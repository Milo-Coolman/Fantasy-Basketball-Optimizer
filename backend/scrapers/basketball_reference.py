"""
Basketball Reference Web Scraper.

This module scrapes player statistics from Basketball Reference for use
in fantasy basketball projections and analysis.

Features:
- Per-game stats for all active NBA players
- Advanced stats (usage rate, PER, true shooting %, etc.)
- Player game logs for recent performance
- Rate limiting to respect robots.txt (1-2 seconds between requests)
- Player name normalization for matching with ESPN data
- Error handling for missing data and HTML changes
- Caching integration to minimize requests

Reference: PRD Section 7.1 - Basketball Reference Scraper
"""

import logging
import re
import time
import unicodedata
from datetime import datetime, date
from typing import Optional, Dict, List, Any, Tuple
from functools import wraps

import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment

# Set up logging
logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

BASE_URL = "https://www.basketball-reference.com"

# Rate limiting settings
MIN_REQUEST_INTERVAL = 1.5  # Minimum seconds between requests
MAX_REQUEST_INTERVAL = 2.5  # Maximum seconds between requests

# Request headers to mimic a browser
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive',
}

# Common stat column mappings
STAT_COLUMNS = {
    'per_game': [
        'player', 'pos', 'age', 'tm', 'g', 'gs', 'mp',
        'fg', 'fga', 'fg_pct', '3p', '3pa', '3p_pct',
        '2p', '2pa', '2p_pct', 'efg_pct', 'ft', 'fta', 'ft_pct',
        'orb', 'drb', 'trb', 'ast', 'stl', 'blk', 'tov', 'pf', 'pts'
    ],
    'advanced': [
        'player', 'pos', 'age', 'tm', 'g', 'mp',
        'per', 'ts_pct', '3par', 'ftr', 'orb_pct', 'drb_pct', 'trb_pct',
        'ast_pct', 'stl_pct', 'blk_pct', 'tov_pct', 'usg_pct',
        'ows', 'dws', 'ws', 'ws_48', 'obpm', 'dbpm', 'bpm', 'vorp'
    ],
    'totals': [
        'player', 'pos', 'age', 'tm', 'g', 'gs', 'mp',
        'fg', 'fga', 'fg_pct', '3p', '3pa', '3p_pct',
        '2p', '2pa', '2p_pct', 'efg_pct', 'ft', 'fta', 'ft_pct',
        'orb', 'drb', 'trb', 'ast', 'stl', 'blk', 'tov', 'pf', 'pts'
    ]
}

# Mapping from new Basketball Reference data-stat attributes to our standard column names
# BBRef updated their HTML structure - this maps new names to legacy names for compatibility
COLUMN_NAME_MAPPING = {
    # Player identification
    'name_display': 'player',
    'team_name_abbr': 'tm',
    # Games
    'games': 'g',
    'games_started': 'gs',
    # Per-game stats (new format has _per_g suffix)
    'mp_per_g': 'mp',
    'fg_per_g': 'fg',
    'fga_per_g': 'fga',
    'fg_pct': 'fg_pct',
    'fg3_per_g': '3p',
    'fg3a_per_g': '3pa',
    'fg3_pct': '3p_pct',
    'fg2_per_g': '2p',
    'fg2a_per_g': '2pa',
    'fg2_pct': '2p_pct',
    'efg_pct': 'efg_pct',
    'ft_per_g': 'ft',
    'fta_per_g': 'fta',
    'ft_pct': 'ft_pct',
    'orb_per_g': 'orb',
    'drb_per_g': 'drb',
    'trb_per_g': 'trb',
    'ast_per_g': 'ast',
    'stl_per_g': 'stl',
    'blk_per_g': 'blk',
    'tov_per_g': 'tov',
    'pf_per_g': 'pf',
    'pts_per_g': 'pts',
    # Totals stats (no suffix)
    'mp': 'mp',
    'fg': 'fg',
    'fga': 'fga',
    'fg3': '3p',
    'fg3a': '3pa',
    'fg2': '2p',
    'fg2a': '2pa',
    'ft': 'ft',
    'fta': 'fta',
    'orb': 'orb',
    'drb': 'drb',
    'trb': 'trb',
    'ast': 'ast',
    'stl': 'stl',
    'blk': 'blk',
    'tov': 'tov',
    'pf': 'pf',
    'pts': 'pts',
    # Advanced stats
    'per': 'per',
    'ts_pct': 'ts_pct',
    'fg3a_per_fga_pct': '3par',
    'fta_per_fga_pct': 'ftr',
    'orb_pct': 'orb_pct',
    'drb_pct': 'drb_pct',
    'trb_pct': 'trb_pct',
    'ast_pct': 'ast_pct',
    'stl_pct': 'stl_pct',
    'blk_pct': 'blk_pct',
    'tov_pct': 'tov_pct',
    'usg_pct': 'usg_pct',
    'ows': 'ows',
    'dws': 'dws',
    'ws': 'ws',
    'ws_per_48': 'ws_48',
    'obpm': 'obpm',
    'dbpm': 'dbpm',
    'bpm': 'bpm',
    'vorp': 'vorp',
    # Keep these as-is
    'age': 'age',
    'pos': 'pos',
    'ranker': 'rk',
}


# =============================================================================
# Custom Exceptions
# =============================================================================

class ScraperError(Exception):
    """Base exception for scraper errors."""
    pass


class RateLimitError(ScraperError):
    """Raised when rate limited by Basketball Reference."""
    pass


class PageNotFoundError(ScraperError):
    """Raised when a requested page doesn't exist."""
    pass


class ParseError(ScraperError):
    """Raised when HTML parsing fails."""
    pass


# =============================================================================
# Rate Limiter
# =============================================================================

class RateLimiter:
    """
    Simple rate limiter to ensure polite scraping.

    Enforces a minimum delay between requests to respect
    Basketball Reference's server resources.
    """

    def __init__(self, min_interval: float = MIN_REQUEST_INTERVAL):
        """
        Initialize rate limiter.

        Args:
            min_interval: Minimum seconds between requests
        """
        self.min_interval = min_interval
        self.last_request_time: Optional[float] = None

    def wait(self) -> None:
        """Wait if necessary to respect rate limiting."""
        if self.last_request_time is not None:
            elapsed = time.time() - self.last_request_time
            if elapsed < self.min_interval:
                sleep_time = self.min_interval - elapsed
                logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
                time.sleep(sleep_time)

        self.last_request_time = time.time()


# Global rate limiter instance
_rate_limiter = RateLimiter()


def rate_limited(func):
    """Decorator to apply rate limiting to a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        _rate_limiter.wait()
        return func(*args, **kwargs)
    return wrapper


# =============================================================================
# Player Name Normalization
# =============================================================================

def normalize_player_name(name: str) -> str:
    """
    Normalize a player name for matching across data sources.

    Handles:
    - Unicode characters (accents, special characters)
    - Suffixes (Jr., III, etc.)
    - Extra whitespace
    - Case normalization

    Args:
        name: Raw player name

    Returns:
        Normalized player name
    """
    if not name:
        return ""

    # Convert to lowercase
    name = name.lower().strip()

    # Normalize unicode characters (é -> e, etc.)
    name = unicodedata.normalize('NFKD', name)
    name = ''.join(c for c in name if not unicodedata.combining(c))

    # Remove common suffixes for matching purposes
    suffixes = [' jr.', ' jr', ' sr.', ' sr', ' iii', ' ii', ' iv', ' v']
    for suffix in suffixes:
        if name.endswith(suffix):
            name = name[:-len(suffix)]

    # Remove periods and extra whitespace
    name = name.replace('.', '')
    name = ' '.join(name.split())

    return name


def create_player_id(name: str) -> str:
    """
    Create a Basketball Reference style player ID from a name.

    Format: first 5 letters of last name + first 2 of first name + 01
    Example: "LeBron James" -> "jamesle01"

    Args:
        name: Player name

    Returns:
        Basketball Reference player ID format
    """
    parts = name.lower().strip().split()
    if len(parts) < 2:
        return ""

    first_name = parts[0]
    last_name = parts[-1]

    # Remove special characters
    first_name = re.sub(r'[^a-z]', '', first_name)
    last_name = re.sub(r'[^a-z]', '', last_name)

    player_id = f"{last_name[:5]}{first_name[:2]}01"
    return player_id


# =============================================================================
# HTTP Request Handler
# =============================================================================

@rate_limited
def fetch_page(url: str, timeout: int = 30) -> BeautifulSoup:
    """
    Fetch and parse a web page.

    Args:
        url: URL to fetch
        timeout: Request timeout in seconds

    Returns:
        BeautifulSoup object of parsed HTML

    Raises:
        PageNotFoundError: If page returns 404
        RateLimitError: If rate limited (429)
        ScraperError: For other HTTP errors
    """
    logger.debug(f"Fetching: {url}")

    try:
        response = requests.get(url, headers=HEADERS, timeout=timeout)

        if response.status_code == 404:
            raise PageNotFoundError(f"Page not found: {url}")

        if response.status_code == 429:
            raise RateLimitError("Rate limited by Basketball Reference")

        response.raise_for_status()

        return BeautifulSoup(response.content, 'html.parser')

    except requests.exceptions.Timeout:
        raise ScraperError(f"Request timeout: {url}")
    except requests.exceptions.RequestException as e:
        raise ScraperError(f"Request failed: {e}")


def extract_table_from_comments(soup: BeautifulSoup, table_id: str) -> Optional[BeautifulSoup]:
    """
    Extract a table that may be hidden in HTML comments.

    Basketball Reference often hides tables in comments for performance.

    Args:
        soup: BeautifulSoup object
        table_id: ID of the table to find

    Returns:
        BeautifulSoup table element or None
    """
    # First try to find the table directly
    table = soup.find('table', {'id': table_id})
    if table:
        return table

    # Search in HTML comments
    comments = soup.find_all(string=lambda text: isinstance(text, Comment))
    for comment in comments:
        if table_id in comment:
            comment_soup = BeautifulSoup(comment, 'html.parser')
            table = comment_soup.find('table', {'id': table_id})
            if table:
                return table

    return None


# =============================================================================
# Data Parsing Functions
# =============================================================================

def parse_stats_table(table: BeautifulSoup, expected_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Parse a Basketball Reference stats table into a DataFrame.

    Args:
        table: BeautifulSoup table element
        expected_columns: Optional list of expected column names

    Returns:
        pandas DataFrame with parsed stats
    """
    if table is None:
        raise ParseError("No table provided")

    rows = []
    headers = []

    # Get headers
    thead = table.find('thead')
    if thead:
        header_row = thead.find_all('tr')[-1]  # Last header row (handles multi-level headers)
        headers = [th.get('data-stat', th.get_text(strip=True).lower())
                   for th in header_row.find_all(['th', 'td'])]

    # Get data rows
    tbody = table.find('tbody')
    if tbody:
        for tr in tbody.find_all('tr'):
            # Skip separator rows
            if 'thead' in tr.get('class', []) or tr.find('th', {'colspan': True}):
                continue

            row = {}
            cells = tr.find_all(['th', 'td'])

            for i, cell in enumerate(cells):
                stat_name = cell.get('data-stat', headers[i] if i < len(headers) else f'col_{i}')

                # Map new BBRef column names to legacy names
                mapped_name = COLUMN_NAME_MAPPING.get(stat_name, stat_name)

                # Get cell value
                value = cell.get_text(strip=True)

                # Handle links (player names have links)
                link = cell.find('a')
                if link and mapped_name == 'player':
                    # Extract player ID from href
                    href = link.get('href', '')
                    if '/players/' in href:
                        player_id = href.split('/')[-1].replace('.html', '')
                        row['player_id'] = player_id

                row[mapped_name] = value

            # Only add rows with player data (check both old and new column names)
            if row and (row.get('player') or row.get('name_display')):
                rows.append(row)

    df = pd.DataFrame(rows)

    # Convert numeric columns
    df = convert_numeric_columns(df)

    return df


def convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert appropriate columns to numeric types.

    Args:
        df: DataFrame with string columns

    Returns:
        DataFrame with numeric columns converted
    """
    # Columns that should remain as strings
    string_columns = {'player', 'player_id', 'pos', 'tm', 'team', 'date', 'opp', 'result'}

    for col in df.columns:
        if col.lower() in string_columns:
            continue

        # Try to convert to numeric
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except (ValueError, TypeError):
            pass

    return df


# =============================================================================
# Main Scraping Functions
# =============================================================================

class BasketballReferenceScraper:
    """
    Scraper for Basketball Reference player statistics.

    Usage:
        scraper = BasketballReferenceScraper()
        per_game = scraper.get_per_game_stats(2025)
        advanced = scraper.get_advanced_stats(2025)
        game_log = scraper.get_player_game_log('jamesle01', 2025)
    """

    def __init__(self, cache_service=None):
        """
        Initialize the scraper.

        Args:
            cache_service: Optional cache service for storing results
        """
        self.cache = cache_service
        self._session = requests.Session()
        self._session.headers.update(HEADERS)

    def get_per_game_stats(self, season: int) -> pd.DataFrame:
        """
        Get per-game stats for all NBA players in a season.

        Args:
            season: Season year (e.g., 2025 for 2024-25 season)

        Returns:
            DataFrame with per-game stats for all players
        """
        cache_key = f"bbref:per_game:{season}"

        # Check cache
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                logger.debug(f"Cache hit for per-game stats {season}")
                return cached

        url = f"{BASE_URL}/leagues/NBA_{season}_per_game.html"
        logger.info(f"Scraping per-game stats for {season}")

        try:
            soup = fetch_page(url)
            table = extract_table_from_comments(soup, 'per_game_stats')

            if table is None:
                raise ParseError(f"Could not find per_game_stats table for {season}")

            df = parse_stats_table(table)
            df['season'] = season

            # Normalize player names
            df['player_normalized'] = df['player'].apply(normalize_player_name)

            # Handle players on multiple teams (TOT = total)
            # Keep only TOT row for players traded mid-season
            df = self._handle_multi_team_players(df)

            # Cache the result
            if self.cache:
                self.cache.set(cache_key, df, ttl=3600)  # 1 hour TTL

            logger.info(f"Scraped {len(df)} players for {season} per-game stats")
            return df

        except PageNotFoundError:
            logger.error(f"Season {season} not found on Basketball Reference")
            raise
        except Exception as e:
            logger.error(f"Error scraping per-game stats: {e}")
            raise ScraperError(f"Failed to scrape per-game stats: {e}")

    def get_advanced_stats(self, season: int) -> pd.DataFrame:
        """
        Get advanced stats for all NBA players in a season.

        Includes: PER, TS%, USG%, WS, BPM, VORP, etc.

        Args:
            season: Season year (e.g., 2025 for 2024-25 season)

        Returns:
            DataFrame with advanced stats for all players
        """
        cache_key = f"bbref:advanced:{season}"

        # Check cache
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                logger.debug(f"Cache hit for advanced stats {season}")
                return cached

        url = f"{BASE_URL}/leagues/NBA_{season}_advanced.html"
        logger.info(f"Scraping advanced stats for {season}")

        try:
            soup = fetch_page(url)
            # Try multiple table IDs - BBRef changed from 'advanced_stats' to 'advanced'
            table = None
            for table_id in ['advanced', 'advanced_stats']:
                table = extract_table_from_comments(soup, table_id)
                if table:
                    break

            if table is None:
                raise ParseError(f"Could not find advanced stats table for {season}")

            df = parse_stats_table(table)
            df['season'] = season

            # Normalize player names
            df['player_normalized'] = df['player'].apply(normalize_player_name)

            # Handle multi-team players
            df = self._handle_multi_team_players(df)

            # Cache the result
            if self.cache:
                self.cache.set(cache_key, df, ttl=3600)

            logger.info(f"Scraped {len(df)} players for {season} advanced stats")
            return df

        except PageNotFoundError:
            logger.error(f"Season {season} not found on Basketball Reference")
            raise
        except Exception as e:
            logger.error(f"Error scraping advanced stats: {e}")
            raise ScraperError(f"Failed to scrape advanced stats: {e}")

    def get_totals_stats(self, season: int) -> pd.DataFrame:
        """
        Get total stats (not per-game) for all NBA players in a season.

        Args:
            season: Season year

        Returns:
            DataFrame with total stats for all players
        """
        cache_key = f"bbref:totals:{season}"

        if self.cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        url = f"{BASE_URL}/leagues/NBA_{season}_totals.html"
        logger.info(f"Scraping totals stats for {season}")

        try:
            soup = fetch_page(url)
            # Try multiple table IDs for resilience
            table = None
            for table_id in ['totals_stats', 'totals']:
                table = extract_table_from_comments(soup, table_id)
                if table:
                    break

            if table is None:
                raise ParseError(f"Could not find totals stats table for {season}")

            df = parse_stats_table(table)
            df['season'] = season
            df['player_normalized'] = df['player'].apply(normalize_player_name)
            df = self._handle_multi_team_players(df)

            if self.cache:
                self.cache.set(cache_key, df, ttl=3600)

            logger.info(f"Scraped {len(df)} players for {season} totals")
            return df

        except Exception as e:
            logger.error(f"Error scraping totals: {e}")
            raise ScraperError(f"Failed to scrape totals: {e}")

    def get_player_game_log(
        self,
        player_id: str,
        season: int,
        last_n_games: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get game-by-game stats for a specific player.

        Args:
            player_id: Basketball Reference player ID (e.g., 'jamesle01')
            season: Season year
            last_n_games: Optional limit to last N games

        Returns:
            DataFrame with game log stats
        """
        cache_key = f"bbref:gamelog:{player_id}:{season}"

        if self.cache and last_n_games is None:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        # Determine the letter subdirectory from player_id
        letter = player_id[0].lower()
        url = f"{BASE_URL}/players/{letter}/{player_id}/gamelog/{season}"
        logger.info(f"Scraping game log for {player_id}, season {season}")

        try:
            soup = fetch_page(url)

            # Game log table ID varies, try common patterns
            table = None
            for table_id in ['pgl_basic', 'gamelog', f'pgl_basic_{season}']:
                table = extract_table_from_comments(soup, table_id)
                if table:
                    break

            if table is None:
                raise ParseError(f"Could not find game log table for {player_id}")

            df = parse_stats_table(table)

            # Filter out inactive games (DNP, etc.)
            if 'gs' in df.columns:
                df = df[df['gs'].notna()]

            # Add metadata
            df['player_id'] = player_id
            df['season'] = season

            # Parse dates
            if 'date_game' in df.columns:
                df['date'] = pd.to_datetime(df['date_game'], errors='coerce')
                df = df.sort_values('date', ascending=False)

            # Limit to last N games if specified
            if last_n_games and len(df) > last_n_games:
                df = df.head(last_n_games)

            # Cache full game log
            if self.cache and last_n_games is None:
                self.cache.set(cache_key, df, ttl=1800)  # 30 min TTL

            logger.info(f"Scraped {len(df)} games for {player_id}")
            return df

        except PageNotFoundError:
            logger.warning(f"Game log not found for {player_id}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error scraping game log for {player_id}: {e}")
            raise ScraperError(f"Failed to scrape game log: {e}")

    def get_player_page(self, player_id: str) -> Dict[str, Any]:
        """
        Get comprehensive data from a player's main page.

        Args:
            player_id: Basketball Reference player ID

        Returns:
            Dictionary with player info and career stats
        """
        letter = player_id[0].lower()
        url = f"{BASE_URL}/players/{letter}/{player_id}.html"
        logger.info(f"Scraping player page for {player_id}")

        try:
            soup = fetch_page(url)

            # Get player info
            info = self._parse_player_info(soup)
            info['player_id'] = player_id

            # Get career stats table
            per_game_table = extract_table_from_comments(soup, 'per_game')
            if per_game_table:
                info['career_stats'] = parse_stats_table(per_game_table)

            return info

        except Exception as e:
            logger.error(f"Error scraping player page for {player_id}: {e}")
            raise ScraperError(f"Failed to scrape player page: {e}")

    def search_player(self, name: str) -> List[Dict[str, str]]:
        """
        Search for a player by name.

        Args:
            name: Player name to search

        Returns:
            List of matching players with IDs
        """
        search_query = name.replace(' ', '+')
        url = f"{BASE_URL}/search/search.fcgi?search={search_query}"

        try:
            soup = fetch_page(url)

            # Check if redirected to player page
            if soup.find('div', {'id': 'meta'}):
                # Direct player page
                player_id = soup.find('link', {'rel': 'canonical'})
                if player_id:
                    href = player_id.get('href', '')
                    pid = href.split('/')[-1].replace('.html', '')
                    return [{'name': name, 'player_id': pid}]

            # Search results page
            results = []
            search_div = soup.find('div', {'id': 'players'})
            if search_div:
                for item in search_div.find_all('div', {'class': 'search-item'}):
                    link = item.find('a')
                    if link:
                        href = link.get('href', '')
                        if '/players/' in href:
                            pid = href.split('/')[-1].replace('.html', '')
                            pname = link.get_text(strip=True)
                            results.append({'name': pname, 'player_id': pid})

            return results

        except Exception as e:
            logger.error(f"Error searching for player {name}: {e}")
            return []

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _handle_multi_team_players(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle players who played for multiple teams in a season.

        Basketball Reference shows separate rows for each team plus a TOT row.
        We keep only the TOT row for these players.

        Args:
            df: DataFrame with potential duplicate players

        Returns:
            DataFrame with duplicates resolved
        """
        if 'tm' not in df.columns:
            return df

        # Find players with TOT (total) row
        tot_players = df[df['tm'] == 'TOT']['player'].unique()

        # Remove individual team rows for these players
        mask = ~((df['player'].isin(tot_players)) & (df['tm'] != 'TOT'))
        df = df[mask].copy()

        return df

    def _parse_player_info(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """
        Parse player biographical info from their page.

        Args:
            soup: BeautifulSoup of player page

        Returns:
            Dictionary with player info
        """
        info = {}

        meta = soup.find('div', {'id': 'meta'})
        if meta:
            # Name
            name_tag = meta.find('h1')
            if name_tag:
                info['name'] = name_tag.get_text(strip=True)

            # Position and other info from paragraphs
            for p in meta.find_all('p'):
                text = p.get_text(strip=True)

                if 'Position:' in text:
                    pos_match = re.search(r'Position:\s*(\w+)', text)
                    if pos_match:
                        info['position'] = pos_match.group(1)

                if 'Shoots:' in text:
                    shoots_match = re.search(r'Shoots:\s*(\w+)', text)
                    if shoots_match:
                        info['shoots'] = shoots_match.group(1)

                if 'Born:' in text:
                    # Extract birth date
                    birth_match = re.search(r'Born:\s*(\w+\s+\d+,\s+\d+)', text)
                    if birth_match:
                        info['birth_date'] = birth_match.group(1)

        return info

    def get_all_player_stats(self, season: int) -> pd.DataFrame:
        """
        Get combined per-game and advanced stats for all players.

        Args:
            season: Season year

        Returns:
            DataFrame with merged stats
        """
        per_game = self.get_per_game_stats(season)
        advanced = self.get_advanced_stats(season)

        # Merge on player and team
        # Select only unique columns from advanced stats
        advanced_cols = ['player', 'tm', 'per', 'ts_pct', 'usg_pct', 'ws', 'bpm', 'vorp']
        advanced_subset = advanced[[c for c in advanced_cols if c in advanced.columns]]

        merged = per_game.merge(
            advanced_subset,
            on=['player', 'tm'],
            how='left',
            suffixes=('', '_adv')
        )

        return merged


# =============================================================================
# Convenience Functions
# =============================================================================

def get_current_season() -> int:
    """
    Get the current NBA season year.

    The NBA season spans two calendar years (e.g., 2024-25 season).
    Basketball Reference uses the end year (2025).

    Returns:
        Current season year
    """
    today = date.today()
    # NBA season typically starts in October
    if today.month >= 10:
        return today.year + 1
    return today.year


def create_scraper(cache_service=None) -> BasketballReferenceScraper:
    """
    Factory function to create a configured scraper instance.

    Args:
        cache_service: Optional cache service

    Returns:
        Configured BasketballReferenceScraper instance
    """
    return BasketballReferenceScraper(cache_service=cache_service)

"""
ESPN Fantasy Basketball API Client Service.

This module wraps the espn-api package to provide a clean interface for
fetching fantasy basketball league data from ESPN.

Features:
- League initialization with authentication
- Fetch league settings and scoring rules
- Get team rosters and standings
- Get free agent pool
- Get matchup data for H2H leagues
- Comprehensive error handling and retry logic
- Logging for debugging and monitoring

Reference: PRD Section 7.2 - ESPN API Integration
"""

import logging
import time
from datetime import date, timedelta
from typing import Optional, Dict, List, Any
from functools import wraps

import requests
from espn_api.basketball import League as ESPNLeague
from requests.exceptions import RequestException

# Set up logging
logger = logging.getLogger(__name__)


def extract_games_played(raw_stats: dict, season: int, player_name: str = "Unknown") -> int:
    """
    Extract games_played from raw ESPN player stats using STRING KEYS.

    ESPN-api library returns stats with STRING KEYS like 'PTS', 'REB', 'GP'
    (not numeric IDs like 0, 6, 41).

    IMPORTANT: ESPN's direct GP field is sometimes incorrect (e.g., showing 1 game
    when player has 51). Calculation from PTS is more reliable.

    Priority order (calculation is PRIMARY, direct lookup is FALLBACK):
    1. Calculate from total['PTS'] / avg['PTS'] - most reliable
    2. Direct GP in 'total' dict (fallback if calculation fails)
    3. Direct GP in 'avg' dict
    4. Direct GP in season_total top-level

    Args:
        raw_stats: Player's stats dict (player.stats)
        season: Season year (e.g., 2026)
        player_name: Player name for logging warnings

    Returns:
        Games played count (0 if not found)
    """
    season_key = f'{season}_total'
    season_total = raw_stats.get(season_key, {})

    if not season_total or not isinstance(season_total, dict):
        return 0

    total_dict = season_total.get('total', {})
    avg_dict = season_total.get('avg', {})

    # GP key variants (STRING KEYS - NOT numeric IDs)
    GP_KEYS = ['GP', 'G', 'gamesPlayed', 'games_played', 'GamesPlayed', 'GAMES_PLAYED']

    # ==========================================================================
    # Method 1 (PRIMARY): Calculate from total['PTS'] / avg['PTS']
    # This is most reliable - ESPN's direct GP field is sometimes wrong
    # ==========================================================================
    calculated_gp = 0
    if avg_dict and total_dict:
        pts_keys = ['PTS', 'pts', 'Points', 'points', 0, '0']

        pts_avg = 0
        pts_total = 0

        for pts_key in pts_keys:
            if pts_key in avg_dict and avg_dict[pts_key]:
                pts_avg = float(avg_dict[pts_key])
                break

        for pts_key in pts_keys:
            if pts_key in total_dict and total_dict[pts_key]:
                pts_total = float(total_dict[pts_key])
                break

        if pts_avg > 0 and pts_total > 0:
            calculated_gp = int(round(pts_total / pts_avg))

    # ==========================================================================
    # Get direct GP for comparison/fallback
    # ==========================================================================
    direct_gp = 0

    # Method 2 (FALLBACK): Direct GP in 'total' dict
    if total_dict and isinstance(total_dict, dict):
        for gp_key in GP_KEYS:
            gp = total_dict.get(gp_key, 0)
            if gp and float(gp) > 0:
                direct_gp = int(float(gp))
                break

    # Method 3: Direct GP in 'avg' dict
    if not direct_gp and avg_dict and isinstance(avg_dict, dict):
        for gp_key in GP_KEYS:
            gp = avg_dict.get(gp_key, 0)
            if gp and float(gp) > 0:
                direct_gp = int(float(gp))
                break

    # Method 4: Direct GP in season_total top-level
    if not direct_gp:
        for gp_key in GP_KEYS + ['applied_total', 'appliedTotal']:
            gp = season_total.get(gp_key, 0)
            if gp and float(gp) > 0:
                direct_gp = int(float(gp))
                break

    # ==========================================================================
    # Determine final GP value with validation
    # ==========================================================================
    if calculated_gp > 0:
        # Calculated GP is available - use it as source of truth
        if direct_gp > 0 and calculated_gp != direct_gp:
            # Check for significant mismatch (>10% difference)
            diff_pct = abs(calculated_gp - direct_gp) / max(calculated_gp, direct_gp)
            if diff_pct > 0.10:
                logger.warning(
                    f"GP mismatch for {player_name}: calculated={calculated_gp}, "
                    f"ESPN_reported={direct_gp}, using calculated"
                )
        return calculated_gp
    elif direct_gp > 0:
        # Fallback to direct GP if calculation failed
        return direct_gp

    return 0


# =============================================================================
# Custom Exceptions
# =============================================================================

class ESPNClientError(Exception):
    """Base exception for ESPN client errors."""
    pass


class ESPNAuthenticationError(ESPNClientError):
    """Raised when ESPN authentication fails."""
    pass


class ESPNLeagueNotFoundError(ESPNClientError):
    """Raised when the specified league cannot be found."""
    pass


class ESPNRateLimitError(ESPNClientError):
    """Raised when ESPN rate limits our requests."""
    pass


class ESPNConnectionError(ESPNClientError):
    """Raised when connection to ESPN fails."""
    pass


# =============================================================================
# Retry Decorator
# =============================================================================

def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Decorator to retry a function on failure with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Multiplier for delay after each retry
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except (RequestException, ConnectionError) as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}: {e}. "
                            f"Retrying in {current_delay:.1f}s..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed for {func.__name__}")
                except Exception as e:
                    # Don't retry on other exceptions
                    raise

            raise ESPNConnectionError(f"Failed after {max_retries + 1} attempts: {last_exception}")
        return wrapper
    return decorator


# =============================================================================
# ESPN Client Service
# =============================================================================

class ESPNClient:
    """
    ESPN Fantasy Basketball API client wrapper.

    Provides a clean interface to the espn-api package with error handling,
    retry logic, and data transformation.

    Usage:
        client = ESPNClient(league_id=12345, year=2025, espn_s2="...", swid="...")
        settings = client.get_league_settings()
        teams = client.get_teams()
    """

    def __init__(
        self,
        league_id: int,
        year: int,
        espn_s2: str,
        swid: str,
        fetch_league: bool = True
    ):
        """
        Initialize ESPN client and connect to league.

        Args:
            league_id: ESPN league ID
            year: Season year (e.g., 2025)
            espn_s2: ESPN_S2 cookie value for authentication
            swid: SWID cookie value for authentication
            fetch_league: If True, immediately fetch league data

        Raises:
            ESPNAuthenticationError: If credentials are invalid
            ESPNLeagueNotFoundError: If league doesn't exist
            ESPNConnectionError: If connection fails
        """
        self.league_id = league_id
        self.year = year
        self.espn_s2 = espn_s2
        self.swid = swid
        self._league: Optional[ESPNLeague] = None
        self._raw_player_data: Dict[int, Dict] = {}  # player_id -> raw API data (includes injuryDetails)

        logger.info(f"Initializing ESPN client for league {league_id}, season {year}")

        if fetch_league:
            self._connect()

    @retry_on_failure(max_retries=3, delay=1.0, backoff=2.0)
    def _connect(self) -> None:
        """
        Connect to ESPN and fetch league data.

        Raises:
            ESPNAuthenticationError: If credentials are invalid
            ESPNLeagueNotFoundError: If league doesn't exist
            ESPNConnectionError: If connection fails
        """
        try:
            self._league = ESPNLeague(
                league_id=self.league_id,
                year=self.year,
                espn_s2=self.espn_s2,
                swid=self.swid
            )
            logger.info(f"Successfully connected to league: {self._league.settings.name}")

        except Exception as e:
            error_msg = str(e).lower()

            # Check for authentication errors
            if 'unauthorized' in error_msg or '401' in error_msg or 'authentication' in error_msg:
                logger.error(f"Authentication failed for league {self.league_id}: {e}")
                raise ESPNAuthenticationError(
                    "ESPN authentication failed. Please verify your espn_s2 and swid cookies are correct and not expired."
                )

            # Check for league not found
            if 'not found' in error_msg or '404' in error_msg or 'does not exist' in error_msg:
                logger.error(f"League {self.league_id} not found: {e}")
                raise ESPNLeagueNotFoundError(
                    f"League {self.league_id} not found for season {self.year}. "
                    "Please verify the league ID and season."
                )

            # Check for private league access
            if 'private' in error_msg:
                logger.error(f"Cannot access private league {self.league_id}: {e}")
                raise ESPNAuthenticationError(
                    "This league is private. Please provide valid espn_s2 and swid cookies."
                )

            # Generic connection error
            logger.error(f"Failed to connect to ESPN league {self.league_id}: {e}")
            raise ESPNConnectionError(f"Failed to connect to ESPN: {e}")

    @property
    def league(self) -> ESPNLeague:
        """Get the ESPN league object, connecting if necessary."""
        if self._league is None:
            self._connect()
        return self._league

    def refresh(self) -> None:
        """Refresh league data from ESPN."""
        logger.info(f"Refreshing data for league {self.league_id}")
        self._league = None
        self._raw_player_data = {}
        self._connect()

    def _fetch_raw_player_data(self) -> None:
        """
        Fetch raw player data directly from ESPN API to get injuryDetails.

        The espn-api library does NOT expose injuryDetails, so we need to make
        a raw HTTP request to get this data. This populates self._raw_player_data
        with a mapping of player_id -> raw player dict from ESPN API.
        """
        if self._raw_player_data:
            # Already fetched
            return

        logger.info("Fetching raw player data from ESPN API for injuryDetails...")

        try:
            endpoint = (
                f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/fba/"
                f"seasons/{self.year}/segments/0/leagues/{self.league_id}"
            )
            # kona_playercard view provides injuryDetails
            params = {'view': ['mTeam', 'mRoster', 'kona_player_info', 'kona_playercard']}
            cookies = {'espn_s2': self.espn_s2, 'SWID': self.swid}

            response = requests.get(endpoint, params=params, cookies=cookies, timeout=30)
            response.raise_for_status()
            raw_data = response.json()

            # Build player_id -> raw data mapping
            for team_data in raw_data.get('teams', []):
                for entry in team_data.get('roster', {}).get('entries', []):
                    player_pool = entry.get('playerPoolEntry', {})
                    player = player_pool.get('player', {})
                    player_id = player.get('id')

                    if player_id:
                        # Store the full player dict plus entry-level data
                        self._raw_player_data[player_id] = {
                            'player': player,
                            'entry': entry,
                            'playerPoolEntry': player_pool,
                        }

            logger.info(f"Cached raw data for {len(self._raw_player_data)} players")

            # Log any players with injuryDetails
            players_with_injury_details = [
                (pid, data['player'].get('fullName', 'Unknown'))
                for pid, data in self._raw_player_data.items()
                if data['player'].get('injuryDetails')
            ]
            if players_with_injury_details:
                logger.info(f"Players with injuryDetails: {players_with_injury_details}")
            else:
                logger.info("No players found with injuryDetails in raw API response")

        except Exception as e:
            logger.warning(f"Failed to fetch raw player data: {e}")
            # Don't fail - just won't have injuryDetails

    def _get_raw_injury_details(self, player_id: int) -> Optional[Dict]:
        """
        Get raw injuryDetails for a player from cached ESPN API data.

        Args:
            player_id: ESPN player ID

        Returns:
            Dict with injury details or None if not available
        """
        # Ensure we have raw data
        self._fetch_raw_player_data()

        raw_data = self._raw_player_data.get(player_id)
        if not raw_data:
            return None

        # Check all possible locations for injuryDetails
        player = raw_data.get('player', {})
        entry = raw_data.get('entry', {})
        player_pool = raw_data.get('playerPoolEntry', {})

        raw_injury_details = (
            player.get('injuryDetails') or
            entry.get('injuryDetails') or
            player_pool.get('injuryDetails')
        )

        if not raw_injury_details or not isinstance(raw_injury_details, dict):
            return None

        # Parse injuryDetails into our format
        injury_details = {}

        # Extract out_for_season flag
        out_for_season = raw_injury_details.get('outForSeason', False)
        injury_details['out_for_season'] = out_for_season

        # Extract injury type
        injury_type = raw_injury_details.get('type', '')
        injury_details['injury_type'] = injury_type

        # Extract expected return date (array format: [year, month, day])
        raw_return_date = raw_injury_details.get('expectedReturnDate')
        if raw_return_date and isinstance(raw_return_date, list) and len(raw_return_date) >= 3:
            try:
                return_date_obj = date(raw_return_date[0], raw_return_date[1], raw_return_date[2])
                injury_details['expected_return_date'] = return_date_obj
            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to parse return date {raw_return_date}: {e}")

        return injury_details

    # =========================================================================
    # League Settings
    # =========================================================================

    def get_league_settings(self) -> Dict[str, Any]:
        """
        Get league settings and configuration.

        Returns:
            Dictionary containing:
                - name: League name
                - size: Number of teams
                - scoring_type: Category or Points
                - roster_settings: Position slots and limits
                - scoring_settings: Category weights or point values
                - reg_season_weeks: Number of regular season weeks
                - playoff_teams: Number of playoff teams
                - current_week: Current matchup period
        """
        try:
            settings = self.league.settings

            # Determine scoring type
            scoring_type = self._determine_scoring_type()

            # Build roster settings
            roster_settings = self._get_roster_settings()

            # Build scoring settings
            scoring_settings = self._get_scoring_settings()

            result = {
                'name': settings.name,
                'size': settings.team_count,
                'scoring_type': scoring_type,
                'roster_settings': roster_settings,
                'scoring_settings': scoring_settings,
                'reg_season_weeks': getattr(settings, 'reg_season_count', None),
                'playoff_teams': getattr(settings, 'playoff_team_count', None),
                'current_week': self.league.current_week,
            }

            logger.debug(f"Retrieved league settings: {result['name']}")
            return result

        except ESPNClientError:
            raise
        except Exception as e:
            logger.error(f"Error getting league settings: {e}")
            raise ESPNClientError(f"Failed to get league settings: {e}")

    def _determine_scoring_type(self) -> str:
        """Determine the league scoring type (H2H_CATEGORY, H2H_POINTS, ROTO)."""
        try:
            settings = self.league.settings

            # Check for scoring type attribute
            if hasattr(settings, 'scoring_type'):
                scoring_type = str(settings.scoring_type).upper()
                if 'ROTO' in scoring_type:
                    return 'ROTO'
                elif 'POINT' in scoring_type:
                    return 'H2H_POINTS'
                else:
                    return 'H2H_CATEGORY'

            # Fallback: Check if league has matchups (H2H) or not (Roto)
            if hasattr(self.league, 'scoreboard') and self.league.scoreboard:
                return 'H2H_CATEGORY'

            return 'H2H_CATEGORY'  # Default

        except Exception as e:
            logger.warning(f"Could not determine scoring type, defaulting to H2H_CATEGORY: {e}")
            return 'H2H_CATEGORY'

    def _get_roster_settings(self) -> Dict[str, Any]:
        """Get roster position settings."""
        try:
            roster_settings = {}

            if hasattr(self.league.settings, 'roster'):
                roster = self.league.settings.roster
                for position, count in roster.items():
                    roster_settings[position] = count

            return roster_settings

        except Exception as e:
            logger.warning(f"Could not get roster settings: {e}")
            return {}

    def _get_scoring_settings(self) -> Dict[str, Any]:
        """Get scoring category settings."""
        try:
            scoring_settings = {}

            if hasattr(self.league.settings, 'scoring_items'):
                for item in self.league.settings.scoring_items:
                    if hasattr(item, 'stat_abbr') and hasattr(item, 'points'):
                        scoring_settings[item.stat_abbr] = item.points

            return scoring_settings

        except Exception as e:
            logger.warning(f"Could not get scoring settings: {e}")
            return {}

    # ESPN Stat ID to category name mapping
    ESPN_STAT_ID_MAP = {
        0: 'PTS',
        1: 'BLK',
        2: 'STL',
        3: 'AST',
        4: 'OREB',
        5: 'DREB',
        6: 'REB',
        9: 'PF',
        10: 'TF',   # Technical Fouls
        11: 'TO',
        12: 'EJ',   # Ejections
        13: 'FGM',
        14: 'FGA',
        15: 'FTM',
        16: 'FTA',
        17: '3PM',
        18: '3PA',
        19: 'FG%',
        20: 'FT%',
        21: '3P%',
        22: 'DD',   # Double-Double
        23: 'TD',   # Triple-Double
        24: 'QD',   # Quadruple-Double
        28: 'MPG',
        29: 'MIN',
        40: 'GS',   # Games Started
        41: 'GP',
    }

    # Standard display order for categories
    STANDARD_CATEGORY_ORDER = [
        'FGM', 'FG%', 'FTM', 'FT%', '3PM', '3P%', 'PTS', 'REB', 'OREB', 'DREB',
        'AST', 'STL', 'BLK', 'TO', 'DD', 'TD', 'PF'
    ]

    def get_scoring_categories(self) -> List[Dict[str, Any]]:
        """
        Fetch the actual scoring categories from ESPN's raw API.

        This method makes a direct API call to get the league's scoringSettings,
        which contains the actual categories used for scoring (not just tracked stats).

        Returns:
            List of category dictionaries with:
                - key: Category abbreviation (e.g., 'PTS', 'REB')
                - label: Full category name (e.g., 'Points', 'Rebounds')
                - abbr: Short abbreviation for display
                - is_reverse: True if lower is better (e.g., TO)
        """
        # Category metadata for display
        CATEGORY_METADATA = {
            'PTS': {'label': 'Points', 'abbr': 'PTS'},
            'BLK': {'label': 'Blocks', 'abbr': 'BLK'},
            'STL': {'label': 'Steals', 'abbr': 'STL'},
            'AST': {'label': 'Assists', 'abbr': 'AST'},
            'REB': {'label': 'Rebounds', 'abbr': 'REB'},
            'OREB': {'label': 'Offensive Rebounds', 'abbr': 'OREB'},
            'DREB': {'label': 'Defensive Rebounds', 'abbr': 'DREB'},
            'FGM': {'label': 'Field Goals Made', 'abbr': 'FGM'},
            'FGA': {'label': 'Field Goals Attempted', 'abbr': 'FGA'},
            'FTM': {'label': 'Free Throws Made', 'abbr': 'FTM'},
            'FTA': {'label': 'Free Throws Attempted', 'abbr': 'FTA'},
            '3PM': {'label': '3-Pointers Made', 'abbr': '3PM'},
            '3PA': {'label': '3-Pointers Attempted', 'abbr': '3PA'},
            'FG%': {'label': 'Field Goal %', 'abbr': 'FG%'},
            'FT%': {'label': 'Free Throw %', 'abbr': 'FT%'},
            '3P%': {'label': '3-Point %', 'abbr': '3P%'},
            'TO': {'label': 'Turnovers', 'abbr': 'TO'},
            'PF': {'label': 'Personal Fouls', 'abbr': 'PF'},
            'DD': {'label': 'Double-Doubles', 'abbr': 'DD'},
            'TD': {'label': 'Triple-Doubles', 'abbr': 'TD'},
            'MIN': {'label': 'Minutes', 'abbr': 'MIN'},
            'MPG': {'label': 'Minutes Per Game', 'abbr': 'MPG'},
            'GP': {'label': 'Games Played', 'abbr': 'GP'},
        }

        try:
            # Make direct API call to get raw league settings
            endpoint = (
                f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/fba/"
                f"seasons/{self.year}/segments/0/leagues/{self.league_id}"
            )
            params = {'view': 'mSettings'}
            cookies = {
                'espn_s2': self.espn_s2,
                'SWID': self.swid
            }

            response = requests.get(endpoint, params=params, cookies=cookies, timeout=10)

            if response.status_code != 200:
                logger.warning(f"Failed to fetch raw settings (status {response.status_code}), falling back to team stats")
                return self._get_scoring_categories_from_team_stats()

            raw_data = response.json()

            # Extract scoring items from settings
            scoring_settings = raw_data.get('settings', {}).get('scoringSettings', {})
            scoring_items = scoring_settings.get('scoringItems', [])

            if not scoring_items:
                logger.warning("No scoringItems found in API response, falling back to team stats")
                return self._get_scoring_categories_from_team_stats()

            # Parse scoring items
            categories = []
            for item in scoring_items:
                stat_id = item.get('statId')
                is_reverse = item.get('isReverseItem', False)

                # Map stat ID to category name
                category_key = self.ESPN_STAT_ID_MAP.get(stat_id)
                if not category_key:
                    logger.debug(f"Unknown stat ID {stat_id}, skipping")
                    continue

                # Get metadata
                metadata = CATEGORY_METADATA.get(category_key, {
                    'label': category_key,
                    'abbr': category_key
                })

                categories.append({
                    'key': category_key,
                    'stat_key': category_key,  # Alias for compatibility with dashboard
                    'label': metadata['label'],
                    'abbr': metadata['abbr'],
                    'is_reverse': is_reverse,
                    'stat_id': stat_id,
                })

            # Sort by standard order
            def sort_key(cat):
                try:
                    return self.STANDARD_CATEGORY_ORDER.index(cat['key'])
                except ValueError:
                    return len(self.STANDARD_CATEGORY_ORDER) + 1

            sorted_categories = sorted(categories, key=sort_key)

            logger.info(f"Fetched {len(sorted_categories)} scoring categories from ESPN API: {[c['key'] for c in sorted_categories]}")
            return sorted_categories

        except Exception as e:
            logger.error(f"Error fetching scoring categories from API: {e}")
            return self._get_scoring_categories_from_team_stats()

    def _get_scoring_categories_from_team_stats(self) -> List[Dict[str, Any]]:
        """
        Fallback method to extract scoring categories from team stats.

        This is less accurate than the API method as it includes all tracked stats,
        not just scored categories.

        Returns:
            List of category dictionaries
        """
        # Stats to exclude - these are typically not scoring categories
        EXCLUDED_STATS = {'GP', 'FGA', 'FTA', 'MIN', 'MPG', 'GS', '3PA'}

        CATEGORY_METADATA = {
            'PTS': {'label': 'Points', 'abbr': 'PTS', 'is_reverse': False},
            'BLK': {'label': 'Blocks', 'abbr': 'BLK', 'is_reverse': False},
            'STL': {'label': 'Steals', 'abbr': 'STL', 'is_reverse': False},
            'AST': {'label': 'Assists', 'abbr': 'AST', 'is_reverse': False},
            'REB': {'label': 'Rebounds', 'abbr': 'REB', 'is_reverse': False},
            'FGM': {'label': 'Field Goals Made', 'abbr': 'FGM', 'is_reverse': False},
            'FTM': {'label': 'Free Throws Made', 'abbr': 'FTM', 'is_reverse': False},
            '3PM': {'label': '3-Pointers Made', 'abbr': '3PM', 'is_reverse': False},
            'FG%': {'label': 'Field Goal %', 'abbr': 'FG%', 'is_reverse': False},
            'FT%': {'label': 'Free Throw %', 'abbr': 'FT%', 'is_reverse': False},
            '3P%': {'label': '3-Point %', 'abbr': '3P%', 'is_reverse': False},
            'TO': {'label': 'Turnovers', 'abbr': 'TO', 'is_reverse': True},
            'DD': {'label': 'Double-Doubles', 'abbr': 'DD', 'is_reverse': False},
            'TD': {'label': 'Triple-Doubles', 'abbr': 'TD', 'is_reverse': False},
        }

        try:
            if not self.league.teams:
                logger.warning("No teams found in league")
                return []

            first_team = self.league.teams[0]
            raw_stats = getattr(first_team, 'stats', None)

            if not raw_stats or not isinstance(raw_stats, dict):
                logger.warning("No valid stats found on team")
                return []

            # Filter out excluded stats
            stat_keys = set(raw_stats.keys()) - EXCLUDED_STATS

            # Build category list
            categories = []
            for key in stat_keys:
                metadata = CATEGORY_METADATA.get(key, {
                    'label': key,
                    'abbr': key,
                    'is_reverse': False
                })
                categories.append({
                    'key': key,
                    'stat_key': key,  # Alias for compatibility with dashboard
                    'label': metadata.get('label', key),
                    'abbr': metadata.get('abbr', key),
                    'is_reverse': metadata.get('is_reverse', False),
                })

            # Sort by standard order
            def sort_key(cat):
                try:
                    return self.STANDARD_CATEGORY_ORDER.index(cat['key'])
                except ValueError:
                    return len(self.STANDARD_CATEGORY_ORDER) + 1

            sorted_categories = sorted(categories, key=sort_key)

            logger.info(f"Extracted {len(sorted_categories)} categories from team stats (fallback): {[c['key'] for c in sorted_categories]}")
            return sorted_categories

        except Exception as e:
            logger.error(f"Error extracting categories from team stats: {e}")
            return []

    # =========================================================================
    # User Team Identification
    # =========================================================================

    def _normalize_swid(self, swid: str) -> str:
        """
        Normalize SWID format for comparison.
        ESPN SWIDs can be with or without curly braces.
        """
        if not swid:
            return ''
        # Remove curly braces and convert to lowercase for comparison
        return swid.strip('{}').lower()

    def get_user_team_id(self) -> Optional[int]:
        """
        Identify the user's team by matching SWID to team owner ID.

        The SWID cookie (e.g., '{2117988A-EE11-4794-8E44-F3E729DBC9E4}')
        should match the 'id' field in team.owners[0].

        Returns:
            ESPN team ID of the user's team, or None if not found
        """
        try:
            normalized_swid = self._normalize_swid(self.swid)

            if not normalized_swid:
                logger.warning("No SWID provided, cannot identify user team")
                return None

            logger.debug(f"Looking for user team with SWID: {self.swid} (normalized: {normalized_swid})")

            for team in self.league.teams:
                owners = getattr(team, 'owners', [])

                if not owners:
                    logger.debug(f"Team {team.team_id} ({team.team_name}) has no owners list")
                    continue

                # Check each owner (usually just one per team)
                for owner in owners:
                    owner_id = None

                    # Owner can be a dict or an object
                    if isinstance(owner, dict):
                        owner_id = owner.get('id', '')
                    elif hasattr(owner, 'id'):
                        owner_id = getattr(owner, 'id', '')

                    if owner_id:
                        normalized_owner_id = self._normalize_swid(str(owner_id))
                        logger.debug(
                            f"Team {team.team_id} ({team.team_name}): "
                            f"owner_id={owner_id} (normalized: {normalized_owner_id})"
                        )

                        if normalized_owner_id == normalized_swid:
                            logger.info(
                                f"Identified user team: {team.team_name} "
                                f"(ID: {team.team_id}) - Owner ID matches SWID"
                            )
                            return team.team_id

            logger.warning(
                f"Could not find user team matching SWID: {self.swid}. "
                f"Checked {len(self.league.teams)} teams."
            )
            return None

        except Exception as e:
            logger.error(f"Error identifying user team: {e}")
            return None

    def get_user_team(self) -> Optional[Dict[str, Any]]:
        """
        Get the user's team data.

        Returns:
            Team dictionary for user's team, or None if not found
        """
        user_team_id = self.get_user_team_id()

        if not user_team_id:
            return None

        team = self._get_team_by_id(user_team_id)
        if not team:
            return None

        return self._parse_team(team, is_user_team=True)

    def _parse_team(self, team, is_user_team: bool = False) -> Dict[str, Any]:
        """
        Parse an ESPN team object into a dictionary.

        Args:
            team: ESPN team object
            is_user_team: Whether this is the user's team

        Returns:
            Team dictionary with all relevant data
        """
        # Get owner info
        owner_name = 'Unknown'
        owners = getattr(team, 'owners', [])
        if owners:
            first_owner = owners[0]
            if isinstance(first_owner, dict):
                # Try different name fields
                owner_name = (
                    first_owner.get('firstName', '') + ' ' +
                    first_owner.get('lastName', '')
                ).strip()
                if not owner_name or owner_name == ' ':
                    owner_name = first_owner.get('displayName', 'Unknown')
            elif hasattr(first_owner, 'firstName'):
                owner_name = f"{getattr(first_owner, 'firstName', '')} {getattr(first_owner, 'lastName', '')}".strip()

        # Fallback to team.owner attribute
        if owner_name == 'Unknown':
            owner_name = getattr(team, 'owner', 'Unknown')

        # Get team stats for Roto leagues
        # ESPN provides team.stats as a dict with keys like 'PTS', 'REB', 'AST', etc.
        team_stats = {}
        raw_stats = getattr(team, 'stats', None)
        if raw_stats:
            if isinstance(raw_stats, dict):
                team_stats = raw_stats.copy()
            else:
                # Try to convert to dict if it's an object
                try:
                    team_stats = vars(raw_stats) if hasattr(raw_stats, '__dict__') else {}
                except Exception:
                    pass

        return {
            'espn_team_id': team.team_id,
            'team_id': team.team_id,
            'team_name': team.team_name,
            'owner_name': owner_name,
            'wins': team.wins,
            'losses': team.losses,
            'ties': getattr(team, 'ties', 0),
            'standing': team.standing,
            'points_for': getattr(team, 'points_for', 0),
            'points_against': getattr(team, 'points_against', 0),
            'stats': team_stats,
            'is_user_team': is_user_team,
        }

    # =========================================================================
    # Teams and Rosters
    # =========================================================================

    def get_teams(self) -> List[Dict[str, Any]]:
        """
        Get all teams in the league with their basic info.

        Returns:
            List of team dictionaries containing:
                - espn_team_id: ESPN's team ID
                - team_name: Team name
                - owner_name: Owner's display name
                - wins: Current wins
                - losses: Current losses
                - ties: Current ties
                - standing: Current standing position
                - points_for: Total points scored (if applicable)
                - points_against: Total points against (if applicable)
                - is_user_team: True if this is the authenticated user's team
        """
        try:
            teams = []

            # Get user's team ID for marking
            user_team_id = self.get_user_team_id()

            for team in self.league.teams:
                is_user_team = (team.team_id == user_team_id) if user_team_id else False
                team_data = self._parse_team(team, is_user_team=is_user_team)
                teams.append(team_data)

            # Log summary
            user_team = next((t for t in teams if t.get('is_user_team')), None)
            if user_team:
                logger.info(f"User's team: {user_team['team_name']} (ID: {user_team['espn_team_id']})")
            else:
                logger.warning("Could not identify user's team in the league")

            logger.debug(f"Retrieved {len(teams)} teams")
            return teams

        except ESPNClientError:
            raise
        except Exception as e:
            logger.error(f"Error getting teams: {e}")
            raise ESPNClientError(f"Failed to get teams: {e}")

    def get_team_roster(self, team_id: int) -> List[Dict[str, Any]]:
        """
        Get the roster for a specific team.

        Args:
            team_id: ESPN team ID

        Returns:
            List of player dictionaries containing:
                - espn_player_id: ESPN's player ID
                - name: Player name
                - position: Primary position
                - nba_team: NBA team abbreviation
                - injury_status: Injury status if any
                - roster_slot: Current roster slot
                - acquisition_type: How player was acquired
        """
        try:
            team = self._get_team_by_id(team_id)
            if not team:
                raise ESPNClientError(f"Team {team_id} not found")

            roster = []
            for player in team.roster:
                player_data = self._parse_player(player)
                player_data['roster_slot'] = getattr(player, 'slot_position', 'BENCH')
                player_data['acquisition_type'] = getattr(player, 'acquisition_type', 'UNKNOWN')
                roster.append(player_data)

            logger.debug(f"Retrieved roster with {len(roster)} players for team {team_id}")
            return roster

        except ESPNClientError:
            raise
        except Exception as e:
            logger.error(f"Error getting team roster: {e}")
            raise ESPNClientError(f"Failed to get team roster: {e}")

    def get_all_rosters(self) -> Dict[int, List[Dict[str, Any]]]:
        """
        Get rosters for all teams in the league with actual lineup slot IDs.

        Makes a direct ESPN API call to get accurate lineup slot assignments,
        which is needed for IR player detection and start limit optimization.

        Returns:
            Dictionary mapping team_id to list of player dictionaries
        """
        try:
            import requests

            # Fetch raw roster data directly from ESPN API to get accurate lineupSlotId
            endpoint = (
                f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/fba/"
                f"seasons/{self.year}/segments/0/leagues/{self.league_id}"
            )
            # Include kona_player_info for full player stats and kona_playercard for injury details
            params = {'view': ['mTeam', 'mRoster', 'kona_player_info', 'kona_playercard']}
            cookies = {'espn_s2': self.espn_s2, 'SWID': self.swid}

            response = requests.get(endpoint, params=params, cookies=cookies, timeout=15)
            response.raise_for_status()
            raw_data = response.json()

            # NBA team ID mapping
            PRO_TEAM_MAP = {
                1: 'ATL', 2: 'BOS', 3: 'NOP', 4: 'CHI', 5: 'CLE',
                6: 'DAL', 7: 'DEN', 8: 'DET', 9: 'GSW', 10: 'HOU',
                11: 'IND', 12: 'LAC', 13: 'LAL', 14: 'MIA', 15: 'MIL',
                16: 'MIN', 17: 'BKN', 18: 'NYK', 19: 'ORL', 20: 'PHI',
                21: 'PHX', 22: 'POR', 23: 'SAC', 24: 'SAS', 25: 'OKC',
                26: 'UTA', 27: 'WAS', 28: 'TOR', 29: 'MEM', 30: 'CHA',
            }

            SLOT_ID_TO_NAME = {
                0: 'PG', 1: 'SG', 2: 'SF', 3: 'PF', 4: 'C',
                5: 'G', 6: 'F', 7: 'SG/SF', 8: 'G/F', 9: 'PF/C',
                10: 'F/C', 11: 'UTIL', 12: 'BE', 13: 'IR', 14: 'IR+',
            }

            all_rosters = {}

            for team_data in raw_data.get('teams', []):
                team_id = team_data.get('id')
                roster_entries = team_data.get('roster', {}).get('entries', [])
                roster = []

                for entry in roster_entries:
                    # Get actual lineup slot ID from raw ESPN data
                    lineup_slot_id = entry.get('lineupSlotId', 12)

                    # Get player info
                    player_pool = entry.get('playerPoolEntry', {})
                    player = player_pool.get('player', {})

                    player_id = player.get('id', 0)
                    player_name = player.get('fullName', 'Unknown')
                    pro_team_id = player.get('proTeamId', 0)
                    nba_team = PRO_TEAM_MAP.get(pro_team_id, 'FA')
                    position = player.get('defaultPositionId', 0)
                    injury_status = player.get('injuryStatus', 'ACTIVE')

                    # =================================================================
                    # DEBUG: Log raw ESPN API injury fields for specific players
                    # =================================================================
                    is_debug_player = any(x in player_name.lower() for x in ['jaren', 'jackson', 'murphy', 'trey'])
                    if is_debug_player:
                        logger.info(f"=== DEBUG RAW API: {player_name} ===")
                        logger.info(f"  Raw API player keys: {list(player.keys())}")

                        # Check ALL possible injury-related fields in raw API response
                        injury_fields = ['injuryStatus', 'injured', 'injuryDate', 'returnDate',
                                        'expectedReturnDate', 'expected_return_date', 'injuryType',
                                        'injuryNotes', 'seasonOutlook', 'healthStatus', 'status']
                        for field in injury_fields:
                            val = player.get(field)
                            if val is not None:
                                logger.info(f"  {field} = {val}")

                        # Also check playerPoolEntry level for injury data
                        entry_injury_fields = ['injuryStatus', 'status', 'injured']
                        for field in entry_injury_fields:
                            entry_val = entry.get(field)
                            pool_val = player_pool.get(field)
                            if entry_val is not None:
                                logger.info(f"  entry.{field} = {entry_val}")
                            if pool_val is not None:
                                logger.info(f"  playerPoolEntry.{field} = {pool_val}")

                        # CRITICAL: Check for injuryDetails at ALL levels
                        logger.info(f"  >>> INJURY_DETAILS CHECK <<<")
                        logger.info(f"  player.injuryDetails = {player.get('injuryDetails')}")
                        logger.info(f"  entry.injuryDetails = {entry.get('injuryDetails')}")
                        logger.info(f"  playerPoolEntry.injuryDetails = {player_pool.get('injuryDetails')}")

                        # Show the full player dict for this player (truncated)
                        player_str = str(player)[:500]
                        logger.info(f"  Full player dict (truncated): {player_str}")
                        logger.info(f"=== END DEBUG RAW API {player_name} ===")

                    if not injury_status or injury_status == 'ACTIVE':
                        injury_status = None
                    season_outlook = player.get('seasonOutlook', '')

                    # Extract injury notes - ESPN may store in different fields
                    injury_notes = player.get('injuryNotes') or player.get('injury', '')

                    # Extract injuryDetails from ESPN API (kona_playercard view)
                    # Contains: expectedReturnDate (array [year, month, day]), outForSeason (bool), type (str)
                    # IMPORTANT: Only set injury_details if ESPN actually provides injuryDetails data
                    # Check multiple locations - injuryDetails could be at player, entry, or playerPoolEntry level
                    injury_details = None
                    raw_injury_details = (
                        player.get('injuryDetails') or
                        entry.get('injuryDetails') or
                        player_pool.get('injuryDetails')
                    )

                    if raw_injury_details and isinstance(raw_injury_details, dict):
                        injury_details = {}

                        # Extract out_for_season flag
                        out_for_season = raw_injury_details.get('outForSeason', False)
                        injury_details['out_for_season'] = out_for_season

                        # Extract injury type
                        injury_type = raw_injury_details.get('type', '')
                        injury_details['injury_type'] = injury_type

                        # Extract expected return date (array format: [year, month, day])
                        raw_return_date = raw_injury_details.get('expectedReturnDate')
                        if raw_return_date and isinstance(raw_return_date, list) and len(raw_return_date) >= 3:
                            try:
                                return_date_obj = date(raw_return_date[0], raw_return_date[1], raw_return_date[2])
                                injury_details['expected_return_date'] = return_date_obj
                                logger.info(f"EXTRACTED injury_details for {player_name}: "
                                           f"injury_type={injury_type}, return_date={return_date_obj}, "
                                           f"out_for_season={out_for_season}")
                            except (ValueError, TypeError) as e:
                                logger.warning(f"Failed to parse return date for {player_name}: {raw_return_date} - {e}")
                        elif out_for_season:
                            logger.info(f"EXTRACTED injury_details for {player_name}: OUT FOR SEASON "
                                       f"(injury_type={injury_type})")
                        else:
                            logger.info(f"EXTRACTED injury_details for {player_name}: "
                                       f"injury_type={injury_type}, no return date, "
                                       f"out_for_season={out_for_season}")

                        if is_debug_player:
                            logger.info(f"  {player_name} raw injuryDetails: {raw_injury_details}")

                    # IMPORTANT: Only set expected_return_date for INJURED players
                    # This prevents healthy players from getting false return dates
                    expected_return_date = None

                    # Log injury processing for ALL players to help debug
                    logger.debug(f"Player {player_name}: injuryStatus={injury_status}, "
                                f"injury_details={injury_details is not None}")

                    # Only process injury data if player is actually injured
                    if injury_status and injury_status not in ['ACTIVE', None]:
                        # Player is injured - use injuryDetails if available
                        if injury_details and injury_details.get('expected_return_date'):
                            expected_return_date = injury_details['expected_return_date']
                            logger.info(f"Player {player_name}: INJURED, using injuryDetails return_date={expected_return_date}")
                        else:
                            # Fall back to parsing from season_outlook/injury_notes
                            expected_return_date = self._parse_expected_return_date(
                                injury_notes=injury_notes,
                                season_outlook=season_outlook,
                                injury_status=injury_status,
                                player_name=player_name,
                                debug=is_debug_player
                            )
                            if expected_return_date:
                                logger.info(f"Player {player_name}: INJURED, parsed return_date={expected_return_date}")
                    elif injury_details:
                        # Has injuryDetails but ACTIVE status - this is suspicious
                        logger.warning(f"Player {player_name}: ACTIVE but has injury_details={injury_details}")

                    droppable = player.get('droppable', True)

                    # Get eligible slots (list of slot IDs)
                    eligible_slots = player.get('eligibleSlots', [])

                    # Get stats - including projections and last season for IR players
                    stats = {}
                    last_season_avg = {}
                    projected_stats = {}
                    current_season_games = 0

                    for stat_set in player.get('stats', []):
                        stat_source_id = stat_set.get('statSourceId', 0)
                        stat_split_type = stat_set.get('statSplitTypeId', 0)
                        season_id = stat_set.get('seasonId', 0)

                        # statSplitTypeId: 0=season total, 1=last 7, 2=last 15, 3=last 30
                        # We want season total (statSplitTypeId == 0)
                        if stat_source_id == 0 and season_id == self.year and stat_split_type == 0:
                            # Current season actual stats (raw API uses numeric stat IDs as strings)
                            avg_stats = stat_set.get('averageStats', {})
                            total_stats = stat_set.get('stats', {})

                            # Extract games played - PRIORITIZE CALCULATION over direct GP
                            # ESPN's direct GP (stat ID 41) is sometimes wrong
                            calculated_gp = 0
                            direct_gp = 0

                            # Method 1 (PRIMARY): Calculate from PTS total / PTS avg (stat ID 0)
                            if avg_stats and total_stats:
                                pts_avg = float(avg_stats.get('0', 0) or avg_stats.get(0, 0) or 0)
                                pts_total = float(total_stats.get('0', 0) or total_stats.get(0, 0) or 0)
                                if pts_avg > 0 and pts_total > 0:
                                    calculated_gp = int(round(pts_total / pts_avg))

                            # Method 2 (FALLBACK): Try stat ID 41 in stats dict (GP)
                            if total_stats:
                                direct_gp = int(total_stats.get('41', 0) or total_stats.get(41, 0) or 0)

                            # Method 3: Fallback to gamesPlayed/appliedTotal in stat_set
                            if not direct_gp:
                                direct_gp = int(stat_set.get('gamesPlayed', 0) or stat_set.get('appliedTotal', 0) or 0)

                            # Determine final GP with validation
                            if calculated_gp > 0:
                                # Use calculated GP as source of truth
                                if direct_gp > 0 and calculated_gp != direct_gp:
                                    diff_pct = abs(calculated_gp - direct_gp) / max(calculated_gp, direct_gp)
                                    if diff_pct > 0.10:
                                        logger.warning(
                                            f"GP mismatch for {player_name}: calculated={calculated_gp}, "
                                            f"ESPN_reported={direct_gp}, using calculated"
                                        )
                                current_season_games = calculated_gp
                            elif direct_gp > 0:
                                current_season_games = direct_gp
                            else:
                                current_season_games = 0

                            stats['current_season'] = {
                                'average': avg_stats,
                                'total': total_stats,
                                'games_played': current_season_games,
                            }

                            # Also store in format hybrid engine expects: '{season}_total' with 'avg' sub-key
                            # Map numeric stat IDs to named keys
                            ESPN_STAT_ID_TO_NAME = {
                                '0': 'PTS', '1': 'BLK', '2': 'STL', '3': 'AST',
                                '6': 'REB', '17': '3PM', '19': 'TO',
                                '13': 'FGM', '14': 'FGA', '15': 'FTM', '16': 'FTA',
                                '40': 'MIN', '41': 'GP',
                            }
                            avg_named = {}
                            for stat_id, val in avg_stats.items():
                                stat_name = ESPN_STAT_ID_TO_NAME.get(str(stat_id))
                                if stat_name and val is not None:
                                    avg_named[stat_name] = float(val)

                            total_key = f'{self.year}_total'
                            stats[total_key] = {
                                'avg': avg_named,
                                'total': {'GP': current_season_games},
                            }

                            logger.debug(f"Raw API: {player_name} - GP={current_season_games} (from season total)")
                            logger.debug(f"{player_name}: Current season avg keys: {list(avg_named.keys())}")
                        elif stat_source_id == 1 and season_id == self.year:
                            # Current season projected stats
                            projected_stats = stat_set.get('stats', {})
                        elif stat_source_id == 0 and season_id == self.year - 1:
                            # Last season actual stats (fallback for IR players)
                            last_season_avg = stat_set.get('averageStats', {})

                    # Store previous season stats for ALL players (used by hybrid engine)
                    if last_season_avg:
                        stats['previous_season'] = {
                            'average': last_season_avg,
                            'source': 'last_season',
                        }
                        logger.debug(f"{player_name}: Previous season stats extracted ({len(last_season_avg)} stats)")

                    # Store ESPN projected stats for ALL players (used by hybrid engine)
                    # Map numeric stat IDs to named keys that hybrid engine expects
                    ESPN_STAT_ID_TO_NAME = {
                        '0': 'PTS', '1': 'BLK', '2': 'STL', '3': 'AST',
                        '6': 'REB', '17': '3PM', '19': 'TO',
                        '13': 'FGM', '14': 'FGA', '15': 'FTM', '16': 'FTA',
                        '40': 'MIN', '41': 'GP',
                    }
                    if projected_stats:
                        # Projected stats are season totals, convert to per-game
                        est_games = 70  # Estimate full season games
                        proj_avg = {}
                        proj_avg_named = {}  # Named keys for hybrid engine
                        for stat_id, total_val in projected_stats.items():
                            if total_val and isinstance(total_val, (int, float)):
                                per_game_val = total_val / est_games
                                proj_avg[stat_id] = per_game_val
                                # Also store with named key if mapped
                                stat_name = ESPN_STAT_ID_TO_NAME.get(str(stat_id))
                                if stat_name:
                                    proj_avg_named[stat_name] = per_game_val

                        # Store in legacy format for backwards compatibility
                        stats['espn_projection'] = {
                            'average': proj_avg,
                            'source': 'espn_projected',
                        }

                        # Store in format hybrid engine expects: '{season}_projected' with 'avg' sub-key
                        # This matches the raw ESPN player.stats structure
                        proj_key = f'{self.year}_projected'
                        stats[proj_key] = {
                            'avg': proj_avg_named,
                            'total': projected_stats,
                        }

                        extracted_keys = list(proj_avg_named.keys())
                        logger.info(f"Extracted ESPN projections for {player_name}: {extracted_keys}")
                        logger.debug(f"{player_name}: ESPN projection - PTS={proj_avg_named.get('PTS', 0):.1f}, REB={proj_avg_named.get('REB', 0):.1f}, AST={proj_avg_named.get('AST', 0):.1f}")

                    # For IR players with 0 games this season, also use last season or projections as fallback
                    if current_season_games == 0 and (last_season_avg or projected_stats):
                        fallback_avg = {}
                        source = 'unknown'

                        if last_season_avg:
                            # Use last season averages (actual per-game when healthy)
                            fallback_avg = last_season_avg
                            source = 'last_season'
                        elif projected_stats:
                            # Fall back to projections if no last season data
                            est_games = 70
                            for stat_id, total_val in projected_stats.items():
                                fallback_avg[stat_id] = total_val / est_games
                            source = 'projection'

                        stats['projected'] = {
                            'average': fallback_avg,
                            'source': source,
                        }

                    # Compute per_game_stats for easy access
                    # Prefer current season averages, fall back to projected for IR players
                    STAT_ID_MAP = {
                        '0': 'pts', '1': 'blk', '2': 'stl', '3': 'ast',
                        '6': 'reb', '17': '3pm', '19': 'to',
                        '13': 'fgm', '14': 'fga', '15': 'ftm', '16': 'fta'
                    }
                    per_game_stats = {}

                    if stats.get('current_season', {}).get('games_played', 0) > 0:
                        # Use current season averages
                        avg = stats.get('current_season', {}).get('average', {})
                        for stat_id, stat_name in STAT_ID_MAP.items():
                            per_game_stats[stat_name] = float(avg.get(stat_id, 0) or 0)
                    elif 'projected' in stats:
                        # Use projected averages for IR players with 0 games
                        avg = stats.get('projected', {}).get('average', {})
                        for stat_id, stat_name in STAT_ID_MAP.items():
                            per_game_stats[stat_name] = float(avg.get(stat_id, 0) or 0)

                    slot_name = SLOT_ID_TO_NAME.get(lineup_slot_id, 'BENCH')

                    roster.append({
                        'espn_player_id': player_id,
                        'player_id': player_id,  # Also store as player_id for consistency
                        'name': player_name,
                        'position': position,
                        'nba_team': nba_team,
                        'injury_status': injury_status,
                        'injury_notes': injury_notes,
                        'season_outlook': season_outlook,
                        'expected_return_date': expected_return_date,
                        'injury_details': injury_details,  # Contains out_for_season, injury_type, expected_return_date
                        'eligible_slots': eligible_slots,
                        'lineupSlotId': lineup_slot_id,
                        'roster_slot': slot_name,
                        'droppable': droppable,
                        'stats': stats,
                        'per_game_stats': per_game_stats,
                        'games_played': current_season_games,  # Top-level for easy access
                    })

                all_rosters[team_id] = roster

            logger.debug(f"Retrieved rosters for {len(all_rosters)} teams with lineup slots")
            return all_rosters

        except Exception as e:
            logger.warning(f"Direct API roster fetch failed, falling back to espn-api: {e}")
            # Fallback to original method
            return self._get_all_rosters_fallback()

    def _get_all_rosters_fallback(self) -> Dict[int, List[Dict[str, Any]]]:
        """Fallback roster fetch using espn-api library."""
        try:
            all_rosters = {}

            for team in self.league.teams:
                roster = []
                for player in team.roster:
                    lineup_slot_id = getattr(player, 'lineupSlot', 12)
                    slot_position = getattr(player, 'slot_position', 'BENCH')

                    if isinstance(slot_position, str):
                        slot_name_to_id = {
                            'PG': 0, 'SG': 1, 'SF': 2, 'PF': 3, 'C': 4,
                            'G': 5, 'F': 6, 'SG/SF': 7, 'G/F': 8, 'PF/C': 9,
                            'F/C': 10, 'UTIL': 11, 'BE': 12, 'BENCH': 12,
                            'IR': 13, 'IR+': 14,
                        }
                        lineup_slot_id = slot_name_to_id.get(slot_position.upper(), lineup_slot_id)

                    player_data = self._parse_player(player, lineup_slot_id=lineup_slot_id)
                    player_data['roster_slot'] = slot_position
                    player_data['acquisition_type'] = getattr(player, 'acquisition_type', 'UNKNOWN')
                    roster.append(player_data)

                all_rosters[team.team_id] = roster

            return all_rosters

        except Exception as e:
            logger.error(f"Error in fallback roster fetch: {e}")
            raise ESPNClientError(f"Failed to get all rosters: {e}")

    def _get_team_by_id(self, team_id: int):
        """Get a team object by its ESPN ID."""
        for team in self.league.teams:
            if team.team_id == team_id:
                return team
        return None

    def _parse_player(self, player, lineup_slot_id: int = 0) -> Dict[str, Any]:
        """
        Parse an ESPN player object into a dictionary.

        Args:
            player: ESPN player object
            lineup_slot_id: Current lineup slot ID (0=PG, 1=SG, ..., 12=BE, 13=IR)
        """
        player_name = getattr(player, 'name', 'Unknown')

        # =================================================================
        # DEBUG: Log raw ESPN player injury-related fields
        # =================================================================
        # Check for specific players to debug
        is_debug_player = any(x in player_name.lower() for x in ['jaren', 'jackson', 'murphy', 'trey'])

        # Get ALL possible injury-related attributes from ESPN player object
        raw_injury_status = getattr(player, 'injuryStatus', None)
        raw_injured = getattr(player, 'injured', None)
        raw_injury_date = getattr(player, 'injuryDate', None)
        raw_expected_return = getattr(player, 'expected_return_date', None)
        raw_return_date = getattr(player, 'returnDate', None)
        raw_injury_type = getattr(player, 'injuryType', None)

        if is_debug_player:
            logger.info(f"=== DEBUG ESPN PLAYER: {player_name} ===")
            logger.info(f"  Raw ESPN fields for {player_name}:")
            logger.info(f"    injuryStatus = {raw_injury_status}")
            logger.info(f"    injured = {raw_injured}")
            logger.info(f"    injuryDate = {raw_injury_date}")
            logger.info(f"    expected_return_date = {raw_expected_return}")
            logger.info(f"    returnDate = {raw_return_date}")
            logger.info(f"    injuryType = {raw_injury_type}")

            # Show ALL attributes on the player object
            all_attrs = [a for a in dir(player) if not a.startswith('_')]
            injury_related = [a for a in all_attrs if any(x in a.lower() for x in
                             ['injury', 'inj', 'return', 'out', 'status', 'health', 'outlook'])]
            logger.info(f"  All injury-related attributes: {injury_related}")

            # Try to get values for each injury-related attribute
            for attr in injury_related:
                try:
                    val = getattr(player, attr, 'N/A')
                    if val is not None and val != 'N/A':
                        val_str = str(val)[:100] if len(str(val)) > 100 else str(val)
                        logger.info(f"    {attr} = {val_str}")
                except Exception:
                    pass

            # Also check player.__dict__ if available
            if hasattr(player, '__dict__'):
                player_dict = player.__dict__
                injury_keys = [k for k in player_dict.keys() if any(x in k.lower() for x in
                              ['injury', 'inj', 'return', 'out', 'status', 'health'])]
                if injury_keys:
                    logger.info(f"  Player __dict__ injury keys: {injury_keys}")
                    for k in injury_keys:
                        v = player_dict.get(k)
                        v_str = str(v)[:100] if len(str(v)) > 100 else str(v)
                        logger.info(f"    __dict__[{k}] = {v_str}")
            logger.info(f"=== END DEBUG {player_name} ===")

        # Get injury status
        injury_status = getattr(player, 'injuryStatus', 'ACTIVE')
        if injury_status == 'ACTIVE' or not injury_status:
            injury_status = None

        # Get additional injury details if available
        injury_date = getattr(player, 'injuryDate', None)

        # Get injuryDetails from raw ESPN API data (espn-api library does NOT expose this!)
        # We fetch this separately via _get_raw_injury_details which makes a raw HTTP request
        injury_details = self._get_raw_injury_details(player_id)

        if injury_details:
            return_date = injury_details.get('expected_return_date')
            out_for_season = injury_details.get('out_for_season', False)
            injury_type = injury_details.get('injury_type', '')

            if return_date:
                logger.info(f"EXTRACTED injury_details for {player_name}: "
                           f"injury_type={injury_type}, return_date={return_date}, "
                           f"out_for_season={out_for_season}")
            elif out_for_season:
                logger.info(f"EXTRACTED injury_details for {player_name}: OUT FOR SEASON "
                           f"(injury_type={injury_type})")
            else:
                logger.info(f"EXTRACTED injury_details for {player_name}: {injury_details}")

            if is_debug_player:
                logger.info(f"  {player_name} injury_details from raw API: {injury_details}")
        elif is_debug_player:
            logger.info(f"  {player_name}: No injuryDetails in raw ESPN API data")

        # ESPN sometimes provides injury notes/details
        # Try multiple possible attribute names
        injury_notes = None
        for attr in ['injuryNotes', 'injury_notes', 'injuryComment', 'injury']:
            notes = getattr(player, attr, None)
            if notes and isinstance(notes, str):
                injury_notes = notes
                break

        # Get season outlook (used for IR return estimation)
        season_outlook = getattr(player, 'seasonOutlook', '')

        # Get eligible slots (position eligibility)
        # ESPN API provides this as a list of slot IDs (integers)
        # espn-api library may convert them to strings
        raw_eligible_slots = getattr(player, 'eligibleSlots', [])
        if not raw_eligible_slots:
            raw_eligible_slots = getattr(player, 'eligible_slots', [])

        # Convert string slot names to slot IDs if needed
        slot_name_to_id = {
            'PG': 0, 'SG': 1, 'SF': 2, 'PF': 3, 'C': 4,
            'G': 5, 'F': 6, 'SG/SF': 7, 'G/F': 8, 'PF/C': 9,
            'F/C': 10, 'UT': 11, 'UTIL': 11, 'BE': 12, 'BENCH': 12,
            'IR': 13, 'IR+': 14,
        }

        eligible_slots = []
        for slot in raw_eligible_slots:
            if isinstance(slot, int):
                eligible_slots.append(slot)
            elif isinstance(slot, str):
                slot_id = slot_name_to_id.get(slot.upper())
                if slot_id is not None:
                    eligible_slots.append(slot_id)
            # Skip unknown formats

        # Get droppable status
        droppable = getattr(player, 'droppable', True)

        # Get player stats if available and extract games_played
        stats = {}
        games_played = 0
        per_game_stats = {}

        raw_stats = getattr(player, 'stats', None)
        player_name = getattr(player, 'name', 'Unknown')

        if raw_stats and isinstance(raw_stats, dict):
            # Use the extract_games_played function with STRING KEYS
            # Pass player_name for mismatch warning logging
            games_played = extract_games_played(raw_stats, self.year, player_name)
            if games_played > 0:
                logger.debug(f"Extracted GP for {player_name}: {games_played} games")
            else:
                logger.debug(f"No GP found for {player_name} in {self.year}_total stats")

            # Store raw stats for reference
            for stat_period, stat_data in raw_stats.items():
                if isinstance(stat_data, dict):
                    stats[stat_period] = stat_data

                    # Extract per-game stats from current season 'avg' dict
                    if '_total' in str(stat_period) and str(self.year) in str(stat_period):
                        if 'avg' in stat_data and isinstance(stat_data['avg'], dict):
                            avg = stat_data['avg']
                            # ESPN uses STRING KEYS like 'PTS', 'REB', etc.
                            STAT_KEY_MAP = {
                                'PTS': 'pts', 'pts': 'pts',
                                'BLK': 'blk', 'blk': 'blk',
                                'STL': 'stl', 'stl': 'stl',
                                'AST': 'ast', 'ast': 'ast',
                                'REB': 'reb', 'reb': 'reb',
                                'OREB': 'oreb', 'DREB': 'dreb',
                                '3PM': '3pm', '3pm': '3pm', 'TPM': '3pm',
                                'TO': 'to', 'to': 'to', 'TOV': 'to',
                                'FGM': 'fgm', 'fgm': 'fgm',
                                'FGA': 'fga', 'fga': 'fga',
                                'FTM': 'ftm', 'ftm': 'ftm',
                                'FTA': 'fta', 'fta': 'fta',
                                'FG%': 'fg_pct', 'FT%': 'ft_pct', '3P%': '3p_pct',
                                'MIN': 'min', 'MPG': 'min',
                            }
                            for stat_key, stat_name in STAT_KEY_MAP.items():
                                if stat_key in avg and stat_name not in per_game_stats:
                                    per_game_stats[stat_name] = float(avg[stat_key] or 0)

            # Also check for 'current_season' format (from get_all_rosters direct API call)
            if 'current_season' in stats:
                current_season = stats['current_season']
                gp = current_season.get('games_played', 0) or current_season.get('gamesPlayed', 0) or 0
                if gp > games_played:
                    games_played = int(gp)

        # Store games_played in a normalized location
        if 'current_season' not in stats:
            stats['current_season'] = {}
        stats['current_season']['games_played'] = games_played

        # IMPORTANT: Only set expected_return_date for INJURED players
        # This prevents healthy players from getting false return dates
        expected_return_date = None

        # Log injury processing for ALL players to help debug
        logger.debug(f"_parse_player {player_name}: injury_status={injury_status}, "
                    f"injury_details={injury_details is not None}")

        # Only process return date if player is actually injured
        if injury_status and injury_status not in ['ACTIVE', None]:
            # Player is injured - use injuryDetails if available
            if injury_details and injury_details.get('expected_return_date'):
                expected_return_date = injury_details['expected_return_date']
                logger.info(f"_parse_player {player_name}: INJURED, "
                           f"using injuryDetails return_date={expected_return_date}")
            else:
                # Parse expected_return_date from injury notes or season outlook
                expected_return_date = self._parse_expected_return_date(
                    injury_notes=injury_notes,
                    season_outlook=season_outlook,
                    injury_status=injury_status,
                    player_name=player_name,
                    debug=is_debug_player
                )
                if expected_return_date:
                    logger.info(f"_parse_player {player_name}: INJURED, "
                               f"parsed return_date={expected_return_date}")
        elif injury_details:
            # Has injuryDetails but ACTIVE status - this would be suspicious
            logger.warning(f"_parse_player {player_name}: ACTIVE but has injury_details={injury_details}")

        return {
            'espn_player_id': player.playerId,
            'player_id': player.playerId,  # Also store as player_id for consistency
            'name': player.name,
            'position': getattr(player, 'position', 'UNKNOWN'),
            'nba_team': getattr(player, 'proTeam', 'FA'),
            'injury_status': injury_status,
            'injury_date': injury_date,
            'injury_notes': injury_notes,
            'season_outlook': season_outlook,
            'expected_return_date': expected_return_date,
            'injury_details': injury_details,  # Contains out_for_season, injury_type, expected_return_date
            'eligible_slots': eligible_slots,
            'lineupSlotId': lineup_slot_id,
            'droppable': droppable,
            'stats': stats,
            'per_game_stats': per_game_stats,
            'games_played': games_played,  # Top-level for easy access
        }

    def _parse_expected_return_date(
        self,
        injury_notes: Optional[str],
        season_outlook: Optional[str],
        injury_status: Optional[str],
        player_name: str = None,
        debug: bool = False
    ) -> Optional[date]:
        """
        Parse expected return date from injury notes or season outlook.

        Looks for common patterns like:
        - "Expected to return Feb 10"
        - "Out until 2/15"
        - "2-3 weeks"
        - "Day-to-day"

        Args:
            injury_notes: Injury notes/comments from ESPN
            season_outlook: Season outlook text
            injury_status: Current injury status
            player_name: Player name for debug logging
            debug: If True, log parsing details

        Returns:
            Estimated return date or None
        """
        import re
        from datetime import timedelta

        today = date.today()

        # Combine text sources
        text = ' '.join(filter(None, [injury_notes, season_outlook])).lower()

        if debug and player_name:
            logger.info(f"  _parse_expected_return_date for {player_name}:")
            logger.info(f"    injury_notes = '{injury_notes}'")
            logger.info(f"    season_outlook = '{season_outlook}'")
            logger.info(f"    injury_status = '{injury_status}'")
            logger.info(f"    combined text = '{text[:200] if text else 'EMPTY'}'")

        if not text:
            if debug:
                logger.info(f"    -> No text to parse, returning None")
            return None

        # Check for season-ending indicators
        if any(x in text for x in ['out for season', 'season-ending', 'season ending']):
            return None

        # Day-to-day: assume next game (today or tomorrow)
        injury_upper = str(injury_status).upper() if injury_status else ''
        if injury_upper in ['DAY_TO_DAY', 'DAY', 'GTD'] or 'day-to-day' in text:
            return today + timedelta(days=1)

        # Month-based patterns: "return in February", "back in March"
        month_map = {
            'january': 1, 'jan': 1, 'february': 2, 'feb': 2,
            'march': 3, 'mar': 3, 'april': 4, 'apr': 4
        }
        for month_name, month_num in month_map.items():
            if month_name in text:
                # Assume mid-month return
                try:
                    year = self.year if month_num >= 10 else self.year
                    return_date = date(year, month_num, 15)
                    if return_date > today:
                        return return_date
                except ValueError:
                    pass

        # Weeks pattern: "2-3 weeks", "4 weeks"
        weeks_match = re.search(r'(\d+)(?:\s*-\s*(\d+))?\s*weeks?', text)
        if weeks_match:
            min_weeks = int(weeks_match.group(1))
            max_weeks = int(weeks_match.group(2)) if weeks_match.group(2) else min_weeks
            avg_weeks = (min_weeks + max_weeks) / 2
            return today + timedelta(weeks=avg_weeks)

        # Days pattern: "7-10 days"
        days_match = re.search(r'(\d+)(?:\s*-\s*(\d+))?\s*days?', text)
        if days_match:
            min_days = int(days_match.group(1))
            max_days = int(days_match.group(2)) if days_match.group(2) else min_days
            avg_days = (min_days + max_days) / 2
            return today + timedelta(days=avg_days)

        # Date patterns: "Feb 10", "2/15"
        date_patterns = [
            r'(?:return|back).*?(\w+\.?\s+\d{1,2})',
            r'(?:return|back).*?(\d{1,2}[/-]\d{1,2})',
        ]
        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                date_str = match.group(1)
                parsed = self._parse_date_string(date_str)
                if parsed and parsed > today:
                    return parsed

        return None

    def _parse_date_string(self, date_str: str) -> Optional[date]:
        """Parse a date string like 'Feb 10' or '2/10' into a date object."""
        import re

        month_map = {
            'jan': 1, 'january': 1, 'feb': 2, 'february': 2,
            'mar': 3, 'march': 3, 'apr': 4, 'april': 4,
            'may': 5, 'jun': 6, 'june': 6, 'jul': 7, 'july': 7,
            'aug': 8, 'august': 8, 'sep': 9, 'september': 9,
            'oct': 10, 'october': 10, 'nov': 11, 'november': 11,
            'dec': 12, 'december': 12
        }

        date_str = date_str.lower().replace('.', '').strip()
        today = date.today()

        try:
            # Try "2/10" or "2-10" format
            if '/' in date_str or '-' in date_str:
                parts = re.split(r'[/-]', date_str)
                if len(parts) >= 2:
                    month = int(parts[0])
                    day = int(parts[1])
                    result = date(self.year, month, day)
                    if result < today:
                        result = date(self.year + 1, month, day)
                    return result

            # Try "Feb 10" format
            parts = date_str.split()
            if len(parts) >= 2:
                month_str = parts[0].replace('.', '')
                month = month_map.get(month_str)
                if month:
                    day = int(re.search(r'\d+', parts[1]).group())
                    result = date(self.year, month, day)
                    if result < today:
                        result = date(self.year + 1, month, day)
                    return result

        except (ValueError, AttributeError):
            pass

        return None

    # =========================================================================
    # Free Agents
    # =========================================================================

    def get_free_agents(
        self,
        size: int = 50,
        position: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get available free agents.

        Args:
            size: Number of free agents to fetch (default 50)
            position: Filter by position (PG, SG, SF, PF, C) or None for all

        Returns:
            List of player dictionaries with stats and projections
        """
        try:
            # Map position strings to ESPN position IDs if needed
            position_filter = None
            if position:
                position_map = {
                    'PG': 1, 'SG': 2, 'SF': 3, 'PF': 4, 'C': 5,
                    'G': 6, 'F': 7, 'UTIL': 8
                }
                position_filter = position_map.get(position.upper())

            # Fetch free agents
            if position_filter:
                free_agents = self.league.free_agents(size=size, position=position_filter)
            else:
                free_agents = self.league.free_agents(size=size)

            result = []
            for player in free_agents:
                player_data = self._parse_player(player)
                player_data['percent_owned'] = getattr(player, 'percent_owned', 0)
                player_data['percent_started'] = getattr(player, 'percent_started', 0)
                result.append(player_data)

            logger.debug(f"Retrieved {len(result)} free agents")
            return result

        except ESPNClientError:
            raise
        except Exception as e:
            logger.error(f"Error getting free agents: {e}")
            raise ESPNClientError(f"Failed to get free agents: {e}")

    # =========================================================================
    # Standings
    # =========================================================================

    def get_standings(self) -> List[Dict[str, Any]]:
        """
        Get current league standings.

        Returns:
            List of team standings sorted by position, containing:
                - standing: Current position
                - espn_team_id: ESPN team ID
                - team_name: Team name
                - owner_name: Owner's display name
                - record: "W-L-T" format
                - wins: Number of wins
                - losses: Number of losses
                - ties: Number of ties
                - win_pct: Winning percentage
                - games_back: Games behind first place
                - points_for: Total points scored
                - points_against: Total points against
                - is_user_team: True if this is the user's team
        """
        try:
            standings = []

            # Get user's team ID for marking
            user_team_id = self.get_user_team_id()

            # Get teams sorted by standing
            sorted_teams = sorted(self.league.teams, key=lambda t: t.standing)

            first_place_wins = sorted_teams[0].wins if sorted_teams else 0
            first_place_losses = sorted_teams[0].losses if sorted_teams else 0

            for team in sorted_teams:
                total_games = team.wins + team.losses + getattr(team, 'ties', 0)
                win_pct = team.wins / total_games if total_games > 0 else 0.0

                # Calculate games back
                games_back = ((first_place_wins - team.wins) + (team.losses - first_place_losses)) / 2

                # Get owner name from owners list
                owner_name = 'Unknown'
                owners = getattr(team, 'owners', [])
                if owners:
                    first_owner = owners[0]
                    if isinstance(first_owner, dict):
                        owner_name = (
                            first_owner.get('firstName', '') + ' ' +
                            first_owner.get('lastName', '')
                        ).strip()
                        if not owner_name or owner_name == ' ':
                            owner_name = first_owner.get('displayName', 'Unknown')
                if owner_name == 'Unknown':
                    owner_name = getattr(team, 'owner', 'Unknown')

                is_user_team = (team.team_id == user_team_id) if user_team_id else False

                standings.append({
                    'standing': team.standing,
                    'espn_team_id': team.team_id,
                    'team_id': team.team_id,
                    'team_name': team.team_name,
                    'owner_name': owner_name,
                    'record': f"{team.wins}-{team.losses}-{getattr(team, 'ties', 0)}",
                    'wins': team.wins,
                    'losses': team.losses,
                    'ties': getattr(team, 'ties', 0),
                    'win_pct': round(win_pct, 3),
                    'games_back': games_back if games_back > 0 else 0,
                    'points_for': getattr(team, 'points_for', 0),
                    'points_against': getattr(team, 'points_against', 0),
                    'is_user_team': is_user_team,
                })

            logger.debug(f"Retrieved standings for {len(standings)} teams")
            return standings

        except ESPNClientError:
            raise
        except Exception as e:
            logger.error(f"Error getting standings: {e}")
            raise ESPNClientError(f"Failed to get standings: {e}")

    # =========================================================================
    # Matchups (H2H Leagues)
    # =========================================================================

    def get_matchups(self, week: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get matchup data for H2H leagues.

        Args:
            week: Matchup period/week number (defaults to current week)

        Returns:
            List of matchup dictionaries containing:
                - week: Matchup period
                - home_team_id: Home team ESPN ID
                - home_team_name: Home team name
                - home_score: Home team score/points
                - away_team_id: Away team ESPN ID
                - away_team_name: Away team name
                - away_score: Away team score/points
                - is_playoff: Whether this is a playoff matchup
                - winner: 'home', 'away', or None if ongoing
        """
        try:
            if week is None:
                week = self.league.current_week

            # Get scoreboard for the specified week
            scoreboard = self.league.scoreboard(matchup_period=week)

            matchups = []
            for matchup in scoreboard:
                home_team = matchup.home_team
                away_team = matchup.away_team

                # Determine winner if matchup is complete
                winner = None
                home_score = getattr(matchup, 'home_score', 0) or 0
                away_score = getattr(matchup, 'away_score', 0) or 0

                if home_score != away_score and home_score > 0 and away_score > 0:
                    winner = 'home' if home_score > away_score else 'away'

                matchup_data = {
                    'week': week,
                    'home_team_id': home_team.team_id,
                    'home_team_name': home_team.team_name,
                    'home_score': home_score,
                    'away_team_id': away_team.team_id if away_team else None,
                    'away_team_name': away_team.team_name if away_team else 'BYE',
                    'away_score': away_score,
                    'is_playoff': getattr(matchup, 'is_playoff', False),
                    'winner': winner,
                }
                matchups.append(matchup_data)

            logger.debug(f"Retrieved {len(matchups)} matchups for week {week}")
            return matchups

        except ESPNClientError:
            raise
        except Exception as e:
            logger.error(f"Error getting matchups: {e}")
            raise ESPNClientError(f"Failed to get matchups: {e}")

    def get_matchup_details(
        self,
        week: Optional[int] = None,
        team_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get detailed matchup data including category breakdowns.

        Args:
            week: Matchup period (defaults to current week)
            team_id: Filter to specific team's matchup

        Returns:
            List of detailed matchup dictionaries with category scores
        """
        try:
            if week is None:
                week = self.league.current_week

            scoreboard = self.league.scoreboard(matchup_period=week)

            matchups = []
            for matchup in scoreboard:
                # Filter by team if specified
                if team_id:
                    if (matchup.home_team.team_id != team_id and
                        (not matchup.away_team or matchup.away_team.team_id != team_id)):
                        continue

                matchup_data = {
                    'week': week,
                    'home_team': {
                        'team_id': matchup.home_team.team_id,
                        'team_name': matchup.home_team.team_name,
                        'score': getattr(matchup, 'home_score', 0),
                    },
                    'away_team': {
                        'team_id': matchup.away_team.team_id if matchup.away_team else None,
                        'team_name': matchup.away_team.team_name if matchup.away_team else 'BYE',
                        'score': getattr(matchup, 'away_score', 0),
                    } if matchup.away_team else None,
                }

                # Add category breakdown if available (for category leagues)
                if hasattr(matchup, 'home_team_cats') and matchup.home_team_cats:
                    matchup_data['home_team']['categories'] = self._parse_category_scores(
                        matchup.home_team_cats
                    )

                if hasattr(matchup, 'away_team_cats') and matchup.away_team_cats:
                    matchup_data['away_team']['categories'] = self._parse_category_scores(
                        matchup.away_team_cats
                    )

                matchups.append(matchup_data)

            return matchups

        except ESPNClientError:
            raise
        except Exception as e:
            logger.error(f"Error getting matchup details: {e}")
            raise ESPNClientError(f"Failed to get matchup details: {e}")

    def _parse_category_scores(self, categories) -> Dict[str, Any]:
        """Parse category scores from ESPN matchup data."""
        result = {}
        try:
            if isinstance(categories, dict):
                return categories

            # Handle list of category objects
            for cat in categories:
                if hasattr(cat, 'stat_abbr') and hasattr(cat, 'value'):
                    result[cat.stat_abbr] = cat.value
        except Exception as e:
            logger.warning(f"Could not parse category scores: {e}")

        return result

    # =========================================================================
    # Player Stats
    # =========================================================================

    def get_player_stats(self, player_id: int) -> Optional[Dict[str, Any]]:
        """
        Get detailed stats for a specific player.

        Args:
            player_id: ESPN player ID

        Returns:
            Player dictionary with full stats or None if not found
        """
        try:
            # Search through all rosters
            for team in self.league.teams:
                for player in team.roster:
                    if player.playerId == player_id:
                        return self._parse_player(player)

            # Search free agents if not found on rosters
            free_agents = self.league.free_agents(size=100)
            for player in free_agents:
                if player.playerId == player_id:
                    return self._parse_player(player)

            return None

        except Exception as e:
            logger.error(f"Error getting player stats: {e}")
            return None

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_league_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of the league.

        Returns:
            Dictionary with settings, standings, and current week matchups
        """
        try:
            return {
                'settings': self.get_league_settings(),
                'standings': self.get_standings(),
                'current_matchups': self.get_matchups(),
                'total_teams': len(self.league.teams),
                'current_week': self.league.current_week,
            }
        except ESPNClientError:
            raise
        except Exception as e:
            logger.error(f"Error getting league summary: {e}")
            raise ESPNClientError(f"Failed to get league summary: {e}")

    def validate_connection(self) -> bool:
        """
        Validate that the ESPN connection is working.

        Returns:
            True if connection is valid, raises exception otherwise
        """
        try:
            # Try to access league name as a simple validation
            _ = self.league.settings.name
            return True
        except Exception as e:
            logger.error(f"Connection validation failed: {e}")
            raise ESPNConnectionError(f"Connection validation failed: {e}")


# =============================================================================
# Factory Function
# =============================================================================

def create_espn_client(
    league_id: int,
    year: int,
    espn_s2: str,
    swid: str
) -> ESPNClient:
    """
    Factory function to create an ESPN client with validation.

    Args:
        league_id: ESPN league ID
        year: Season year
        espn_s2: ESPN_S2 cookie
        swid: SWID cookie

    Returns:
        Configured ESPNClient instance

    Raises:
        ESPNAuthenticationError: If credentials are invalid
        ESPNLeagueNotFoundError: If league doesn't exist
        ESPNConnectionError: If connection fails
    """
    return ESPNClient(
        league_id=league_id,
        year=year,
        espn_s2=espn_s2,
        swid=swid
    )

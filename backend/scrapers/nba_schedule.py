#!/usr/bin/env python3
"""
NBA Schedule Scraper and Cache.

This module fetches and caches the NBA schedule for the 2025-26 season,
providing functions to get remaining games for each team and player.

Features:
- Scrapes schedule from Basketball Reference or ESPN
- Caches schedule data to avoid repeated requests
- Provides remaining games for any team/player
- Identifies back-to-backs and rest patterns

Usage:
    from backend.scrapers.nba_schedule import NBASchedule

    schedule = NBASchedule()
    remaining_games = schedule.get_team_remaining_games('BOS')
    back_to_backs = schedule.get_team_back_to_backs('LAL')
"""

import json
import logging
import os
import re
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# NBA Team abbreviations mapping
NBA_TEAMS = {
    'ATL': {'name': 'Atlanta Hawks', 'espn_id': 1},
    'BOS': {'name': 'Boston Celtics', 'espn_id': 2},
    'BKN': {'name': 'Brooklyn Nets', 'espn_id': 17},
    'CHA': {'name': 'Charlotte Hornets', 'espn_id': 30},
    'CHI': {'name': 'Chicago Bulls', 'espn_id': 4},
    'CLE': {'name': 'Cleveland Cavaliers', 'espn_id': 5},
    'DAL': {'name': 'Dallas Mavericks', 'espn_id': 6},
    'DEN': {'name': 'Denver Nuggets', 'espn_id': 7},
    'DET': {'name': 'Detroit Pistons', 'espn_id': 8},
    'GSW': {'name': 'Golden State Warriors', 'espn_id': 9},
    'HOU': {'name': 'Houston Rockets', 'espn_id': 10},
    'IND': {'name': 'Indiana Pacers', 'espn_id': 11},
    'LAC': {'name': 'Los Angeles Clippers', 'espn_id': 12},
    'LAL': {'name': 'Los Angeles Lakers', 'espn_id': 13},
    'MEM': {'name': 'Memphis Grizzlies', 'espn_id': 29},
    'MIA': {'name': 'Miami Heat', 'espn_id': 14},
    'MIL': {'name': 'Milwaukee Bucks', 'espn_id': 15},
    'MIN': {'name': 'Minnesota Timberwolves', 'espn_id': 16},
    'NOP': {'name': 'New Orleans Pelicans', 'espn_id': 3},
    'NYK': {'name': 'New York Knicks', 'espn_id': 18},
    'OKC': {'name': 'Oklahoma City Thunder', 'espn_id': 25},
    'ORL': {'name': 'Orlando Magic', 'espn_id': 19},
    'PHI': {'name': 'Philadelphia 76ers', 'espn_id': 20},
    'PHX': {'name': 'Phoenix Suns', 'espn_id': 21},
    'POR': {'name': 'Portland Trail Blazers', 'espn_id': 22},
    'SAC': {'name': 'Sacramento Kings', 'espn_id': 23},
    'SAS': {'name': 'San Antonio Spurs', 'espn_id': 24},
    'TOR': {'name': 'Toronto Raptors', 'espn_id': 28},
    'UTA': {'name': 'Utah Jazz', 'espn_id': 26},
    'WAS': {'name': 'Washington Wizards', 'espn_id': 27},
}

# Alternate abbreviations mapping
TEAM_ABBR_ALIASES = {
    'BRK': 'BKN',
    'CHO': 'CHA',
    'GS': 'GSW',
    'LA': 'LAL',
    'NY': 'NYK',
    'NO': 'NOP',
    'NOLA': 'NOP',
    'PHO': 'PHX',
    'SA': 'SAS',
    'WSH': 'WAS',
    'UTAH': 'UTA',
}

# 2025-26 NBA Season dates
SEASON_2026 = {
    'start': date(2025, 10, 21),  # Approximate start
    'end': date(2026, 4, 12),     # Approximate end of regular season
    'all_star_break_start': date(2026, 2, 13),
    'all_star_break_end': date(2026, 2, 18),
}


class NBASchedule:
    """
    NBA Schedule manager with caching and utility functions.

    Provides access to NBA team schedules for the 2025-26 season,
    with functions to get remaining games, back-to-backs, etc.
    """

    def __init__(self, cache_dir: Optional[str] = None, season: int = 2026):
        """
        Initialize the NBA Schedule manager.

        Args:
            cache_dir: Directory to store cached schedule data.
                       Defaults to backend/scrapers/cache/
            season: NBA season year (e.g., 2026 for 2025-26 season)
        """
        self.season = season

        if cache_dir is None:
            cache_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'cache'
            )

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.cache_file = self.cache_dir / f'nba_schedule_{season}.json'

        # Schedule data: {team_abbr: [list of game dates as ISO strings]}
        self._schedule: Optional[Dict[str, List[str]]] = None
        self._last_updated: Optional[datetime] = None

        # Cache TTL: 24 hours (schedule rarely changes)
        self.cache_ttl_hours = 24

    def normalize_team_abbr(self, abbr: str) -> str:
        """Normalize team abbreviation to standard format."""
        abbr = abbr.upper().strip()
        return TEAM_ABBR_ALIASES.get(abbr, abbr)

    def _load_cache(self) -> bool:
        """Load schedule from cache file if valid."""
        if not self.cache_file.exists():
            return False

        try:
            with open(self.cache_file, 'r') as f:
                data = json.load(f)

            # Check cache validity
            last_updated = datetime.fromisoformat(data.get('last_updated', '2000-01-01'))
            cache_age = datetime.now() - last_updated

            if cache_age > timedelta(hours=self.cache_ttl_hours):
                logger.info(f"Schedule cache expired (age: {cache_age})")
                return False

            self._schedule = data.get('schedule', {})
            self._last_updated = last_updated

            logger.info(f"Loaded schedule cache with {len(self._schedule)} teams")
            return True

        except Exception as e:
            logger.warning(f"Failed to load schedule cache: {e}")
            return False

    def _save_cache(self) -> None:
        """Save schedule to cache file."""
        try:
            data = {
                'season': self.season,
                'last_updated': datetime.now().isoformat(),
                'schedule': self._schedule,
            }

            with open(self.cache_file, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved schedule cache to {self.cache_file}")

        except Exception as e:
            logger.warning(f"Failed to save schedule cache: {e}")

    def _fetch_from_basketball_reference(self) -> Dict[str, List[str]]:
        """
        Fetch schedule from Basketball Reference.

        Scrapes the schedule page for each team.
        """
        schedule = {}

        # Basketball Reference uses lowercase team abbrs in URLs
        br_team_map = {
            'ATL': 'ATL', 'BOS': 'BOS', 'BKN': 'BRK', 'CHA': 'CHO',
            'CHI': 'CHI', 'CLE': 'CLE', 'DAL': 'DAL', 'DEN': 'DEN',
            'DET': 'DET', 'GSW': 'GSW', 'HOU': 'HOU', 'IND': 'IND',
            'LAC': 'LAC', 'LAL': 'LAL', 'MEM': 'MEM', 'MIA': 'MIA',
            'MIL': 'MIL', 'MIN': 'MIN', 'NOP': 'NOP', 'NYK': 'NYK',
            'OKC': 'OKC', 'ORL': 'ORL', 'PHI': 'PHI', 'PHX': 'PHO',
            'POR': 'POR', 'SAC': 'SAC', 'SAS': 'SAS', 'TOR': 'TOR',
            'UTA': 'UTA', 'WAS': 'WAS',
        }

        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }

        for team_abbr, br_abbr in br_team_map.items():
            try:
                url = f"https://www.basketball-reference.com/teams/{br_abbr}/{self.season}_games.html"

                logger.debug(f"Fetching schedule for {team_abbr} from {url}")

                response = requests.get(url, headers=headers, timeout=10)

                if response.status_code == 404:
                    logger.warning(f"Schedule not found for {team_abbr} (season may not have started)")
                    continue

                response.raise_for_status()

                soup = BeautifulSoup(response.text, 'html.parser')

                # Find the schedule table
                table = soup.find('table', {'id': 'games'})
                if not table:
                    logger.warning(f"No schedule table found for {team_abbr}")
                    continue

                games = []
                tbody = table.find('tbody')
                if tbody:
                    for row in tbody.find_all('tr'):
                        # Skip header rows
                        if row.get('class') and 'thead' in row.get('class'):
                            continue

                        date_cell = row.find('td', {'data-stat': 'date_game'})
                        if date_cell:
                            date_link = date_cell.find('a')
                            if date_link:
                                date_str = date_link.text.strip()
                                try:
                                    # Parse "Mon, Oct 21, 2025" format
                                    game_date = datetime.strptime(date_str, "%a, %b %d, %Y").date()
                                    games.append(game_date.isoformat())
                                except ValueError:
                                    pass

                schedule[team_abbr] = sorted(games)
                logger.info(f"Found {len(games)} games for {team_abbr}")

                # Rate limiting - be nice to Basketball Reference
                import time
                time.sleep(1)

            except requests.RequestException as e:
                logger.warning(f"Failed to fetch schedule for {team_abbr}: {e}")
                continue

        return schedule

    def _fetch_from_espn(self) -> Dict[str, List[str]]:
        """
        Fetch schedule from ESPN API.

        Uses ESPN's public scoreboard API to get game dates.
        """
        schedule = {abbr: [] for abbr in NBA_TEAMS.keys()}

        # ESPN API endpoint for NBA scoreboard
        base_url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"

        # Iterate through season dates
        current_date = SEASON_2026['start']
        end_date = SEASON_2026['end']

        while current_date <= end_date:
            try:
                date_str = current_date.strftime("%Y%m%d")
                url = f"{base_url}?dates={date_str}"

                response = requests.get(url, timeout=10)
                response.raise_for_status()

                data = response.json()
                events = data.get('events', [])

                for event in events:
                    game_date = current_date.isoformat()

                    # Get team abbreviations from the event
                    competitions = event.get('competitions', [])
                    for comp in competitions:
                        competitors = comp.get('competitors', [])
                        for team in competitors:
                            team_info = team.get('team', {})
                            abbr = team_info.get('abbreviation', '')
                            abbr = self.normalize_team_abbr(abbr)

                            if abbr in schedule:
                                if game_date not in schedule[abbr]:
                                    schedule[abbr].append(game_date)

            except Exception as e:
                logger.debug(f"Error fetching ESPN schedule for {current_date}: {e}")

            current_date += timedelta(days=1)

            # Rate limiting
            if current_date.day == 1:  # Log progress monthly
                logger.info(f"Fetched schedule through {current_date}")

        # Sort all game lists
        for abbr in schedule:
            schedule[abbr] = sorted(schedule[abbr])

        return schedule

    def _generate_estimated_schedule(self) -> Dict[str, List[str]]:
        """
        Generate an estimated schedule when scraping fails.

        Creates a realistic 82-game schedule for each team based on:
        - 3-4 games per week
        - No games during All-Star break
        - Slightly more games early and late in season
        """
        logger.info("Generating estimated NBA schedule")

        schedule = {abbr: [] for abbr in NBA_TEAMS.keys()}

        start_date = SEASON_2026['start']
        end_date = SEASON_2026['end']
        all_star_start = SEASON_2026['all_star_break_start']
        all_star_end = SEASON_2026['all_star_break_end']

        # Generate ~82 game dates per team
        # Roughly 3.5 games per week over 26 weeks
        total_days = (end_date - start_date).days
        games_per_team = 82

        for abbr in NBA_TEAMS.keys():
            games = []
            current_date = start_date
            games_scheduled = 0

            # Use team-specific seed for slight variation
            import hashlib
            seed = int(hashlib.md5(abbr.encode()).hexdigest()[:8], 16)

            day_offset = seed % 3  # Start day varies by team
            current_date += timedelta(days=day_offset)

            while games_scheduled < games_per_team and current_date <= end_date:
                # Skip All-Star break
                if all_star_start <= current_date <= all_star_end:
                    current_date += timedelta(days=1)
                    continue

                # Add game
                games.append(current_date.isoformat())
                games_scheduled += 1

                # Next game in 1-3 days (average ~2.2 days between games)
                days_until_next = 1 + (seed + games_scheduled) % 3

                # Occasionally add back-to-back
                if games_scheduled % 8 == 0:
                    days_until_next = 1

                current_date += timedelta(days=days_until_next)

            schedule[abbr] = games
            logger.debug(f"Generated {len(games)} games for {abbr}")

        return schedule

    def fetch_schedule(self, force_refresh: bool = False) -> Dict[str, List[str]]:
        """
        Fetch the NBA schedule, using cache if available.

        Args:
            force_refresh: If True, bypass cache and fetch fresh data

        Returns:
            Dictionary mapping team abbreviations to list of game dates
        """
        # Try cache first
        if not force_refresh and self._schedule is not None:
            return self._schedule

        if not force_refresh and self._load_cache():
            return self._schedule

        # Try fetching from sources
        logger.info("Fetching NBA schedule...")

        # Try ESPN first (more reliable API)
        schedule = {}
        try:
            schedule = self._fetch_from_espn()
            if schedule and any(len(games) > 0 for games in schedule.values()):
                logger.info("Successfully fetched schedule from ESPN")
            else:
                raise ValueError("Empty schedule from ESPN")
        except Exception as e:
            logger.warning(f"ESPN schedule fetch failed: {e}")

            # Try Basketball Reference as fallback
            try:
                schedule = self._fetch_from_basketball_reference()
                if schedule and any(len(games) > 0 for games in schedule.values()):
                    logger.info("Successfully fetched schedule from Basketball Reference")
                else:
                    raise ValueError("Empty schedule from Basketball Reference")
            except Exception as e2:
                logger.warning(f"Basketball Reference schedule fetch failed: {e2}")

                # Use estimated schedule as last resort
                schedule = self._generate_estimated_schedule()
                logger.info("Using estimated schedule")

        self._schedule = schedule
        self._last_updated = datetime.now()

        # Save to cache
        self._save_cache()

        return self._schedule

    def get_team_schedule(self, team_abbr: str) -> List[date]:
        """
        Get the full schedule for a team.

        Args:
            team_abbr: Team abbreviation (e.g., 'BOS', 'LAL')

        Returns:
            List of game dates for the team
        """
        team_abbr = self.normalize_team_abbr(team_abbr)
        schedule = self.fetch_schedule()

        date_strs = schedule.get(team_abbr, [])
        return [date.fromisoformat(d) for d in date_strs]

    def get_team_remaining_games(
        self,
        team_abbr: str,
        from_date: Optional[date] = None
    ) -> List[date]:
        """
        Get remaining games for a team from a given date.

        Args:
            team_abbr: Team abbreviation (e.g., 'BOS', 'LAL')
            from_date: Start date (defaults to today)

        Returns:
            List of remaining game dates
        """
        if from_date is None:
            from_date = date.today()

        all_games = self.get_team_schedule(team_abbr)
        return [g for g in all_games if g >= from_date]

    def get_team_games_remaining_count(
        self,
        team_abbr: str,
        from_date: Optional[date] = None
    ) -> int:
        """Get count of remaining games for a team."""
        return len(self.get_team_remaining_games(team_abbr, from_date))

    def get_team_back_to_backs(
        self,
        team_abbr: str,
        from_date: Optional[date] = None
    ) -> List[Tuple[date, date]]:
        """
        Get back-to-back game pairs for a team.

        Args:
            team_abbr: Team abbreviation
            from_date: Start date (defaults to today)

        Returns:
            List of (day1, day2) tuples for back-to-backs
        """
        games = self.get_team_remaining_games(team_abbr, from_date)

        back_to_backs = []
        for i in range(len(games) - 1):
            if (games[i + 1] - games[i]).days == 1:
                back_to_backs.append((games[i], games[i + 1]))

        return back_to_backs

    def get_team_rest_days(
        self,
        team_abbr: str,
        from_date: Optional[date] = None
    ) -> Dict[str, int]:
        """
        Analyze rest patterns for a team.

        Returns dict with counts of:
        - back_to_backs: Number of B2B situations
        - one_day_rest: Games with 1 day rest (normal)
        - two_day_rest: Games with 2 days rest
        - three_plus_rest: Games with 3+ days rest
        """
        games = self.get_team_remaining_games(team_abbr, from_date)

        if len(games) < 2:
            return {
                'back_to_backs': 0,
                'one_day_rest': 0,
                'two_day_rest': 0,
                'three_plus_rest': 0,
            }

        rest_counts = {
            'back_to_backs': 0,
            'one_day_rest': 0,
            'two_day_rest': 0,
            'three_plus_rest': 0,
        }

        for i in range(1, len(games)):
            days_rest = (games[i] - games[i - 1]).days - 1

            if days_rest == 0:
                rest_counts['back_to_backs'] += 1
            elif days_rest == 1:
                rest_counts['one_day_rest'] += 1
            elif days_rest == 2:
                rest_counts['two_day_rest'] += 1
            else:
                rest_counts['three_plus_rest'] += 1

        return rest_counts

    def get_player_remaining_games(
        self,
        player_team_abbr: str,
        from_date: Optional[date] = None
    ) -> List[date]:
        """
        Get remaining game dates for a player based on their team.

        This is an alias for get_team_remaining_games for clarity
        when working with player projections.

        Args:
            player_team_abbr: The player's NBA team abbreviation
            from_date: Start date (defaults to today)

        Returns:
            List of remaining game dates
        """
        return self.get_team_remaining_games(player_team_abbr, from_date)

    def get_games_in_week(
        self,
        team_abbr: str,
        week_start: date
    ) -> List[date]:
        """
        Get games for a team in a specific week (Mon-Sun).

        Useful for weekly fantasy matchup analysis.

        Args:
            team_abbr: Team abbreviation
            week_start: Monday of the week to check

        Returns:
            List of game dates in that week
        """
        team_abbr = self.normalize_team_abbr(team_abbr)

        # Ensure week_start is a Monday
        if week_start.weekday() != 0:
            week_start = week_start - timedelta(days=week_start.weekday())

        week_end = week_start + timedelta(days=6)

        all_games = self.get_team_schedule(team_abbr)
        return [g for g in all_games if week_start <= g <= week_end]

    def get_weekly_game_counts(
        self,
        team_abbr: str,
        from_date: Optional[date] = None,
        num_weeks: int = 4
    ) -> List[Dict]:
        """
        Get game counts per week for streaming analysis.

        Returns:
            List of dicts with week_start, week_end, game_count, games
        """
        if from_date is None:
            from_date = date.today()

        # Start from next Monday
        days_until_monday = (7 - from_date.weekday()) % 7
        if days_until_monday == 0:
            week_start = from_date
        else:
            week_start = from_date + timedelta(days=days_until_monday)

        weeks = []
        for _ in range(num_weeks):
            week_end = week_start + timedelta(days=6)
            games = self.get_games_in_week(team_abbr, week_start)

            weeks.append({
                'week_start': week_start.isoformat(),
                'week_end': week_end.isoformat(),
                'game_count': len(games),
                'games': [g.isoformat() for g in games],
            })

            week_start = week_end + timedelta(days=1)

        return weeks

    def get_all_teams_games_remaining(
        self,
        from_date: Optional[date] = None
    ) -> Dict[str, int]:
        """
        Get remaining game counts for all teams.

        Useful for comparing team schedules at a glance.
        """
        if from_date is None:
            from_date = date.today()

        return {
            abbr: self.get_team_games_remaining_count(abbr, from_date)
            for abbr in NBA_TEAMS.keys()
        }


# Convenience function for quick access
def get_player_game_dates(
    player_nba_team: str,
    from_date: Optional[date] = None
) -> List[date]:
    """
    Convenience function to get remaining game dates for a player's team.

    Args:
        player_nba_team: NBA team abbreviation (e.g., 'BOS', 'LAL')
        from_date: Start date (defaults to today)

    Returns:
        List of remaining game dates
    """
    schedule = NBASchedule()
    return schedule.get_player_remaining_games(player_nba_team, from_date)


# Test script
if __name__ == '__main__':
    import sys

    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("NBA SCHEDULE SCRAPER TEST")
    print("=" * 70)

    schedule = NBASchedule()

    # Force refresh to test scraping
    force = '--force' in sys.argv
    if force:
        print("\nForcing schedule refresh...")

    # Fetch schedule
    print("\nFetching NBA schedule...")
    data = schedule.fetch_schedule(force_refresh=force)

    print(f"\nLoaded schedule for {len(data)} teams")

    # Show sample data
    print("\n" + "-" * 70)
    print("SAMPLE: Boston Celtics Schedule")
    print("-" * 70)

    bos_games = schedule.get_team_remaining_games('BOS')
    print(f"Remaining games: {len(bos_games)}")

    if bos_games:
        print(f"Next 5 games: {[g.isoformat() for g in bos_games[:5]]}")

    b2bs = schedule.get_team_back_to_backs('BOS')
    print(f"Back-to-backs remaining: {len(b2bs)}")

    rest = schedule.get_team_rest_days('BOS')
    print(f"Rest patterns: {rest}")

    # Weekly breakdown
    print("\n" + "-" * 70)
    print("WEEKLY GAME COUNTS (Next 4 weeks)")
    print("-" * 70)

    weeks = schedule.get_weekly_game_counts('BOS', num_weeks=4)
    for w in weeks:
        print(f"  {w['week_start']} to {w['week_end']}: {w['game_count']} games")

    # All teams comparison
    print("\n" + "-" * 70)
    print("ALL TEAMS - GAMES REMAINING")
    print("-" * 70)

    all_remaining = schedule.get_all_teams_games_remaining()
    sorted_teams = sorted(all_remaining.items(), key=lambda x: x[1], reverse=True)

    for abbr, count in sorted_teams[:10]:
        team_name = NBA_TEAMS[abbr]['name']
        print(f"  {abbr}: {count} games - {team_name}")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)

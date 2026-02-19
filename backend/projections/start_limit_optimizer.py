#!/usr/bin/env python3
"""
Day-by-Day Start Limit Optimizer.

This module simulates the entire remaining NBA season day-by-day to accurately
determine how many games each player will actually start, respecting position
start limits.

Algorithm:
1. Get roster, position limits, and remaining NBA schedule
2. For each day remaining in the season:
   a) Identify players with games that day (check their NBA team's schedule)
   b) Get each player's eligible positions and fantasy value
   c) Assign players to starting slots using greedy optimization
   d) Respect position start limits (skip filled positions)
   e) Track starts per player and per position
3. Return per-player start counts and percentages

This approach accurately handles:
- Players with limited position eligibility
- Back-to-back games
- Position slot competition
- Mid-season roster changes (current roster assumed)
"""

import logging
import sqlite3
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict
import requests

logger = logging.getLogger(__name__)

# ESPN Stat ID to standard key mapping
ESPN_STAT_ID_MAP = {
    0: 'pts', 1: 'blk', 2: 'stl', 3: 'ast',
    4: 'oreb', 5: 'dreb', 6: 'reb', 9: 'pf',
    11: 'to', 13: 'fgm', 14: 'fga', 15: 'ftm',
    16: 'fta', 17: '3pm', 18: '3pa',
    19: 'fg_pct', 20: 'ft_pct', 21: '3p_pct',
    40: 'min', 41: 'gp',
}

# Reverse mapping for convenience
STAT_KEY_TO_ESPN_ID = {v: k for k, v in ESPN_STAT_ID_MAP.items()}

# Categories that are calculated from component stats (percentages)
PERCENTAGE_CATEGORIES = {'fg_pct', 'ft_pct', '3p_pct'}

# Component stats needed for percentage calculations
PERCENTAGE_COMPONENTS = {
    'fg_pct': ('fgm', 'fga'),
    'ft_pct': ('ftm', 'fta'),
    '3p_pct': ('3pm', '3pa'),
}

# Categories where lower is better
REVERSE_CATEGORIES = {'to', 'pf'}


# ESPN Position Slot ID Mapping
SLOT_ID_TO_POSITION = {
    0: 'PG',
    1: 'SG',
    2: 'SF',
    3: 'PF',
    4: 'C',
    5: 'G',      # PG/SG eligible
    6: 'F',      # SF/PF eligible
    7: 'SG/SF',
    8: 'G/F',
    9: 'PF/C',
    10: 'F/C',
    11: 'UTIL',
    12: 'BE',    # Bench
    13: 'IR',    # Injured Reserve
    14: 'IR+',   # Injured Reserve Plus
}

POSITION_TO_SLOT_ID = {v: k for k, v in SLOT_ID_TO_POSITION.items()}

# Active starting positions (not bench or IR)
STARTING_SLOT_IDS = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}

# Default games per slot for the season
DEFAULT_GAMES_PER_SLOT = 82

# IR slot ID
IR_SLOT_ID = 13


@dataclass
class IRPlayerInfo:
    """Information about a player on IR and their projected return."""
    player_id: int
    player_name: str
    nba_team: str
    eligible_slots: List[int]
    per_game_value: float
    injury_status: str
    projected_return_date: Optional[date]
    will_return_before_season_end: bool
    replacing_player_id: Optional[int] = None
    replacing_player_name: Optional[str] = None
    games_after_return: int = 0
    season_outlook: str = ""
    per_game_stats: Dict[str, float] = field(default_factory=dict)  # For calculating adjusted stats


@dataclass
class PlayerGameLog:
    """Tracks a player's starts throughout the season simulation."""
    player_id: int
    player_name: str
    nba_team: str
    eligible_slots: List[int]
    per_game_value: float
    total_games_available: int = 0
    games_started: int = 0
    games_benched: int = 0
    starts_by_position: Dict[int, int] = field(default_factory=dict)
    daily_log: List[Dict] = field(default_factory=list)

    @property
    def start_percentage(self) -> float:
        if self.total_games_available == 0:
            return 0.0
        return (self.games_started / self.total_games_available) * 100

    @property
    def bench_percentage(self) -> float:
        if self.total_games_available == 0:
            return 0.0
        return (self.games_benched / self.total_games_available) * 100


@dataclass
class DaySimulation:
    """Results of simulating one day's lineup."""
    game_date: date
    players_with_games: List[int]  # player_ids
    assignments: Dict[int, int]  # slot_id -> player_id
    benched_players: List[int]  # player_ids with games but not starting
    slots_at_limit: List[int]  # slot_ids that hit their limit


@dataclass
class SeasonSimulationResult:
    """Complete results of the season simulation."""
    start_date: date
    end_date: date
    total_days: int
    game_days: int
    player_logs: Dict[int, PlayerGameLog]
    position_starts_used: Dict[int, int]
    position_limits: Dict[int, int]
    daily_simulations: List[DaySimulation]
    ir_players: List[IRPlayerInfo] = field(default_factory=list)
    dropped_players: List[int] = field(default_factory=list)  # player_ids dropped to make room for IR returns


@dataclass
class DropCandidateAnalysis:
    """Analysis of a potential drop candidate for IR player activation."""
    player_id: int
    player_name: str
    is_droppable: bool
    projected_starts: int  # Games they'd actually start rest of season
    total_games_available: int  # Total games their NBA team has
    start_percentage: float  # projected_starts / total_games_available
    marginal_value: float  # Total value contribution to team (starts × per_game_value)
    per_game_value: float
    position_scarcity_score: float  # How scarce their positions are on roster
    eligible_positions: List[str]
    reason: str  # Why this analysis was made


@dataclass
class RotoDropScenario:
    """Analysis of dropping a player for Roto standings optimization."""
    drop_player_id: int
    drop_player_name: str
    projected_roto_points: float  # Total Roto points if this player is dropped
    projected_final_rank: int  # Projected final standing (1-10)
    category_ranks: Dict[str, int]  # Rank in each category
    category_totals: Dict[str, float]  # End-of-season totals in each category
    rest_of_season_contribution: Dict[str, float]  # What this roster adds rest of season
    reason: str  # Why this scenario is good/bad


class StartLimitOptimizer:
    """
    Day-by-day season simulator for accurate start limit projections.

    Uses greedy optimization to assign players to starting slots each day,
    respecting position start limits and player eligibility.
    """

    def __init__(
        self,
        espn_s2: str,
        swid: str,
        league_id: int,
        season: int,
        verbose: bool = True,
        projection_method: str = 'adaptive',
        flat_game_rate: float = 0.85
    ):
        self.espn_s2 = espn_s2
        self.swid = swid
        self.league_id = league_id
        self.season = season
        self.verbose = verbose
        self.projection_method = projection_method
        self.flat_game_rate = flat_game_rate

        # Log projection settings
        logger.info(f"[StartLimitOptimizer] Initialized with projection_method={projection_method}, flat_game_rate={flat_game_rate:.1%}")

        # Cached data
        self._roster_settings: Optional[Dict] = None
        self._lineup_slot_counts: Optional[Dict[int, int]] = None
        self._lineup_slot_stat_limits: Optional[Dict[int, int]] = None
        self._nba_schedule: Optional[Any] = None
        self._starts_used_cache: Optional[Dict[int, Dict[int, int]]] = None  # team_id -> slot_id -> starts

        if self.verbose:
            logger.setLevel(logging.DEBUG)
            if not logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter(
                    '%(asctime)s - %(levelname)s - %(message)s'
                ))
                logger.addHandler(handler)

    def _log(self, message: str, level: str = "info"):
        """Log with verbosity control."""
        if self.verbose:
            getattr(logger, level)(message)

    def fetch_scoring_categories(self, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Fetch the league's actual scoring categories from ESPN API.

        Returns list of category dicts with:
            - stat_key: lowercase stat key (pts, reb, ast, fg_pct, etc.)
            - name: Display name
            - is_reverse: True if lower is better (turnovers)
            - is_percentage: True if it's a percentage stat
        """
        try:
            endpoint = (
                f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/fba/"
                f"seasons/{self.season}/segments/0/leagues/{self.league_id}"
            )
            params = {'view': 'mSettings'}
            cookies = {'espn_s2': self.espn_s2, 'SWID': self.swid}

            response = requests.get(endpoint, params=params, cookies=cookies, timeout=15)
            response.raise_for_status()
            data = response.json()

            scoring_settings = data.get('settings', {}).get('scoringSettings', {})
            scoring_items = scoring_settings.get('scoringItems', [])

            categories = []
            for item in scoring_items:
                stat_id = item.get('statId')
                if stat_id is None:
                    continue

                stat_key = ESPN_STAT_ID_MAP.get(stat_id)
                if not stat_key:
                    continue

                # Skip component stats that are only used for percentage calc
                # unless they're also scoring categories themselves
                is_scoring_cat = item.get('isReverseItem', False) or True  # All items here are scoring

                categories.append({
                    'stat_key': stat_key,
                    'stat_id': stat_id,
                    'name': stat_key.upper(),
                    'is_reverse': stat_key in REVERSE_CATEGORIES,
                    'is_percentage': stat_key in PERCENTAGE_CATEGORIES,
                })

            if categories:
                self._log(f"  Fetched {len(categories)} scoring categories: {[c['stat_key'] for c in categories]}")
                return categories

            # Fallback to default categories if none found
            self._log("  No scoring items found, using default categories", "warning")
            return self._get_default_categories()

        except Exception as e:
            self._log(f"Error fetching scoring categories: {e}", "warning")
            return self._get_default_categories()

    def _get_default_categories(self) -> List[Dict[str, Any]]:
        """Return default Roto scoring categories."""
        return [
            {'stat_key': 'pts', 'name': 'PTS', 'is_reverse': False, 'is_percentage': False},
            {'stat_key': 'reb', 'name': 'REB', 'is_reverse': False, 'is_percentage': False},
            {'stat_key': 'ast', 'name': 'AST', 'is_reverse': False, 'is_percentage': False},
            {'stat_key': 'stl', 'name': 'STL', 'is_reverse': False, 'is_percentage': False},
            {'stat_key': 'blk', 'name': 'BLK', 'is_reverse': False, 'is_percentage': False},
            {'stat_key': '3pm', 'name': '3PM', 'is_reverse': False, 'is_percentage': False},
            {'stat_key': 'fg_pct', 'name': 'FG%', 'is_reverse': False, 'is_percentage': True},
            {'stat_key': 'ft_pct', 'name': 'FT%', 'is_reverse': False, 'is_percentage': True},
            {'stat_key': 'to', 'name': 'TO', 'is_reverse': True, 'is_percentage': False},
        ]

    def get_hybrid_projection(
        self,
        player_id: int,
        player_data: Dict[str, Any],
        current_stats: Optional[Dict[str, float]] = None,
        espn_projection: Optional[Dict[str, float]] = None,
        previous_season_stats: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Get player projection using the hybrid projection engine.

        Returns per-game stat projections including all stats needed for
        percentage calculations (fgm, fga, ftm, fta, etc.)
        """
        try:
            from backend.projections.hybrid_engine import HybridProjectionEngine

            engine = HybridProjectionEngine()

            projection = engine.project_player(
                player_id=str(player_id),
                player_data=player_data,
                season_stats=current_stats,
                previous_season_stats=previous_season_stats,
                espn_projection=espn_projection,
                injury_status=player_data.get('injury_status', 'ACTIVE'),
                projection_method=self.projection_method,
                flat_game_rate=self.flat_game_rate,
            )

            return projection.projected_stats

        except Exception as e:
            self._log(f"Hybrid engine failed for player {player_id}: {e}, using current stats", "warning")
            # Fall back to current stats if available
            return current_stats or {}

    def build_roster_with_hybrid_projections(
        self,
        espn_players: List[Dict],
        stat_keys: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Build roster data using hybrid projection engine for each player.

        This method takes raw ESPN player data and generates projections
        using the hybrid engine for each player, including all stats needed
        for percentage calculations (fgm, fga, ftm, fta, 3pm, 3pa).

        Args:
            espn_players: List of player dicts from ESPN client with:
                - espn_player_id: ESPN player ID
                - name: Player name
                - nba_team: NBA team abbreviation
                - eligible_slots: List of eligible slot IDs
                - lineupSlotId: Current lineup slot
                - stats: Dict with current_season, previous_season, projected
                - injury_status: Injury status string
                - droppable: Whether player can be dropped
            stat_keys: List of stat keys to project. If None, uses default set.

        Returns:
            List of roster dicts ready for simulation with:
                - player_id, name, nba_team, eligible_slots, lineupSlotId
                - per_game_stats: Hybrid-projected per-game stats
                - per_game_value: Calculated fantasy value
                - droppable
        """
        if stat_keys is None:
            # Default stats including percentage components
            stat_keys = [
                'pts', 'reb', 'ast', 'stl', 'blk', '3pm', 'to',
                'fgm', 'fga', 'ftm', 'fta', '3pa',
                'fg_pct', 'ft_pct', '3p_pct'
            ]

        # Mapping between ESPN stat keys and hybrid engine stat keys
        # ESPN uses: reb, to, 3pm, 3pa
        # Hybrid engine uses: trb, tov, 3p
        STAT_KEY_ALIASES = {
            'trb': 'reb',   # hybrid -> ESPN
            'tov': 'to',    # hybrid -> ESPN
            '3p': '3pm',    # hybrid -> ESPN
        }
        REVERSE_ALIASES = {v: k for k, v in STAT_KEY_ALIASES.items()}

        roster_data = []
        engine = None

        # Try to initialize hybrid engine once
        try:
            from backend.projections.hybrid_engine import HybridProjectionEngine
            engine = HybridProjectionEngine()
            self._log("Hybrid projection engine initialized")
        except Exception as e:
            self._log(f"Hybrid engine not available: {e}, using ESPN stats directly", "warning")

        for player in espn_players:
            player_id = player.get('espn_player_id')
            player_name = player.get('name', 'Unknown')

            # Extract stats from ESPN player data
            stats_info = player.get('stats', {})
            current_season = stats_info.get('current_season', {})
            previous_season = stats_info.get('previous_season', {})
            espn_projection = stats_info.get('projected', {})

            # Get average stats from current/previous season
            current_avg = current_season.get('average', {})
            previous_avg = previous_season.get('average', {})
            projected_avg = espn_projection.get('average', {})

            games_played = current_season.get('games_played', 0)
            injury_status = player.get('injury_status', 'ACTIVE')

            # Convert ESPN stat IDs to stat keys
            def convert_stats(espn_stats: Dict) -> Dict[str, float]:
                """Convert ESPN stat ID format to stat key format."""
                result = {}
                for stat_id_str, value in espn_stats.items():
                    try:
                        stat_id = int(stat_id_str)
                        stat_key = ESPN_STAT_ID_MAP.get(stat_id)
                        if stat_key:
                            result[stat_key] = float(value)
                            # Also add hybrid engine alias if applicable
                            hybrid_key = REVERSE_ALIASES.get(stat_key)
                            if hybrid_key:
                                result[hybrid_key] = float(value)
                    except (ValueError, TypeError):
                        pass
                return result

            current_stats = convert_stats(current_avg)
            previous_stats = convert_stats(previous_avg)
            espn_proj_stats = convert_stats(projected_avg)

            # Build player data for hybrid engine
            player_data = {
                'player_id': str(player_id),
                'name': player_name,
                'team': player.get('nba_team', 'UNK'),
                'position': player.get('position', ''),
                'games_played': games_played,
                'injury_status': injury_status,
            }

            # Get hybrid projection or fall back to ESPN stats
            per_game_stats = {}

            if engine is not None:
                try:
                    projection = engine.project_player(
                        player_id=str(player_id),
                        player_data=player_data,
                        season_stats=current_stats if current_stats else None,
                        previous_season_stats=previous_stats if previous_stats else None,
                        espn_projection=espn_proj_stats if espn_proj_stats else None,
                        injury_status=injury_status,
                        projection_method=self.projection_method,
                        flat_game_rate=self.flat_game_rate,
                    )

                    # Use hybrid projected stats and normalize to ESPN format
                    raw_stats = projection.projected_stats.copy()

                    # Normalize hybrid engine keys to ESPN format
                    for hybrid_key, espn_key in STAT_KEY_ALIASES.items():
                        if hybrid_key in raw_stats and espn_key not in raw_stats:
                            raw_stats[espn_key] = raw_stats[hybrid_key]
                        elif espn_key in raw_stats and hybrid_key not in raw_stats:
                            raw_stats[hybrid_key] = raw_stats[espn_key]

                    per_game_stats = raw_stats

                    # Ensure all required stat keys are present
                    for stat_key in stat_keys:
                        if stat_key not in per_game_stats:
                            per_game_stats[stat_key] = 0.0

                    self._log(f"  {player_name}: Hybrid projection (tier: {projection.season_phase})")

                except Exception as e:
                    self._log(f"  {player_name}: Hybrid failed ({e}), using ESPN stats", "warning")
                    per_game_stats = current_stats or espn_proj_stats or previous_stats or {}
            else:
                # No hybrid engine - use ESPN stats directly
                if current_stats:
                    per_game_stats = current_stats
                elif games_played == 0 and espn_proj_stats:
                    # IR players with no games - use ESPN projection
                    per_game_stats = espn_proj_stats
                    self._log(f"  {player_name}: Using ESPN projection (IR/no games)")
                elif previous_stats:
                    per_game_stats = previous_stats
                else:
                    per_game_stats = {}

            # Calculate fantasy value
            per_game_value = self.calculate_player_value(per_game_stats)

            roster_data.append({
                'player_id': player_id,
                'name': player_name,
                'nba_team': player.get('nba_team', 'UNK'),
                'eligible_slots': player.get('eligible_slots', []),
                'lineupSlotId': player.get('lineupSlotId', 0),
                'per_game_stats': per_game_stats,
                'per_game_value': per_game_value,
                'droppable': player.get('droppable', True),
                'injury_status': injury_status,
                'games_played': games_played,
            })

        return roster_data

    def calculate_percentage_stat(
        self,
        stats: Dict[str, float],
        pct_stat: str
    ) -> float:
        """
        Calculate a percentage stat from component totals.

        Args:
            stats: Dict with total stats (fgm, fga, ftm, fta, etc.)
            pct_stat: Which percentage to calculate (fg_pct, ft_pct, 3p_pct)

        Returns:
            Percentage as decimal (0.0 to 1.0)
        """
        if pct_stat not in PERCENTAGE_COMPONENTS:
            return 0.0

        made_key, attempt_key = PERCENTAGE_COMPONENTS[pct_stat]
        made = stats.get(made_key, 0)
        attempts = stats.get(attempt_key, 0)

        if attempts > 0:
            return made / attempts
        return 0.0

    def _get_nba_schedule(self):
        """Get or create NBA schedule instance."""
        if self._nba_schedule is None:
            import sys
            import os

            # Ensure project root is in path for imports
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)

            try:
                from backend.scrapers.nba_schedule import NBASchedule
                self._nba_schedule = NBASchedule(season=self.season)
            except ImportError as e:
                self._log(f"NBASchedule not available ({e}), using estimates", "warning")
                return None
        return self._nba_schedule

    def fetch_roster_settings(self) -> Dict:
        """Fetch rosterSettings from ESPN API."""
        if self._roster_settings is not None:
            return self._roster_settings

        endpoint = (
            f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/fba/"
            f"seasons/{self.season}/segments/0/leagues/{self.league_id}"
        )

        params = {'view': 'mSettings'}
        cookies = {'espn_s2': self.espn_s2, 'SWID': self.swid}

        try:
            response = requests.get(endpoint, params=params, cookies=cookies, timeout=15)
            response.raise_for_status()
            data = response.json()

            self._roster_settings = data.get('settings', {}).get('rosterSettings', {})
            return self._roster_settings

        except Exception as e:
            self._log(f"Error fetching roster settings: {e}", "error")
            return {}

    def get_lineup_slot_counts(self) -> Dict[int, int]:
        """Get the number of each lineup slot type."""
        if self._lineup_slot_counts is not None:
            return self._lineup_slot_counts

        roster_settings = self.fetch_roster_settings()
        slot_counts = roster_settings.get('lineupSlotCounts', {})

        self._lineup_slot_counts = {
            int(k): int(v) for k, v in slot_counts.items()
        }

        return self._lineup_slot_counts

    def get_lineup_slot_stat_limits(self) -> Dict[int, int]:
        """Get the max games limit for each position slot."""
        if self._lineup_slot_stat_limits is not None:
            return self._lineup_slot_stat_limits

        roster_settings = self.fetch_roster_settings()
        stat_limits_raw = roster_settings.get('lineupSlotStatLimits', {})

        self._lineup_slot_stat_limits = {}
        for k, v in stat_limits_raw.items():
            slot_id = int(k)
            if isinstance(v, dict):
                limit_value = v.get('limitValue', 82)
                self._lineup_slot_stat_limits[slot_id] = int(limit_value)
            elif isinstance(v, (int, float)):
                self._lineup_slot_stat_limits[slot_id] = int(v)
            else:
                slot_counts = self.get_lineup_slot_counts()
                count = slot_counts.get(slot_id, 1)
                self._lineup_slot_stat_limits[slot_id] = count * DEFAULT_GAMES_PER_SLOT

        return self._lineup_slot_stat_limits

    def fetch_starts_used(self, team_id: int) -> Dict[int, int]:
        """
        Fetch actual starts-used-per-position from ESPN API.

        ESPN stores this in schedule[].teams[].cumulativeScore.statBySlot
        where each slot has {value: X, statId: 42, limitExceeded: bool}

        Args:
            team_id: ESPN team ID to fetch starts for

        Returns:
            Dict mapping slot_id -> starts used this season
            Example: {0: 54, 1: 54, 2: 54, 3: 54, 4: 58, 5: 50, 6: 51, 11: 150}
        """
        self._log(f"Fetching starts_used for team {team_id}")

        # Check cache first
        if self._starts_used_cache is not None and team_id in self._starts_used_cache:
            cached = self._starts_used_cache[team_id]
            self._log(f"Returning cached starts_used for team {team_id}: {cached}")
            return cached

        endpoint = (
            f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/fba/"
            f"seasons/{self.season}/segments/0/leagues/{self.league_id}"
        )

        params = {'view': ['mMatchup', 'mMatchupScore', 'mTeam', 'mSchedule']}
        cookies = {'espn_s2': self.espn_s2, 'SWID': self.swid}

        self._log(f"ESPN API endpoint: {endpoint}")
        self._log(f"Views: {params['view']}")

        try:
            response = requests.get(endpoint, params=params, cookies=cookies, timeout=15)
            response.raise_for_status()
            data = response.json()

            self._log(f"ESPN API response status: {response.status_code}")
            self._log(f"Top-level keys in response: {list(data.keys())}")

            # Initialize cache
            if self._starts_used_cache is None:
                self._starts_used_cache = {}

            # Search through schedule for team's statBySlot
            schedule = data.get('schedule', [])
            self._log(f"Schedule entries found: {len(schedule)}")

            teams_found = 0
            for matchup in schedule:
                teams_in_matchup = matchup.get('teams', [])

                for team_data in teams_in_matchup:
                    tid = team_data.get('teamId')
                    cumulative = team_data.get('cumulativeScore', {})
                    stat_by_slot = cumulative.get('statBySlot', {})

                    if stat_by_slot:
                        teams_found += 1
                        starts_used = {}
                        for slot_id_key, slot_data in stat_by_slot.items():
                            # Handle both string and int keys from ESPN API
                            slot_id = int(slot_id_key) if isinstance(slot_id_key, str) else slot_id_key
                            value = slot_data.get('value', 0)
                            starts_used[slot_id] = int(value)

                        self._starts_used_cache[tid] = starts_used

                        if tid == team_id:
                            self._log(f"SUCCESS: Found starts_used for team {team_id}")
                            self._log(f"ESPN raw statBySlot: {stat_by_slot}")
                            self._log(f"Parsed starts_used: {starts_used}")

            self._log(f"Total teams with statBySlot data: {teams_found}")
            self._log(f"Teams in cache: {list(self._starts_used_cache.keys())}")

            # Return for requested team
            if team_id in self._starts_used_cache:
                result = self._starts_used_cache[team_id]
                self._log(f"Returning starts_used for team {team_id}: {result}")
                return result

            self._log(f"WARNING: Could not find starts data for team {team_id}", "warning")
            self._log(f"Available team IDs: {list(self._starts_used_cache.keys())}", "warning")
            return {}

        except Exception as e:
            self._log(f"ERROR fetching starts used: {e}", "error")
            import traceback
            self._log(f"Traceback: {traceback.format_exc()}", "error")
            return {}

    def fetch_ir_players(
        self,
        team_id: int,
        roster: List[Dict],
        season_end_date: date,
        manual_return_dates: Optional[Dict[int, date]] = None,
        include_out_players: bool = True
    ) -> List[IRPlayerInfo]:
        """
        Detect players on IR who will return and play games.

        SIMPLIFIED: Uses projected_games from hybrid engine to determine
        if player will contribute. If projected_games > 0, player is expected
        to return and play.

        Args:
            team_id: ESPN team ID
            roster: List of player dicts from roster (with lineupSlotId, projected_games)
            season_end_date: End of fantasy season
            manual_return_dates: Optional dict of player_id -> manually specified return date
            include_out_players: If True, also detect OUT players in active slots

        Returns:
            List of IRPlayerInfo for players on IR who will return
        """
        if manual_return_dates is None:
            manual_return_dates = {}

        ir_players = []
        today = date.today()

        for player in roster:
            lineup_slot_id = player.get('lineupSlotId', 0)
            projected_games = player.get('original_projected_games', 0)

            # Only process players on IR slot
            if lineup_slot_id != IR_SLOT_ID:
                continue

            player_id = player.get('player_id', 0)
            player_name = player.get('name', 'Unknown')
            nba_team = player.get('nba_team', 'UNK')
            eligible_slots = player.get('eligible_slots', [])
            per_game_value = player.get('per_game_value', 0)
            per_game_stats = player.get('per_game_stats', {})

            # SIMPLE RULE: If hybrid engine says 0 games, player won't return
            if projected_games == 0:
                self._log(f"IR: {player_name} - projected_games=0, will NOT return")
                continue

            # Get team's remaining schedule
            full_team_games = self.get_player_nba_team_schedule(nba_team, today, season_end_date)

            # Get ACTUAL injury return date from ESPN (not back-calculated!)
            injury_details = player.get('injury_details')
            actual_return_date = None
            if injury_details and isinstance(injury_details, dict):
                actual_return_date = injury_details.get('expected_return_date')

            # Determine return date - only use REAL data, NEVER invent dates
            projected_return = None
            has_real_return_date = False

            if player_id in manual_return_dates:
                # Use manually specified date
                projected_return = manual_return_dates[player_id]
                has_real_return_date = True
                self._log(f"IR: {player_name} - Manual return date: {projected_return}")
            elif actual_return_date:
                # Use ESPN's actual return date - this is real data
                projected_return = actual_return_date
                has_real_return_date = True
                self._log(f"IR: {player_name} - ESPN return date: {projected_return}")
            else:
                # No return date available - DO NOT invent one
                # Player has projected_games > 0, so hybrid engine expects them to play
                # We'll treat them as available from today in simulation, but don't set a fake date
                projected_return = None
                self._log(f"IR: {player_name} - No ESPN return date (projected_games={projected_games})")

            # Check if player will return before season ends
            # If no return date but projected_games > 0, assume they will return
            if projected_return is not None:
                will_return = projected_return < season_end_date
            else:
                # No date but has games projected - assume will return (from today)
                will_return = projected_games > 0

            # Use projected_games directly as games_after_return
            # The hybrid engine already calculated this correctly
            games_after_return = projected_games if will_return else 0

            ir_info = IRPlayerInfo(
                player_id=player_id,
                player_name=player_name,
                nba_team=nba_team,
                eligible_slots=[s for s in eligible_slots if s in STARTING_SLOT_IDS],
                per_game_value=per_game_value,
                injury_status='IR',  # Simplified - we know they're on IR
                projected_return_date=projected_return,
                will_return_before_season_end=will_return,
                games_after_return=games_after_return,
                season_outlook="",  # Not needed - using projected_games instead
                per_game_stats=per_game_stats
            )

            ir_players.append(ir_info)

            if will_return:
                if projected_return:
                    self._log(f"IR: {player_name} - IR return {projected_return}, "
                             f"games={games_after_return}, value={per_game_value:.1f}")
                else:
                    self._log(f"IR: {player_name} - No return date, "
                             f"games={games_after_return}, value={per_game_value:.1f} (available now)")
            else:
                self._log(f"IR: {player_name} - will NOT return before season end")

        return ir_players

    def find_optimal_drop_candidate(
        self,
        roster: List[Dict],
        ir_player: IRPlayerInfo,
        initial_starts_used: Dict[int, int],
        exclude_player_ids: Optional[Set[int]] = None,
        my_team_id: Optional[int] = None,
        categories: Optional[List[Dict]] = None
    ) -> Tuple[Optional[Dict], List[RotoDropScenario]]:
        """
        Find the optimal player to drop for an IR return using END-OF-SEASON Roto optimization.

        This method optimizes for final Roto standings by:
        1. Fetching the league's actual scoring categories from ESPN
        2. Fetching all teams' current season stats (already accumulated)
        3. For each potential drop candidate:
           a) Remove that player, add the IR player
           b) Run day-by-day optimizer for REST of season with new roster
           c) Calculate END-OF-SEASON totals = current stats + rest-of-season projections
           d) For percentage stats, calculate from total FGM/FGA, FTM/FTA
           e) Rank all teams in each category
           f) Convert ranks to Roto points (1st=N, 2nd=N-1, etc for N-team league)
           g) Sum total Roto points
        4. Drop the player whose removal results in HIGHEST end-of-season Roto points

        Args:
            roster: List of player dicts for MY team
            ir_player: The IR player who is returning
            initial_starts_used: Current starts used per position from ESPN
            exclude_player_ids: Player IDs to exclude from consideration
            my_team_id: My ESPN team ID (for fetching league-wide stats)
            categories: List of category dicts for scoring (if None, fetches from ESPN)

        Returns:
            Tuple of (best player dict to drop, list of all scenario analyses)
        """
        if exclude_player_ids is None:
            exclude_player_ids = set()

        # Fetch actual scoring categories from ESPN if not provided
        if categories is None:
            categories = self.fetch_scoring_categories()

        self._log("\n" + "=" * 70)
        self._log(f"ROTO STANDINGS OPTIMIZATION FOR {ir_player.player_name} RETURN")
        self._log("=" * 70)

        # Show categories being used
        cat_names = [c.get('stat_key', '').upper() for c in categories]
        self._log(f"  Scoring Categories ({len(categories)}): {', '.join(cat_names)}")

        # Get season dates and position limits
        start_date, end_date = self.get_season_dates()

        # Use IR player's return date as simulation start if later than today
        sim_start = ir_player.projected_return_date if ir_player.projected_return_date else start_date
        if sim_start < start_date:
            sim_start = start_date

        stat_limits = self.get_lineup_slot_stat_limits()

        # Build category info for analysis
        category_info = []
        for cat in categories:
            stat_key = cat.get('stat_key', '').lower()
            category_info.append({
                'stat_key': stat_key,
                'name': cat.get('name', stat_key.upper()),
                'is_reverse': cat.get('is_reverse', stat_key in REVERSE_CATEGORIES),
                'is_percentage': cat.get('is_percentage', stat_key in PERCENTAGE_CATEGORIES),
            })

        # Extract category keys for simulation (include component stats for percentages)
        category_keys = [c['stat_key'] for c in category_info]

        # Add component stats needed for percentage calculations
        component_stats = set()
        for cat in category_info:
            if cat['is_percentage'] and cat['stat_key'] in PERCENTAGE_COMPONENTS:
                made, attempts = PERCENTAGE_COMPONENTS[cat['stat_key']]
                component_stats.add(made)
                component_stats.add(attempts)

        # All stats we need to track (categories + components)
        all_stat_keys = list(set(category_keys) | component_stats)

        # Fetch all teams' current season stats from ESPN
        self._log("\n  Fetching league-wide current season stats...")
        all_teams_current_stats = self._fetch_all_teams_current_stats(my_team_id, all_stat_keys)
        num_teams = len(all_teams_current_stats)
        self._log(f"  Found {num_teams} teams in league")
        self._log(f"  Max possible Roto points: {num_teams * len(category_keys)}")

        # Get my team's current stats
        my_current_stats = all_teams_current_stats.get(my_team_id, {})

        # Fetch all teams' rosters for proper projections
        self._log("\n  Fetching rosters for all teams...")
        all_teams_rosters = self._fetch_all_teams_rosters()
        self._log(f"  Fetched rosters for {len(all_teams_rosters)} teams")

        # Calculate projected EOS totals for ALL teams BEFORE analyzing drop scenarios
        # This ensures fair comparison: your_projected vs other_teams_projected
        all_teams_baseline_eos = self._calculate_all_teams_eos_projections(
            all_teams_rosters=all_teams_rosters,
            all_teams_current_stats=all_teams_current_stats,
            sim_start=sim_start,
            end_date=end_date,
            stat_limits=stat_limits,
            all_stat_keys=all_stat_keys,
            category_info=category_info,
            my_team_id=my_team_id
        )

        # Get active droppable players
        candidates = []
        for player in roster:
            slot_id = player.get('lineupSlotId', 12)
            player_id = player.get('player_id', 0)

            # Only consider active roster (slots 0-12, not IR)
            if slot_id <= 12 and player_id not in exclude_player_ids:
                if player.get('droppable', True):
                    candidates.append(player)

        if not candidates:
            self._log("  No droppable candidates found")
            return None, []

        self._log(f"\n  Analyzing {len(candidates)} drop scenarios for Roto optimization...")
        self._log(f"  Simulation period: {sim_start} to {end_date}")

        # Calculate IR player's per-game stats
        ir_per_game = ir_player.per_game_stats or {}
        return_str = f"returning {ir_player.projected_return_date}" if ir_player.projected_return_date else "available now"
        self._log(f"  IR Player {ir_player.player_name}: {return_str}")

        # Analyze each drop scenario
        scenarios: List[RotoDropScenario] = []

        for candidate in candidates:
            scenario = self._analyze_roto_drop_scenario(
                drop_candidate=candidate,
                roster=roster,
                ir_player=ir_player,
                sim_start=sim_start,
                end_date=end_date,
                initial_starts_used=initial_starts_used,
                stat_limits=stat_limits,
                exclude_player_ids=exclude_player_ids,
                my_team_id=my_team_id,
                my_current_stats=my_current_stats,
                all_teams_eos_projections=all_teams_baseline_eos,
                category_info=category_info,
                all_stat_keys=all_stat_keys,
                num_teams=num_teams
            )
            scenarios.append(scenario)

        # Sort by projected Roto points (HIGHEST first = best scenario)
        scenarios.sort(key=lambda s: s.projected_roto_points, reverse=True)

        # Log the analysis results
        max_pts = num_teams * len(category_info)
        best_pts = scenarios[0].projected_roto_points if scenarios else 0
        worst_pts = scenarios[-1].projected_roto_points if scenarios else 0
        pts_spread = best_pts - worst_pts

        self._log("\n  " + "-" * 70)
        self._log("  DROP SCENARIO RANKINGS (comparing YOUR projected vs ALL TEAMS projected)")
        self._log("  " + "-" * 70)
        self._log(f"  Roto Points spread across scenarios: {pts_spread:.1f} pts difference")
        self._log(f"  (Best: {best_pts:.1f}, Worst: {worst_pts:.1f}, Max possible: {max_pts})")

        for i, s in enumerate(scenarios):
            rank = i + 1
            diff_from_best = s.projected_roto_points - best_pts
            self._log(f"\n  {rank}. Drop {s.drop_player_name}")
            self._log(f"     Projected Roto Points: {s.projected_roto_points:.1f} / {max_pts}")
            if diff_from_best < 0:
                self._log(f"     vs Best Option: {diff_from_best:.1f} pts")
            self._log(f"     Projected Final Rank: #{s.projected_final_rank}")

            # Show ALL category breakdown
            cat_summary = []
            for cat in category_info:
                stat_key = cat['stat_key']
                cat_rank = s.category_ranks.get(stat_key, 0)
                cat_total = s.category_totals.get(stat_key, 0)
                # Format percentage stats as percentages
                if cat['is_percentage']:
                    cat_summary.append(f"{stat_key.upper()}:{cat_total:.1%}(#{cat_rank})")
                else:
                    cat_summary.append(f"{stat_key.upper()}:{cat_total:.0f}(#{cat_rank})")
            self._log(f"     Categories: {', '.join(cat_summary)}")
            self._log(f"     Reason: {s.reason}")

        # Select the best scenario (highest Roto points)
        if scenarios:
            best = scenarios[0]
            second_best = scenarios[1] if len(scenarios) > 1 else None
            self._log(f"\n  {'='*70}")
            self._log(f"  OPTIMAL DROP: {best.drop_player_name}")
            self._log(f"  Projected Final: {best.projected_roto_points:.1f} Roto pts, Rank #{best.projected_final_rank}")
            if second_best:
                margin = best.projected_roto_points - second_best.projected_roto_points
                self._log(f"  Margin over next best: +{margin:.1f} pts vs dropping {second_best.drop_player_name}")
            self._log(f"  {'='*70}")

            # Find the player dict
            for player in candidates:
                if player.get('player_id') == best.drop_player_id:
                    return player, scenarios

        return None, scenarios

    def _fetch_all_teams_current_stats(
        self,
        my_team_id: Optional[int] = None,
        category_keys: Optional[List[str]] = None
    ) -> Dict[int, Dict[str, float]]:
        """
        Fetch current season stats for ALL teams in the league.

        Fetches all stats needed for the league's scoring categories,
        including component stats for percentage calculations.

        Returns dict mapping team_id -> {stat_key: total_value}
        """
        try:
            endpoint = (
                f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/fba/"
                f"seasons/{self.season}/segments/0/leagues/{self.league_id}"
            )
            params = {'view': ['mTeam', 'mRoster', 'mMatchup', 'mMatchupScore']}
            cookies = {'espn_s2': self.espn_s2, 'SWID': self.swid}

            response = requests.get(endpoint, params=params, cookies=cookies, timeout=15)
            response.raise_for_status()
            data = response.json()

            all_teams_stats = {}

            for team in data.get('teams', []):
                team_id = team.get('id')
                values_by_stat = team.get('valuesByStat', {})

                team_stats = {}
                # Fetch ALL stats using the global mapping
                for stat_id_str, value in values_by_stat.items():
                    stat_id = int(stat_id_str)
                    stat_key = ESPN_STAT_ID_MAP.get(stat_id)
                    if stat_key:
                        team_stats[stat_key] = float(value)

                # DON'T pre-calculate percentages here - they need to be
                # calculated from totals after projections are added
                # Store the raw component stats (fgm, fga, ftm, fta, 3pm, 3pa)

                all_teams_stats[team_id] = team_stats

            return all_teams_stats

        except Exception as e:
            self._log(f"Error fetching team stats: {e}", "warning")
            return {}

    def _analyze_roto_drop_scenario(
        self,
        drop_candidate: Dict,
        roster: List[Dict],
        ir_player: IRPlayerInfo,
        sim_start: date,
        end_date: date,
        initial_starts_used: Dict[int, int],
        stat_limits: Dict[int, int],
        exclude_player_ids: Set[int],
        my_team_id: int,
        my_current_stats: Dict[str, float],
        all_teams_eos_projections: Dict[int, Dict[str, float]],
        category_info: List[Dict[str, Any]],
        all_stat_keys: List[str],
        num_teams: int
    ) -> RotoDropScenario:
        """
        Analyze a single drop scenario for Roto standings impact.

        Simulates rest of season with MODIFIED roster (drop + IR return),
        then compares against PRE-CALCULATED projections for all other teams.
        This ensures fair comparison: your_projected vs other_teams_projected.

        Args:
            all_teams_eos_projections: Pre-calculated EOS projections for ALL teams
                                       (from _calculate_all_teams_eos_projections)
            category_info: List of category dicts with stat_key, is_reverse, is_percentage
            all_stat_keys: All stat keys to track (categories + component stats for percentages)
        """
        drop_player_id = drop_candidate.get('player_id', 0)
        drop_player_name = drop_candidate.get('name', 'Unknown')

        # Build modified roster: remove dropped player, add IR player
        modified_roster = []
        for player in roster:
            pid = player.get('player_id', 0)
            if pid == drop_player_id:
                continue  # Skip dropped player
            if pid == ir_player.player_id:
                continue  # Skip IR player (will add with active status)
            modified_roster.append(player)

        # Add IR player as active
        modified_roster.append({
            'player_id': ir_player.player_id,
            'name': ir_player.player_name,
            'nba_team': ir_player.nba_team,
            'eligible_slots': ir_player.eligible_slots,
            'per_game_stats': ir_player.per_game_stats,
            'per_game_value': ir_player.per_game_value,
            'lineupSlotId': 0,  # Active roster
            'droppable': True,
        })

        # Run simulation to get rest-of-season stats with MODIFIED roster
        ros_stats = self._simulate_roster_ros_stats(
            roster=modified_roster,
            sim_start=sim_start,
            end_date=end_date,
            initial_starts_used=initial_starts_used,
            stat_limits=stat_limits,
            category_keys=all_stat_keys  # Track all stats including components
        )

        # Calculate my team's end-of-season TOTALS for all stats
        my_eos_totals_raw = {}
        for stat_key in all_stat_keys:
            current = my_current_stats.get(stat_key, 0)
            ros = ros_stats.get(stat_key, 0)
            my_eos_totals_raw[stat_key] = current + ros

        # Now calculate the final category values (including percentages from totals)
        my_eos_totals = {}
        for cat in category_info:
            stat_key = cat['stat_key']
            if cat['is_percentage']:
                # Calculate percentage from end-of-season component totals
                my_eos_totals[stat_key] = self.calculate_percentage_stat(
                    my_eos_totals_raw, stat_key
                )
            else:
                my_eos_totals[stat_key] = my_eos_totals_raw.get(stat_key, 0)

        # Build all teams' EOS projections for ranking
        # Use PRE-CALCULATED projections for other teams (fair comparison)
        # Only update MY team with the modified roster projection
        all_teams_eos = {}
        for team_id, team_eos in all_teams_eos_projections.items():
            if team_id == my_team_id:
                # Use the new projection with modified roster
                all_teams_eos[team_id] = my_eos_totals
            else:
                # Use pre-calculated projection (from start-limit-aware optimizer)
                all_teams_eos[team_id] = team_eos

        # Rank teams in each SCORING category
        category_ranks = {}
        for cat in category_info:
            stat_key = cat['stat_key']

            # Get all teams' values for this category
            team_values = [(tid, stats.get(stat_key, 0)) for tid, stats in all_teams_eos.items()]

            # Sort by value (higher is better, unless is_reverse)
            reverse = not cat['is_reverse']
            team_values.sort(key=lambda x: x[1], reverse=reverse)

            # Find my rank
            for rank, (tid, _) in enumerate(team_values, 1):
                if tid == my_team_id:
                    category_ranks[stat_key] = rank
                    break

        # Calculate Roto points (1st place = num_teams points, last = 1 point)
        roto_points = 0
        for cat in category_info:
            stat_key = cat['stat_key']
            rank = category_ranks.get(stat_key, num_teams)
            points = num_teams - rank + 1
            roto_points += points

        # Calculate projected final rank based on total Roto points
        num_categories = len(category_info)
        max_possible = num_teams * num_categories
        pct_of_max = roto_points / max_possible if max_possible > 0 else 0
        projected_rank = max(1, min(num_teams, int((1 - pct_of_max) * num_teams) + 1))

        # Generate reason
        strong_cats = [cat['stat_key'] for cat in category_info
                      if category_ranks.get(cat['stat_key'], num_teams) <= 3]
        weak_cats = [cat['stat_key'] for cat in category_info
                    if category_ranks.get(cat['stat_key'], 1) >= num_teams - 2]

        if len(strong_cats) >= 4:
            reason = f"Strong in {len(strong_cats)} categories: {', '.join(strong_cats[:3]).upper()}"
        elif len(weak_cats) >= 3:
            reason = f"Weak in {', '.join(weak_cats[:2]).upper()}, dropping {drop_player_name} doesn't help enough"
        else:
            avg_rank = sum(category_ranks.values()) / len(category_ranks) if category_ranks else num_teams
            reason = f"Average rank {avg_rank:.1f} across {num_categories} categories"

        return RotoDropScenario(
            drop_player_id=drop_player_id,
            drop_player_name=drop_player_name,
            projected_roto_points=roto_points,
            projected_final_rank=projected_rank,
            category_ranks=category_ranks,
            category_totals=my_eos_totals,
            rest_of_season_contribution=ros_stats,
            reason=reason
        )

    def _simulate_roster_ros_stats(
        self,
        roster: List[Dict],
        sim_start: date,
        end_date: date,
        initial_starts_used: Dict[int, int],
        stat_limits: Dict[int, int],
        category_keys: List[str]
    ) -> Dict[str, float]:
        """
        Simulate rest-of-season stats for a roster configuration.

        Returns dict of {stat_key: total_projected_value}
        """
        # Build player info
        player_info = {}  # player_id -> (nba_team, eligible_slots, per_game_value, per_game_stats)
        player_schedules = {}  # player_id -> set of game dates

        for player in roster:
            pid = player.get('player_id', 0)
            if player.get('lineupSlotId', 0) == IR_SLOT_ID:
                continue

            nba_team = player.get('nba_team', 'UNK')
            eligible_slots = [s for s in player.get('eligible_slots', []) if s in STARTING_SLOT_IDS]
            per_game_value = player.get('per_game_value', 0)
            per_game_stats = player.get('per_game_stats', {})

            player_info[pid] = (nba_team, eligible_slots, per_game_value, per_game_stats)
            team_games = self.get_player_nba_team_schedule(nba_team, sim_start, end_date)
            player_schedules[pid] = set(team_games)

        # Get lineup slots
        slots = self.expand_lineup_slots()

        # Track starts and stats
        starts_used = defaultdict(int, initial_starts_used)
        total_stats = defaultdict(float)

        # Simulate each day
        current_date = sim_start
        while current_date <= end_date:
            # Find players with games today
            players_today = []
            for pid, games in player_schedules.items():
                if current_date in games:
                    nba_t, elig, val, stats = player_info[pid]
                    players_today.append((pid, elig, val, stats))

            # Sort by value (highest first)
            players_today.sort(key=lambda x: x[2], reverse=True)

            # Greedy assignment
            slots_used_today = set()
            assigned_today = set()

            for pid, elig, val, stats in players_today:
                if pid in assigned_today:
                    continue

                # Find available slot
                for slot_id, slot_name in slots:
                    if slot_id in slots_used_today:
                        continue
                    if slot_id not in elig:
                        continue
                    limit = stat_limits.get(slot_id, DEFAULT_GAMES_PER_SLOT)
                    if starts_used[slot_id] >= limit:
                        continue

                    # Assign player
                    slots_used_today.add(slot_id)
                    assigned_today.add(pid)
                    starts_used[slot_id] += 1

                    # Add stats
                    for cat in category_keys:
                        stat_val = stats.get(cat, 0) or stats.get(cat.upper(), 0)
                        total_stats[cat] += stat_val
                    break

            current_date += timedelta(days=1)

        return dict(total_stats)

    def _project_team_ros(
        self,
        current_stats: Dict[str, float],
        sim_start: date,
        end_date: date
    ) -> Dict[str, float]:
        """
        Project a team's end-of-season totals based on current pace.

        Simplified: assumes they maintain current per-game averages.
        """
        # Estimate games played so far (rough)
        season_start = date(self.season - 1, 10, 21)
        days_played = (sim_start - season_start).days
        games_played = max(1, int(days_played / 7 * 3.5))  # ~3.5 games per week

        # Days remaining
        days_remaining = (end_date - sim_start).days
        games_remaining = int(days_remaining / 7 * 3.5)

        # Project totals
        eos_stats = {}
        for stat, current_total in current_stats.items():
            per_game = current_total / games_played if games_played > 0 else 0
            ros_projection = per_game * games_remaining
            eos_stats[stat] = current_total + ros_projection

        return eos_stats

    def _fetch_all_teams_rosters(self) -> Dict[int, List[Dict]]:
        """
        Fetch rosters for ALL teams in the league from ESPN API.

        Returns dict mapping team_id -> list of player dicts suitable for simulation.
        """
        try:
            endpoint = (
                f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/fba/"
                f"seasons/{self.season}/segments/0/leagues/{self.league_id}"
            )
            params = {'view': ['mTeam', 'mRoster']}
            cookies = {'espn_s2': self.espn_s2, 'SWID': self.swid}

            response = requests.get(endpoint, params=params, cookies=cookies, timeout=15)
            response.raise_for_status()
            data = response.json()

            # NBA team mapping from proTeamId
            PRO_TEAM_MAP = {
                1: 'ATL', 2: 'BOS', 3: 'NOP', 4: 'CHI', 5: 'CLE',
                6: 'DAL', 7: 'DEN', 8: 'DET', 9: 'GSW', 10: 'HOU',
                11: 'IND', 12: 'LAC', 13: 'LAL', 14: 'MIA', 15: 'MIL',
                16: 'MIN', 17: 'BKN', 18: 'NYK', 19: 'ORL', 20: 'PHI',
                21: 'PHX', 22: 'POR', 23: 'SAC', 24: 'SAS', 25: 'OKC',
                26: 'UTA', 27: 'WAS', 28: 'TOR', 29: 'MEM', 30: 'CHA',
            }

            all_teams_rosters = {}

            for team in data.get('teams', []):
                team_id = team.get('id')
                roster_entries = team.get('roster', {}).get('entries', [])

                roster = []
                for entry in roster_entries:
                    player_data = entry.get('playerPoolEntry', {}).get('player', {})
                    player_id = player_data.get('id', 0)
                    player_name = player_data.get('fullName', 'Unknown')
                    pro_team_id = player_data.get('proTeamId', 0)
                    nba_team = PRO_TEAM_MAP.get(pro_team_id, 'UNK')
                    lineup_slot_id = entry.get('lineupSlotId', 12)

                    # Get eligible slots
                    eligible_slots = player_data.get('eligibleSlots', [])
                    eligible_slots = [s for s in eligible_slots if s in STARTING_SLOT_IDS]

                    # Get per-game stats from player stats
                    per_game_stats = {}
                    per_game_value = 0.0
                    stats_list = player_data.get('stats', [])
                    for stat_entry in stats_list:
                        # Look for season average stats (statSplitTypeId = 0 or 1)
                        if stat_entry.get('seasonId') == self.season:
                            avg_stats = stat_entry.get('averageStats', {}) or stat_entry.get('stats', {})
                            if avg_stats:
                                for stat_id_str, val in avg_stats.items():
                                    stat_id = int(stat_id_str)
                                    stat_key = ESPN_STAT_ID_MAP.get(stat_id)
                                    if stat_key:
                                        per_game_stats[stat_key] = float(val)
                                # Simple fantasy value estimate
                                per_game_value = (
                                    per_game_stats.get('pts', 0) +
                                    per_game_stats.get('reb', 0) * 1.2 +
                                    per_game_stats.get('ast', 0) * 1.5 +
                                    per_game_stats.get('stl', 0) * 3 +
                                    per_game_stats.get('blk', 0) * 3 -
                                    per_game_stats.get('to', 0) * 2
                                )
                                break

                    roster.append({
                        'player_id': player_id,
                        'name': player_name,
                        'nba_team': nba_team,
                        'eligible_slots': eligible_slots,
                        'per_game_stats': per_game_stats,
                        'per_game_value': per_game_value,
                        'lineupSlotId': lineup_slot_id,
                    })

                all_teams_rosters[team_id] = roster

            return all_teams_rosters

        except Exception as e:
            self._log(f"Error fetching all teams rosters: {e}", "warning")
            return {}

    def _calculate_all_teams_eos_projections(
        self,
        all_teams_rosters: Dict[int, List[Dict]],
        all_teams_current_stats: Dict[int, Dict[str, float]],
        sim_start: date,
        end_date: date,
        stat_limits: Dict[int, int],
        all_stat_keys: List[str],
        category_info: List[Dict[str, Any]],
        my_team_id: int
    ) -> Dict[int, Dict[str, float]]:
        """
        Calculate projected end-of-season totals for ALL teams in the league.

        This runs the start-limit-aware optimizer for EACH team to get fair
        projections, not just pace-based estimates.

        Returns dict mapping team_id -> {stat_key: projected_eos_total}
        """
        num_teams = len(all_teams_rosters)
        self._log(f"\n  Calculating projections for all {num_teams} teams for fair comparison...")

        all_teams_eos = {}

        for team_id, roster in all_teams_rosters.items():
            # Get team's current stats
            current_stats = all_teams_current_stats.get(team_id, {})

            # Fetch starts_used for this team
            team_starts_used = self.fetch_starts_used(team_id)

            # Build roster for simulation (exclude IR players)
            sim_roster = [p for p in roster if p.get('lineupSlotId', 12) != IR_SLOT_ID]

            # Run rest-of-season simulation for this team
            ros_stats = self._simulate_roster_ros_stats(
                roster=sim_roster,
                sim_start=sim_start,
                end_date=end_date,
                initial_starts_used=team_starts_used,
                stat_limits=stat_limits,
                category_keys=all_stat_keys
            )

            # Calculate end-of-season totals (current + ROS projection)
            eos_totals_raw = {}
            for stat_key in all_stat_keys:
                current = current_stats.get(stat_key, 0)
                ros = ros_stats.get(stat_key, 0)
                eos_totals_raw[stat_key] = current + ros

            # Calculate final category values (including percentages from totals)
            eos_totals = {}
            for cat in category_info:
                stat_key = cat['stat_key']
                if cat['is_percentage']:
                    eos_totals[stat_key] = self.calculate_percentage_stat(
                        eos_totals_raw, stat_key
                    )
                else:
                    eos_totals[stat_key] = eos_totals_raw.get(stat_key, 0)

            all_teams_eos[team_id] = eos_totals

            # Log progress for debugging
            if team_id == my_team_id:
                self._log(f"    Team {team_id} (YOU): Projected {len(ros_stats)} stat categories")
            else:
                # Minimal logging for other teams
                pass

        self._log(f"  Completed projections for all {num_teams} teams")
        return all_teams_eos

    def _analyze_drop_candidate(
        self,
        candidate: Dict,
        roster: List[Dict],
        ir_player: IRPlayerInfo,
        sim_start: date,
        end_date: date,
        initial_starts_used: Dict[int, int],
        stat_limits: Dict[int, int],
        exclude_player_ids: Set[int]
    ) -> DropCandidateAnalysis:
        """
        Analyze a single drop candidate's marginal contribution.

        Runs a quick simulation to determine how many games this player would
        actually start if kept, then calculates their marginal value.
        """
        player_id = candidate.get('player_id', 0)
        player_name = candidate.get('name', 'Unknown')
        nba_team = candidate.get('nba_team', 'UNK')
        eligible_slots = candidate.get('eligible_slots', [])
        per_game_value = candidate.get('per_game_value', 0)
        per_game_stats = candidate.get('per_game_stats', {})

        # Get player's game schedule
        player_games = self.get_player_nba_team_schedule(nba_team, sim_start, end_date)
        total_games = len(player_games)

        # Filter to starting-eligible slots
        eligible_starting = [s for s in eligible_slots if s in STARTING_SLOT_IDS]
        position_names = [SLOT_ID_TO_POSITION.get(s, f'?{s}') for s in eligible_starting[:5]]

        # Calculate position scarcity (how many other players share these positions)
        scarcity_score = self._calculate_position_scarcity(
            eligible_starting, roster, exclude_player_ids | {player_id}
        )

        # Run quick simulation to determine actual starts
        projected_starts = self._quick_simulate_player_starts(
            player_id=player_id,
            player_name=player_name,
            nba_team=nba_team,
            eligible_slots=eligible_starting,
            per_game_value=per_game_value,
            roster=roster,
            sim_start=sim_start,
            end_date=end_date,
            initial_starts_used=initial_starts_used,
            stat_limits=stat_limits,
            exclude_player_ids=exclude_player_ids
        )

        # Calculate marginal value = starts × per_game_value
        marginal_value = projected_starts * per_game_value

        # Calculate start percentage
        start_percentage = (projected_starts / total_games * 100) if total_games > 0 else 0

        # Generate reason
        if projected_starts == 0:
            reason = "Never starts due to position conflicts"
        elif start_percentage < 30:
            reason = f"Rarely starts ({start_percentage:.0f}%) - limited position eligibility"
        elif start_percentage < 60:
            reason = f"Moderate starter ({start_percentage:.0f}%) - position competition"
        elif per_game_value < 5:
            reason = f"Low per-game value ({per_game_value:.1f}) despite {start_percentage:.0f}% start rate"
        else:
            reason = f"Regular starter ({start_percentage:.0f}%) with good value"

        return DropCandidateAnalysis(
            player_id=player_id,
            player_name=player_name,
            is_droppable=True,
            projected_starts=projected_starts,
            total_games_available=total_games,
            start_percentage=start_percentage,
            marginal_value=marginal_value,
            per_game_value=per_game_value,
            position_scarcity_score=scarcity_score,
            eligible_positions=position_names,
            reason=reason
        )

    def _calculate_position_scarcity(
        self,
        eligible_slots: List[int],
        roster: List[Dict],
        exclude_ids: Set[int]
    ) -> float:
        """
        Calculate how scarce a player's positions are on the roster.

        Lower score = positions are well-covered by other players (more replaceable)
        Higher score = positions are scarce (harder to replace)
        """
        if not eligible_slots:
            return 0.0

        total_coverage = 0
        for slot in eligible_slots:
            coverage = 0
            for player in roster:
                pid = player.get('player_id', 0)
                if pid in exclude_ids:
                    continue
                if player.get('lineupSlotId', 12) == IR_SLOT_ID:
                    continue
                player_slots = player.get('eligible_slots', [])
                if slot in player_slots:
                    coverage += 1
            total_coverage += coverage

        # Normalize: fewer players covering = higher scarcity
        avg_coverage = total_coverage / len(eligible_slots) if eligible_slots else 0
        # Invert so higher = more scarce
        scarcity = 1.0 / (avg_coverage + 0.5) if avg_coverage >= 0 else 0
        return scarcity

    def _quick_simulate_player_starts(
        self,
        player_id: int,
        player_name: str,
        nba_team: str,
        eligible_slots: List[int],
        per_game_value: float,
        roster: List[Dict],
        sim_start: date,
        end_date: date,
        initial_starts_used: Dict[int, int],
        stat_limits: Dict[int, int],
        exclude_player_ids: Set[int]
    ) -> int:
        """
        Run a quick simulation to determine how many games a player would start.

        This is a lightweight version of simulate_season focused on counting
        starts for a single player within the full roster context.
        """
        # Build all player schedules and info
        player_info = {}  # player_id -> (nba_team, eligible_slots, per_game_value)
        player_schedules = {}  # player_id -> set of game dates

        for p in roster:
            pid = p.get('player_id', 0)
            if pid in exclude_player_ids:
                continue
            if p.get('lineupSlotId', 12) == IR_SLOT_ID:
                continue

            p_nba_team = p.get('nba_team', 'UNK')
            p_eligible = [s for s in p.get('eligible_slots', []) if s in STARTING_SLOT_IDS]
            p_value = p.get('per_game_value', 0)

            player_info[pid] = (p_nba_team, p_eligible, p_value)
            team_games = self.get_player_nba_team_schedule(p_nba_team, sim_start, end_date)
            player_schedules[pid] = set(team_games)

        # Track starts
        starts_used = defaultdict(int, initial_starts_used)
        target_player_starts = 0

        # Get lineup slots
        slots = self.expand_lineup_slots()

        # Simulate each day
        current_date = sim_start
        while current_date <= end_date:
            # Find players with games today
            players_today = []
            for pid, games in player_schedules.items():
                if current_date in games:
                    nba_t, elig, val = player_info[pid]
                    players_today.append((pid, elig, val))

            # Sort by value (highest first for greedy assignment)
            players_today.sort(key=lambda x: x[2], reverse=True)

            # Track which slots are used today and which players are assigned
            slots_used_today = set()
            assigned_today = set()

            # Greedy assignment
            for pid, elig, val in players_today:
                if pid in assigned_today:
                    continue

                # Try to find an available slot
                best_slot = None
                for slot_id, slot_name in slots:
                    if slot_id in slots_used_today:
                        continue
                    if slot_id not in elig:
                        continue
                    # Check position limit
                    limit = stat_limits.get(slot_id, DEFAULT_GAMES_PER_SLOT)
                    if starts_used[slot_id] >= limit:
                        continue
                    best_slot = slot_id
                    break

                if best_slot is not None:
                    slots_used_today.add(best_slot)
                    assigned_today.add(pid)
                    starts_used[best_slot] += 1

                    if pid == player_id:
                        target_player_starts += 1

            current_date += timedelta(days=1)

        return target_player_starts

    def find_worst_active_player(
        self,
        roster: List[Dict],
        exclude_player_ids: Optional[Set[int]] = None
    ) -> Optional[Dict]:
        """
        Legacy method: Find the worst player on the active roster by projected value.

        NOTE: This method is kept for backward compatibility but find_optimal_drop_candidate
        should be preferred as it uses marginal contribution analysis.

        Args:
            roster: List of player dicts
            exclude_player_ids: Player IDs to exclude from consideration

        Returns:
            Player dict of worst active player, or None
        """
        if exclude_player_ids is None:
            exclude_player_ids = set()

        active_players = []
        for player in roster:
            slot_id = player.get('lineupSlotId', 12)
            player_id = player.get('player_id', 0)

            # Only consider active roster (slots 0-12, not IR)
            if slot_id <= 12 and player_id not in exclude_player_ids:
                # Check if droppable
                if player.get('droppable', True):
                    active_players.append(player)

        if not active_players:
            return None

        # Sort by per-game value (lowest first)
        active_players.sort(key=lambda p: p.get('per_game_value', 0))

        return active_players[0]

    def get_season_dates(self) -> Tuple[date, date]:
        """Get the start and end dates for the remaining season."""
        today = date.today()

        # 2025-26 NBA season approximate dates
        if self.season == 2026:
            season_start = date(2025, 10, 21)
            season_end = date(2026, 4, 12)
        else:
            # Generic calculation
            season_start = date(self.season - 1, 10, 21)
            season_end = date(self.season, 4, 12)

        # Start from today if season has started
        start_date = max(today, season_start)

        return start_date, season_end

    def get_player_nba_team_schedule(
        self,
        nba_team: str,
        start_date: date,
        end_date: date
    ) -> List[date]:
        """Get game dates for an NBA team in the given date range."""
        schedule = self._get_nba_schedule()

        if schedule is None:
            # Fallback: estimate ~3.5 games per week
            days = (end_date - start_date).days
            games = int(days / 7 * 3.5)
            # Generate evenly spaced dates
            if games == 0:
                return []
            interval = days // games
            return [start_date + timedelta(days=i * interval) for i in range(games)]

        try:
            remaining_games = schedule.get_team_remaining_games(nba_team, start_date)
            return [g for g in remaining_games if g <= end_date]
        except Exception as e:
            self._log(f"Error getting schedule for {nba_team}: {e}", "warning")
            return []

    def calculate_player_value(
        self,
        per_game_stats: Dict[str, float],
        categories: Optional[List[str]] = None
    ) -> float:
        """Calculate a player's fantasy value from per-game stats."""
        if categories is None:
            categories = ['PTS', 'REB', 'AST', 'STL', 'BLK', '3PM']

        # Value weights (can be customized per league)
        # Support both uppercase and lowercase stat keys
        weights = {
            'PTS': 1.0, 'pts': 1.0,
            'REB': 1.2, 'reb': 1.2,
            'AST': 1.5, 'ast': 1.5,
            'STL': 2.5, 'stl': 2.5,
            'BLK': 2.5, 'blk': 2.5,
            '3PM': 1.0, '3pm': 1.0,
            'TO': -1.5, 'to': -1.5,
        }

        value = 0.0
        for stat, weight in weights.items():
            stat_val = per_game_stats.get(stat, 0)
            if stat_val:
                value += stat_val * weight

        return value

    def expand_lineup_slots(self) -> List[Tuple[int, str]]:
        """
        Expand lineup slot counts into individual slot instances.

        Returns list of (slot_id, slot_name) for each slot in the lineup.
        Example: If lineup has 1 PG, 1 SG, 3 UTIL, returns:
            [(0, 'PG'), (1, 'SG'), ..., (11, 'UTIL_1'), (11, 'UTIL_2'), (11, 'UTIL_3')]
        """
        slot_counts = self.get_lineup_slot_counts()

        slots = []
        for slot_id in sorted(slot_counts.keys()):
            count = slot_counts[slot_id]
            if count == 0 or slot_id not in STARTING_SLOT_IDS:
                continue

            pos_name = SLOT_ID_TO_POSITION.get(slot_id, f'SLOT_{slot_id}')

            if count == 1:
                slots.append((slot_id, pos_name))
            else:
                for i in range(count):
                    slots.append((slot_id, f"{pos_name}_{i+1}"))

        return slots

    def simulate_day(
        self,
        game_date: date,
        available_players: List[Dict],
        slots: List[Tuple[int, str]],
        starts_used: Dict[int, int],
        stat_limits: Dict[int, int],
        player_logs: Dict[int, PlayerGameLog]
    ) -> DaySimulation:
        """
        Simulate one day's lineup assignments.

        Uses greedy algorithm: assign highest-value players first,
        to their most constrained eligible positions.
        """
        # Sort players by value (highest first)
        players_sorted = sorted(
            available_players,
            key=lambda p: p['per_game_value'],
            reverse=True
        )

        assignments = {}  # slot_index -> player_id
        used_players = set()
        slots_at_limit = []

        # Track which slot instances are still open
        # slot_instances[slot_id] = list of open instance indices
        slot_instances = defaultdict(list)
        for idx, (slot_id, _) in enumerate(slots):
            slot_instances[slot_id].append(idx)

        # Check which positions have hit their limit
        for slot_id in list(slot_instances.keys()):
            limit = stat_limits.get(slot_id, DEFAULT_GAMES_PER_SLOT * len(slot_instances[slot_id]))
            if starts_used.get(slot_id, 0) >= limit:
                slots_at_limit.append(slot_id)
                slot_instances.pop(slot_id, None)

        # Assign players using greedy approach
        for player in players_sorted:
            player_id = player['player_id']
            eligible_slots = player['eligible_slots']

            # Find an available slot this player can fill
            # Prefer most constrained slots (fewest remaining capacity)
            best_slot_idx = None
            best_slot_id = None
            min_remaining = float('inf')

            for slot_id in eligible_slots:
                if slot_id not in slot_instances or not slot_instances[slot_id]:
                    continue

                # Check limit
                limit = stat_limits.get(slot_id, DEFAULT_GAMES_PER_SLOT)
                used = starts_used.get(slot_id, 0)
                remaining = limit - used

                if remaining <= 0:
                    continue

                # Pick slot with least remaining capacity (fill constrained slots first)
                if remaining < min_remaining:
                    min_remaining = remaining
                    best_slot_id = slot_id
                    best_slot_idx = slot_instances[slot_id][0]

            if best_slot_idx is not None:
                # Assign player to this slot
                assignments[best_slot_idx] = player_id
                used_players.add(player_id)

                # Remove this slot instance from available
                slot_instances[best_slot_id].pop(0)

                # Update starts used
                starts_used[best_slot_id] = starts_used.get(best_slot_id, 0) + 1

                # Update player log
                if player_id in player_logs:
                    player_logs[player_id].games_started += 1
                    player_logs[player_id].starts_by_position[best_slot_id] = \
                        player_logs[player_id].starts_by_position.get(best_slot_id, 0) + 1

        # Players with games but not assigned are benched
        benched = []
        for player in players_sorted:
            player_id = player['player_id']
            if player_id not in used_players:
                benched.append(player_id)
                if player_id in player_logs:
                    player_logs[player_id].games_benched += 1

        return DaySimulation(
            game_date=game_date,
            players_with_games=[p['player_id'] for p in players_sorted],
            assignments=assignments,
            benched_players=benched,
            slots_at_limit=slots_at_limit
        )

    def simulate_season(
        self,
        roster: List[Dict],
        categories: Optional[List[Dict]] = None,
        initial_starts_used: Optional[Dict[int, int]] = None,
        ir_players: Optional[List[IRPlayerInfo]] = None,
        include_ir_returns: bool = False,
        team_id: Optional[int] = None
    ) -> SeasonSimulationResult:
        """
        Simulate the entire remaining season day by day.

        Args:
            roster: List of player dicts with:
                - player_id: int
                - name: str
                - nba_team: str (abbreviation)
                - eligible_slots: List[int]
                - per_game_stats: Dict[str, float]
            initial_starts_used: Dict of slot_id -> starts already used this season
                                 (fetched from ESPN). If None, starts from 0.
            ir_players: List of IRPlayerInfo for players on IR
            include_ir_returns: If True, simulate IR player returns (dropping worst player)
            team_id: ESPN team ID for Roto standings optimization

        Returns:
            SeasonSimulationResult with detailed breakdown
        """
        self._log("=" * 70)
        self._log("SEASON SIMULATION START")
        self._log("=" * 70)

        # Get season dates
        start_date, end_date = self.get_season_dates()
        self._log(f"Simulating from {start_date} to {end_date}")

        # Get lineup configuration
        slots = self.expand_lineup_slots()
        self._log(f"Lineup slots: {[s[1] for s in slots]}")

        stat_limits = self.get_lineup_slot_stat_limits()
        self._log(f"Position limits: {stat_limits}")

        # Initialize tracking with actual starts used from ESPN (or zeros if not available)
        self._log(f"initial_starts_used parameter: {initial_starts_used}")
        self._log(f"initial_starts_used type: {type(initial_starts_used)}")

        if initial_starts_used:
            starts_used = defaultdict(int, initial_starts_used)
            self._log(f"Starting with actual ESPN starts: {dict(initial_starts_used)}")
            # Log slot name mapping for clarity
            slot_names = {0: 'PG', 1: 'SG', 2: 'SF', 3: 'PF', 4: 'C', 5: 'G', 6: 'F', 11: 'UTIL'}
            for slot_id, count in sorted(initial_starts_used.items()):
                slot_name = slot_names.get(slot_id, f'Slot{slot_id}')
                self._log(f"  {slot_name} (slot {slot_id}): {count} starts used")
        else:
            starts_used = defaultdict(int)
            self._log("WARNING: No initial starts data - starting from 0 for all positions!")
            self._log("This will cause inaccurate projections. Check fetch_starts_used() call.")
        player_logs = {}
        daily_simulations = []

        # Process IR players if enabled
        ir_players_processed = ir_players or []
        dropped_player_ids: Set[int] = set()
        ir_return_schedule: Dict[date, List[IRPlayerInfo]] = defaultdict(list)

        if include_ir_returns and ir_players_processed:
            self._log("\n" + "-" * 70)
            self._log("IR PLAYER PROCESSING")
            self._log("-" * 70)

            # Find optimal players to drop for each IR return using marginal contribution analysis
            already_dropping = set()
            for ir_player in ir_players_processed:
                if not ir_player.will_return_before_season_end:
                    self._log(f"  {ir_player.player_name}: Excluded (not returning)")
                    continue

                # Use Roto standings optimization to find optimal drop
                optimal_drop, scenarios = self.find_optimal_drop_candidate(
                    roster=roster,
                    ir_player=ir_player,
                    initial_starts_used=initial_starts_used or {},
                    exclude_player_ids=already_dropping,
                    my_team_id=team_id,
                    categories=categories
                )

                if optimal_drop:
                    ir_player.replacing_player_id = optimal_drop.get('player_id')
                    ir_player.replacing_player_name = optimal_drop.get('name', 'Unknown')
                    already_dropping.add(optimal_drop.get('player_id'))

                    # Log return info - show actual date only if we have one
                    if ir_player.projected_return_date:
                        return_info = f"IR return {ir_player.projected_return_date}"
                    else:
                        return_info = "Available now (no ESPN return date)"
                    self._log(f"\n  {ir_player.player_name}: {return_info}, "
                             f"Replacing {ir_player.replacing_player_name}, "
                             f"+{ir_player.games_after_return} games")

                    # Schedule this return - use start_date if no return date
                    effective_return = ir_player.projected_return_date or start_date
                    ir_return_schedule[effective_return].append(ir_player)
                else:
                    self._log(f"  {ir_player.player_name}: No droppable player found")

        # Build player schedules and logs
        self._log("\nBuilding player game schedules...")
        player_schedules = {}  # player_id -> set of game dates

        # Track which players are currently active (excludes IR/OUT players not yet returned)
        active_player_ids = set()
        ir_player_ids = {ir.player_id for ir in ir_players_processed} if ir_players_processed else set()

        # Track OUT players with return dates (in active slots but injured)
        out_player_return_schedule: Dict[date, List[Dict]] = defaultdict(list)
        out_player_ids: Set[int] = set()

        # =================================================================
        # SIMPLIFIED: Use projected_games from hybrid engine
        # The hybrid engine already calculated games based on injuries,
        # return dates, out-for-season status, etc. Trust it!
        # =================================================================
        self._log("\n" + "=" * 70)
        self._log("PLAYER PROJECTED GAMES (from hybrid engine)")
        self._log("=" * 70)

        for player in roster:
            player_id = player.get('player_id', 0)
            player_name = player.get('name', 'Unknown')
            nba_team = player.get('nba_team', 'UNK')
            eligible_slots = player.get('eligible_slots', [])
            per_game_stats = player.get('per_game_stats', {})
            lineup_slot_id = player.get('lineupSlotId', 0)
            projected_games = player.get('original_projected_games', 0)

            # Filter to starting-eligible slots only
            eligible_starting = [s for s in eligible_slots if s in STARTING_SLOT_IDS]

            # Calculate player value
            per_game_value = self.calculate_player_value(per_game_stats)

            # SIMPLE RULE: If hybrid engine says 0 games, exclude player
            if projected_games == 0:
                self._log(f"  {player_name} ({nba_team}): projected_games=0, EXCLUDING from optimization")
                continue

            # Skip IR players - they're handled separately via ir_players list
            if lineup_slot_id == IR_SLOT_ID:
                self._log(f"  {player_name} ({nba_team}): ON IR slot, projected_games={projected_games}, "
                         f"handled via IR return logic")
                continue

            # Get team's full schedule
            full_team_games = self.get_player_nba_team_schedule(nba_team, start_date, end_date)
            total_team_games = len(full_team_games)

            # Check for ACTUAL injury data from ESPN (not back-calculated)
            # Only use return date logic if player has real injury_details with expected_return_date
            injury_details = player.get('injury_details')
            actual_return_date = None

            if injury_details and isinstance(injury_details, dict):
                actual_return_date = injury_details.get('expected_return_date')

            # If player has ACTUAL ESPN injury return date, handle as OUT player
            if actual_return_date and actual_return_date > start_date and actual_return_date <= end_date:
                out_player_ids.add(player_id)

                # Get schedule from return date onward only (for daily simulation)
                team_games = self.get_player_nba_team_schedule(nba_team, actual_return_date, end_date)
                player_schedules[player_id] = set(team_games)

                # Use projected_games from hybrid engine (already accounts for game_rate)
                # NOT len(team_games) which is raw schedule count
                player_projected_games = projected_games

                # Create player log
                player_logs[player_id] = PlayerGameLog(
                    player_id=player_id,
                    player_name=player_name,
                    nba_team=nba_team,
                    eligible_slots=eligible_starting,
                    per_game_value=per_game_value,
                    total_games_available=player_projected_games,  # Use hybrid engine's projection
                )

                # Schedule their return
                out_player_return_schedule[actual_return_date].append({
                    'player_id': player_id,
                    'player_name': player_name,
                })

                self._log(f"  {player_name} ({nba_team}): INJURED, ESPN return={actual_return_date}, "
                         f"projected_games={player_projected_games} (from hybrid engine), value={per_game_value:.1f}")
                continue

            # Regular active player - available for all remaining team games
            # NOTE: projected_games < total_team_games does NOT mean injured!
            # The tier system reduces projected_games for players with fewer games played.
            team_games = full_team_games
            player_schedules[player_id] = set(team_games)

            # Create player log
            player_logs[player_id] = PlayerGameLog(
                player_id=player_id,
                player_name=player_name,
                nba_team=nba_team,
                eligible_slots=eligible_starting,
                per_game_value=per_game_value,
                total_games_available=len(team_games),
            )

            # Track as active player
            active_player_ids.add(player_id)

            self._log(f"  {player_name} ({nba_team}): ACTIVE, {total_team_games} games remaining, "
                     f"value={per_game_value:.1f}")

        # Pre-build IR player schedules and logs (they start with 0 games available until return)
        if include_ir_returns:
            for ir_player in ir_players_processed:
                if not ir_player.will_return_before_season_end:
                    continue

                # Get schedule from return date onward (for daily simulation)
                effective_return = ir_player.projected_return_date or start_date
                team_games = self.get_player_nba_team_schedule(
                    ir_player.nba_team,
                    effective_return,
                    end_date
                )
                player_schedules[ir_player.player_id] = set(team_games)

                # Use games_after_return from IR player info (already set from projected_games)
                # NOT len(team_games) which is raw schedule count
                ir_projected_games = ir_player.games_after_return

                # Create player log (starts with 0 games available, updated on return)
                player_logs[ir_player.player_id] = PlayerGameLog(
                    player_id=ir_player.player_id,
                    player_name=ir_player.player_name,
                    nba_team=ir_player.nba_team,
                    eligible_slots=ir_player.eligible_slots,
                    per_game_value=ir_player.per_game_value,
                    total_games_available=ir_projected_games,  # Use hybrid engine's projection
                )

                if ir_player.projected_return_date:
                    self._log(f"  {ir_player.player_name} ({ir_player.nba_team}): "
                             f"IR return {ir_player.projected_return_date}, "
                             f"projected_games={ir_projected_games} (from hybrid engine)")
                else:
                    self._log(f"  {ir_player.player_name} ({ir_player.nba_team}): "
                             f"No ESPN return date, available now, "
                             f"projected_games={ir_projected_games} (from hybrid engine)")

        # =================================================================
        # DEBUG: Summary of player categorization
        # =================================================================
        self._log("\n" + "=" * 70)
        self._log("DEBUG: PLAYER CATEGORIZATION SUMMARY")
        self._log("=" * 70)
        self._log(f"  Active player IDs ({len(active_player_ids)}): {sorted(active_player_ids)}")
        self._log(f"  OUT player IDs ({len(out_player_ids)}): {sorted(out_player_ids)}")
        self._log(f"  IR player IDs ({len(ir_player_ids)}): {sorted(ir_player_ids)}")

        # Find JJJ's player_id and show their categorization
        for p in roster:
            p_name = p.get('name', '')
            if 'jaren' in p_name.lower() or 'jackson' in p_name.lower():
                p_id = p.get('player_id', 0)
                self._log(f"  >>> JJJ (ID={p_id}) STATUS:")
                self._log(f"  >>>   In active_player_ids: {p_id in active_player_ids}")
                self._log(f"  >>>   In out_player_ids: {p_id in out_player_ids}")
                self._log(f"  >>>   In ir_player_ids: {p_id in ir_player_ids}")
                if p_id in out_player_ids:
                    # Find their return date
                    for ret_date, players in out_player_return_schedule.items():
                        for p_info in players:
                            if p_info['player_id'] == p_id:
                                self._log(f"  >>>   Scheduled to return: {ret_date}")
        self._log("=" * 70 + "\n")

        # Simulate each day
        self._log("\n" + "-" * 70)
        self._log("DAY-BY-DAY SIMULATION")
        self._log("-" * 70)

        current_date = start_date
        game_days = 0

        # Track starts per player to enforce projected_games limit
        player_starts_used = defaultdict(int)

        # Build projected_games lookup from roster
        player_projected_games = {}
        for p in roster:
            pid = p.get('player_id', 0)
            proj_games = p.get('original_projected_games', 0)
            player_projected_games[pid] = proj_games
            self._log(f"  Player {p.get('name', 'Unknown')} (ID={pid}): projected_games limit = {proj_games}")

        while current_date <= end_date:
            # Check for IR player returns today
            if include_ir_returns and current_date in ir_return_schedule:
                for ir_player in ir_return_schedule[current_date]:
                    # Activate IR player
                    active_player_ids.add(ir_player.player_id)
                    self._log(f"  {current_date}: IR RETURN - {ir_player.player_name} activated")

                    # Drop the replaced player
                    if ir_player.replacing_player_id:
                        dropped_player_ids.add(ir_player.replacing_player_id)
                        active_player_ids.discard(ir_player.replacing_player_id)
                        self._log(f"  {current_date}: DROPPED - {ir_player.replacing_player_name}")

            # Check for OUT player returns today (players in active slots with return dates)
            if current_date in out_player_return_schedule:
                for out_player_info in out_player_return_schedule[current_date]:
                    player_id = out_player_info['player_id']
                    player_name = out_player_info['player_name']
                    active_player_ids.add(player_id)
                    # Calculate remaining games for logging
                    remaining_games = len([d for d in player_schedules.get(player_id, set()) if d >= current_date])
                    self._log(f"  {current_date}: OUT RETURN - {player_name} activated, available for {remaining_games} games")

            # Find players with games today (only active players, not dropped)
            available_today = []
            for player_id, game_dates in player_schedules.items():
                log = player_logs.get(player_id)
                player_name = log.player_name if log else f"ID:{player_id}"

                # DEBUG: Check JJJ specifically on first few days
                is_jjj_player = log and ('jaren' in log.player_name.lower() or 'jackson' in log.player_name.lower())
                if is_jjj_player and game_days < 5:  # Only log first 5 game days
                    has_game = current_date in game_dates
                    is_dropped = player_id in dropped_player_ids
                    is_ir_not_returned = player_id in ir_player_ids and player_id not in active_player_ids
                    is_out_not_returned = player_id in out_player_ids and player_id not in active_player_ids
                    is_active = player_id in active_player_ids
                    self._log(f"  >>> JJJ DAY CHECK {current_date}: has_game={has_game}, "
                             f"dropped={is_dropped}, ir_wait={is_ir_not_returned}, "
                             f"out_wait={is_out_not_returned}, active={is_active}")
                    self._log(f"  >>> JJJ in out_player_ids: {player_id in out_player_ids}, "
                             f"in active_player_ids: {player_id in active_player_ids}")

                # Skip dropped players
                if player_id in dropped_player_ids:
                    continue

                # Skip IR players not yet returned
                if player_id in ir_player_ids and player_id not in active_player_ids:
                    continue

                # Skip OUT players not yet returned
                if player_id in out_player_ids and player_id not in active_player_ids:
                    continue

                if current_date in game_dates:
                    if log:
                        # Check if player has exhausted their projected_games limit
                        proj_limit = player_projected_games.get(player_id, 0)
                        starts_so_far = player_starts_used[player_id]
                        if proj_limit > 0 and starts_so_far >= proj_limit:
                            # Player has reached their projected games limit - skip
                            if is_jjj_player or self.verbose:
                                self._log(f"  >>> {log.player_name}: LIMIT REACHED ({starts_so_far}/{proj_limit} games)")
                            continue

                        # DEBUG: Log when JJJ is added to available players
                        if is_jjj_player and game_days < 5:
                            self._log(f"  >>> JJJ ADDED TO available_today on {current_date}!")

                        available_today.append({
                            'player_id': player_id,
                            'name': log.player_name,
                            'eligible_slots': log.eligible_slots,
                            'per_game_value': log.per_game_value,
                        })

            if available_today:
                game_days += 1

                # Simulate this day
                day_result = self.simulate_day(
                    game_date=current_date,
                    available_players=available_today,
                    slots=slots,
                    starts_used=starts_used,
                    stat_limits=stat_limits,
                    player_logs=player_logs
                )
                daily_simulations.append(day_result)

                # Update player starts tracking based on assignments
                # assignments is Dict[int, int] mapping slot_id -> player_id
                for slot_id, assigned_player_id in day_result.assignments.items():
                    player_starts_used[assigned_player_id] += 1

                # Log summary for this day
                if self.verbose and len(daily_simulations) <= 10:
                    started = len(day_result.assignments)
                    benched = len(day_result.benched_players)
                    self._log(f"  {current_date}: {len(available_today)} players with games, "
                             f"{started} started, {benched} benched")

            current_date += timedelta(days=1)

        # Log final summary
        self._log("\n" + "=" * 70)
        self._log("SIMULATION COMPLETE")
        self._log("=" * 70)

        total_days = (end_date - start_date).days + 1
        self._log(f"Total days: {total_days}, Game days: {game_days}")

        self._log("\nPosition starts used:")
        for slot_id, count in sorted(starts_used.items()):
            pos_name = SLOT_ID_TO_POSITION.get(slot_id, f'SLOT_{slot_id}')
            limit = stat_limits.get(slot_id, 0)
            pct = (count / limit * 100) if limit > 0 else 0
            self._log(f"  {pos_name}: {count}/{limit} ({pct:.1f}%)")

        self._log("\nPlayer start summary (with projected_games enforcement):")
        for player_id, log in sorted(player_logs.items(), key=lambda x: x[1].games_started, reverse=True):
            # Mark dropped players
            dropped_marker = " [DROPPED]" if player_id in dropped_player_ids else ""
            proj_limit = player_projected_games.get(player_id, 0)
            actual_starts = player_starts_used.get(player_id, 0)
            # Only show LIMIT REACHED when player actually hit their limit
            if proj_limit > 0 and actual_starts >= proj_limit:
                limit_status = "LIMIT REACHED"
            else:
                limit_status = f"{log.games_benched} benched"
            self._log(f"  {log.player_name}: {actual_starts}/{proj_limit} starts ({limit_status}){dropped_marker}")

        # Validation: Check no player exceeded their projected_games limit
        self._log("\nValidation - checking projected_games enforcement:")
        violations_found = False
        for player_id, starts in player_starts_used.items():
            proj_limit = player_projected_games.get(player_id, 0)
            if proj_limit > 0 and starts > proj_limit:
                log = player_logs.get(player_id)
                player_name = log.player_name if log else f"ID:{player_id}"
                logger.warning(f"BUG: {player_name} used {starts} starts but projected for only {proj_limit}")
                self._log(f"  [VIOLATION] {player_name}: {starts} > {proj_limit}")
                violations_found = True
        if not violations_found:
            self._log("  [OK] All players within projected_games limits")

        # Log IR summary
        if include_ir_returns and ir_players_processed:
            self._log("\nIR Player Summary:")
            for ir_player in ir_players_processed:
                if ir_player.will_return_before_season_end:
                    log = player_logs.get(ir_player.player_id)
                    games_started = log.games_started if log else 0
                    return_str = f"IR return {ir_player.projected_return_date}" if ir_player.projected_return_date else "Available now"
                    self._log(f"  IR: {ir_player.player_name} - "
                             f"{return_str}, "
                             f"Replacing: {ir_player.replacing_player_name or 'N/A'}, "
                             f"Adding {games_started} starts")
                else:
                    self._log(f"  IR: {ir_player.player_name} - "
                             f"Not returning (excluded from projections)")

        return SeasonSimulationResult(
            start_date=start_date,
            end_date=end_date,
            total_days=total_days,
            game_days=game_days,
            player_logs=player_logs,
            position_starts_used=dict(starts_used),
            position_limits=stat_limits,
            daily_simulations=daily_simulations,
            ir_players=ir_players_processed,
            dropped_players=list(dropped_player_ids)
        )

    def optimize_team_projections(
        self,
        team_id: int,
        team_name: str,
        roster: List[Dict],
        categories: List[Dict],
        include_ir_returns: bool = False,
        manual_return_dates: Optional[Dict[int, date]] = None
    ) -> Tuple[Dict[str, float], List[Dict], List[IRPlayerInfo]]:
        """
        Main entry point: Optimize projections using day-by-day simulation.

        Args:
            team_id: ESPN team ID
            team_name: Team name for logging
            roster: List of player dicts with projected_stats, projected_games, eligible_slots
            categories: List of scoring category dicts
            include_ir_returns: If True, simulate IR player returns
            manual_return_dates: Optional dict of player_id -> manually specified return date

        Returns:
            Tuple of (adjusted_totals, player_assignments, ir_players)
        """
        self._log(f"\n{'#'*60}")
        self._log(f"OPTIMIZING: {team_name}")
        self._log(f"{'#'*60}")

        # Get season end date for IR processing
        _, season_end = self.get_season_dates()

        # Convert roster format for simulation
        sim_roster = []
        for player in roster:
            # Extract per-game stats from total projections
            projected_stats = player.get('projected_stats', {})
            projected_games = player.get('projected_games', 1)
            player_name = player.get('name', 'Unknown')
            self._log(f"[OPTIMIZER RECEIVED] {player_name}: projected_games={projected_games}")

            per_game_stats = {}
            if projected_games > 0:
                for key, total in projected_stats.items():
                    if key not in ['FG%', 'FT%']:
                        per_game_stats[key] = total / projected_games
                    else:
                        per_game_stats[key] = total

            sim_roster.append({
                'player_id': player.get('player_id', 0),
                'name': player.get('name', 'Unknown'),
                'nba_team': player.get('nba_team', 'UNK'),
                'eligible_slots': player.get('eligible_slots', []),
                'per_game_stats': per_game_stats,
                'per_game_value': self.calculate_player_value(per_game_stats),
                'original_projected_stats': projected_stats,
                'original_projected_games': projected_games,
                'lineupSlotId': player.get('lineupSlotId', 0),
                'injury_status': player.get('injury_status', 'ACTIVE'),
                'injury_details': player.get('injury_details'),  # ESPN injury data with expected_return_date
                'season_outlook': player.get('season_outlook', ''),
                'droppable': player.get('droppable', True),
            })

        # Fetch actual starts used from ESPN
        initial_starts = self.fetch_starts_used(team_id)
        if initial_starts:
            self._log(f"Actual ESPN starts used: {initial_starts}")
        else:
            self._log("WARNING: Could not fetch actual starts - simulation will start from 0", "warning")

        # Fetch IR players if enabled
        ir_players = []
        if include_ir_returns:
            ir_players = self.fetch_ir_players(
                team_id=team_id,
                roster=sim_roster,
                season_end_date=season_end,
                manual_return_dates=manual_return_dates
            )

        # Run simulation with actual starts and IR handling
        result = self.simulate_season(
            sim_roster,
            categories,
            initial_starts_used=initial_starts,
            ir_players=ir_players,
            include_ir_returns=include_ir_returns,
            team_id=team_id
        )

        # Build adjusted totals based on actual starts
        category_keys = [c.get('stat_key', c.get('name', '')) for c in categories]
        adjusted_totals = defaultdict(float)
        fgm_total, fga_total, ftm_total, fta_total = 0.0, 0.0, 0.0, 0.0

        player_assignments = []

        # Build set of IR player IDs to skip in main loop (they're added separately)
        ir_player_ids = {ir.player_id for ir in ir_players} if ir_players else set()

        for player in sim_roster:
            player_id = player['player_id']

            # Skip IR players - they're added in the IR return section
            if player_id in ir_player_ids:
                continue

            log = result.player_logs.get(player_id)

            if log is None:
                continue

            # Calculate start ratio
            games_available = log.total_games_available
            games_started = log.games_started
            start_ratio = games_started / games_available if games_available > 0 else 0

            # Adjust stats by start ratio
            original_stats = player.get('original_projected_stats', {})
            adjusted_stats = {}

            for key, total in original_stats.items():
                if key in ['FG%', 'FT%']:
                    adjusted_stats[key] = total  # Percentages stay same
                else:
                    adjusted_stats[key] = total * start_ratio

            # Accumulate team totals
            for key in category_keys:
                if key not in ['FG%', 'FT%']:
                    adjusted_totals[key] += adjusted_stats.get(key, 0)

            fgm_total += adjusted_stats.get('FGM', 0)
            fga_total += adjusted_stats.get('FGA', 0)
            ftm_total += adjusted_stats.get('FTM', 0)
            fta_total += adjusted_stats.get('FTA', 0)

            # Build assignment info
            primary_position = max(log.starts_by_position.items(), key=lambda x: x[1])[0] \
                if log.starts_by_position else 12  # Bench

            # Check if player was dropped for IR return
            is_dropped = player_id in result.dropped_players

            player_assignments.append({
                'player_id': player_id,
                'name': log.player_name,
                'nba_team': log.nba_team,
                'assigned_position': SLOT_ID_TO_POSITION.get(primary_position, 'BE'),
                'projected_games': games_available,
                'actual_games_to_start': games_started,
                'games_benched': log.games_benched,
                'start_percentage': log.start_percentage,
                'is_benched': games_started == 0,
                'is_dropped': is_dropped,
                'starts_by_position': {
                    SLOT_ID_TO_POSITION.get(k, f'?{k}'): v
                    for k, v in log.starts_by_position.items()
                },
                'adjusted_stats': adjusted_stats,
            })

        # Add IR players who returned to player_assignments
        if include_ir_returns and result.ir_players:
            for ir_player in result.ir_players:
                if not ir_player.will_return_before_season_end:
                    continue

                log = result.player_logs.get(ir_player.player_id)
                if log is None:
                    continue

                games_available = log.total_games_available
                games_started = log.games_started

                # Calculate adjusted stats from per-game stats * games started
                adjusted_stats = {}
                per_game = ir_player.per_game_stats
                for key in category_keys:
                    if key not in ['FG%', 'FT%']:
                        adjusted_stats[key] = per_game.get(key, 0) * games_started

                # Add to team totals
                for key in category_keys:
                    if key not in ['FG%', 'FT%']:
                        adjusted_totals[key] += adjusted_stats.get(key, 0)

                # Add to FG/FT totals for percentage calculation
                fgm_total += per_game.get('FGM', 0) * games_started
                fga_total += per_game.get('FGA', 0) * games_started
                ftm_total += per_game.get('FTM', 0) * games_started
                fta_total += per_game.get('FTA', 0) * games_started

                primary_position = max(log.starts_by_position.items(), key=lambda x: x[1])[0] \
                    if log.starts_by_position else 12

                player_assignments.append({
                    'player_id': ir_player.player_id,
                    'name': ir_player.player_name,
                    'nba_team': ir_player.nba_team,
                    'assigned_position': SLOT_ID_TO_POSITION.get(primary_position, 'BE'),
                    'projected_games': games_available,
                    'actual_games_to_start': games_started,
                    'games_benched': log.games_benched,
                    'start_percentage': log.start_percentage,
                    'is_benched': games_started == 0,
                    'is_dropped': False,
                    'is_ir_return': True,
                    'ir_return_date': ir_player.projected_return_date.isoformat() if ir_player.projected_return_date else None,
                    'replacing_player': ir_player.replacing_player_name,
                    'starts_by_position': {
                        SLOT_ID_TO_POSITION.get(k, f'?{k}'): v
                        for k, v in log.starts_by_position.items()
                    },
                    'adjusted_stats': adjusted_stats,
                })

        # Calculate percentages
        adjusted_totals['FGM'] = fgm_total
        adjusted_totals['FGA'] = fga_total
        adjusted_totals['FTM'] = ftm_total
        adjusted_totals['FTA'] = fta_total
        adjusted_totals['FG%'] = fgm_total / fga_total if fga_total > 0 else 0.0
        adjusted_totals['FT%'] = ftm_total / fta_total if fta_total > 0 else 0.0

        # Log summary
        self._log(f"\nAdjusted team totals for {team_name}:")
        for key, value in sorted(adjusted_totals.items()):
            if key in ['FG%', 'FT%']:
                self._log(f"  {key}: {value:.3f}")
            else:
                self._log(f"  {key}: {value:.1f}")

        return dict(adjusted_totals), player_assignments, result.ir_players

    def build_team_start_limits(
        self,
        team_id: int,
        team_name: str
    ) -> Dict[str, Any]:
        """
        Build start limit info for a team (for API response).
        """
        slot_counts = self.get_lineup_slot_counts()
        stat_limits = self.get_lineup_slot_stat_limits()

        position_limits = {}
        for slot_id, count in slot_counts.items():
            if count == 0 or slot_id not in STARTING_SLOT_IDS:
                continue

            pos_name = SLOT_ID_TO_POSITION.get(slot_id, f'SLOT_{slot_id}')
            limit = stat_limits.get(slot_id, count * DEFAULT_GAMES_PER_SLOT)

            position_limits[pos_name] = {
                'slot_id': slot_id,
                'slots': count,
                'games_limit': limit,
            }

        return {
            'position_limits': position_limits,
            'total_limit': sum(stat_limits.values()),
        }

    def get_slot(self, slot_id: int) -> Optional[Dict]:
        """Get info for a specific slot (for compatibility)."""
        slot_counts = self.get_lineup_slot_counts()
        stat_limits = self.get_lineup_slot_stat_limits()

        count = slot_counts.get(slot_id, 0)
        if count == 0:
            return None

        return {
            'slot_id': slot_id,
            'position_name': SLOT_ID_TO_POSITION.get(slot_id, f'SLOT_{slot_id}'),
            'slot_count': count,
            'games_limit': stat_limits.get(slot_id, count * DEFAULT_GAMES_PER_SLOT),
            'games_used': 0,  # Would need actual tracking
            'games_remaining': stat_limits.get(slot_id, count * DEFAULT_GAMES_PER_SLOT),
        }


def create_optimizer_from_league(
    league_id: int,
    espn_s2: str,
    swid: str,
    season: int,
    verbose: bool = True
) -> StartLimitOptimizer:
    """Factory function to create an optimizer from league credentials."""
    return StartLimitOptimizer(
        espn_s2=espn_s2,
        swid=swid,
        league_id=league_id,
        season=season,
        verbose=verbose
    )


# Test script
if __name__ == '__main__':
    import sys
    import os
    import sqlite3

    print("=" * 70)
    print("DAY-BY-DAY START LIMIT OPTIMIZER TEST")
    print("=" * 70)

    # Get credentials from database
    # __file__ is backend/projections/start_limit_optimizer.py
    # Go up 2 levels to get project root, then into instance/
    db_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        'instance',
        'fantasy_basketball.db'
    )

    if not os.path.exists(db_path):
        print(f"Database not found at {db_path}")
        sys.exit(1)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT espn_league_id, espn_s2_cookie, swid_cookie, season
        FROM leagues LIMIT 1
    ''')
    row = cursor.fetchone()
    conn.close()

    if not row:
        print("No league found in database")
        sys.exit(1)

    league_id, espn_s2, swid, season = row

    print(f"League ID: {league_id}, Season: {season}")

    # Create optimizer
    optimizer = StartLimitOptimizer(
        espn_s2=espn_s2,
        swid=swid,
        league_id=int(league_id),
        season=int(season),
        verbose=True
    )

    # Fetch roster
    print("\nFetching roster...")
    endpoint = (
        f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/fba/"
        f"seasons/{season}/segments/0/leagues/{league_id}"
    )
    params = {'view': ['mTeam', 'mRoster']}
    cookies = {'espn_s2': espn_s2, 'SWID': swid}

    response = requests.get(endpoint, params=params, cookies=cookies, timeout=15)
    data = response.json()
    teams = data.get('teams', [])

    # Find Doncic Donuts or first team
    my_team = None
    for team in teams:
        name = team.get('name', '') or team.get('nickname', '')
        if 'doncic' in name.lower():
            my_team = team
            break
    if not my_team:
        my_team = teams[0]

    team_id = my_team.get('id')
    team_name = my_team.get('name', '') or my_team.get('nickname', 'Unknown')

    print(f"Team: {team_name}")

    # Build roster for simulation
    roster_entries = my_team.get('roster', {}).get('entries', [])
    sim_roster = []

    # NBA team mapping from proTeamId
    PRO_TEAM_MAP = {
        1: 'ATL', 2: 'BOS', 3: 'NOP', 4: 'CHI', 5: 'CLE',
        6: 'DAL', 7: 'DEN', 8: 'DET', 9: 'GSW', 10: 'HOU',
        11: 'IND', 12: 'LAC', 13: 'LAL', 14: 'MIA', 15: 'MIL',
        16: 'MIN', 17: 'BKN', 18: 'NYK', 19: 'ORL', 20: 'PHI',
        21: 'PHX', 22: 'POR', 23: 'SAC', 24: 'SAS', 25: 'OKC',
        26: 'UTA', 27: 'WAS', 28: 'TOR', 29: 'MEM', 30: 'CHA',
    }

    for entry in roster_entries:
        player = entry.get('playerPoolEntry', {}).get('player', {})
        player_id = player.get('id', 0)
        player_name = player.get('fullName', 'Unknown')
        pro_team_id = player.get('proTeamId', 0)
        nba_team = PRO_TEAM_MAP.get(pro_team_id, 'UNK')
        eligible_slots = player.get('eligibleSlots', [])
        lineup_slot_id = entry.get('lineupSlotId', 0)
        injury_status = player.get('injuryStatus', 'ACTIVE')
        season_outlook = player.get('seasonOutlook', '')
        droppable = player.get('droppable', True)

        # Get per-game stats
        per_game_stats = {}
        for stat_set in player.get('stats', []):
            if stat_set.get('statSourceId') == 0:  # Current season
                avg_stats = stat_set.get('averageStats', {})
                stat_id_map = {
                    0: 'PTS', 6: 'REB', 3: 'AST', 2: 'STL', 1: 'BLK',
                    17: '3PM', 13: 'FGM', 14: 'FGA', 15: 'FTM', 16: 'FTA',
                }
                for stat_id, stat_name in stat_id_map.items():
                    per_game_stats[stat_name] = avg_stats.get(str(stat_id), 0) or 0
                break

        # Estimate projected games (40 remaining)
        projected_games = 40
        projected_stats = {k: v * projected_games for k, v in per_game_stats.items()}

        # Show IR players
        slot_name = SLOT_ID_TO_POSITION.get(lineup_slot_id, f'?{lineup_slot_id}')
        if lineup_slot_id == 13:
            print(f"  IR PLAYER: {player_name} ({nba_team}) - {injury_status}")

        sim_roster.append({
            'player_id': player_id,
            'name': player_name,
            'nba_team': nba_team,
            'eligible_slots': eligible_slots,
            'per_game_stats': per_game_stats,
            'projected_stats': projected_stats,
            'projected_games': projected_games,
            'lineupSlotId': lineup_slot_id,
            'injury_status': injury_status,
            'season_outlook': season_outlook,
            'droppable': droppable,
        })

    print(f"Roster: {len(sim_roster)} players")

    # Fetch and display actual ESPN starts used
    print("\n" + "=" * 70)
    print("ACTUAL ESPN STARTS USED (fetched from API)")
    print("=" * 70)
    actual_starts = optimizer.fetch_starts_used(team_id)
    stat_limits = optimizer.get_lineup_slot_stat_limits()

    if actual_starts:
        print(f"\n{'Position':<8} {'Used':<8} {'Limit':<8} {'Remaining':<10}")
        print("-" * 40)
        for slot_id in sorted(actual_starts.keys()):
            pos_name = SLOT_ID_TO_POSITION.get(slot_id, f'?{slot_id}')
            used = actual_starts[slot_id]
            limit = stat_limits.get(slot_id, 82)
            remaining = limit - used
            print(f"{pos_name:<8} {used:<8} {limit:<8} {remaining:<10}")
    else:
        print("Could not fetch starts data from ESPN")

    # Run simulation with IR handling enabled
    categories = [
        {'stat_key': 'PTS'}, {'stat_key': 'REB'}, {'stat_key': 'AST'},
        {'stat_key': 'STL'}, {'stat_key': 'BLK'}, {'stat_key': '3PM'},
        {'stat_key': 'FG%'}, {'stat_key': 'FT%'},
    ]

    # Test WITH IR returns
    print("\n" + "=" * 70)
    print("RUNNING OPTIMIZATION WITH IR RETURNS ENABLED")
    print("=" * 70)

    adjusted_totals, assignments, ir_players = optimizer.optimize_team_projections(
        team_id=team_id,
        team_name=team_name,
        roster=sim_roster,
        categories=categories,
        include_ir_returns=True  # Enable IR handling
    )

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    # Show IR players first
    if ir_players:
        print("\nIR PLAYERS:")
        print("-" * 70)
        for ir in ir_players:
            status = "RETURNING" if ir.will_return_before_season_end else "NOT RETURNING"
            return_info = f"Return: {ir.projected_return_date}" if ir.projected_return_date else "N/A"
            replacing = f"Replacing: {ir.replacing_player_name}" if ir.replacing_player_name else ""
            games = f"+{ir.games_after_return} games" if ir.games_after_return else ""
            print(f"  {ir.player_name} ({ir.nba_team}) - {status}")
            print(f"    {return_info}, {replacing} {games}")

    print(f"\n{'Player':<25} {'Team':<5} {'Games':<8} {'Started':<8} {'%':<6} {'Status':<12}")
    print("-" * 75)

    for a in sorted(assignments, key=lambda x: x['actual_games_to_start'], reverse=True):
        # Determine status
        if a.get('is_dropped'):
            status = "DROPPED"
        elif a.get('is_ir_return'):
            status = "IR RETURN"
        elif a['actual_games_to_start'] == 0:
            status = "BENCHED"
        else:
            status = "ACTIVE"

        print(f"{a['name'][:24]:<25} {a['nba_team']:<5} "
              f"{a['projected_games']:<8} {a['actual_games_to_start']:<8} "
              f"{a['start_percentage']:.0f}%{'':<3} {status:<12}")

    print("\nAdjusted Team Totals:")
    for key, value in sorted(adjusted_totals.items()):
        if key in ['FG%', 'FT%']:
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value:.0f}")

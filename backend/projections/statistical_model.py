#!/usr/bin/env python3
"""
Statistical Projection Component for Fantasy Basketball.

This module provides statistical projections by aggregating data from multiple
sources (Basketball Reference, ESPN) and applying adjustments for:
- Games remaining in season
- Recent performance trends (last 15 games)
- Injury status
- Schedule strength

The statistical projections complement the ML models and are combined
in the hybrid engine for final projections.

Reference: PRD Section 3.3 - Projection Engine
"""

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# NBA season constants
NBA_SEASON_GAMES = 82
NBA_REGULAR_SEASON_START_MONTH = 10  # October
NBA_REGULAR_SEASON_END_MONTH = 4     # April
NBA_PLAYOFF_START_MONTH = 4          # April

# Source reliability weights (sum to 1.0)
SOURCE_WEIGHTS = {
    'basketball_reference': 0.45,
    'espn': 0.35,
    'season_average': 0.20,
}

# Recency weights for trending (exponential decay)
RECENCY_WEIGHTS = {
    'last_7': 0.40,
    'last_15': 0.35,
    'last_30': 0.25,
}

# Injury status adjustments (multiplier for games played projection)
INJURY_ADJUSTMENTS = {
    'ACTIVE': 1.0,
    'DAY_TO_DAY': 0.90,
    'OUT': 0.0,           # Currently out
    'DOUBTFUL': 0.25,
    'QUESTIONABLE': 0.75,
    'PROBABLE': 0.95,
    'SUSPENSION': 0.0,
    'INJ_RESERVE': 0.0,   # Injured reserve
    'GTD': 0.85,          # Game time decision
}

# Stat categories for fantasy basketball
STAT_CATEGORIES = [
    'pts', 'trb', 'ast', 'stl', 'blk', 'tov',
    '3p', 'fg_pct', 'ft_pct', 'fgm', 'fga', 'ftm', 'fta'
]

# Counting stats (per-game, scale with games played)
COUNTING_STATS = ['pts', 'trb', 'ast', 'stl', 'blk', 'tov', '3p', 'fgm', 'fga', 'ftm', 'fta']

# Rate stats (percentages, don't scale directly)
RATE_STATS = ['fg_pct', 'ft_pct', '3p_pct']

# Default confidence level for intervals
DEFAULT_CONFIDENCE = 0.90


# =============================================================================
# Data Classes
# =============================================================================

class ProjectionSource(Enum):
    """Enumeration of projection data sources."""
    BASKETBALL_REFERENCE = "basketball_reference"
    ESPN = "espn"
    SEASON_AVERAGE = "season_average"
    RECENT_TREND = "recent_trend"


@dataclass
class SourceProjection:
    """Projection from a single source."""
    source: ProjectionSource
    stats: Dict[str, float]
    games_projected: int
    reliability: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PlayerProjection:
    """Complete projection for a player."""
    player_id: str
    player_name: str
    team: str
    position: str

    # Projected per-game stats
    projected_stats: Dict[str, float]

    # Rest-of-season totals
    ros_totals: Dict[str, float]

    # Games projections
    games_played: int
    games_remaining: int
    games_projected_ros: int

    # Confidence intervals
    confidence_intervals: Dict[str, Tuple[float, float]]
    confidence_level: float

    # Adjustments applied
    injury_adjustment: float
    trend_adjustment: Dict[str, float]
    schedule_adjustment: float

    # Metadata
    sources_used: List[str]
    projection_date: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'player_id': self.player_id,
            'player_name': self.player_name,
            'team': self.team,
            'position': self.position,
            'projected_stats': self.projected_stats,
            'ros_totals': self.ros_totals,
            'games_played': self.games_played,
            'games_remaining': self.games_remaining,
            'games_projected_ros': self.games_projected_ros,
            'confidence_intervals': {
                k: {'low': v[0], 'high': v[1]}
                for k, v in self.confidence_intervals.items()
            },
            'confidence_level': self.confidence_level,
            'injury_adjustment': self.injury_adjustment,
            'schedule_adjustment': self.schedule_adjustment,
            'sources_used': self.sources_used,
            'projection_date': self.projection_date.isoformat(),
        }


@dataclass
class TeamScheduleStrength:
    """Schedule strength metrics for a team."""
    team: str
    remaining_games: int
    avg_opponent_rating: float  # Higher = harder schedule
    back_to_backs: int
    home_games: int
    away_games: int
    strength_rating: float  # 0.8-1.2 multiplier


# =============================================================================
# Statistical Projection Engine
# =============================================================================

class StatisticalProjectionEngine:
    """
    Statistical projection engine that aggregates multiple data sources
    and applies various adjustments to generate player projections.
    """

    def __init__(
        self,
        source_weights: Optional[Dict[str, float]] = None,
        confidence_level: float = DEFAULT_CONFIDENCE
    ):
        """
        Initialize the statistical projection engine.

        Args:
            source_weights: Custom weights for data sources
            confidence_level: Confidence level for intervals (0-1)
        """
        self.source_weights = source_weights or SOURCE_WEIGHTS
        self.confidence_level = confidence_level

        # Normalize weights to sum to 1.0
        total_weight = sum(self.source_weights.values())
        self.source_weights = {
            k: v / total_weight for k, v in self.source_weights.items()
        }

        # Cache for schedule strength
        self._schedule_cache: Dict[str, TeamScheduleStrength] = {}

    def project_player(
        self,
        player_data: Dict[str, Any],
        season_stats: Optional[Dict[str, float]] = None,
        recent_stats: Optional[Dict[str, List[float]]] = None,
        espn_projection: Optional[Dict[str, float]] = None,
        bbref_projection: Optional[Dict[str, float]] = None,
        injury_status: str = 'ACTIVE',
        team_schedule: Optional[TeamScheduleStrength] = None,
        games_remaining_in_season: Optional[int] = None
    ) -> PlayerProjection:
        """
        Generate a statistical projection for a player.

        Args:
            player_data: Basic player info (id, name, team, position)
            season_stats: Current season per-game averages
            recent_stats: Recent game stats (last 7, 15, 30 games)
            espn_projection: ESPN's rest-of-season projection
            bbref_projection: Basketball Reference projection
            injury_status: Current injury status
            team_schedule: Team's remaining schedule strength
            games_remaining_in_season: Games left in NBA season

        Returns:
            PlayerProjection object with all projections and metadata
        """
        player_id = player_data.get('player_id', 'unknown')
        player_name = player_data.get('name', 'Unknown Player')
        team = player_data.get('team', 'UNK')
        position = player_data.get('position', 'N/A')
        games_played = player_data.get('games_played', 0)

        logger.debug(f"Projecting: {player_name} ({team})")

        # Calculate games remaining
        if games_remaining_in_season is None:
            games_remaining_in_season = self._estimate_games_remaining()

        # Estimate player's remaining games based on current pace and injury
        player_game_rate = games_played / max(1, self._games_into_season())
        projected_ros_games = int(games_remaining_in_season * player_game_rate)

        # Apply injury adjustment
        injury_multiplier = INJURY_ADJUSTMENTS.get(
            injury_status.upper(), INJURY_ADJUSTMENTS['ACTIVE']
        )
        projected_ros_games = int(projected_ros_games * injury_multiplier)

        # Collect projections from all sources
        source_projections = []

        # Source 1: Basketball Reference
        if bbref_projection:
            source_projections.append(SourceProjection(
                source=ProjectionSource.BASKETBALL_REFERENCE,
                stats=bbref_projection,
                games_projected=projected_ros_games,
                reliability=self.source_weights.get('basketball_reference', 0.45)
            ))

        # Source 2: ESPN
        if espn_projection:
            source_projections.append(SourceProjection(
                source=ProjectionSource.ESPN,
                stats=espn_projection,
                games_projected=projected_ros_games,
                reliability=self.source_weights.get('espn', 0.35)
            ))

        # Source 3: Season average
        if season_stats:
            source_projections.append(SourceProjection(
                source=ProjectionSource.SEASON_AVERAGE,
                stats=season_stats,
                games_projected=projected_ros_games,
                reliability=self.source_weights.get('season_average', 0.20)
            ))

        # Aggregate projections with weights
        projected_stats = self._aggregate_projections(source_projections)

        # Calculate recent trend adjustments
        trend_adjustments = {}
        if recent_stats and season_stats:
            trend_adjustments = self._calculate_trend_adjustments(
                season_stats, recent_stats
            )
            projected_stats = self._apply_trend_adjustments(
                projected_stats, trend_adjustments
            )

        # Apply schedule strength adjustment
        schedule_multiplier = 1.0
        if team_schedule:
            schedule_multiplier = team_schedule.strength_rating
            projected_stats = self._apply_schedule_adjustment(
                projected_stats, schedule_multiplier
            )

        # Calculate rest-of-season totals
        ros_totals = self._calculate_ros_totals(
            projected_stats, projected_ros_games
        )

        # Calculate confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(
            projected_stats,
            source_projections,
            self.confidence_level
        )

        # Build projection object
        projection = PlayerProjection(
            player_id=player_id,
            player_name=player_name,
            team=team,
            position=position,
            projected_stats=projected_stats,
            ros_totals=ros_totals,
            games_played=games_played,
            games_remaining=games_remaining_in_season,
            games_projected_ros=projected_ros_games,
            confidence_intervals=confidence_intervals,
            confidence_level=self.confidence_level,
            injury_adjustment=injury_multiplier,
            trend_adjustment=trend_adjustments,
            schedule_adjustment=schedule_multiplier,
            sources_used=[sp.source.value for sp in source_projections]
        )

        return projection

    def project_roster(
        self,
        players: List[Dict[str, Any]],
        season_stats_map: Dict[str, Dict[str, float]],
        recent_stats_map: Optional[Dict[str, Dict[str, List[float]]]] = None,
        espn_projections: Optional[Dict[str, Dict[str, float]]] = None,
        bbref_projections: Optional[Dict[str, Dict[str, float]]] = None,
        injury_statuses: Optional[Dict[str, str]] = None,
        team_schedules: Optional[Dict[str, TeamScheduleStrength]] = None
    ) -> List[PlayerProjection]:
        """
        Generate projections for an entire roster.

        Args:
            players: List of player data dictionaries
            season_stats_map: Map of player_id -> season stats
            recent_stats_map: Map of player_id -> recent stats
            espn_projections: Map of player_id -> ESPN projection
            bbref_projections: Map of player_id -> BBRef projection
            injury_statuses: Map of player_id -> injury status
            team_schedules: Map of team -> schedule strength

        Returns:
            List of PlayerProjection objects
        """
        projections = []
        games_remaining = self._estimate_games_remaining()

        for player in players:
            player_id = player.get('player_id', '')
            team = player.get('team', '')

            try:
                projection = self.project_player(
                    player_data=player,
                    season_stats=season_stats_map.get(player_id),
                    recent_stats=recent_stats_map.get(player_id) if recent_stats_map else None,
                    espn_projection=espn_projections.get(player_id) if espn_projections else None,
                    bbref_projection=bbref_projections.get(player_id) if bbref_projections else None,
                    injury_status=injury_statuses.get(player_id, 'ACTIVE') if injury_statuses else 'ACTIVE',
                    team_schedule=team_schedules.get(team) if team_schedules else None,
                    games_remaining_in_season=games_remaining
                )
                projections.append(projection)
            except Exception as e:
                logger.warning(f"Failed to project {player.get('name', player_id)}: {e}")

        return projections

    # =========================================================================
    # Aggregation Methods
    # =========================================================================

    def _aggregate_projections(
        self,
        source_projections: List[SourceProjection]
    ) -> Dict[str, float]:
        """
        Aggregate projections from multiple sources using weighted average.

        Args:
            source_projections: List of source projections with reliability weights

        Returns:
            Aggregated per-game stats
        """
        if not source_projections:
            return {}

        # Normalize reliability weights
        total_reliability = sum(sp.reliability for sp in source_projections)
        if total_reliability == 0:
            total_reliability = 1.0

        aggregated = {}

        # Get all stat categories across sources
        all_stats = set()
        for sp in source_projections:
            all_stats.update(sp.stats.keys())

        for stat in all_stats:
            weighted_sum = 0.0
            weight_sum = 0.0

            for sp in source_projections:
                if stat in sp.stats and sp.stats[stat] is not None:
                    value = sp.stats[stat]
                    weight = sp.reliability / total_reliability
                    weighted_sum += value * weight
                    weight_sum += weight

            if weight_sum > 0:
                aggregated[stat] = weighted_sum / weight_sum

        return aggregated

    def _calculate_trend_adjustments(
        self,
        season_stats: Dict[str, float],
        recent_stats: Dict[str, List[float]]
    ) -> Dict[str, float]:
        """
        Calculate adjustments based on recent performance trends.

        Compares recent performance to season average to identify
        hot/cold streaks and adjust projections accordingly.

        Args:
            season_stats: Season per-game averages
            recent_stats: Dictionary with 'last_7', 'last_15', 'last_30' lists

        Returns:
            Dictionary of adjustment multipliers per stat
        """
        adjustments = {}

        for stat in COUNTING_STATS:
            if stat not in season_stats or season_stats[stat] == 0:
                continue

            season_avg = season_stats[stat]

            # Calculate weighted recent average
            recent_weighted = 0.0
            weight_sum = 0.0

            for period, weight in RECENCY_WEIGHTS.items():
                if period in recent_stats and recent_stats[period]:
                    # Get the stat values for this period
                    period_values = recent_stats[period]
                    if isinstance(period_values, dict) and stat in period_values:
                        period_avg = period_values[stat]
                    elif isinstance(period_values, list):
                        # Assume it's a list of game stats
                        period_avg = np.mean(period_values) if period_values else season_avg
                    else:
                        continue

                    recent_weighted += period_avg * weight
                    weight_sum += weight

            if weight_sum > 0:
                recent_avg = recent_weighted / weight_sum

                # Calculate adjustment as ratio of recent to season
                # Cap adjustments to prevent extreme swings
                adjustment_ratio = recent_avg / season_avg
                adjustment_ratio = np.clip(adjustment_ratio, 0.75, 1.25)

                # Apply dampening to prevent overreacting
                # Final adjustment = 1 + (ratio - 1) * 0.5
                dampened_adjustment = 1.0 + (adjustment_ratio - 1.0) * 0.5
                adjustments[stat] = dampened_adjustment

        return adjustments

    def _apply_trend_adjustments(
        self,
        projected_stats: Dict[str, float],
        trend_adjustments: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Apply trend adjustments to projected stats.

        Args:
            projected_stats: Base projected stats
            trend_adjustments: Adjustment multipliers

        Returns:
            Adjusted projected stats
        """
        adjusted = projected_stats.copy()

        for stat, adjustment in trend_adjustments.items():
            if stat in adjusted:
                adjusted[stat] = adjusted[stat] * adjustment

        return adjusted

    def _apply_schedule_adjustment(
        self,
        projected_stats: Dict[str, float],
        schedule_multiplier: float
    ) -> Dict[str, float]:
        """
        Apply schedule strength adjustment.

        Harder schedules (multiplier > 1.0) slightly reduce counting stats.
        Easier schedules (multiplier < 1.0) slightly increase counting stats.

        Args:
            projected_stats: Projected stats
            schedule_multiplier: Schedule strength (0.8-1.2)

        Returns:
            Adjusted stats
        """
        adjusted = projected_stats.copy()

        # Invert multiplier for stats (harder schedule = lower stats)
        stat_multiplier = 2.0 - schedule_multiplier  # 0.8 -> 1.2, 1.2 -> 0.8

        # Apply to counting stats only, with dampening
        for stat in COUNTING_STATS:
            if stat in adjusted:
                # Apply only 50% of the adjustment to be conservative
                dampened = 1.0 + (stat_multiplier - 1.0) * 0.5
                adjusted[stat] = adjusted[stat] * dampened

        return adjusted

    def _calculate_ros_totals(
        self,
        per_game_stats: Dict[str, float],
        games_remaining: int
    ) -> Dict[str, float]:
        """
        Calculate rest-of-season totals from per-game stats.

        Args:
            per_game_stats: Per-game projected stats
            games_remaining: Projected games to play

        Returns:
            Rest-of-season total stats
        """
        totals = {}

        for stat, value in per_game_stats.items():
            if stat in COUNTING_STATS:
                # Scale counting stats by games
                totals[stat] = value * games_remaining
            elif stat in RATE_STATS:
                # Rate stats stay as-is
                totals[stat] = value
            else:
                # Unknown stats, keep per-game value
                totals[stat] = value

        return totals

    # =========================================================================
    # Confidence Interval Calculation
    # =========================================================================

    def _calculate_confidence_intervals(
        self,
        projected_stats: Dict[str, float],
        source_projections: List[SourceProjection],
        confidence_level: float
    ) -> Dict[str, Tuple[float, float]]:
        """
        Calculate confidence intervals for projected stats.

        Uses variance across sources and historical accuracy to
        determine uncertainty bands.

        Args:
            projected_stats: Aggregated projected stats
            source_projections: Individual source projections
            confidence_level: Confidence level (e.g., 0.90 for 90%)

        Returns:
            Dictionary of (low, high) tuples per stat
        """
        intervals = {}

        # Z-score for confidence level
        z_score = scipy_stats.norm.ppf((1 + confidence_level) / 2)

        for stat in projected_stats:
            values = []

            # Collect values from all sources
            for sp in source_projections:
                if stat in sp.stats and sp.stats[stat] is not None:
                    values.append(sp.stats[stat])

            if len(values) >= 2:
                # Calculate standard error from source variance
                std_dev = np.std(values, ddof=1)
                mean = projected_stats[stat]

                # Margin of error
                margin = z_score * std_dev

                # Apply minimum margin based on stat type
                min_margin = self._get_minimum_margin(stat, mean)
                margin = max(margin, min_margin)

                intervals[stat] = (
                    max(0, mean - margin),  # Lower bound (non-negative)
                    mean + margin            # Upper bound
                )
            elif len(values) == 1:
                # Single source - use default uncertainty
                mean = projected_stats[stat]
                default_pct = 0.15  # 15% default uncertainty
                margin = mean * default_pct

                intervals[stat] = (
                    max(0, mean - margin),
                    mean + margin
                )
            else:
                # No data - no interval
                intervals[stat] = (0, 0)

        return intervals

    def _get_minimum_margin(self, stat: str, mean: float) -> float:
        """
        Get minimum margin of error for a stat based on typical variance.

        Args:
            stat: Stat category
            mean: Projected mean value

        Returns:
            Minimum margin of error
        """
        # Percentage-based minimums
        pct_minimums = {
            'pts': 0.10,      # 10% of points
            'trb': 0.12,      # 12% of rebounds
            'ast': 0.15,      # 15% of assists
            'stl': 0.20,      # 20% of steals (more volatile)
            'blk': 0.25,      # 25% of blocks (more volatile)
            'tov': 0.15,      # 15% of turnovers
            '3p': 0.20,       # 20% of 3PM
            'fg_pct': 0.02,   # 2 percentage points
            'ft_pct': 0.03,   # 3 percentage points
        }

        pct = pct_minimums.get(stat, 0.15)
        return mean * pct if stat not in RATE_STATS else pct

    # =========================================================================
    # Season & Schedule Utilities
    # =========================================================================

    def _games_into_season(self) -> int:
        """
        Estimate how many games into the NBA season we are.

        Returns:
            Estimated games played in season so far
        """
        today = date.today()

        # Approximate season start (October 22)
        if today.month >= NBA_REGULAR_SEASON_START_MONTH:
            season_start = date(today.year, NBA_REGULAR_SEASON_START_MONTH, 22)
        else:
            season_start = date(today.year - 1, NBA_REGULAR_SEASON_START_MONTH, 22)

        days_elapsed = (today - season_start).days

        if days_elapsed < 0:
            return 0

        # Approximate: 82 games over ~175 days
        games_per_day = NBA_SEASON_GAMES / 175
        estimated_games = int(days_elapsed * games_per_day)

        return min(estimated_games, NBA_SEASON_GAMES)

    def _estimate_games_remaining(self) -> int:
        """
        Estimate games remaining in the NBA season.

        Returns:
            Estimated games remaining
        """
        games_played = self._games_into_season()
        return max(0, NBA_SEASON_GAMES - games_played)

    def calculate_schedule_strength(
        self,
        team: str,
        remaining_opponents: List[Dict[str, Any]],
        league_avg_rating: float = 100.0
    ) -> TeamScheduleStrength:
        """
        Calculate schedule strength for a team's remaining games.

        Args:
            team: Team abbreviation
            remaining_opponents: List of opponent info dicts
            league_avg_rating: League average opponent rating

        Returns:
            TeamScheduleStrength object
        """
        if not remaining_opponents:
            return TeamScheduleStrength(
                team=team,
                remaining_games=0,
                avg_opponent_rating=league_avg_rating,
                back_to_backs=0,
                home_games=0,
                away_games=0,
                strength_rating=1.0
            )

        remaining_games = len(remaining_opponents)

        # Calculate average opponent rating
        opponent_ratings = [
            opp.get('rating', league_avg_rating)
            for opp in remaining_opponents
        ]
        avg_rating = np.mean(opponent_ratings)

        # Count back-to-backs
        back_to_backs = sum(
            1 for opp in remaining_opponents
            if opp.get('is_back_to_back', False)
        )

        # Count home/away
        home_games = sum(
            1 for opp in remaining_opponents
            if opp.get('is_home', True)
        )
        away_games = remaining_games - home_games

        # Calculate strength rating (0.8 to 1.2 scale)
        # Higher rating = harder schedule
        rating_factor = avg_rating / league_avg_rating
        b2b_penalty = 1 + (back_to_backs / remaining_games) * 0.1
        away_penalty = 1 + ((away_games / remaining_games) - 0.5) * 0.05

        strength_rating = np.clip(
            rating_factor * b2b_penalty * away_penalty,
            0.8, 1.2
        )

        return TeamScheduleStrength(
            team=team,
            remaining_games=remaining_games,
            avg_opponent_rating=avg_rating,
            back_to_backs=back_to_backs,
            home_games=home_games,
            away_games=away_games,
            strength_rating=strength_rating
        )


# =============================================================================
# Source-Specific Adapters
# =============================================================================

class BasketballReferenceAdapter:
    """
    Adapter for converting Basketball Reference data to projection format.
    """

    @staticmethod
    def convert_stats(bbref_stats: Dict[str, Any]) -> Dict[str, float]:
        """
        Convert Basketball Reference stats to standard projection format.

        Args:
            bbref_stats: Raw stats from Basketball Reference scraper

        Returns:
            Normalized stats dictionary
        """
        # Mapping from BBRef column names to our standard names
        mapping = {
            'pts': 'pts',
            'trb': 'trb',
            'ast': 'ast',
            'stl': 'stl',
            'blk': 'blk',
            'tov': 'tov',
            '3p': '3p',
            'fg_pct': 'fg_pct',
            'ft_pct': 'ft_pct',
            'fg': 'fgm',
            'fga': 'fga',
            'ft': 'ftm',
            'fta': 'fta',
            'mp': 'minutes',
            'g': 'games',
        }

        converted = {}
        for bbref_key, standard_key in mapping.items():
            if bbref_key in bbref_stats:
                value = bbref_stats[bbref_key]
                if value is not None and not (isinstance(value, float) and math.isnan(value)):
                    converted[standard_key] = float(value)

        return converted

    @staticmethod
    def convert_recent_games(game_logs: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Convert game log DataFrame to recent stats format.

        Args:
            game_logs: DataFrame of recent game logs

        Returns:
            Dictionary with 'last_7', 'last_15', 'last_30' averages
        """
        if game_logs.empty:
            return {}

        recent = {}

        # Sort by date descending
        if 'date' in game_logs.columns:
            game_logs = game_logs.sort_values('date', ascending=False)

        for period, n_games in [('last_7', 7), ('last_15', 15), ('last_30', 30)]:
            subset = game_logs.head(n_games)
            if len(subset) > 0:
                period_avg = {}
                for col in ['pts', 'trb', 'ast', 'stl', 'blk', 'tov', '3p']:
                    if col in subset.columns:
                        period_avg[col] = subset[col].mean()
                recent[period] = period_avg

        return recent


class ESPNAdapter:
    """
    Adapter for converting ESPN API data to projection format.
    """

    @staticmethod
    def convert_player_stats(espn_player: Dict[str, Any]) -> Dict[str, float]:
        """
        Convert ESPN player stats to standard format.

        Args:
            espn_player: ESPN API player data

        Returns:
            Normalized stats dictionary
        """
        # ESPN uses different stat IDs
        # This mapping depends on the espn-api package structure
        stats = espn_player.get('stats', {})

        if isinstance(stats, dict):
            # Already in dict format
            mapping = {
                'PTS': 'pts',
                'REB': 'trb',
                'AST': 'ast',
                'STL': 'stl',
                'BLK': 'blk',
                'TO': 'tov',
                '3PM': '3p',
                'FG%': 'fg_pct',
                'FT%': 'ft_pct',
                'FGM': 'fgm',
                'FGA': 'fga',
                'FTM': 'ftm',
                'FTA': 'fta',
            }

            converted = {}
            for espn_key, standard_key in mapping.items():
                if espn_key in stats:
                    converted[standard_key] = float(stats[espn_key])

            return converted

        return {}

    @staticmethod
    def convert_projection(espn_projection: Dict[str, Any]) -> Dict[str, float]:
        """
        Convert ESPN ROS projection to standard format.

        Args:
            espn_projection: ESPN projection data

        Returns:
            Normalized projection dictionary
        """
        return ESPNAdapter.convert_player_stats({'stats': espn_projection})


# =============================================================================
# Convenience Functions
# =============================================================================

def create_statistical_engine(
    source_weights: Optional[Dict[str, float]] = None,
    confidence_level: float = 0.90
) -> StatisticalProjectionEngine:
    """
    Factory function to create a configured statistical engine.

    Args:
        source_weights: Custom weights for sources
        confidence_level: Confidence level for intervals

    Returns:
        Configured StatisticalProjectionEngine
    """
    return StatisticalProjectionEngine(
        source_weights=source_weights,
        confidence_level=confidence_level
    )


def quick_projection(
    player_name: str,
    season_stats: Dict[str, float],
    games_played: int,
    injury_status: str = 'ACTIVE'
) -> PlayerProjection:
    """
    Quick single-player projection with minimal inputs.

    Args:
        player_name: Player's name
        season_stats: Current season per-game averages
        games_played: Games played so far
        injury_status: Injury status string

    Returns:
        PlayerProjection object
    """
    engine = StatisticalProjectionEngine()

    player_data = {
        'player_id': player_name.lower().replace(' ', '_'),
        'name': player_name,
        'team': 'UNK',
        'position': 'N/A',
        'games_played': games_played
    }

    return engine.project_player(
        player_data=player_data,
        season_stats=season_stats,
        injury_status=injury_status
    )


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """Demo/test entry point for statistical projections."""
    logger.info("=" * 60)
    logger.info("Statistical Projection Engine Demo")
    logger.info("=" * 60)

    # Create engine
    engine = StatisticalProjectionEngine()

    # Example player data
    player_data = {
        'player_id': 'test_player',
        'name': 'Demo Player',
        'team': 'LAL',
        'position': 'PG',
        'games_played': 45
    }

    # Example season stats
    season_stats = {
        'pts': 25.5,
        'trb': 5.2,
        'ast': 8.1,
        'stl': 1.3,
        'blk': 0.4,
        'tov': 3.2,
        '3p': 2.8,
        'fg_pct': 0.485,
        'ft_pct': 0.875,
    }

    # Example ESPN projection (slightly different)
    espn_projection = {
        'pts': 24.8,
        'trb': 5.0,
        'ast': 8.5,
        'stl': 1.2,
        'blk': 0.5,
        'tov': 3.0,
        '3p': 2.6,
        'fg_pct': 0.480,
        'ft_pct': 0.880,
    }

    # Example BBRef projection
    bbref_projection = {
        'pts': 26.0,
        'trb': 5.4,
        'ast': 7.8,
        'stl': 1.4,
        'blk': 0.4,
        'tov': 3.4,
        '3p': 3.0,
        'fg_pct': 0.490,
        'ft_pct': 0.870,
    }

    # Generate projection
    projection = engine.project_player(
        player_data=player_data,
        season_stats=season_stats,
        espn_projection=espn_projection,
        bbref_projection=bbref_projection,
        injury_status='ACTIVE'
    )

    # Print results
    print("\n" + "=" * 60)
    print(f"Projection for: {projection.player_name}")
    print("=" * 60)
    print(f"Team: {projection.team} | Position: {projection.position}")
    print(f"Games Played: {projection.games_played}")
    print(f"Projected ROS Games: {projection.games_projected_ros}")
    print(f"Injury Adjustment: {projection.injury_adjustment:.2f}")
    print(f"Schedule Adjustment: {projection.schedule_adjustment:.2f}")

    print("\nProjected Per-Game Stats:")
    print("-" * 40)
    for stat, value in sorted(projection.projected_stats.items()):
        ci = projection.confidence_intervals.get(stat, (0, 0))
        print(f"  {stat.upper():<8}: {value:>6.2f}  [{ci[0]:.2f} - {ci[1]:.2f}]")

    print("\nRest-of-Season Totals:")
    print("-" * 40)
    for stat in ['pts', 'trb', 'ast', 'stl', 'blk', '3p']:
        if stat in projection.ros_totals:
            print(f"  {stat.upper():<8}: {projection.ros_totals[stat]:>7.1f}")

    print("\nSources Used:", projection.sources_used)
    print("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())

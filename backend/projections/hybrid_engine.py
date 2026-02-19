#!/usr/bin/env python3
"""
Hybrid Projection Engine for Fantasy Basketball.

This module combines ML-based and statistical projections with dynamic
weighting based on season progress, adjusts for league-specific scoring,
and provides comprehensive player projections with confidence scores.

Weighting Strategy:
- All season phases: 80% statistical, 20% ML
- FT% uses regression-to-mean instead of ML (FT% is highly stable year-to-year)

The 80/20 ratio favors current season performance, as actual stats are
generally more predictive than ML model outputs for fantasy purposes.

Reference: PRD Section 3.3.4 - Hybrid Projection Engine
"""

import logging
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Try relative imports first (when used as a package), then absolute imports
ML_AVAILABLE = False
STAT_ENGINE_AVAILABLE = False
SIMPLE_ENGINE_AVAILABLE = False

try:
    from .ml_model import (
        FantasyBasketballMLModel,
        TrainedModel,
        COUNTING_STATS as ML_COUNTING_STATS,
        SHOOTING_STATS as ML_SHOOTING_STATS,
    )
    ML_AVAILABLE = True
except ImportError as e:
    logging.getLogger(__name__).warning(f"ML model import failed: {e}")
    FantasyBasketballMLModel = None
    TrainedModel = None
    ML_COUNTING_STATS = []
    ML_SHOOTING_STATS = []

try:
    from .statistical_model import (
        StatisticalProjectionEngine,
        PlayerProjection as StatisticalProjection,
        STAT_CATEGORIES,
        COUNTING_STATS,
        RATE_STATS,
    )
    STAT_ENGINE_AVAILABLE = True
except ImportError as e:
    logging.getLogger(__name__).warning(f"Statistical model import failed: {e}")
    StatisticalProjectionEngine = None
    StatisticalProjection = None
    STAT_CATEGORIES = ['pts', 'trb', 'ast', 'stl', 'blk', '3p', 'tov', 'fg_pct', 'ft_pct']
    COUNTING_STATS = ['pts', 'trb', 'ast', 'stl', 'blk', '3p', 'tov']
    RATE_STATS = ['fg_pct', 'ft_pct']

try:
    from .simple_projection import SimpleProjectionEngine
    SIMPLE_ENGINE_AVAILABLE = True
except ImportError:
    SimpleProjectionEngine = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Season progress thresholds for weight adjustment
EARLY_SEASON_THRESHOLD = 0.25   # First 25% of season
LATE_SEASON_THRESHOLD = 0.75    # Last 25% of season

# Games played thresholds for tiered weighting
GAMES_TIER_1_MAX = 5    # 0-5 games: rely heavily on projections
GAMES_TIER_2_MAX = 15   # 6-15 games: blend projections with emerging stats
GAMES_TIER_3_MAX = 35   # 16-35 games: current stats dominant
# 35+ games: 100% current season stats

# Tiered weights based on player's games played this season
# Sources: espn_projection, current_season, ml
# Note: Previous season stats removed - ESPN API doesn't provide them for current season connection
TIERED_WEIGHTS = {
    'tier_1': {  # 0-5 games
        'espn_projection': 0.90,
        'current_season': 0.00,
        'ml': 0.10,
    },
    'tier_2': {  # 6-15 games
        'espn_projection': 0.55,
        'current_season': 0.35,
        'ml': 0.10,
    },
    'tier_3': {  # 16-35 games
        'espn_projection': 0.15,
        'current_season': 0.80,
        'ml': 0.05,
    },
    'tier_4': {  # 35+ games
        'espn_projection': 0.00,
        'current_season': 1.00,
        'ml': 0.00,
    },
}

# Legacy weights (kept for backward compatibility)
SEASON_WEIGHTS = {
    'early': {'statistical': 0.80, 'ml': 0.20},
    'mid': {'statistical': 0.80, 'ml': 0.20},
    'late': {'statistical': 0.80, 'ml': 0.20},
}

# NBA season length
NBA_SEASON_GAMES = 82

# Standard fantasy stat categories
FANTASY_CATEGORIES = [
    'pts', 'trb', 'ast', 'stl', 'blk', 'tov',
    '3p', 'fg_pct', 'ft_pct', 'fgm', 'fga', 'ftm', 'fta'
]

# Default scoring weights for points leagues
DEFAULT_POINTS_SCORING = {
    'pts': 1.0,
    'trb': 1.2,
    'ast': 1.5,
    'stl': 3.0,
    'blk': 3.0,
    'tov': -1.0,
    '3p': 0.5,      # Additional bonus for 3PM
    'fgm': 0.0,     # Often not scored separately
    'fga': 0.0,
    'ftm': 0.0,
    'fta': 0.0,
    'fg_pct': 0.0,  # Handled via makes/attempts
    'ft_pct': 0.0,
}

# Category value weights for H2H category leagues
DEFAULT_CATEGORY_VALUES = {
    'pts': 1.0,
    'trb': 1.0,
    'ast': 1.0,
    'stl': 1.5,     # Scarcity premium
    'blk': 1.5,     # Scarcity premium
    'tov': -1.0,    # Negative value
    '3p': 1.0,
    'fg_pct': 1.2,  # Efficiency premium
    'ft_pct': 1.0,
}


# =============================================================================
# Data Classes
# =============================================================================

class SeasonPhase(Enum):
    """NBA season phase enumeration."""
    EARLY = "early"
    MID = "mid"
    LATE = "late"
    OFFSEASON = "offseason"


class LeagueType(Enum):
    """Fantasy league type enumeration."""
    H2H_CATEGORY = "h2h_category"
    H2H_POINTS = "h2h_points"
    ROTO = "roto"
    POINTS = "points"


@dataclass
class LeagueScoringSettings:
    """League-specific scoring configuration."""
    league_type: LeagueType
    scoring_weights: Dict[str, float] = field(default_factory=dict)
    categories: List[str] = field(default_factory=list)
    negative_categories: List[str] = field(default_factory=lambda: ['tov'])

    @classmethod
    def from_espn_settings(cls, espn_settings: Dict[str, Any]) -> 'LeagueScoringSettings':
        """
        Create scoring settings from ESPN league settings.

        Args:
            espn_settings: ESPN API league settings

        Returns:
            LeagueScoringSettings instance
        """
        # Determine league type
        scoring_type = espn_settings.get('scoring_type', 'H2H_CATEGORY')
        scoring_type_str = str(scoring_type).upper() if scoring_type else 'H2H_CATEGORY'
        if 'POINTS' in scoring_type_str:
            league_type = LeagueType.H2H_POINTS
        elif 'ROTO' in scoring_type_str:
            league_type = LeagueType.ROTO
        else:
            league_type = LeagueType.H2H_CATEGORY

        # Extract scoring weights
        scoring_weights = {}
        stat_categories = espn_settings.get('stat_categories', {})

        # ESPN stat ID mapping
        espn_stat_mapping = {
            '0': 'pts', '1': 'blk', '2': 'stl', '3': 'ast',
            '6': 'trb', '11': 'tov', '17': '3p',
            '19': 'fg_pct', '20': 'ft_pct',
            '13': 'fgm', '14': 'fga', '15': 'ftm', '16': 'fta'
        }

        for stat_id, stat_info in stat_categories.items():
            standard_stat = espn_stat_mapping.get(str(stat_id))
            if standard_stat:
                # Get point value or default to 1
                points = stat_info.get('points', 1.0) if isinstance(stat_info, dict) else 1.0
                scoring_weights[standard_stat] = float(points)

        # Default weights if not found
        if not scoring_weights:
            scoring_weights = DEFAULT_POINTS_SCORING.copy()

        # Determine categories
        categories = list(scoring_weights.keys())
        negative_cats = [cat for cat, val in scoring_weights.items() if val < 0]

        return cls(
            league_type=league_type,
            scoring_weights=scoring_weights,
            categories=categories,
            negative_categories=negative_cats
        )

    @classmethod
    def default_h2h_category(cls) -> 'LeagueScoringSettings':
        """Create default H2H category settings."""
        return cls(
            league_type=LeagueType.H2H_CATEGORY,
            scoring_weights=DEFAULT_CATEGORY_VALUES.copy(),
            categories=['pts', 'trb', 'ast', 'stl', 'blk', '3p', 'fg_pct', 'ft_pct', 'tov'],
            negative_categories=['tov']
        )

    @classmethod
    def default_points(cls) -> 'LeagueScoringSettings':
        """Create default points league settings."""
        return cls(
            league_type=LeagueType.H2H_POINTS,
            scoring_weights=DEFAULT_POINTS_SCORING.copy(),
            categories=list(DEFAULT_POINTS_SCORING.keys()),
            negative_categories=['tov']
        )


@dataclass
class HybridProjection:
    """Complete hybrid projection for a player."""
    # Player identification
    player_id: str
    player_name: str
    team: str
    position: str

    # Core projections
    projected_stats: Dict[str, float]
    ml_contribution: Dict[str, float]
    statistical_contribution: Dict[str, float]

    # Rest of season
    ros_totals: Dict[str, float]
    games_remaining: int
    games_projected: int

    # Fantasy value
    fantasy_points: float
    fantasy_value_rank: Optional[int] = None
    category_values: Dict[str, float] = field(default_factory=dict)

    # Confidence
    confidence_score: float = 0.0
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    # Weights used
    ml_weight: float = 0.5
    statistical_weight: float = 0.5
    season_phase: str = "mid"

    # Metadata
    projection_date: datetime = field(default_factory=datetime.now)
    league_id: Optional[int] = None

    # Injury data (passed through from ESPN)
    injury_details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'player_id': self.player_id,
            'player_name': self.player_name,
            'team': self.team,
            'position': self.position,
            'projected_stats': self.projected_stats,
            'ros_totals': self.ros_totals,
            'games_remaining': self.games_remaining,
            'games_projected': self.games_projected,
            'fantasy_points': round(self.fantasy_points, 2),
            'fantasy_value_rank': self.fantasy_value_rank,
            'category_values': {k: round(v, 2) for k, v in self.category_values.items()},
            'confidence_score': round(self.confidence_score, 2),
            'confidence_intervals': {
                k: {'low': round(v[0], 2), 'high': round(v[1], 2)}
                for k, v in self.confidence_intervals.items()
            },
            'ml_weight': round(self.ml_weight, 2),
            'statistical_weight': round(self.statistical_weight, 2),
            'season_phase': self.season_phase,
            'projection_date': self.projection_date.isoformat(),
            'league_id': self.league_id,
            'injury_details': self.injury_details,
        }


# =============================================================================
# Hybrid Projection Engine
# =============================================================================

class HybridProjectionEngine:
    """
    Combines ML and statistical projections with dynamic weighting.

    The engine automatically adjusts the balance between data sources
    based on how far into the NBA season we are, with statistical
    projections weighted more heavily early (when we have less
    current-season data) and ML weighted more heavily late (when
    we have more data to learn from).

    Falls back to simple projection when ML/statistical engines aren't available.
    """

    def __init__(
        self,
        ml_model: Optional['FantasyBasketballMLModel'] = None,
        statistical_engine: Optional['StatisticalProjectionEngine'] = None,
        models_dir: Optional[str] = None,
        use_simple_fallback: bool = True
    ):
        """
        Initialize the hybrid projection engine.

        Args:
            ml_model: Pre-loaded ML model instance
            statistical_engine: Pre-loaded statistical engine
            models_dir: Directory containing trained ML models
            use_simple_fallback: Use simple projection when other engines unavailable
        """
        self.ml_models_loaded = False
        self.use_simple_fallback = use_simple_fallback

        # Initialize ML model
        if ml_model:
            self.ml_model = ml_model
            self.ml_models_loaded = True
        elif ML_AVAILABLE and FantasyBasketballMLModel is not None:
            try:
                self.ml_model = FantasyBasketballMLModel(models_dir=models_dir)
                self._load_ml_models()
            except Exception as e:
                logger.warning(f"Could not initialize ML model: {e}")
                self.ml_model = None
        else:
            logger.info("ML model not available - using statistical/simple projections only")
            self.ml_model = None

        # Initialize statistical engine
        if statistical_engine:
            self.stat_engine = statistical_engine
        elif STAT_ENGINE_AVAILABLE and StatisticalProjectionEngine is not None:
            try:
                self.stat_engine = StatisticalProjectionEngine()
            except Exception as e:
                logger.warning(f"Could not initialize statistical engine: {e}")
                self.stat_engine = None
        else:
            logger.info("Statistical engine not available")
            self.stat_engine = None

        # Initialize simple fallback engine
        if use_simple_fallback and SIMPLE_ENGINE_AVAILABLE and SimpleProjectionEngine is not None:
            try:
                self.simple_engine = SimpleProjectionEngine()
                logger.info("Simple projection engine initialized as fallback")
            except Exception as e:
                logger.warning(f"Could not initialize simple engine: {e}")
                self.simple_engine = None
        else:
            self.simple_engine = None

        # Caches
        self._league_settings_cache: Dict[int, LeagueScoringSettings] = {}
        self._projection_cache: Dict[str, HybridProjection] = {}
        self._nba_schedule = None  # Lazy-loaded NBA schedule

        # Log initialization status
        logger.info(f"HybridProjectionEngine initialized: ML={self.ml_model is not None}, "
                   f"Stat={self.stat_engine is not None}, Simple={self.simple_engine is not None}")

    def _use_simple_fallback(
        self,
        player_id: str,
        player_data: Dict[str, Any],
        season_stats: Dict[str, float],
        games_played: int,
        injury_status: str,
        league_id: Optional[int] = None
    ) -> HybridProjection:
        """
        Use simple projection as fallback when other methods fail.

        Takes current season averages and projects rest-of-season totals.
        """
        if self.simple_engine is None:
            raise RuntimeError("Simple projection engine not available")

        player_name = player_data.get('name', f'Player {player_id}')
        team = player_data.get('team', 'UNK')
        position = player_data.get('position', 'N/A')

        logger.debug(f"Using simple fallback projection for {player_name}")

        # Get simple projection
        simple_proj = self.simple_engine.project_player(
            player_id=player_id,
            player_name=player_name,
            current_stats=season_stats,
            games_played=games_played,
            injury_status=injury_status,
        )

        # Convert to HybridProjection format
        return HybridProjection(
            player_id=player_id,
            player_name=player_name,
            team=team,
            position=position,
            projected_stats=simple_proj.projected_per_game,
            ml_contribution={},
            statistical_contribution=simple_proj.projected_per_game,
            ros_totals=simple_proj.ros_totals,
            games_remaining=simple_proj.games_remaining,
            games_projected=simple_proj.games_projected,
            fantasy_points=0.0,  # Not calculated in simple mode
            category_values={},
            confidence_score=simple_proj.confidence_score,
            confidence_intervals={},
            ml_weight=0.0,
            statistical_weight=1.0,
            season_phase='simple_fallback',
            league_id=league_id,
            injury_details=player_data.get('injury_details'),  # Pass through ESPN injury data
        )

    def _load_ml_models(self) -> None:
        """Load ML models from disk if available."""
        try:
            if self.ml_model.load_all_models():
                logger.info("ML models loaded successfully")
            else:
                logger.warning("No ML models found - ML projections will be unavailable")
        except Exception as e:
            logger.warning(f"Could not load ML models: {e}")

    # =========================================================================
    # Main Projection Interface
    # =========================================================================

    def project_player(
        self,
        player_id: str,
        league_id: Optional[int] = None,
        player_data: Optional[Dict[str, Any]] = None,
        season_stats: Optional[Dict[str, float]] = None,
        recent_stats: Optional[Dict[str, Dict[str, float]]] = None,
        espn_projection: Optional[Dict[str, float]] = None,
        bbref_projection: Optional[Dict[str, float]] = None,
        injury_status: str = 'ACTIVE',
        injury_notes: Optional[str] = None,
        expected_return_date: Optional[date] = None,
        league_settings: Optional[LeagueScoringSettings] = None,
        force_weights: Optional[Dict[str, float]] = None,
        league_season: Optional[int] = None,
        projection_method: str = 'adaptive',
        flat_game_rate: float = 0.85
    ) -> HybridProjection:
        """
        Generate a comprehensive hybrid projection for a player.

        Uses a tiered weighting system based on games played (3 sources: ESPN, Current, ML):
        - 0-5 games:   90% ESPN proj, 0% current, 10% ML
        - 6-15 games:  55% ESPN proj, 35% current, 10% ML
        - 16-35 games: 15% ESPN proj, 80% current, 5% ML
        - 35+ games:   100% current season stats

        Note: Previous season stats are not used because ESPN API doesn't provide them
        for the current season connection.

        Args:
            player_id: Unique player identifier
            league_id: Optional league ID for league-specific adjustments
            player_data: Player info (name, team, position, games_played)
            season_stats: Current season per-game averages
            recent_stats: Recent performance (last_7, last_15, last_30)
            espn_projection: ESPN's preseason/ROS projection
            bbref_projection: Basketball Reference projection
            injury_status: Current injury status
            injury_notes: Injury notes/comments (may contain return date info)
            expected_return_date: ESPN's expected return date for injured players
            league_settings: League scoring configuration
            force_weights: Override automatic weight calculation (legacy)
            league_season: The season year (e.g., 2026) for dynamic stat access
            projection_method: 'adaptive' (tiered rates) or 'flat_rate' (fixed rate)
            flat_game_rate: Fixed rate when projection_method='flat_rate' (0.50-1.00)

        Returns:
            HybridProjection with complete projection data
        """
        # Default player data
        if player_data is None:
            player_data = {
                'player_id': player_id,
                'name': f'Player {player_id}',
                'team': 'UNK',
                'position': 'N/A',
                'games_played': 0
            }

        player_name = player_data.get('name', f'Player {player_id}')
        team = player_data.get('team', 'UNK')
        nba_team = player_data.get('nba_team', team)  # Use nba_team if available, fallback to team
        position = player_data.get('position', 'N/A')
        games_played = player_data.get('games_played', 0)

        # Determine season for stat extraction
        if league_season:
            season = league_season
        else:
            # Default to current season based on date
            from datetime import date
            today = date.today()
            season = today.year if today.month >= 10 else today.year

        logger.debug(f"Using season {season} for projections")

        # =================================================================
        # DEBUG: Check player_data structure before extraction
        # =================================================================
        logger.info(f"=== project_player() called for: {player_name} ===")
        logger.info(f"[SETTINGS RECEIVED] projection_method={projection_method}, flat_game_rate={flat_game_rate}")
        logger.info(f"Player {player_name} - player_data type: {type(player_data).__name__}")
        logger.info(f"Player {player_name} - player_data keys: {list(player_data.keys()) if isinstance(player_data, dict) else 'NOT A DICT'}")
        logger.info(f"Player {player_name} - has 'stats' key: {'stats' in player_data if isinstance(player_data, dict) else False}")
        if 'stats' in player_data:
            stats_obj = player_data['stats']
            logger.info(f"Player {player_name} - stats type: {type(stats_obj).__name__}")
            if isinstance(stats_obj, dict):
                logger.info(f"Player {player_name} - stats keys: {list(stats_obj.keys())}")
                proj_key = f'{season}_projected'
                logger.info(f"Player {player_name} - has '{proj_key}': {proj_key in stats_obj}")
                if proj_key in stats_obj:
                    proj_obj = stats_obj[proj_key]
                    logger.info(f"Player {player_name} - {proj_key} type: {type(proj_obj).__name__}")
                    if isinstance(proj_obj, dict):
                        logger.info(f"Player {player_name} - {proj_key} keys: {list(proj_obj.keys())}")

        # Auto-extract ESPN projections if not provided
        if espn_projection is None and 'stats' in player_data:
            extracted_espn = self._extract_espn_projection(player_data, season, player_name)
            if extracted_espn:
                espn_projection = extracted_espn
                logger.info(f"[{player_name}] Auto-extracted ESPN projection: {len(espn_projection)} stat keys")
            else:
                logger.warning(f"[{player_name}] ESPN projection extraction returned None")

        # Auto-extract current season stats if not provided
        if season_stats is None and 'stats' in player_data:
            extracted_curr = self._extract_current_season_stats(player_data, season, player_name)
            if extracted_curr:
                season_stats = extracted_curr
                logger.info(f"[{player_name}] Auto-extracted current season: {len(season_stats)} stat keys")

        # Calculate tier for logging
        if games_played <= GAMES_TIER_1_MAX:
            tier = 1
            tier_desc = f"Tier 1 (0-{GAMES_TIER_1_MAX} games): 90% ESPN, 0% curr, 10% ML"
        elif games_played <= GAMES_TIER_2_MAX:
            tier = 2
            tier_desc = f"Tier 2 ({GAMES_TIER_1_MAX+1}-{GAMES_TIER_2_MAX} games): 55% ESPN, 35% curr, 10% ML"
        elif games_played <= GAMES_TIER_3_MAX:
            tier = 3
            tier_desc = f"Tier 3 ({GAMES_TIER_2_MAX+1}-{GAMES_TIER_3_MAX} games): 15% ESPN, 80% curr, 5% ML"
        else:
            tier = 4
            tier_desc = f"Tier 4 ({GAMES_TIER_3_MAX}+ games): 100% curr"

        logger.info(f"Extracted GP for {player_name}: {games_played} games -> {tier_desc}")

        # If no season stats and no ESPN projection, use simple fallback immediately
        if not season_stats and not espn_projection:
            if self.simple_engine and self.use_simple_fallback:
                logger.debug(f"No stats available for {player_name}, using simple fallback")
                return self._use_simple_fallback(
                    player_id=player_id,
                    player_data=player_data,
                    season_stats={},
                    games_played=games_played,
                    injury_status=injury_status,
                    league_id=league_id,
                )
            else:
                # Return empty projection
                return HybridProjection(
                    player_id=player_id,
                    player_name=player_name,
                    team=team,
                    position=position,
                    projected_stats={},
                    ml_contribution={},
                    statistical_contribution={},
                    ros_totals={},
                    games_remaining=0,
                    games_projected=0,
                    fantasy_points=0.0,
                    category_values={},
                    confidence_score=0.0,
                    confidence_intervals={},
                    ml_weight=0.0,
                    statistical_weight=0.0,
                    season_phase='no_data',
                    league_id=league_id,
                    injury_details=player_data.get('injury_details'),  # Pass through ESPN injury data
                )

        try:
            # Get tiered weights based on games played
            tiered_weights = self._get_tiered_weights(games_played)
            weight_tier = self._get_weight_tier_name(games_played)

            # Determine season phase (for metadata)
            season_phase = self._get_season_phase()

            # Get league settings
            if league_settings is None:
                if league_id and league_id in self._league_settings_cache:
                    league_settings = self._league_settings_cache[league_id]
                else:
                    league_settings = LeagueScoringSettings.default_h2h_category()

            # Generate ML projection (only if weight > 0 and ML available)
            ml_projection = None
            if tiered_weights['ml'] > 0 and self.ml_model is not None:
                ml_projection = self._get_ml_projection(
                    player_data=player_data,
                    season_stats=season_stats
                )

            # Combine projections using tiered system (3 sources: ESPN, Current, ML)
            combined_stats, contributions = self._combine_projections_tiered(
                espn_projection=espn_projection,
                current_season=season_stats,
                ml_projection=ml_projection,
                games_played=games_played,
                player_name=player_name
            )

            # For backward compatibility, extract ml and stat contributions
            ml_contrib = contributions.get('ml', {})
            stat_contrib = contributions.get('current_season', {})

            # Apply league-specific adjustments
            adjusted_stats = self._apply_league_adjustments(
                combined_stats, league_settings
            )

            # Calculate games projection with injury-aware logic
            games_remaining = self._estimate_games_remaining()

            # Check injury_details for out_for_season flag (from ESPN injuryDetails)
            # injury_details can be None for healthy players or dict for injured
            injury_details = player_data.get('injury_details')
            out_for_season = False

            if injury_details and isinstance(injury_details, dict):
                out_for_season = injury_details.get('out_for_season', False)
                logger.info(f"RECEIVED injury_details for {player_name}: {injury_details}")

            if out_for_season:
                # Player is out for the season - 0 games projected
                games_projected = 0
                logger.info(f"Player {player_name} OUT FOR SEASON - 0 games projected")
            else:
                # Check for expected_return_date from injury_details or player_data
                if expected_return_date is None and injury_details:
                    expected_return_date = injury_details.get('expected_return_date')
                if expected_return_date is None:
                    expected_return_date = player_data.get('expected_return_date')
                if expected_return_date is None:
                    expected_return_date = player_data.get('injury_date')

                # Get player_schedule from player_data if available
                player_schedule = player_data.get('player_schedule')

                games_projected = self._estimate_player_games(
                    games_played, games_remaining, injury_status, injury_notes,
                    expected_return_date=expected_return_date,
                    player_name=player_name,
                    nba_team=nba_team,
                    player_schedule=player_schedule,
                    projection_method=projection_method,
                    flat_game_rate=flat_game_rate
                )

            # Log the final projected_games value
            logger.info(f"[HYBRID OUTPUT] {player_name}: projected_games={games_projected}")

            # Calculate ROS totals
            ros_totals = self._calculate_ros_totals(adjusted_stats, games_projected)

            # Calculate fantasy value
            fantasy_points = self._calculate_fantasy_points(
                adjusted_stats, league_settings
            )
            category_values = self._calculate_category_values(
                adjusted_stats, league_settings
            )

            # Calculate confidence score based on data availability and games played
            # More sources and more games = higher confidence (3 sources max: ESPN, Curr, ML)
            sources_available = sum([
                1 if espn_projection else 0,
                1 if season_stats else 0,
                1 if ml_projection else 0
            ])
            confidence_score = self._calculate_confidence_score_tiered(
                games_played=games_played,
                sources_available=sources_available,
                injury_status=injury_status
            )

            # Build confidence intervals
            confidence_intervals = self._build_confidence_intervals(
                combined_stats,
                {},  # No stat projection intervals in tiered system
                confidence_score
            )

            # Create hybrid projection
            projection = HybridProjection(
                player_id=player_id,
                player_name=player_name,
                team=team,
                position=position,
                projected_stats=adjusted_stats,
                ml_contribution=ml_contrib,
                statistical_contribution=stat_contrib,
                ros_totals=ros_totals,
                games_remaining=games_remaining,
                games_projected=games_projected,
                fantasy_points=fantasy_points,
                category_values=category_values,
                confidence_score=confidence_score,
                confidence_intervals=confidence_intervals,
                ml_weight=tiered_weights['ml'],
                statistical_weight=tiered_weights['current_season'],
                season_phase=weight_tier,  # Show which tier was used
                league_id=league_id,
                injury_details=injury_details,  # Pass through ESPN injury data
            )

            # Cache the projection
            cache_key = f"{player_id}_{league_id}"
            self._projection_cache[cache_key] = projection

            return projection

        except Exception as e:
            logger.warning(f"Hybrid projection failed for {player_name}: {e}")

            # Try simple fallback
            if self.simple_engine and self.use_simple_fallback and season_stats:
                logger.info(f"Using simple fallback for {player_name}")
                return self._use_simple_fallback(
                    player_id=player_id,
                    player_data=player_data,
                    season_stats=season_stats,
                    games_played=games_played,
                    injury_status=injury_status,
                    league_id=league_id,
                )
            else:
                # Re-raise if no fallback available
                raise

    def project_roster(
        self,
        players: List[Dict[str, Any]],
        league_id: int,
        season_stats_map: Dict[str, Dict[str, float]],
        league_settings: Optional[LeagueScoringSettings] = None,
        espn_projections: Optional[Dict[str, Dict[str, float]]] = None,
        bbref_projections: Optional[Dict[str, Dict[str, float]]] = None,
        injury_statuses: Optional[Dict[str, str]] = None,
        league_season: Optional[int] = None
    ) -> List[HybridProjection]:
        """
        Generate projections for an entire roster.

        Args:
            players: List of player data dictionaries
            league_id: League identifier
            season_stats_map: Map of player_id -> season stats
            league_settings: League scoring configuration
            espn_projections: Map of player_id -> ESPN projection
            bbref_projections: Map of player_id -> BBRef projection
            injury_statuses: Map of player_id -> injury status

        Returns:
            List of HybridProjection objects, sorted by fantasy value
        """
        projections = []

        for player in players:
            player_id = player.get('player_id', '')

            try:
                projection = self.project_player(
                    player_id=player_id,
                    league_id=league_id,
                    player_data=player,
                    season_stats=season_stats_map.get(player_id),
                    espn_projection=espn_projections.get(player_id) if espn_projections else None,
                    bbref_projection=bbref_projections.get(player_id) if bbref_projections else None,
                    injury_status=injury_statuses.get(player_id, 'ACTIVE') if injury_statuses else 'ACTIVE',
                    league_settings=league_settings,
                    league_season=league_season
                )
                projections.append(projection)
            except Exception as e:
                logger.warning(f"Failed to project {player.get('name', player_id)}: {e}")

        # Sort by fantasy points descending
        projections.sort(key=lambda p: p.fantasy_points, reverse=True)

        # Assign ranks
        for i, proj in enumerate(projections, 1):
            proj.fantasy_value_rank = i

        return projections

    def set_league_settings(
        self,
        league_id: int,
        settings: LeagueScoringSettings
    ) -> None:
        """
        Cache league settings for a specific league.

        Args:
            league_id: League identifier
            settings: League scoring settings
        """
        self._league_settings_cache[league_id] = settings
        logger.debug(f"Cached settings for league {league_id}")

    # =========================================================================
    # Data Extraction from ESPN Player Stats
    # =========================================================================

    def _extract_espn_projection(
        self,
        player_data: Dict[str, Any],
        season: int,
        player_name: str = "Unknown"
    ) -> Optional[Dict[str, float]]:
        """
        Extract ESPN projection from player data.

        ESPN projections are at: player.stats['{season}_projected']['avg']
        Stats use STRING KEYS: 'PTS', 'REB', 'AST', 'STL', 'BLK', '3PM', 'FGM', 'FGA', 'FTM', 'FTA'

        Args:
            player_data: Player data dict (may contain 'stats' sub-dict)
            season: Current season year
            player_name: Player name for logging

        Returns:
            Dictionary of projected per-game stats (10 keys) or None
        """
        # =================================================================
        # DEBUG LOGGING: Check player object structure
        # =================================================================
        logger.info(f"=== ESPN PROJECTION EXTRACTION: {player_name} ===")
        logger.info(f"Player {player_name} - player_data type: {type(player_data).__name__}")
        logger.info(f"Player {player_name} - player_data keys: {list(player_data.keys()) if isinstance(player_data, dict) else 'NOT A DICT'}")

        # Check if player_data has 'stats' key or is the stats dict itself
        stats_dict = player_data.get('stats', {})
        logger.info(f"Player {player_name} stats keys: {list(stats_dict.keys()) if stats_dict else 'EMPTY'}")

        # Also check if stats might be directly on player_data (structure issue)
        if not stats_dict and isinstance(player_data, dict):
            # Maybe stats ARE player_data directly?
            potential_keys = [k for k in player_data.keys() if 'projected' in str(k).lower() or 'total' in str(k).lower()]
            if potential_keys:
                logger.info(f"Player {player_name} - FOUND projection keys directly in player_data: {potential_keys}")
                stats_dict = player_data  # Use player_data directly

        if not stats_dict:
            logger.info(f"[{player_name}] ESPN projections not available - no stats dict")
            return None

        # Primary key: '{season}_projected' (e.g., '2026_projected')
        proj_key = f'{season}_projected'
        logger.info(f"Has {proj_key}: {proj_key in stats_dict}")

        if proj_key not in stats_dict:
            # Log available keys for debugging
            available_keys = list(stats_dict.keys())
            logger.info(f"[{player_name}] ESPN projections not available - '{proj_key}' not in stats (available: {available_keys})")
            return None

        proj_data = stats_dict[proj_key]
        logger.info(f"{proj_key} structure: {list(proj_data.keys()) if isinstance(proj_data, dict) else type(proj_data).__name__}")

        if not isinstance(proj_data, dict) or 'avg' not in proj_data:
            logger.info(f"[{player_name}] ESPN projections not available - no 'avg' in {proj_key}")
            return None

        avg_stats = proj_data['avg']
        logger.info(f"Player {player_name} avg_stats keys: {list(avg_stats.keys()) if isinstance(avg_stats, dict) else 'NOT A DICT'}")

        if not avg_stats or not isinstance(avg_stats, dict):
            logger.info(f"[{player_name}] ESPN projections not available - avg is empty or not dict")
            return None

        # Extract the 10 component stats using STRING KEYS
        # ESPN uses uppercase keys like 'PTS', 'REB', 'AST', etc.
        result = {}

        # Map of our standard keys to possible ESPN keys
        stat_mappings = {
            'pts': ['PTS', 'pts', 'Points', 'points'],
            'reb': ['REB', 'reb', 'Rebounds', 'rebounds', 'TRB', 'trb'],
            'ast': ['AST', 'ast', 'Assists', 'assists'],
            'stl': ['STL', 'stl', 'Steals', 'steals'],
            'blk': ['BLK', 'blk', 'Blocks', 'blocks'],
            '3p': ['3PM', '3pm', '3P', '3p', 'ThreePointersMade', 'threePointersMade'],
            'fgm': ['FGM', 'fgm', 'FieldGoalsMade', 'fieldGoalsMade'],
            'fga': ['FGA', 'fga', 'FieldGoalsAttempted', 'fieldGoalsAttempted'],
            'ftm': ['FTM', 'ftm', 'FreeThrowsMade', 'freeThrowsMade'],
            'fta': ['FTA', 'fta', 'FreeThrowsAttempted', 'freeThrowsAttempted'],
            'to': ['TO', 'to', 'TOV', 'tov', 'Turnovers', 'turnovers'],
            'fg_pct': ['FG%', 'fg%', 'FG_PCT', 'fg_pct', 'FieldGoalPercentage'],
            'ft_pct': ['FT%', 'ft%', 'FT_PCT', 'ft_pct', 'FreeThrowPercentage'],
        }

        for standard_key, possible_keys in stat_mappings.items():
            found = False
            for espn_key in possible_keys:
                if espn_key in avg_stats:
                    value = avg_stats[espn_key]
                    if isinstance(value, (int, float)):
                        # Convert percentages if needed (>1 means 0-100 scale)
                        if 'pct' in standard_key and value > 1:
                            value = value / 100.0
                        result[standard_key] = float(value)
                        logger.debug(f"[{player_name}] MATCHED {standard_key} <- '{espn_key}' = {value}")
                        found = True
                        break
            if not found:
                logger.debug(f"[{player_name}] NO MATCH for {standard_key} (tried: {possible_keys})")

        # Log what we found
        found_keys = list(result.keys())
        if result:
            sample_stats = {k: f"{v:.1f}" for k, v in list(result.items())[:5]}
            logger.info(f"[{player_name}] ESPN projections found: {len(result)} keys {found_keys}")
            logger.info(f"[{player_name}] ESPN sample: {sample_stats}")
        else:
            # Log the raw keys to help debug
            raw_keys = list(avg_stats.keys())
            logger.warning(f"[{player_name}] ESPN projection extraction failed - raw keys: {raw_keys}")

        return result if result else None

    def _extract_current_season_stats(
        self,
        player_data: Dict[str, Any],
        season: int,
        player_name: str = "Unknown"
    ) -> Optional[Dict[str, float]]:
        """
        Extract current season stats from player data.

        Current season stats are at: player.stats['{season}_total']['avg']
        Stats use STRING KEYS: 'PTS', 'REB', 'AST', 'STL', 'BLK', '3PM', 'FGM', 'FGA', 'FTM', 'FTA'

        Args:
            player_data: Player data dict (may contain 'stats' sub-dict)
            season: Current season year
            player_name: Player name for logging

        Returns:
            Dictionary of current season per-game stats or None
        """
        stats_dict = player_data.get('stats', {})
        if not stats_dict:
            logger.debug(f"[{player_name}] Current season not available - no stats dict")
            return None

        # Primary key: '{season}_total' (e.g., '2026_total')
        total_key = f'{season}_total'

        if total_key not in stats_dict:
            logger.debug(f"[{player_name}] Current season not available - '{total_key}' not in stats")
            return None

        total_data = stats_dict[total_key]

        if not isinstance(total_data, dict) or 'avg' not in total_data:
            logger.debug(f"[{player_name}] Current season not available - no 'avg' in {total_key}")
            return None

        avg_stats = total_data['avg']

        if not avg_stats or not isinstance(avg_stats, dict):
            logger.debug(f"[{player_name}] Current season not available - avg is empty or not dict")
            return None

        # Extract the 10 component stats using STRING KEYS
        result = {}

        # Map of our standard keys to possible ESPN keys
        stat_mappings = {
            'pts': ['PTS', 'pts', 'Points', 'points'],
            'reb': ['REB', 'reb', 'Rebounds', 'rebounds', 'TRB', 'trb'],
            'ast': ['AST', 'ast', 'Assists', 'assists'],
            'stl': ['STL', 'stl', 'Steals', 'steals'],
            'blk': ['BLK', 'blk', 'Blocks', 'blocks'],
            '3p': ['3PM', '3pm', '3P', '3p', 'ThreePointersMade', 'threePointersMade'],
            'fgm': ['FGM', 'fgm', 'FieldGoalsMade', 'fieldGoalsMade'],
            'fga': ['FGA', 'fga', 'FieldGoalsAttempted', 'fieldGoalsAttempted'],
            'ftm': ['FTM', 'ftm', 'FreeThrowsMade', 'freeThrowsMade'],
            'fta': ['FTA', 'fta', 'FreeThrowsAttempted', 'freeThrowsAttempted'],
            'to': ['TO', 'to', 'TOV', 'tov', 'Turnovers', 'turnovers'],
            'fg_pct': ['FG%', 'fg%', 'FG_PCT', 'fg_pct', 'FieldGoalPercentage'],
            'ft_pct': ['FT%', 'ft%', 'FT_PCT', 'ft_pct', 'FreeThrowPercentage'],
        }

        for standard_key, possible_keys in stat_mappings.items():
            for espn_key in possible_keys:
                if espn_key in avg_stats:
                    value = avg_stats[espn_key]
                    if isinstance(value, (int, float)):
                        # Convert percentages if needed (>1 means 0-100 scale)
                        if 'pct' in standard_key and value > 1:
                            value = value / 100.0
                        result[standard_key] = float(value)
                        break

        # Log what we found
        if result:
            found_keys = list(result.keys())
            sample_stats = {k: f"{v:.1f}" for k, v in list(result.items())[:5]}
            logger.info(f"[{player_name}] Current season found: {len(result)} keys {found_keys}")
            logger.debug(f"[{player_name}] Current season sample: {sample_stats}")
        else:
            raw_keys = list(avg_stats.keys())
            logger.debug(f"[{player_name}] Current season extraction failed - raw keys: {raw_keys}")

        return result if result else None

    def _normalize_stat_keys(
        self,
        stats: Dict[str, Any],
        player_name: str = "Unknown",
        source: str = ""
    ) -> Dict[str, float]:
        """
        Normalize stat keys from various formats to standard keys.

        Handles:
        - ESPN stat IDs (0=pts, 6=reb, 3=ast, etc.)
        - Various key formats (PTS, pts, Points, etc.)
        - Filters out non-numeric values

        Args:
            stats: Raw stats dictionary
            player_name: Player name for logging
            source: Source name for logging

        Returns:
            Normalized stats dictionary
        """
        # ESPN stat ID mapping
        ESPN_STAT_ID_MAP = {
            '0': 'pts', '1': 'blk', '2': 'stl', '3': 'ast',
            '4': 'oreb', '5': 'dreb', '6': 'reb',
            '9': 'pf', '11': 'to', '13': 'fgm', '14': 'fga',
            '15': 'ftm', '16': 'fta', '17': '3pm', '18': '3pa',
            '19': 'fg_pct', '20': 'ft_pct', '21': '3p_pct',
            '40': 'min', '41': 'gp',
        }

        # Key normalization mapping
        KEY_NORMALIZE = {
            'points': 'pts', 'rebounds': 'reb', 'assists': 'ast',
            'steals': 'stl', 'blocks': 'blk', 'turnovers': 'to',
            '3pm': '3p', 'threes': '3p', 'three_pointers': '3p',
            'trb': 'reb', 'tov': 'to',
            'fg%': 'fg_pct', 'ft%': 'ft_pct', '3p%': '3p_pct',
            'fgpct': 'fg_pct', 'ftpct': 'ft_pct',
        }

        normalized = {}
        for key, value in stats.items():
            # Skip non-numeric values
            if not isinstance(value, (int, float)):
                continue

            key_str = str(key).lower().strip()

            # Check if it's an ESPN stat ID
            if key_str in ESPN_STAT_ID_MAP:
                norm_key = ESPN_STAT_ID_MAP[key_str]
            elif key_str in KEY_NORMALIZE:
                norm_key = KEY_NORMALIZE[key_str]
            else:
                norm_key = key_str

            # Convert percentage values if needed (>1 means it's 0-100 scale)
            if 'pct' in norm_key and value > 1:
                value = value / 100.0

            normalized[norm_key] = float(value)

        if normalized:
            core_keys = [k for k in ['pts', 'reb', 'ast', 'stl', 'blk', '3p'] if k in normalized]
            logger.debug(f"[{player_name}] Normalized {source}: {len(normalized)} keys, core: {core_keys}")

        return normalized

    # =========================================================================
    # Projection Components
    # =========================================================================

    def _get_statistical_projection(
        self,
        player_data: Dict[str, Any],
        season_stats: Optional[Dict[str, float]],
        recent_stats: Optional[Dict[str, Dict[str, float]]],
        espn_projection: Optional[Dict[str, float]],
        bbref_projection: Optional[Dict[str, float]],
        injury_status: str
    ) -> Optional[StatisticalProjection]:
        """Get projection from statistical engine."""
        try:
            return self.stat_engine.project_player(
                player_data=player_data,
                season_stats=season_stats,
                recent_stats=recent_stats,
                espn_projection=espn_projection,
                bbref_projection=bbref_projection,
                injury_status=injury_status
            )
        except Exception as e:
            logger.warning(f"Statistical projection failed: {e}")
            return None

    def _get_ml_projection(
        self,
        player_data: Dict[str, Any],
        season_stats: Optional[Dict[str, float]]
    ) -> Dict[str, float]:
        """Get projection from ML models."""
        if not season_stats:
            return {}

        # Check if ML models are loaded
        if not self.ml_model.counting_models and not self.ml_model.shooting_models:
            return {}

        try:
            # Build feature DataFrame for prediction
            features = self._build_ml_features(player_data, season_stats)

            if features.empty:
                return {}

            # Get predictions
            predictions = self.ml_model.predict_player_stats(features)

            # Convert to single values (predictions are arrays)
            return {
                stat: float(values[0]) if len(values) > 0 else 0.0
                for stat, values in predictions.items()
            }
        except Exception as e:
            logger.warning(f"ML projection failed: {e}")
            return {}

    def _build_ml_features(
        self,
        player_data: Dict[str, Any],
        season_stats: Dict[str, float]
    ) -> pd.DataFrame:
        """
        Build feature DataFrame for ML prediction.

        Args:
            player_data: Player information
            season_stats: Current season stats

        Returns:
            DataFrame with features for ML models
        """
        features = {}

        # Demographics
        features['age'] = player_data.get('age', 25)

        # Playing time
        features['mp'] = season_stats.get('mp', season_stats.get('minutes', 25))
        features['g'] = player_data.get('games_played', 40)
        features['gs_pct'] = season_stats.get('gs_pct', 0.5)

        # Stats from season averages
        for stat in ['pts', 'trb', 'ast', 'stl', 'blk', 'tov', '3p',
                     'fg_pct', 'ft_pct', 'fga', 'fta']:
            features[stat] = season_stats.get(stat, 0)

        # Advanced stats
        features['usg_pct'] = season_stats.get('usg_pct', 0.20)
        features['ts_pct'] = season_stats.get('ts_pct', 0.55)
        features['per'] = season_stats.get('per', 15.0)
        features['bpm'] = season_stats.get('bpm', 0.0)
        features['vorp'] = season_stats.get('vorp', 0.0)
        features['ws'] = season_stats.get('ws', 0.0)
        features['efg_pct'] = season_stats.get('efg_pct', features.get('fg_pct', 0.45))

        # Shooting profile
        fga = features.get('fga', 1)
        features['3p_rate'] = season_stats.get('3pa', 0) / max(fga, 1)
        features['ft_rate'] = season_stats.get('fta', 0) / max(fga, 1)

        # Position encoding
        # Handle position as int (ESPN position ID) or string
        raw_position = player_data.get('position', '')
        if isinstance(raw_position, int):
            # ESPN position IDs: 1=PG, 2=SG, 3=SF, 4=PF, 5=C
            ESPN_POSITION_MAP = {
                0: 'PG', 1: 'PG', 2: 'SG', 3: 'SF', 4: 'PF', 5: 'C',
                6: 'G', 7: 'F', 8: 'UTIL', 9: 'G/F', 10: 'F/C', 11: 'UTIL'
            }
            position = ESPN_POSITION_MAP.get(raw_position, 'UTIL')
        else:
            position = str(raw_position).upper() if raw_position else ''
        features['is_guard'] = 1 if 'G' in position or position in ['PG', 'SG'] else 0
        features['is_forward'] = 1 if 'F' in position or position in ['SF', 'PF'] else 0
        features['is_center'] = 1 if 'C' in position else 0

        # Per-36 estimates
        mp = features.get('mp', 25)
        if mp > 0:
            for stat in ['pts', 'trb', 'ast', 'stl', 'blk']:
                features[f'{stat}_per36'] = (features.get(stat, 0) / mp) * 36

        # Previous season (use current as proxy if not available)
        for stat in ['pts', 'trb', 'ast', 'mp', 'g']:
            features[f'{stat}_prev_season'] = features.get(stat, 0) * 0.95

        return pd.DataFrame([features])

    def _combine_projections_tiered(
        self,
        espn_projection: Optional[Dict[str, float]],
        current_season: Optional[Dict[str, float]],
        ml_projection: Optional[Dict[str, float]],
        games_played: int,
        player_name: str = "Unknown"
    ) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
        """
        Combine projections from 3 sources using tiered weights based on games played.

        Tiered system (3 sources: ESPN, Current, ML):
        - 0-5 games:   90% ESPN proj, 0% current, 10% ML
        - 6-15 games:  55% ESPN proj, 35% current, 10% ML
        - 16-35 games: 15% ESPN proj, 80% current, 5% ML
        - 35+ games:   100% current season stats

        Note: Previous season stats are not used because ESPN API doesn't provide them
        for the current season connection.

        FT% uses regression-to-mean instead of the tiered system.

        Args:
            espn_projection: ESPN's preseason/ROS projection
            current_season: Player's current season stats
            ml_projection: ML model projection
            games_played: Number of games player has played this season
            player_name: Player name for logging

        Returns:
            Tuple of (combined_stats, contributions_by_source)
        """
        combined = {}
        contributions = {
            'espn_projection': {},
            'current_season': {},
            'ml': {}
        }

        # Get tiered weights based on games played
        weights = self._get_tiered_weights(games_played)
        tier_name = self._get_weight_tier_name(games_played)

        # Core fantasy categories to always try to project
        CORE_CATEGORIES = ['pts', 'reb', 'trb', 'ast', 'stl', 'blk', '3pm', '3p', 'to', 'tov']
        PERCENTAGE_CATEGORIES = ['fg_pct', 'ft_pct', '3p_pct']
        COMPONENT_STATS = ['fgm', 'fga', 'ftm', 'fta', '3pa']

        # Stats that should be clamped to [0, 1]
        percentage_stats = {'fg_pct', 'ft_pct', '3p_pct', 'efg_pct', 'ts_pct'}

        # League average FT% for regression-to-mean
        LEAGUE_AVG_FT_PCT = 0.78
        FT_PRIOR_WEIGHT = 100

        # Ensure all sources are dicts (not None)
        espn_projection = espn_projection or {}
        current_season = current_season or {}
        ml_projection = ml_projection or {}

        # Log tier and source availability (3 sources: ESPN, Curr, ML)
        logger.info(f"[HYBRID] {player_name}: Tier={tier_name}, Games={games_played}")
        logger.info(f"[HYBRID] {player_name}: Weights: ESPN={weights['espn_projection']:.0%}, Curr={weights['current_season']:.0%}, ML={weights['ml']:.0%}")

        # Detailed source availability logging
        logger.info(f"[HYBRID] {player_name}: Sources: ESPN={len(espn_projection)} keys, Curr={len(current_season)} keys, ML={len(ml_projection)} keys")

        # Log sample values from each source
        if espn_projection and len(espn_projection) > 0:
            sample_espn = {k: f"{v:.1f}" for k, v in list(espn_projection.items())[:3]}
            logger.info(f"[HYBRID] {player_name}: ESPN sample: {sample_espn}")
        elif weights['espn_projection'] > 0:
            logger.warning(f"[HYBRID] {player_name}: ESPN projection MISSING but weight is {weights['espn_projection']:.0%}")

        # Log actual current season stats for debugging
        if current_season:
            core_stats = {k: v for k, v in current_season.items() if k in ['pts', 'trb', 'reb', 'ast', 'stl', 'blk', '3p', '3pm', 'fgm', 'fga', 'ftm', 'fta']}
            logger.info(f"[HYBRID] {player_name}: Current season stats: {core_stats}")
        else:
            logger.warning(f"[HYBRID] {player_name}: NO CURRENT SEASON STATS AVAILABLE!")

        # Get all stats from all 3 sources, prioritizing core categories
        all_stats = (
            set(espn_projection.keys()) |
            set(current_season.keys()) |
            set(ml_projection.keys()) |
            set(CORE_CATEGORIES) |
            set(PERCENTAGE_CATEGORIES) |
            set(COMPONENT_STATS)
        )

        for stat in all_stats:
            # Special handling for FT%: use regression-to-mean
            if stat == 'ft_pct':
                # Prefer current season FT%, then ESPN
                ft_pct_value = (
                    current_season.get('ft_pct') or
                    espn_projection.get('ft_pct')
                )

                if ft_pct_value is not None:
                    # Ensure it's a decimal (0-1), not percentage (0-100)
                    if ft_pct_value > 1:
                        ft_pct_value = ft_pct_value / 100.0

                    # Get FTA for sample size weighting
                    fta = 0
                    if current_season:
                        fta = current_season.get('fta', 0) or 0
                        games = current_season.get('g', games_played) or games_played
                        if games > 0 and fta < 10:  # Looks like per-game
                            fta = fta * games

                    # Regression to mean
                    if fta > 0:
                        value = (ft_pct_value * fta + LEAGUE_AVG_FT_PCT * FT_PRIOR_WEIGHT) / (fta + FT_PRIOR_WEIGHT)
                    else:
                        value = ft_pct_value  # Use raw value if no FTA data

                    value = max(0.0, min(1.0, value))
                    combined[stat] = value
                    contributions['current_season'][stat] = value
                continue

            # Get values from each source, with alias handling
            # Handle stat key aliases (reb/trb, to/tov, 3pm/3p)
            stat_aliases = {
                'reb': 'trb', 'trb': 'reb',
                'to': 'tov', 'tov': 'to',
                '3pm': '3p', '3p': '3pm'
            }

            def get_stat_value(source_dict, stat_key):
                """Get stat value, trying aliases if needed."""
                val = source_dict.get(stat_key)
                if val is None and stat_key in stat_aliases:
                    val = source_dict.get(stat_aliases[stat_key])
                return val

            sources = {
                'espn_projection': get_stat_value(espn_projection, stat),
                'current_season': get_stat_value(current_season, stat),
                'ml': get_stat_value(ml_projection, stat)
            }

            # Calculate weighted average using only available sources
            total_weight = 0.0
            weighted_sum = 0.0
            source_details = []

            for source_name, source_value in sources.items():
                if source_value is not None:
                    # Ensure percentage stats are in decimal form
                    if stat in percentage_stats and source_value > 1:
                        source_value = source_value / 100.0

                    weight = weights[source_name]
                    if weight > 0:
                        weighted_sum += source_value * weight
                        total_weight += weight
                        contributions[source_name][stat] = source_value * weight
                        source_details.append(f"{source_name[:4]}:{source_value:.2f}*{weight:.0%}")

            if total_weight > 0:
                value = weighted_sum / total_weight

                # Clamp percentages to [0, 1] range
                if stat in percentage_stats:
                    value = max(0.0, min(1.0, value))
                # Counting stats should be >= 0
                elif value < 0:
                    value = 0.0

                combined[stat] = value

                # Log core stat calculations
                if stat in CORE_CATEGORIES or stat in PERCENTAGE_CATEGORIES:
                    logger.debug(f"[{player_name}] {stat}: {value:.3f} <- {', '.join(source_details)}")

        # Log final projection summary for core stats
        core_summary = {k: f"{v:.2f}" for k, v in combined.items()
                       if k in ['pts', 'reb', 'trb', 'ast', 'stl', 'blk', '3pm', '3p', 'fg_pct', 'ft_pct', 'fgm', 'fga', 'ftm', 'fta']}
        logger.info(f"[HYBRID] {player_name}: Combined per-game projection: {core_summary}")

        # Warn if all zeros
        counting_stats = sum(combined.get(k, 0) for k in ['pts', 'trb', 'reb', 'ast', 'stl', 'blk', '3pm', '3p'])
        if counting_stats == 0 and len(combined) > 0:
            logger.warning(f"[HYBRID] {player_name}: ALL COUNTING STATS ARE ZERO! Check stat extraction.")
        elif len(combined) == 0:
            logger.warning(f"[HYBRID] {player_name}: COMBINED STATS DICT IS EMPTY!")

        return combined, contributions

    def _combine_projections(
        self,
        stat_projection: Dict[str, float],
        ml_projection: Dict[str, float],
        ml_weight: float,
        stat_weight: float,
        season_stats: Optional[Dict[str, float]] = None
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        """
        DEPRECATED: Use _combine_projections_tiered instead.

        Legacy method for backward compatibility.
        """
        # Convert to tiered format using current season as primary
        combined, contributions = self._combine_projections_tiered(
            espn_projection=None,
            current_season=stat_projection,
            ml_projection=ml_projection,
            games_played=35,  # Use tier_4 weights (100% current)
            player_name="Legacy"
        )

        # Convert contributions to legacy format
        ml_contrib = contributions.get('ml', {})
        stat_contrib = contributions.get('current_season', {})

        return combined, ml_contrib, stat_contrib

    # =========================================================================
    # League Adjustments
    # =========================================================================

    def _apply_league_adjustments(
        self,
        stats: Dict[str, float],
        league_settings: LeagueScoringSettings
    ) -> Dict[str, float]:
        """
        Apply league-specific adjustments to projections.

        For category leagues, adjustments are minimal (raw stats matter).
        For points leagues, we may adjust based on scoring weights.

        Args:
            stats: Raw projected stats
            league_settings: League configuration

        Returns:
            Adjusted stats dictionary
        """
        adjusted = stats.copy()

        # For points leagues, no stat adjustment needed
        # (fantasy points calculation handles scoring)

        # For category leagues, ensure all tracked categories are present
        if league_settings.league_type in [LeagueType.H2H_CATEGORY, LeagueType.ROTO]:
            for cat in league_settings.categories:
                if cat not in adjusted:
                    adjusted[cat] = 0.0

        return adjusted

    def _calculate_fantasy_points(
        self,
        stats: Dict[str, float],
        league_settings: LeagueScoringSettings
    ) -> float:
        """
        Calculate total fantasy points based on league scoring.

        Args:
            stats: Projected per-game stats
            league_settings: League scoring configuration

        Returns:
            Fantasy points per game
        """
        total = 0.0

        for stat, value in stats.items():
            weight = league_settings.scoring_weights.get(stat, 0)
            total += value * weight

        return total

    def _calculate_category_values(
        self,
        stats: Dict[str, float],
        league_settings: LeagueScoringSettings
    ) -> Dict[str, float]:
        """
        Calculate value contribution by category.

        For category leagues, this shows relative strength in each category.
        For points leagues, this shows point contribution by stat.

        Args:
            stats: Projected stats
            league_settings: League configuration

        Returns:
            Dictionary of category -> value
        """
        values = {}

        for stat in league_settings.categories:
            if stat in stats:
                weight = league_settings.scoring_weights.get(stat, 1.0)
                values[stat] = stats[stat] * abs(weight)

        return values

    # =========================================================================
    # Confidence Calculation
    # =========================================================================

    def _calculate_confidence_score(
        self,
        stat_projection: Optional[StatisticalProjection],
        ml_available: bool,
        games_played: int,
        injury_status: str
    ) -> float:
        """
        Calculate overall confidence in the projection.

        Factors:
        - Number of data sources available
        - Games played (more games = more confidence)
        - Injury status (healthy = more confidence)
        - Agreement between sources

        Args:
            stat_projection: Statistical projection (contains source info)
            ml_available: Whether ML projection was available
            games_played: Number of games played this season
            injury_status: Current injury status

        Returns:
            Confidence score from 0 to 100
        """
        confidence = 50.0  # Base confidence

        # Source availability bonus
        sources = 0
        if stat_projection and stat_projection.sources_used:
            sources = len(stat_projection.sources_used)
        if ml_available:
            sources += 1

        # More sources = more confidence (up to +20)
        confidence += min(sources * 5, 20)

        # Games played bonus (up to +20)
        games_factor = min(games_played / 50, 1.0)  # Max out at 50 games
        confidence += games_factor * 20

        # Injury penalty
        injury_penalties = {
            'ACTIVE': 0,
            'PROBABLE': -2,
            'QUESTIONABLE': -10,
            'DOUBTFUL': -20,
            'DAY_TO_DAY': -5,
            'GTD': -5,
            'OUT': -30,
            'INJ_RESERVE': -30,
        }
        injury_status_str = str(injury_status).upper() if injury_status else 'ACTIVE'
        confidence += injury_penalties.get(injury_status_str, -10)

        # Season phase adjustment
        phase = self._get_season_phase()
        if phase == SeasonPhase.EARLY:
            confidence -= 10  # Less confident early
        elif phase == SeasonPhase.LATE:
            confidence += 5   # More confident late

        return max(0, min(100, confidence))

    def _calculate_confidence_score_tiered(
        self,
        games_played: int,
        sources_available: int,
        injury_status: str
    ) -> float:
        """
        Calculate confidence score for tiered projection system.

        Confidence increases with:
        - More games played (larger sample = more reliable)
        - More data sources available
        - Healthy injury status

        Args:
            games_played: Number of games played this season
            sources_available: Number of data sources (0-4)
            injury_status: Current injury status

        Returns:
            Confidence score from 0 to 100
        """
        # Base confidence depends on games played tier
        if games_played <= GAMES_TIER_1_MAX:
            # 0-5 games: low confidence, relying mostly on projections
            base_confidence = 40.0
        elif games_played <= GAMES_TIER_2_MAX:
            # 6-15 games: moderate confidence
            base_confidence = 55.0
        elif games_played <= GAMES_TIER_3_MAX:
            # 16-35 games: good confidence
            base_confidence = 70.0
        else:
            # 35+ games: high confidence
            base_confidence = 85.0

        confidence = base_confidence

        # Source availability bonus (up to +10)
        confidence += min(sources_available * 2.5, 10)

        # Games played fine-tuning within tier (up to +5)
        if games_played <= GAMES_TIER_1_MAX:
            tier_progress = games_played / GAMES_TIER_1_MAX
        elif games_played <= GAMES_TIER_2_MAX:
            tier_progress = (games_played - GAMES_TIER_1_MAX) / (GAMES_TIER_2_MAX - GAMES_TIER_1_MAX)
        elif games_played <= GAMES_TIER_3_MAX:
            tier_progress = (games_played - GAMES_TIER_2_MAX) / (GAMES_TIER_3_MAX - GAMES_TIER_2_MAX)
        else:
            tier_progress = min((games_played - GAMES_TIER_3_MAX) / 20, 1.0)
        confidence += tier_progress * 5

        # Injury penalty
        injury_penalties = {
            'ACTIVE': 0,
            'PROBABLE': -2,
            'QUESTIONABLE': -8,
            'DOUBTFUL': -15,
            'DAY_TO_DAY': -5,
            'DAY': -5,
            'GTD': -5,
            'OUT': -20,
            'INJ_RESERVE': -25,
            'INJURED_RESERVE': -25,
            'IR': -25,
        }
        injury_status_str = str(injury_status).upper() if injury_status else 'ACTIVE'
        confidence += injury_penalties.get(injury_status_str, -10)

        return max(0, min(100, confidence))

    def _build_confidence_intervals(
        self,
        combined_stats: Dict[str, float],
        stat_intervals: Dict[str, Tuple[float, float]],
        confidence_score: float
    ) -> Dict[str, Tuple[float, float]]:
        """
        Build confidence intervals for combined projections.

        Args:
            combined_stats: Combined projected stats
            stat_intervals: Intervals from statistical model
            confidence_score: Overall confidence score

        Returns:
            Dictionary of stat -> (low, high) intervals
        """
        intervals = {}

        # Confidence affects interval width
        # Higher confidence = narrower intervals
        width_multiplier = 2.0 - (confidence_score / 100)  # 1.0-2.0 range

        # Stats that should be clamped to [0, 1]
        percentage_stats = {'fg_pct', 'ft_pct', '3p_pct', 'efg_pct', 'ts_pct'}

        for stat, value in combined_stats.items():
            if stat in stat_intervals and stat_intervals[stat][0] is not None:
                # Use statistical interval width, but center on projected value
                low, high = stat_intervals[stat]
                half_width = (high - low) / 2
                adjusted_width = half_width * width_multiplier
            else:
                # Default interval based on stat volatility
                volatility = self._get_stat_volatility(stat)
                adjusted_width = abs(value) * volatility * width_multiplier

            # Always center the interval on the projected value
            low_bound = value - adjusted_width
            high_bound = value + adjusted_width

            # Apply bounds based on stat type
            if stat in percentage_stats:
                # Percentages must be in [0, 1]
                intervals[stat] = (
                    max(0.0, min(1.0, low_bound)),
                    max(0.0, min(1.0, high_bound))
                )
            else:
                # Counting stats must be >= 0
                intervals[stat] = (
                    max(0.0, low_bound),
                    high_bound
                )

        return intervals

    def _get_stat_volatility(self, stat: str) -> float:
        """Get typical volatility for a stat category."""
        volatilities = {
            'pts': 0.12,
            'trb': 0.15,
            'ast': 0.18,
            'stl': 0.25,
            'blk': 0.30,
            'tov': 0.20,
            '3p': 0.22,
            'fg_pct': 0.05,
            'ft_pct': 0.03,  # FT% is highly stable - narrower CI
        }
        return volatilities.get(stat, 0.15)

    # =========================================================================
    # Season & Timing Utilities
    # =========================================================================

    def _get_season_phase(self) -> SeasonPhase:
        """
        Determine current NBA season phase.

        Returns:
            SeasonPhase enumeration value
        """
        progress = self._get_season_progress()

        if progress < 0:
            return SeasonPhase.OFFSEASON
        elif progress < EARLY_SEASON_THRESHOLD:
            return SeasonPhase.EARLY
        elif progress < LATE_SEASON_THRESHOLD:
            return SeasonPhase.MID
        else:
            return SeasonPhase.LATE

    def _get_season_progress(self) -> float:
        """
        Calculate season progress as fraction (0.0 to 1.0).

        Returns:
            Season progress fraction, or -1 if offseason
        """
        today = date.today()

        # Approximate season dates
        if today.month >= 10:
            season_start = date(today.year, 10, 22)
            season_end = date(today.year + 1, 4, 15)
        elif today.month <= 4:
            season_start = date(today.year - 1, 10, 22)
            season_end = date(today.year, 4, 15)
        else:
            # Offseason (May-September)
            return -1.0

        if today < season_start:
            return -1.0
        if today > season_end:
            return 1.0

        total_days = (season_end - season_start).days
        elapsed_days = (today - season_start).days

        return elapsed_days / total_days

    def _get_dynamic_weights(
        self,
        phase: SeasonPhase
    ) -> Tuple[float, float]:
        """
        Get ML and statistical weights based on season phase.

        DEPRECATED: Use _get_tiered_weights instead for games-based weighting.

        Args:
            phase: Current season phase

        Returns:
            Tuple of (ml_weight, statistical_weight)
        """
        if phase == SeasonPhase.OFFSEASON:
            # Offseason: rely more on projections/historical
            return 0.20, 0.80

        weights = SEASON_WEIGHTS.get(phase.value, SEASON_WEIGHTS['mid'])
        return weights['ml'], weights['statistical']

    def _get_tiered_weights(self, games_played: int) -> Dict[str, float]:
        """
        Get projection weights based on player's games played this season.

        Tiered system:
        - 0-5 games:   60% ESPN proj, 30% prev season, 0% current, 10% ML
        - 6-15 games:  35% ESPN proj, 20% prev season, 35% current, 10% ML
        - 16-35 games: 15% ESPN proj, 10% prev season, 70% current, 5% ML
        - 35+ games:   0% ESPN proj, 0% prev season, 100% current, 0% ML

        Args:
            games_played: Number of games player has played this season

        Returns:
            Dictionary with weights for each source
        """
        if games_played <= GAMES_TIER_1_MAX:
            return TIERED_WEIGHTS['tier_1'].copy()
        elif games_played <= GAMES_TIER_2_MAX:
            return TIERED_WEIGHTS['tier_2'].copy()
        elif games_played <= GAMES_TIER_3_MAX:
            return TIERED_WEIGHTS['tier_3'].copy()
        else:
            return TIERED_WEIGHTS['tier_4'].copy()

    def _get_weight_tier_name(self, games_played: int) -> str:
        """Get the name of the weight tier for display purposes."""
        if games_played <= GAMES_TIER_1_MAX:
            return f"tier_1 (0-{GAMES_TIER_1_MAX} games)"
        elif games_played <= GAMES_TIER_2_MAX:
            return f"tier_2 ({GAMES_TIER_1_MAX+1}-{GAMES_TIER_2_MAX} games)"
        elif games_played <= GAMES_TIER_3_MAX:
            return f"tier_3 ({GAMES_TIER_2_MAX+1}-{GAMES_TIER_3_MAX} games)"
        else:
            return f"tier_4 ({GAMES_TIER_3_MAX}+ games)"

    def _estimate_games_remaining(self) -> int:
        """Estimate games remaining in season."""
        progress = self._get_season_progress()
        if progress < 0:
            return NBA_SEASON_GAMES
        return max(0, int(NBA_SEASON_GAMES * (1 - progress)))

    def _parse_return_date_from_notes(
        self,
        injury_notes: Optional[str],
        injury_status: str
    ) -> Optional[date]:
        """
        Parse expected return date from injury notes.

        Looks for common patterns in ESPN injury notes like:
        - "Expected to return Feb 10"
        - "Out until 2/15"
        - "Day-to-day, expected back Friday"
        - "Will return after All-Star break"

        Args:
            injury_notes: Injury notes/comments from ESPN
            injury_status: Current injury status

        Returns:
            Estimated return date or None if cannot determine
        """
        if not injury_notes:
            return None

        notes_lower = injury_notes.lower()
        today = date.today()

        # Pattern: "expected to return [date]" or "return [date]"
        # e.g., "Expected to return Feb 10" or "return 2/15"
        date_patterns = [
            # "Feb 10", "February 10", "Feb. 10"
            r'(?:return|back|play).*?(?:on\s+)?(\w+\.?\s+\d{1,2})',
            # "2/10", "2-10", "02/10"
            r'(?:return|back|play).*?(\d{1,2}[/-]\d{1,2})',
            # "expected [date]"
            r'expected.*?(\w+\.?\s+\d{1,2})',
        ]

        for pattern in date_patterns:
            match = re.search(pattern, notes_lower)
            if match:
                date_str = match.group(1)
                parsed_date = self._parse_date_string(date_str, today.year)
                if parsed_date and parsed_date > today:
                    return parsed_date

        # Pattern: "X weeks" or "X-Y weeks"
        weeks_match = re.search(r'(\d+)(?:\s*-\s*(\d+))?\s*weeks?', notes_lower)
        if weeks_match:
            min_weeks = int(weeks_match.group(1))
            max_weeks = int(weeks_match.group(2)) if weeks_match.group(2) else min_weeks
            avg_weeks = (min_weeks + max_weeks) / 2
            return today + timedelta(weeks=avg_weeks)

        # Pattern: "X days" or "X-Y days"
        days_match = re.search(r'(\d+)(?:\s*-\s*(\d+))?\s*days?', notes_lower)
        if days_match:
            min_days = int(days_match.group(1))
            max_days = int(days_match.group(2)) if days_match.group(2) else min_days
            avg_days = (min_days + max_days) / 2
            return today + timedelta(days=avg_days)

        # Common phrases
        if 'next game' in notes_lower or 'tonight' in notes_lower:
            return today  # Available now/today
        if 'tomorrow' in notes_lower:
            return today + timedelta(days=1)
        if 'this week' in notes_lower:
            return today + timedelta(days=3)
        if 'next week' in notes_lower:
            return today + timedelta(days=7)
        if 'all-star' in notes_lower:
            # All-Star break is typically mid-February
            return date(today.year, 2, 20)
        if 'season' in notes_lower and ('out' in notes_lower or 'end' in notes_lower):
            # Season-ending injury
            return None  # Will use INJ_RESERVE logic

        return None

    def _parse_date_string(self, date_str: str, year: int) -> Optional[date]:
        """Parse a date string like 'Feb 10' or '2/10' into a date object."""
        month_map = {
            'jan': 1, 'january': 1, 'feb': 2, 'february': 2,
            'mar': 3, 'march': 3, 'apr': 4, 'april': 4,
            'may': 5, 'jun': 6, 'june': 6, 'jul': 7, 'july': 7,
            'aug': 8, 'august': 8, 'sep': 9, 'september': 9,
            'oct': 10, 'october': 10, 'nov': 11, 'november': 11,
            'dec': 12, 'december': 12
        }

        date_str = date_str.lower().replace('.', '').strip()

        try:
            # Try "2/10" or "2-10" format
            if '/' in date_str or '-' in date_str:
                parts = re.split(r'[/-]', date_str)
                if len(parts) >= 2:
                    month = int(parts[0])
                    day = int(parts[1])
                    # If date is in the past, assume next year
                    result = date(year, month, day)
                    if result < date.today():
                        result = date(year + 1, month, day)
                    return result

            # Try "Feb 10" format
            parts = date_str.split()
            if len(parts) >= 2:
                month_str = parts[0].replace('.', '')
                month = month_map.get(month_str)
                if month:
                    day = int(re.search(r'\d+', parts[1]).group())
                    result = date(year, month, day)
                    if result < date.today():
                        result = date(year + 1, month, day)
                    return result

        except (ValueError, AttributeError):
            pass

        return None

    def _get_nba_schedule(self):
        """Lazy-load NBA schedule."""
        if self._nba_schedule is None:
            try:
                from backend.scrapers.nba_schedule import NBASchedule
                # Use current season (Oct-Apr spans two calendar years)
                current_month = date.today().month
                if current_month >= 10:  # Oct-Dec
                    season = date.today().year + 1
                else:  # Jan-Sep
                    season = date.today().year
                self._nba_schedule = NBASchedule(season=season)
                logger.info(f"Loaded NBA schedule for {season} season")
            except Exception as e:
                logger.warning(f"Could not load NBA schedule: {e}")
                self._nba_schedule = None
        return self._nba_schedule

    def _get_team_schedule_dates(
        self,
        nba_team: str,
        player_schedule: Optional[set] = None
    ) -> Optional[List[date]]:
        """
        Get the full schedule for a team as a sorted list of dates.

        Uses player_schedule if provided, otherwise fetches from NBASchedule.

        Args:
            nba_team: NBA team abbreviation (e.g., 'BOS', 'LAL')
            player_schedule: Optional pre-computed set of game dates

        Returns:
            Sorted list of game dates, or None if unavailable
        """
        if player_schedule:
            return sorted(player_schedule)

        if not nba_team:
            return None

        schedule = self._get_nba_schedule()
        if schedule:
            try:
                all_games = schedule.get_team_schedule(nba_team)
                return sorted(all_games) if all_games else None
            except Exception as e:
                logger.debug(f"Could not get schedule for {nba_team}: {e}")

        return None

    def _count_team_games_played(
        self,
        nba_team: str,
        player_schedule: Optional[set] = None
    ) -> int:
        """
        Count team games that have already been played (before today).

        Args:
            nba_team: NBA team abbreviation
            player_schedule: Optional pre-computed set of game dates

        Returns:
            Number of games played so far
        """
        schedule_dates = self._get_team_schedule_dates(nba_team, player_schedule)
        if not schedule_dates:
            # Fallback: estimate based on season progress
            progress = self._get_season_progress()
            return int(NBA_SEASON_GAMES * progress)

        today = date.today()
        return len([d for d in schedule_dates if d < today])

    def _count_team_games_remaining(
        self,
        nba_team: str,
        player_schedule: Optional[set] = None
    ) -> int:
        """
        Count team games remaining (from today onwards).

        Args:
            nba_team: NBA team abbreviation
            player_schedule: Optional pre-computed set of game dates

        Returns:
            Number of games remaining
        """
        schedule_dates = self._get_team_schedule_dates(nba_team, player_schedule)
        if not schedule_dates:
            # Fallback: use existing estimate
            return self._estimate_games_remaining()

        today = date.today()
        return len([d for d in schedule_dates if d >= today])

    def _count_team_games_after_date(
        self,
        nba_team: str,
        after_date: date,
        player_schedule: Optional[set] = None
    ) -> int:
        """
        Count team games from a specific date onwards.

        Args:
            nba_team: NBA team abbreviation
            after_date: Date from which to start counting (inclusive)
            player_schedule: Optional pre-computed set of game dates

        Returns:
            Number of games from after_date onwards
        """
        schedule_dates = self._get_team_schedule_dates(nba_team, player_schedule)
        if not schedule_dates:
            # Fallback: estimate based on days remaining
            today = date.today()
            season_end = date(today.year if today.month <= 4 else today.year + 1, 4, 13)
            days_from_date = (season_end - after_date).days
            total_days = (season_end - today).days
            if total_days <= 0:
                return 0
            games_remaining = self._estimate_games_remaining()
            return int(games_remaining * (days_from_date / total_days))

        return len([d for d in schedule_dates if d >= after_date])

    def _estimate_games_until_return(
        self,
        return_date: Optional[date],
        games_remaining: int,
        nba_team: str = None,
        player_name: str = "Unknown"
    ) -> int:
        """
        Count actual games missed until return date using NBA schedule.

        Uses actual team schedule to count games between today and return_date,
        not calendar day approximations.

        Args:
            return_date: Expected return date
            games_remaining: Total games remaining in season
            nba_team: Player's NBA team abbreviation (e.g., 'NOP', 'MEM')
            player_name: Player name for logging

        Returns:
            Actual games missed
        """
        if return_date is None:
            return games_remaining  # Season-ending

        today = date.today()
        if return_date <= today:
            return 0  # Already back or returning today

        # Try to use actual schedule data
        if nba_team:
            schedule = self._get_nba_schedule()
            if schedule:
                try:
                    # Get remaining games from today
                    remaining_games = schedule.get_team_remaining_games(nba_team, today)
                    # Count games BEFORE return_date (player is available ON return_date)
                    # e.g., if return_date = Feb 20, count games from today through Feb 19
                    games_missed = len([g for g in remaining_games if g < return_date])

                    # Calculate day before return for clearer logging
                    day_before_return = return_date - timedelta(days=1)
                    logger.info(f"Player {player_name} ({nba_team}): missing {games_missed} actual games "
                               f"({today} through {day_before_return}), available starting {return_date}")
                    return min(games_missed, games_remaining)
                except Exception as e:
                    logger.debug(f"Could not get schedule for {nba_team}: {e}")

        # Fallback: approximate if no schedule data
        days_until_return = (return_date - today).days
        games_missed = int(days_until_return * 0.5)  # ~3.5 games/week = 0.5 games/day
        logger.debug(f"Player {player_name}: using estimated {games_missed} games missed (no schedule data)")

        return min(games_missed, games_remaining)

    def _estimate_player_games(
        self,
        games_played: int,
        games_remaining: int,
        injury_status: str,
        injury_notes: Optional[str] = None,
        games_until_return: Optional[int] = None,
        expected_return_date: Optional[date] = None,
        player_name: str = "Unknown",
        nba_team: str = None,
        player_schedule: Optional[set] = None,
        projection_method: str = 'adaptive',
        flat_game_rate: float = 0.85
    ) -> int:
        """
        Estimate how many more games a player will play.

        Uses actual NBA schedule data when available to calculate game rates
        and games after return. Supports two projection methods:
        - 'adaptive': 0-4 GP uses 90% rate, 5+ GP uses actual rate (min 75%)
        - 'flat_rate': Uses fixed percentage for all players

        Args:
            games_played: Games played so far
            games_remaining: Team games remaining (fallback if no schedule)
            injury_status: Current injury status
            injury_notes: Injury notes/comments that may contain return info
            games_until_return: Optional override for expected games missed
            expected_return_date: ESPN's expected return date for injured players
            player_name: Player name for logging
            nba_team: Player's NBA team abbreviation for schedule lookup
            player_schedule: Optional pre-computed set of game dates for player's team
            projection_method: 'adaptive' (tiered rates) or 'flat_rate' (fixed rate)
            flat_game_rate: Fixed rate to use when projection_method='flat_rate' (0.50-1.00)

        Returns:
            Estimated games to play
        """
        # Handle None values - default to adaptive if not specified
        if projection_method is None:
            projection_method = 'adaptive'
        if flat_game_rate is None:
            flat_game_rate = 0.85

        # Log projection settings being used (INFO level for visibility)
        logger.info(f"[PROJECTION SETTINGS] {player_name}: method={projection_method}, flat_rate={flat_game_rate:.1%}")

        injury_status_str = str(injury_status).upper() if injury_status else 'ACTIVE'
        today = date.today()

        # Get actual team games from schedule
        team_games_so_far = self._count_team_games_played(nba_team, player_schedule)
        team_games_remaining = self._count_team_games_remaining(nba_team, player_schedule)

        # Calculate game_rate based on projection method
        if projection_method == 'flat_rate':
            # User-configured flat rate for all players
            game_rate = max(0.50, min(1.0, flat_game_rate))
            logger.info(f"Player {player_name}: {games_played}/{team_games_so_far} games played, "
                       f"rate={game_rate:.1%} (flat_rate mode)")
        else:
            # Adaptive mode: tiered rates based on games played
            if games_played < 5:
                # 0-4 games: Use default 90% rate (grace period for new/returning players)
                game_rate = 0.9
                logger.info(f"Player {player_name}: {games_played}/{team_games_so_far} games played, "
                           f"rate=90.0% (< 5 GP grace period)")
            else:
                # 5+ games: Use actual rate with 75% floor
                if team_games_so_far > 0:
                    calculated_rate = games_played / team_games_so_far
                    game_rate = max(0.75, min(1.0, calculated_rate))
                else:
                    game_rate = 0.9  # Fallback if no team games data
                logger.info(f"Player {player_name}: {games_played}/{team_games_so_far} games played, "
                           f"rate={game_rate:.1%}")

        # Baseline projection without injury consideration
        baseline_games = int(team_games_remaining * game_rate)

        # If explicit games until return provided, use that
        if games_until_return is not None:
            projected = max(0, baseline_games - games_until_return)
            logger.info(f"Player {player_name}: explicit games_until_return={games_until_return}, "
                       f"projected {projected} games")
            return projected

        # DTD (day-to-day) players: assume they return next game (minimal reduction)
        if injury_status_str in ['DAY_TO_DAY', 'DAY', 'GTD', 'QUESTIONABLE', 'PROBABLE']:
            games_missed = 1 if injury_status_str in ['DAY_TO_DAY', 'DAY', 'GTD', 'QUESTIONABLE'] else 0
            projected = max(0, baseline_games - games_missed)
            logger.info(f"Player {player_name} {injury_status_str} - projecting {projected} games (minor reduction)")
            return projected

        # Check for expected_return_date from ESPN (highest priority for OUT/IR players)
        return_date = None
        if expected_return_date:
            if isinstance(expected_return_date, date):
                return_date = expected_return_date
            elif isinstance(expected_return_date, str):
                # Parse string date
                try:
                    return_date = datetime.strptime(expected_return_date, '%Y-%m-%d').date()
                except ValueError:
                    pass

        # If no ESPN return date, try to parse from injury notes
        if return_date is None:
            return_date = self._parse_return_date_from_notes(injury_notes, injury_status)

        # Calculate games based on return date using actual schedule
        if return_date:
            if return_date <= today:
                # Already returned or returning today - full games at game_rate
                projected = int(team_games_remaining * game_rate)
                logger.info(f"Player {player_name} return date {return_date} is today or past - "
                           f"projecting {projected} games")
                return max(0, projected)
            else:
                # Player returns in the future - count actual games after return
                team_games_after_return = self._count_team_games_after_date(
                    nba_team, return_date, player_schedule
                )
                games_missed = team_games_remaining - team_games_after_return
                projected = int(team_games_after_return * game_rate)

                logger.info(f"Player {player_name} returns {return_date}, "
                           f"will miss {games_missed}, projected {projected} games after return")

                return max(0, projected)
        else:
            # No return date available - use heuristics
            games_missed = 0

            if injury_status_str in ['OUT', 'O']:
                # Check injury notes for season-ending indicators
                notes_lower = (injury_notes or '').lower()
                if any(x in notes_lower for x in ['season', 'acl', 'achilles', 'surgery']):
                    # Likely season-ending
                    logger.info(f"Player {player_name} OUT (likely season-ending) - projecting 0 games")
                    return 0
                else:
                    # Generic OUT without return date - assume extended absence
                    games_missed = min(team_games_remaining, int(team_games_remaining * 0.5))
                    projected = int((team_games_remaining - games_missed) * game_rate)
                    logger.info(f"Player {player_name} OUT (no return date) - "
                               f"projecting {projected} games (conservative)")
            elif injury_status_str in ['INJ_RESERVE', 'INJURED_RESERVE', 'IR']:
                # IR without return date = season-ending
                logger.info(f"Player {player_name} IR (no return date) - projecting 0 games (season-ending)")
                return 0
            else:
                # Other statuses (ACTIVE, SUSPENSION, etc.)
                injury_games_missed = {
                    'ACTIVE': 0,
                    'SUSPENSION': 5,
                }
                games_missed = injury_games_missed.get(injury_status_str, 2)
                projected = int((team_games_remaining - games_missed) * game_rate)

            # Apply injury risk factor for non-healthy players
            if injury_status_str not in ['ACTIVE', 'PROBABLE']:
                injury_risk_factor = 0.95
                projected = int(projected * injury_risk_factor)

            return max(0, projected)

    def _calculate_ros_totals(
        self,
        per_game: Dict[str, float],
        games: int
    ) -> Dict[str, float]:
        """
        Calculate rest-of-season totals from per-game averages.

        Counting stats are multiplied by games remaining.
        Percentage/rate stats stay as-is.

        Args:
            per_game: Per-game averages
            games: Number of games to project

        Returns:
            ROS total stats
        """
        totals = {}

        # All counting stats (including aliases)
        COUNTING_STAT_KEYS = {
            'pts', 'reb', 'trb', 'ast', 'stl', 'blk', 'tov', 'to',
            '3p', '3pm', '3pa', 'fgm', 'fga', 'ftm', 'fta',
            'oreb', 'dreb', 'pf', 'min'
        }

        for stat, value in per_game.items():
            if stat in COUNTING_STAT_KEYS:
                totals[stat] = value * games
            else:
                # Rate stats (fg_pct, ft_pct, etc.) stay as-is
                totals[stat] = value

        logger.info(f"[HYBRID] ROS totals ({games} games): pts={totals.get('pts', 0):.1f}, reb={totals.get('reb', totals.get('trb', 0)):.1f}, ast={totals.get('ast', 0):.1f}, fgm={totals.get('fgm', 0):.1f}, fga={totals.get('fga', 0):.1f}")

        # Warn if ROS totals are all zero
        if games > 0 and all(totals.get(k, 0) == 0 for k in ['pts', 'trb', 'reb', 'ast']):
            logger.warning(f"[HYBRID] ROS totals are ALL ZERO despite {games} projected games!")
            logger.warning(f"[HYBRID] Input per_game stats were: {per_game}")

        return totals


# =============================================================================
# Convenience Functions
# =============================================================================

def create_hybrid_engine(
    models_dir: Optional[str] = None
) -> HybridProjectionEngine:
    """
    Factory function to create a configured hybrid engine.

    Args:
        models_dir: Optional directory for ML models

    Returns:
        Configured HybridProjectionEngine
    """
    return HybridProjectionEngine(models_dir=models_dir)


def quick_hybrid_projection(
    player_name: str,
    season_stats: Dict[str, float],
    games_played: int,
    injury_status: str = 'ACTIVE',
    league_type: str = 'h2h_category'
) -> HybridProjection:
    """
    Quick single-player hybrid projection.

    Args:
        player_name: Player's name
        season_stats: Current season per-game averages
        games_played: Games played so far
        injury_status: Injury status
        league_type: League type for scoring

    Returns:
        HybridProjection object
    """
    engine = HybridProjectionEngine()

    # Get league settings
    if league_type.lower() in ['points', 'h2h_points']:
        settings = LeagueScoringSettings.default_points()
    else:
        settings = LeagueScoringSettings.default_h2h_category()

    player_data = {
        'player_id': player_name.lower().replace(' ', '_'),
        'name': player_name,
        'team': 'UNK',
        'position': season_stats.get('position', 'N/A'),
        'games_played': games_played,
        'age': season_stats.get('age', 25)
    }

    return engine.project_player(
        player_id=player_data['player_id'],
        player_data=player_data,
        season_stats=season_stats,
        injury_status=injury_status,
        league_settings=settings
    )


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """Demo/test entry point for hybrid projections."""
    logger.info("=" * 60)
    logger.info("Hybrid Projection Engine Demo")
    logger.info("=" * 60)

    # Create engine
    engine = HybridProjectionEngine()

    # Show current season phase
    phase = engine._get_season_phase()
    progress = engine._get_season_progress()
    ml_weight, stat_weight = engine._get_dynamic_weights(phase)

    print(f"\nSeason Status:")
    print(f"  Phase: {phase.value}")
    print(f"  Progress: {progress*100:.1f}%" if progress >= 0 else "  Progress: Offseason")
    print(f"  ML Weight: {ml_weight:.0%}")
    print(f"  Statistical Weight: {stat_weight:.0%}")

    # Example player
    player_data = {
        'player_id': 'demo_player',
        'name': 'Demo Star',
        'team': 'LAL',
        'position': 'PG',
        'games_played': 45,
        'age': 27
    }

    season_stats = {
        'pts': 28.5,
        'trb': 7.2,
        'ast': 8.1,
        'stl': 1.5,
        'blk': 0.6,
        'tov': 3.2,
        '3p': 3.2,
        'fg_pct': 0.505,
        'ft_pct': 0.890,
        'fga': 20.5,
        'fta': 8.2,
        'mp': 35.5,
        'usg_pct': 0.32,
        'ts_pct': 0.62,
    }

    espn_projection = {
        'pts': 27.8,
        'trb': 7.0,
        'ast': 8.5,
        'stl': 1.4,
        'blk': 0.5,
        '3p': 3.0,
    }

    # Generate projection with default H2H category settings
    projection = engine.project_player(
        player_id='demo_player',
        player_data=player_data,
        season_stats=season_stats,
        espn_projection=espn_projection,
        injury_status='ACTIVE'
    )

    # Print results
    print("\n" + "=" * 60)
    print(f"Hybrid Projection: {projection.player_name}")
    print("=" * 60)
    print(f"Team: {projection.team} | Position: {projection.position}")
    print(f"Games Projected ROS: {projection.games_projected}")
    print(f"Confidence Score: {projection.confidence_score:.0f}/100")
    print(f"Season Phase: {projection.season_phase}")

    print("\nProjected Per-Game Stats:")
    print("-" * 50)
    print(f"{'Stat':<8} {'Proj':>8} {'ML':>8} {'Stat':>8} {'CI':>20}")
    print("-" * 50)

    for stat in ['pts', 'trb', 'ast', 'stl', 'blk', '3p', 'tov', 'fg_pct', 'ft_pct']:
        if stat in projection.projected_stats:
            proj = projection.projected_stats[stat]
            ml = projection.ml_contribution.get(stat, 0)
            st = projection.statistical_contribution.get(stat, 0)
            ci = projection.confidence_intervals.get(stat, (0, 0))
            ci_str = f"[{ci[0]:.2f}-{ci[1]:.2f}]"
            print(f"{stat.upper():<8} {proj:>8.2f} {ml:>8.2f} {st:>8.2f} {ci_str:>20}")

    print("\nFantasy Value:")
    print("-" * 40)
    print(f"  Fantasy Points/Game: {projection.fantasy_points:.2f}")
    print(f"  ROS Total Points:    {projection.ros_totals.get('pts', 0):.0f}")

    print("\nCategory Values:")
    for cat, val in sorted(projection.category_values.items(), key=lambda x: -x[1]):
        print(f"  {cat.upper():<8}: {val:>6.2f}")

    print("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())

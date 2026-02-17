# Projections Package
"""
Fantasy Basketball Projection Engine.

This package provides player projections using a hybrid approach combining:
- ML models (Gradient Boosting for counting stats, Ridge for percentages)
- Statistical models (weighted averages from multiple sources)

Main classes:
- HybridProjectionEngine: Main projection interface
- HybridProjection: Projection result dataclass
- SimpleProjectionEngine: Fallback engine using current stats

Usage:
    from backend.projections import HybridProjectionEngine, HybridProjection

    engine = HybridProjectionEngine()
    projection = engine.project_player(
        player_id='12345',
        player_data={'name': 'Player Name', 'games_played': 40},
        season_stats={'pts': 25.0, 'trb': 8.0, 'ast': 6.0}
    )
"""

# Import main classes for convenient access
try:
    from .hybrid_engine import (
        HybridProjectionEngine,
        HybridProjection,
        LeagueScoringSettings,
        LeagueType,
        SeasonPhase,
        create_hybrid_engine,
    )
    from .simple_projection import SimpleProjectionEngine
except ImportError:
    # Allow package to be imported even if dependencies are missing
    pass

# Import start limit optimizer
try:
    from .start_limit_optimizer import (
        StartLimitOptimizer,
        PlayerGameLog,
        DaySimulation,
        SeasonSimulationResult,
        SLOT_ID_TO_POSITION,
        POSITION_TO_SLOT_ID,
        STARTING_SLOT_IDS,
        create_optimizer_from_league,
    )
except ImportError:
    # Allow package to be imported even if dependencies are missing
    pass

__all__ = [
    'HybridProjectionEngine',
    'HybridProjection',
    'LeagueScoringSettings',
    'LeagueType',
    'SeasonPhase',
    'SimpleProjectionEngine',
    'create_hybrid_engine',
    # Start limit optimizer
    'StartLimitOptimizer',
    'PlayerGameLog',
    'DaySimulation',
    'SeasonSimulationResult',
    'SLOT_ID_TO_POSITION',
    'POSITION_TO_SLOT_ID',
    'STARTING_SLOT_IDS',
    'create_optimizer_from_league',
]

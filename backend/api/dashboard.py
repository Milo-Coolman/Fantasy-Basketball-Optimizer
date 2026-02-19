"""
Dashboard API Endpoints.

Provides comprehensive league dashboard data including standings,
projections, team highlights, and quick insights.

Reference: PRD Section 5.5 - Dashboard Endpoints
"""

import importlib
import logging
import os
import sys
import traceback
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Tuple

from flask import Blueprint, jsonify, request
from flask_login import login_required, current_user

from backend.extensions import db
from backend.models import League, Team, Player, PlayerStats, Roster
from backend.services.espn_client import (
    ESPNClient,
    ESPNClientError,
    ESPNAuthenticationError,
    ESPNConnectionError,
)
from backend.services.league_service import (
    get_league_or_404,
    get_league_credentials,
    LeagueNotFoundError,
    LeagueAccessDeniedError,
)
from backend.services.team_service import (
    get_league_teams,
    get_league_standings,
)
from backend.services.cache_service import get_cache, CacheTTL

logger = logging.getLogger(__name__)

# =============================================================================
# Projection Engine Imports (with robust fallback)
# =============================================================================

HYBRID_ENGINE_AVAILABLE = False
SIMPLE_ENGINE_AVAILABLE = False
HybridProjectionEngine = None
SimpleProjectionEngine = None

def _import_projection_engines():
    """Import projection engines with multiple fallback strategies."""
    global HYBRID_ENGINE_AVAILABLE, SIMPLE_ENGINE_AVAILABLE
    global HybridProjectionEngine, SimpleProjectionEngine

    # Strategy 1: Try standard imports
    try:
        from backend.projections.hybrid_engine import HybridProjectionEngine as HE
        HybridProjectionEngine = HE
        HYBRID_ENGINE_AVAILABLE = True
        logger.info("Hybrid projection engine imported via standard import")
    except ImportError as e:
        logger.debug(f"Standard hybrid import failed: {e}")

        # Strategy 2: Try adding backend to path
        try:
            backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if backend_dir not in sys.path:
                sys.path.insert(0, backend_dir)

            from projections.hybrid_engine import HybridProjectionEngine as HE
            HybridProjectionEngine = HE
            HYBRID_ENGINE_AVAILABLE = True
            logger.info("Hybrid projection engine imported via path modification")
        except ImportError as e2:
            logger.warning(f"Hybrid engine import failed: {e2}")

    # Import simple projection engine
    try:
        from backend.projections.simple_projection import SimpleProjectionEngine as SE
        SimpleProjectionEngine = SE
        SIMPLE_ENGINE_AVAILABLE = True
        logger.info("Simple projection engine imported via standard import")
    except ImportError:
        try:
            from projections.simple_projection import SimpleProjectionEngine as SE
            SimpleProjectionEngine = SE
            SIMPLE_ENGINE_AVAILABLE = True
            logger.info("Simple projection engine imported via path modification")
        except ImportError as e:
            logger.debug(f"Simple engine import failed: {e}")

# Run imports at module load
_import_projection_engines()

# =============================================================================
# Start Limit Optimizer Import
# =============================================================================

START_LIMIT_OPTIMIZER_AVAILABLE = False
StartLimitOptimizer = None

try:
    from backend.projections.start_limit_optimizer import StartLimitOptimizer as SLO
    StartLimitOptimizer = SLO
    START_LIMIT_OPTIMIZER_AVAILABLE = True
    logger.info("Start limit optimizer imported via standard import")
except ImportError:
    try:
        from projections.start_limit_optimizer import StartLimitOptimizer as SLO
        StartLimitOptimizer = SLO
        START_LIMIT_OPTIMIZER_AVAILABLE = True
        logger.info("Start limit optimizer imported via path modification")
    except ImportError as e:
        logger.debug(f"Start limit optimizer import failed: {e}")

dashboard_bp = Blueprint('dashboard', __name__)


# =============================================================================
# Constants
# =============================================================================

# Category metadata - labels, abbreviations, and scoring direction
# Used to build full category definitions from stat keys stored in database
CATEGORY_METADATA = {
    'FGM': {'label': 'Field Goals Made', 'abbr': 'FGM', 'lower_is_better': False},
    'FG%': {'label': 'Field Goal %', 'abbr': 'FG%', 'lower_is_better': False},
    'FTM': {'label': 'Free Throws Made', 'abbr': 'FTM', 'lower_is_better': False},
    'FT%': {'label': 'Free Throw %', 'abbr': 'FT%', 'lower_is_better': False},
    '3PM': {'label': '3-Pointers Made', 'abbr': '3PM', 'lower_is_better': False},
    'PTS': {'label': 'Points', 'abbr': 'PTS', 'lower_is_better': False},
    'REB': {'label': 'Rebounds', 'abbr': 'REB', 'lower_is_better': False},
    'AST': {'label': 'Assists', 'abbr': 'AST', 'lower_is_better': False},
    'STL': {'label': 'Steals', 'abbr': 'STL', 'lower_is_better': False},
    'BLK': {'label': 'Blocks', 'abbr': 'BLK', 'lower_is_better': False},
    'TO': {'label': 'Turnovers', 'abbr': 'TO', 'lower_is_better': True},
}

# Default category order for display
DEFAULT_CATEGORY_ORDER = ['FGM', 'FG%', 'FTM', 'FT%', '3PM', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TO']

# Mapping from ESPN category names to hybrid engine internal stat names
# ESPN uses uppercase (PTS, REB, 3PM), hybrid engine uses lowercase (pts, trb, 3p)
ESPN_TO_ENGINE_STAT_MAP = {
    'PTS': 'pts',
    'REB': 'trb',
    'AST': 'ast',
    'STL': 'stl',
    'BLK': 'blk',
    '3PM': '3p',
    'FG%': 'fg_pct',
    'FT%': 'ft_pct',
    'TO': 'tov',
    'FGM': 'fgm',
    'FTM': 'ftm',
    'FGA': 'fga',
    'FTA': 'fta',
}

# Reverse mapping: engine stat names to ESPN names
ENGINE_TO_ESPN_STAT_MAP = {v: k for k, v in ESPN_TO_ENGINE_STAT_MAP.items()}

# Counting stats that should be summed for team totals
COUNTING_STATS_ESPN = {'PTS', 'REB', 'AST', 'STL', 'BLK', '3PM', 'FGM', 'FTM', 'FGA', 'FTA', 'TO'}

# Percentage stats that need weighted average calculation
PERCENTAGE_STATS_ESPN = {'FG%', 'FT%'}


# =============================================================================
# Helper Functions
# =============================================================================

def get_espn_client_for_league(league_id: int, user_id: int) -> ESPNClient:
    """
    Create an ESPN client for a league.

    Args:
        league_id: League database ID
        user_id: User ID for authorization

    Returns:
        Connected ESPNClient instance
    """
    credentials = get_league_credentials(league_id, user_id)

    return ESPNClient(
        league_id=credentials['espn_league_id'],
        year=credentials['season'],
        espn_s2=credentials['espn_s2'],
        swid=credentials['swid']
    )


def get_league_scoring_categories(league: League, espn_client: Optional[ESPNClient] = None) -> List[Dict]:
    """
    Get the scoring categories configured for the league from database.

    Categories can be stored in two formats:
    - Old format: list of strings ['PTS', 'REB', ...]
    - New format: list of dicts [{'key': 'PTS', 'label': 'Points', ...}, ...]

    This function normalizes both formats to full category definitions.

    Args:
        league: League database object
        espn_client: Optional ESPN client for fallback detection

    Returns:
        List of category definitions with:
            - key: Stat key (e.g., 'PTS')
            - stat_key: Same as key
            - label: Human-readable label
            - abbr: Abbreviation for display
            - lower_is_better: True for TO, False for others
            - enabled: Always True for detected categories
    """
    categories = []

    # Try to get categories from database scoring_settings
    scoring_settings = league.scoring_settings or {}
    stored_categories = scoring_settings.get('categories', [])

    if stored_categories and isinstance(stored_categories, list):
        logger.info(f"Using stored categories for league {league.id}: {len(stored_categories)} categories")

        # Check format: list of strings or list of dicts
        if stored_categories and isinstance(stored_categories[0], dict):
            # New format: list of dicts with full category info
            logger.debug("Categories in new dict format")

            # Sort by default order
            def sort_key(cat):
                cat_key = cat.get('key', '')
                try:
                    return DEFAULT_CATEGORY_ORDER.index(cat_key)
                except ValueError:
                    return len(DEFAULT_CATEGORY_ORDER) + 1

            sorted_cats = sorted(stored_categories, key=sort_key)

            for cat in sorted_cats:
                cat_key = cat.get('key', '')
                if not cat_key:
                    continue

                # Use stored metadata, fall back to defaults
                categories.append({
                    'key': cat_key,
                    'stat_key': cat_key,
                    'label': cat.get('label', CATEGORY_METADATA.get(cat_key, {}).get('label', cat_key)),
                    'abbr': cat.get('abbr', CATEGORY_METADATA.get(cat_key, {}).get('abbr', cat_key)),
                    'lower_is_better': cat.get('is_reverse', CATEGORY_METADATA.get(cat_key, {}).get('lower_is_better', False)),
                    'enabled': True
                })
        else:
            # Old format: list of strings
            logger.debug("Categories in old string format")

            def sort_key(cat_key):
                try:
                    return DEFAULT_CATEGORY_ORDER.index(cat_key)
                except ValueError:
                    return len(DEFAULT_CATEGORY_ORDER) + 1

            sorted_cat_keys = sorted(stored_categories, key=sort_key)

            for cat_key in sorted_cat_keys:
                # Get metadata for this category
                metadata = CATEGORY_METADATA.get(cat_key, {
                    'label': cat_key,
                    'abbr': cat_key,
                    'lower_is_better': False
                })

                categories.append({
                    'key': cat_key,
                    'stat_key': cat_key,
                    'label': metadata['label'],
                    'abbr': metadata['abbr'],
                    'lower_is_better': metadata.get('lower_is_better', False),
                    'enabled': True
                })

    # Fallback: try to detect from ESPN client if no stored categories
    if not categories and espn_client:
        logger.info(f"No stored categories for league {league.id}, detecting from ESPN")
        try:
            detected_categories = espn_client.get_scoring_categories()

            # get_scoring_categories now returns list of dicts
            for cat in detected_categories:
                if isinstance(cat, dict):
                    cat_key = cat.get('key', '')
                    if not cat_key:
                        continue
                    categories.append({
                        'key': cat_key,
                        'stat_key': cat_key,
                        'label': cat.get('label', cat_key),
                        'abbr': cat.get('abbr', cat_key),
                        'lower_is_better': cat.get('is_reverse', False),
                        'enabled': True
                    })
                else:
                    # Backward compatibility: string format
                    cat_key = cat
                    metadata = CATEGORY_METADATA.get(cat_key, {
                        'label': cat_key,
                        'abbr': cat_key,
                        'lower_is_better': False
                    })
                    categories.append({
                        'key': cat_key,
                        'stat_key': cat_key,
                        'label': metadata['label'],
                        'abbr': metadata['abbr'],
                        'lower_is_better': metadata.get('lower_is_better', False),
                        'enabled': True
                    })
        except Exception as e:
            logger.warning(f"Could not detect categories from ESPN: {e}")

    # Final fallback: use default category order
    if not categories:
        logger.warning(f"Using default categories for league {league.id}")
        for cat_key in DEFAULT_CATEGORY_ORDER:
            metadata = CATEGORY_METADATA.get(cat_key)
            if metadata:
                categories.append({
                    'key': cat_key,
                    'stat_key': cat_key,
                    'label': metadata['label'],
                    'abbr': metadata['abbr'],
                    'lower_is_better': metadata.get('lower_is_better', False),
                    'enabled': True
                })

    logger.debug(f"Scoring categories for league {league.id}: {[c['abbr'] for c in categories]}")
    return categories


def calculate_roto_category_ranks(
    teams: List[Dict],
    categories: List[Dict],
    espn_client: ESPNClient
) -> Dict[int, Dict]:
    """
    Calculate category ranks for all teams in a Roto league.

    Roto scoring works as follows:
    1. For each category, teams are ranked 1 to N (1 = best)
    2. For counting stats (PTS, REB, AST, etc.), HIGHER values get rank 1
    3. For percentage stats (FG%, FT%), HIGHER percentages get rank 1
    4. For turnovers (TO), LOWER values get rank 1 (lower_is_better)
    5. Ranks are converted to points: points = num_teams - rank + 1
       (1st place gets N points, 2nd gets N-1, etc.)
    6. Total points = sum of all category points
    7. Teams are ranked by total points (highest total = rank 1)

    Args:
        teams: List of team data from ESPN (must include 'stats' dict)
        categories: Scoring categories (excludes GP)
        espn_client: Connected ESPN client

    Returns:
        Dict mapping team_id to:
            - category_ranks: rank in each category (1-N)
            - category_values: raw stat value in each category
            - category_points: points earned in each category (N-1)
            - total_points: sum of all category points
            - roto_points: same as total_points (for compatibility)
    """
    num_teams = len(teams)
    logger.info(f"=== ROTO STANDINGS CALCULATION START ===")
    logger.info(f"Number of teams: {num_teams}")
    logger.info(f"Categories being scored: {[c['abbr'] for c in categories]}")

    # Step 1: Extract raw stats for each team
    team_stats = {}
    for team in teams:
        team_id = team['espn_team_id']
        team_name = team.get('team_name', 'Unknown')
        stats = team.get('stats', {})

        logger.debug(f"Team {team_id} ({team_name}) raw stats: {stats}")

        team_stats[team_id] = {
            'team_name': team_name,
            'stats': stats
        }

    # Initialize result structure
    team_ranks = {}
    for team_id in team_stats:
        team_ranks[team_id] = {
            'team_name': team_stats[team_id]['team_name'],
            'category_ranks': {},      # Rank in each category (1 = best)
            'category_values': {},     # Raw stat value
            'category_points': {},     # Points earned (num_teams - rank + 1)
        }

    # Step 2: For each category, rank all teams and assign points
    logger.info(f"\n--- Category Rankings ---")
    for cat in categories:
        stat_key = cat['stat_key']
        abbr = cat['abbr']
        lower_is_better = cat.get('lower_is_better', False)

        logger.info(f"\nCategory: {abbr} (stat_key: {stat_key}, lower_is_better: {lower_is_better})")

        # Get values for all teams
        team_values = []
        for team_id, data in team_stats.items():
            raw_value = data['stats'].get(stat_key, 0)
            # Handle None values
            value = float(raw_value) if raw_value is not None else 0.0
            team_values.append({
                'team_id': team_id,
                'team_name': data['team_name'],
                'value': value
            })

        # Sort teams by value
        # For most stats: higher is better (reverse=True)
        # For TO: lower is better (reverse=False)
        team_values.sort(
            key=lambda x: x['value'],
            reverse=(not lower_is_better)
        )

        # Log the sorted order
        logger.info(f"  Sorted order (best to worst):")
        for i, tv in enumerate(team_values, 1):
            logger.info(f"    {i}. {tv['team_name']}: {tv['value']:.2f}")

        # Assign ranks and points
        for rank, tv in enumerate(team_values, 1):
            team_id = tv['team_id']
            value = tv['value']
            # Points = num_teams - rank + 1
            # 1st place gets num_teams points, last place gets 1 point
            points = num_teams - rank + 1

            team_ranks[team_id]['category_ranks'][stat_key] = rank
            team_ranks[team_id]['category_values'][stat_key] = value
            team_ranks[team_id]['category_points'][stat_key] = points

            logger.debug(f"    {tv['team_name']}: rank={rank}, value={value:.2f}, points={points}")

    # Step 3: Calculate total points for each team
    logger.info(f"\n--- Total Points Calculation ---")
    for team_id, data in team_ranks.items():
        total_points = sum(data['category_points'].values())
        data['total_points'] = total_points
        data['roto_points'] = total_points  # Alias for compatibility

        logger.info(f"  {data['team_name']}: {total_points} total points")
        logger.debug(f"    Breakdown: {data['category_points']}")

    # Step 4: Calculate overall ranking (by total points, highest = rank 1)
    sorted_by_total = sorted(
        team_ranks.items(),
        key=lambda x: x[1]['total_points'],
        reverse=True
    )

    logger.info(f"\n--- Final Standings ---")
    for final_rank, (team_id, data) in enumerate(sorted_by_total, 1):
        data['overall_rank'] = final_rank
        logger.info(f"  {final_rank}. {data['team_name']}: {data['total_points']} points")

    logger.info(f"=== ROTO STANDINGS CALCULATION COMPLETE ===\n")

    return team_ranks


def _extract_games_played(player: Dict, stats: Dict, season: Optional[int] = None) -> int:
    """
    Extract games_played from all possible locations in player data.

    Searches in order of priority:
    1. Top-level player['games_played'] (set by get_all_rosters and _parse_player)
    2. stats['current_season']['games_played']
    3. stats['{season}_total'] with various keys (GP, gamesPlayed, applied_total)
    4. Calculated from total_points / avg_points if both available

    Args:
        player: Player dictionary
        stats: Player stats dictionary
        season: Season year for dynamic key access

    Returns:
        Games played count (0 if not found)
    """
    games_played = 0
    player_name = player.get('name', 'Unknown')

    # 1. Check top-level games_played (preferred - set by ESPN client)
    if 'games_played' in player:
        gp = player.get('games_played', 0)
        if gp and int(gp) > 0:
            games_played = int(gp)
            logger.debug(f"  {player_name}: games_played from top-level: {games_played}")
            return games_played

    # 2. Check stats['current_season']['games_played']
    current_season = stats.get('current_season', {})
    if current_season:
        gp = current_season.get('games_played', 0) or current_season.get('gamesPlayed', 0) or 0
        if gp and int(gp) > 0:
            games_played = int(gp)
            logger.debug(f"  {player_name}: games_played from current_season: {games_played}")
            return games_played

    # 3. Check dynamic season keys: {season}_total, 00{season}, etc.
    if season:
        season_keys = [
            f'{season}_total',
            f'00{season}',
            f'{season}',
        ]
    else:
        # Fallback to common years
        season_keys = ['2026_total', '2025_total', '002026', '002025', 'total']

    for key in season_keys:
        if key in stats and isinstance(stats[key], dict):
            stat_data = stats[key]

            # Try various field names for games played
            gp = (
                stat_data.get('GP', 0) or
                stat_data.get('gamesPlayed', 0) or
                stat_data.get('games_played', 0) or
                stat_data.get('applied_total', 0) or
                stat_data.get('appliedTotal', 0) or
                0
            )
            if gp and int(gp) > 0:
                games_played = int(gp)
                logger.debug(f"  {player_name}: games_played from {key}: {games_played}")
                return games_played

            # Check inside 'total' sub-dict for stat ID 41 (GP)
            if 'total' in stat_data and isinstance(stat_data['total'], dict):
                total_dict = stat_data['total']
                for gp_key in ['41', 41, 'GP', 'gamesPlayed']:
                    gp = total_dict.get(gp_key, 0)
                    if gp and int(gp) > 0:
                        games_played = int(gp)
                        logger.debug(f"  {player_name}: games_played from {key}.total[{gp_key}]: {games_played}")
                        return games_played

    # 4. Try to calculate from total_points / avg_points
    # If we have both, we can derive games played
    for key in season_keys:
        if key in stats and isinstance(stats[key], dict):
            stat_data = stats[key]
            avg = stat_data.get('avg', {})
            total = stat_data.get('total', {})

            if avg and total:
                # Use PTS (stat ID 0) to calculate
                avg_pts = avg.get('0', 0) or avg.get(0, 0) or avg.get('PTS', 0) or 0
                total_pts = total.get('0', 0) or total.get(0, 0) or total.get('PTS', 0) or 0

                if avg_pts and avg_pts > 0 and total_pts and total_pts > 0:
                    calculated_gp = int(total_pts / avg_pts)
                    if calculated_gp > 0:
                        games_played = calculated_gp
                        logger.debug(f"  {player_name}: games_played calculated from PTS ({total_pts}/{avg_pts}): {games_played}")
                        return games_played

    logger.debug(f"  {player_name}: games_played not found, returning 0")
    return 0


def get_player_current_stats(player: Dict, season: Optional[int] = None) -> Tuple[Dict[str, float], int]:
    """
    Extract current season per-game stats from a player's stats dictionary.

    Handles multiple stat formats:
    1. get_all_rosters format: stats.current_season.average, per_game_stats
    2. Legacy ESPN format: stats['{season}_total']['avg']
    3. Top-level games_played field

    Args:
        player: Player dictionary with 'stats' field
        season: The season year (e.g., 2026) for dynamic stat key access.
                If not provided, falls back to trying multiple years.

    Returns:
        Tuple of (stats_dict, games_played)
    """
    stats = player.get('stats', {})
    games_played = 0

    # Log the season being used
    if season:
        logger.debug(f"get_player_current_stats: Using season {season} for stat access")

    # FIRST: Try to get games_played from all possible locations
    # Priority: top-level > current_season > dynamic season key > calculated
    games_played = _extract_games_played(player, stats, season)

    # Strategy 1: Use pre-computed per_game_stats from get_all_rosters (preferred)
    if 'per_game_stats' in player and player['per_game_stats']:
        per_game = player['per_game_stats']
        # Convert engine format to ESPN format for consistency
        espn_stats = {}
        engine_to_espn = {
            'pts': 'PTS', 'reb': 'REB', 'ast': 'AST', 'stl': 'STL', 'blk': 'BLK',
            '3pm': '3PM', 'to': 'TO', 'fgm': 'FGM', 'fga': 'FGA', 'ftm': 'FTM', 'fta': 'FTA',
        }
        for eng_key, espn_key in engine_to_espn.items():
            if eng_key in per_game:
                espn_stats[espn_key] = float(per_game[eng_key])

        # Calculate percentages from makes/attempts
        if espn_stats.get('FGA', 0) > 0:
            espn_stats['FG%'] = espn_stats.get('FGM', 0) / espn_stats['FGA']
        if espn_stats.get('FTA', 0) > 0:
            espn_stats['FT%'] = espn_stats.get('FTM', 0) / espn_stats['FTA']

        # games_played already extracted above, don't overwrite
        if espn_stats:
            return espn_stats, int(games_played)

    # Strategy 2: Use current_season from get_all_rosters format
    if 'current_season' in stats:
        current_season = stats['current_season']
        avg_stats = current_season.get('average', {})
        # games_played already extracted above, don't overwrite

        if avg_stats:
            # Convert ESPN stat IDs to names
            STAT_ID_TO_NAME = {
                '0': 'PTS', '1': 'BLK', '2': 'STL', '3': 'AST',
                '6': 'REB', '17': '3PM', '19': 'TO',
                '13': 'FGM', '14': 'FGA', '15': 'FTM', '16': 'FTA'
            }
            espn_stats = {}
            for stat_id, stat_name in STAT_ID_TO_NAME.items():
                if stat_id in avg_stats:
                    espn_stats[stat_name] = float(avg_stats[stat_id] or 0)

            # Calculate percentages
            if espn_stats.get('FGA', 0) > 0:
                espn_stats['FG%'] = espn_stats.get('FGM', 0) / espn_stats['FGA']
            if espn_stats.get('FTA', 0) > 0:
                espn_stats['FT%'] = espn_stats.get('FTM', 0) / espn_stats['FTA']

            return espn_stats, int(games_played)

    # Strategy 3: Use projected stats for IR players with 0 games
    if 'projected' in stats:
        projected = stats['projected']
        avg_stats = projected.get('average', {})
        if avg_stats:
            STAT_ID_TO_NAME = {
                '0': 'PTS', '1': 'BLK', '2': 'STL', '3': 'AST',
                '6': 'REB', '17': '3PM', '19': 'TO',
                '13': 'FGM', '14': 'FGA', '15': 'FTM', '16': 'FTA'
            }
            espn_stats = {}
            for stat_id, stat_name in STAT_ID_TO_NAME.items():
                if stat_id in avg_stats:
                    espn_stats[stat_name] = float(avg_stats[stat_id] or 0)

            if espn_stats.get('FGA', 0) > 0:
                espn_stats['FG%'] = espn_stats.get('FGM', 0) / espn_stats['FGA']
            if espn_stats.get('FTA', 0) > 0:
                espn_stats['FT%'] = espn_stats.get('FTM', 0) / espn_stats['FTA']

            return espn_stats, 0  # 0 games played for IR players using projected

    # Strategy 4: Legacy format fallback
    # Build dynamic season keys based on provided season, or try common years
    if season:
        # Use provided season for dynamic keys
        season_keys = [
            f'{season}_total',       # e.g., '2026_total'
            f'{season - 1}_total',   # e.g., '2025_total' (previous season)
            f'00{season}',           # e.g., '002026'
            f'00{season - 1}',       # e.g., '002025'
            'total',
        ]
        logger.debug(f"get_player_current_stats: Trying dynamic keys for season {season}: {season_keys}")
    else:
        # Fallback to trying common years (2026, 2025)
        season_keys = ['2026_total', '2025_total', 'total', '002026', '002025']

    for key in season_keys:
        if key in stats and isinstance(stats[key], dict):
            stat_data = stats[key]
            if 'avg' in stat_data:
                per_game = stat_data['avg']
                games = (
                    stat_data.get('GP', 0) or
                    stat_data.get('applied_total', 0) or
                    per_game.get('GP', 0) or
                    per_game.get('G', 0) or
                    0
                )
                logger.debug(f"get_player_current_stats: Found stats in key '{key}' with {games} games")
                return per_game, int(games) if games else 0

    return {}, 0


def convert_espn_stats_to_engine_format(espn_stats: Dict[str, float]) -> Dict[str, float]:
    """
    Convert ESPN stat names to hybrid engine format.

    ESPN uses: PTS, REB, AST, 3PM, FG%, FT%, TO
    Engine uses: pts, trb, ast, 3p, fg_pct, ft_pct, tov

    Args:
        espn_stats: Stats dictionary with ESPN naming

    Returns:
        Stats dictionary with engine naming
    """
    engine_stats = {}
    for espn_name, engine_name in ESPN_TO_ENGINE_STAT_MAP.items():
        if espn_name in espn_stats:
            value = espn_stats[espn_name]
            if value is not None:
                engine_stats[engine_name] = float(value)

    # Also copy over common variants
    stat_aliases = {
        'MP': 'mp', 'MIN': 'mp', 'minutes': 'mp',
        'GP': 'g', 'games': 'g',
        'FG_PCT': 'fg_pct', 'FT_PCT': 'ft_pct',
        '3PA': '3pa',
    }
    for alias, engine_name in stat_aliases.items():
        if alias in espn_stats and engine_name not in engine_stats:
            value = espn_stats[alias]
            if value is not None:
                engine_stats[engine_name] = float(value)

    return engine_stats


def _estimate_games_remaining(season: Optional[int] = None) -> int:
    """
    Estimate games remaining in the NBA season.

    Args:
        season: The season year (e.g., 2026). If not provided, calculates based on current date.
                For a 2025-26 season, pass 2026.
    """
    today = date.today()

    # Determine season year if not provided
    if season is None:
        # NBA season starts in October, so:
        # - If it's Oct-Dec, we're in the season ending next year
        # - If it's Jan-Apr, we're in the season that started last year
        if today.month >= 10:
            season = today.year + 1  # e.g., Oct 2025 -> 2026 season
        else:
            season = today.year  # e.g., Feb 2026 -> 2026 season

    # NBA season typically runs late October to mid-April
    # Season year refers to the ending year (e.g., 2026 means 2025-26 season)
    season_start = date(season - 1, 10, 22)  # e.g., Oct 22, 2025 for 2026 season
    season_end = date(season, 4, 13)         # e.g., Apr 13, 2026 for 2026 season

    if today < season_start:
        return 82
    if today > season_end:
        return 0

    total_days = (season_end - season_start).days
    elapsed_days = (today - season_start).days
    progress = elapsed_days / total_days

    return max(0, int(82 * (1 - progress)))


def calculate_projected_team_totals(
    roster: List[Dict],
    categories: List[Dict],
    team_name: str = "Unknown",
    hybrid_engine: Optional[Any] = None,
    simple_engine: Optional[Any] = None,
    league_id: Optional[int] = None,
    season: Optional[int] = None,
    projection_method: str = 'adaptive',
    flat_game_rate: float = 0.85
) -> Tuple[Dict[str, float], List[Dict]]:
    """
    Calculate REST-OF-SEASON projected totals for a team's roster.

    NOTE: This function calculates ONLY the ROS projections for rostered players.
    The caller should add these to ESPN's current team totals to get end-of-season projections:
        EOS_Total = ESPN_Team_Current_Total + sum(player_ROS_projections)

    For counting stats (PTS, REB, AST, STL, BLK, 3PM, TO):
        Sum each player's ROS projection

    For percentage stats (FG%, FT%):
        These are calculated from ROS makes/attempts only (caller handles combining with current)

    Args:
        roster: List of player dictionaries
        categories: Scoring categories
        team_name: Team name for logging
        hybrid_engine: Optional hybrid projection engine instance
        simple_engine: Optional simple projection engine instance
        league_id: Optional league ID for projections
        season: The season year (e.g., 2026) for dynamic stat access
        projection_method: 'adaptive' or 'flat_rate' for game projections
        flat_game_rate: Fixed rate when using flat_rate method (0.50-1.00)

    Returns:
        Tuple of (ros_totals dict, player_projections list)
    """
    logger.info(f"    Calculating ROS projections for {team_name} ({len(roster)} players)")
    if season:
        logger.info(f"    Using season {season} for projections")

    # Initialize ROS totals (rest-of-season projections only)
    ros_team_totals = {cat['stat_key']: 0.0 for cat in categories}
    ros_team_totals['FGM'] = 0.0
    ros_team_totals['FGA'] = 0.0
    ros_team_totals['FTM'] = 0.0
    ros_team_totals['FTA'] = 0.0

    player_projections = []
    players_projected = 0
    games_remaining_estimate = _estimate_games_remaining(season=season)

    # Counting stats for ROS calculation
    COUNTING_STATS_ENGINE = ['pts', 'trb', 'ast', 'stl', 'blk', '3p', 'tov', 'fgm', 'fga', 'ftm', 'fta']

    for player in roster:
        player_name = player.get('name', 'Unknown')
        player_id = str(player.get('espn_player_id', ''))
        injury_status = player.get('injury_status', 'ACTIVE') or 'ACTIVE'

        # Get current per-game stats and games played
        current_stats, games_played = get_player_current_stats(player, season=season)

        if not current_stats:
            logger.warning(f"      Skipping {player_name}: no current stats extracted")
            logger.debug(f"        Raw player stats keys: {list(player.get('stats', {}).keys())}")
            logger.debug(f"        Has per_game_stats: {bool(player.get('per_game_stats'))}")
            continue

        # Log extracted stats
        logger.debug(f"      {player_name}: Extracted ESPN stats: PTS={current_stats.get('PTS', 0):.1f}, REB={current_stats.get('REB', 0):.1f}, GP={games_played}")

        # Convert to engine format for hybrid engine
        engine_stats = convert_espn_stats_to_engine_format(current_stats)

        # Log converted stats
        logger.debug(f"      {player_name}: Engine stats: pts={engine_stats.get('pts', 0):.1f}, trb={engine_stats.get('trb', 0):.1f}")

        ros_totals = None
        games_projected = 0
        method_used = None  # Track which projection method was used (hybrid/simple/fallback)

        # Strategy 1: Try hybrid projection engine
        if hybrid_engine is not None:
            try:
                # Get expected_return_date from player data (ESPN provides this for injured players)
                expected_return_date = player.get('expected_return_date') or player.get('injury_date')

                player_data = {
                    'player_id': player_id,
                    'name': player_name,
                    'team': player.get('nba_team', 'UNK'),
                    'position': player.get('position', 'N/A'),
                    'games_played': games_played,
                    'stats': player.get('stats', {}),  # Include ESPN stats for projection extraction
                    'expected_return_date': expected_return_date,
                }

                projection = hybrid_engine.project_player(
                    player_id=player_id,
                    league_id=league_id,
                    player_data=player_data,
                    season_stats=engine_stats,
                    injury_status=injury_status,
                    injury_notes=player.get('injury_notes'),
                    expected_return_date=expected_return_date,
                    league_season=season,
                    projection_method=projection_method,
                    flat_game_rate=flat_game_rate,
                )

                ros_totals = projection.ros_totals
                games_projected = projection.games_projected
                method_used = 'hybrid'
                logger.info(f"[CALCULATE_TEAM] {player_name}: games_projected from hybrid={games_projected}")

                # Log what ros_totals contains
                if ros_totals:
                    core_ros = {k: v for k, v in ros_totals.items() if k in ['pts', 'reb', 'trb', 'ast', 'stl', 'blk', '3p', '3pm', 'fgm', 'fga']}
                    logger.debug(f"      {player_name}: ROS totals keys: {list(ros_totals.keys())}")
                    logger.debug(f"      {player_name}: ROS core stats: {core_ros}")
                else:
                    logger.warning(f"      {player_name}: ROS totals is EMPTY from hybrid engine!")

            except Exception as e:
                logger.warning(f"      Hybrid projection failed for {player_name}: {e}")
                import traceback
                logger.debug(traceback.format_exc())

        # Strategy 2: Try simple projection engine
        if ros_totals is None and simple_engine is not None:
            try:
                simple_proj = simple_engine.project_player(
                    player_id=player_id,
                    player_name=player_name,
                    current_stats=current_stats,
                    games_played=games_played,
                    injury_status=injury_status,
                )
                ros_totals = simple_proj.ros_totals
                games_projected = simple_proj.games_projected
                method_used = 'simple'

            except Exception as e:
                logger.debug(f"      Simple projection failed for {player_name}: {e}")

        # Strategy 3: Inline fallback calculation
        if ros_totals is None:
            # Calculate games remaining for this player
            games_remaining = games_remaining_estimate

            # Apply injury reduction
            injury_factors = {
                'ACTIVE': 1.0,
                'PROBABLE': 0.95,
                'QUESTIONABLE': 0.75,
                'DOUBTFUL': 0.4,
                'DAY_TO_DAY': 0.85,
                'DAY': 0.85,
                'GTD': 0.85,
                'OUT': 0.2,
                'INJ_RESERVE': 0.0,
                'INJURED_RESERVE': 0.0,
                'IR': 0.0,
            }
            injury_factor = injury_factors.get(injury_status.upper(), 0.7)

            # Calculate player's historical game rate
            if games_played > 0:
                # Estimate how many team games have been played
                team_games_played = 82 - games_remaining_estimate
                if team_games_played > 0:
                    game_rate = min(1.0, games_played / team_games_played)
                else:
                    game_rate = 0.85
            else:
                game_rate = 0.75  # New player, assume 75%

            games_projected = int(games_remaining * game_rate * injury_factor)

            ros_totals = {}
            for engine_stat, value in engine_stats.items():
                if engine_stat in ['pts', 'trb', 'ast', 'stl', 'blk', '3p', 'tov', 'fgm', 'fga', 'ftm', 'fta']:
                    ros_totals[engine_stat] = value * games_projected
                else:
                    ros_totals[engine_stat] = value

            method_used = 'fallback'

        # Log the ROS projection
        pts_ros = ros_totals.get('pts', 0)
        reb_ros = ros_totals.get('trb', 0)
        logger.debug(f"      {player_name}: {method_used}, GP_remaining={games_projected}, "
                    f"ROS: PTS={pts_ros:.0f}, REB={reb_ros:.0f}")

        # Save player projection details
        player_projections.append({
            'name': player_name,
            'games_played': games_played,
            'games_projected': games_projected,
            'method': method_used,
            'injury_status': injury_status,
            'ros_totals': ros_totals.copy(),
        })

        # Add ROS totals to team ROS totals
        # Handle stat key aliases (hybrid engine may use 'reb'/'trb', '3pm'/'3p', 'to'/'tov')
        STAT_ALIASES = {
            'trb': ['reb', 'trb'],  # rebounds
            'reb': ['reb', 'trb'],
            '3p': ['3pm', '3p'],    # 3-pointers
            '3pm': ['3pm', '3p'],
            'tov': ['to', 'tov'],   # turnovers
            'to': ['to', 'tov'],
        }

        def get_ros_stat(ros_dict, engine_key):
            """Get stat value, trying aliases if primary key not found."""
            val = ros_dict.get(engine_key, 0)
            if val == 0 and engine_key in STAT_ALIASES:
                for alias in STAT_ALIASES[engine_key]:
                    val = ros_dict.get(alias, 0)
                    if val != 0:
                        break
            return val

        for cat in categories:
            stat_key = cat['stat_key']
            engine_stat = ESPN_TO_ENGINE_STAT_MAP.get(stat_key, stat_key.lower())

            if stat_key in COUNTING_STATS_ESPN:
                val = get_ros_stat(ros_totals, engine_stat)
                ros_team_totals[stat_key] += val
                if val > 0:
                    logger.debug(f"        {stat_key}: +{val:.1f} from {player_name}")

        # Track ROS makes/attempts for percentage calculations
        ros_team_totals['FGM'] += ros_totals.get('fgm', 0)
        ros_team_totals['FGA'] += ros_totals.get('fga', 0)
        ros_team_totals['FTM'] += ros_totals.get('ftm', 0)
        ros_team_totals['FTA'] += ros_totals.get('fta', 0)

        players_projected += 1

    # Calculate ROS percentage stats (caller will combine with current)
    if ros_team_totals['FGA'] > 0:
        ros_team_totals['FG%'] = ros_team_totals['FGM'] / ros_team_totals['FGA']
    else:
        ros_team_totals['FG%'] = 0.0  # No ROS FGA

    if ros_team_totals['FTA'] > 0:
        ros_team_totals['FT%'] = ros_team_totals['FTM'] / ros_team_totals['FTA']
    else:
        ros_team_totals['FT%'] = 0.0  # No ROS FTA

    # Log team ROS summary
    logger.info(f"      {team_name} ROS projections: "
               f"PTS={ros_team_totals.get('PTS', 0):.0f}, "
               f"REB={ros_team_totals.get('REB', 0):.0f}, "
               f"AST={ros_team_totals.get('AST', 0):.0f}")

    # Validation: warn if ROS projections are all zeros
    total_counting_stats = sum([
        ros_team_totals.get('PTS', 0),
        ros_team_totals.get('REB', 0),
        ros_team_totals.get('AST', 0),
        ros_team_totals.get('STL', 0),
        ros_team_totals.get('BLK', 0),
        ros_team_totals.get('3PM', 0),
    ])

    if total_counting_stats == 0 and len(player_projections) > 0:
        logger.warning(f"      [WARNING] All ROS projections are ZERO for {team_name}!")
        logger.warning(f"      This may indicate stats extraction issues.")
        logger.warning(f"      Players processed: {len(player_projections)}, but no stats accumulated.")
    elif players_projected == 0:
        logger.warning(f"      [WARNING] No players could be projected for {team_name}")
    else:
        logger.info(f"      [OK] {players_projected} players projected, total ROS PTS: {ros_team_totals.get('PTS', 0):.0f}")

    return ros_team_totals, player_projections


def apply_start_limits_to_projections(
    team_id: int,
    team_name: str,
    roster: List[Dict],
    ros_totals: Dict[str, float],
    player_projections: List[Dict],
    categories: List[Dict],
    start_limit_optimizer: Any,
    include_ir_returns: bool = True
) -> Tuple[Dict[str, float], List[Dict], Dict[str, Any]]:
    """
    Apply position start limits to adjust rest-of-season projections using day-by-day simulation.

    This function uses the start limit optimizer to:
    1. Fetch actual starts-used-per-position from ESPN
    2. Simulate the remaining season day-by-day
    3. Determine which players actually start vs bench based on position limits
    4. Adjust projections: per_game_stats × games_started (not total games available)
    5. Handle IR player returns (optionally)

    Args:
        team_id: ESPN team ID
        team_name: Team name for logging
        roster: Raw roster data with eligible_slots, nba_team, injury info
        ros_totals: Raw ROS totals before start limit adjustment
        player_projections: Individual player projections with per-game stats
        categories: Scoring categories
        start_limit_optimizer: Initialized StartLimitOptimizer instance
        include_ir_returns: Whether to simulate IR player returns

    Returns:
        Tuple of (adjusted_ros_totals, adjusted_player_projections, start_limit_info)
    """
    if start_limit_optimizer is None:
        return ros_totals, player_projections, {'available': False}

    try:
        # Import position mapping
        from backend.projections.start_limit_optimizer import SLOT_ID_TO_POSITION, IR_SLOT_ID

        # Build comprehensive player data for optimizer
        optimizer_roster = []
        for i, player in enumerate(roster):
            player_proj = player_projections[i] if i < len(player_projections) else None
            if player_proj is None:
                continue

            # Get per-game stats from projection (engine format)
            ros_stats = player_proj.get('ros_totals', {})
            projected_games = player_proj.get('games_projected', 0)
            player_name_for_log = player_proj.get('name', player.get('name', 'Unknown'))
            logger.info(f"[DASHBOARD->OPTIMIZER] {player_name_for_log}: projected_games={projected_games}")

            # Calculate per-game stats from ROS totals
            per_game_stats = {}
            if projected_games > 0:
                for engine_stat, total in ros_stats.items():
                    if engine_stat in ['fg_pct', 'ft_pct']:
                        per_game_stats[ENGINE_TO_ESPN_STAT_MAP.get(engine_stat, engine_stat.upper())] = total
                    else:
                        espn_stat = ENGINE_TO_ESPN_STAT_MAP.get(engine_stat, engine_stat.upper())
                        per_game_stats[espn_stat] = total / projected_games

            # Map ROS totals to ESPN format
            projected_stats = {}
            for engine_stat, value in ros_stats.items():
                espn_stat = ENGINE_TO_ESPN_STAT_MAP.get(engine_stat, engine_stat.upper())
                projected_stats[espn_stat] = value

            # Get player details from roster
            lineup_slot_id = player.get('lineupSlotId', player.get('lineup_slot_id', 0))
            injury_status = player.get('injury_status', 'ACTIVE') or 'ACTIVE'
            season_outlook = player.get('season_outlook', '')
            droppable = player.get('droppable', True)

            # Get NBA team - try multiple field names
            nba_team = (
                player.get('nba_team') or
                player.get('pro_team') or
                player.get('team') or
                'UNK'
            )

            optimizer_roster.append({
                'player_id': player.get('espn_player_id', i),
                'name': player_proj.get('name', player.get('name', 'Unknown')),
                'nba_team': nba_team,
                'eligible_slots': player.get('eligible_slots', []),
                'per_game_stats': per_game_stats,
                'projected_stats': projected_stats,
                'projected_games': projected_games,
                'lineupSlotId': lineup_slot_id,
                'injury_status': injury_status,
                'injury_details': player.get('injury_details'),  # ESPN injury data with expected_return_date
                'season_outlook': season_outlook,
                'droppable': droppable,
            })

        if not optimizer_roster:
            logger.warning(f"    No players to optimize for {team_name}")
            return ros_totals, player_projections, {'available': False, 'reason': 'no_players'}

        # Run the day-by-day optimizer
        logger.info(f"    Running day-by-day optimization for {team_name} ({len(optimizer_roster)} players)")
        logger.info(f"      IR returns: {'enabled' if include_ir_returns else 'disabled'}")

        adjusted_totals, assignments, ir_players = start_limit_optimizer.optimize_team_projections(
            team_id=team_id,
            team_name=team_name,
            roster=optimizer_roster,
            categories=categories,
            include_ir_returns=include_ir_returns
        )

        # Fetch actual starts used from ESPN for position limits display
        actual_starts_used = start_limit_optimizer.fetch_starts_used(team_id)
        slot_counts = start_limit_optimizer.get_lineup_slot_counts()
        stat_limits = start_limit_optimizer.get_lineup_slot_stat_limits()

        # Log starts_used for debugging
        logger.info(f"      Actual starts_used from ESPN: {actual_starts_used}")
        logger.info(f"      Slot counts: {slot_counts}")
        logger.info(f"      Stat limits: {stat_limits}")

        if not actual_starts_used:
            logger.warning(f"      [ISSUE 3] starts_used is EMPTY! Check fetch_starts_used method.")
        elif all(v == 0 for v in actual_starts_used.values()):
            logger.warning(f"      [ISSUE 3] All starts_used values are ZERO! This is unexpected.")

        # Build position limits with actual ESPN data
        position_limits = {}
        for slot_id, count in slot_counts.items():
            if count > 0 and slot_id not in [12, 13, 14]:  # Skip bench and IR
                pos_name = SLOT_ID_TO_POSITION.get(slot_id, f'SLOT_{slot_id}')
                limit = stat_limits.get(slot_id, count * 82)
                games_used = actual_starts_used.get(slot_id, 0)

                position_limits[pos_name] = {
                    'slot_id': slot_id,
                    'slots': count,
                    'games_limit': limit,
                    'games_used': games_used,
                    'games_remaining': limit - games_used,
                }
                logger.debug(f"        Position {pos_name}: {games_used}/{limit} starts used")

        # Build adjusted player projections from optimizer assignments (now dicts, not objects)
        adjusted_player_projections = []
        for assignment in assignments:
            # Handle both dict and object formats for compatibility
            if isinstance(assignment, dict):
                adj_proj = {
                    'name': assignment.get('name', 'Unknown'),
                    'nba_team': assignment.get('nba_team', 'UNK'),
                    'assigned_position': assignment.get('assigned_position', 'BE'),
                    'projected_games': assignment.get('projected_games', 0),
                    'actual_games_to_start': assignment.get('actual_games_to_start', 0),
                    'games_benched': assignment.get('games_benched', 0),
                    'start_percentage': assignment.get('start_percentage', 0),
                    'is_benched': assignment.get('is_benched', False),
                    'is_dropped': assignment.get('is_dropped', False),
                    'is_ir_return': assignment.get('is_ir_return', False),
                    'ir_return_date': assignment.get('ir_return_date'),
                    'replacing_player': assignment.get('replacing_player'),
                    'starts_by_position': assignment.get('starts_by_position', {}),
                    'adjusted_stats': assignment.get('adjusted_stats', {}),
                }
            else:
                # Object format (legacy compatibility)
                adj_proj = {
                    'name': getattr(assignment, 'name', 'Unknown'),
                    'assigned_position': getattr(assignment, 'assigned_position', 'BE'),
                    'projected_games': getattr(assignment, 'projected_games', 0),
                    'actual_games_to_start': getattr(assignment, 'actual_games_to_start', 0),
                    'start_percentage': getattr(assignment, 'start_percentage', 0),
                    'is_benched': getattr(assignment, 'is_benched', False),
                    'adjusted_stats': getattr(assignment, 'adjusted_stats', {}),
                }
            adjusted_player_projections.append(adj_proj)

        # Build IR player info for API response
        ir_player_info = []
        for ir_player in ir_players:
            ir_player_info.append({
                'player_id': ir_player.player_id,
                'name': ir_player.player_name,
                'nba_team': ir_player.nba_team,
                'injury_status': ir_player.injury_status,
                'projected_return_date': (
                    ir_player.projected_return_date.isoformat()
                    if ir_player.projected_return_date else None
                ),
                'will_return': ir_player.will_return_before_season_end,
                'games_after_return': ir_player.games_after_return,
                'replacing_player': ir_player.replacing_player_name,
            })

        # Copy adjusted totals to ROS totals format
        adjusted_ros_totals = {}
        for cat in categories:
            stat_key = cat['stat_key']
            if stat_key in adjusted_totals:
                adjusted_ros_totals[stat_key] = adjusted_totals[stat_key]
            elif stat_key in ros_totals:
                adjusted_ros_totals[stat_key] = ros_totals[stat_key]

        # Include FGM/FGA/FTM/FTA for percentage calculations
        for key in ['FGM', 'FGA', 'FTM', 'FTA']:
            if key in adjusted_totals:
                adjusted_ros_totals[key] = adjusted_totals[key]

        # Use percentages from optimizer (already calculated)
        if 'FG%' in adjusted_totals:
            adjusted_ros_totals['FG%'] = adjusted_totals['FG%']
        elif adjusted_ros_totals.get('FGA', 0) > 0:
            adjusted_ros_totals['FG%'] = adjusted_ros_totals['FGM'] / adjusted_ros_totals['FGA']
        else:
            adjusted_ros_totals['FG%'] = 0.0

        if 'FT%' in adjusted_totals:
            adjusted_ros_totals['FT%'] = adjusted_totals['FT%']
        elif adjusted_ros_totals.get('FTA', 0) > 0:
            adjusted_ros_totals['FT%'] = adjusted_ros_totals['FTM'] / adjusted_ros_totals['FTA']
        else:
            adjusted_ros_totals['FT%'] = 0.0

        # Count starting vs benched vs dropped players
        starting_count = sum(1 for a in adjusted_player_projections
                           if not a.get('is_benched') and not a.get('is_dropped'))
        benched_count = sum(1 for a in adjusted_player_projections
                          if a.get('is_benched') and not a.get('is_dropped'))
        dropped_count = sum(1 for a in adjusted_player_projections if a.get('is_dropped'))
        ir_returning_count = sum(1 for a in adjusted_player_projections if a.get('is_ir_return'))

        # Calculate optimization impact
        original_pts = ros_totals.get('PTS', 0)
        adjusted_pts = adjusted_ros_totals.get('PTS', 0)
        reduction_pct = ((1 - adjusted_pts / original_pts) * 100) if original_pts > 0 else 0

        # Log position_limits summary
        if position_limits:
            logger.info(f"      Built position_limits for {len(position_limits)} positions: {list(position_limits.keys())}")
        else:
            logger.warning(f"      [ISSUE 3] position_limits is EMPTY!")

        start_limit_info = {
            'available': True,
            'position_limits': position_limits,
            'starts_used': actual_starts_used,
            'starting_players': starting_count,
            'benched_players': benched_count,
            'dropped_players': dropped_count,
            'ir_returning_players': ir_returning_count,
            'ir_players': ir_player_info,
            'player_assignments': adjusted_player_projections,
            'adjustment_summary': {
                'original_pts': original_pts,
                'adjusted_pts': adjusted_pts,
                'reduction_pct': reduction_pct,
            },
        }

        # Log optimization results
        logger.info(f"      ESPN Starts Used: {actual_starts_used}")
        logger.info(f"      Players: {starting_count} starting, {benched_count} benched, "
                   f"{dropped_count} dropped, {ir_returning_count} IR returns")
        logger.info(f"      Adjusted PTS: {adjusted_pts:.0f} (was {original_pts:.0f}, "
                   f"{reduction_pct:.1f}% reduction)")

        if ir_player_info:
            for ir in ir_player_info:
                if ir['will_return']:
                    logger.info(f"      IR: {ir['name']} returns {ir['projected_return_date']}, "
                               f"replacing {ir['replacing_player']}, +{ir['games_after_return']} games")

        return adjusted_ros_totals, adjusted_player_projections, start_limit_info

    except Exception as e:
        logger.warning(f"    Start limit optimization failed for {team_name}: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return ros_totals, player_projections, {'available': False, 'error': str(e)}


def calculate_projected_roto_standings(
    teams: List[Dict],
    current_ranks: Dict[int, Dict],
    categories: List[Dict],
    espn_client: ESPNClient
) -> List[Dict[str, Any]]:
    """
    Calculate projected end-of-season Roto standings using the projection engine.

    This function performs REAL projections - the projected standings MUST be different
    from current standings as they include rest-of-season (ROS) projections.

    For each team:
    1. Get the team roster from ESPN
    2. Call start_limit_optimizer to run projections with IR handling and start limits
    3. Calculate end-of-season totals = current_stats + optimized_ROS_projections
    4. Rank ALL teams in each of the 8 scoring categories based on projected EOS totals
    5. Convert category ranks to Roto points (1st=10, 2nd=9, etc. for 10-team league)
    6. Calculate total Roto points for each team
    7. Sort teams by total Roto points for projected_standings

    Args:
        teams: List of team data with current stats
        current_ranks: Current category ranks/points for each team (from ESPN)
        categories: Scoring categories (8 standard categories)
        espn_client: Connected ESPN client

    Returns:
        List of teams with projected standings (DIFFERENT from current standings)
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("PROJECTED ROTO STANDINGS CALCULATION - FULL PROJECTION MODE")
    logger.info("=" * 80)
    logger.info(f"Calculating projected END-OF-SEASON standings for {len(teams)} teams")
    logger.info(f"Using season {espn_client.year} for projections")
    logger.info(f"Categories to project: {[c['abbr'] for c in categories]}")
    logger.info(f"Games remaining estimate: {_estimate_games_remaining(season=espn_client.year)}")
    logger.info("")

    num_teams = len(teams)
    projected = []
    all_player_projections = {}  # Store for detailed logging

    # =========================================================================
    # STEP 1: Initialize projection engines
    # =========================================================================
    logger.info("STEP 1: INITIALIZING PROJECTION ENGINES")
    logger.info("-" * 60)
    hybrid_engine = None
    simple_engine = None
    engines_available = []

    if HYBRID_ENGINE_AVAILABLE and HybridProjectionEngine is not None:
        try:
            hybrid_engine = HybridProjectionEngine()
            engines_available.append("Hybrid")
            logger.info("  [OK] Hybrid projection engine initialized")
        except Exception as e:
            logger.warning(f"  [FAIL] Hybrid engine failed: {e}")

    if SIMPLE_ENGINE_AVAILABLE and SimpleProjectionEngine is not None:
        try:
            simple_engine = SimpleProjectionEngine()
            engines_available.append("Simple")
            logger.info("  [OK] Simple projection engine initialized")
        except Exception as e:
            logger.warning(f"  [FAIL] Simple engine failed: {e}")

    if not engines_available:
        engines_available.append("Fallback")
        logger.warning("  [WARN] No projection engine available - using inline fallback calculations")

    logger.info(f"  Available engines: {engines_available}")

    # Initialize start limit optimizer for accurate start limit projections
    start_limit_optimizer = None
    if START_LIMIT_OPTIMIZER_AVAILABLE and StartLimitOptimizer is not None:
        try:
            # Get league credentials for the optimizer
            from backend.models import League as LeagueModel
            # Try both string and int formats for espn_league_id
            league_record = LeagueModel.query.filter_by(
                espn_league_id=str(espn_client.league_id)
            ).first()
            if not league_record:
                league_record = LeagueModel.query.filter_by(
                    espn_league_id=int(espn_client.league_id)
                ).first()

            if league_record:
                # Extract projection settings from league record FIRST
                projection_method = league_record.projection_method or 'adaptive'
                flat_game_rate = league_record.flat_game_rate or 0.85
                logger.info(f"  [OK] Projection settings: method={projection_method}, flat_rate={flat_game_rate}")

                start_limit_optimizer = StartLimitOptimizer(
                    espn_s2=league_record.espn_s2_cookie,
                    swid=league_record.swid_cookie,
                    league_id=espn_client.league_id,
                    season=espn_client.year,
                    verbose=True,  # Enable verbose logging for debugging
                    projection_method=projection_method,
                    flat_game_rate=flat_game_rate,
                )
                logger.info("  [OK] Start limit optimizer initialized with IR handling and projection settings")
            else:
                logger.warning(f"  [FAIL] Could not find league record for espn_league_id={espn_client.league_id}")
                projection_method = 'adaptive'
                flat_game_rate = 0.85
        except Exception as e:
            logger.warning(f"  [FAIL] Start limit optimizer failed: {e}")
            logger.debug(traceback.format_exc())
            projection_method = 'adaptive'
            flat_game_rate = 0.85
    else:
        logger.info("  [WARN] Start limit optimizer not available - projections will not account for position limits")
        projection_method = 'adaptive'
        flat_game_rate = 0.85

    logger.info("")

    # =========================================================================
    # STEP 2: Fetch all team rosters from ESPN
    # =========================================================================
    logger.info("STEP 2: FETCHING ALL TEAM ROSTERS FROM ESPN")
    logger.info("-" * 60)
    try:
        all_rosters = espn_client.get_all_rosters()
        total_players = sum(len(r) for r in all_rosters.values())
        logger.info(f"  Fetched rosters for {len(all_rosters)} teams ({total_players} total players)")
        for team_id, roster in all_rosters.items():
            team_name = next((t.get('team_name', 'Unknown') for t in teams if t['espn_team_id'] == team_id), 'Unknown')
            ir_count = sum(1 for p in roster if p.get('lineupSlotId') == 13)
            logger.info(f"    Team {team_id} ({team_name[:20]}): {len(roster)} players, {ir_count} on IR")
    except Exception as e:
        logger.error(f"  [ERROR] Failed to fetch rosters: {e}")
        logger.error(f"  Falling back to current standings as projected (NO PROJECTIONS)")
        return _fallback_projected_standings(teams, current_ranks, categories, num_teams)

    logger.info("")

    # =========================================================================
    # STEP 3: Calculate projected team totals for EACH team
    # =========================================================================
    logger.info("STEP 3: CALCULATING PROJECTED TEAM TOTALS FOR ALL TEAMS")
    logger.info("-" * 60)
    logger.info("For each team: Current Stats + ROS Projections = End-of-Season Totals")
    logger.info("")
    team_projected_totals = {}

    for team_idx, team in enumerate(teams, 1):
        team_id = team['espn_team_id']
        team_name = team.get('team_name', 'Unknown')
        roster = all_rosters.get(team_id, [])

        logger.info(f"")
        logger.info(f"  [{team_idx}/{num_teams}] TEAM: {team_name} (ESPN ID: {team_id})")
        logger.info(f"  " + "-" * 50)
        logger.info(f"  Roster size: {len(roster)} players")

        # Get ESPN's current team totals (accurate data from ESPN)
        espn_current_totals = team.get('stats', {})
        current_fgm = espn_current_totals.get('FGM', 0) or 0
        current_fga = espn_current_totals.get('FGA', 0) or 0
        current_ftm = espn_current_totals.get('FTM', 0) or 0
        current_fta = espn_current_totals.get('FTA', 0) or 0

        # Log current stats
        logger.info(f"  CURRENT STATS (from ESPN):")
        for cat in categories:
            stat_key = cat['stat_key']
            val = espn_current_totals.get(stat_key, 0) or 0
            if stat_key in ['FG%', 'FT%']:
                logger.info(f"    {stat_key}: {val:.3f}")
            else:
                logger.info(f"    {stat_key}: {val:.1f}")

        if roster:
            # Get ROS projections for all rostered players
            logger.info(f"  CALCULATING ROS PROJECTIONS...")
            ros_totals, player_projections = calculate_projected_team_totals(
                roster=roster,
                categories=categories,
                team_name=team_name,
                hybrid_engine=hybrid_engine,
                simple_engine=simple_engine,
                league_id=None,
                season=espn_client.year,
                projection_method=projection_method,
                flat_game_rate=flat_game_rate,
            )

            # Log individual player projections
            logger.info(f"  PLAYER-BY-PLAYER ROS PROJECTIONS:")
            for pp in player_projections:
                pname = pp.get('name', 'Unknown')[:20]
                method = pp.get('method', 'unknown')
                gp = pp.get('games_played', 0)
                gproj = pp.get('games_projected', 0)
                ros = pp.get('ros_totals', {})
                pts_ros = ros.get('pts', 0)
                reb_ros = ros.get('trb', 0)
                logger.info(f"    {pname}: GP={gp}, Proj Games={gproj}, Method={method}, ROS PTS={pts_ros:.0f}, REB={reb_ros:.0f}")

            # Apply start limits to adjust projections (if optimizer available)
            start_limit_info = {'available': False}
            if start_limit_optimizer is not None:
                logger.info(f"  APPLYING START LIMIT OPTIMIZATION (with IR handling)...")
                ros_totals, player_projections, start_limit_info = apply_start_limits_to_projections(
                    team_id=team_id,
                    team_name=team_name,
                    roster=roster,
                    ros_totals=ros_totals,
                    player_projections=player_projections,
                    categories=categories,
                    start_limit_optimizer=start_limit_optimizer,
                    include_ir_returns=True  # Enable IR return projections
                )

                if start_limit_info.get('available'):
                    adj_summary = start_limit_info.get('adjustment_summary', {})
                    logger.info(f"    Start limits applied: {adj_summary.get('reduction_pct', 0):.1f}% reduction")
                    logger.info(f"    Players: {start_limit_info.get('starting_players', 0)} starting, "
                               f"{start_limit_info.get('benched_players', 0)} benched, "
                               f"{start_limit_info.get('ir_returning_players', 0)} IR returns")
            else:
                logger.info(f"  [WARN] Start limit optimizer not available - using raw projections")

            all_player_projections[team_id] = {
                'players': player_projections,
                'start_limits': start_limit_info,
            }

            # Log ROS totals (after optimization if applied)
            logger.info(f"  ROS PROJECTIONS (rest-of-season totals):")
            for cat in categories:
                stat_key = cat['stat_key']
                val = ros_totals.get(stat_key, 0) or 0
                if stat_key in ['FG%', 'FT%']:
                    logger.info(f"    {stat_key}: {val:.3f}")
                else:
                    logger.info(f"    {stat_key}: {val:.1f}")

            # Calculate END-OF-SEASON totals = ESPN current + ROS projections
            eos_totals = {}
            for cat in categories:
                stat_key = cat['stat_key']
                current_val = espn_current_totals.get(stat_key, 0) or 0
                ros_val = ros_totals.get(stat_key, 0) or 0
                # For percentage stats, we'll recalculate from makes/attempts below
                if stat_key not in ['FG%', 'FT%']:
                    eos_totals[stat_key] = current_val + ros_val

            # Calculate EOS percentage stats: (current_makes + ros_makes) / (current_attempts + ros_attempts)
            ros_fgm = ros_totals.get('FGM', 0) or 0
            ros_fga = ros_totals.get('FGA', 0) or 0
            ros_ftm = ros_totals.get('FTM', 0) or 0
            ros_fta = ros_totals.get('FTA', 0) or 0

            eos_fgm = current_fgm + ros_fgm
            eos_fga = current_fga + ros_fga
            eos_ftm = current_ftm + ros_ftm
            eos_fta = current_fta + ros_fta

            eos_totals['FGM'] = eos_fgm
            eos_totals['FGA'] = eos_fga
            eos_totals['FTM'] = eos_ftm
            eos_totals['FTA'] = eos_fta
            eos_totals['FG%'] = eos_fgm / eos_fga if eos_fga > 0 else 0.45
            eos_totals['FT%'] = eos_ftm / eos_fta if eos_fta > 0 else 0.78

            # Log END-OF-SEASON totals
            logger.info(f"  END-OF-SEASON TOTALS (Current + ROS):")
            for cat in categories:
                stat_key = cat['stat_key']
                curr = espn_current_totals.get(stat_key, 0) or 0
                ros = ros_totals.get(stat_key, 0) or 0
                eos = eos_totals.get(stat_key, 0) or 0
                if stat_key in ['FG%', 'FT%']:
                    logger.info(f"    {stat_key}: {curr:.3f} + ROS -> {eos:.3f}")
                else:
                    logger.info(f"    {stat_key}: {curr:.0f} + {ros:.0f} = {eos:.0f}")

        else:
            logger.warning(f"  [WARN] No roster for {team_name}, using ESPN current as projection (NO PROJECTION)")
            ros_totals = {}  # No ROS projections when no roster
            eos_totals = {}
            for cat in categories:
                stat_key = cat['stat_key']
                eos_totals[stat_key] = espn_current_totals.get(stat_key, 0) or 0
            eos_totals['FGM'] = current_fgm
            eos_totals['FGA'] = current_fga
            eos_totals['FTM'] = current_ftm
            eos_totals['FTA'] = current_fta
            start_limit_info = {'available': False, 'reason': 'no_roster'}
            player_projections = []  # No player projections

        team_projected_totals[team_id] = {
            'team_name': team_name,
            'espn_current': espn_current_totals,
            'ros_totals': ros_totals,
            'totals': eos_totals,  # End-of-season totals = Current + ROS
            'start_limits': start_limit_info,
        }

    logger.info("")

    # =========================================================================
    # STEP 4: Compare Current vs Projected End-of-Season Totals
    # =========================================================================
    logger.info("STEP 4: CURRENT VS PROJECTED END-OF-SEASON COMPARISON")
    logger.info("-" * 60)
    logger.info("Showing how each team's stats will change by end of season")
    logger.info("")

    # Create a comprehensive comparison table
    header = f"  {'Team':<22} | {'Stat':<5} | {'Current':>10} | {'EOS Proj':>10} | {'Change':>10}"
    logger.info(header)
    logger.info("  " + "-" * 70)

    for team in teams:
        team_id = team['espn_team_id']
        team_name = team.get('team_name', 'Unknown')[:22]
        current_values = current_ranks.get(team_id, {}).get('category_values', {})
        projected_totals = team_projected_totals[team_id]['totals']

        for cat in categories:
            stat_key = cat['stat_key']
            current_val = current_values.get(stat_key, 0) or 0
            projected_val = projected_totals.get(stat_key, 0) or 0
            diff = projected_val - current_val

            if stat_key in ['FG%', 'FT%']:
                diff_str = f"+{diff:.3f}" if diff >= 0 else f"{diff:.3f}"
                logger.info(f"  {team_name:<22} | {stat_key:<5} | {current_val:>10.3f} | {projected_val:>10.3f} | {diff_str:>10}")
            else:
                diff_str = f"+{diff:.0f}" if diff >= 0 else f"{diff:.0f}"
                logger.info(f"  {team_name:<22} | {stat_key:<5} | {current_val:>10.1f} | {projected_val:>10.1f} | {diff_str:>10}")
        logger.info("  " + "-" * 70)

    logger.info("")

    # =========================================================================
    # STEP 5: Rank ALL teams in each PROJECTED category (EOS totals)
    # =========================================================================
    logger.info("STEP 5: RANKING TEAMS IN EACH PROJECTED CATEGORY")
    logger.info("-" * 60)
    logger.info("Ranking based on END-OF-SEASON projected totals (not current stats)")
    logger.info(f"Roto points: 1st place = {num_teams} points, 2nd = {num_teams-1}, ... last = 1 point")
    logger.info("")

    team_projected_ranks = {team_id: {'category_ranks': {}, 'category_points': {}} for team_id in team_projected_totals}

    for cat in categories:
        stat_key = cat['stat_key']
        abbr = cat['abbr']
        lower_is_better = cat.get('lower_is_better', False)

        logger.info(f"  CATEGORY: {abbr} ({stat_key}) - {'Lower is better' if lower_is_better else 'Higher is better'}")

        # Get EOS projected values for all teams
        team_values = []
        for team_id, data in team_projected_totals.items():
            value = data['totals'].get(stat_key, 0) or 0
            team_values.append({
                'team_id': team_id,
                'team_name': data['team_name'],
                'value': value,
                'current_value': current_ranks.get(team_id, {}).get('category_values', {}).get(stat_key, 0) or 0
            })

        # Sort teams by EOS projected value (best first)
        team_values.sort(
            key=lambda x: x['value'],
            reverse=(not lower_is_better)
        )

        # Log full category ranking
        logger.info(f"    Rank | Team                   | Current    | EOS Proj   | Roto Pts")
        logger.info(f"    " + "-" * 65)

        for rank, tv in enumerate(team_values, 1):
            team_id = tv['team_id']
            roto_points = num_teams - rank + 1

            # Store rank and points
            team_projected_ranks[team_id]['category_ranks'][stat_key] = rank
            team_projected_ranks[team_id]['category_points'][stat_key] = roto_points

            # Log the ranking
            team_name_display = tv['team_name'][:22]
            current_val = tv['current_value']
            eos_val = tv['value']

            if stat_key in ['FG%', 'FT%']:
                logger.info(f"    {rank:>4} | {team_name_display:<22} | {current_val:>10.3f} | {eos_val:>10.3f} | {roto_points:>8}")
            else:
                logger.info(f"    {rank:>4} | {team_name_display:<22} | {current_val:>10.0f} | {eos_val:>10.0f} | {roto_points:>8}")

        logger.info("")

    # =========================================================================
    # STEP 6: Calculate TOTAL Roto points for each team
    # =========================================================================
    logger.info("STEP 6: CALCULATING TOTAL ROTO POINTS FOR EACH TEAM")
    logger.info("-" * 60)
    logger.info("Total = Sum of Roto points across all categories")
    logger.info("")

    header = f"  {'Team':<25} | {'Current':>8} | {'Projected':>8} | {'Change':>8} | Category Breakdown"
    logger.info(header)
    logger.info("  " + "-" * 90)

    for team_id, ranks_data in team_projected_ranks.items():
        total_points = sum(ranks_data['category_points'].values())
        ranks_data['total_points'] = total_points

        team_name = team_projected_totals[team_id]['team_name'][:25]
        current_pts = current_ranks.get(team_id, {}).get('total_points', 0)
        diff = total_points - current_pts
        diff_str = f"+{diff}" if diff > 0 else str(diff)

        # Build category breakdown string
        cat_breakdown = " ".join([f"{c['abbr']}:{ranks_data['category_points'].get(c['stat_key'], 0)}" for c in categories])

        logger.info(f"  {team_name:<25} | {current_pts:>8} | {total_points:>8} | {diff_str:>8} | {cat_breakdown}")

    logger.info("")

    # Validate that projections are different from current
    total_changes = sum(
        abs(team_projected_ranks[t]['total_points'] - current_ranks.get(t, {}).get('total_points', 0))
        for t in team_projected_ranks
    )
    if total_changes == 0:
        logger.warning("  [WARNING] PROJECTED ROTO POINTS ARE IDENTICAL TO CURRENT!")
        logger.warning("  This indicates projections may not be calculating correctly.")
    else:
        logger.info(f"  [OK] Projections show changes from current standings (total delta: {total_changes} points)")

    logger.info("")

    # =========================================================================
    # STEP 7: Build projected standings response
    # =========================================================================
    logger.info("STEP 7: BUILDING PROJECTED STANDINGS RESPONSE")
    logger.info("-" * 60)

    for team in teams:
        team_id = team['espn_team_id']
        current_data = current_ranks.get(team_id, {})
        projected_data = team_projected_ranks.get(team_id, {})

        current_total_points = current_data.get('total_points', 0)
        projected_total_points = projected_data.get('total_points', 0)

        # Build categories display dict - projected POINTS (10 for 1st, 9 for 2nd, etc.)
        categories_display = {}
        projected_values = {}
        totals = team_projected_totals[team_id]['totals']
        for cat in categories:
            stat_key = cat['stat_key']
            categories_display[stat_key] = projected_data.get('category_points', {}).get(stat_key, 0)
            projected_values[stat_key] = totals.get(stat_key, 0)

        # Include FGM/FGA/FTM/FTA for percentage verification
        projected_values['FGM'] = totals.get('FGM', 0)
        projected_values['FGA'] = totals.get('FGA', 0)
        projected_values['FTM'] = totals.get('FTM', 0)
        projected_values['FTA'] = totals.get('FTA', 0)

        # Calculate win probability based on projected points
        max_possible_points = num_teams * len(categories)
        if max_possible_points > 0:
            win_probability = (projected_total_points / max_possible_points) * 100
        else:
            win_probability = 50

        # Get start limit info for this team
        team_start_limits = team_projected_totals[team_id].get('start_limits', {})

        projected.append({
            'espn_team_id': team_id,
            'team_name': team_projected_totals[team_id]['team_name'],
            'owner_name': team.get('owner_name', 'Unknown'),
            # PRIMARY DISPLAY: Projected points earned in each category
            'categories': categories_display,
            # Projected category values (raw totals for tooltips)
            'projected_category_values': projected_values,
            # Current values for comparison
            'current_category_values': current_data.get('category_values', {}),
            # ROS projections (what was added)
            'ros_projections': team_projected_totals[team_id].get('ros_totals', {}),
            # Totals
            'current_total_points': current_total_points,
            'projected_total_points': projected_total_points,
            'current_roto_points': current_total_points,
            'projected_roto_points': projected_total_points,
            # Reference data
            'current_category_ranks': current_data.get('category_ranks', {}),
            'current_category_points': current_data.get('category_points', {}),
            'projected_category_ranks': projected_data.get('category_ranks', {}),
            'projected_category_points': projected_data.get('category_points', {}),
            'win_probability': round(min(95, max(5, win_probability)), 1),
            # Start limit information
            'start_limits': team_start_limits,
        })

    # Sort by projected total points (higher is better)
    projected.sort(key=lambda x: x['projected_total_points'], reverse=True)

    # Assign projected standings and calculate movement
    current_order = sorted(projected, key=lambda x: x['current_total_points'], reverse=True)
    current_standing_map = {t['espn_team_id']: idx + 1 for idx, t in enumerate(current_order)}

    logger.info("")

    # =========================================================================
    # STEP 8: FINAL PROJECTED STANDINGS WITH MOVEMENT
    # =========================================================================
    logger.info("STEP 8: FINAL PROJECTED STANDINGS (SORTED BY PROJECTED ROTO POINTS)")
    logger.info("-" * 60)
    logger.info("")

    # Build comprehensive final standings table
    header = f"  {'Proj':<5} | {'Team':<25} | {'Proj Pts':>8} | {'Curr Pts':>8} | {'Curr Rk':>7} | {'Move':>6}"
    logger.info(header)
    logger.info("  " + "-" * 75)

    for i, team in enumerate(projected, 1):
        team['projected_standing'] = i
        team['current_standing'] = current_standing_map.get(team['espn_team_id'], i)
        team['movement'] = team['current_standing'] - i  # Positive = improving

        movement_str = f"+{team['movement']}" if team['movement'] > 0 else str(team['movement'])
        movement_indicator = "↑" if team['movement'] > 0 else ("↓" if team['movement'] < 0 else "→")

        logger.info(f"  {i:<5} | {team['team_name'][:25]:<25} | {team['projected_total_points']:>8} | "
                   f"{team['current_total_points']:>8} | {team['current_standing']:>7} | {movement_str:>4} {movement_indicator}")

    logger.info("")
    logger.info("  " + "-" * 75)

    # Summary statistics
    teams_improving = sum(1 for t in projected if t['movement'] > 0)
    teams_declining = sum(1 for t in projected if t['movement'] < 0)
    teams_same = sum(1 for t in projected if t['movement'] == 0)

    logger.info(f"  MOVEMENT SUMMARY: {teams_improving} teams improving, {teams_declining} declining, {teams_same} unchanged")

    # Validate projections are different from current
    current_standings_order = [t['espn_team_id'] for t in sorted(projected, key=lambda x: x['current_standing'])]
    projected_standings_order = [t['espn_team_id'] for t in projected]

    if current_standings_order == projected_standings_order:
        logger.warning("")
        logger.warning("  [WARNING] PROJECTED STANDINGS ORDER IS IDENTICAL TO CURRENT ORDER!")
        logger.warning("  This may indicate projections are not being calculated properly.")
        logger.warning("  Check that ROS projections are producing meaningful values.")
    else:
        logger.info("")
        logger.info("  [OK] Projected standings differ from current standings")

    logger.info("")
    logger.info("=" * 80)
    logger.info("PROJECTED ROTO STANDINGS CALCULATION COMPLETE")
    logger.info(f"Processed {num_teams} teams, {len(categories)} categories")
    logger.info(f"Start limit optimization: {'ENABLED' if any(t.get('start_limits', {}).get('available') for t in projected) else 'NOT AVAILABLE'}")
    logger.info("=" * 80)
    logger.info("")

    return projected


def _fallback_projected_standings(
    teams: List[Dict],
    current_ranks: Dict[int, Dict],
    categories: List[Dict],
    num_teams: int
) -> List[Dict[str, Any]]:
    """
    Fallback: use current standings as projection when rosters unavailable.

    WARNING: This means NO PROJECTIONS are being calculated - projected standings
    will be identical to current standings.
    """
    logger.warning("")
    logger.warning("=" * 80)
    logger.warning("FALLBACK MODE: USING CURRENT STANDINGS AS PROJECTED STANDINGS")
    logger.warning("=" * 80)
    logger.warning("This means NO PROJECTIONS are being calculated!")
    logger.warning("Projected standings will be IDENTICAL to current standings.")
    logger.warning("Check roster fetching and projection engine initialization.")
    logger.warning("")

    projected = []
    for team in teams:
        team_id = team['espn_team_id']
        ranks_data = current_ranks.get(team_id, {})

        current_total_points = ranks_data.get('total_points', 0)

        categories_display = {}
        for cat in categories:
            stat_key = cat['stat_key']
            categories_display[stat_key] = ranks_data.get('category_points', {}).get(stat_key, 0)

        max_possible_points = num_teams * len(categories)
        win_probability = (current_total_points / max_possible_points) * 100 if max_possible_points > 0 else 50

        projected.append({
            'espn_team_id': team_id,
            'team_name': ranks_data.get('team_name', team.get('team_name', 'Unknown')),
            'owner_name': team.get('owner_name', 'Unknown'),
            'categories': categories_display,
            'current_total_points': current_total_points,
            'projected_total_points': current_total_points,
            'current_roto_points': current_total_points,
            'projected_roto_points': current_total_points,
            'projected_category_ranks': ranks_data.get('category_ranks', {}),
            'projected_category_points': ranks_data.get('category_points', {}),
            'win_probability': round(min(95, max(5, win_probability)), 1),
        })

    projected.sort(key=lambda x: x['projected_total_points'], reverse=True)

    current_order = sorted(projected, key=lambda x: x['current_total_points'], reverse=True)
    current_standing_map = {t['espn_team_id']: idx + 1 for idx, t in enumerate(current_order)}

    for i, team in enumerate(projected, 1):
        team['projected_standing'] = i
        team['current_standing'] = current_standing_map.get(team['espn_team_id'], i)
        team['movement'] = team['current_standing'] - i

    return projected


def calculate_projected_standings(
    teams: List[Dict],
    league_type: str,
    espn_client: ESPNClient
) -> List[Dict[str, Any]]:
    """
    Calculate projected end-of-season standings for H2H leagues.

    Args:
        teams: List of team data from ESPN
        league_type: H2H_CATEGORY, H2H_POINTS, or ROTO
        espn_client: Connected ESPN client

    Returns:
        List of teams with projected standings and win probabilities
    """
    projected = []

    # Get current standings
    standings = espn_client.get_standings()
    standings_map = {s['espn_team_id']: s for s in standings}

    for team in teams:
        team_id = team['espn_team_id']
        standing_data = standings_map.get(team_id, {})

        current_rank = standing_data.get('standing', 0)
        wins = team.get('wins', 0)
        losses = team.get('losses', 0)
        total_games = wins + losses

        # For H2H: calculate playoff probability based on win rate
        if total_games > 0:
            win_rate = wins / total_games
            win_probability = min(95, max(5, win_rate * 100))
        else:
            win_probability = 50.0

        projected_standing = current_rank

        projected.append({
            'espn_team_id': team_id,
            'team_name': team.get('team_name', 'Unknown'),
            'owner_name': team.get('owner_name', 'Unknown'),
            'current_standing': current_rank,
            'projected_standing': projected_standing,
            'current_record': f"{wins}-{losses}",
            'win_probability': round(win_probability, 1),
            'playoff_probability': round(win_probability, 1),
            'movement': 0
        })

    # Sort by projected standing
    projected.sort(key=lambda x: x['current_standing'])

    # Calculate movement
    for i, team in enumerate(projected, 1):
        team['projected_standing'] = i
        team['movement'] = team['current_standing'] - i

    return projected


def get_category_analysis(
    roster: List[Dict],
    league_type: str,
    season: Optional[int] = None
) -> Dict[str, Any]:
    """
    Analyze team's category strengths and weaknesses.

    Args:
        roster: Team roster with player stats
        league_type: League scoring type
        season: The season year (e.g., 2026) for dynamic stat access

    Returns:
        Dict with strengths, weaknesses, and category totals
    """
    # Category totals
    categories = {
        'pts': 0, 'reb': 0, 'ast': 0, 'stl': 0, 'blk': 0,
        '3pm': 0, 'fg_pct': [], 'ft_pct': [], 'to': 0
    }

    player_count = 0

    # Build dynamic season keys
    if season:
        season_keys = [f'{season}_total', f'{season - 1}_total', 'total']
    else:
        season_keys = ['2026_total', '2025_total', 'total']

    for player in roster:
        stats = player.get('stats', {})

        # Get current season stats
        season_stats = None
        for key in season_keys:
            if key in stats and isinstance(stats[key], dict):
                if 'avg' in stats[key]:
                    season_stats = stats[key]['avg']
                    break

        if not season_stats:
            continue

        player_count += 1

        # Sum counting stats
        categories['pts'] += season_stats.get('PTS', 0) or 0
        categories['reb'] += season_stats.get('REB', 0) or 0
        categories['ast'] += season_stats.get('AST', 0) or 0
        categories['stl'] += season_stats.get('STL', 0) or 0
        categories['blk'] += season_stats.get('BLK', 0) or 0
        categories['3pm'] += season_stats.get('3PM', 0) or 0
        categories['to'] += season_stats.get('TO', 0) or 0

        # Collect percentages for averaging
        fg_pct = season_stats.get('FG%', 0)
        ft_pct = season_stats.get('FT%', 0)
        if fg_pct:
            categories['fg_pct'].append(fg_pct)
        if ft_pct:
            categories['ft_pct'].append(ft_pct)

    # Calculate average percentages
    if categories['fg_pct']:
        categories['fg_pct'] = sum(categories['fg_pct']) / len(categories['fg_pct'])
    else:
        categories['fg_pct'] = 0

    if categories['ft_pct']:
        categories['ft_pct'] = sum(categories['ft_pct']) / len(categories['ft_pct'])
    else:
        categories['ft_pct'] = 0

    # Determine strengths and weaknesses
    thresholds = {
        'pts': {'strong': 18, 'weak': 12},
        'reb': {'strong': 6, 'weak': 4},
        'ast': {'strong': 5, 'weak': 3},
        'stl': {'strong': 1.2, 'weak': 0.8},
        'blk': {'strong': 1.0, 'weak': 0.5},
        '3pm': {'strong': 2.0, 'weak': 1.0},
        'fg_pct': {'strong': 0.48, 'weak': 0.44},
        'ft_pct': {'strong': 0.80, 'weak': 0.72},
        'to': {'strong': 2.0, 'weak': 3.5}
    }

    strengths = []
    weaknesses = []

    if player_count > 0:
        for cat, values in thresholds.items():
            if cat == 'to':
                avg = categories[cat] / player_count
                if avg < values['strong']:
                    strengths.append(cat.upper() if cat != '3pm' else '3PM')
                elif avg > values['weak']:
                    weaknesses.append(cat.upper() if cat != '3pm' else '3PM')
            elif cat in ['fg_pct', 'ft_pct']:
                val = categories[cat]
                if val > values['strong']:
                    strengths.append(cat.upper().replace('_', ''))
                elif val < values['weak']:
                    weaknesses.append(cat.upper().replace('_', ''))
            else:
                avg = categories[cat] / player_count
                if avg > values['strong']:
                    strengths.append(cat.upper() if cat != '3pm' else '3PM')
                elif avg < values['weak']:
                    weaknesses.append(cat.upper() if cat != '3pm' else '3PM')

    return {
        'strengths': strengths[:3],
        'weaknesses': weaknesses[:3],
        'totals': {k: round(v, 2) if isinstance(v, float) else v for k, v in categories.items()},
        'player_count': player_count
    }


def get_waiver_targets(
    espn_client: ESPNClient,
    team_weaknesses: List[str],
    limit: int = 3,
    season: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Get top waiver wire targets based on team needs.

    Args:
        espn_client: ESPN client instance
        team_weaknesses: List of weakness categories
        limit: Number of targets to return
        season: The season year (e.g., 2026) for dynamic stat access
    """
    try:
        free_agents = espn_client.get_free_agents(size=50)
    except Exception as e:
        logger.warning(f"Could not fetch free agents: {e}")
        return []

    targets = []
    weakness_cats = [w.lower() for w in team_weaknesses]

    # Build dynamic season keys
    if season:
        season_keys = [f'{season}_total', f'{season - 1}_total', 'total']
    else:
        season_keys = ['2026_total', '2025_total', 'total']

    for player in free_agents[:limit * 3]:
        player_stats = player.get('stats', {})

        season_stats = None
        for key in season_keys:
            if key in player_stats and isinstance(player_stats[key], dict):
                if 'avg' in player_stats[key]:
                    season_stats = player_stats[key]['avg']
                    break

        if not season_stats:
            continue

        impact_score = 50
        percent_owned = player.get('percent_owned', 0) or 0

        if percent_owned > 50:
            impact_score += 20
        elif percent_owned > 25:
            impact_score += 10

        # Build reason string
        reasons = []
        stat_map = {'pts': 'PTS', 'reb': 'REB', 'ast': 'AST', 'stl': 'STL', 'blk': 'BLK', '3pm': '3PM'}

        for weakness in weakness_cats:
            if weakness in stat_map:
                stat_val = season_stats.get(stat_map[weakness], 0) or 0
                if stat_val > 0:
                    impact_score += 15
                    reasons.append(f"Strong {stat_map[weakness]}")

        targets.append({
            'id': player.get('espn_player_id'),
            'name': player.get('name'),
            'position': player.get('position'),
            'nba_team': player.get('nba_team'),
            'percent_owned': percent_owned,
            'impact_score': min(100, impact_score),
            'reason': ', '.join(reasons[:2]) if reasons else 'Available pickup',
            'trending': 'up' if percent_owned > 30 else None,
        })

    targets.sort(key=lambda x: x['impact_score'], reverse=True)
    return targets[:limit]


def get_trade_opportunities(
    espn_client: ESPNClient,
    my_team_id: int,
    strengths: List[str],
    weaknesses: List[str],
    limit: int = 2,
    season: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Identify potential trade opportunities.

    Args:
        espn_client: ESPN client instance
        my_team_id: User's team ID to exclude from targets
        strengths: List of strength categories
        weaknesses: List of weakness categories
        limit: Number of opportunities to return
        season: The season year (e.g., 2026) for dynamic stat access
    """
    opportunities = []

    if not strengths or not weaknesses:
        return opportunities

    try:
        all_rosters = espn_client.get_all_rosters()
        teams = espn_client.get_teams()
        team_names = {t['espn_team_id']: t['team_name'] for t in teams}
    except Exception as e:
        logger.warning(f"Could not fetch rosters for trade analysis: {e}")
        return []

    strength_cats = [s.lower() for s in strengths]
    weakness_cats = [w.lower() for w in weaknesses]

    # Build dynamic season keys
    if season:
        season_keys = [f'{season}_total', f'{season - 1}_total', 'total']
    else:
        season_keys = ['2026_total', '2025_total', 'total']

    for team_id, roster in all_rosters.items():
        if team_id == my_team_id:
            continue

        for player in roster:
            stats = player.get('stats', {})
            season_stats = None

            for key in season_keys:
                if key in stats and isinstance(stats[key], dict):
                    if 'avg' in stats[key]:
                        season_stats = stats[key]['avg']
                        break

            if not season_stats:
                continue

            helps_weakness = False
            for weakness in weakness_cats:
                stat_map = {'pts': 'PTS', 'reb': 'REB', 'ast': 'AST', 'stl': 'STL', 'blk': 'BLK', '3pm': '3PM'}
                if weakness in stat_map:
                    val = season_stats.get(stat_map[weakness], 0) or 0
                    threshold = 15 if weakness == 'pts' else 5 if weakness == 'reb' else 4
                    if val > threshold:
                        helps_weakness = True
                        break

            if helps_weakness:
                opportunities.append({
                    'target_player': player.get('name'),
                    'target_team': team_names.get(team_id, 'Unknown'),
                    'target_team_id': team_id,
                    'reason': f"Addresses {weakness_cats[0].upper()} weakness",
                    'benefit': f"Improves your {weakness_cats[0].upper()}"
                })

    return opportunities[:limit]


# =============================================================================
# API Endpoints
# =============================================================================

@dashboard_bp.route('/leagues/<int:league_id>/dashboard', methods=['GET'])
@login_required
def get_dashboard(league_id: int):
    """
    Get comprehensive dashboard data for a league.

    Returns different data structures for H2H vs Roto leagues:

    H2H Response:
        - current_standings: Teams with records and ranks
        - projected_standings: Teams with playoff probabilities

    Roto Response:
        - current_standings: Teams with category ranks and total points
        - projected_standings: Teams with projected category ranks
        - scoring_categories: List of enabled categories
        - category_data: All teams' values for each category (for charts)
    """
    # Track current step for error reporting
    current_step = "initialization"
    user_id = current_user.id if current_user else None

    logger.info(f"=== DASHBOARD REQUEST START === league_id={league_id}, user_id={user_id}")

    try:
        # Step 1: Get league from database
        current_step = "fetching league from database"
        logger.info(f"Step 1: {current_step}")
        league = get_league_or_404(league_id, current_user.id)
        logger.info(f"  League found: {league.league_name} (type: {league.league_type})")

        # Step 2: Check cache
        current_step = "checking cache"
        logger.debug(f"Step 2: {current_step}")
        force_refresh = request.args.get('refresh', 'false').lower() == 'true'
        cache = get_cache()
        cache_key = cache.league_key(league_id, 'dashboard')

        if not force_refresh:
            cached = cache.get(cache_key)
            if cached:
                logger.info(f"  Returning cached dashboard data")
                return jsonify(cached), 200

        # Step 3: Connect to ESPN
        current_step = "connecting to ESPN API"
        logger.info(f"Step 3: {current_step}")
        espn_client = get_espn_client_for_league(league_id, current_user.id)
        logger.info(f"  ESPN client created successfully")

        # Step 4: Fetch teams from ESPN
        current_step = "fetching teams from ESPN"
        logger.info(f"Step 4: {current_step}")
        teams = espn_client.get_teams()
        logger.info(f"  Fetched {len(teams)} teams")

        # Step 5: Fetch standings from ESPN
        current_step = "fetching standings from ESPN"
        logger.info(f"Step 5: {current_step}")
        standings = espn_client.get_standings()
        logger.info(f"  Fetched standings for {len(standings)} teams")

        # Step 6: Identify user's team
        current_step = "identifying user's team"
        logger.info(f"Step 6: {current_step}")
        my_team_id = request.args.get('team_id', type=int)
        if not my_team_id:
            # Use SWID matching to identify the user's team
            my_team_id = espn_client.get_user_team_id()
            if my_team_id:
                logger.info(f"Identified user's team via SWID matching: team_id={my_team_id}")
            else:
                logger.warning("Could not identify user's team via SWID matching")
                # Fallback: find team marked as is_user_team from get_teams()
                user_team = next((t for t in teams if t.get('is_user_team')), None)
                if user_team:
                    my_team_id = user_team['espn_team_id']
                    logger.info(f"Found user team from teams list: team_id={my_team_id}")

        # Log the identified team
        if my_team_id:
            user_team_name = next(
                (t.get('team_name', 'Unknown') for t in teams if t.get('espn_team_id') == my_team_id),
                'Unknown'
            )
            logger.info(f"Dashboard for league {league_id}: User's team is '{user_team_name}' (ID: {my_team_id})")
        else:
            logger.warning(f"Dashboard for league {league_id}: Could not identify user's team")

        is_roto = league.league_type == 'ROTO'
        logger.info(f"  League type: {'ROTO' if is_roto else 'H2H'}")

        # Step 7: Get scoring categories
        current_step = "fetching scoring categories"
        logger.info(f"Step 7: {current_step}")
        scoring_categories = get_league_scoring_categories(league, espn_client)
        enabled_categories = [c for c in scoring_categories if c.get('enabled', True)]
        logger.info(f"  Found {len(enabled_categories)} scoring categories: {[c['abbr'] for c in enabled_categories]}")

        if is_roto:
            # =====================================================
            # ROTO LEAGUE RESPONSE
            # =====================================================
            logger.info("Processing as ROTO league")

            # Step 8: Calculate category ranks
            current_step = "calculating Roto category ranks"
            logger.info(f"Step 8: {current_step}")
            category_ranks = calculate_roto_category_ranks(teams, enabled_categories, espn_client)
            logger.info(f"  Calculated ranks for {len(category_ranks)} teams")

            # Step 9: Build current standings
            current_step = "building current standings"
            logger.info(f"Step 9: {current_step}")
            # NOTE: For display, use 'categories' dict which contains POINTS (10 for 1st, 9 for 2nd, etc.)
            # The 'category_ranks', 'category_values', and 'category_points' are kept for reference
            current_standings = []
            for team in teams:
                team_id = team['espn_team_id']
                ranks_data = category_ranks.get(team_id, {})

                # Build the 'categories' dict for easy display - POINTS earned in each category
                # This is what should be shown in the standings table columns
                categories_display = {}
                for cat in enabled_categories:
                    stat_key = cat['stat_key']
                    categories_display[stat_key] = ranks_data.get('category_points', {}).get(stat_key, 0)

                standing_entry = {
                    'team_id': team_id,
                    'espn_team_id': team_id,
                    'team_name': ranks_data.get('team_name', team.get('team_name', 'Unknown')),
                    'owner_name': team.get('owner_name', 'Unknown'),
                    'is_user_team': team_id == my_team_id,
                    # PRIMARY DISPLAY: Points earned in each category (10 for 1st, 9 for 2nd, etc.)
                    'categories': categories_display,
                    # Reference data (kept for tooltips/details):
                    'category_ranks': ranks_data.get('category_ranks', {}),      # Rank 1-10 in each cat
                    'category_values': ranks_data.get('category_values', {}),    # Raw stat totals
                    'category_points': ranks_data.get('category_points', {}),    # Same as 'categories'
                    # Total column
                    'total_points': ranks_data.get('total_points', 0),
                    'roto_points': ranks_data.get('roto_points', 0),  # Alias for total_points
                }
                current_standings.append(standing_entry)

            # Sort by total_points (higher is better - this is the sum of category points)
            current_standings.sort(key=lambda x: x['total_points'], reverse=True)

            # Assign overall ranks based on total points
            for i, team in enumerate(current_standings, 1):
                team['rank'] = i

            # Log the final standings
            logger.info("Final Roto standings for dashboard:")
            for team in current_standings:
                logger.debug(f"  Rank {team['rank']}: {team['team_name']} - {team['total_points']} points"
                            f" (user_team={team['is_user_team']})")
            logger.info(f"  Built current standings with {len(current_standings)} teams")

            # Step 10: Calculate projected standings
            current_step = "calculating projected Roto standings"
            logger.info(f"Step 10: {current_step}")
            projected_standings = calculate_projected_roto_standings(
                teams, category_ranks, enabled_categories, espn_client
            )
            logger.info(f"  Calculated projections for {len(projected_standings)} teams")

            # Add user team flag to projected
            for team in projected_standings:
                team['team_id'] = team['espn_team_id']
                team['is_user_team'] = team['espn_team_id'] == my_team_id

            # Step 11: Build category comparison data
            current_step = "building category comparison data"
            logger.info(f"Step 11: {current_step}")
            category_data = {}
            for cat in enabled_categories:
                stat_key = cat['stat_key']
                category_data[stat_key] = {
                    'label': cat['label'],
                    'abbr': cat['abbr'],
                    'lower_is_better': cat.get('lower_is_better', False),
                    'teams': []
                }
                for team in current_standings:
                    category_data[stat_key]['teams'].append({
                        'team_id': team['team_id'],
                        'team_name': team['team_name'],
                        'value': team['category_values'].get(stat_key, 0),
                        'rank': team['category_ranks'].get(stat_key, 0),
                        'points': team['category_points'].get(stat_key, 0),
                    })
            logger.info(f"  Built category data for {len(category_data)} categories")

            # Step 12: Get user's team data
            current_step = "building user team data"
            logger.info(f"Step 12: {current_step}")
            my_team_data = None
            if my_team_id:
                my_team_current = next(
                    (s for s in current_standings if s['team_id'] == my_team_id), None
                )
                my_team_projected = next(
                    (p for p in projected_standings if p['espn_team_id'] == my_team_id), None
                )

                if my_team_current and my_team_projected:
                    logger.info(f"  Found user team: {my_team_current['team_name']}")
                    my_team_data = {
                        'team_id': my_team_id,
                        'team_name': my_team_current['team_name'],
                        'current_rank': my_team_current['rank'],
                        'projected_rank': my_team_projected['projected_standing'],
                        'movement': my_team_projected['movement'],
                        'total_points': my_team_current['total_points'],
                        'roto_points': my_team_current['roto_points'],
                        'win_probability': my_team_projected['win_probability'] / 100,
                        # Points earned in each category (for display)
                        'categories': my_team_current['categories'],
                        # Reference data
                        'category_ranks': my_team_current['category_ranks'],
                        'category_points': my_team_current['category_points'],
                    }

            # Step 13: Fetch roster for analysis
            current_step = "fetching roster for Roto insights"
            logger.info(f"Step 13: {current_step}")
            roster = []
            if my_team_id:
                try:
                    roster = espn_client.get_team_roster(my_team_id)
                    logger.info(f"  Fetched roster with {len(roster)} players")
                except Exception as e:
                    logger.warning(f"Could not fetch roster: {e}")

            # Step 14: Build insights
            current_step = "building Roto insights"
            logger.info(f"Step 14: {current_step}")
            category_analysis = get_category_analysis(roster, league.league_type, season=espn_client.year)
            waiver_targets = get_waiver_targets(espn_client, category_analysis.get('weaknesses', []), season=espn_client.year)
            trade_opportunities = get_trade_opportunities(
                espn_client, my_team_id,
                category_analysis.get('strengths', []),
                category_analysis.get('weaknesses', []),
                season=espn_client.year
            )
            logger.info(f"  Category analysis: strengths={category_analysis.get('strengths', [])}, weaknesses={category_analysis.get('weaknesses', [])}")
            logger.info(f"  Found {len(waiver_targets)} waiver targets and {len(trade_opportunities)} trade opportunities")

            # Step 15: Build final response
            current_step = "building Roto dashboard response"
            logger.info(f"Step 15: {current_step}")

            # Build start limits summary from projected standings
            start_limits_enabled = any(
                t.get('start_limits', {}).get('available', False)
                for t in projected_standings
            )
            start_limits_summary = {
                'enabled': start_limits_enabled,
                'description': (
                    'Projections are adjusted for position start limits using day-by-day simulation. '
                    'Each position has a maximum number of games that can be started.'
                ) if start_limits_enabled else (
                    'Start limit optimization is not available.'
                ),
            }

            # Get detailed optimization data from user's team if available
            if start_limits_enabled and my_team_id:
                user_proj = next(
                    (t for t in projected_standings if t.get('espn_team_id') == my_team_id),
                    None
                )
                logger.info(f"  Looking for user team {my_team_id} in projected_standings to get start_limits")
                if user_proj:
                    logger.info(f"  Found user team. start_limits available: {user_proj.get('start_limits', {}).get('available', False)}")
                    logger.debug(f"  User team start_limits keys: {list(user_proj.get('start_limits', {}).keys())}")

                if user_proj and user_proj.get('start_limits', {}).get('available'):
                    user_start_limits = user_proj['start_limits']
                    start_limits_summary['position_limits'] = user_start_limits.get('position_limits', {})
                    start_limits_summary['starts_used'] = user_start_limits.get('starts_used', {})
                    logger.info(f"  [ISSUE 3] Copied starts_used to response: {start_limits_summary['starts_used']}")
                else:
                    logger.warning(f"  [ISSUE 3] Could not find start_limits for user team!")
                    start_limits_summary['starting_players'] = user_start_limits.get('starting_players', 0)
                    start_limits_summary['benched_players'] = user_start_limits.get('benched_players', 0)
                    start_limits_summary['dropped_players'] = user_start_limits.get('dropped_players', 0)
                    start_limits_summary['ir_returning_players'] = user_start_limits.get('ir_returning_players', 0)
                    start_limits_summary['ir_players'] = user_start_limits.get('ir_players', [])
                    start_limits_summary['adjustment_summary'] = user_start_limits.get('adjustment_summary', {})
                    start_limits_summary['player_assignments'] = user_start_limits.get('player_assignments', [])

            dashboard_data = {
                'league': {
                    'id': league_id,
                    'league_name': league.league_name,
                    'league_type': league.league_type,
                    'season': league.season,
                },
                'is_roto': True,
                'scoring_categories': [
                    {
                        'key': c['stat_key'],
                        'stat_key': c['stat_key'],
                        'label': c['label'],
                        'abbr': c['abbr'],
                        'lower_is_better': c.get('lower_is_better', False),
                    }
                    for c in enabled_categories
                ],
                'current_standings': current_standings,
                'projected_standings': projected_standings,
                'category_data': category_data,
                'user_team': my_team_data,
                'insights': {
                    'category_analysis': category_analysis,
                    'waiver_targets': waiver_targets,
                    'trade_opportunities': trade_opportunities,
                },
                'start_limits': start_limits_summary,
                'last_updated': datetime.utcnow().isoformat(),
            }

        else:
            # =====================================================
            # H2H LEAGUE RESPONSE
            # =====================================================
            logger.info("Processing as H2H league")

            # Step 8 (H2H): Build current standings
            current_step = "building H2H current standings"
            logger.info(f"Step 8: {current_step}")
            current_standings = []
            for standing in standings:
                team_data = next(
                    (t for t in teams if t['espn_team_id'] == standing['espn_team_id']), {}
                )
                team_id = standing['espn_team_id']

                current_standings.append({
                    'team_id': team_id,
                    'espn_team_id': team_id,
                    'rank': standing.get('standing', 0),
                    'team_name': standing.get('team_name', team_data.get('team_name', 'Unknown')),
                    'owner_name': standing.get('owner_name', team_data.get('owner_name', 'Unknown')),
                    'record': standing.get('record', '0-0'),
                    'wins': standing.get('wins', 0),
                    'losses': standing.get('losses', 0),
                    'win_pct': standing.get('win_pct', 0),
                    'is_user_team': team_id == my_team_id,
                })

            current_standings.sort(key=lambda x: x['rank'])
            logger.info(f"  Built current standings with {len(current_standings)} teams")

            # Step 9 (H2H): Calculate projected standings
            current_step = "calculating H2H projected standings"
            logger.info(f"Step 9: {current_step}")
            projected_standings = calculate_projected_standings(teams, league.league_type, espn_client)
            logger.info(f"  Calculated projections for {len(projected_standings)} teams")

            # Add team_id and user flag to projected
            for team in projected_standings:
                team['team_id'] = team['espn_team_id']
                team['is_user_team'] = team['espn_team_id'] == my_team_id

            # Step 10 (H2H): Get user's team data
            current_step = "building H2H user team data"
            logger.info(f"Step 10: {current_step}")
            my_team_data = None
            if my_team_id:
                my_team_current = next(
                    (s for s in current_standings if s['team_id'] == my_team_id), None
                )
                my_team_projected = next(
                    (p for p in projected_standings if p['espn_team_id'] == my_team_id), None
                )

                if my_team_current and my_team_projected:
                    logger.info(f"  Found user team: {my_team_current['team_name']}")
                    my_team_data = {
                        'team_id': my_team_id,
                        'team_name': my_team_current['team_name'],
                        'current_rank': my_team_current['rank'],
                        'projected_rank': my_team_projected['projected_standing'],
                        'movement': my_team_projected['movement'],
                        'record': my_team_current['record'],
                        'win_probability': my_team_projected['win_probability'] / 100,
                        'playoff_probability': my_team_projected.get('playoff_probability', 0) / 100,
                    }

            # Step 11 (H2H): Get roster for analysis
            current_step = "fetching roster for H2H insights"
            logger.info(f"Step 11: {current_step}")
            roster = []
            if my_team_id:
                try:
                    roster = espn_client.get_team_roster(my_team_id)
                    logger.info(f"  Fetched roster with {len(roster)} players")
                except Exception as e:
                    logger.warning(f"Could not fetch roster: {e}")

            # Step 12 (H2H): Build insights
            current_step = "building H2H insights"
            logger.info(f"Step 12: {current_step}")
            category_analysis = get_category_analysis(roster, league.league_type, season=espn_client.year)
            waiver_targets = get_waiver_targets(espn_client, category_analysis.get('weaknesses', []), season=espn_client.year)
            trade_opportunities = get_trade_opportunities(
                espn_client, my_team_id,
                category_analysis.get('strengths', []),
                category_analysis.get('weaknesses', []),
                season=espn_client.year
            )
            logger.info(f"  Category analysis: strengths={category_analysis.get('strengths', [])}, weaknesses={category_analysis.get('weaknesses', [])}")
            logger.info(f"  Found {len(waiver_targets)} waiver targets and {len(trade_opportunities)} trade opportunities")

            # Step 13 (H2H): Build final response
            current_step = "building H2H dashboard response"
            logger.info(f"Step 13: {current_step}")
            dashboard_data = {
                'league': {
                    'id': league_id,
                    'league_name': league.league_name,
                    'league_type': league.league_type,
                    'season': league.season,
                },
                'is_roto': False,
                'current_standings': current_standings,
                'projected_standings': projected_standings,
                'user_team': my_team_data,
                'insights': {
                    'category_analysis': category_analysis,
                    'waiver_targets': waiver_targets,
                    'trade_opportunities': trade_opportunities,
                },
                'last_updated': datetime.utcnow().isoformat(),
            }

        # Cache the result
        current_step = "caching and returning response"
        logger.info(f"Step 16: {current_step}")
        cache.set(cache_key, dashboard_data, ttl=300)

        logger.info(f"=== DASHBOARD REQUEST COMPLETE === league_id={league_id}")
        return jsonify(dashboard_data), 200

    except LeagueNotFoundError:
        logger.warning(f"Dashboard error: League {league_id} not found (step: {current_step})")
        return jsonify({'error': 'League not found'}), 404

    except LeagueAccessDeniedError:
        logger.warning(f"Dashboard error: Access denied for league {league_id} (step: {current_step})")
        return jsonify({'error': 'Access denied'}), 403

    except ESPNAuthenticationError as e:
        logger.error(f"Dashboard error: ESPN auth failed for league {league_id} (step: {current_step}): {e}")
        return jsonify({
            'error': 'ESPN authentication failed',
            'message': 'Your ESPN cookies may have expired.'
        }), 401

    except ESPNConnectionError as e:
        logger.error(f"Dashboard error: ESPN connection failed for league {league_id} (step: {current_step}): {e}")
        return jsonify({'error': 'ESPN connection failed'}), 503

    except ESPNClientError as e:
        logger.error(f"Dashboard error: ESPN client error for league {league_id} (step: {current_step}): {e}")
        return jsonify({'error': str(e)}), 400

    except Exception as e:
        # Log full traceback with context
        error_details = traceback.format_exc()
        logger.error(f"=== DASHBOARD ERROR ===")
        logger.error(f"  League ID: {league_id}")
        logger.error(f"  User ID: {user_id}")
        logger.error(f"  Failed at step: {current_step}")
        logger.error(f"  Exception: {type(e).__name__}: {e}")
        logger.error(f"  Full traceback:\n{error_details}")
        return jsonify({
            'error': 'An unexpected error occurred',
            'step': current_step,
        }), 500


@dashboard_bp.route('/leagues/<int:league_id>/dashboard/standings', methods=['GET'])
@login_required
def get_standings_only(league_id: int):
    """
    Get just the standings data (lightweight endpoint).
    """
    try:
        league = get_league_or_404(league_id, current_user.id)
        espn_client = get_espn_client_for_league(league_id, current_user.id)

        teams = espn_client.get_teams()
        standings = espn_client.get_standings()
        is_roto = league.league_type == 'ROTO'

        # Identify user's team via SWID matching
        my_team_id = espn_client.get_user_team_id()
        if my_team_id:
            logger.debug(f"Standings: User's team identified as team_id={my_team_id}")

        if is_roto:
            scoring_categories = get_league_scoring_categories(league, espn_client)
            category_ranks = calculate_roto_category_ranks(teams, scoring_categories, espn_client)

            current_standings = []
            for team in teams:
                team_id = team['espn_team_id']
                ranks_data = category_ranks.get(team_id, {})

                # Build 'categories' dict for display - POINTS earned (10 for 1st, 9 for 2nd, etc.)
                categories_display = {}
                for cat in scoring_categories:
                    stat_key = cat['stat_key']
                    categories_display[stat_key] = ranks_data.get('category_points', {}).get(stat_key, 0)

                current_standings.append({
                    'team_id': team_id,
                    'team_name': ranks_data.get('team_name', team.get('team_name')),
                    'is_user_team': team_id == my_team_id,
                    # PRIMARY DISPLAY: Points earned in each category (10 for 1st, 9 for 2nd, etc.)
                    'categories': categories_display,
                    # Reference data:
                    'category_ranks': ranks_data.get('category_ranks', {}),
                    'category_values': ranks_data.get('category_values', {}),
                    'category_points': ranks_data.get('category_points', {}),
                    'total_points': ranks_data.get('total_points', 0),
                    'roto_points': ranks_data.get('roto_points', 0),
                })

            current_standings.sort(key=lambda x: x['total_points'], reverse=True)
            for i, team in enumerate(current_standings, 1):
                team['rank'] = i

            projected_standings = calculate_projected_roto_standings(
                teams, category_ranks, scoring_categories, espn_client
            )
            # Add is_user_team to projected standings
            for team in projected_standings:
                team['is_user_team'] = team.get('espn_team_id') == my_team_id
        else:
            current_standings = []
            for standing in standings:
                team_data = next(
                    (t for t in teams if t['espn_team_id'] == standing['espn_team_id']), {}
                )
                team_id = standing['espn_team_id']
                current_standings.append({
                    'rank': standing.get('standing', 0),
                    'team_id': team_id,
                    'team_name': standing.get('team_name', team_data.get('team_name')),
                    'record': standing.get('record', '0-0'),
                    'is_user_team': team_id == my_team_id,
                })

            current_standings.sort(key=lambda x: x['rank'])
            projected_standings = calculate_projected_standings(teams, league.league_type, espn_client)
            # Add is_user_team to projected standings
            for team in projected_standings:
                team['is_user_team'] = team.get('espn_team_id') == my_team_id

        # Build response
        response_data = {
            'league_id': league_id,
            'league_type': league.league_type,
            'is_roto': is_roto,
            'user_team_id': my_team_id,
            'current_standings': current_standings,
            'projected_standings': projected_standings,
            'last_updated': datetime.utcnow().isoformat(),
        }

        # Include scoring categories for Roto leagues so frontend knows which columns to display
        if is_roto:
            response_data['scoring_categories'] = [
                {
                    'key': c['stat_key'],
                    'stat_key': c['stat_key'],
                    'label': c['label'],
                    'abbr': c['abbr'],
                    'lower_is_better': c.get('lower_is_better', False),
                }
                for c in scoring_categories
            ]

        return jsonify(response_data), 200

    except LeagueNotFoundError:
        return jsonify({'error': 'League not found'}), 404

    except LeagueAccessDeniedError:
        return jsonify({'error': 'Access denied'}), 403

    except ESPNClientError as e:
        return jsonify({'error': str(e)}), 400

    except Exception as e:
        logger.exception(f"Standings error for league {league_id}: {e}")
        return jsonify({'error': 'Failed to fetch standings'}), 500


@dashboard_bp.route('/leagues/<int:league_id>/dashboard/insights', methods=['GET'])
@login_required
def get_insights_only(league_id: int):
    """
    Get just the quick insights data (lightweight endpoint).
    """
    try:
        league = get_league_or_404(league_id, current_user.id)
        espn_client = get_espn_client_for_league(league_id, current_user.id)

        # Determine user's team using SWID matching
        my_team_id = request.args.get('team_id', type=int)

        if not my_team_id:
            # Use SWID matching to identify the user's team
            my_team_id = espn_client.get_user_team_id()
            if my_team_id:
                logger.info(f"Insights: User's team identified via SWID matching: team_id={my_team_id}")
            else:
                logger.warning("Insights: Could not identify user's team via SWID matching")
                # Fallback: find team marked as is_user_team from get_teams()
                teams = espn_client.get_teams()
                user_team = next((t for t in teams if t.get('is_user_team')), None)
                if user_team:
                    my_team_id = user_team['espn_team_id']
                    logger.info(f"Insights: Found user team from teams list: team_id={my_team_id}")

        roster = []
        if my_team_id:
            try:
                roster = espn_client.get_team_roster(my_team_id)
            except Exception as e:
                logger.warning(f"Could not fetch roster: {e}")

        category_analysis = get_category_analysis(roster, league.league_type, season=espn_client.year)
        waiver_targets = get_waiver_targets(espn_client, category_analysis.get('weaknesses', []), season=espn_client.year)
        trade_opportunities = get_trade_opportunities(
            espn_client, my_team_id,
            category_analysis.get('strengths', []),
            category_analysis.get('weaknesses', []),
            season=espn_client.year
        )

        return jsonify({
            'league_id': league_id,
            'team_id': my_team_id,
            'category_analysis': category_analysis,
            'waiver_targets': waiver_targets,
            'trade_opportunities': trade_opportunities,
            'last_updated': datetime.utcnow().isoformat(),
        }), 200

    except LeagueNotFoundError:
        return jsonify({'error': 'League not found'}), 404

    except LeagueAccessDeniedError:
        return jsonify({'error': 'Access denied'}), 403

    except ESPNClientError as e:
        return jsonify({'error': str(e)}), 400

    except Exception as e:
        logger.exception(f"Insights error for league {league_id}: {e}")
        return jsonify({'error': 'Failed to fetch insights'}), 500

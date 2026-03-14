"""
Player Rankings API Endpoint

Provides league-specific player rankings based on z-scores across all NBA players.
Uses 24-hour caching for performance.
"""

from flask import Blueprint, jsonify, request
from backend.services.espn_client import ESPNClient
from backend.models import League
from datetime import datetime, timedelta
import logging
import numpy as np

logger = logging.getLogger(__name__)

players_bp = Blueprint('players', __name__)

# Simple in-memory cache
_rankings_cache = {}
CACHE_DURATION = timedelta(hours=24)


@players_bp.route('/leagues/<int:league_id>/player-rankings', methods=['GET'])
def get_player_rankings(league_id):
    """
    Get all NBA players ranked by z-score for this league's categories.

    Combines rostered players and free agents into a unified ranking.
    Cached for 24 hours for performance.

    Query Parameters:
        force_refresh (bool): If true, bypass cache and recalculate

    Returns:
        JSON with ranked players, categories, and league averages
    """
    try:
        logger.info('=' * 80)
        logger.info('PLAYER RANKINGS REQUEST')
        logger.info('=' * 80)

        force_refresh = request.args.get('force_refresh', 'false').lower() == 'true'

        # Check cache first (unless force refresh)
        cache_key = f'rankings_{league_id}'
        if not force_refresh and cache_key in _rankings_cache:
            cached_data, cached_time = _rankings_cache[cache_key]
            if datetime.now() - cached_time < CACHE_DURATION:
                cache_age = datetime.now() - cached_time
                logger.info(f'   Returning CACHED rankings for league {league_id}')
                logger.info(f'   Cache age: {cache_age}')
                logger.info('=' * 80)
                return jsonify(cached_data)

        logger.info(f'   Calculating FRESH rankings for league {league_id}')

        # Get league and settings
        league = League.query.get_or_404(league_id)

        # Extract category keys from scoring_settings
        # scoring_settings is a dict with 'categories' key containing list of category objects
        categories = []
        scoring_settings = league.scoring_settings

        if scoring_settings:
            if isinstance(scoring_settings, str):
                import json
                scoring_settings = json.loads(scoring_settings)

            if isinstance(scoring_settings, dict) and 'categories' in scoring_settings:
                # Extract keys from category objects
                categories = [
                    c.get('stat_key') or c.get('key') or c
                    for c in scoring_settings['categories']
                    if isinstance(c, dict) or isinstance(c, str)
                ]
            elif isinstance(scoring_settings, list):
                # Direct list of category names
                categories = scoring_settings

        if not categories:
            return jsonify({'error': 'League has no scoring categories configured'}), 400

        logger.info(f'   Categories: {categories}')

        # Initialize ESPN client
        espn_client = ESPNClient(
            league_id=league.espn_league_id,
            year=league.season,
            espn_s2=league.espn_s2_cookie,
            swid=league.swid_cookie
        )

        # Get ALL NBA players (rostered + free agents)
        all_players = get_all_nba_players(espn_client)
        logger.info(f'   Fetched {len(all_players)} total NBA players')

        # Filter to players with minimum games played
        MIN_GAMES = 5
        active_players = [p for p in all_players if p.get('games_played', 0) >= MIN_GAMES]
        logger.info(f'   Filtered to {len(active_players)} players with >= {MIN_GAMES} games')

        if not active_players:
            return jsonify({
                'error': 'No active players found with sufficient games played',
                'players': [],
                'categories': categories,
                'total_players': 0
            })

        # Calculate league-wide averages
        league_averages = calculate_league_wide_averages(active_players, categories)
        logger.info(f'   Calculated averages for {len(league_averages)} categories')

        # Calculate z-scores for each player
        ranked_players = calculate_player_zscores(active_players, categories, league_averages)

        # Sort by total z-score (descending)
        ranked_players.sort(key=lambda p: p['total_z_score'], reverse=True)

        # Add rank numbers
        for idx, player in enumerate(ranked_players, 1):
            player['rank'] = idx

        response = {
            'players': ranked_players,
            'categories': categories,
            'total_players': len(ranked_players),
            'cached_at': datetime.now().isoformat(),
            'cache_expires_at': (datetime.now() + CACHE_DURATION).isoformat(),
            'league_averages': {
                k: {'mean': round(v['mean'], 3), 'std': round(v['std'], 3)}
                for k, v in league_averages.items()
            }
        }

        # Cache the results
        _rankings_cache[cache_key] = (response, datetime.now())

        # Log completion with top players
        logger.info('=' * 80)
        logger.info('PLAYER RANKINGS COMPLETE')
        logger.info(f'   Total players ranked: {len(ranked_players)}')
        logger.info(f'   Categories: {categories}')
        logger.info('   Top 5 Players:')
        for i, p in enumerate(ranked_players[:5], 1):
            logger.info(f'      {i}. {p["name"]} ({p["team"]}) - z={p["total_z_score"]}')
        logger.info('=' * 80)

        return jsonify(response)

    except Exception as e:
        logger.error('=' * 80)
        logger.error('PLAYER RANKINGS ERROR')
        logger.error(f'   League ID: {league_id}')
        logger.error(f'   Error: {str(e)}')
        logger.error('=' * 80, exc_info=True)
        return jsonify({'error': str(e)}), 500


def get_all_nba_players(espn_client):
    """
    Fetch all NBA players by combining rostered players and free agents.
    Deduplicates by player_id.

    Returns:
        List of player dictionaries with per_game_stats
    """
    all_players = []
    seen_ids = set()

    # Get all rostered players from all teams
    try:
        all_rosters = espn_client.get_all_rosters()
        for team_id, roster in all_rosters.items():
            for player in roster:
                pid = player.get('player_id') or player.get('espn_player_id')
                if pid and pid not in seen_ids:
                    seen_ids.add(pid)
                    # Mark as rostered
                    player['is_rostered'] = True
                    player['roster_team_id'] = team_id
                    all_players.append(player)
        logger.debug(f'      Rostered players: {len(all_players)}')
    except Exception as e:
        logger.error(f'      Error fetching rostered players: {e}')

    # Get free agents (ESPN typically limits to ~200-250)
    try:
        # Fetch in batches to get more players
        free_agents = espn_client.get_free_agents(size=250, include_injury_details=False)
        fa_count = 0
        for player in free_agents:
            pid = player.get('player_id') or player.get('espn_player_id')
            if pid and pid not in seen_ids:
                seen_ids.add(pid)
                # Mark as free agent
                player['is_rostered'] = False
                player['roster_team_id'] = None
                all_players.append(player)
                fa_count += 1
        logger.debug(f'      Free agents: {fa_count}')
    except Exception as e:
        logger.error(f'      Error fetching free agents: {e}')

    logger.debug(f'      Total unique: {len(all_players)}')
    return all_players


def calculate_league_wide_averages(players, categories):
    """
    Calculate mean and standard deviation for each category
    across all NBA players (league-wide, not just rostered).

    Args:
        players: List of player dictionaries with per_game_stats
        categories: List of category names (e.g., ['PTS', 'REB', 'AST', ...])

    Returns:
        Dictionary mapping stat_key to {'mean': float, 'std': float}
    """
    league_averages = {}

    for cat in categories:
        stat_key = map_category_to_stat(cat)

        # Collect all player values for this stat
        values = []
        for p in players:
            per_game_stats = p.get('per_game_stats', {})
            value = per_game_stats.get(stat_key)

            # For percentage stats, scale from decimal to percentage
            if stat_key in ['fg_pct', 'ft_pct'] and value is not None:
                value = value * 100  # 0.476 -> 47.6

            if value is not None and not np.isnan(value):
                values.append(value)

        if values:
            mean_val = float(np.mean(values))
            std_val = float(np.std(values))
            # Ensure std is never 0 to avoid division errors
            if std_val == 0:
                std_val = 1.0
            league_averages[stat_key] = {
                'mean': mean_val,
                'std': std_val
            }
            logger.debug(f'{cat} ({stat_key}): mean={mean_val:.2f}, std={std_val:.2f}, n={len(values)}')
        else:
            league_averages[stat_key] = {'mean': 0.0, 'std': 1.0}
            logger.warning(f'No values found for {cat} ({stat_key}), using defaults')

    return league_averages


def calculate_player_zscores(players, categories, league_averages):
    """
    Calculate z-scores for each player in each category.

    Args:
        players: List of player dictionaries
        categories: List of category names
        league_averages: Dictionary from calculate_league_wide_averages

    Returns:
        List of player dictionaries with z-score data added
    """
    ranked_players = []

    for player in players:
        per_game_stats = player.get('per_game_stats', {})

        if not per_game_stats:
            continue

        # Calculate z-score for each category
        category_z_scores = {}
        total_z_score = 0

        for cat in categories:
            stat_key = map_category_to_stat(cat)
            player_value = per_game_stats.get(stat_key, 0) or 0

            # For percentage stats, scale from decimal to percentage
            if stat_key in ['fg_pct', 'ft_pct']:
                player_value = player_value * 100

            avg_data = league_averages.get(stat_key, {'mean': 0, 'std': 1})
            avg = avg_data['mean']
            std = avg_data['std']

            if std > 0:
                z_score = (player_value - avg) / std
            else:
                z_score = 0

            # For turnovers, flip the sign (lower is better)
            if cat == 'TO':
                z_score = -z_score

            category_z_scores[cat] = round(z_score, 2)
            total_z_score += z_score

        # Get eligible positions as a list
        eligible_positions = get_eligible_positions(player)

        ranked_players.append({
            'player_id': player.get('player_id') or player.get('espn_player_id'),
            'name': player.get('name', 'Unknown'),
            'team': player.get('nba_team', ''),
            'positions': eligible_positions,
            'injury_status': player.get('injury_status') or 'ACTIVE',
            'is_rostered': player.get('is_rostered', False),
            'roster_team_id': player.get('roster_team_id'),
            'total_z_score': round(total_z_score, 2),
            'category_z_scores': category_z_scores,
            'per_game_stats': format_per_game_stats(per_game_stats),
            'games_played': player.get('games_played', 0)
        })

    return ranked_players


def get_eligible_positions(player):
    """
    Extract eligible positions from player data.

    Returns:
        List of position strings (e.g., ['PG', 'SG'])
    """
    SLOT_ID_TO_POS = {
        0: 'PG', 1: 'SG', 2: 'SF', 3: 'PF', 4: 'C',
        5: 'G', 6: 'F', 7: 'SG/SF', 8: 'G/F', 9: 'PF/C',
        10: 'F/C', 11: 'UTIL'
    }

    # Try eligible_slots first (array of slot IDs)
    eligible_slots = player.get('eligible_slots', [])
    if eligible_slots:
        positions = []
        for slot_id in eligible_slots:
            if slot_id in [0, 1, 2, 3, 4]:  # Only primary positions
                pos = SLOT_ID_TO_POS.get(slot_id)
                if pos and pos not in positions:
                    positions.append(pos)
        if positions:
            return positions

    # Fallback to eligible_positions if it exists
    existing_positions = player.get('eligible_positions', [])
    if existing_positions:
        return existing_positions

    # Final fallback to position field
    position = player.get('position', '')
    if position:
        if isinstance(position, str):
            return position.split('/')
        elif isinstance(position, int):
            # Position ID mapping
            POS_ID_TO_NAME = {1: 'PG', 2: 'SG', 3: 'SF', 4: 'PF', 5: 'C'}
            return [POS_ID_TO_NAME.get(position, 'UTIL')]

    return ['UTIL']


def format_per_game_stats(per_game_stats):
    """
    Format per-game stats for API response.
    Rounds values appropriately.
    """
    formatted = {}
    for key, value in per_game_stats.items():
        if value is None:
            formatted[key] = 0
        elif key in ['fg_pct', 'ft_pct']:
            # Keep percentages as decimals but round to 3 places
            formatted[key] = round(value, 3)
        else:
            # Round counting stats to 1 decimal
            formatted[key] = round(value, 1)
    return formatted


def map_category_to_stat(category):
    """
    Map ESPN category name to stat key in per_game_stats.

    Args:
        category: Category name (e.g., 'PTS', 'FG%')

    Returns:
        Stat key used in per_game_stats dictionary
    """
    mapping = {
        'FG%': 'fg_pct',
        'FT%': 'ft_pct',
        '3PM': '3pm',
        'PTS': 'pts',
        'REB': 'reb',
        'AST': 'ast',
        'STL': 'stl',
        'BLK': 'blk',
        'TO': 'to',
        # Alternative names
        'FGM': 'fgm',
        'FGA': 'fga',
        'FTM': 'ftm',
        'FTA': 'fta',
        '3PTM': '3pm',
        'THREES': '3pm',
    }
    return mapping.get(category, category.lower())


@players_bp.route('/leagues/<int:league_id>/player-rankings/clear-cache', methods=['POST'])
def clear_rankings_cache(league_id):
    """
    Clear the rankings cache for a specific league.
    Useful when you want to force a refresh.
    """
    cache_key = f'rankings_{league_id}'
    if cache_key in _rankings_cache:
        del _rankings_cache[cache_key]
        logger.info(f'Cleared rankings cache for league {league_id}')
        return jsonify({'message': 'Cache cleared successfully'})
    return jsonify({'message': 'No cache found for this league'})

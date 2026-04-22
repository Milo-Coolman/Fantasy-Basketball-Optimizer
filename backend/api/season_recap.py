"""
Season Recap API Endpoint

Provides end-of-season summary with MVP and final standings.
"""

from flask import Blueprint, jsonify
from flask_login import login_required, current_user
from backend.services.espn_client import ESPNClient
from backend.models import League
from backend.api.players import (
    get_all_nba_players,
    calculate_league_wide_averages,
    calculate_player_zscores,
    map_category_to_stat,
)
from backend.api.dashboard import (
    get_league_scoring_categories,
    calculate_roto_category_ranks,
)
from datetime import date, datetime
import logging

logger = logging.getLogger(__name__)

season_recap_bp = Blueprint('season_recap', __name__)


def is_season_complete(season: int) -> bool:
    """
    Check if NBA regular season is complete.

    Args:
        season: The season year (e.g., 2026)

    Returns:
        True if current date is past the regular season end
    """
    # NBA regular season typically ends mid-April
    season_end = date(season, 4, 13)
    return date.today() > season_end


@season_recap_bp.route('/leagues/<int:league_id>/season-recap', methods=['GET'])
@login_required
def get_season_recap(league_id: int):
    """
    Get season recap data including MVP and final standings.

    Returns:
        JSON with season status, final standings, MVP, and highlights
    """
    try:
        logger.info('=' * 80)
        logger.info('SEASON RECAP REQUEST')
        logger.info(f'   League ID: {league_id}')
        logger.info('=' * 80)

        # Get league
        league = League.query.filter_by(
            id=league_id,
            user_id=current_user.id
        ).first_or_404()

        # Check if season is complete
        season_complete = is_season_complete(league.season)

        if not season_complete:
            return jsonify({
                'season_status': 'in_progress',
                'message': 'Season is still in progress',
                'season': league.season,
            }), 200

        # Get scoring categories using the same function as dashboard
        scoring_categories = get_league_scoring_categories(league)
        if not scoring_categories:
            return jsonify({'error': 'League has no scoring categories configured'}), 400

        # Extract category keys for z-score calculation
        categories = [c.get('stat_key') or c.get('key') for c in scoring_categories]
        logger.info(f'   Categories: {[c["abbr"] for c in scoring_categories]}')

        # Initialize ESPN client
        espn_client = ESPNClient(
            league_id=league.espn_league_id,
            year=league.season,
            espn_s2=league.espn_s2_cookie,
            swid=league.swid_cookie
        )

        # Get user's team ID
        user_team_id = espn_client.get_user_team_id()
        logger.info(f'   User team ID: {user_team_id}')

        # Get final standings using the same calculation as dashboard
        final_standings = get_final_standings(espn_client, user_team_id, scoring_categories, league.league_type)
        logger.info(f'   Got {len(final_standings)} teams in standings')

        # Get user's team recap
        user_team_recap = None
        for team in final_standings:
            if team.get('is_user_team'):
                user_team_recap = {
                    'team_name': team.get('team_name'),
                    'final_rank': team.get('rank'),
                    'total_points': team.get('total_points'),
                    'best_category': find_best_category(team, categories),
                    'worst_category': find_worst_category(team, categories, len(final_standings)),
                }
                break

        # Get user's roster and calculate MVP
        season_mvp = None
        category_leaders = []

        if user_team_id:
            roster = espn_client.get_team_roster(user_team_id)
            logger.info(f'   User roster size: {len(roster)}')

            if roster:
                # Get all players for z-score calculation
                all_players = get_all_nba_players(espn_client)
                MIN_GAMES = 5
                active_players = [p for p in all_players if p.get('games_played', 0) >= MIN_GAMES]

                # Calculate league averages
                league_averages = calculate_league_wide_averages(active_players, categories)

                # Calculate z-scores for user's roster
                roster_with_zscores = calculate_player_zscores(roster, categories, league_averages)

                # Sort by total z-score to find MVP
                roster_with_zscores.sort(key=lambda p: p.get('total_z_score', 0), reverse=True)

                if roster_with_zscores:
                    mvp = roster_with_zscores[0]
                    season_mvp = {
                        'player_id': mvp.get('player_id'),
                        'name': mvp.get('name'),
                        'team': mvp.get('team'),
                        'positions': mvp.get('positions', []),
                        'total_z_score': mvp.get('total_z_score'),
                        'category_z_scores': mvp.get('category_z_scores', {}),
                        'per_game_stats': mvp.get('per_game_stats', {}),
                        'games_played': mvp.get('games_played', 0),
                    }
                    logger.info(f'   MVP: {mvp.get("name")} (z={mvp.get("total_z_score")})')

                # Find category leaders from user's roster
                category_leaders = find_category_leaders(roster_with_zscores, categories)

        response = {
            'season_status': 'completed',
            'league': {
                'id': league.id,
                'name': league.league_name,
                'season': league.season,
                'league_type': league.league_type,
            },
            'scoring_categories': categories,
            'final_standings': final_standings,
            'user_team_recap': user_team_recap,
            'season_mvp': season_mvp,
            'category_leaders': category_leaders,
            'generated_at': datetime.utcnow().isoformat(),
        }

        logger.info('=' * 80)
        logger.info('SEASON RECAP COMPLETE')
        logger.info('=' * 80)

        return jsonify(response), 200

    except Exception as e:
        logger.error('=' * 80)
        logger.error('SEASON RECAP ERROR')
        logger.error(f'   League ID: {league_id}')
        logger.error(f'   Error: {str(e)}')
        logger.error('=' * 80, exc_info=True)
        return jsonify({'error': str(e)}), 500


def get_final_standings(espn_client, user_team_id, scoring_categories, league_type):
    """
    Get final standings using the same calculation as dashboard.

    For Roto leagues, uses calculate_roto_category_ranks from dashboard.
    For H2H leagues, uses win/loss standings.

    Returns:
        List of team standings sorted by rank
    """
    try:
        is_roto = league_type == 'ROTO'

        if is_roto:
            return get_roto_standings(espn_client, user_team_id, scoring_categories)
        else:
            return get_h2h_standings(espn_client, user_team_id)

    except Exception as e:
        logger.error(f'Error getting standings: {e}', exc_info=True)
        return []


def get_h2h_standings(espn_client, user_team_id):
    """Get H2H league standings based on wins/losses."""
    standings = espn_client.get_standings()

    result = []
    for team in standings:
        team_id = team.get('team_id') or team.get('espn_team_id')
        result.append({
            'rank': team.get('rank') or team.get('standing'),
            'team_id': team_id,
            'team_name': team.get('team_name'),
            'owner_name': team.get('owner_name'),
            'total_points': team.get('wins', 0),
            'is_user_team': team_id == user_team_id,
            'category_ranks': {},
        })

    result.sort(key=lambda t: t.get('rank', 99))
    return result


def get_roto_standings(espn_client, user_team_id, scoring_categories):
    """
    Calculate Roto standings using the same function as dashboard.

    Uses calculate_roto_category_ranks to ensure consistent results.
    """
    # Get teams with their stats (same as dashboard)
    teams = espn_client.get_teams()

    # Calculate category ranks using dashboard function
    category_ranks = calculate_roto_category_ranks(teams, scoring_categories, espn_client)

    # Build standings list
    result = []
    for team in teams:
        team_id = team['espn_team_id']
        ranks_data = category_ranks.get(team_id, {})

        result.append({
            'team_id': team_id,
            'team_name': ranks_data.get('team_name', team.get('team_name', 'Unknown')),
            'owner_name': team.get('owner_name', 'Unknown'),
            'is_user_team': team_id == user_team_id,
            'total_points': ranks_data.get('total_points', 0),
            'roto_points': ranks_data.get('roto_points', 0),
            'category_ranks': ranks_data.get('category_ranks', {}),
            'category_points': ranks_data.get('category_points', {}),
        })

    # Sort by total points and assign ranks
    result.sort(key=lambda x: x['total_points'], reverse=True)
    for i, team in enumerate(result, 1):
        team['rank'] = i

    return result


def find_best_category(team, categories):
    """Find the category where the team ranked best (lowest rank number)."""
    category_ranks = team.get('category_ranks', {})
    if not category_ranks:
        return None

    best_cat = None
    best_rank = float('inf')

    for cat in categories:
        rank = category_ranks.get(cat)
        if rank is not None and rank < best_rank:
            best_rank = rank
            best_cat = cat

    if best_cat:
        return {
            'name': best_cat,
            'rank': best_rank,
        }
    return None


def find_worst_category(team, categories, num_teams):
    """Find the category where the team ranked worst (highest rank number)."""
    category_ranks = team.get('category_ranks', {})
    if not category_ranks:
        return None

    worst_cat = None
    worst_rank = 0

    for cat in categories:
        rank = category_ranks.get(cat)
        if rank is not None and rank > worst_rank:
            worst_rank = rank
            worst_cat = cat

    if worst_cat:
        return {
            'name': worst_cat,
            'rank': worst_rank,
        }
    return None


def find_category_leaders(roster_with_zscores, categories):
    """
    Find the best player on the roster for each category.

    Returns:
        List of category leaders with player name and z-score
    """
    leaders = []

    for cat in categories:
        best_player = None
        best_z = float('-inf')

        for player in roster_with_zscores:
            z_scores = player.get('category_z_scores', {})
            z = z_scores.get(cat, 0)

            if z > best_z:
                best_z = z
                best_player = player

        if best_player:
            # Get per-game value for this category
            stat_key = map_category_to_stat(cat)
            per_game_stats = best_player.get('per_game_stats', {})
            per_game_value = per_game_stats.get(stat_key, 0)

            leaders.append({
                'category': cat,
                'player_name': best_player.get('name'),
                'z_score': round(best_z, 2),
                'per_game_value': per_game_value,
            })

    return leaders

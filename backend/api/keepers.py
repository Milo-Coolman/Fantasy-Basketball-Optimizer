"""
Keepers API Endpoint

Provides keeper league data and player rankings for keeper selection.
"""

from flask import Blueprint, jsonify, request
from flask_login import login_required, current_user
from backend.services.espn_client import ESPNClient, get_player_ages
from backend.models import League
from backend.api.players import get_all_nba_players, calculate_player_zscores, calculate_league_wide_averages
import logging

logger = logging.getLogger(__name__)

keepers_bp = Blueprint('keepers', __name__)

# Slot ID to position name mapping
SLOT_ID_TO_NAME = {
    0: 'PG', 1: 'SG', 2: 'SF', 3: 'PF', 4: 'C',
    5: 'G', 6: 'F', 7: 'SG/SF', 8: 'G/F', 9: 'PF/C',
    10: 'F/C', 11: 'UTIL', 12: 'BE', 13: 'IR', 14: 'IR+'
}

def get_eligible_positions(eligible_slots):
    """Convert slot IDs to position names, filtering out non-position slots."""
    positions = []
    for slot_id in eligible_slots:
        if slot_id in SLOT_ID_TO_NAME and slot_id <= 4:  # Only primary positions (PG, SG, SF, PF, C)
            positions.append(SLOT_ID_TO_NAME[slot_id])
    return positions


@keepers_bp.route('/leagues/<int:league_id>/keepers', methods=['GET'])
@login_required
def get_keepers(league_id):
    """
    Get keeper information and player rankings for keeper selection.

    Query Parameters:
        team_id (optional): ESPN team ID to view. Defaults to user's team.

    Returns roster players ranked by z-score with age and games played
    for keeper evaluation.
    """
    try:
        logger.info('=' * 80)
        logger.info('KEEPERS REQUEST')
        logger.info('=' * 80)

        # Get league
        league = League.query.get_or_404(league_id)

        # Verify user owns this league
        if league.user_id != current_user.id:
            return jsonify({'error': 'Unauthorized'}), 403

        # Initialize ESPN client
        espn_client = ESPNClient(
            league_id=league.espn_league_id,
            year=league.season,
            espn_s2=league.espn_s2_cookie,
            swid=league.swid_cookie
        )

        # Get keeper settings
        keeper_settings = espn_client.get_keeper_settings()

        if not keeper_settings.get('is_keeper_league', False):
            return jsonify({
                'is_keeper_league': False,
                'message': 'This league is not configured as a keeper league'
            })

        keeper_count = keeper_settings.get('keeper_count', 0)

        # Get all teams for the dropdown
        all_teams = espn_client.get_teams()
        teams_list = [
            {
                'team_id': t.get('espn_team_id') or t.get('team_id'),
                'team_name': t.get('team_name'),
                'owner_name': t.get('owner_name'),
                'is_user_team': t.get('is_user_team', False)
            }
            for t in all_teams
        ]

        # Get user's team as default
        user_team = espn_client.get_user_team()
        if not user_team:
            return jsonify({'error': 'Could not find user team'}), 404

        # Check if a specific team was requested
        requested_team_id = request.args.get('team_id', type=int)

        if requested_team_id:
            # Find the requested team
            selected_team = next(
                (t for t in teams_list if t['team_id'] == requested_team_id),
                None
            )
            if not selected_team:
                return jsonify({'error': 'Team not found'}), 404
            team_name = selected_team['team_name']
            team_id = requested_team_id
        else:
            team_name = user_team.get('team_name', 'My Team')
            team_id = user_team.get('team_id') or user_team.get('espn_team_id')

        # Get roster separately using get_team_roster
        roster = espn_client.get_team_roster(team_id)

        logger.info(f'   User team: {team_name}')
        logger.info(f'   Roster size: {len(roster)}')
        logger.info(f'   Keeper count: {keeper_count}')

        # Get scoring categories
        categories = []
        scoring_settings = league.scoring_settings

        if scoring_settings:
            if isinstance(scoring_settings, str):
                import json
                scoring_settings = json.loads(scoring_settings)

            if isinstance(scoring_settings, dict) and 'categories' in scoring_settings:
                categories = [
                    c.get('stat_key') or c.get('key') or c
                    for c in scoring_settings['categories']
                    if isinstance(c, dict) or isinstance(c, str)
                ]
            elif isinstance(scoring_settings, list):
                categories = scoring_settings

        if not categories:
            return jsonify({'error': 'League has no scoring categories configured'}), 400

        logger.info(f'   Categories extracted: {categories}')

        # Calculate z-scores for roster players
        # First get all players to establish league baseline
        all_players = get_all_nba_players(espn_client)

        # Filter to players with stats
        players_with_stats = []
        for player in all_players:
            per_game = player.get('per_game_stats', {})
            games = player.get('games_played', 0)
            if games > 0 and any(per_game.get(cat.lower(), 0) for cat in categories):
                players_with_stats.append(player)

        logger.info(f'   Players with stats: {len(players_with_stats)}')

        # Calculate league averages for z-score baseline
        league_averages = calculate_league_wide_averages(players_with_stats, categories)

        # Calculate z-scores using all players as baseline
        ranked_players = calculate_player_zscores(players_with_stats, categories, league_averages)

        # Build a lookup dict by player name
        player_z_lookup = {p['name']: p for p in ranked_players}

        # Build keeper rankings for user's roster
        keeper_rankings = []

        for roster_player in roster:
            player_name = roster_player.get('name', 'Unknown')
            player_id = roster_player.get('player_id') or roster_player.get('espn_player_id')

            # Find z-score data for this player
            z_data = player_z_lookup.get(player_name, {})

            if not z_data:
                # Player has no stats, skip
                logger.debug(f'   Skipping {player_name}: no z-score data')
                continue

            # Get per-game stats from roster data or z-score data
            per_game = roster_player.get('per_game_stats', {})
            if not per_game:
                per_game = z_data.get('per_game_stats', {})

            # Get eligible positions from slot IDs
            eligible_slots = roster_player.get('eligible_slots', [])
            eligible_positions = get_eligible_positions(eligible_slots)

            # Debug: log first player's data
            if len(keeper_rankings) == 0:
                logger.info(f'   Sample player {player_name}: age={roster_player.get("age")}, '
                           f'slots={eligible_slots}, positions={eligible_positions}')
                logger.info(f'   Sample z_data keys: {list(z_data.keys())}')
                logger.info(f'   Sample category_z_scores: {z_data.get("category_z_scores", {})}')

            keeper_rankings.append({
                'name': player_name,
                'player_id': player_id,
                'position': roster_player.get('position', 'UNKNOWN'),
                'eligible_positions': eligible_positions,
                'nba_team': roster_player.get('nba_team', 'FA'),
                'age': roster_player.get('age'),
                'games_played': roster_player.get('games_played', 0),
                'injury_status': roster_player.get('injury_status'),
                'total_z_score': z_data.get('total_z_score', 0),
                'category_z_scores': z_data.get('category_z_scores', {}),
                'per_game_stats': per_game
            })

        # Fetch ages for all players in keeper rankings
        player_ids = [p.get('player_id') for p in keeper_rankings if p.get('player_id')]
        if player_ids:
            logger.info(f'   Fetching ages for {len(player_ids)} players')
            ages = get_player_ages(player_ids, league.season)

            # Add ages to keeper rankings
            for player in keeper_rankings:
                pid = player.get('player_id')
                if pid and pid in ages:
                    player['age'] = ages[pid]

        # Sort by total z-score (highest first)
        keeper_rankings.sort(key=lambda x: x.get('total_z_score', 0), reverse=True)

        # Add ranking
        for i, player in enumerate(keeper_rankings):
            player['rank'] = i + 1

        logger.info(f'   Keeper rankings calculated: {len(keeper_rankings)} players')
        if keeper_rankings:
            logger.info(f'   Top player: {keeper_rankings[0]["name"]} (z={keeper_rankings[0]["total_z_score"]:.2f})')

        return jsonify({
            'is_keeper_league': True,
            'keeper_count': keeper_count,
            'team_name': team_name,
            'team_id': team_id,
            'teams': teams_list,
            'players': keeper_rankings,
            'categories': categories
        })

    except Exception as e:
        logger.exception(f'Error fetching keepers for league {league_id}')
        return jsonify({'error': str(e)}), 500


@keepers_bp.route('/leagues/<int:league_id>/keeper-settings', methods=['GET'])
@login_required
def get_keeper_settings(league_id):
    """
    Check if a league is a keeper league and get keeper count.
    """
    try:
        league = League.query.get_or_404(league_id)

        if league.user_id != current_user.id:
            return jsonify({'error': 'Unauthorized'}), 403

        espn_client = ESPNClient(
            league_id=league.espn_league_id,
            year=league.season,
            espn_s2=league.espn_s2_cookie,
            swid=league.swid_cookie
        )

        keeper_settings = espn_client.get_keeper_settings()

        return jsonify(keeper_settings)

    except Exception as e:
        logger.exception(f'Error fetching keeper settings for league {league_id}')
        return jsonify({'error': str(e)}), 500

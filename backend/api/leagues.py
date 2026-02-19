"""
League API Endpoints.

Handles CRUD operations for ESPN Fantasy Basketball leagues.
Uses ESPN client service for data fetching and database services for persistence.

Reference: PRD Section 5.2 - League Endpoints
"""

import logging
from datetime import datetime

from flask import Blueprint, request, jsonify
from flask_login import login_required, current_user

from backend.services.espn_client import (
    ESPNClient,
    ESPNClientError,
    ESPNAuthenticationError,
    ESPNLeagueNotFoundError,
    ESPNConnectionError,
)
from backend.services.league_service import (
    create_league,
    get_league_or_404,
    get_user_leagues,
    update_league_settings,
    mark_league_updated,
    delete_league,
    get_league_credentials,
    LeagueAlreadyExistsError,
    LeagueNotFoundError,
    LeagueAccessDeniedError,
)
from backend.services.team_service import (
    create_or_update_team,
    get_league_teams,
    get_league_standings,
    delete_league_teams,
)
from backend.services.player_service import (
    create_or_update_player,
)
from backend.services.cache_service import get_cache, CacheTTL

logger = logging.getLogger(__name__)

leagues_bp = Blueprint('leagues', __name__)


# =============================================================================
# Helper Functions
# =============================================================================

def sync_league_data(league, espn_client):
    """
    Sync league data from ESPN to database.

    Args:
        league: League database object
        espn_client: Connected ESPNClient instance

    Returns:
        Dictionary with sync results
    """
    results = {
        'settings_updated': False,
        'teams_synced': 0,
        'players_synced': 0,
        'errors': []
    }

    try:
        # 1. Sync league settings
        settings = espn_client.get_league_settings()

        # Extract actual scoring categories from team stats
        scoring_categories = espn_client.get_scoring_categories()
        logger.info(f"Detected scoring categories for league {league.id}: {scoring_categories}")

        # Build scoring_settings with categories list
        scoring_settings = settings.get('scoring_settings', {})
        scoring_settings['categories'] = scoring_categories

        update_league_settings(
            league_id=league.id,
            user_id=league.user_id,
            league_name=settings.get('name'),
            league_type=settings.get('scoring_type'),
            num_teams=settings.get('size'),
            roster_settings=settings.get('roster_settings'),
            scoring_settings=scoring_settings
        )
        results['settings_updated'] = True
        results['scoring_categories'] = scoring_categories
        logger.info(f"Updated settings for league {league.id}")

    except Exception as e:
        logger.error(f"Error syncing league settings: {e}")
        results['errors'].append(f"Settings sync failed: {str(e)}")

    try:
        # 2. Sync teams and standings
        teams = espn_client.get_teams()
        standings = espn_client.get_standings()

        # Create a standings lookup
        standings_map = {s['espn_team_id']: s for s in standings}

        for team_data in teams:
            espn_team_id = team_data['espn_team_id']
            standing_data = standings_map.get(espn_team_id, {})

            create_or_update_team(
                league_id=league.id,
                espn_team_id=espn_team_id,
                team_name=team_data['team_name'],
                owner_name=team_data.get('owner_name'),
                current_record=standing_data.get('record'),
                current_standing=standing_data.get('standing')
            )
            results['teams_synced'] += 1

        logger.info(f"Synced {results['teams_synced']} teams for league {league.id}")

    except Exception as e:
        logger.error(f"Error syncing teams: {e}")
        results['errors'].append(f"Teams sync failed: {str(e)}")

    try:
        # 3. Sync rosters and players
        all_rosters = espn_client.get_all_rosters()

        player_ids_seen = set()
        for team_id, roster in all_rosters.items():
            for player_data in roster:
                espn_player_id = player_data['espn_player_id']

                # Only create/update each player once
                if espn_player_id not in player_ids_seen:
                    create_or_update_player(
                        espn_player_id=espn_player_id,
                        name=player_data['name'],
                        position=player_data.get('position'),
                        nba_team=player_data.get('nba_team'),
                        injury_status=player_data.get('injury_status')
                    )
                    player_ids_seen.add(espn_player_id)
                    results['players_synced'] += 1

        logger.info(f"Synced {results['players_synced']} players for league {league.id}")

    except Exception as e:
        logger.error(f"Error syncing players: {e}")
        results['errors'].append(f"Players sync failed: {str(e)}")

    # Mark league as updated
    mark_league_updated(league.id)

    # Invalidate cache for this league
    cache = get_cache()
    cache.invalidate_league(league.id)

    return results


# =============================================================================
# API Endpoints
# =============================================================================

@leagues_bp.route('', methods=['GET'])
@login_required
def list_leagues():
    """
    Get all leagues for the authenticated user.

    Returns:
        JSON array of league objects with basic info.

    Response:
        200: { leagues: [{ league_id, league_name, season, last_updated }] }
    """
    try:
        leagues = get_user_leagues(current_user.id)

        return jsonify({
            'leagues': [league.to_dict() for league in leagues]
        }), 200

    except Exception as e:
        logger.error(f"Error fetching leagues for user {current_user.id}: {e}")
        return jsonify({'error': 'Failed to fetch leagues'}), 500


@leagues_bp.route('', methods=['POST'])
@login_required
def add_league():
    """
    Add a new ESPN league for the authenticated user.

    Validates ESPN credentials, fetches league data, and stores in database.

    Request JSON:
        espn_league_id: ESPN league ID (required)
        season: Season year e.g. 2025 (required)
        espn_s2: ESPN_S2 cookie value (required)
        swid: SWID cookie value (required)

    Returns:
        201: { league_id, league_name, settings }
        400: Validation error
        401: ESPN authentication failed
        404: League not found on ESPN
        409: League already exists
        500: Server error
    """
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    # Validate required fields
    required_fields = ['espn_league_id', 'season', 'espn_s2', 'swid']
    missing = [f for f in required_fields if not data.get(f)]
    if missing:
        return jsonify({
            'error': f"Missing required fields: {', '.join(missing)}"
        }), 400

    espn_league_id = data['espn_league_id']
    season = data['season']
    espn_s2 = data['espn_s2']
    swid = data['swid']

    # Validate data types
    try:
        espn_league_id = int(espn_league_id)
        season = int(season)
    except (ValueError, TypeError):
        return jsonify({'error': 'Invalid league_id or season format'}), 400

    # Validate SWID format (should be like {GUID})
    if not swid.startswith('{') or not swid.endswith('}'):
        return jsonify({
            'error': 'SWID should be in format {XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX}'
        }), 400

    try:
        # Step 1: Verify ESPN credentials and fetch league data
        logger.info(f"Connecting to ESPN league {espn_league_id} for user {current_user.id}")

        espn_client = ESPNClient(
            league_id=espn_league_id,
            year=season,
            espn_s2=espn_s2,
            swid=swid
        )

        # Get league settings to verify connection and get league name
        settings = espn_client.get_league_settings()
        league_name = settings.get('name', f"League {espn_league_id}")

        # Extract actual scoring categories from team stats
        scoring_categories = espn_client.get_scoring_categories()
        logger.info(f"Detected scoring categories for league {espn_league_id}: {scoring_categories}")

        # Build scoring_settings with categories list
        scoring_settings = settings.get('scoring_settings', {})
        scoring_settings['categories'] = scoring_categories

        # Step 2: Create league in database
        league = create_league(
            user_id=current_user.id,
            espn_league_id=espn_league_id,
            season=season,
            espn_s2=espn_s2,
            swid=swid,
            league_name=league_name,
            league_type=settings.get('scoring_type', 'H2H_CATEGORY'),
            num_teams=settings.get('size'),
            roster_settings=settings.get('roster_settings'),
            scoring_settings=scoring_settings
        )

        # Step 3: Sync teams and players
        sync_results = sync_league_data(league, espn_client)

        logger.info(f"Successfully added league {league.id} for user {current_user.id}")

        return jsonify({
            'message': 'League added successfully',
            'league_id': league.id,
            'league_name': league_name,
            'settings': {
                'scoring_type': settings.get('scoring_type'),
                'num_teams': settings.get('size'),
                'current_week': settings.get('current_week')
            },
            'sync_results': {
                'teams_synced': sync_results['teams_synced'],
                'players_synced': sync_results['players_synced']
            }
        }), 201

    except LeagueAlreadyExistsError:
        return jsonify({
            'error': f'League {espn_league_id} for season {season} already exists'
        }), 409

    except ESPNAuthenticationError as e:
        logger.warning(f"ESPN auth failed for user {current_user.id}: {e}")
        return jsonify({
            'error': 'ESPN authentication failed. Please verify your espn_s2 and swid cookies are correct and not expired.'
        }), 401

    except ESPNLeagueNotFoundError:
        return jsonify({
            'error': f'League {espn_league_id} not found on ESPN for season {season}'
        }), 404

    except ESPNConnectionError as e:
        logger.error(f"ESPN connection error: {e}")
        return jsonify({
            'error': 'Failed to connect to ESPN. Please try again later.'
        }), 503

    except ESPNClientError as e:
        logger.error(f"ESPN client error: {e}")
        return jsonify({'error': str(e)}), 400

    except Exception as e:
        logger.exception(f"Unexpected error adding league: {e}")
        return jsonify({'error': 'An unexpected error occurred'}), 500


@leagues_bp.route('/<int:league_id>', methods=['GET'])
@login_required
def get_league_details(league_id):
    """
    Get detailed information for a specific league.

    Args:
        league_id: League ID

    Returns:
        JSON with league info, teams, and settings.

    Response:
        200: { league_info, teams, settings }
        404: League not found
    """
    try:
        league = get_league_or_404(league_id, current_user.id)
        teams = get_league_teams(league_id)
        standings = get_league_standings(league_id)

        return jsonify({
            'league': league.to_dict(),
            'teams': [team.to_dict() for team in teams],
            'standings': standings,
            'settings': {
                'league_type': league.league_type,
                'num_teams': league.num_teams,
                'roster_settings': league.roster_settings,
                'scoring_settings': league.scoring_settings
            }
        }), 200

    except LeagueNotFoundError:
        return jsonify({'error': 'League not found'}), 404

    except LeagueAccessDeniedError:
        return jsonify({'error': 'Access denied'}), 403

    except Exception as e:
        logger.error(f"Error fetching league {league_id}: {e}")
        return jsonify({'error': 'Failed to fetch league details'}), 500


@leagues_bp.route('/<int:league_id>/refresh', methods=['POST'])
@login_required
def refresh_league_data(league_id):
    """
    Manually trigger a data refresh for a league.

    Fetches latest data from ESPN and updates database.

    Args:
        league_id: League ID

    Returns:
        JSON with refresh status and results.

    Response:
        200: { success: true, updated_at, sync_results }
        401: ESPN authentication failed
        404: League not found
        503: ESPN connection failed
    """
    try:
        league = get_league_or_404(league_id, current_user.id)

        # Get ESPN credentials
        credentials = get_league_credentials(league_id, current_user.id)

        logger.info(f"Refreshing league {league_id} for user {current_user.id}")

        # Connect to ESPN
        espn_client = ESPNClient(
            league_id=credentials['espn_league_id'],
            year=credentials['season'],
            espn_s2=credentials['espn_s2'],
            swid=credentials['swid']
        )

        # Sync all data
        sync_results = sync_league_data(league, espn_client)

        # Get updated league
        league = get_league_or_404(league_id, current_user.id)

        return jsonify({
            'success': True,
            'message': 'League data refreshed successfully',
            'updated_at': league.last_updated.isoformat() if league.last_updated else None,
            'sync_results': {
                'settings_updated': sync_results['settings_updated'],
                'teams_synced': sync_results['teams_synced'],
                'players_synced': sync_results['players_synced'],
                'errors': sync_results['errors']
            }
        }), 200

    except LeagueNotFoundError:
        return jsonify({'error': 'League not found'}), 404

    except LeagueAccessDeniedError:
        return jsonify({'error': 'Access denied'}), 403

    except ESPNAuthenticationError:
        return jsonify({
            'error': 'ESPN authentication failed. Your cookies may have expired. Please update them in league settings.'
        }), 401

    except ESPNConnectionError as e:
        logger.error(f"ESPN connection error during refresh: {e}")
        return jsonify({
            'error': 'Failed to connect to ESPN. Please try again later.'
        }), 503

    except ESPNClientError as e:
        logger.error(f"ESPN client error during refresh: {e}")
        return jsonify({'error': str(e)}), 400

    except Exception as e:
        logger.exception(f"Unexpected error refreshing league {league_id}: {e}")
        return jsonify({'error': 'An unexpected error occurred'}), 500


@leagues_bp.route('/<int:league_id>', methods=['DELETE'])
@login_required
def remove_league(league_id):
    """
    Remove a league from the user's account.

    Deletes the league and all associated data (teams, rosters, projections).

    Args:
        league_id: League ID

    Returns:
        JSON success message.

    Response:
        200: { success: true }
        404: League not found
    """
    try:
        # Verify ownership and get league
        league = get_league_or_404(league_id, current_user.id)
        league_name = league.league_name

        # Delete teams first (handles rosters via cascade/explicit delete)
        teams_deleted = delete_league_teams(league_id)

        # Delete the league
        delete_league(league_id, current_user.id)

        # Invalidate cache
        cache = get_cache()
        cache.invalidate_league(league_id)

        logger.info(f"Deleted league {league_id} ({league_name}) for user {current_user.id}")

        return jsonify({
            'success': True,
            'message': f'League "{league_name}" removed successfully',
            'teams_removed': teams_deleted
        }), 200

    except LeagueNotFoundError:
        return jsonify({'error': 'League not found'}), 404

    except LeagueAccessDeniedError:
        return jsonify({'error': 'Access denied'}), 403

    except Exception as e:
        logger.exception(f"Error deleting league {league_id}: {e}")
        return jsonify({'error': 'Failed to delete league'}), 500


@leagues_bp.route('/<int:league_id>/settings', methods=['PUT'])
@login_required
def update_league_config(league_id):
    """
    Update league configuration (cookies, name, etc.).

    Request JSON (all optional):
        league_name: New league name
        espn_s2: New ESPN_S2 cookie
        swid: New SWID cookie

    Returns:
        Updated league info.
    """
    try:
        league = get_league_or_404(league_id, current_user.id)

        data = request.get_json() or {}

        # Update allowed fields
        if 'league_name' in data:
            league.league_name = data['league_name']

        if 'espn_s2' in data and 'swid' in data:
            from backend.services.league_service import update_league_cookies
            update_league_cookies(league_id, current_user.id, data['espn_s2'], data['swid'])

        from backend.extensions import db
        db.session.commit()

        return jsonify({
            'success': True,
            'league': league.to_dict()
        }), 200

    except LeagueNotFoundError:
        return jsonify({'error': 'League not found'}), 404

    except LeagueAccessDeniedError:
        return jsonify({'error': 'Access denied'}), 403

    except Exception as e:
        logger.error(f"Error updating league settings: {e}")
        return jsonify({'error': 'Failed to update league settings'}), 500


@leagues_bp.route('/<int:league_id>/matchups', methods=['GET'])
@login_required
def get_league_matchups(league_id):
    """
    Get current or specific week matchups for H2H leagues.

    Query params:
        week: Week number (optional, defaults to current)

    Returns:
        JSON with matchup data.
    """
    try:
        league = get_league_or_404(league_id, current_user.id)

        # Check if H2H league
        if league.league_type == 'ROTO':
            return jsonify({
                'error': 'Matchups not available for Roto leagues'
            }), 400

        week = request.args.get('week', type=int)

        # Get credentials and connect to ESPN
        credentials = get_league_credentials(league_id, current_user.id)

        espn_client = ESPNClient(
            league_id=credentials['espn_league_id'],
            year=credentials['season'],
            espn_s2=credentials['espn_s2'],
            swid=credentials['swid']
        )

        matchups = espn_client.get_matchups(week=week)

        return jsonify({
            'league_id': league_id,
            'week': week or espn_client.league.current_week,
            'matchups': matchups
        }), 200

    except LeagueNotFoundError:
        return jsonify({'error': 'League not found'}), 404

    except LeagueAccessDeniedError:
        return jsonify({'error': 'Access denied'}), 403

    except ESPNClientError as e:
        logger.error(f"ESPN error fetching matchups: {e}")
        return jsonify({'error': str(e)}), 400

    except Exception as e:
        logger.error(f"Error fetching matchups: {e}")
        return jsonify({'error': 'Failed to fetch matchups'}), 500


@leagues_bp.route('/<int:league_id>/free-agents', methods=['GET'])
@login_required
def get_free_agents(league_id):
    """
    Get available free agents for the league.

    Query params:
        limit: Number of players (default 50)
        position: Filter by position (PG, SG, SF, PF, C)

    Returns:
        JSON with free agent data.
    """
    try:
        league = get_league_or_404(league_id, current_user.id)

        limit = request.args.get('limit', default=50, type=int)
        position = request.args.get('position', type=str)

        # Check cache first
        cache = get_cache()
        cache_key = cache.league_key(league_id, f"free_agents:{position or 'all'}:{limit}")
        cached = cache.get(cache_key)

        if cached:
            return jsonify(cached), 200

        # Get from ESPN
        credentials = get_league_credentials(league_id, current_user.id)

        espn_client = ESPNClient(
            league_id=credentials['espn_league_id'],
            year=credentials['season'],
            espn_s2=credentials['espn_s2'],
            swid=credentials['swid']
        )

        free_agents = espn_client.get_free_agents(size=limit, position=position)

        result = {
            'league_id': league_id,
            'count': len(free_agents),
            'free_agents': free_agents
        }

        # Cache for 5 minutes
        cache.set(cache_key, result, ttl=CacheTTL.FREE_AGENTS)

        return jsonify(result), 200

    except LeagueNotFoundError:
        return jsonify({'error': 'League not found'}), 404

    except LeagueAccessDeniedError:
        return jsonify({'error': 'Access denied'}), 403

    except ESPNClientError as e:
        logger.error(f"ESPN error fetching free agents: {e}")
        return jsonify({'error': str(e)}), 400

    except Exception as e:
        logger.error(f"Error fetching free agents: {e}")
        return jsonify({'error': 'Failed to fetch free agents'}), 500


@leagues_bp.route('/<int:league_id>/projection-settings', methods=['GET', 'PUT'])
@login_required
def projection_settings(league_id):
    """
    Get or update game projection settings for the league.

    GET Response:
        {
            projection_method: 'adaptive' | 'flat_rate',
            flat_game_rate: float (0.70-1.00)
        }

    PUT Request JSON:
        projection_method: 'adaptive' or 'flat_rate'
        flat_game_rate: float between 0.70 and 1.00 (only used when method='flat_rate')

    Projection Methods:
        - 'adaptive': Uses tiered game_rate based on games played
            * 0 GP with return date: 90%
            * 0 GP no return date: 85%
            * Otherwise: max(75%, games_played / team_games_so_far)
        - 'flat_rate': Uses fixed percentage for all players
            * All players get the same game_rate (e.g., 85%)
    """
    from backend.extensions import db

    try:
        league = get_league_or_404(league_id, current_user.id)

        if request.method == 'GET':
            return jsonify({
                'projection_method': league.projection_method or 'adaptive',
                'flat_game_rate': league.flat_game_rate or 0.85,
                'method_descriptions': {
                    'adaptive': 'Uses tiered rates based on games played (90% for injured w/return, 85% for new players, 75% floor)',
                    'flat_rate': 'Uses the same fixed percentage for all players'
                }
            }), 200

        # PUT - Update settings
        data = request.get_json() or {}

        # Validate projection_method
        projection_method = data.get('projection_method', 'adaptive')
        if projection_method not in ('adaptive', 'flat_rate'):
            return jsonify({
                'error': 'Invalid projection_method. Must be "adaptive" or "flat_rate"'
            }), 400

        # Validate flat_game_rate
        flat_game_rate = data.get('flat_game_rate', 0.85)
        try:
            flat_game_rate = float(flat_game_rate)
            if not (0.50 <= flat_game_rate <= 1.00):
                return jsonify({
                    'error': 'flat_game_rate must be between 0.50 and 1.00'
                }), 400
        except (ValueError, TypeError):
            return jsonify({
                'error': 'flat_game_rate must be a valid number'
            }), 400

        # Update league settings
        league.projection_method = projection_method
        league.flat_game_rate = flat_game_rate
        db.session.commit()

        logger.info(f"Updated projection settings for league {league_id}: "
                   f"method={projection_method}, flat_rate={flat_game_rate}")

        return jsonify({
            'success': True,
            'projection_method': league.projection_method,
            'flat_game_rate': league.flat_game_rate
        }), 200

    except LeagueNotFoundError:
        return jsonify({'error': 'League not found'}), 404

    except LeagueAccessDeniedError:
        return jsonify({'error': 'Access denied'}), 403

    except Exception as e:
        logger.error(f"Error updating projection settings: {e}")
        return jsonify({'error': 'Failed to update projection settings'}), 500

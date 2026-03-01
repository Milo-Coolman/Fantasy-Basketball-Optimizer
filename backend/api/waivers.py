"""
Waiver wire recommendation API endpoints.

Uses z-score based waiver analyzer for:
- Add/drop analysis with category impact
- Best pickup recommendations
- Drop candidate suggestions
"""

import logging
from datetime import datetime, date
from flask import Blueprint, request, jsonify
from flask_login import login_required, current_user

from backend.extensions import db
from backend.models import League, Team
from backend.analysis.waiver_analyzer import WaiverAnalyzer
from backend.services.espn_client import ESPNClient

logger = logging.getLogger(__name__)

# Injury statuses that indicate a player is not available to play soon
UNAVAILABLE_STATUSES = {'OUT', 'SUSPENSION', 'INACTIVE'}

# How many days until return before filtering out (default: 14 days = 2 weeks)
MAX_DAYS_UNTIL_RETURN = 14

waivers_bp = Blueprint('waivers', __name__)


def _is_player_available(player: dict, max_days_until_return: int = MAX_DAYS_UNTIL_RETURN) -> bool:
    """
    Check if a player is available to play soon.

    Filters out:
    - Players out for the season
    - Players with long-term injuries (return date > max_days_until_return)
    - Players with OUT/SUSPENSION/INACTIVE status (but keeps DTD/QUESTIONABLE)

    Args:
        player: Player dictionary from ESPN client
        max_days_until_return: Max days until return to be considered available

    Returns:
        True if player is available or will return soon, False otherwise
    """
    # Check injury_details for out_for_season flag
    injury_details = player.get('injury_details') or {}
    if injury_details.get('out_for_season', False):
        return False

    # Check expected return date
    expected_return = player.get('expected_return_date')
    if not expected_return and injury_details:
        expected_return = injury_details.get('expected_return_date')

    if expected_return:
        # Handle both date objects and strings
        if isinstance(expected_return, str):
            try:
                expected_return = datetime.strptime(expected_return, '%Y-%m-%d').date()
            except ValueError:
                pass  # Can't parse, skip this check

        if isinstance(expected_return, date):
            today = date.today()
            days_until_return = (expected_return - today).days
            if days_until_return > max_days_until_return:
                return False

    # Check injury status (OUT, SUSPENSION, INACTIVE are not available)
    injury_status = (player.get('injury_status') or '').upper()
    if injury_status in UNAVAILABLE_STATUSES:
        return False

    # Player is available (ACTIVE, DTD, QUESTIONABLE, or no status)
    return True


def _filter_available_players(
    players: list,
    max_days_until_return: int = MAX_DAYS_UNTIL_RETURN
) -> list:
    """
    Filter a list of players to only include those available to play soon.

    Args:
        players: List of player dictionaries
        max_days_until_return: Max days until return to be considered available

    Returns:
        List of available players
    """
    available = []
    filtered_count = 0

    for player in players:
        if _is_player_available(player, max_days_until_return):
            available.append(player)
        else:
            filtered_count += 1
            injury_status = player.get('injury_status', 'UNKNOWN')
            injury_details = player.get('injury_details') or {}
            logger.debug(
                f"Filtered out {player.get('name', 'Unknown')}: "
                f"status={injury_status}, "
                f"out_for_season={injury_details.get('out_for_season', False)}, "
                f"return_date={player.get('expected_return_date')}"
            )

    if filtered_count > 0:
        logger.info(
            f"Filtered {filtered_count} unavailable players "
            f"(from {len(players)} to {len(available)} available)"
        )

    return available


@waivers_bp.route('/leagues/<int:league_id>/waivers/analyze', methods=['POST'])
@login_required
def analyze_waiver(league_id):
    """
    Analyze a specific waiver wire add/drop move.

    Args:
        league_id: League ID

    Request JSON:
        player_to_add: Player data for free agent to add
        player_to_drop: (optional) Player data for roster player to drop
        current_roster: List of current roster players
        league_averages: Dict of stat averages for z-score calculation

    Returns:
        JSON with WaiverAnalysis results.
    """
    league = League.query.filter_by(id=league_id, user_id=current_user.id).first()

    if not league:
        return jsonify({'error': 'League not found'}), 404

    data = request.get_json()

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    if 'player_to_add' not in data:
        return jsonify({'error': 'player_to_add is required'}), 400

    if 'current_roster' not in data:
        return jsonify({'error': 'current_roster is required'}), 400

    try:
        player_to_add = data['player_to_add']
        current_roster = data['current_roster']
        player_to_drop = data.get('player_to_drop')
        league_averages = data.get('league_averages', {})

        logger.info(f"=== WAIVER ANALYSIS REQUEST ===")
        logger.info(f"League ID: {league_id}")
        logger.info(f"Player to add: {player_to_add.get('name', 'Unknown')}")
        logger.info(f"Player to drop: {player_to_drop.get('name', 'Auto-select') if player_to_drop else 'Auto-select'}")
        logger.info(f"Roster size: {len(current_roster)}")

        # Get scoring categories from league settings
        scoring_categories = _get_scoring_categories(league)

        # Initialize waiver analyzer
        analyzer = WaiverAnalyzer(
            league_averages=league_averages,
            categories=scoring_categories,
            num_teams=league.num_teams or 10
        )

        # Run analysis
        analysis = analyzer.analyze_add_drop(
            player_to_add=player_to_add,
            current_roster=current_roster,
            player_to_drop=player_to_drop,
        )

        return jsonify({
            'success': True,
            'analysis': analysis.to_dict(),
        }), 200

    except Exception as e:
        logger.error(f"Waiver analysis error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Waiver analysis failed: {str(e)}'}), 500


@waivers_bp.route('/leagues/<int:league_id>/waivers/suggestions', methods=['POST'])
@login_required
def get_waiver_suggestions(league_id):
    """
    Get suggested waiver wire pickups based on team needs.

    Uses TWO-TIER LOGIC for targeting categories:
    1. If ANY category in bottom half → suggest for ALL bottom-half categories
    2. If ALL categories in top half → suggest for weakest category only

    Args:
        league_id: League ID

    Request JSON:
        team_id: (optional) Team ID for personalized recommendations
        current_roster: List of current roster players
        league_averages: Dict of stat averages for z-score calculation
        projected_category_ranks: (optional) Dict of {category: rank/roto_points} for targeting
        weak_categories: (optional) Legacy param - explicit categories to target
        max_suggestions: (optional) Number of recommendations (default 10)
        position: (optional) Filter free agents by position

    Returns:
        JSON with ranked waiver recommendations.
    """
    league = League.query.filter_by(id=league_id, user_id=current_user.id).first()

    if not league:
        return jsonify({'error': 'League not found'}), 404

    try:
        data = request.get_json() or {}

        logger.info("=" * 60)
        logger.info("=== WAIVER SUGGESTIONS DEBUG ===")
        logger.info("=" * 60)

        # Initialize ESPN client
        espn_client = ESPNClient(
            league_id=league.espn_league_id,
            year=league.season,
            espn_s2=league.espn_s2_cookie,
            swid=league.swid_cookie
        )

        # Get team_id from request or auto-detect
        team_id = data.get('team_id') or data.get('user_team_id')
        if not team_id:
            team_id = espn_client.get_user_team_id()

        logger.info(f"[DEBUG] team_id: {team_id}")

        if not team_id:
            return jsonify({
                'error': 'Could not determine user team. Please provide team_id.'
            }), 400

        # Get current roster from request or fetch from ESPN
        current_roster = data.get('current_roster')
        if not current_roster:
            all_rosters = espn_client.get_all_rosters()
            current_roster = all_rosters.get(team_id, [])

        logger.info(f"[DEBUG] current_roster size: {len(current_roster) if current_roster else 0}")
        if current_roster:
            logger.info(f"[DEBUG] First roster player keys: {list(current_roster[0].keys()) if current_roster else 'N/A'}")
            logger.info(f"[DEBUG] First roster player: {current_roster[0].get('name', 'N/A')}, z_score_value: {current_roster[0].get('z_score_value', 'NOT_PRESENT')}, per_game_value: {current_roster[0].get('per_game_value', 'NOT_PRESENT')}")

        if not current_roster:
            return jsonify({
                'error': 'Could not fetch roster',
                'suggestions': []
            }), 200

        # Get league averages from request or calculate
        league_averages = data.get('league_averages')
        if not league_averages:
            all_rosters = espn_client.get_all_rosters()
            league_averages = _calculate_league_averages(all_rosters, league.season)

        logger.info(f"[DEBUG] league_averages keys: {list(league_averages.keys()) if league_averages else 'EMPTY'}")
        if league_averages:
            sample_stat = list(league_averages.keys())[0]
            logger.info(f"[DEBUG] Sample league_averages['{sample_stat}']: {league_averages[sample_stat]}")

        # Get free agents from ESPN
        position = data.get('position')
        free_agent_limit = data.get('free_agent_limit', 50)
        all_free_agents = espn_client.get_free_agents(
            size=free_agent_limit,
            position=position
        )

        logger.info(f"[DEBUG] Fetched {len(all_free_agents)} free agents from ESPN")

        # Filter out players who are out for season or won't play soon
        # This removes: OUT, SUSPENSION, INACTIVE, out for season, long-term injured
        available_players = _filter_available_players(all_free_agents)

        logger.info(f"[DEBUG] After availability filter: {len(available_players)} available free agents")
        if available_players:
            logger.info(f"[DEBUG] First free agent keys: {list(available_players[0].keys())}")
            fa = available_players[0]
            logger.info(f"[DEBUG] First free agent: {fa.get('name', 'N/A')}, per_game_stats: {fa.get('per_game_stats', 'NOT_PRESENT')}")

        # Get category targeting data from request
        # New: projected_category_ranks for two-tier logic
        # Legacy: weak_categories for explicit targeting
        projected_category_ranks = data.get('projected_category_ranks', {})
        weak_categories = data.get('weak_categories', [])
        max_suggestions = data.get('max_suggestions', 10)

        logger.info(f"[DEBUG] projected_category_ranks: {projected_category_ranks}")
        logger.info(f"[DEBUG] weak_categories: {weak_categories}")
        logger.info(f"[DEBUG] max_suggestions: {max_suggestions}")

        # Get scoring categories from league settings
        scoring_categories = _get_scoring_categories(league)
        logger.info(f"[DEBUG] scoring_categories: {scoring_categories}")
        logger.info(f"[DEBUG] num_teams: {league.num_teams}")

        # Initialize waiver analyzer
        analyzer = WaiverAnalyzer(
            league_averages=league_averages,
            categories=scoring_categories,
            num_teams=league.num_teams or 10
        )

        logger.info("[DEBUG] WaiverAnalyzer initialized, calling find_best_pickups...")

        # Find best pickups using two-tier logic
        recommendations = analyzer.find_best_pickups(
            available_players=available_players,
            current_roster=current_roster,
            weak_categories=weak_categories if weak_categories else None,
            projected_category_ranks=projected_category_ranks if projected_category_ranks else None,
            max_suggestions=max_suggestions,
        )

        logger.info(f"[DEBUG] find_best_pickups returned {len(recommendations)} recommendations")

        # Also get drop candidates for reference
        drop_candidates = analyzer.get_drop_candidates(current_roster, limit=3)

        return jsonify({
            'success': True,
            'team_id': team_id,
            'suggestions': [rec.to_dict() for rec in recommendations],
            'drop_candidates': [dc.to_dict() for dc in drop_candidates],
            'weak_categories': weak_categories,
            'free_agents_analyzed': len(available_players),
            'free_agents_filtered': len(all_free_agents) - len(available_players),
            'free_agents_total': len(all_free_agents),
        }), 200

    except Exception as e:
        logger.error(f"Waiver suggestions error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            'error': f'Failed to generate waiver suggestions: {str(e)}',
            'suggestions': []
        }), 500


@waivers_bp.route('/leagues/<int:league_id>/waivers/recommendations', methods=['GET'])
@login_required
def get_waiver_recommendations(league_id):
    """
    Get waiver wire pickup recommendations (GET endpoint).

    Simplified endpoint that fetches everything from ESPN.

    Args:
        league_id: League ID

    Query params:
        team_id: (optional) Team ID for personalized recommendations
        position: (optional) Filter by position (PG, SG, SF, PF, C)
        limit: (optional) Number of recommendations (default 10)

    Returns:
        JSON with ranked list of waiver recommendations.
    """
    league = League.query.filter_by(id=league_id, user_id=current_user.id).first()

    if not league:
        return jsonify({'error': 'League not found'}), 404

    team_id = request.args.get('team_id', type=int)
    position = request.args.get('position', type=str)
    limit = request.args.get('limit', default=10, type=int)

    try:
        # Initialize ESPN client
        espn_client = ESPNClient(
            league_id=league.espn_league_id,
            year=league.season,
            espn_s2=league.espn_s2_cookie,
            swid=league.swid_cookie
        )

        # Get team_id if not provided
        if not team_id:
            team_id = espn_client.get_user_team_id()

        if not team_id:
            return jsonify({
                'error': 'Could not determine user team',
                'recommendations': []
            }), 200

        # Fetch all data from ESPN
        all_rosters = espn_client.get_all_rosters()
        current_roster = all_rosters.get(team_id, [])
        all_free_agents = espn_client.get_free_agents(size=50, position=position)

        # Filter out unavailable players (OUT, suspended, out for season, long-term injured)
        available_players = _filter_available_players(all_free_agents)

        league_averages = _calculate_league_averages(all_rosters, league.season)

        # Get scoring categories
        scoring_categories = _get_scoring_categories(league)

        # Initialize analyzer
        analyzer = WaiverAnalyzer(
            league_averages=league_averages,
            categories=scoring_categories,
            num_teams=league.num_teams or 10
        )

        # Get recommendations
        recommendations = analyzer.find_best_pickups(
            available_players=available_players,
            current_roster=current_roster,
            max_suggestions=limit,
        )

        return jsonify({
            'success': True,
            'recommendations': [rec.to_dict() for rec in recommendations],
            'filters': {
                'team_id': team_id,
                'position': position,
                'limit': limit
            },
            'free_agents_analyzed': len(available_players),
            'free_agents_filtered': len(all_free_agents) - len(available_players),
        }), 200

    except Exception as e:
        logger.error(f"Waiver recommendations error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            'error': f'Failed to get recommendations: {str(e)}',
            'recommendations': []
        }), 500


@waivers_bp.route('/leagues/<int:league_id>/waivers/drop-candidates', methods=['POST'])
@login_required
def get_drop_candidates(league_id):
    """
    Get ranked list of drop candidates from roster.

    Args:
        league_id: League ID

    Request JSON:
        current_roster: List of current roster players
        league_averages: Dict of stat averages for z-score calculation
        limit: (optional) Number of candidates (default 5)

    Returns:
        JSON with ranked drop candidates.
    """
    league = League.query.filter_by(id=league_id, user_id=current_user.id).first()

    if not league:
        return jsonify({'error': 'League not found'}), 404

    data = request.get_json() or {}

    if 'current_roster' not in data:
        return jsonify({'error': 'current_roster is required'}), 400

    try:
        current_roster = data['current_roster']
        league_averages = data.get('league_averages', {})
        limit = data.get('limit', 5)

        scoring_categories = _get_scoring_categories(league)

        analyzer = WaiverAnalyzer(
            league_averages=league_averages,
            categories=scoring_categories,
            num_teams=league.num_teams or 10
        )

        candidates = analyzer.get_drop_candidates(current_roster, limit=limit)

        return jsonify({
            'success': True,
            'drop_candidates': [c.to_dict() for c in candidates],
        }), 200

    except Exception as e:
        logger.error(f"Drop candidates error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Failed to get drop candidates: {str(e)}'}), 500


@waivers_bp.route('/leagues/<int:league_id>/waivers/player/<int:player_id>', methods=['GET'])
@login_required
def get_player_analysis(league_id, player_id):
    """
    Get detailed analysis for a specific free agent.

    Args:
        league_id: League ID
        player_id: ESPN Player ID

    Returns:
        JSON with player stats, projections, and add/drop suggestion.
    """
    league = League.query.filter_by(id=league_id, user_id=current_user.id).first()

    if not league:
        return jsonify({'error': 'League not found'}), 404

    try:
        # Initialize ESPN client
        espn_client = ESPNClient(
            league_id=league.espn_league_id,
            year=league.season,
            espn_s2=league.espn_s2_cookie,
            swid=league.swid_cookie
        )

        # Get player data
        player_data = espn_client.get_player_by_id(player_id)
        if not player_data:
            return jsonify({'error': 'Player not found'}), 404

        # Get user's roster and league averages
        team_id = espn_client.get_user_team_id()
        all_rosters = espn_client.get_all_rosters()
        current_roster = all_rosters.get(team_id, []) if team_id else []
        league_averages = _calculate_league_averages(all_rosters, league.season)

        scoring_categories = _get_scoring_categories(league)

        analyzer = WaiverAnalyzer(
            league_averages=league_averages,
            categories=scoring_categories,
            num_teams=league.num_teams or 10
        )

        # Analyze adding this player
        analysis = None
        if current_roster:
            analysis = analyzer.analyze_add_drop(
                player_to_add=player_data,
                current_roster=current_roster,
            )

        return jsonify({
            'success': True,
            'player': player_data,
            'analysis': analysis.to_dict() if analysis else None,
        }), 200

    except Exception as e:
        logger.error(f"Player analysis error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Player analysis failed: {str(e)}'}), 500


@waivers_bp.route('/leagues/<int:league_id>/waivers/streaming', methods=['GET'])
@login_required
def get_streaming_recommendations(league_id):
    """
    Get streaming recommendations for current week (H2H leagues only).

    Finds players with favorable weekly schedules for short-term adds.

    Args:
        league_id: League ID

    Query params:
        team_id: (optional) Team ID for personalized recommendations

    Returns:
        JSON with players having favorable weekly schedules.
    """
    league = League.query.filter_by(id=league_id, user_id=current_user.id).first()

    if not league:
        return jsonify({'error': 'League not found'}), 404

    if league.league_type == 'ROTO':
        return jsonify({'error': 'Streaming recommendations not available for Roto leagues'}), 400

    team_id = request.args.get('team_id', type=int)

    # TODO: Implement streaming recommendations with schedule analysis
    # This would require:
    # 1. Fetch weekly NBA schedule
    # 2. Find free agents with 4+ games this week
    # 3. Factor in back-to-back games
    # 4. Combine with z-score analysis

    return jsonify({
        'message': 'Streaming recommendations coming soon',
        'recommendations': [],
        'note': 'This feature will analyze weekly schedules to find optimal streaming targets'
    }), 200


# =============================================================================
# Helper Functions
# =============================================================================

def _get_scoring_categories(league: League) -> list:
    """Extract scoring categories from league settings."""
    scoring_categories = None

    if league.scoring_settings:
        scoring_settings = league.scoring_settings
        if isinstance(scoring_settings, str):
            import json
            scoring_settings = json.loads(scoring_settings)

        if 'categories' in scoring_settings:
            scoring_categories = [
                c.get('stat_key') or c.get('key') or c
                for c in scoring_settings['categories']
            ]
        elif isinstance(scoring_settings, list):
            scoring_categories = scoring_settings

    logger.debug(f"League scoring categories: {scoring_categories}")
    return scoring_categories


def _calculate_league_averages(all_rosters: dict, season: int) -> dict:
    """Calculate league-wide stat averages for z-score calculation."""
    from collections import defaultdict
    import statistics

    stat_values = defaultdict(list)
    stats_to_track = ['pts', 'reb', 'ast', 'stl', 'blk', '3pm', 'fg_pct', 'ft_pct', 'to']

    # Build season keys
    season_keys = [f'{season}_total', f'{season - 1}_total', 'total']

    for team_id, roster in all_rosters.items():
        for player in roster:
            # Try per_game_stats first (from optimizer)
            per_game = player.get('per_game_stats', {})

            # Fall back to stats structure
            if not per_game:
                stats = player.get('stats', {})
                for key in season_keys:
                    if key in stats and isinstance(stats[key], dict):
                        if 'avg' in stats[key]:
                            per_game = stats[key]['avg']
                            break

            if not per_game:
                continue

            # Collect stat values
            for stat in stats_to_track:
                val = _get_stat_value(per_game, stat)
                if val is not None and val > 0:
                    stat_values[stat].append(val)

    # Calculate mean and std for each stat
    averages = {}
    for stat, values in stat_values.items():
        if len(values) >= 2:
            averages[stat] = {
                'mean': statistics.mean(values),
                'std': statistics.stdev(values) or 1.0,
            }
        elif len(values) == 1:
            averages[stat] = {
                'mean': values[0],
                'std': 1.0,
            }

    return averages


def _get_stat_value(per_game: dict, stat: str) -> float:
    """Get a stat value from per-game stats."""
    if not per_game:
        return 0.0

    # Try various key formats
    for key in [stat, stat.upper(), stat.lower(), stat.replace('_', '')]:
        if key in per_game:
            return float(per_game[key] or 0.0)

    # Special mappings
    mappings = {
        'pts': ['PTS', 'points'],
        'reb': ['REB', 'rebounds'],
        'ast': ['AST', 'assists'],
        'stl': ['STL', 'steals'],
        'blk': ['BLK', 'blocks'],
        '3pm': ['3PM', '3P'],
        'fg_pct': ['FG%', 'FGP', 'fgPct'],
        'ft_pct': ['FT%', 'FTP', 'ftPct'],
        'to': ['TO', 'TOV', 'turnovers'],
    }

    if stat.lower() in mappings:
        for alt in mappings[stat.lower()]:
            if alt in per_game:
                return float(per_game[alt] or 0.0)

    return 0.0

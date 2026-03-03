"""
Trade analysis API endpoints.

Uses z-score based trade analyzer for:
- Per-game value comparison
- Category impact analysis
- Fairness assessment
- Trade recommendations
"""

import logging
from flask import Blueprint, request, jsonify
from flask_login import login_required, current_user

from backend.extensions import db
from backend.models import League, Team, TradeHistory
from backend.analysis.trade_analyzer import TradeAnalyzer, TradePlayer
from backend.analysis.trade_suggestions import generate_trade_suggestions
from backend.services.espn_client import ESPNClient

logger = logging.getLogger(__name__)

trades_bp = Blueprint('trades', __name__)


@trades_bp.route('/leagues/<int:league_id>/trades/analyze', methods=['POST'])
@login_required
def analyze_trade(league_id):
    """
    Analyze a potential trade between two teams using z-score comparison.

    Supports multi-player trades (1-for-1, 2-for-1, 2-for-2, 3-for-1, etc.)
    and automatically determines additional drops needed to fit roster limits.

    Args:
        league_id: League ID

    Request JSON:
        team1_id: First team ID (your team)
        team1_players: Array of player IDs from team 1 (players you give)
        team2_id: Second team ID (trade partner)
        team2_players: Array of player IDs from team 2 (players you receive)
        team1_player_data: Full player data for team1's players in trade
        team2_player_data: Full player data for team2's players in trade
        league_averages: Dict of stat averages for z-score calculation
        current_roster: (optional) Full roster for team1 to check roster limits
        roster_size_limit: (optional) Maximum roster size (default: from league settings or 15)

    Returns:
        JSON with z-score based trade analysis results including additional_drops.
    """
    league = League.query.filter_by(id=league_id, user_id=current_user.id).first()

    if not league:
        return jsonify({'error': 'League not found'}), 404

    data = request.get_json()

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    required_fields = ['team1_id', 'team1_players', 'team2_id', 'team2_players']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'{field} is required'}), 400

    # Validate teams exist
    team1 = Team.query.filter_by(id=data['team1_id'], league_id=league_id).first()
    team2 = Team.query.filter_by(id=data['team2_id'], league_id=league_id).first()

    if not team1 or not team2:
        return jsonify({'error': 'One or both teams not found'}), 404

    try:
        # Get player data from request
        team1_player_data = data.get('team1_player_data', [])
        team2_player_data = data.get('team2_player_data', [])
        league_averages = data.get('league_averages', {})

        # Get roster data for multi-player trade support
        current_roster = data.get('current_roster', [])
        roster_size_limit = data.get('roster_size_limit')

        # Get roster limit from cached league setting or fetch from ESPN
        if not roster_size_limit:
            # First check if we have it cached in the league model
            if league.active_roster_limit:
                roster_size_limit = league.active_roster_limit
                logger.info(f"Using cached active_roster_limit: {roster_size_limit}")
            else:
                # Fetch from ESPN and cache it
                try:
                    espn_client = ESPNClient(
                        league_id=league.espn_league_id,
                        year=league.season,
                        espn_s2=league.espn_s2_cookie,
                        swid=league.swid_cookie
                    )
                    roster_info = espn_client.get_roster_size_info()
                    roster_size_limit = roster_info['active_roster_size']

                    # Cache it for future use
                    league.active_roster_limit = roster_size_limit
                    db.session.commit()
                    logger.info(f"Fetched and cached active_roster_limit from ESPN: {roster_size_limit} "
                               f"(total: {roster_info['total_roster_size']}, IR: {roster_info['ir_slots']})")
                except Exception as e:
                    logger.warning(f"Could not fetch roster size from ESPN: {e}")
                    # Fall back to league.roster_settings if available
                    if league.roster_settings:
                        roster_settings = league.roster_settings
                        if isinstance(roster_settings, str):
                            import json
                            roster_settings = json.loads(roster_settings)
                        # Try to calculate active slots (total - IR)
                        total = sum(roster_settings.values()) if isinstance(roster_settings, dict) else 15
                        ir_slots = roster_settings.get('IR', 0) + roster_settings.get('IR+', 0)
                        roster_size_limit = total - ir_slots
                    else:
                        roster_size_limit = 13  # Reasonable default (15 total - 2 IR)

        logger.info(f"Roster limit for league {league_id}: {roster_size_limit} (excluding IR)")

        logger.info(f"=== TRADE ANALYSIS REQUEST ===")
        logger.info(f"League ID: {league_id}")
        logger.info(f"Trade type: {len(team1_player_data)}-for-{len(team2_player_data)}")
        logger.info(f"Team 1 players (giving): {len(team1_player_data)}")
        logger.info(f"Team 2 players (receiving): {len(team2_player_data)}")
        logger.info(f"Current roster provided: {len(current_roster)} players")
        logger.info(f"Roster size limit: {roster_size_limit}")
        logger.info(f"League averages provided: {bool(league_averages)}")

        # Get league scoring categories
        scoring_categories = None
        if league.scoring_settings:
            scoring_settings = league.scoring_settings
            if isinstance(scoring_settings, str):
                import json
                scoring_settings = json.loads(scoring_settings)
            if 'categories' in scoring_settings:
                scoring_categories = [c.get('stat_key') or c.get('key') for c in scoring_settings['categories']]
                logger.info(f"League scoring categories: {scoring_categories}")

        # Initialize trade analyzer
        analyzer = TradeAnalyzer(
            league_averages=league_averages,
            categories=scoring_categories,
            num_teams=data.get('num_teams', 10)
        )

        logger.info(f"Analyzer using categories: {analyzer.categories}")

        # Convert player data to TradePlayer objects
        players_out = []
        for player_data in team1_player_data:
            trade_player = analyzer.create_trade_player(player_data, league_averages)
            players_out.append(trade_player)

        players_in = []
        for player_data in team2_player_data:
            trade_player = analyzer.create_trade_player(player_data, league_averages)
            players_in.append(trade_player)

        # Run z-score based trade analysis (supports multi-player trades)
        analysis = analyzer.analyze_trade(
            players_out=players_out,
            players_in=players_in,
            current_roster=current_roster if current_roster else None,
            roster_size_limit=roster_size_limit,
        )

        # Log the trade analysis
        trade_record = TradeHistory(
            league_id=league_id,
            team1_id=data['team1_id'],
            team2_id=data['team2_id'],
            team1_players=data['team1_players'],
            team2_players=data['team2_players'],
            value_differential=analysis.net_z_score_change,
            was_suggested=False
        )
        db.session.add(trade_record)
        db.session.commit()

        return jsonify({
            'success': True,
            'trade_id': trade_record.id,
            'analysis': analysis.to_dict(),
            'team1': team1.to_dict(),
            'team2': team2.to_dict()
        }), 200

    except Exception as e:
        logger.error(f"Trade analysis error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Trade analysis failed: {str(e)}'}), 500


@trades_bp.route('/leagues/<int:league_id>/trades/suggestions', methods=['POST'])
@login_required
def get_trade_suggestions(league_id):
    """
    Get AI-generated trade suggestions for a team.

    Analyzes the user's weak categories and finds fair 1-for-1 trade
    opportunities with other teams who are strong in those categories.

    Uses PROJECTED category ranks passed from the frontend (from dashboard data)
    to identify weak categories based on end-of-season projections.

    Args:
        league_id: League ID

    Request JSON:
        team_id: Team ID to get suggestions for
        projected_category_ranks: Dict of category -> projected rank (from dashboard)
        all_teams_projected_ranks: Dict of team_id -> projected_category_ranks (optional)
        max_suggestions: Maximum number of suggestions (default 5)

    Returns:
        JSON with ranked trade suggestions.
    """
    league = League.query.filter_by(id=league_id, user_id=current_user.id).first()

    if not league:
        return jsonify({'error': 'League not found'}), 404

    try:
        data = request.get_json() or {}

        # Initialize ESPN client
        espn_client = ESPNClient(
            league_id=league.espn_league_id,
            year=league.season,
            espn_s2=league.espn_s2_cookie,
            swid=league.swid_cookie
        )

        # Get team_id from request body or auto-detect
        team_id = data.get('team_id') or data.get('user_team_id')
        if not team_id:
            team_id = espn_client.get_user_team_id()

        if not team_id:
            return jsonify({
                'error': 'Could not determine user team. Please provide team_id in request body.'
            }), 400

        # Get projected ranks from request (passed from frontend dashboard data)
        user_projected_ranks = data.get('projected_category_ranks', {})
        all_teams_projected_ranks = data.get('all_teams_projected_ranks', {})
        max_suggestions = data.get('max_suggestions', 5)

        # Get rosters and league averages from request (from dashboard - has z-scores from optimizer)
        frontend_team_rosters = data.get('team_rosters')
        frontend_league_averages = data.get('league_averages')

        logger.info(f"Received projected ranks for team {team_id}: {user_projected_ranks}")
        logger.info(f"Received team_rosters from frontend: {frontend_team_rosters is not None}")
        logger.info(f"Received league_averages from frontend: {frontend_league_averages is not None}")

        # DEBUG: Log sample player structure and league_averages keys
        if frontend_team_rosters:
            for tid, roster in frontend_team_rosters.items():
                if roster and len(roster) > 0:
                    sample_player = roster[0]
                    logger.info(f"=== SAMPLE PLAYER STRUCTURE (team {tid}) ===")
                    logger.info(f"  Keys: {list(sample_player.keys())}")
                    if 'per_game_stats' in sample_player:
                        logger.info(f"  per_game_stats keys: {list(sample_player['per_game_stats'].keys())}")
                        # Check for percentage stats
                        pgs = sample_player['per_game_stats']
                        logger.info(f"  fg_pct in per_game_stats: {pgs.get('fg_pct')}")
                        logger.info(f"  ft_pct in per_game_stats: {pgs.get('ft_pct')}")
                        logger.info(f"  FG% in per_game_stats: {pgs.get('FG%')}")
                        logger.info(f"  FT% in per_game_stats: {pgs.get('FT%')}")
                    if 'stats' in sample_player:
                        logger.info(f"  stats keys: {list(sample_player['stats'].keys()) if isinstance(sample_player['stats'], dict) else 'not a dict'}")
                    break  # Only log first team

        if frontend_league_averages:
            logger.info(f"=== LEAGUE AVERAGES KEYS ===")
            logger.info(f"  Keys: {list(frontend_league_averages.keys())}")
            logger.info(f"  fg_pct: {frontend_league_averages.get('fg_pct')}")
            logger.info(f"  ft_pct: {frontend_league_averages.get('ft_pct')}")
            logger.info(f"  FG%: {frontend_league_averages.get('FG%')}")

        # Get team info from ESPN (for team names)
        teams = espn_client.get_teams()
        num_teams = len(teams)

        # Use rosters from frontend if provided (already have z-scores from optimizer)
        # Otherwise fetch fresh from ESPN (will need to recalculate z-scores)
        if frontend_team_rosters:
            all_rosters = frontend_team_rosters
            # Convert string keys to int if needed
            if all_rosters and all(isinstance(k, str) for k in all_rosters.keys()):
                all_rosters = {int(k): v for k, v in all_rosters.items()}
            logger.info(f"Using rosters from frontend (has z-scores from optimizer)")
        else:
            all_rosters = espn_client.get_all_rosters()
            logger.info(f"Fetching fresh rosters from ESPN (will recalculate z-scores)")

        # Build all_teams_data structure using provided projected ranks
        all_teams_data = {}
        for team in teams:
            tid = team['espn_team_id']
            current_ranks = team.get('category_ranks', {})

            # Use projected ranks from frontend if available
            if tid == team_id and user_projected_ranks:
                projected_ranks = user_projected_ranks
                logger.debug(f"Using FRONTEND projected ranks for user team {tid}")
            elif str(tid) in all_teams_projected_ranks:
                projected_ranks = all_teams_projected_ranks[str(tid)]
            elif tid in all_teams_projected_ranks:
                projected_ranks = all_teams_projected_ranks[tid]
            else:
                # Fall back to current ranks if no projections provided
                projected_ranks = current_ranks

            all_teams_data[tid] = {
                'name': team.get('team_name', 'Unknown'),
                'roster': all_rosters.get(tid, []),
                'category_ranks': current_ranks,
                'projected_category_ranks': projected_ranks if projected_ranks else current_ranks,
            }

        # Get user's roster
        user_roster = all_rosters.get(team_id, [])
        if not user_roster:
            return jsonify({
                'error': 'Could not fetch user roster',
                'suggestions': []
            }), 200

        # Use league averages from frontend if provided, otherwise calculate
        if frontend_league_averages:
            league_averages = frontend_league_averages
            logger.info(f"Using league_averages from frontend (consistent with optimizer)")
        else:
            league_averages = _calculate_league_averages(all_rosters, league.season)
            logger.info(f"Calculated league_averages from rosters")

        # Extract scoring categories from league settings
        scoring_categories = None
        if league.scoring_settings:
            scoring_settings = league.scoring_settings
            if isinstance(scoring_settings, str):
                import json
                scoring_settings = json.loads(scoring_settings)
            if 'categories' in scoring_settings:
                scoring_categories = [c.get('stat_key') or c.get('key') or c for c in scoring_settings['categories']]
            elif isinstance(scoring_settings, list):
                # If scoring_settings is a direct list of category names
                scoring_categories = scoring_settings

        logger.info(f"League scoring categories for suggestions: {scoring_categories}")

        # Get trade suggestion mode from league settings
        trade_suggestion_mode = league.trade_suggestion_mode or 'normal'
        logger.info(f"Trade suggestion mode: {trade_suggestion_mode}")

        # Generate suggestions with league-specific categories and mode
        suggestions = generate_trade_suggestions(
            user_team_id=team_id,
            user_roster=user_roster,
            all_teams_data=all_teams_data,
            league_averages=league_averages,
            categories=scoring_categories,
            trade_suggestion_mode=trade_suggestion_mode,
            max_suggestions=max_suggestions,
            num_teams=num_teams,
        )

        logger.info(f"Generated {len(suggestions)} trade suggestions for team {team_id}")

        # Debug: Log sample suggestion to verify player names are present
        if suggestions:
            sample = suggestions[0]
            logger.info(f"Sample suggestion fields: {list(sample.keys())}")
            logger.info(f"  target_player: {sample.get('target_player')}")
            logger.info(f"  give_player: {sample.get('give_player')}")
            logger.info(f"  value_gain: {sample.get('value_gain')}")

        return jsonify({
            'success': True,
            'team_id': team_id,
            'suggestions': suggestions,
            'weak_categories': _get_weak_categories(all_teams_data.get(team_id, {}), num_teams),
        }), 200

    except Exception as e:
        logger.error(f"Trade suggestions error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            'error': f'Failed to generate trade suggestions: {str(e)}',
            'suggestions': []
        }), 500


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
            stats = player.get('stats', {})

            # Find per-game stats
            per_game = None
            for key in season_keys:
                if key in stats and isinstance(stats[key], dict):
                    if 'avg' in stats[key]:
                        per_game = stats[key]['avg']
                        break

            if not per_game:
                continue

            # Collect stat values
            for stat in stats_to_track:
                val = _get_stat_from_player(per_game, stat)
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


def _get_stat_from_player(per_game: dict, stat: str) -> float:
    """Get a stat value from per-game stats."""
    if not per_game:
        return 0.0

    # Try various key formats
    for key in [stat, stat.upper(), stat.lower(), stat.replace('_', '')]:
        if key in per_game:
            return per_game[key] or 0.0

    # Special mappings
    mappings = {
        'pts': ['PTS', 'points'],
        'reb': ['REB', 'rebounds'],
        'ast': ['AST', 'assists'],
        'stl': ['STL', 'steals'],
        'blk': ['BLK', 'blocks'],
        '3pm': ['3PM', '3P'],
        'fg_pct': ['FG%', 'FGP'],
        'ft_pct': ['FT%', 'FTP'],
        'to': ['TO', 'TOV'],
    }

    if stat.lower() in mappings:
        for alt in mappings[stat.lower()]:
            if alt in per_game:
                return per_game[alt] or 0.0

    return 0.0


def _get_weak_categories(team_data: dict, num_teams: int) -> list:
    """
    Get list of weak categories for a team.

    Uses PROJECTED category ranks (end-of-season projections) when available,
    falls back to current ranks if projections aren't available.

    A category is considered "weak" if the team ranks in the bottom half
    of the league (rank >= num_teams/2 + 1, minimum threshold of 6).
    """
    # Prefer projected ranks over current ranks
    projected_ranks = team_data.get('projected_category_ranks', {})
    current_ranks = team_data.get('category_ranks', {})

    if projected_ranks:
        ranks = projected_ranks
        rank_type = 'projected'
    else:
        ranks = current_ranks
        rank_type = 'current'

    threshold = max(num_teams // 2 + 1, 6)

    weak = []
    for cat, rank in ranks.items():
        if rank >= threshold:
            weak.append({
                'category': cat,
                'rank': rank,
                'rank_type': rank_type,
            })

    weak.sort(key=lambda x: x['rank'], reverse=True)

    if weak:
        logger.debug(f"Weak categories ({rank_type} ranks, threshold={threshold}): {[w['category'] for w in weak]}")

    return weak


@trades_bp.route('/leagues/<int:league_id>/trades/history', methods=['GET'])
@login_required
def get_trade_history(league_id):
    """
    Get history of analyzed trades for a league.

    Args:
        league_id: League ID

    Returns:
        JSON array of past trade analyses.
    """
    league = League.query.filter_by(id=league_id, user_id=current_user.id).first()

    if not league:
        return jsonify({'error': 'League not found'}), 404

    trades = TradeHistory.query.filter_by(league_id=league_id)\
        .order_by(TradeHistory.analyzed_at.desc())\
        .limit(50)\
        .all()

    return jsonify({
        'trades': [{
            'id': t.id,
            'team1_id': t.team1_id,
            'team2_id': t.team2_id,
            'team1_players': t.team1_players,
            'team2_players': t.team2_players,
            'analyzed_at': t.analyzed_at.isoformat() if t.analyzed_at else None,
            'value_differential': float(t.value_differential) if t.value_differential else None,
            'was_suggested': t.was_suggested,
            'was_accepted': t.was_accepted
        } for t in trades]
    }), 200

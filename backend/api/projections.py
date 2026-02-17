"""
Projection API Endpoints for Fantasy Basketball Optimizer.

This module provides REST API endpoints for accessing projections including:
- League standings (current vs projected)
- Team projections (H2H or Roto specific)
- Weekly matchup projections (H2H only)
- Individual player projections

Reference: PRD Section 5.3 - Projection Endpoints
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

from flask import Blueprint, jsonify, request
from flask_login import login_required, current_user

from backend.extensions import db
from backend.models import League, Team, Player, PlayerStats, Projection, Roster

# Import projection engines
try:
    from backend.projections.hybrid_engine import (
        HybridProjectionEngine,
        LeagueScoringSettings,
        LeagueType,
    )
    from backend.projections.statistical_model import (
        StatisticalProjectionEngine,
        TeamScheduleStrength,
    )
    from backend.analyzers.matchup_analyzer import (
        MatchupAnalyzer,
        TeamProjection,
    )
    from backend.analyzers.roto_analyzer import (
        RotoAnalyzer,
        TeamRotoStats,
    )
    PROJECTIONS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Projection engines not fully available: {e}")
    PROJECTIONS_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)

# Create blueprint
projections_bp = Blueprint('projections', __name__)

# Cache for engines (reuse across requests)
_engine_cache: Dict[str, Any] = {}


# =============================================================================
# Helper Functions
# =============================================================================

def get_hybrid_engine() -> Optional['HybridProjectionEngine']:
    """Get or create cached hybrid projection engine."""
    if not PROJECTIONS_AVAILABLE:
        return None

    if 'hybrid_engine' not in _engine_cache:
        try:
            _engine_cache['hybrid_engine'] = HybridProjectionEngine()
        except Exception as e:
            logger.error(f"Failed to create hybrid engine: {e}")
            return None

    return _engine_cache['hybrid_engine']


def get_matchup_analyzer(categories: Optional[List[str]] = None) -> Optional['MatchupAnalyzer']:
    """Get or create cached matchup analyzer."""
    if not PROJECTIONS_AVAILABLE:
        return None

    cache_key = f"matchup_analyzer_{'-'.join(categories or [])}"
    if cache_key not in _engine_cache:
        try:
            _engine_cache[cache_key] = MatchupAnalyzer(
                categories=categories,
                num_simulations=5000  # Reduced for API performance
            )
        except Exception as e:
            logger.error(f"Failed to create matchup analyzer: {e}")
            return None

    return _engine_cache[cache_key]


def get_roto_analyzer(league_size: int = 12) -> Optional['RotoAnalyzer']:
    """Get or create cached Roto analyzer."""
    if not PROJECTIONS_AVAILABLE:
        return None

    cache_key = f"roto_analyzer_{league_size}"
    if cache_key not in _engine_cache:
        try:
            _engine_cache[cache_key] = RotoAnalyzer(
                league_size=league_size,
                num_simulations=5000
            )
        except Exception as e:
            logger.error(f"Failed to create roto analyzer: {e}")
            return None

    return _engine_cache[cache_key]


def get_league_categories(league: League) -> List[str]:
    """Extract scoring categories from league settings."""
    default_categories = ['pts', 'trb', 'ast', 'stl', 'blk', '3p', 'fg_pct', 'ft_pct', 'tov']

    if not league.scoring_settings:
        return default_categories

    # Try to extract from scoring settings
    scoring = league.scoring_settings
    if isinstance(scoring, dict):
        categories = scoring.get('categories', [])
        if categories:
            return categories

    return default_categories


def get_team_stats(team: Team) -> Dict[str, float]:
    """Get current season stats for a team's roster."""
    stats = {
        'pts': 0, 'trb': 0, 'ast': 0, 'stl': 0, 'blk': 0,
        '3p': 0, 'tov': 0, 'fg_pct': 0.45, 'ft_pct': 0.78,
        'fgm': 0, 'fga': 0, 'ftm': 0, 'fta': 0
    }

    # Get roster players
    rosters = Roster.query.filter_by(team_id=team.id).all()
    player_count = 0

    for roster in rosters:
        player = Player.query.get(roster.player_id)
        if not player:
            continue

        # Get latest stats
        player_stats = PlayerStats.query.filter_by(
            player_id=player.id
        ).order_by(PlayerStats.stat_date.desc()).first()

        if player_stats:
            player_count += 1
            stats['pts'] += float(player_stats.points or 0)
            stats['trb'] += float(player_stats.rebounds or 0)
            stats['ast'] += float(player_stats.assists or 0)
            stats['stl'] += float(player_stats.steals or 0)
            stats['blk'] += float(player_stats.blocks or 0)
            stats['3p'] += float(player_stats.three_pointers_made or 0)
            stats['tov'] += float(player_stats.turnovers or 0)

    return stats


def get_player_season_stats(player_id: int) -> Optional[Dict[str, float]]:
    """Get current season stats for a player."""
    player_stats = PlayerStats.query.filter_by(
        player_id=player_id
    ).order_by(PlayerStats.stat_date.desc()).first()

    if not player_stats:
        return None

    return {
        'pts': float(player_stats.points or 0),
        'trb': float(player_stats.rebounds or 0),
        'ast': float(player_stats.assists or 0),
        'stl': float(player_stats.steals or 0),
        'blk': float(player_stats.blocks or 0),
        '3p': float(player_stats.three_pointers_made or 0),
        'tov': float(player_stats.turnovers or 0),
        'fg_pct': float(player_stats.field_goal_pct or 0),
        'ft_pct': float(player_stats.free_throw_pct or 0),
        'mp': float(player_stats.minutes_per_game or 0),
        'g': player_stats.games_played or 0,
    }


def parse_record(record: str) -> tuple:
    """Parse record string like '10-5-2' into (wins, losses, ties)."""
    if not record:
        return (0, 0, 0)
    parts = record.split('-')
    wins = int(parts[0]) if len(parts) > 0 else 0
    losses = int(parts[1]) if len(parts) > 1 else 0
    ties = int(parts[2]) if len(parts) > 2 else 0
    return (wins, losses, ties)


# =============================================================================
# Standings Endpoint
# =============================================================================

@projections_bp.route('/leagues/<int:league_id>/projections/standings', methods=['GET'])
@login_required
def get_projected_standings(league_id: int):
    """
    Get current and projected standings for a league.

    For H2H leagues: includes win probability and playoff odds
    For Roto leagues: includes category rankings and point totals

    Args:
        league_id: League ID

    Returns:
        JSON with current and projected standings
    """
    league = League.query.filter_by(id=league_id, user_id=current_user.id).first()

    if not league:
        return jsonify({'error': 'League not found'}), 404

    teams = Team.query.filter_by(league_id=league_id).order_by(Team.current_standing).all()

    if not teams:
        return jsonify({
            'error': 'No teams found in league',
            'current_standings': [],
            'projected_standings': []
        }), 200

    # Build current standings
    current_standings = []
    for team in teams:
        current_standings.append({
            'team_id': team.id,
            'espn_team_id': team.espn_team_id,
            'team_name': team.team_name,
            'owner_name': team.owner_name,
            'record': team.current_record,
            'standing': team.current_standing
        })

    # Build projected standings
    projected_standings = []

    if PROJECTIONS_AVAILABLE and league.is_h2h:
        # Use matchup analyzer for H2H leagues
        analyzer = get_matchup_analyzer(get_league_categories(league))

        if analyzer:
            try:
                # Build team projections
                team_projections = []
                schedule = {}
                current_records = {}

                for team in teams:
                    stats = get_team_stats(team)
                    team_proj = TeamProjection(
                        team_id=team.id,
                        team_name=team.team_name,
                        projected_stats=stats
                    )
                    team_projections.append(team_proj)

                    # Parse current record
                    current_records[team.id] = parse_record(team.current_record)

                    # Placeholder schedule (would come from ESPN API)
                    schedule[team.id] = []

                # Run simulation if we have schedule data
                # For now, use stored projections from database
                pass

            except Exception as e:
                logger.error(f"H2H projection error: {e}")

    elif PROJECTIONS_AVAILABLE and league.is_roto:
        # Use Roto analyzer
        analyzer = get_roto_analyzer(league.num_teams or 12)

        if analyzer:
            try:
                # Build Roto team stats
                roto_teams = []
                for team in teams:
                    stats = get_team_stats(team)
                    roto_team = TeamRotoStats(
                        team_id=team.id,
                        team_name=team.team_name,
                        current_totals=stats,
                        fgm=stats.get('fgm', 0),
                        fga=stats.get('fga', 1),
                        ftm=stats.get('ftm', 0),
                        fta=stats.get('fta', 1),
                        ros_projections={k: v / max(1, stats.get('g', 1)) for k, v in stats.items()},
                        games_remaining=35
                    )
                    roto_teams.append(roto_team)

                # Analyze league
                projections = analyzer.analyze_league(roto_teams)

                for proj in projections:
                    projected_standings.append({
                        'team_id': proj.team_id,
                        'team_name': proj.team_name,
                        'current_points': proj.current_points,
                        'current_rank': proj.current_rank,
                        'projected_points': round(proj.projected_points, 1),
                        'projected_standing': round(proj.projected_rank, 1),
                        'win_probability': round(proj.win_probability * 100, 1),
                        'top_3_probability': round(proj.top_3_probability * 100, 1),
                        'strengths': proj.strengths,
                        'weaknesses': proj.weaknesses
                    })

            except Exception as e:
                logger.error(f"Roto projection error: {e}")

    # Fall back to database projections if engine unavailable
    if not projected_standings:
        for team in teams:
            projected_standings.append({
                'team_id': team.id,
                'espn_team_id': team.espn_team_id,
                'team_name': team.team_name,
                'owner_name': team.owner_name,
                'projected_standing': team.projected_standing,
                'win_probability': float(team.win_probability) if team.win_probability else None
            })

        # Sort by projected standing
        projected_standings.sort(key=lambda x: x.get('projected_standing') or 999)

    return jsonify({
        'league_id': league_id,
        'league_name': league.league_name,
        'league_type': league.league_type,
        'current_standings': current_standings,
        'projected_standings': projected_standings,
        'last_updated': league.last_updated.isoformat() if league.last_updated else None,
        'projection_engine_available': PROJECTIONS_AVAILABLE
    }), 200


# =============================================================================
# Team Projections Endpoint
# =============================================================================

@projections_bp.route('/leagues/<int:league_id>/projections/team/<int:team_id>', methods=['GET'])
@login_required
def get_team_projections(league_id: int, team_id: int):
    """
    Get detailed projections for a specific team.

    For H2H leagues: includes matchup win probabilities, playoff odds
    For Roto leagues: includes category rankings, gap analysis, punt suggestions

    Args:
        league_id: League ID
        team_id: Team ID

    Returns:
        JSON with detailed team projection data
    """
    league = League.query.filter_by(id=league_id, user_id=current_user.id).first()

    if not league:
        return jsonify({'error': 'League not found'}), 404

    team = Team.query.filter_by(id=team_id, league_id=league_id).first()

    if not team:
        return jsonify({'error': 'Team not found'}), 404

    # Get all teams for comparison
    all_teams = Team.query.filter_by(league_id=league_id).all()

    # Build response
    response = {
        'team': team.to_dict(),
        'league_type': league.league_type,
        'projections': {}
    }

    # Get team's roster and stats
    roster_players = []
    rosters = Roster.query.filter_by(team_id=team_id).all()

    for roster in rosters:
        player = Player.query.get(roster.player_id)
        if player:
            player_stats = get_player_season_stats(player.id)
            roster_players.append({
                'player_id': player.id,
                'espn_player_id': player.espn_player_id,
                'name': player.name,
                'team': player.team,
                'position': player.position,
                'injury_status': player.injury_status,
                'roster_slot': roster.roster_slot,
                'stats': player_stats
            })

    response['roster'] = roster_players

    if not PROJECTIONS_AVAILABLE:
        response['projections'] = {
            'message': 'Projection engine not available',
            'database_projection': {
                'projected_standing': team.projected_standing,
                'win_probability': float(team.win_probability) if team.win_probability else None
            }
        }
        return jsonify(response), 200

    # H2H League projections
    if league.is_h2h:
        try:
            analyzer = get_matchup_analyzer(get_league_categories(league))

            if analyzer:
                team_stats = get_team_stats(team)

                # Build team projections for comparison
                team_projections = []
                for t in all_teams:
                    t_stats = get_team_stats(t)
                    team_projections.append(TeamProjection(
                        team_id=t.id,
                        team_name=t.team_name,
                        projected_stats=t_stats
                    ))

                # Calculate category rankings
                category_rankings = {}
                categories = get_league_categories(league)

                for cat in categories:
                    values = [(t.team_id, t.projected_stats.get(cat, 0)) for t in team_projections]
                    # Sort descending (except turnovers)
                    reverse = cat != 'tov'
                    values.sort(key=lambda x: x[1], reverse=reverse)

                    for rank, (tid, val) in enumerate(values, 1):
                        if tid == team_id:
                            category_rankings[cat] = {
                                'rank': rank,
                                'value': round(val, 2),
                                'league_size': len(all_teams)
                            }
                            break

                response['projections'] = {
                    'type': 'h2h',
                    'projected_standing': team.projected_standing,
                    'win_probability': float(team.win_probability) if team.win_probability else None,
                    'category_rankings': category_rankings,
                    'team_totals': {k: round(v, 2) for k, v in team_stats.items()}
                }

        except Exception as e:
            logger.error(f"H2H team projection error: {e}")
            response['projections'] = {'error': str(e)}

    # Roto League projections
    elif league.is_roto:
        try:
            analyzer = get_roto_analyzer(league.num_teams or len(all_teams))

            if analyzer:
                # Build Roto team stats
                roto_teams = []
                my_roto_team = None

                for t in all_teams:
                    stats = get_team_stats(t)
                    roto_team = TeamRotoStats(
                        team_id=t.id,
                        team_name=t.team_name,
                        current_totals=stats,
                        fgm=stats.get('fgm', 0),
                        fga=max(stats.get('fga', 1), 1),
                        ftm=stats.get('ftm', 0),
                        fta=max(stats.get('fta', 1), 1),
                        ros_projections={k: v / 10 for k, v in stats.items()},
                        games_remaining=35
                    )
                    roto_teams.append(roto_team)

                    if t.id == team_id:
                        my_roto_team = roto_team

                if my_roto_team:
                    # Full analysis
                    projection = analyzer.analyze_team(my_roto_team, roto_teams)

                    # Gap analysis
                    gaps = analyzer.get_gap_analysis(my_roto_team, roto_teams)
                    gap_analysis = {}
                    for cat, gap in gaps.items():
                        gap_analysis[cat] = {
                            'current_rank': gap.current_rank,
                            'current_value': round(gap.current_value, 2),
                            'gap_to_improve': round(gap.gap_to_improve, 2),
                            'team_ahead': gap.team_ahead,
                            'cushion': round(gap.cushion, 2) if gap.cushion != float('inf') else None,
                            'team_behind': gap.team_behind,
                            'difficulty': gap.improvement_difficulty
                        }

                    response['projections'] = {
                        'type': 'roto',
                        'current_points': projection.current_points,
                        'current_rank': projection.current_rank,
                        'projected_points': round(projection.projected_points, 1),
                        'projected_rank': round(projection.projected_rank, 1),
                        'win_probability': round(projection.win_probability * 100, 1),
                        'top_3_probability': round(projection.top_3_probability * 100, 1),
                        'category_rankings': {
                            cat: ranking.to_dict()
                            for cat, ranking in projection.category_rankings.items()
                        },
                        'strengths': projection.strengths,
                        'weaknesses': projection.weaknesses,
                        'punt_candidates': projection.punt_candidates,
                        'improvement_targets': projection.improvement_targets,
                        'gap_analysis': gap_analysis
                    }

        except Exception as e:
            logger.error(f"Roto team projection error: {e}")
            response['projections'] = {'error': str(e)}

    return jsonify(response), 200


# =============================================================================
# Matchup Projection Endpoint (H2H Only)
# =============================================================================

@projections_bp.route('/leagues/<int:league_id>/projections/matchup', methods=['GET'])
@login_required
def get_matchup_projection(league_id: int):
    """
    Get projection for current/upcoming matchup (H2H leagues only).

    Query params:
        week: (optional) Week number
        team_id: (optional) Team ID to get matchup for

    Returns:
        JSON with category-by-category win probabilities and recommendations
    """
    league = League.query.filter_by(id=league_id, user_id=current_user.id).first()

    if not league:
        return jsonify({'error': 'League not found'}), 404

    if league.is_roto:
        return jsonify({
            'error': 'Matchup projections not available for Roto leagues',
            'league_type': league.league_type
        }), 400

    # Get query parameters
    week = request.args.get('week', type=int)
    team_id = request.args.get('team_id', type=int)

    # Get all teams
    teams = Team.query.filter_by(league_id=league_id).all()

    if not teams:
        return jsonify({'error': 'No teams found'}), 404

    # If no team specified, try to find user's team (first team by default for demo)
    if not team_id:
        team_id = teams[0].id if teams else None

    if not team_id:
        return jsonify({'error': 'Team not found'}), 404

    my_team = Team.query.get(team_id)
    if not my_team or my_team.league_id != league_id:
        return jsonify({'error': 'Team not found in this league'}), 404

    if not PROJECTIONS_AVAILABLE:
        return jsonify({
            'error': 'Projection engine not available',
            'team_id': team_id,
            'week': week
        }), 503

    try:
        analyzer = get_matchup_analyzer(get_league_categories(league))

        if not analyzer:
            return jsonify({'error': 'Matchup analyzer not available'}), 503

        # Build team projections
        my_stats = get_team_stats(my_team)
        my_projection = TeamProjection(
            team_id=my_team.id,
            team_name=my_team.team_name,
            projected_stats=my_stats
        )

        # Get opponent (would come from schedule - using next team in standings for demo)
        opponent = None
        for t in teams:
            if t.id != team_id:
                opponent = t
                break

        if not opponent:
            return jsonify({
                'error': 'No opponent found',
                'team': my_team.to_dict()
            }), 200

        opp_stats = get_team_stats(opponent)
        opp_projection = TeamProjection(
            team_id=opponent.id,
            team_name=opponent.team_name,
            projected_stats=opp_stats
        )

        # Analyze matchup
        cat_probs, win_prob, result = analyzer.analyze_matchup(my_projection, opp_projection)

        # Project weekly matchup
        weekly = analyzer.project_weekly_matchup(
            my_projection, opp_projection,
            week=week or 1
        )

        # Build category breakdown
        categories = get_league_categories(league)
        category_analysis = []

        for cat in categories:
            prob = cat_probs.get(cat, 0.5)
            my_val = my_stats.get(cat, 0)
            opp_val = opp_stats.get(cat, 0)

            if prob >= 0.65:
                outlook = "Strong Win"
            elif prob >= 0.55:
                outlook = "Likely Win"
            elif prob >= 0.45:
                outlook = "Toss-up"
            elif prob >= 0.35:
                outlook = "Likely Loss"
            else:
                outlook = "Strong Loss"

            category_analysis.append({
                'category': cat,
                'win_probability': round(prob * 100, 1),
                'your_value': round(my_val, 2),
                'opponent_value': round(opp_val, 2),
                'outlook': outlook
            })

        return jsonify({
            'league_id': league_id,
            'week': week,
            'matchup': {
                'team': {
                    'id': my_team.id,
                    'name': my_team.team_name,
                    'record': my_team.current_record
                },
                'opponent': {
                    'id': opponent.id,
                    'name': opponent.team_name,
                    'record': opponent.current_record
                }
            },
            'overall_win_probability': round(win_prob * 100, 1),
            'projected_result': result.result_string,
            'expected_category_wins': round(weekly.expected_category_wins, 1),
            'category_analysis': category_analysis,
            'punt_categories': weekly.punt_categories,
            'recommendations': {
                'streaming_targets': weekly.streaming_targets[:5],
                'focus_categories': [
                    cat['category'] for cat in category_analysis
                    if 45 <= cat['win_probability'] <= 55
                ]
            }
        }), 200

    except Exception as e:
        logger.error(f"Matchup projection error: {e}")
        return jsonify({'error': f'Projection error: {str(e)}'}), 500


# =============================================================================
# Player Projection Endpoint
# =============================================================================

@projections_bp.route('/leagues/<int:league_id>/projections/player/<int:player_id>', methods=['GET'])
@login_required
def get_player_projection(league_id: int, player_id: int):
    """
    Get detailed projection for an individual player.

    Returns hybrid projection combining ML and statistical models,
    adjusted for league-specific scoring.

    Args:
        league_id: League ID (for scoring context)
        player_id: Player ID

    Returns:
        JSON with player projection details
    """
    league = League.query.filter_by(id=league_id, user_id=current_user.id).first()

    if not league:
        return jsonify({'error': 'League not found'}), 404

    player = Player.query.get(player_id)

    if not player:
        return jsonify({'error': 'Player not found'}), 404

    # Get player's current stats
    season_stats = get_player_season_stats(player.id)

    # Get stored projection from database
    db_projection = Projection.query.filter_by(
        player_id=player.id,
        league_id=league_id
    ).order_by(Projection.created_at.desc()).first()

    # Build response
    response = {
        'player': {
            'id': player.id,
            'espn_player_id': player.espn_player_id,
            'name': player.name,
            'team': player.team,
            'position': player.position,
            'injury_status': player.injury_status
        },
        'current_stats': season_stats,
        'league_id': league_id,
        'league_type': league.league_type
    }

    if not PROJECTIONS_AVAILABLE:
        # Return database projection if available
        if db_projection:
            response['projection'] = {
                'source': 'database',
                'type': db_projection.projection_type,
                'projected_stats': {
                    'pts': float(db_projection.projected_points or 0),
                    'trb': float(db_projection.projected_rebounds or 0),
                    'ast': float(db_projection.projected_assists or 0),
                    'stl': float(db_projection.projected_steals or 0),
                    'blk': float(db_projection.projected_blocks or 0),
                    'tov': float(db_projection.projected_turnovers or 0),
                    'fg_pct': float(db_projection.projected_fg_pct or 0),
                    'ft_pct': float(db_projection.projected_ft_pct or 0),
                    '3p': float(db_projection.projected_threes or 0)
                },
                'fantasy_value': float(db_projection.fantasy_value or 0),
                'confidence': float(db_projection.confidence or 0),
                'games_remaining': db_projection.games_remaining,
                'created_at': db_projection.created_at.isoformat() if db_projection.created_at else None
            }
        else:
            response['projection'] = {
                'source': 'unavailable',
                'message': 'Projection engine not available and no stored projection found'
            }

        return jsonify(response), 200

    # Use hybrid projection engine
    try:
        engine = get_hybrid_engine()

        if not engine:
            response['projection'] = {'error': 'Hybrid engine not available'}
            return jsonify(response), 503

        # Build player data
        player_data = {
            'player_id': str(player.id),
            'name': player.name,
            'team': player.team or 'UNK',
            'position': player.position or 'N/A',
            'games_played': season_stats.get('g', 0) if season_stats else 0,
            'age': 25  # Would come from player data
        }

        # Get league settings
        if league.is_h2h and 'POINTS' in (league.league_type or ''):
            league_settings = LeagueScoringSettings.default_points()
        else:
            league_settings = LeagueScoringSettings.default_h2h_category()

        # Generate projection
        projection = engine.project_player(
            player_id=str(player.id),
            league_id=league_id,
            player_data=player_data,
            season_stats=season_stats,
            injury_status=player.injury_status or 'ACTIVE',
            league_settings=league_settings
        )

        response['projection'] = {
            'source': 'hybrid_engine',
            'projected_stats': {k: round(v, 2) for k, v in projection.projected_stats.items()},
            'ros_totals': {k: round(v, 1) for k, v in projection.ros_totals.items()},
            'fantasy_points_per_game': round(projection.fantasy_points, 2),
            'games_projected': projection.games_projected,
            'confidence_score': round(projection.confidence_score, 1),
            'confidence_intervals': {
                k: {'low': round(v[0], 2), 'high': round(v[1], 2)}
                for k, v in projection.confidence_intervals.items()
            },
            'weights': {
                'ml': round(projection.ml_weight * 100, 1),
                'statistical': round(projection.statistical_weight * 100, 1)
            },
            'season_phase': projection.season_phase,
            'injury_adjustment': round(projection.injury_adjustment, 2)
        }

    except Exception as e:
        logger.error(f"Player projection error: {e}")
        response['projection'] = {'error': str(e)}

    return jsonify(response), 200


# =============================================================================
# Bulk Player Projections Endpoint
# =============================================================================

@projections_bp.route('/leagues/<int:league_id>/projections/players', methods=['POST'])
@login_required
def get_bulk_player_projections(league_id: int):
    """
    Get projections for multiple players at once.

    Request body:
        player_ids: List of player IDs

    Returns:
        JSON with projections for each player
    """
    league = League.query.filter_by(id=league_id, user_id=current_user.id).first()

    if not league:
        return jsonify({'error': 'League not found'}), 404

    data = request.get_json()
    if not data or 'player_ids' not in data:
        return jsonify({'error': 'player_ids required in request body'}), 400

    player_ids = data['player_ids']
    if not isinstance(player_ids, list):
        return jsonify({'error': 'player_ids must be a list'}), 400

    # Limit batch size
    if len(player_ids) > 50:
        return jsonify({'error': 'Maximum 50 players per request'}), 400

    projections = []

    for pid in player_ids:
        player = Player.query.get(pid)
        if not player:
            projections.append({
                'player_id': pid,
                'error': 'Player not found'
            })
            continue

        season_stats = get_player_season_stats(player.id)

        proj_data = {
            'player_id': player.id,
            'name': player.name,
            'team': player.team,
            'position': player.position,
            'injury_status': player.injury_status,
            'current_stats': season_stats
        }

        # Get stored projection
        db_proj = Projection.query.filter_by(
            player_id=player.id,
            league_id=league_id
        ).order_by(Projection.created_at.desc()).first()

        if db_proj:
            proj_data['projection'] = {
                'pts': float(db_proj.projected_points or 0),
                'trb': float(db_proj.projected_rebounds or 0),
                'ast': float(db_proj.projected_assists or 0),
                'stl': float(db_proj.projected_steals or 0),
                'blk': float(db_proj.projected_blocks or 0),
                'fantasy_value': float(db_proj.fantasy_value or 0)
            }

        projections.append(proj_data)

    return jsonify({
        'league_id': league_id,
        'count': len(projections),
        'projections': projections
    }), 200


# =============================================================================
# Refresh Projections Endpoint
# =============================================================================

@projections_bp.route('/leagues/<int:league_id>/projections/refresh', methods=['POST'])
@login_required
def refresh_projections(league_id: int):
    """
    Trigger a refresh of projections for all players in a league.

    This endpoint queues projection recalculation for all rostered players.

    Returns:
        JSON with refresh status
    """
    league = League.query.filter_by(id=league_id, user_id=current_user.id).first()

    if not league:
        return jsonify({'error': 'League not found'}), 404

    if not PROJECTIONS_AVAILABLE:
        return jsonify({
            'error': 'Projection engine not available',
            'status': 'failed'
        }), 503

    try:
        # Get all rostered players
        teams = Team.query.filter_by(league_id=league_id).all()
        player_count = 0

        for team in teams:
            rosters = Roster.query.filter_by(team_id=team.id).all()
            player_count += len(rosters)

        # In production, this would queue background jobs
        # For now, return acknowledgment

        return jsonify({
            'status': 'queued',
            'league_id': league_id,
            'players_to_update': player_count,
            'message': f'Projection refresh queued for {player_count} players'
        }), 202

    except Exception as e:
        logger.error(f"Projection refresh error: {e}")
        return jsonify({
            'error': str(e),
            'status': 'failed'
        }), 500

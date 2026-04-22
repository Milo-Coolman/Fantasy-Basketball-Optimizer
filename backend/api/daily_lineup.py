"""
Daily Lineup API - Provides optimized daily lineups for fantasy basketball teams.

This module exposes endpoints for getting day-by-day optimized lineup assignments,
showing which players should start and which should be benched based on z-score values.
"""

from flask import Blueprint, jsonify, request
from datetime import datetime, date, timedelta
import logging

from backend.models import League
from backend.projections.start_limit_optimizer import (
    StartLimitOptimizer,
    SLOT_ID_TO_POSITION,
    STARTING_SLOT_IDS
)

logger = logging.getLogger(__name__)

daily_lineup_bp = Blueprint('daily_lineup', __name__)

# Cache for simulation results (league_id -> {timestamp, results})
_simulation_cache = {}
CACHE_TTL_SECONDS = 300  # 5 minutes


def get_current_nba_season() -> int:
    """
    Auto-detect the current NBA season based on today's date.

    NBA seasons span two calendar years (e.g., 2025-26 season).
    ESPN uses the ending year as the season identifier.

    - Oct-Dec: season = current_year + 1 (e.g., Oct 2025 -> 2026)
    - Jan-Sep: season = current_year (e.g., Mar 2026 -> 2026)
    """
    today = date.today()
    if today.month >= 10:  # October or later
        return today.year + 1
    else:
        return today.year


def get_cached_simulation(league_id: int, league: League):
    """
    Get cached simulation results or run new simulation.

    Returns SeasonSimulationResult from the optimizer.
    """
    cache_key = league_id
    now = datetime.now()

    # Check cache
    if cache_key in _simulation_cache:
        cached = _simulation_cache[cache_key]
        age_seconds = (now - cached['timestamp']).total_seconds()
        if age_seconds < CACHE_TTL_SECONDS:
            logger.info(f'Using cached simulation for league {league_id} (age: {age_seconds:.0f}s)')
            return cached['results'], cached.get('slots', []), cached.get('user_roster', [])

    # Run new simulation
    logger.info(f'Running new simulation for league {league_id}')

    # Auto-detect season if not set on league
    season = league.season or get_current_nba_season()
    logger.info(f'Using season {season} for simulation')

    optimizer = StartLimitOptimizer(
        espn_s2=league.espn_s2_cookie,
        swid=league.swid_cookie,
        league_id=league.espn_league_id,
        season=season,
        verbose=False,
        projection_method=league.projection_method or 'adaptive',
        flat_game_rate=league.flat_game_rate or 0.85
    )

    # Fetch required data
    categories = optimizer.fetch_scoring_categories()
    roster_data = optimizer._fetch_all_teams_rosters()

    # Calculate league-wide averages for z-score calculations
    # This is required for proper player value calculation
    all_rosters_list = list(roster_data.values())
    optimizer.calculate_league_averages(all_rosters_list, categories)
    logger.info(f'Calculated league averages from {len(all_rosters_list)} teams')

    # Get user's team ID - either from query param or from ESPN
    user_team_id = request.args.get('team_id', type=int)

    if not user_team_id:
        # Get user's team from ESPN client
        from backend.services.espn_client import ESPNClient
        espn_client = ESPNClient(
            league_id=league.espn_league_id,
            year=season,
            espn_s2=league.espn_s2_cookie,
            swid=league.swid_cookie
        )
        user_team_id = espn_client.get_user_team_id()
        logger.info(f'User team ID from ESPN: {user_team_id}')

    if user_team_id and user_team_id in roster_data:
        roster = roster_data[user_team_id]
        logger.info(f'Using roster for team {user_team_id}: {len(roster)} players')
    else:
        # Fallback to first team
        logger.warning(f'Team {user_team_id} not found, using first team')
        roster = list(roster_data.values())[0] if roster_data else []

    # Calculate projected_games for each player based on team schedule
    # This is needed because _fetch_all_teams_rosters doesn't include projected_games
    start_date, end_date = optimizer.get_season_dates()
    game_rate = league.flat_game_rate or 0.85

    # Statuses that indicate player is unavailable
    UNAVAILABLE_STATUSES = {'SUSPENSION', 'INACTIVE', 'SUSPENDED'}

    for player in roster:
        nba_team = player.get('nba_team')

        # Check if player is out for the season
        injury_details = player.get('injury_details')
        out_for_season = False
        if injury_details and isinstance(injury_details, dict):
            out_for_season = injury_details.get('out_for_season', False)

        # Check if player is suspended or inactive
        injury_status = player.get('injury_status', '')
        is_suspended = injury_status and injury_status.upper() in UNAVAILABLE_STATUSES

        if out_for_season:
            # Player is out for the season - no projected games
            player['projected_games'] = 0
            player['original_projected_games'] = 0
            logger.info(f"Player {player.get('name')}: OUT FOR SEASON, projected_games=0")
        elif is_suspended:
            # Player is suspended/inactive - no projected games
            player['projected_games'] = 0
            player['original_projected_games'] = 0
            logger.info(f"Player {player.get('name')}: {injury_status}, projected_games=0")
        elif nba_team:
            # Get remaining games for this player's team
            remaining_games = optimizer.get_player_nba_team_schedule(
                nba_team, start_date, end_date
            )
            team_remaining = len(remaining_games)

            # Apply game rate to get projected games
            # Use adaptive or flat rate based on league settings
            projected = int(team_remaining * game_rate)
            # Set both field names - simulator uses 'original_projected_games'
            player['projected_games'] = max(projected, 1)
            player['original_projected_games'] = max(projected, 1)
            logger.debug(f"Player {player.get('name')}: {team_remaining} team games * {game_rate:.0%} = {player['projected_games']} projected")
        else:
            player['projected_games'] = 0
            player['original_projected_games'] = 0

    # Fetch actual starts used from ESPN for accurate start limit tracking
    starts_used = optimizer.fetch_starts_used(user_team_id)
    logger.info(f'Fetched starts_used for team {user_team_id}: {starts_used}')

    # Run simulation with actual starts used
    results = optimizer.simulate_season(
        roster=roster,
        categories=categories,
        include_ir_returns=True,
        initial_starts_used=starts_used,
        team_id=user_team_id
    )

    # Get expanded slots list for proper position mapping
    slots = optimizer.expand_lineup_slots()

    # Cache results
    _simulation_cache[cache_key] = {
        'timestamp': now,
        'results': results,
        'roster_data': roster_data,
        'user_roster': roster,  # Store user's roster with injury data
        'slots': slots  # Store for position mapping
    }

    return results, slots, roster


@daily_lineup_bp.route('/leagues/<int:league_id>/daily-lineup', methods=['GET'])
def get_daily_lineup(league_id):
    """
    Get optimized lineup for a specific date.

    Query params:
        - date: YYYY-MM-DD format, defaults to today
        - team_id: ESPN team ID (optional, defaults to user's team)

    Returns:
        - lineup_slots: {position: {player, z_score, game_info}}
        - bench: [players not starting]
        - ir: [injured reserve players]
        - starts_used/starts_limit per position
        - date info
    """
    try:
        # Get date from query param or default to today
        date_str = request.args.get('date')
        if date_str:
            try:
                selected_date = datetime.strptime(date_str, '%Y-%m-%d').date()
            except ValueError:
                return jsonify({'error': 'Invalid date format. Use YYYY-MM-DD'}), 400
        else:
            selected_date = date.today()

        logger.info(f'=== DAILY LINEUP REQUEST ===')
        logger.info(f'League: {league_id}, Date: {selected_date}')

        # Get league
        league = League.query.get_or_404(league_id)

        # Get simulation results (cached or fresh)
        results, slots, user_roster = get_cached_simulation(league_id, league)

        # Build player injury lookup from roster
        player_injury_status = {}
        for player in user_roster:
            pid = player.get('player_id')
            if pid:
                player_injury_status[pid] = player.get('injury_status')

        # Build slot index to position name mapping
        slot_idx_to_position = {}
        for idx, (slot_id, slot_name) in enumerate(slots):
            # Use the slot_name which already handles duplicates (e.g., UTIL_1, UTIL_2)
            slot_idx_to_position[idx] = slot_name

        # Find the day simulation for selected date
        daily_sim = None
        for day_sim in results.daily_simulations:
            if day_sim.game_date == selected_date:
                daily_sim = day_sim
                break

        # Check if date is outside simulation range entirely
        if not daily_sim and (selected_date < results.start_date or selected_date > results.end_date):
            available_dates = [
                d.game_date.strftime('%Y-%m-%d')
                for d in results.daily_simulations[:10]
            ]
            return jsonify({
                'error': f'No lineup available for {selected_date}',
                'message': 'Date is outside the simulation range',
                'available_dates': available_dates,
                'simulation_range': {
                    'start': results.start_date.strftime('%Y-%m-%d'),
                    'end': results.end_date.strftime('%Y-%m-%d')
                }
            }), 404

        # If no daily_sim but date is within range, it means no games today
        # Show all players in the "no game" section
        if not daily_sim:
            logger.info(f'No games on {selected_date} - showing all players as no game')

            # Build player info from roster
            no_game_players = []
            ir_players = []

            for player in user_roster:
                pid = player.get('player_id')
                lineup_slot_id = player.get('lineupSlotId', 0)
                injury_status = player.get('injury_status')

                player_entry = {
                    'player': {
                        'id': pid,
                        'name': player.get('name', 'Unknown'),
                        'team': player.get('nba_team', ''),
                        'positions': [SLOT_ID_TO_POSITION.get(s, str(s)) for s in player.get('eligible_slots', [])]
                    },
                    'z_score': player.get('per_game_value', 0),
                    'has_game_today': False
                }

                if injury_status:
                    player_entry['injury_status'] = injury_status

                # Check if player is on IR
                if lineup_slot_id == 13:  # IR slot
                    ir_players.append(player_entry)
                else:
                    no_game_players.append(player_entry)

            # Sort by z-score
            no_game_players.sort(key=lambda p: p['z_score'], reverse=True)

            return jsonify({
                'date': selected_date.strftime('%Y-%m-%d'),
                'day_of_week': selected_date.strftime('%A'),
                'formatted_date': selected_date.strftime('%B %d, %Y'),
                'lineup_slots': {},
                'bench': [],
                'injured': [],
                'no_game': no_game_players,
                'ir': ir_players,
                'summary': {
                    'players_with_games': 0,
                    'starters': 0,
                    'benched': 0,
                    'injured': 0,
                    'no_game': len(no_game_players)
                },
                'simulation_info': {
                    'start_date': results.start_date.strftime('%Y-%m-%d'),
                    'end_date': results.end_date.strftime('%Y-%m-%d'),
                    'total_days': results.total_days,
                    'game_days': results.game_days
                },
                'start_limits': {
                    'position_starts_used': {
                        SLOT_ID_TO_POSITION.get(k, str(k)): v
                        for k, v in results.position_starts_used.items()
                        if k in STARTING_SLOT_IDS
                    },
                    'position_limits': {
                        SLOT_ID_TO_POSITION.get(k, str(k)): v
                        for k, v in results.position_limits.items()
                        if k in STARTING_SLOT_IDS
                    }
                },
                'no_games_today': True
            })

        # Build lineup slots from assignments
        lineup_slots = {}
        benched_players = []

        # Get player info from player_logs
        player_info = {}
        for pid, log in results.player_logs.items():
            player_info[pid] = {
                'player_id': log.player_id,
                'name': log.player_name,
                'nba_team': log.nba_team,
                'eligible_positions': [SLOT_ID_TO_POSITION.get(s, str(s)) for s in log.eligible_slots],
                'z_score_value': log.per_game_value
            }

        # Process assignments (slot_idx -> player_id)
        # Note: assignments keys are slot INDICES, not slot IDs
        for slot_idx, player_id in daily_sim.assignments.items():
            # Map index to position name using the slots list
            position = slot_idx_to_position.get(slot_idx, f'SLOT_{slot_idx}')
            player = player_info.get(player_id, {})

            # Get the actual slot_id for this index (for reference)
            actual_slot_id = slots[slot_idx][0] if slot_idx < len(slots) else slot_idx

            lineup_slots[position] = {
                'player': {
                    'id': player.get('player_id'),
                    'name': player.get('name', 'Unknown'),
                    'team': player.get('nba_team', ''),
                    'positions': player.get('eligible_positions', [])
                },
                'z_score': player.get('z_score_value', 0),
                'slot_id': actual_slot_id,
                'game': {
                    'has_game': True,
                    'opponent': 'TBD',  # Would need schedule data
                    'is_home': True,
                    'time': 'TBD'
                }
            }

        # Process benched players
        for player_id in daily_sim.benched_players:
            player = player_info.get(player_id, {})
            benched_players.append({
                'player': {
                    'id': player.get('player_id'),
                    'name': player.get('name', 'Unknown'),
                    'team': player.get('nba_team', ''),
                    'positions': player.get('eligible_positions', [])
                },
                'z_score': player.get('z_score_value', 0),
                'has_game_today': True,
                'game': {
                    'has_game': True,
                    'opponent': 'TBD',
                    'is_home': True,
                    'time': 'TBD'
                }
            })

        # Sort bench by z-score (highest first)
        benched_players.sort(key=lambda p: p['z_score'], reverse=True)

        # Build set of teams with games today
        # First from players in simulation
        teams_with_games_today = set()
        for pid in daily_sim.players_with_games:
            if pid in player_info:
                teams_with_games_today.add(player_info[pid].get('nba_team'))

        # Also check all roster players' teams using the NBA schedule
        # This catches OUT players whose team has a game
        try:
            from backend.scrapers.nba_schedule import NBASchedule
            season = league.season or get_current_nba_season()
            nba_schedule = NBASchedule(season=season)
            for player in user_roster:
                nba_team = player.get('nba_team')
                if nba_team and nba_team not in teams_with_games_today:
                    # Check if this team has a game today
                    team_games = nba_schedule.get_team_remaining_games(nba_team, selected_date)
                    if team_games and selected_date in team_games:
                        teams_with_games_today.add(nba_team)
        except ImportError:
            logger.warning("NBASchedule not available, cannot check injured player games")

        # Process players not starting or benched
        # Separate into: injured (OUT + team has game), no_game (team has no game)
        no_game_players = []
        injured_players = []
        players_with_games_set = set(daily_sim.players_with_games)
        assigned_players = set(daily_sim.assignments.values())
        benched_player_ids = set(daily_sim.benched_players)
        ir_player_ids = {ir.player_id for ir in results.ir_players}

        # Get all roster player IDs
        all_roster_pids = set(p.get('player_id') for p in user_roster if p.get('player_id'))

        for pid in all_roster_pids:
            # Skip if already accounted for (starter, bench, IR)
            if pid in assigned_players or pid in benched_player_ids or pid in ir_player_ids:
                continue

            # Get player info - prefer from player_logs, fallback to roster
            if pid in player_info:
                info = player_info[pid]
            else:
                # Find in user_roster
                roster_player = next((p for p in user_roster if p.get('player_id') == pid), None)
                if not roster_player:
                    continue
                info = {
                    'player_id': pid,
                    'name': roster_player.get('name', 'Unknown'),
                    'nba_team': roster_player.get('nba_team', ''),
                    'eligible_positions': [SLOT_ID_TO_POSITION.get(s, str(s)) for s in roster_player.get('eligible_slots', [])],
                    'z_score_value': roster_player.get('per_game_value', 0)
                }

            nba_team = info.get('nba_team', '')
            injury_status = player_injury_status.get(pid)
            team_has_game = nba_team in teams_with_games_today

            player_entry = {
                'player': {
                    'id': info.get('player_id'),
                    'name': info.get('name', 'Unknown'),
                    'team': nba_team,
                    'positions': info.get('eligible_positions', [])
                },
                'z_score': info.get('z_score_value', 0),
            }

            # Check if player is OUT and their team has a game
            if injury_status and injury_status.upper() == 'OUT' and team_has_game:
                player_entry['injury_status'] = injury_status
                player_entry['has_game_today'] = True
                injured_players.append(player_entry)
            elif not team_has_game:
                player_entry['has_game_today'] = False
                # Include injury status if present (DTD, etc.)
                if injury_status:
                    player_entry['injury_status'] = injury_status
                no_game_players.append(player_entry)
            # else: player has game but not OUT - shouldn't happen normally

        # Sort by z-score (highest first)
        no_game_players.sort(key=lambda p: p['z_score'], reverse=True)
        injured_players.sort(key=lambda p: p['z_score'], reverse=True)

        # Process IR players
        ir_players = []
        for ir_info in results.ir_players:
            ir_players.append({
                'player': {
                    'id': ir_info.player_id,
                    'name': ir_info.player_name,
                    'team': ir_info.nba_team,
                    'positions': [SLOT_ID_TO_POSITION.get(s, str(s)) for s in ir_info.eligible_slots]
                },
                'z_score': ir_info.per_game_value,
                'injury_status': ir_info.injury_status,
                'return_date': ir_info.projected_return_date.strftime('%Y-%m-%d') if ir_info.projected_return_date else None
            })

        # Build response
        response = {
            'date': selected_date.strftime('%Y-%m-%d'),
            'day_of_week': selected_date.strftime('%A'),
            'formatted_date': selected_date.strftime('%B %d, %Y'),
            'lineup_slots': lineup_slots,
            'bench': benched_players,
            'injured': injured_players,
            'no_game': no_game_players,
            'ir': ir_players,
            'summary': {
                'players_with_games': len(daily_sim.players_with_games) + len(injured_players),
                'starters': len(daily_sim.assignments),
                'benched': len(daily_sim.benched_players),
                'injured': len(injured_players),
                'no_game': len(no_game_players)
            },
            'simulation_info': {
                'start_date': results.start_date.strftime('%Y-%m-%d'),
                'end_date': results.end_date.strftime('%Y-%m-%d'),
                'total_days': results.total_days,
                'game_days': results.game_days
            },
            'start_limits': {
                'position_starts_used': {
                    SLOT_ID_TO_POSITION.get(k, str(k)): v
                    for k, v in results.position_starts_used.items()
                    if k in STARTING_SLOT_IDS
                },
                'position_limits': {
                    SLOT_ID_TO_POSITION.get(k, str(k)): v
                    for k, v in results.position_limits.items()
                    if k in STARTING_SLOT_IDS
                }
            }
        }

        logger.info(f'Returning lineup: {len(lineup_slots)} starters, {len(benched_players)} benched')

        return jsonify(response)

    except Exception as e:
        logger.error(f'Error getting daily lineup: {e}', exc_info=True)
        return jsonify({'error': str(e)}), 500


@daily_lineup_bp.route('/leagues/<int:league_id>/daily-lineup/dates', methods=['GET'])
def get_available_dates(league_id):
    """
    Get list of available dates for daily lineup.

    Returns:
        - dates: list of available dates with game counts
        - range: start and end of simulation
    """
    try:
        league = League.query.get_or_404(league_id)
        results, _ = get_cached_simulation(league_id, league)

        dates = []
        for day_sim in results.daily_simulations:
            dates.append({
                'date': day_sim.game_date.strftime('%Y-%m-%d'),
                'day_of_week': day_sim.game_date.strftime('%A'),
                'players_with_games': len(day_sim.players_with_games),
                'starters': len(day_sim.assignments),
                'benched': len(day_sim.benched_players)
            })

        return jsonify({
            'dates': dates,
            'range': {
                'start': results.start_date.strftime('%Y-%m-%d'),
                'end': results.end_date.strftime('%Y-%m-%d')
            },
            'total_days': len(dates)
        })

    except Exception as e:
        logger.error(f'Error getting available dates: {e}', exc_info=True)
        return jsonify({'error': str(e)}), 500


@daily_lineup_bp.route('/leagues/<int:league_id>/daily-lineup/clear-cache', methods=['POST'])
def clear_lineup_cache(league_id):
    """Clear the simulation cache for a league."""
    if league_id in _simulation_cache:
        del _simulation_cache[league_id]
        return jsonify({'message': 'Cache cleared', 'league_id': league_id})
    return jsonify({'message': 'No cache to clear', 'league_id': league_id})

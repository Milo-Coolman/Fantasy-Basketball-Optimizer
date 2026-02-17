#!/usr/bin/env python3
"""
Debug script to analyze the IR drop decision logic.

This script shows exactly what's happening in the Roto optimization:
1. Your team's CURRENT stats (accumulated this season)
2. Rest-of-season projections for each player
3. Category rankings vs all teams
4. Roto point calculations
5. Before/after comparison for dropping Jaylen Brown

Usage:
    cd /Users/milo/fantasy-basketball-optimizer
    source venv/bin/activate
    python backend/debug_drop_decision.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlite3
from datetime import date, timedelta
from collections import defaultdict
from backend.services.espn_client import ESPNClient
from backend.projections.start_limit_optimizer import (
    StartLimitOptimizer, create_optimizer_from_league,
    STARTING_SLOT_IDS, IR_SLOT_ID, DEFAULT_GAMES_PER_SLOT,
    ESPN_STAT_ID_MAP, PERCENTAGE_CATEGORIES, PERCENTAGE_COMPONENTS, REVERSE_CATEGORIES
)


def get_credentials():
    """Get league credentials from database."""
    db_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'instance', 'fantasy_basketball.db'
    )
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT espn_league_id, espn_s2_cookie, swid_cookie, season
        FROM leagues LIMIT 1
    ''')
    row = cursor.fetchone()
    conn.close()
    return row


def fetch_all_teams_stats(optimizer, my_team_id):
    """Fetch current season stats for all teams including all component stats."""
    import requests

    endpoint = (
        f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/fba/"
        f"seasons/{optimizer.season}/segments/0/leagues/{optimizer.league_id}"
    )
    params = {'view': ['mTeam', 'mRoster']}
    cookies = {'espn_s2': optimizer.espn_s2, 'SWID': optimizer.swid}

    response = requests.get(endpoint, params=params, cookies=cookies, timeout=15)
    response.raise_for_status()
    data = response.json()

    all_teams = {}
    team_names = {}

    for team in data.get('teams', []):
        team_id = team.get('id')
        team_name = team.get('name', '') or team.get('nickname', f'Team {team_id}')
        team_names[team_id] = team_name

        values_by_stat = team.get('valuesByStat', {})
        team_stats = {}

        # Use full stat ID map to get all stats
        for stat_id_str, value in values_by_stat.items():
            stat_id = int(stat_id_str)
            stat_key = ESPN_STAT_ID_MAP.get(stat_id)
            if stat_key:
                team_stats[stat_key] = float(value)

        all_teams[team_id] = team_stats

    return all_teams, team_names


def calculate_percentage(stats, pct_stat):
    """Calculate percentage from component totals."""
    if pct_stat not in PERCENTAGE_COMPONENTS:
        return 0.0
    made_key, attempt_key = PERCENTAGE_COMPONENTS[pct_stat]
    made = stats.get(made_key, 0)
    attempts = stats.get(attempt_key, 0)
    if attempts > 0:
        return made / attempts
    return 0.0


def simulate_ros_stats(optimizer, roster, sim_start, end_date, initial_starts_used, category_keys):
    """Simulate rest-of-season stats for a roster."""
    stat_limits = optimizer.get_lineup_slot_stat_limits()
    slots = optimizer.expand_lineup_slots()

    # Build player info
    player_info = {}
    player_schedules = {}

    for player in roster:
        pid = player.get('player_id', 0)
        if player.get('lineupSlotId', 0) == IR_SLOT_ID:
            continue

        nba_team = player.get('nba_team', 'UNK')
        eligible_slots = [s for s in player.get('eligible_slots', []) if s in STARTING_SLOT_IDS]
        per_game_value = player.get('per_game_value', 0)
        per_game_stats = player.get('per_game_stats', {})

        player_info[pid] = {
            'name': player.get('name', 'Unknown'),
            'nba_team': nba_team,
            'eligible_slots': eligible_slots,
            'per_game_value': per_game_value,
            'per_game_stats': per_game_stats,
        }

        team_games = optimizer.get_player_nba_team_schedule(nba_team, sim_start, end_date)
        player_schedules[pid] = set(team_games)

    # Track starts and stats per player
    starts_used = defaultdict(int, initial_starts_used)
    player_starts = defaultdict(int)
    player_ros_stats = defaultdict(lambda: defaultdict(float))
    total_stats = defaultdict(float)

    # Simulate each day
    current_date = sim_start
    while current_date <= end_date:
        # Find players with games today
        players_today = []
        for pid, games in player_schedules.items():
            if current_date in games:
                info = player_info[pid]
                players_today.append((pid, info['eligible_slots'], info['per_game_value'], info['per_game_stats']))

        # Sort by value (highest first)
        players_today.sort(key=lambda x: x[2], reverse=True)

        # Greedy assignment
        slots_used_today = set()
        assigned_today = set()

        for pid, elig, val, stats in players_today:
            if pid in assigned_today:
                continue

            for slot_id, slot_name in slots:
                if slot_id in slots_used_today:
                    continue
                if slot_id not in elig:
                    continue
                limit = stat_limits.get(slot_id, DEFAULT_GAMES_PER_SLOT)
                if starts_used[slot_id] >= limit:
                    continue

                # Assign player
                slots_used_today.add(slot_id)
                assigned_today.add(pid)
                starts_used[slot_id] += 1
                player_starts[pid] += 1

                # Add stats
                for cat in category_keys:
                    stat_val = stats.get(cat, 0) or stats.get(cat.upper(), 0)
                    player_ros_stats[pid][cat] += stat_val
                    total_stats[cat] += stat_val
                break

        current_date += timedelta(days=1)

    return dict(total_stats), dict(player_starts), {k: dict(v) for k, v in player_ros_stats.items()}, player_info


def rank_teams_in_category(all_teams_eos, category, my_team_id, is_reverse=False):
    """Rank all teams in a single category. Returns (my_rank, sorted_list)."""
    team_values = [(tid, stats.get(category, 0)) for tid, stats in all_teams_eos.items()]

    # Higher is better unless is_reverse (e.g., turnovers)
    reverse = not is_reverse
    team_values.sort(key=lambda x: x[1], reverse=reverse)

    my_rank = None
    for rank, (tid, val) in enumerate(team_values, 1):
        if tid == my_team_id:
            my_rank = rank
            break

    return my_rank, team_values


def main():
    print("=" * 80)
    print("DEBUG: IR DROP DECISION ANALYSIS")
    print("=" * 80)

    # Get credentials
    creds = get_credentials()
    if not creds:
        print("No league found in database")
        return 1

    league_id, espn_s2, swid, season = creds
    print(f"\nLeague ID: {league_id}, Season: {season}")

    # Create client and optimizer
    client = ESPNClient(league_id, season, espn_s2, swid)
    optimizer = create_optimizer_from_league(league_id, espn_s2, swid, season)

    # Find my team
    rosters = client.get_all_rosters()
    teams = client.get_teams()

    my_team_id = None
    my_team_name = None
    for team in teams:
        if 'doncic' in team.get('team_name', '').lower():
            my_team_id = team.get('team_id')
            my_team_name = team.get('team_name')
            break

    if not my_team_id:
        my_team_id = teams[0].get('team_id')
        my_team_name = teams[0].get('team_name')

    print(f"My Team: {my_team_name} (ID: {my_team_id})")

    # ==========================================================================
    # FETCH LEAGUE SCORING CATEGORIES (DYNAMIC)
    # ==========================================================================
    print("\n" + "=" * 80)
    print("LEAGUE SCORING CATEGORIES")
    print("=" * 80)

    # Fetch actual scoring categories from ESPN
    scoring_categories = optimizer.fetch_scoring_categories()
    num_teams = len(client.get_teams())

    print(f"\nLeague has {len(scoring_categories)} scoring categories:")
    for cat in scoring_categories:
        pct_marker = " (percentage)" if cat.get('is_percentage') else ""
        rev_marker = " (lower=better)" if cat.get('is_reverse') else ""
        print(f"  - {cat['stat_key'].upper()}{pct_marker}{rev_marker}")

    print(f"\nMax possible Roto points: {num_teams} teams × {len(scoring_categories)} categories = {num_teams * len(scoring_categories)}")

    # Build list of all stat keys needed (categories + components for percentages)
    category_keys = [c['stat_key'] for c in scoring_categories]
    all_stat_keys = set(category_keys)
    for cat in scoring_categories:
        if cat.get('is_percentage') and cat['stat_key'] in PERCENTAGE_COMPONENTS:
            made, attempts = PERCENTAGE_COMPONENTS[cat['stat_key']]
            all_stat_keys.add(made)
            all_stat_keys.add(attempts)
    all_stat_keys = list(all_stat_keys)

    # ==========================================================================
    # SECTION 1: Current stats from ESPN
    # ==========================================================================
    print("\n" + "=" * 80)
    print("1. YOUR TEAM'S CURRENT STATS (Accumulated this season from ESPN)")
    print("=" * 80)

    all_teams_current, team_names = fetch_all_teams_stats(optimizer, my_team_id)
    my_current = all_teams_current.get(my_team_id, {})

    print(f"\n{my_team_name}'s Current Season Stats:")
    for cat in scoring_categories:
        stat_key = cat['stat_key']
        if cat.get('is_percentage'):
            # Calculate percentage from component totals
            val = calculate_percentage(my_current, stat_key)
            print(f"  {stat_key.upper():>6}: {val:.1%}")
        else:
            val = my_current.get(stat_key, 0)
            print(f"  {stat_key.upper():>6}: {val:,.1f}")

    # ==========================================================================
    # SECTION 2: Rest-of-season projections
    # ==========================================================================
    print("\n" + "=" * 80)
    print("2. REST-OF-SEASON PROJECTIONS (Per Player)")
    print("=" * 80)

    # Get season dates
    start_date, end_date = optimizer.get_season_dates()
    initial_starts = optimizer.fetch_starts_used(my_team_id)

    print(f"\nSimulation period: {start_date} to {end_date}")
    print(f"Initial starts used: {dict(initial_starts)}")

    # Build roster data using hybrid projection engine
    players = rosters.get(my_team_id, [])

    print("\nBuilding roster with hybrid projections...")
    roster_data = optimizer.build_roster_with_hybrid_projections(
        espn_players=players,
        stat_keys=all_stat_keys
    )
    print(f"  Built roster data for {len(roster_data)} players")

    # Simulate current roster
    ros_totals, player_starts, player_ros_stats, player_info = simulate_ros_stats(
        optimizer, roster_data, start_date, end_date, initial_starts, category_keys
    )

    print("\nPlayer ROS Projections (based on starts they'd actually get):")
    print(f"{'Player':<20} {'Starts':<8} {'PTS':<10} {'REB':<10} {'AST':<10} {'STL':<8} {'BLK':<8} {'3PM':<8}")
    print("-" * 90)

    for pid, starts in sorted(player_starts.items(), key=lambda x: x[1], reverse=True):
        info = player_info.get(pid, {})
        name = info.get('name', 'Unknown')[:18]
        ros = player_ros_stats.get(pid, {})

        print(f"{name:<20} {starts:<8} {ros.get('pts', 0):<10.1f} {ros.get('reb', 0):<10.1f} "
              f"{ros.get('ast', 0):<10.1f} {ros.get('stl', 0):<8.1f} {ros.get('blk', 0):<8.1f} "
              f"{ros.get('3pm', 0):<8.1f}")

    print("-" * 90)
    print(f"{'ROS TOTALS':<20} {'':<8} {ros_totals.get('pts', 0):<10.1f} {ros_totals.get('reb', 0):<10.1f} "
          f"{ros_totals.get('ast', 0):<10.1f} {ros_totals.get('stl', 0):<8.1f} {ros_totals.get('blk', 0):<8.1f} "
          f"{ros_totals.get('3pm', 0):<8.1f}")

    # ==========================================================================
    # SECTION 3: End-of-season projections and rankings
    # ==========================================================================
    print("\n" + "=" * 80)
    print("3. END-OF-SEASON PROJECTIONS & RANKINGS (Current Roster)")
    print("=" * 80)

    # My EOS totals (raw stats including components)
    my_eos_raw = {}
    for stat_key in all_stat_keys:
        current = my_current.get(stat_key, 0)
        ros = ros_totals.get(stat_key, 0)
        my_eos_raw[stat_key] = current + ros

    # Calculate final category values including percentages
    my_eos = {}
    print(f"\n{my_team_name}'s End-of-Season Totals:")
    print(f"{'Category':<10} {'Current':<12} {'ROS Proj':<12} {'EOS Total':<12}")
    print("-" * 50)
    for cat in scoring_categories:
        stat_key = cat['stat_key']
        if cat.get('is_percentage'):
            current_pct = calculate_percentage(my_current, stat_key)
            eos_pct = calculate_percentage(my_eos_raw, stat_key)
            my_eos[stat_key] = eos_pct
            print(f"{stat_key.upper():<10} {current_pct:<12.1%} {'N/A':<12} {eos_pct:<12.1%}")
        else:
            current = my_current.get(stat_key, 0)
            ros = ros_totals.get(stat_key, 0)
            eos = current + ros
            my_eos[stat_key] = eos
            print(f"{stat_key.upper():<10} {current:<12.1f} {ros:<12.1f} {eos:<12.1f}")

    # Project other teams (simple pace projection)
    all_teams_eos = {}
    days_played = (start_date - date(season - 1, 10, 21)).days
    games_played = max(1, int(days_played / 7 * 3.5))
    days_remaining = (end_date - start_date).days
    games_remaining = int(days_remaining / 7 * 3.5)

    print(f"\n(Other teams projected assuming they maintain current pace)")
    print(f"Days played: {days_played}, Games estimated: {games_played}")
    print(f"Days remaining: {days_remaining}, Games remaining: {games_remaining}")

    for team_id, current_stats in all_teams_current.items():
        if team_id == my_team_id:
            all_teams_eos[team_id] = my_eos
        else:
            # Project raw stats for other teams
            eos_raw = {}
            for stat_key in all_stat_keys:
                current = current_stats.get(stat_key, 0)
                per_game = current / games_played if games_played > 0 else 0
                ros = per_game * games_remaining
                eos_raw[stat_key] = current + ros

            # Calculate category values including percentages
            eos = {}
            for cat in scoring_categories:
                stat_key = cat['stat_key']
                if cat.get('is_percentage'):
                    eos[stat_key] = calculate_percentage(eos_raw, stat_key)
                else:
                    eos[stat_key] = eos_raw.get(stat_key, 0)
            all_teams_eos[team_id] = eos

    # Rank in each category
    print("\n" + "-" * 80)
    print("CATEGORY RANKINGS (Current Roster)")
    print("-" * 80)

    total_roto = 0
    num_teams = len(all_teams_eos)
    num_categories = len(scoring_categories)

    for cat in scoring_categories:
        stat_key = cat['stat_key']
        is_reverse = cat.get('is_reverse', False)
        is_percentage = cat.get('is_percentage', False)

        my_rank, team_values = rank_teams_in_category(all_teams_eos, stat_key, my_team_id, is_reverse)
        roto_pts = num_teams - my_rank + 1
        total_roto += roto_pts

        print(f"\n{stat_key.upper()} - My Rank: #{my_rank} ({roto_pts} Roto pts)")
        print(f"  Rank  Team                          Value")
        for rank, (tid, val) in enumerate(team_values, 1):
            marker = " <-- YOU" if tid == my_team_id else ""
            tname = team_names.get(tid, f'Team {tid}')[:25]
            if is_percentage:
                print(f"  #{rank:<3}  {tname:<28} {val:>10.1%}{marker}")
            else:
                print(f"  #{rank:<3}  {tname:<28} {val:>10.1f}{marker}")

    print("\n" + "=" * 80)
    print(f"TOTAL ROTO POINTS (Current Roster): {total_roto} / {num_teams * num_categories}")
    print("=" * 80)

    # ==========================================================================
    # SECTION 4: Dropping Jaylen Brown analysis
    # ==========================================================================
    print("\n" + "=" * 80)
    print("4. DROPPING JAYLEN BROWN - DETAILED ANALYSIS")
    print("=" * 80)

    # Find Jaylen Brown and Kyrie Irving
    jb_player = None
    kyrie_player = None
    for p in roster_data:
        if 'jaylen brown' in p.get('name', '').lower():
            jb_player = p
        if 'kyrie' in p.get('name', '').lower():
            kyrie_player = p

    if not jb_player:
        print("ERROR: Jaylen Brown not found in roster!")
        return 1

    print(f"\nJaylen Brown's per-game stats:")
    for cat in scoring_categories:
        stat_key = cat['stat_key']
        if not cat.get('is_percentage'):  # Skip percentage stats for per-game display
            val = jb_player.get('per_game_stats', {}).get(stat_key, 0)
            print(f"  {stat_key.upper()}: {val:.2f}")

    print(f"\nJaylen Brown's projected starts: {player_starts.get(jb_player['player_id'], 0)}")
    print(f"Jaylen Brown's ROS contribution:")
    jb_ros = player_ros_stats.get(jb_player['player_id'], {})
    for cat in scoring_categories:
        stat_key = cat['stat_key']
        if not cat.get('is_percentage'):  # Skip percentage stats
            print(f"  {stat_key.upper()}: {jb_ros.get(stat_key, 0):.1f}")

    # Find Kyrie from IR
    # Get Kyrie's data from the already-built roster_data (includes hybrid projections)
    kyrie_info = None
    kyrie_stats_source = "hybrid projection"
    for p in roster_data:
        if 'kyrie' in p.get('name', '').lower():
            # Get a copy of the player data with active slot
            kyrie_info = dict(p)
            kyrie_info['lineupSlotId'] = 0  # Set to active (not IR)
            break

    if kyrie_info:
        print(f"\nKyrie Irving's per-game stats (source: {kyrie_stats_source}):")
        for cat in scoring_categories:
            stat_key = cat['stat_key']
            if not cat.get('is_percentage'):  # Skip percentage stats
                val = kyrie_info.get('per_game_stats', {}).get(stat_key, 0)
                print(f"  {stat_key.upper()}: {val:.2f}")

    # Build modified roster: drop JB, add Kyrie
    modified_roster = [p for p in roster_data if p['player_id'] != jb_player['player_id']]
    if kyrie_info:
        modified_roster.append(kyrie_info)

    # Simulate modified roster (track all stats including components)
    print("\n--- Simulating roster WITHOUT Jaylen Brown + WITH Kyrie ---")
    ros_totals_new, player_starts_new, player_ros_stats_new, player_info_new = simulate_ros_stats(
        optimizer, modified_roster, start_date, end_date, initial_starts, all_stat_keys
    )

    # New EOS totals (raw stats)
    my_eos_new_raw = {}
    for stat_key in all_stat_keys:
        current = my_current.get(stat_key, 0)
        ros_new = ros_totals_new.get(stat_key, 0)
        my_eos_new_raw[stat_key] = current + ros_new

    # Calculate final category values including percentages
    my_eos_new = {}
    print(f"\n{'Category':<10} {'Current':<12} {'ROS (new)':<12} {'EOS (new)':<12} {'Change':<12}")
    print("-" * 60)
    for cat in scoring_categories:
        stat_key = cat['stat_key']
        if cat.get('is_percentage'):
            current_pct = calculate_percentage(my_current, stat_key)
            eos_new_pct = calculate_percentage(my_eos_new_raw, stat_key)
            my_eos_new[stat_key] = eos_new_pct
            old_pct = my_eos.get(stat_key, 0)
            change = eos_new_pct - old_pct
            print(f"{stat_key.upper():<10} {current_pct:<12.1%} {'N/A':<12} {eos_new_pct:<12.1%} {change:+.1%}")
        else:
            current = my_current.get(stat_key, 0)
            ros_new = ros_totals_new.get(stat_key, 0)
            eos_new = current + ros_new
            my_eos_new[stat_key] = eos_new
            old_eos = my_eos.get(stat_key, 0)
            change = eos_new - old_eos
            print(f"{stat_key.upper():<10} {current:<12.1f} {ros_new:<12.1f} {eos_new:<12.1f} {change:+.1f}")

    # Update all teams EOS with new totals
    all_teams_eos_new = dict(all_teams_eos)
    all_teams_eos_new[my_team_id] = my_eos_new

    # New rankings
    print("\n" + "-" * 80)
    print("CATEGORY RANKINGS COMPARISON (Before vs After Drop)")
    print("-" * 80)

    total_roto_new = 0

    for cat in scoring_categories:
        stat_key = cat['stat_key']
        is_reverse = cat.get('is_reverse', False)
        is_percentage = cat.get('is_percentage', False)

        old_rank, _ = rank_teams_in_category(all_teams_eos, stat_key, my_team_id, is_reverse)
        new_rank, new_values = rank_teams_in_category(all_teams_eos_new, stat_key, my_team_id, is_reverse)

        old_pts = num_teams - old_rank + 1
        new_pts = num_teams - new_rank + 1
        total_roto_new += new_pts

        change = new_pts - old_pts
        change_str = f"({change:+d})" if change != 0 else ""

        print(f"\n{stat_key.upper()}: Rank #{old_rank} -> #{new_rank}  |  Roto: {old_pts} -> {new_pts} {change_str}")

        # Show where we are in rankings
        for rank, (tid, val) in enumerate(new_values, 1):
            if tid == my_team_id:
                if is_percentage:
                    print(f"  Our new value: {val:.1%}")
                else:
                    print(f"  Our new value: {val:.1f}")
                break

    print("\n" + "=" * 80)
    print(f"ROTO POINTS: {total_roto} -> {total_roto_new} (Change: {total_roto_new - total_roto:+d})")
    print("=" * 80)

    # ==========================================================================
    # SECTION 5: Verification checks
    # ==========================================================================
    print("\n" + "=" * 80)
    print("5. VERIFICATION CHECKS")
    print("=" * 80)

    print("\n5a. Are current stats reasonable?")
    for cat in scoring_categories:
        stat_key = cat['stat_key']
        if cat.get('is_percentage'):
            pct = calculate_percentage(my_current, stat_key)
            print(f"  {stat_key.upper()}: {pct:.1%}")
        else:
            val = my_current.get(stat_key, 0)
            per_game = val / games_played if games_played > 0 else 0
            print(f"  {stat_key.upper()}: {val:.1f} total / {games_played} games = {per_game:.1f}/game")

    print("\n5b. Are ROS projections reasonable?")
    print(f"  Simulating {days_remaining} days, ~{games_remaining} team games")
    for cat in scoring_categories:
        stat_key = cat['stat_key']
        if not cat.get('is_percentage'):
            ros = ros_totals.get(stat_key, 0)
            per_game = ros / games_remaining if games_remaining > 0 else 0
            print(f"  {stat_key.upper()}: {ros:.1f} projected / ~{games_remaining} games = {per_game:.1f}/game")

    print("\n5c. Sample of other teams' projected EOS totals (for comparison):")
    # Show first few stats for sample teams
    sample_stats = ['pts', 'reb', 'ast']
    for tid, stats in list(all_teams_eos.items())[:5]:
        tname = team_names.get(tid, f'Team {tid}')
        stat_strs = []
        for skey in sample_stats:
            if skey in stats:
                stat_strs.append(f"{skey.upper()}:{stats[skey]:>8.1f}")
        marker = " <-- YOU" if tid == my_team_id else ""
        print(f"  {tname[:20]:<22} {'  '.join(stat_strs)}{marker}")

    print("\n" + "=" * 80)
    print("DEBUG COMPLETE")
    print("=" * 80)

    return 0


if __name__ == '__main__':
    sys.exit(main())

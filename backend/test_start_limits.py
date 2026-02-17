#!/usr/bin/env python3
"""
Test Start Limit Optimizer with Real League Data.

This script tests the start limit optimizer by:
1. Loading league credentials from the database
2. Connecting to ESPN API
3. Running the optimizer on your team (Doncic Donuts)
4. Showing detailed position limits and player assignments
5. Comparing raw vs adjusted projections

Usage:
    cd /Users/milo/fantasy-basketball-optimizer
    source venv/bin/activate
    python backend/test_start_limits.py
"""

import os
import sys
import sqlite3
from datetime import date

# Add paths for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import requests


def get_league_credentials():
    """Load league credentials from the database."""
    db_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'instance',
        'fantasy_basketball.db'
    )

    if not os.path.exists(db_path):
        print(f"ERROR: Database not found at {db_path}")
        return None

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT espn_league_id, espn_s2_cookie, swid_cookie, league_name, season
        FROM leagues LIMIT 1
    ''')
    row = cursor.fetchone()
    conn.close()

    if not row:
        print("ERROR: No league found in database")
        return None

    return {
        'espn_league_id': int(row[0]),
        'espn_s2': row[1],
        'swid': row[2],
        'league_name': row[3],
        'season': int(row[4]),
    }


def fetch_teams_and_rosters(credentials):
    """Fetch teams and rosters from ESPN API."""
    endpoint = (
        f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/fba/"
        f"seasons/{credentials['season']}/segments/0/leagues/{credentials['espn_league_id']}"
    )

    params = {'view': ['mTeam', 'mRoster', 'mSettings']}
    cookies = {'espn_s2': credentials['espn_s2'], 'SWID': credentials['swid']}

    response = requests.get(endpoint, params=params, cookies=cookies, timeout=15)

    if response.status_code != 200:
        print(f"ERROR: API request failed with status {response.status_code}")
        return None, None

    data = response.json()
    return data.get('teams', []), data.get('settings', {})


def find_my_team(teams, swid):
    """Find the user's team (Doncic Donuts or by SWID)."""
    # First try to find by name
    for team in teams:
        name = team.get('name', '') or team.get('nickname', '')
        if 'doncic' in name.lower():
            return team

    # Fallback: find by SWID
    for team in teams:
        owners = team.get('owners', [])
        for owner in owners:
            owner_id = owner.get('id') if isinstance(owner, dict) else owner
            if owner_id == swid:
                return team

    # Last resort: return first team
    return teams[0] if teams else None


def estimate_games_remaining():
    """Estimate games remaining in NBA season."""
    today = date.today()
    # 2025-26 NBA season runs Oct 2025 to Apr 2026
    season_start = date(2025, 10, 22)
    season_end = date(2026, 4, 13)

    if today < season_start:
        return 82
    if today > season_end:
        return 0

    total_days = (season_end - season_start).days
    elapsed_days = (today - season_start).days
    progress = elapsed_days / total_days

    return max(0, int(82 * (1 - progress)))


def main():
    print("=" * 80)
    print("START LIMIT OPTIMIZER TEST")
    print("=" * 80)

    # Step 1: Load credentials
    print("\n1. Loading league credentials...")
    credentials = get_league_credentials()
    if not credentials:
        return 1

    print(f"   League: {credentials['league_name']}")
    print(f"   ESPN League ID: {credentials['espn_league_id']}")
    print(f"   Season: {credentials['season']}")

    # Step 2: Import the optimizer
    print("\n2. Importing start limit optimizer...")
    try:
        from projections.start_limit_optimizer import (
            StartLimitOptimizer,
            SLOT_ID_TO_POSITION,
            POSITION_TO_SLOT_ID,
        )
        print("   OK - StartLimitOptimizer imported")
    except ImportError as e:
        print(f"   ERROR: Could not import StartLimitOptimizer: {e}")
        return 1

    # Step 3: Create optimizer instance
    print("\n3. Creating optimizer instance...")
    optimizer = StartLimitOptimizer(
        espn_s2=credentials['espn_s2'],
        swid=credentials['swid'],
        league_id=credentials['espn_league_id'],
        season=credentials['season'],
        verbose=False  # We'll do our own logging
    )
    print("   OK - Optimizer created")

    # Step 4: Fetch league data
    print("\n4. Fetching teams and rosters from ESPN...")
    teams, settings = fetch_teams_and_rosters(credentials)
    if not teams:
        print("   ERROR: Could not fetch teams")
        return 1
    print(f"   Fetched {len(teams)} teams")

    # Step 5: Find your team
    print("\n5. Finding your team...")
    my_team = find_my_team(teams, credentials['swid'])
    if not my_team:
        print("   ERROR: Could not find your team")
        return 1

    team_name = my_team.get('name', '') or my_team.get('nickname', 'Unknown')
    team_id = my_team.get('id', 0)
    print(f"   Found: {team_name} (ID: {team_id})")

    # Step 6: Get lineup slot configuration
    print("\n" + "=" * 80)
    print("POSITION START LIMITS")
    print("=" * 80)

    slot_counts = optimizer.get_lineup_slot_counts()
    stat_limits = optimizer.get_lineup_slot_stat_limits()

    print(f"\n{'Position':<12} {'Slots':<8} {'Games Limit':<14} {'Per Slot':<10}")
    print("-" * 50)

    active_positions = {}
    for slot_id, count in sorted(slot_counts.items()):
        if count == 0:
            continue
        pos_name = SLOT_ID_TO_POSITION.get(slot_id, f'SLOT_{slot_id}')
        limit = stat_limits.get(slot_id, count * 82)
        per_slot = limit // count if count > 0 else 0

        if slot_id not in [12, 13, 14]:  # Skip BE, IR, IR+
            active_positions[slot_id] = {
                'name': pos_name,
                'count': count,
                'limit': limit,
            }
            print(f"{pos_name:<12} {count:<8} {limit:<14} {per_slot:<10}")
        else:
            print(f"{pos_name:<12} {count:<8} {'unlimited':<14} {'-':<10}")

    # Step 7: Get starts used (estimated)
    print("\n" + "=" * 80)
    print("STARTS USED VS REMAINING (ESTIMATED)")
    print("=" * 80)

    team_limits = optimizer.build_team_start_limits(team_id, team_name)

    print(f"\n{'Position':<12} {'Used':<10} {'Remaining':<12} {'Limit':<10} {'%Used':<10}")
    print("-" * 60)

    for slot_id in sorted(active_positions.keys()):
        slot = team_limits.get_slot(slot_id)
        if slot:
            pct_used = (slot.games_used / slot.games_limit * 100) if slot.games_limit > 0 else 0
            print(f"{slot.position_name:<12} {slot.games_used:<10} {slot.games_remaining:<12} "
                  f"{slot.games_limit:<10} {pct_used:.1f}%")

    # Step 8: Show roster with eligible positions
    print("\n" + "=" * 80)
    print(f"ROSTER: {team_name}")
    print("=" * 80)

    roster_entries = my_team.get('roster', {}).get('entries', [])
    print(f"\nRoster size: {len(roster_entries)} players")

    print(f"\n{'Player':<25} {'Current Slot':<12} {'Eligible Positions':<40}")
    print("-" * 80)

    roster_for_optimizer = []
    games_remaining = estimate_games_remaining()
    print(f"\n(Estimated games remaining in season: {games_remaining})")
    print()

    for entry in roster_entries:
        player = entry.get('playerPoolEntry', {}).get('player', {})
        player_name = player.get('fullName', 'Unknown')[:24]
        player_id = player.get('id', 0)

        lineup_slot_id = entry.get('lineupSlotId', -1)
        current_slot = SLOT_ID_TO_POSITION.get(lineup_slot_id, f'?{lineup_slot_id}')

        eligible_slots = player.get('eligibleSlots', [])
        eligible_active = [s for s in eligible_slots if s not in [12, 13, 14]]
        eligible_names = [SLOT_ID_TO_POSITION.get(s, f'?{s}') for s in eligible_active]

        print(f"{player_name:<25} {current_slot:<12} {', '.join(eligible_names):<40}")

        # Get player stats for projection estimate
        stats = player.get('stats', [])
        per_game_stats = {}
        games_played = 0

        for stat_set in stats:
            if stat_set.get('id') == '002026' or stat_set.get('statSourceId') == 0:
                if 'averageStats' in stat_set:
                    per_game_stats = stat_set['averageStats']
                if 'appliedTotal' in stat_set:
                    games_played = int(stat_set.get('appliedTotal', 0))
                break

        # Estimate projected games for this player
        if games_played > 0:
            # Use their game rate to estimate remaining games
            team_games_played = 82 - games_remaining
            if team_games_played > 0:
                game_rate = min(1.0, games_played / team_games_played)
            else:
                game_rate = 0.85
        else:
            game_rate = 0.75

        projected_games = int(games_remaining * game_rate)

        # Simple projection: per-game * projected games
        projected_stats = {}
        stat_id_map = {
            0: 'PTS', 6: 'REB', 3: 'AST', 2: 'STL', 1: 'BLK',
            17: '3PM', 13: 'FGM', 14: 'FGA', 15: 'FTM', 16: 'FTA',
        }

        for stat_id, stat_name in stat_id_map.items():
            per_game_val = per_game_stats.get(str(stat_id), 0) or 0
            projected_stats[stat_name] = per_game_val * projected_games

        roster_for_optimizer.append({
            'player_id': player_id,
            'name': player_name,
            'eligible_slots': eligible_active,
            'projected_games': projected_games,
            'projected_stats': projected_stats,
            'per_game': per_game_stats,
            'games_played': games_played,
        })

    # Step 9: Run the optimizer
    print("\n" + "=" * 80)
    print("OPTIMIZATION RESULTS")
    print("=" * 80)

    categories = [
        {'stat_key': 'PTS', 'abbr': 'PTS'},
        {'stat_key': 'REB', 'abbr': 'REB'},
        {'stat_key': 'AST', 'abbr': 'AST'},
        {'stat_key': 'STL', 'abbr': 'STL'},
        {'stat_key': 'BLK', 'abbr': 'BLK'},
        {'stat_key': '3PM', 'abbr': '3PM'},
        {'stat_key': 'FG%', 'abbr': 'FG%'},
        {'stat_key': 'FT%', 'abbr': 'FT%'},
    ]

    adjusted_totals, assignments = optimizer.optimize_team_projections(
        team_id=team_id,
        team_name=team_name,
        roster=roster_for_optimizer,
        categories=categories
    )

    # Show assignments
    print(f"\n{'Player':<25} {'Assigned To':<12} {'Proj Games':<12} {'Actual':<10} {'Status':<10}")
    print("-" * 75)

    starting_count = 0
    benched_count = 0

    for assignment in assignments:
        status = "BENCHED" if assignment.is_benched else "Starting"
        if assignment.is_benched:
            benched_count += 1
        else:
            starting_count += 1

        print(f"{assignment.player_name[:24]:<25} {assignment.assigned_position:<12} "
              f"{assignment.projected_games:<12} {assignment.actual_games_to_start:<10} {status:<10}")

    print(f"\nSummary: {starting_count} starting, {benched_count} benched")

    # Step 10: Compare raw vs adjusted projections
    print("\n" + "=" * 80)
    print("PROJECTIONS: RAW vs ADJUSTED")
    print("=" * 80)

    # Calculate raw totals (no start limits)
    raw_totals = {cat['stat_key']: 0.0 for cat in categories}
    raw_totals['FGM'] = 0.0
    raw_totals['FGA'] = 0.0
    raw_totals['FTM'] = 0.0
    raw_totals['FTA'] = 0.0

    for player in roster_for_optimizer:
        stats = player['projected_stats']
        for key in ['PTS', 'REB', 'AST', 'STL', 'BLK', '3PM', 'FGM', 'FGA', 'FTM', 'FTA']:
            raw_totals[key] = raw_totals.get(key, 0) + stats.get(key, 0)

    # Calculate raw percentages
    if raw_totals['FGA'] > 0:
        raw_totals['FG%'] = raw_totals['FGM'] / raw_totals['FGA']
    else:
        raw_totals['FG%'] = 0.0

    if raw_totals['FTA'] > 0:
        raw_totals['FT%'] = raw_totals['FTM'] / raw_totals['FTA']
    else:
        raw_totals['FT%'] = 0.0

    print(f"\n{'Category':<12} {'Raw Projection':<18} {'Adjusted':<18} {'Reduction':<12}")
    print("-" * 65)

    for cat in categories:
        key = cat['stat_key']
        raw_val = raw_totals.get(key, 0)
        adj_val = adjusted_totals.get(key, 0)

        if key in ['FG%', 'FT%']:
            raw_str = f"{raw_val:.3f}"
            adj_str = f"{adj_val:.3f}"
            reduction = ""
        else:
            raw_str = f"{raw_val:.0f}"
            adj_str = f"{adj_val:.0f}"
            if raw_val > 0:
                reduction_pct = (1 - adj_val / raw_val) * 100
                reduction = f"-{reduction_pct:.1f}%"
            else:
                reduction = "0%"

        print(f"{key:<12} {raw_str:<18} {adj_str:<18} {reduction:<12}")

    # Show impact summary
    print("\n" + "=" * 80)
    print("IMPACT SUMMARY")
    print("=" * 80)

    raw_pts = raw_totals.get('PTS', 0)
    adj_pts = adjusted_totals.get('PTS', 0)
    pts_lost = raw_pts - adj_pts

    print(f"\nTotal ROS Points Lost Due to Start Limits: {pts_lost:.0f}")
    if raw_pts > 0:
        print(f"Percentage Reduction: {(pts_lost / raw_pts) * 100:.1f}%")

    print(f"\nBenched Players (will contribute 0 stats to ROS projections):")
    for assignment in assignments:
        if assignment.is_benched:
            # Find original projection
            orig = next((p for p in roster_for_optimizer if p['name'].startswith(assignment.player_name[:20])), None)
            if orig:
                pts_lost = orig['projected_stats'].get('PTS', 0)
                print(f"  - {assignment.player_name}: {pts_lost:.0f} PTS lost")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

    return 0


if __name__ == '__main__':
    sys.exit(main())

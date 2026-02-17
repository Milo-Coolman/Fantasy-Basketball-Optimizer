#!/usr/bin/env python3
"""
Debug script for dashboard projections.

This script simulates what happens when the dashboard API is called and
compares the actual API output with what the start_limit_optimizer produces.

It helps identify where projection calculations might be breaking in the
dashboard flow.
"""

import sys
import os
import json
import sqlite3
from datetime import datetime

# Add project root to path BEFORE any backend imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from espn_api.basketball import League as ESPNLeague
from backend.projections.start_limit_optimizer import StartLimitOptimizer

def normalize_swid(swid: str) -> str:
    """Normalize SWID for comparison - strips braces and lowercases."""
    if not swid:
        return ''
    return swid.strip('{}').lower()


def get_database_path():
    """Get the path to the SQLite database."""
    db_path = os.path.join(PROJECT_ROOT, 'instance', 'fantasy_basketball.db')
    return db_path


def get_espn_credentials():
    """Get ESPN credentials from database using direct SQLite query."""
    db_path = get_database_path()

    print(f"\nDatabase path: {db_path}")
    print(f"Database exists: {os.path.exists(db_path)}")

    if not os.path.exists(db_path):
        print(f"ERROR: Database not found at {db_path}")
        return None

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # First, let's see what leagues exist
        print("\nQuerying all leagues in database...")
        cursor.execute('''
            SELECT id, espn_league_id, league_name, season, espn_s2_cookie IS NOT NULL as has_s2, swid_cookie IS NOT NULL as has_swid
            FROM leagues
        ''')
        all_leagues = cursor.fetchall()

        if not all_leagues:
            print("ERROR: No leagues found in database!")
            print("Make sure you've set up a league through the frontend first.")
            conn.close()
            return None

        print(f"\nFound {len(all_leagues)} league(s) in database:")
        for league in all_leagues:
            print(f"  ID: {league[0]}, ESPN ID: {league[1]}, Name: {league[2]}, Season: {league[3]}, Has S2: {league[4]}, Has SWID: {league[5]}")

        # Get the first league with credentials
        cursor.execute('''
            SELECT id, espn_league_id, espn_s2_cookie, swid_cookie, league_name, season
            FROM leagues
            WHERE espn_s2_cookie IS NOT NULL AND swid_cookie IS NOT NULL
            LIMIT 1
        ''')
        row = cursor.fetchone()
        conn.close()

        if not row:
            print("\nERROR: No league with valid credentials found!")
            print("Leagues exist but none have ESPN cookies set.")
            return None

        credentials = {
            'db_id': row[0],
            'espn_league_id': int(row[1]),
            'espn_s2': row[2],
            'swid': row[3],
            'league_name': row[4],
            'season': int(row[5]),
        }

        print(f"\nUsing league: {credentials['league_name']}")
        print(f"  Database ID: {credentials['db_id']}")
        print(f"  ESPN League ID: {credentials['espn_league_id']}")
        print(f"  Season: {credentials['season']}")
        print(f"  ESPN_S2: {'*' * 10}...{credentials['espn_s2'][-10:] if credentials['espn_s2'] else 'MISSING'}")
        print(f"  SWID: {credentials['swid'][:20] if credentials['swid'] else 'MISSING'}...")

        return credentials

    except sqlite3.Error as e:
        print(f"ERROR: Database query failed: {e}")
        return None


def connect_to_espn(credentials):
    """Connect to ESPN API using credentials from database."""
    if not credentials:
        print("ERROR: No credentials provided")
        sys.exit(1)

    espn_s2 = credentials['espn_s2']
    swid = credentials['swid']
    league_id = credentials['espn_league_id']
    season = credentials['season']

    if not espn_s2 or not swid:
        print("ERROR: Missing ESPN credentials in database")
        sys.exit(1)

    print(f"\nConnecting to ESPN League {league_id} (Season {season})...")
    league = ESPNLeague(league_id=league_id, year=season, espn_s2=espn_s2, swid=swid)
    print(f"Connected! League: {league.settings.name}")
    return league


def find_user_team(league, credentials):
    """Find the user's team by matching SWID to team owner IDs."""
    user_swid = credentials.get('swid', '')
    user_swid_normalized = normalize_swid(user_swid)

    print(f"\n{'='*60}")
    print("FINDING USER'S TEAM (SWID Matching)")
    print('='*60)
    print(f"\nYour SWID (normalized): {user_swid_normalized[:20]}...")

    match_found = None

    for team in league.teams:
        owners = getattr(team, 'owners', None)

        if owners:
            for owner in owners:
                # Handle dict vs object
                if isinstance(owner, dict):
                    owner_id = owner.get('id', '')
                else:
                    owner_id = getattr(owner, 'id', '')

                if owner_id:
                    owner_id_normalized = normalize_swid(str(owner_id))

                    if owner_id_normalized == user_swid_normalized:
                        match_found = team
                        print(f"\nMATCH FOUND!")
                        print(f"  Team: {team.team_name}")
                        print(f"  Team ID: {team.team_id}")
                        print(f"  Owner ID matches SWID")
                        break

            if match_found:
                break

    if match_found:
        return match_found

    # No match found - show all teams for debugging
    print("\nERROR: Could not find team matching your SWID")
    print("\nAll teams and their owner IDs:")
    for team in league.teams:
        owners = getattr(team, 'owners', None)
        owner_ids = []
        if owners:
            for owner in owners:
                if isinstance(owner, dict):
                    oid = owner.get('id', '')
                else:
                    oid = getattr(owner, 'id', '')
                if oid:
                    owner_ids.append(normalize_swid(str(oid))[:20])
        print(f"  - {team.team_name}: {owner_ids if owner_ids else 'No owners found'}")

    print(f"\nYour SWID (normalized): {user_swid_normalized}")
    print("\nPossible issues:")
    print("  1. SWID cookie might be incorrect or expired")
    print("  2. You might not be an owner in this league")
    sys.exit(1)


def get_current_stats_from_espn(team):
    """Get current accumulated stats from ESPN for a team."""
    print(f"\n{'='*60}")
    print(f"CURRENT STATS FROM ESPN: {team.team_name}")
    print('='*60)

    # ESPN stores current standings data
    stats = {}

    # Try to get from team standings
    if hasattr(team, 'stats'):
        stats = team.stats

    print(f"\nTeam Stats (from ESPN):")
    if stats:
        for key, value in sorted(stats.items()):
            print(f"  {key}: {value}")
    else:
        print("  No stats available directly on team object")

    # Also show current standing info
    print(f"\nCurrent Standing: {team.standing}")
    if hasattr(team, 'wins'):
        print(f"Category Record: {team.wins}-{team.losses}")
    if hasattr(team, 'points'):
        print(f"Total Points: {team.points}")

    return stats


def get_roster_info(team, league):
    """Get detailed roster information."""
    print(f"\n{'='*60}")
    print(f"ROSTER INFO: {team.team_name}")
    print('='*60)

    roster_data = []
    ir_players = []

    for player in team.roster:
        player_info = {
            'name': player.name,
            'player_id': player.playerId,
            'position': player.position,
            'lineup_slot': player.lineupSlot,
            'injury_status': getattr(player, 'injuryStatus', 'ACTIVE'),
            'pro_team': getattr(player, 'proTeam', 'UNK'),
        }

        # Check if on IR
        if player.lineupSlot == 'IR':
            ir_players.append(player_info)
            print(f"  [IR] {player.name} ({player.position}) - {player_info['injury_status']}")
        else:
            print(f"  {player.name} ({player.position}) - Slot: {player.lineupSlot}")

        # Get player stats
        if hasattr(player, 'stats'):
            player_info['stats'] = player.stats

        roster_data.append(player_info)

    print(f"\nTotal roster size: {len(roster_data)}")
    print(f"Players on IR: {len(ir_players)}")

    return roster_data, ir_players


def run_start_limit_optimizer(league, team, credentials):
    """Run the start limit optimizer and show detailed output."""
    print(f"\n{'='*60}")
    print("START LIMIT OPTIMIZER OUTPUT")
    print('='*60)

    # Initialize optimizer with correct parameters (like test_start_limits.py)
    print("\nInitializing StartLimitOptimizer...")
    optimizer = StartLimitOptimizer(
        espn_s2=credentials['espn_s2'],
        swid=credentials['swid'],
        league_id=credentials['espn_league_id'],
        season=credentials['season'],
        verbose=True
    )
    print("  Optimizer created successfully")

    # Fetch scoring categories from the optimizer
    print("\nFetching scoring categories...")
    scoring_categories = optimizer.fetch_scoring_categories()
    print(f"  Found {len(scoring_categories)} categories:")
    for cat in scoring_categories[:5]:
        print(f"    - {cat.get('stat_key', 'unknown')}: {cat.get('display_name', '')}")
    if len(scoring_categories) > 5:
        print(f"    ... and {len(scoring_categories) - 5} more")

    # Get stat keys from scoring categories
    stat_keys = [c.get('stat_key', '') for c in scoring_categories]
    stat_keys = [k for k in stat_keys if k]  # Remove empty
    print(f"\nStat keys for projections: {stat_keys}")

    # Build roster data with hybrid projections
    print("\nBuilding roster with hybrid projections...")
    espn_players = []
    for player in team.roster:
        player_data = {
            'name': player.name,
            'playerId': player.playerId,
            'position': player.position,
            'lineupSlotId': getattr(player, 'lineupSlotId', 0),
            'injuryStatus': getattr(player, 'injuryStatus', 'ACTIVE'),
            'proTeam': getattr(player, 'proTeam', 'UNK'),
        }

        # Add stats if available
        if hasattr(player, 'stats'):
            player_data['stats'] = player.stats

        espn_players.append(player_data)

    print(f"  Built data for {len(espn_players)} players")

    # Build roster with projections
    roster_data = optimizer.build_roster_with_hybrid_projections(
        espn_players=espn_players,
        stat_keys=stat_keys
    )

    print(f"\nRoster data built for {len(roster_data)} players")

    # Show projection details for each player
    print("\n--- Player Projections ---")
    for player in roster_data[:5]:  # Show first 5
        print(f"\n{player.get('name', 'Unknown')}:")
        print(f"  Position: {player.get('position')}")
        print(f"  Games remaining: {player.get('games_remaining', 'N/A')}")
        print(f"  Projected stats (per game):")
        for stat in stat_keys[:5]:  # Show first 5 stats
            val = player.get('projected_stats', {}).get(stat, 'N/A')
            print(f"    {stat}: {val}")

    # Get starts used for this team
    print(f"\n\nFetching starts used for team {team.team_id}...")
    try:
        starts_used = optimizer.fetch_starts_used(team.team_id)
        print(f"  Starts used by slot: {starts_used}")
    except Exception as e:
        print(f"  Error fetching starts used: {e}")
        starts_used = {}

    # Get lineup slot counts
    print("\nFetching lineup slot configuration...")
    try:
        slot_counts = optimizer.get_lineup_slot_counts()
        slot_limits = optimizer.get_lineup_slot_stat_limits()
        print(f"  Slot counts: {slot_counts}")
        print(f"  Slot limits: {slot_limits}")
    except Exception as e:
        print(f"  Error fetching slot config: {e}")

    # Build result summary
    result = {
        'roster_data': roster_data,
        'scoring_categories': scoring_categories,
        'stat_keys': stat_keys,
        'starts_used': starts_used,
    }

    print(f"\nOptimizer data collection complete!")
    print(f"Result keys: {list(result.keys())}")

    # Show start limits from starts_used
    if starts_used:
        print("\n--- Starts Used by Position Slot ---")
        from backend.projections.start_limit_optimizer import SLOT_ID_TO_POSITION
        for slot_id, used in starts_used.items():
            pos_name = SLOT_ID_TO_POSITION.get(slot_id, f'Slot {slot_id}')
            limit = slot_limits.get(slot_id, 82) if slot_limits else 82
            remaining = limit - used
            print(f"  {pos_name}: {used}/{limit} used ({remaining} remaining)")

    # Show IR players on roster
    ir_on_roster = [p for p in roster_data if p.get('lineupSlotId') == 13]
    if ir_on_roster:
        print("\n--- IR Players on Roster ---")
        for ir_player in ir_on_roster:
            print(f"  {ir_player.get('name')} - {ir_player.get('injuryStatus', 'Unknown status')}")

    return result, optimizer


def call_dashboard_api(credentials):
    """Call the actual dashboard API and return the response."""
    print(f"\n{'='*60}")
    print("DASHBOARD API RESPONSE")
    print('='*60)

    # Import here to avoid circular imports
    from backend.app import create_app

    db_league_id = credentials['db_id']

    app = create_app()
    with app.app_context():
        with app.test_client() as client:
            print(f"Calling dashboard API for league ID: {db_league_id}")

            try:
                # Call the dashboard API endpoint
                response = client.get(f'/api/dashboard/{db_league_id}')

                if response.status_code == 200:
                    data = response.get_json()
                    print(f"API returned successfully!")
                    print(f"Response keys: {list(data.keys()) if data else 'None'}")
                    return data
                elif response.status_code == 401:
                    print("API requires authentication - cannot test without login")
                    print("To test the full API flow, log in via the frontend.")
                    return None
                else:
                    print(f"API returned status {response.status_code}")
                    try:
                        print(f"Response: {response.get_json()}")
                    except:
                        print(f"Response text: {response.data[:500]}")
                    return None

            except Exception as e:
                print(f"Error calling API: {e}")
                import traceback
                traceback.print_exc()
                return None


def simulate_dashboard_calculation(league, user_team, optimizer_result):
    """Simulate what the dashboard should calculate."""
    print(f"\n{'='*60}")
    print("SIMULATED DASHBOARD CALCULATION")
    print('='*60)

    # Get current standings
    print("\nCurrent Standings:")
    for i, team in enumerate(sorted(league.teams, key=lambda t: t.standing), 1):
        marker = " <-- YOU" if team.team_id == user_team.team_id else ""
        print(f"  {i}. {team.team_name} (Standing: {team.standing}){marker}")

    # The dashboard should:
    # 1. Get current accumulated stats for all teams
    # 2. Add projected ROS stats (from optimizer)
    # 3. Re-rank teams based on end-of-season totals

    print("\n--- What Dashboard Should Do ---")
    print("1. Get current stats from ESPN for all teams")
    print("2. Run start_limit_optimizer for each team to get ROS projections")
    print("3. Add current + ROS to get end-of-season projections")
    print("4. Rank teams in each category based on projected totals")
    print("5. Calculate Roto points (rank points) for each team")
    print("6. Sort by total Roto points for projected standings")

    # Show what we calculated for user's team
    if optimizer_result and 'projected_totals' in optimizer_result:
        print(f"\n--- Your Team's Projected ROS Stats ---")
        for stat, value in optimizer_result['projected_totals'].items():
            if isinstance(value, (int, float)):
                print(f"  {stat}: {value:.1f}")

    return True


def compare_api_vs_expected(api_response, optimizer_result, user_team):
    """Compare API response with expected values."""
    print(f"\n{'='*60}")
    print("COMPARISON: API vs EXPECTED")
    print('='*60)

    if not api_response:
        print("\nCannot compare - API response not available (requires authentication)")
        print("To test the API, log in via the frontend and check network requests.")
        return

    # Check if API response has projected standings
    projected = api_response.get('projected_standings', [])
    if not projected:
        print("\nWARNING: API response has no projected_standings!")
        print("This suggests the dashboard is not calling the optimizer.")
        return

    # Find user's team in projected standings
    user_projected = None
    for team in projected:
        if team.get('team_name', '').lower() == user_team.team_name.lower():
            user_projected = team
            break

    if user_projected:
        print(f"\nYour team in API response:")
        print(f"  Projected Rank: {user_projected.get('projected_rank')}")
        print(f"  Roto Points: {user_projected.get('roto_points')}")

        if 'categories' in user_projected:
            print(f"  Category Points: {user_projected['categories']}")

    # Check if start_limits is in response
    start_limits = api_response.get('start_limits', {})
    if start_limits.get('enabled'):
        print("\n Start limits ARE enabled in API response")
        print(f"  Position limits: {list(start_limits.get('position_limits', {}).keys())}")
    else:
        print("\nWARNING: Start limits NOT enabled in API response!")
        print("The dashboard may not be using the optimizer.")


def check_dashboard_code():
    """Check if the dashboard API is calling the optimizer."""
    print(f"\n{'='*60}")
    print("CODE ANALYSIS: Dashboard API")
    print('='*60)

    dashboard_path = os.path.join(
        os.path.dirname(__file__),
        'api',
        'dashboard.py'
    )

    if not os.path.exists(dashboard_path):
        print(f"Dashboard file not found at: {dashboard_path}")
        return

    with open(dashboard_path, 'r') as f:
        content = f.read()

    # Check for optimizer imports/usage
    checks = [
        ('StartLimitOptimizer import', 'StartLimitOptimizer' in content),
        ('Optimizer instantiation', 'StartLimitOptimizer(' in content),
        ('optimize_ros_with_ir call', 'optimize_ros_with_ir' in content),
        ('build_roster_with_hybrid_projections', 'build_roster_with_hybrid_projections' in content),
        ('start_limits in response', "'start_limits'" in content or '"start_limits"' in content),
        ('projected_standings calculation', 'projected_standings' in content),
    ]

    print("\nDashboard API Code Checks:")
    all_passed = True
    for check_name, passed in checks:
        status = "✓" if passed else "✗"
        print(f"  {status} {check_name}")
        if not passed:
            all_passed = False

    if not all_passed:
        print("\nSome checks failed - the dashboard may not be fully integrated with the optimizer.")
        print("Review backend/api/dashboard.py to ensure it:")
        print("  1. Imports and instantiates StartLimitOptimizer")
        print("  2. Calls build_roster_with_hybrid_projections for each team")
        print("  3. Calls optimize_ros_with_ir to get projections")
        print("  4. Includes start_limits in the response")


def main():
    """Main debug function."""
    print("="*60)
    print("DASHBOARD PROJECTIONS DEBUG SCRIPT")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # Step 1: Get credentials from database
    credentials = get_espn_credentials()
    if not credentials:
        print("\nFailed to get credentials. Exiting.")
        sys.exit(1)

    # Step 2: Connect to ESPN
    league = connect_to_espn(credentials)

    # Step 3: Find user's team using SWID matching
    user_team = find_user_team(league, credentials)
    print(f"\nYour team: {user_team.team_name} (ID: {user_team.team_id})")

    # Step 4a: Show current stats from ESPN
    current_stats = get_current_stats_from_espn(user_team)

    # Step 4b: Show roster info
    roster_data, ir_players = get_roster_info(user_team, league)

    # Step 4c: Run start limit optimizer
    optimizer_result, optimizer = run_start_limit_optimizer(league, user_team, credentials)

    # Step 5: Call dashboard API
    api_response = call_dashboard_api(credentials)

    # Step 6: Simulate what dashboard should calculate
    simulate_dashboard_calculation(league, user_team, optimizer_result)

    # Step 7: Compare API vs expected
    compare_api_vs_expected(api_response, optimizer_result, user_team)

    # Step 8: Check dashboard code
    check_dashboard_code()

    print(f"\n{'='*60}")
    print("DEBUG COMPLETE")
    print("="*60)
    print("\nSummary:")
    print("- If start_limits is not in API response, dashboard isn't using optimizer")
    print("- If projected_standings matches current_standings, no projection happening")
    print("- Check the code analysis section for missing integrations")
    print("\nNext steps:")
    print("1. Log into the frontend and navigate to your league dashboard")
    print("2. Open browser DevTools > Network tab")
    print("3. Look for the /api/dashboard/{id} request")
    print("4. Check if response contains 'start_limits' with 'enabled: true'")


if __name__ == '__main__':
    main()

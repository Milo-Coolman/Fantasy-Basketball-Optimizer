#!/usr/bin/env python3
"""
Debug script to investigate why healthy players might be getting false injuries.

Compares healthy players (Luka, Scottie, Brunson) vs injured players (Trey Murphy, JJJ)
to identify where false injury data is coming from.

Usage:
    cd /Users/milo/fantasy-basketball-optimizer
    source venv/bin/activate
    python backend/debug_healthy_players_injury.py
"""

import json
import sys
import requests
from datetime import date
from typing import Dict, Any, Optional, List

# Add project root to path
sys.path.insert(0, '/Users/milo/fantasy-basketball-optimizer')

from backend.app import create_app
from backend.models import League
from backend.services.espn_client import ESPNClient


# Players to investigate
TARGET_PLAYERS = {
    # Healthy players (should have NO injury data)
    'healthy': ['Luka Doncic', 'Scottie Barnes', 'Jalen Brunson'],
    # Injured players (should have injury data)
    'injured': ['Trey Murphy', 'Jaren Jackson'],
}


def print_section(title: str, char: str = "="):
    """Print a section header."""
    print(f"\n{char * 80}")
    print(title)
    print(f"{char * 80}")


def safe_json(obj: Any) -> str:
    """Safely convert object to JSON string."""
    def default_handler(o):
        if isinstance(o, date):
            return o.isoformat()
        return str(o)
    return json.dumps(obj, indent=2, default=default_handler)


def find_player_in_raw_api(raw_data: Dict, player_names: List[str]) -> Dict[str, Dict]:
    """Find players in raw API response."""
    found = {}

    for team_data in raw_data.get('teams', []):
        team_name = team_data.get('name', 'Unknown Team')
        for entry in team_data.get('roster', {}).get('entries', []):
            player_pool = entry.get('playerPoolEntry', {})
            player = player_pool.get('player', {})
            full_name = player.get('fullName', '')

            for target_name in player_names:
                if target_name.lower() in full_name.lower():
                    found[full_name] = {
                        'player': player,
                        'entry': entry,
                        'playerPoolEntry': player_pool,
                        'team_name': team_name,
                    }

    return found


def analyze_player(name: str, data: Dict, expected_injured: bool):
    """Analyze a single player's injury data."""
    player = data['player']
    entry = data['entry']
    player_pool = data['playerPoolEntry']
    team_name = data['team_name']

    print(f"\n{'─' * 60}")
    print(f"PLAYER: {name}")
    print(f"TEAM: {team_name}")
    print(f"EXPECTED STATUS: {'INJURED' if expected_injured else 'HEALTHY'}")
    print(f"{'─' * 60}")

    # Raw ESPN fields
    print("\n[RAW ESPN API DATA]")

    # injuryStatus
    injury_status = player.get('injuryStatus')
    print(f"  player.injuryStatus = {injury_status}")

    # injured boolean
    injured = player.get('injured')
    print(f"  player.injured = {injured}")

    # injuryDetails (the key field we're investigating)
    injury_details_player = player.get('injuryDetails')
    injury_details_entry = entry.get('injuryDetails')
    injury_details_pool = player_pool.get('injuryDetails')

    print(f"  player.injuryDetails = {injury_details_player}")
    print(f"  entry.injuryDetails = {injury_details_entry}")
    print(f"  playerPoolEntry.injuryDetails = {injury_details_pool}")

    # Other injury-related fields
    print("\n[OTHER INJURY FIELDS]")
    injury_fields = ['injuryDate', 'injuryNotes', 'seasonOutlook', 'status',
                     'healthStatus', 'returnDate', 'expectedReturnDate']
    for field in injury_fields:
        val = player.get(field)
        if val is not None:
            print(f"  player.{field} = {val}")

    # Check if any injuryDetails exists anywhere
    has_injury_details = any([injury_details_player, injury_details_entry, injury_details_pool])

    print("\n[ANALYSIS]")
    if expected_injured:
        if has_injury_details:
            print(f"  ✓ CORRECT: Injured player HAS injuryDetails")
            # Show the details
            details = injury_details_player or injury_details_entry or injury_details_pool
            if details:
                print(f"    injuryDetails content:")
                for k, v in details.items():
                    print(f"      {k}: {v}")
        else:
            print(f"  ✗ ISSUE: Injured player has NO injuryDetails")
            print(f"    ESPN may not provide injuryDetails for this injury type")
    else:
        if has_injury_details:
            print(f"  ✗ ISSUE: Healthy player HAS injuryDetails (FALSE POSITIVE)")
            details = injury_details_player or injury_details_entry or injury_details_pool
            if details:
                print(f"    FALSE injuryDetails content:")
                for k, v in details.items():
                    print(f"      {k}: {v}")
        else:
            print(f"  ✓ CORRECT: Healthy player has NO injuryDetails")

    if injury_status and injury_status != 'ACTIVE':
        if expected_injured:
            print(f"  ✓ injuryStatus='{injury_status}' matches expected INJURED")
        else:
            print(f"  ✗ ISSUE: Healthy player has injuryStatus='{injury_status}'")
    else:
        if expected_injured:
            print(f"  ? injuryStatus is ACTIVE/None but player expected to be injured")
        else:
            print(f"  ✓ injuryStatus is ACTIVE/None as expected for healthy player")

    return {
        'name': name,
        'expected_injured': expected_injured,
        'has_injury_details': has_injury_details,
        'injury_status': injury_status,
        'injured_bool': injured,
    }


def main():
    print("=" * 80)
    print("HEALTHY vs INJURED PLAYERS - INJURY DATA INVESTIGATION")
    print("=" * 80)

    # Initialize Flask app and get credentials
    app = create_app()
    with app.app_context():
        league = League.query.first()
        if not league:
            print("ERROR: No league found in database!")
            sys.exit(1)

        print(f"\nLeague: {league.league_name}")
        print(f"League ID: {league.espn_league_id}")
        print(f"Season: {league.season}")

        # =====================================================================
        # STEP 1: Raw ESPN API Call
        # =====================================================================
        print_section("STEP 1: RAW ESPN API RESPONSE")

        print("\nFetching raw API data with kona_playercard view...")

        endpoint = (
            f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/fba/"
            f"seasons/{league.season}/segments/0/leagues/{league.espn_league_id}"
        )
        params = {'view': ['mTeam', 'mRoster', 'kona_player_info', 'kona_playercard']}
        cookies = {'espn_s2': league.espn_s2_cookie, 'SWID': league.swid_cookie}

        response = requests.get(endpoint, params=params, cookies=cookies, timeout=30)
        response.raise_for_status()
        raw_api_data = response.json()

        # Find all target players
        all_targets = TARGET_PLAYERS['healthy'] + TARGET_PLAYERS['injured']
        found_players = find_player_in_raw_api(raw_api_data, all_targets)

        print(f"\nFound {len(found_players)} of {len(all_targets)} target players:")
        for name in found_players:
            print(f"  - {name}")

        # =====================================================================
        # STEP 2: Analyze Each Player
        # =====================================================================
        print_section("STEP 2: INDIVIDUAL PLAYER ANALYSIS")

        results = []

        # Analyze healthy players first
        print("\n" + "=" * 60)
        print("HEALTHY PLAYERS (should have NO injury data)")
        print("=" * 60)

        for target in TARGET_PLAYERS['healthy']:
            for found_name, data in found_players.items():
                if target.lower() in found_name.lower():
                    result = analyze_player(found_name, data, expected_injured=False)
                    results.append(result)
                    break
            else:
                print(f"\n⚠ {target}: NOT FOUND in any roster")

        # Analyze injured players
        print("\n" + "=" * 60)
        print("INJURED PLAYERS (should have injury data)")
        print("=" * 60)

        for target in TARGET_PLAYERS['injured']:
            for found_name, data in found_players.items():
                if target.lower() in found_name.lower():
                    result = analyze_player(found_name, data, expected_injured=True)
                    results.append(result)
                    break
            else:
                print(f"\n⚠ {target}: NOT FOUND in any roster")

        # =====================================================================
        # STEP 3: Check ESPNClient Processing
        # =====================================================================
        print_section("STEP 3: ESPNCLIENT PROCESSING")

        print("\nChecking what ESPNClient._get_raw_injury_details returns...")

        client = ESPNClient(
            league_id=league.espn_league_id,
            year=league.season,
            espn_s2=league.espn_s2_cookie,
            swid=league.swid_cookie
        )

        # Get all rosters to trigger raw data fetch
        all_rosters = client.get_all_rosters()

        # Find our target players in the processed rosters
        for target in all_targets:
            found = False
            for team_id, team_data in all_rosters.items():
                for player in team_data.get('roster', []):
                    name = player.get('name', '')
                    if target.lower() in name.lower():
                        found = True
                        expected_injured = target in TARGET_PLAYERS['injured']

                        print(f"\n--- {name} (via get_all_rosters) ---")
                        print(f"  Expected: {'INJURED' if expected_injured else 'HEALTHY'}")
                        print(f"  injury_status: {player.get('injury_status')}")
                        print(f"  injury_details: {player.get('injury_details')}")
                        print(f"  expected_return_date: {player.get('expected_return_date')}")

                        # Check for false positives
                        injury_details = player.get('injury_details')
                        if not expected_injured and injury_details:
                            print(f"  ✗ FALSE POSITIVE: Healthy player got injury_details!")
                        elif expected_injured and injury_details:
                            print(f"  ✓ Correct: Injured player has injury_details")
                        elif expected_injured and not injury_details:
                            print(f"  ? Injured player missing injury_details")
                        else:
                            print(f"  ✓ Correct: Healthy player has no injury_details")
                        break
                if found:
                    break

            if not found:
                print(f"\n--- {target}: NOT FOUND in processed rosters ---")

        # =====================================================================
        # STEP 4: Summary
        # =====================================================================
        print_section("STEP 4: SUMMARY")

        false_positives = [r for r in results if not r['expected_injured'] and r['has_injury_details']]
        missing_injury = [r for r in results if r['expected_injured'] and not r['has_injury_details']]

        print("\n[FALSE POSITIVES - Healthy players with injuryDetails]")
        if false_positives:
            for r in false_positives:
                print(f"  ✗ {r['name']}: has_injury_details={r['has_injury_details']}")
            print("\n  DIAGNOSIS: ESPN is returning injuryDetails for healthy players")
            print("  OR our extraction code is creating false injury data")
        else:
            print("  ✓ No false positives found in raw API data")

        print("\n[MISSING INJURY DATA - Injured players without injuryDetails]")
        if missing_injury:
            for r in missing_injury:
                print(f"  ? {r['name']}: has_injury_details={r['has_injury_details']}, "
                      f"injury_status={r['injury_status']}")
            print("\n  NOTE: ESPN may not provide injuryDetails for all injury types")
        else:
            print("  ✓ All injured players have injuryDetails")

        print("\n[CONCLUSION]")
        print("If false positives exist in RAW API → ESPN is the source")
        print("If false positives exist only in get_all_rosters → Our code is the source")
        print("If no false positives anywhere → The issue may be in hybrid_engine or optimizer")


if __name__ == "__main__":
    main()

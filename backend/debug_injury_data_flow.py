#!/usr/bin/env python3
"""
Debug script to trace JJJ's injury_details through the entire data pipeline.

This script identifies EXACTLY where injury_details data is lost by printing
the data structure at each step of the flow:
  0. Direct raw HTTP API call (bypassing espn-api library)
  1. espn-api library Player object attributes
  2. After _parse_player - in player_data dict
  3. After get_all_rosters returns - in the returned data structure

Usage:
    cd /Users/milo/fantasy-basketball-optimizer
    source venv/bin/activate
    python backend/debug_injury_data_flow.py
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


def find_jjj_in_dict(data: Dict, path: str = "") -> list:
    """Recursively find any dict containing JJJ."""
    results = []

    if isinstance(data, dict):
        name = data.get('name', '') or data.get('fullName', '') or ''
        if 'jaren' in name.lower() or 'jackson' in name.lower():
            results.append((path, data))
        for key, value in data.items():
            results.extend(find_jjj_in_dict(value, f"{path}.{key}" if path else key))
    elif isinstance(data, list):
        for i, item in enumerate(data):
            results.extend(find_jjj_in_dict(item, f"{path}[{i}]"))

    return results


def extract_injury_fields(player_data: Dict) -> Dict:
    """Extract all injury-related fields from a player dict."""
    injury_fields = {}
    injury_keywords = ['injury', 'return', 'out', 'status', 'health', 'ir', 'details']

    for key, value in player_data.items():
        key_lower = key.lower()
        if any(kw in key_lower for kw in injury_keywords):
            injury_fields[key] = value

    return injury_fields


def main():
    print("=" * 80)
    print("JJJ INJURY_DETAILS DATA FLOW TRACE")
    print("Tracking injury_details through the entire pipeline")
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

        # Create ESPN client
        client = ESPNClient(
            league_id=league.espn_league_id,
            year=league.season,
            espn_s2=league.espn_s2_cookie,
            swid=league.swid_cookie
        )

        # =====================================================================
        # STEP 0: Direct Raw HTTP API Call (bypassing espn-api library)
        # =====================================================================
        print_section("STEP 0: DIRECT RAW HTTP API CALL")

        print("\nMaking raw HTTP request with kona_playercard view...")
        print("(This bypasses the espn-api library to see what ESPN actually returns)")

        endpoint = (
            f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/fba/"
            f"seasons/{league.season}/segments/0/leagues/{league.espn_league_id}"
        )
        params = {'view': ['mTeam', 'mRoster', 'kona_player_info', 'kona_playercard']}
        cookies = {'espn_s2': league.espn_s2_cookie, 'SWID': league.swid_cookie}

        response = requests.get(endpoint, params=params, cookies=cookies, timeout=30)
        response.raise_for_status()
        raw_api_data = response.json()

        # Find JJJ in raw API response
        jjj_raw_api = None
        jjj_entry_api = None
        jjj_pool_api = None

        for team_data in raw_api_data.get('teams', []):
            for entry in team_data.get('roster', {}).get('entries', []):
                player_pool = entry.get('playerPoolEntry', {})
                player = player_pool.get('player', {})
                name = player.get('fullName', '')

                if 'jaren' in name.lower() or 'jackson' in name.lower():
                    jjj_raw_api = player
                    jjj_entry_api = entry
                    jjj_pool_api = player_pool
                    print(f"\nFound in raw API: {name}")
                    break
            if jjj_raw_api:
                break

        if jjj_raw_api:
            print("\n--- RAW API: player object keys ---")
            print(f"  Keys: {list(jjj_raw_api.keys())}")

            print("\n--- RAW API: injuryDetails at each level ---")
            print(f"  player.injuryDetails = {jjj_raw_api.get('injuryDetails')}")
            print(f"  entry.injuryDetails = {jjj_entry_api.get('injuryDetails')}")
            print(f"  playerPoolEntry.injuryDetails = {jjj_pool_api.get('injuryDetails')}")

            print("\n--- RAW API: All injury-related fields in player ---")
            injury_keywords = ['injury', 'return', 'out', 'status', 'health']
            for key, value in jjj_raw_api.items():
                if any(kw in key.lower() for kw in injury_keywords):
                    print(f"  {key} = {value}")

            if jjj_raw_api.get('injuryDetails'):
                print("\n>>> SUCCESS: injuryDetails found in RAW API!")
                print(f">>> injuryDetails = {jjj_raw_api.get('injuryDetails')}")
            else:
                print("\n>>> WARNING: No injuryDetails in RAW API response")
                print(">>> ESPN may not provide this data for this player")
        else:
            print("ERROR: JJJ not found in raw API response!")

        # =====================================================================
        # STEP 1: espn-api Library Player Object
        # =====================================================================
        print_section("STEP 1: ESPN-API LIBRARY PLAYER OBJECT")

        print("\nAccessing player through espn-api library...")

        # Access the underlying espn-api league object
        espn_league = client.league

        # Find JJJ in teams
        jjj_raw = None
        jjj_team_name = None

        for team in espn_league.teams:
            for player in team.roster:
                if 'jaren' in player.name.lower() or 'jackson' in player.name.lower():
                    jjj_raw = player
                    jjj_team_name = team.team_name
                    break
            if jjj_raw:
                break

        if not jjj_raw:
            print("ERROR: Could not find Jaren Jackson Jr. in any roster!")
            print("\nSearching all teams for players with 'jackson' in name...")
            for team in espn_league.teams:
                for player in team.roster:
                    if 'jackson' in player.name.lower():
                        print(f"  Found: {player.name} on {team.team_name}")
            sys.exit(1)

        print(f"\nFound: {jjj_raw.name} on team: {jjj_team_name}")

        # Print ALL attributes of the raw player object
        print("\n--- Raw Player Object Attributes ---")
        attrs = [attr for attr in dir(jjj_raw) if not attr.startswith('_')]
        print(f"Available attributes: {attrs}")

        print("\n--- Injury-Related Raw Attributes ---")
        injury_attrs = ['injuryStatus', 'injured', 'injuryDetails', 'injuryDate',
                       'injuryNotes', 'seasonOutlook', 'status', 'lineupSlot']
        for attr in injury_attrs:
            value = getattr(jjj_raw, attr, 'NOT_FOUND')
            print(f"  {attr}: {value}")

        # Special focus on injuryDetails
        raw_injury_details = getattr(jjj_raw, 'injuryDetails', None)
        print(f"\n>>> RAW injuryDetails type: {type(raw_injury_details)}")
        print(f">>> RAW injuryDetails value: {raw_injury_details}")

        if raw_injury_details and isinstance(raw_injury_details, dict):
            print(">>> injuryDetails IS a dict with keys:", list(raw_injury_details.keys()))
            for k, v in raw_injury_details.items():
                print(f"    {k}: {v} (type: {type(v).__name__})")

        # =====================================================================
        # STEP 2: After _parse_player
        # =====================================================================
        print_section("STEP 2: AFTER _parse_player() METHOD")

        print("\nCalling client._parse_player() directly...")

        # Get team ID for this player
        team_id = None
        for team in espn_league.teams:
            for player in team.roster:
                if player.name == jjj_raw.name:
                    team_id = team.team_id
                    break
            if team_id:
                break

        # Call _parse_player
        parsed_data = client._parse_player(jjj_raw, team_id, is_debug_player=True)

        print("\n--- Parsed Player Data (Full) ---")
        print(safe_json(parsed_data))

        print("\n--- injury_details After _parse_player ---")
        injury_details_after_parse = parsed_data.get('injury_details')
        print(f"  Type: {type(injury_details_after_parse)}")
        print(f"  Value: {injury_details_after_parse}")

        if injury_details_after_parse is None:
            print("  >>> ALERT: injury_details is None!")
        elif isinstance(injury_details_after_parse, dict):
            if injury_details_after_parse:
                print("  >>> injury_details has data")
                for k, v in injury_details_after_parse.items():
                    print(f"      {k}: {v}")
            else:
                print("  >>> ALERT: injury_details is EMPTY dict {}")

        # =====================================================================
        # STEP 3: After get_all_rosters
        # =====================================================================
        print_section("STEP 3: AFTER get_all_rosters() METHOD")

        print("\nCalling client.get_all_rosters()...")
        all_rosters = client.get_all_rosters()

        # Find JJJ in the returned rosters
        jjj_from_rosters = None
        jjj_roster_team = None

        for team_id, team_data in all_rosters.items():
            for player in team_data.get('roster', []):
                name = player.get('name', '')
                if 'jaren' in name.lower() or 'jackson' in name.lower():
                    jjj_from_rosters = player
                    jjj_roster_team = team_data.get('team_name', team_id)
                    break
            if jjj_from_rosters:
                break

        if not jjj_from_rosters:
            print("ERROR: JJJ not found in get_all_rosters() output!")
        else:
            print(f"\nFound JJJ in rosters: {jjj_from_rosters.get('name')} on {jjj_roster_team}")

            print("\n--- JJJ Data From get_all_rosters ---")
            injury_fields = extract_injury_fields(jjj_from_rosters)
            print(f"Injury-related fields: {safe_json(injury_fields)}")

            print("\n--- injury_details After get_all_rosters ---")
            injury_details_from_rosters = jjj_from_rosters.get('injury_details')
            print(f"  Type: {type(injury_details_from_rosters)}")
            print(f"  Value: {injury_details_from_rosters}")

            if injury_details_from_rosters is None:
                print("  >>> injury_details is None (correct for healthy players)")
            elif isinstance(injury_details_from_rosters, dict):
                if injury_details_from_rosters:
                    print("  >>> injury_details has data:")
                    for k, v in injury_details_from_rosters.items():
                        print(f"      {k}: {v}")
                else:
                    print("  >>> ALERT: injury_details is EMPTY dict {}")

            # Check all keys in the player dict
            print("\n--- All Keys in JJJ Player Dict ---")
            print(f"  Keys: {list(jjj_from_rosters.keys())}")

        # =====================================================================
        # STEP 4: Data Flow Summary
        # =====================================================================
        print_section("STEP 4: DATA FLOW SUMMARY")

        print("\n=== COMPARISON AT EACH STEP ===\n")

        # Step 0 data (raw HTTP)
        step0_injury_details = jjj_raw_api.get('injuryDetails') if jjj_raw_api else None
        print(f"STEP 0 (Raw HTTP API):")
        print(f"  injuryDetails = {step0_injury_details}")
        print(f"  Type = {type(step0_injury_details)}")

        # Step 1 data (espn-api library)
        step1_injury_details = raw_injury_details
        print(f"\nSTEP 1 (espn-api library Player object):")
        print(f"  injuryDetails = {step1_injury_details}")
        print(f"  Type = {type(step1_injury_details)}")

        # Step 2 data
        step2_injury_details = parsed_data.get('injury_details')
        print(f"\nSTEP 2 (After _parse_player):")
        print(f"  injury_details = {step2_injury_details}")
        print(f"  Type = {type(step2_injury_details)}")

        # Step 3 data
        step3_injury_details = jjj_from_rosters.get('injury_details') if jjj_from_rosters else None
        print(f"\nSTEP 3 (After get_all_rosters):")
        print(f"  injury_details = {step3_injury_details}")
        print(f"  Type = {type(step3_injury_details)}")

        # Identify where data changes
        print("\n=== DATA FLOW ANALYSIS ===\n")

        if step0_injury_details and not step1_injury_details:
            print(">>> DATA LOST between Step 0 and Step 1")
            print("    The espn-api library is NOT exposing injuryDetails!")
            print("    Solution: Use raw HTTP API calls instead of espn-api for injury data")
        elif step0_injury_details and step3_injury_details:
            print(">>> SUCCESS: get_all_rosters() captures injuryDetails from raw API!")
            print("    The data flows correctly when using raw HTTP requests")
        elif step1_injury_details and not step2_injury_details:
            print(">>> DATA LOST between Step 1 and Step 2 (_parse_player)")
            print("    Check _parse_player() method for parsing issues")
        elif step2_injury_details and not step3_injury_details:
            print(">>> DATA LOST between Step 2 and Step 3 (get_all_rosters)")
            print("    Check get_all_rosters() method for data transfer issues")
        elif step0_injury_details and step1_injury_details and step2_injury_details and step3_injury_details:
            print(">>> DATA PRESERVED through all steps!")
            print("    injury_details flows correctly from ESPN to output")
        elif not step0_injury_details:
            print(">>> NO DATA from ESPN API (even raw HTTP)")
            print("    ESPN may not be providing injuryDetails for this player")
            print("    Check if player has active injury or is healthy")

        # Check if player is actually injured
        print("\n=== PLAYER INJURY STATUS ===\n")
        injury_status = getattr(jjj_raw, 'injuryStatus', None) if jjj_raw else None
        lineup_slot = getattr(jjj_raw, 'lineupSlot', None) if jjj_raw else None

        # Also check raw API data
        raw_api_injury_status = jjj_raw_api.get('injuryStatus') if jjj_raw_api else None

        print(f"espn-api injuryStatus: {injury_status}")
        print(f"raw API injuryStatus: {raw_api_injury_status}")
        print(f"lineupSlot: {lineup_slot}")

        if raw_api_injury_status in ['ACTIVE', None] and lineup_slot not in ['IR', 'IR+']:
            print("\n>>> Player appears to be HEALTHY or not on IR")
            print("    ESPN does not provide injuryDetails for healthy players")
            print("    injury_details = None is CORRECT for this player")
        else:
            print("\n>>> Player appears to be INJURED or on IR")
            if not step0_injury_details:
                print("    But ESPN is NOT providing injuryDetails in raw API")
                print("    This may be an ESPN data limitation or the kona_playercard")
                print("    view may need different parameters")
            elif step0_injury_details and step3_injury_details:
                print("    And injuryDetails IS available via raw API + get_all_rosters!")
                print("    SUCCESS: The pipeline is working correctly")


if __name__ == "__main__":
    main()

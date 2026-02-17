#!/usr/bin/env python3
"""
Comprehensive Roster Limits Analysis Script.

This script does a deep dive into ESPN Fantasy Basketball API to find
all roster configuration, lineup settings, position limits, and game limits.

Usage:
    cd backend
    source ../venv/bin/activate
    python debug_roster_limits.py
"""

import os
import sys
import sqlite3
import json
import requests
from pprint import pprint

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def get_league_credentials():
    """Get league credentials from database."""
    db_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'instance',
        'fantasy_basketball.db'
    )

    if not os.path.exists(db_path):
        db_path = os.path.join(
            os.path.dirname(__file__),
            'instance',
            'fantasy_basketball.db'
        )

    print(f"Database path: {db_path}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT espn_league_id, espn_s2_cookie, swid_cookie, league_name, season
        FROM leagues LIMIT 1
    ''')
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    return {
        'espn_league_id': row[0],
        'espn_s2': row[1],
        'swid': row[2],
        'league_name': row[3],
        'season': row[4],
    }


# ESPN Position ID to Name mapping
POSITION_ID_MAP = {
    0: 'PG',
    1: 'SG',
    2: 'SF',
    3: 'PF',
    4: 'C',
    5: 'G',      # PG/SG
    6: 'F',      # SF/PF
    7: 'SG/SF',
    8: 'G/F',
    9: 'PF/C',
    10: 'F/C',
    11: 'UTIL',
    12: 'BE',    # Bench
    13: 'IR',    # Injured Reserve
    14: 'IR+',   # Injured Reserve Plus
}


def search_for_keywords(obj, keywords, path="", max_depth=5, results=None):
    """Recursively search for fields containing keywords."""
    if results is None:
        results = []

    if max_depth <= 0:
        return results

    keywords_lower = [k.lower() for k in keywords]

    if isinstance(obj, dict):
        for key, value in obj.items():
            key_lower = key.lower()
            # Check if key contains any keyword
            for kw in keywords_lower:
                if kw in key_lower:
                    results.append({
                        'path': f"{path}.{key}" if path else key,
                        'key': key,
                        'matched_keyword': kw,
                        'value_type': type(value).__name__,
                        'value_preview': str(value)[:200] if not isinstance(value, (dict, list)) else f"<{type(value).__name__}>"
                    })
                    break

            # Recurse into nested structures
            if isinstance(value, (dict, list)):
                search_for_keywords(value, keywords, f"{path}.{key}" if path else key, max_depth - 1, results)

    elif isinstance(obj, list) and len(obj) > 0:
        # Only check first item of lists
        search_for_keywords(obj[0], keywords, f"{path}[0]", max_depth - 1, results)

    return results


def fetch_raw_api_all_views(league_id, espn_s2, swid, season):
    """Fetch raw ESPN API with multiple views."""
    endpoint = (
        f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/fba/"
        f"seasons/{season}/segments/0/leagues/{league_id}"
    )

    # Request all possible views
    all_views = [
        'mSettings', 'mRoster', 'mTeam', 'mMatchup', 'mStatus',
        'mLiveScoring', 'mPositionalRatings', 'mSchedule', 'mScoreboard'
    ]

    params = {'view': all_views}
    cookies = {'espn_s2': espn_s2, 'SWID': swid}

    response = requests.get(endpoint, params=params, cookies=cookies, timeout=15)

    if response.status_code != 200:
        print(f"API request failed with status {response.status_code}")
        return None

    return response.json()


def print_all_top_level_keys(raw_data):
    """Print ALL top-level keys in the API response."""
    print("\n" + "=" * 80)
    print("STEP 1: ALL TOP-LEVEL KEYS IN API RESPONSE")
    print("=" * 80)

    for key in sorted(raw_data.keys()):
        value = raw_data[key]
        if isinstance(value, dict):
            print(f"  {key}: <dict with {len(value)} keys>")
            # Show first 5 sub-keys
            sub_keys = list(value.keys())[:5]
            for sk in sub_keys:
                print(f"    - {sk}")
            if len(value) > 5:
                print(f"    ... and {len(value) - 5} more keys")
        elif isinstance(value, list):
            print(f"  {key}: <list with {len(value)} items>")
        else:
            print(f"  {key}: {value}")


def print_all_settings_keys(raw_data):
    """Print ALL keys in the settings object."""
    print("\n" + "=" * 80)
    print("STEP 2: ALL KEYS IN 'settings' OBJECT")
    print("=" * 80)

    settings = raw_data.get('settings', {})
    if not settings:
        print("  No 'settings' key found in API response")
        return

    def print_dict_structure(d, indent=0, max_depth=3):
        if max_depth <= 0:
            return
        prefix = "  " * indent
        for key in sorted(d.keys()):
            value = d[key]
            if isinstance(value, dict):
                print(f"{prefix}{key}: <dict>")
                print_dict_structure(value, indent + 1, max_depth - 1)
            elif isinstance(value, list):
                print(f"{prefix}{key}: <list[{len(value)}]>")
                if len(value) > 0 and isinstance(value[0], dict):
                    print(f"{prefix}  [0]: <dict with keys: {list(value[0].keys())[:5]}...>")
            else:
                val_str = str(value)
                if len(val_str) > 60:
                    val_str = val_str[:60] + "..."
                print(f"{prefix}{key}: {val_str}")

    print_dict_structure(settings)


def search_for_roster_keywords(raw_data):
    """Look for any field containing roster-related keywords."""
    print("\n" + "=" * 80)
    print("STEP 3: SEARCH FOR ROSTER-RELATED KEYWORDS")
    print("=" * 80)

    keywords = ['roster', 'lineup', 'position', 'slot', 'limit', 'game', 'start', 'max', 'cap']

    print(f"\nSearching for fields containing: {', '.join(keywords)}")
    print("-" * 80)

    results = search_for_keywords(raw_data, keywords, max_depth=6)

    # Group by keyword
    by_keyword = {}
    for r in results:
        kw = r['matched_keyword']
        if kw not in by_keyword:
            by_keyword[kw] = []
        by_keyword[kw].append(r)

    for kw in sorted(by_keyword.keys()):
        matches = by_keyword[kw]
        print(f"\n  Keyword '{kw}' found in {len(matches)} fields:")
        for m in matches[:10]:  # Limit to 10 per keyword
            print(f"    Path: {m['path']}")
            print(f"      Type: {m['value_type']}, Preview: {m['value_preview'][:100]}")
        if len(matches) > 10:
            print(f"    ... and {len(matches) - 10} more matches")


def inspect_espn_api_package(credentials):
    """Use espn-api package and inspect league.settings."""
    print("\n" + "=" * 80)
    print("STEP 4: INSPECT espn-api PACKAGE OBJECTS")
    print("=" * 80)

    try:
        from espn_api.basketball import League

        league = League(
            league_id=int(credentials['espn_league_id']),
            year=int(credentials['season']),
            espn_s2=credentials['espn_s2'],
            swid=credentials['swid']
        )

        print(f"\nLeague: {league.settings.name}")

        # Inspect league object
        print("\n--- league attributes (dir) ---")
        league_attrs = [a for a in dir(league) if not a.startswith('_')]
        print(f"  {league_attrs}")

        # Inspect league.settings
        print("\n--- league.settings attributes (dir) ---")
        settings = league.settings
        settings_attrs = [a for a in dir(settings) if not a.startswith('_')]
        print(f"  {settings_attrs}")

        # Print each attribute value
        print("\n--- league.settings attribute values ---")
        for attr in settings_attrs:
            try:
                value = getattr(settings, attr)
                if callable(value):
                    continue
                val_str = str(value)
                if len(val_str) > 100:
                    val_str = val_str[:100] + "..."
                print(f"  {attr}: {val_str}")
            except Exception as e:
                print(f"  {attr}: <error: {e}>")

        # Check for __dict__
        print("\n--- league.settings.__dict__ ---")
        if hasattr(settings, '__dict__'):
            for key, value in settings.__dict__.items():
                val_str = str(value)
                if len(val_str) > 100:
                    val_str = val_str[:100] + "..."
                print(f"  {key}: {val_str}")
        else:
            print("  No __dict__ available")

        # Check roster attribute
        print("\n--- league.settings.roster (if exists) ---")
        if hasattr(settings, 'roster'):
            roster = settings.roster
            print(f"  Type: {type(roster)}")
            if isinstance(roster, dict):
                for k, v in roster.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  Value: {roster}")
        else:
            print("  No 'roster' attribute")

        # Check lineup settings
        print("\n--- league.settings.lineup_slot_counts (if exists) ---")
        for attr_name in ['lineup_slot_counts', 'lineupSlotCounts', 'lineup_slots', 'roster_slots']:
            if hasattr(settings, attr_name):
                print(f"  {attr_name}: {getattr(settings, attr_name)}")

        return league

    except ImportError:
        print("  espn-api package not installed")
        return None
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def check_teams_for_position_info(raw_data):
    """Check if there's position information in teams data."""
    print("\n" + "=" * 80)
    print("STEP 5: POSITION INFO IN TEAMS DATA")
    print("=" * 80)

    teams = raw_data.get('teams', [])
    if not teams:
        print("  No teams data found")
        return

    print(f"\n  Found {len(teams)} teams")

    # Check first team structure
    team = teams[0]
    print(f"\n--- First team keys ---")
    for key in sorted(team.keys()):
        value = team[key]
        if isinstance(value, dict):
            print(f"  {key}: <dict with {len(value)} keys>")
        elif isinstance(value, list):
            print(f"  {key}: <list with {len(value)} items>")
        else:
            print(f"  {key}: {value}")

    # Check roster structure
    roster = team.get('roster', {})
    if roster:
        print(f"\n--- team.roster keys ---")
        for key in roster.keys():
            value = roster[key]
            if isinstance(value, list):
                print(f"  {key}: <list with {len(value)} items>")
            else:
                print(f"  {key}: {value}")


def print_sample_team_roster(raw_data, credentials):
    """Print a sample team's roster with player positions."""
    print("\n" + "=" * 80)
    print("STEP 6: SAMPLE TEAM ROSTER (YOUR TEAM)")
    print("=" * 80)

    teams = raw_data.get('teams', [])
    swid = credentials['swid']

    # Find user's team
    my_team = None
    for team in teams:
        owners = team.get('owners', [])
        for owner in owners:
            owner_id = owner.get('id') if isinstance(owner, dict) else owner
            if owner_id == swid:
                my_team = team
                break
        if my_team:
            break

    if not my_team:
        for team in teams:
            name = team.get('name', '') or team.get('nickname', '')
            if 'doncic' in name.lower():
                my_team = team
                break

    if not my_team:
        my_team = teams[0] if teams else None

    if not my_team:
        print("  No team found")
        return

    team_name = my_team.get('name', '') or my_team.get('nickname', 'Unknown')
    print(f"\n  Team: {team_name}")

    roster_entries = my_team.get('roster', {}).get('entries', [])
    print(f"  Roster entries: {len(roster_entries)}")

    if not roster_entries:
        print("  No roster entries found")
        return

    # Print first entry structure completely
    print("\n--- FIRST PLAYER COMPLETE STRUCTURE ---")
    first_entry = roster_entries[0]

    def print_nested(obj, indent=0, max_depth=4):
        if max_depth <= 0:
            return
        prefix = "  " * indent
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, dict):
                    print(f"{prefix}{key}: <dict>")
                    print_nested(value, indent + 1, max_depth - 1)
                elif isinstance(value, list):
                    print(f"{prefix}{key}: <list[{len(value)}]>")
                    if len(value) > 0 and max_depth > 1:
                        print_nested(value[0], indent + 1, max_depth - 1)
                else:
                    val_str = str(value)
                    if len(val_str) > 80:
                        val_str = val_str[:80] + "..."
                    print(f"{prefix}{key}: {val_str}")

    print_nested(first_entry, indent=1)

    # Print all players with positions
    print("\n--- ALL PLAYERS ---")
    print(f"{'Player':<25} {'SlotID':<8} {'Slot':<10} {'DefPosID':<10} {'EligibleSlots':<40}")
    print("-" * 100)

    for entry in roster_entries:
        player = entry.get('playerPoolEntry', {}).get('player', {})
        name = player.get('fullName', 'Unknown')[:24]

        lineup_slot_id = entry.get('lineupSlotId', -1)
        slot_name = POSITION_ID_MAP.get(lineup_slot_id, f'?{lineup_slot_id}')

        default_pos_id = player.get('defaultPositionId', -1)

        eligible_slots = player.get('eligibleSlots', [])
        eligible_names = [POSITION_ID_MAP.get(s, f'?{s}') for s in eligible_slots[:8]]
        eligible_str = ','.join(eligible_names)
        if len(eligible_slots) > 8:
            eligible_str += f'...(+{len(eligible_slots)-8})'

        print(f"{name:<25} {lineup_slot_id:<8} {slot_name:<10} {default_pos_id:<10} {eligible_str:<40}")


def search_for_limits(raw_data):
    """Search specifically for start/games limits."""
    print("\n" + "=" * 80)
    print("STEP 7: DEEP SEARCH FOR LIMITS/CAPS")
    print("=" * 80)

    settings = raw_data.get('settings', {})

    # List all settings that might contain limits
    limit_candidates = [
        'rosterSettings', 'scoringSettings', 'acquisitionSettings',
        'tradeSettings', 'draftSettings', 'scheduleSettings'
    ]

    for candidate in limit_candidates:
        if candidate in settings:
            print(f"\n--- {candidate} ---")
            obj = settings[candidate]
            if isinstance(obj, dict):
                for key, value in obj.items():
                    key_lower = key.lower()
                    if any(kw in key_lower for kw in ['limit', 'max', 'cap', 'count', 'slot', 'start', 'game']):
                        print(f"  ** {key}: {value}")
                    else:
                        val_str = str(value)
                        if len(val_str) > 60:
                            val_str = val_str[:60] + "..."
                        print(f"  {key}: {val_str}")

    # Check for statLimits at various levels
    print("\n--- Searching for 'statLimits' anywhere ---")
    stat_limit_results = search_for_keywords(raw_data, ['statLimit', 'stat_limit', 'gameLim', 'game_limit', 'startLim', 'start_limit'], max_depth=6)
    for r in stat_limit_results:
        print(f"  Found: {r['path']} = {r['value_preview']}")

    if not stat_limit_results:
        print("  No explicit stat/game/start limits found")

    # Print rosterSettings completely
    roster_settings = settings.get('rosterSettings', {})
    if roster_settings:
        print("\n--- COMPLETE rosterSettings ---")
        for key, value in roster_settings.items():
            print(f"  {key}: {value}")


def main():
    print("=" * 80)
    print("COMPREHENSIVE ROSTER CONFIGURATION SEARCH")
    print("=" * 80)

    # Get credentials
    print("\nLoading league credentials...")
    credentials = get_league_credentials()

    if not credentials:
        print("ERROR: No league found in database")
        return 1

    print(f"  League: {credentials['league_name']}")
    print(f"  ESPN League ID: {credentials['espn_league_id']}")
    print(f"  Season: {credentials['season']}")

    # Fetch raw API data with all views
    print("\nFetching raw ESPN API data (all views)...")
    raw_data = fetch_raw_api_all_views(
        credentials['espn_league_id'],
        credentials['espn_s2'],
        credentials['swid'],
        credentials['season']
    )

    if not raw_data:
        print("ERROR: Failed to fetch API data")
        return 1

    print("  ✓ API data fetched successfully")

    # Step 1: Print ALL top-level keys
    print_all_top_level_keys(raw_data)

    # Step 2: Print ALL settings keys
    print_all_settings_keys(raw_data)

    # Step 3: Search for roster-related keywords
    search_for_roster_keywords(raw_data)

    # Step 4: Inspect espn-api package
    inspect_espn_api_package(credentials)

    # Step 5: Check teams for position info
    check_teams_for_position_info(raw_data)

    # Step 6: Print sample team roster
    print_sample_team_roster(raw_data, credentials)

    # Step 7: Deep search for limits
    search_for_limits(raw_data)

    print("\n" + "=" * 80)
    print("SEARCH COMPLETE")
    print("=" * 80)
    print("\nLook for fields containing:")
    print("  - lineupSlotCounts: Number of each position slot")
    print("  - lineupSlotStatLimits: Max starts per position")
    print("  - positionLimits: Position-specific limits")
    print("  - acquisitionLimit, moveLimit: Transaction limits")
    print("  - Any field with 'limit', 'max', 'cap' in the name")

    return 0


if __name__ == '__main__':
    sys.exit(main())

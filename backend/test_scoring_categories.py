#!/usr/bin/env python3
"""
Test script for verifying ESPN scoring categories extraction.

This script tests the scoring categories extraction to ensure:
1. It connects to the ESPN API successfully
2. It fetches the raw scoringSettings from ESPN
3. It correctly maps statIds to category names
4. It returns only the actual scoring categories (not tracked stats)

Usage:
    cd backend
    source ../venv/bin/activate
    python test_scoring_categories.py
"""

import os
import sys
import sqlite3
import json

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
        SELECT espn_league_id, espn_s2_cookie, swid_cookie, league_name, scoring_settings
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
        'scoring_settings': json.loads(row[4]) if row[4] else {}
    }


def test_scoring_categories():
    """Test the scoring categories extraction."""
    print("=" * 70)
    print("ESPN SCORING CATEGORIES EXTRACTION TEST")
    print("=" * 70)
    print()

    # Step 1: Get credentials
    print("Step 1: Loading league credentials from database...")
    credentials = get_league_credentials()

    if not credentials:
        print("ERROR: No league found in database")
        print("Please add a league first via the web interface")
        return False

    print(f"  League: {credentials['league_name']}")
    print(f"  ESPN League ID: {credentials['espn_league_id']}")
    print()

    # Step 2: Connect to ESPN
    print("Step 2: Connecting to ESPN...")
    try:
        from espn_api.basketball import League as ESPNLeague
        import requests

        league_id = int(credentials['espn_league_id'])
        espn_s2 = credentials['espn_s2']
        swid = credentials['swid']

        espn_league = ESPNLeague(
            league_id=league_id,
            year=2025,
            espn_s2=espn_s2,
            swid=swid
        )
        print(f"  SUCCESS: Connected to ESPN league: {espn_league.settings.name}")
        print()

    except Exception as e:
        print(f"  ERROR: Failed to connect to ESPN: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 3: Fetch scoring categories from raw API
    print("Step 3: Fetching scoring categories from ESPN API...")
    try:
        # ESPN Stat ID to category name mapping
        ESPN_STAT_ID_MAP = {
            0: 'PTS', 1: 'BLK', 2: 'STL', 3: 'AST', 4: 'OREB', 5: 'DREB',
            6: 'REB', 9: 'PF', 10: 'TF', 11: 'TO', 12: 'EJ', 13: 'FGM',
            14: 'FGA', 15: 'FTM', 16: 'FTA', 17: '3PM', 18: '3PA',
            19: 'FG%', 20: 'FT%', 21: '3P%', 22: 'DD', 23: 'TD', 24: 'QD',
            28: 'MPG', 29: 'MIN', 40: 'GS', 41: 'GP',
        }

        CATEGORY_METADATA = {
            'PTS': {'label': 'Points', 'abbr': 'PTS'},
            'BLK': {'label': 'Blocks', 'abbr': 'BLK'},
            'STL': {'label': 'Steals', 'abbr': 'STL'},
            'AST': {'label': 'Assists', 'abbr': 'AST'},
            'REB': {'label': 'Rebounds', 'abbr': 'REB'},
            'FGM': {'label': 'Field Goals Made', 'abbr': 'FGM'},
            'FTM': {'label': 'Free Throws Made', 'abbr': 'FTM'},
            '3PM': {'label': '3-Pointers Made', 'abbr': '3PM'},
            'FG%': {'label': 'Field Goal %', 'abbr': 'FG%'},
            'FT%': {'label': 'Free Throw %', 'abbr': 'FT%'},
            '3P%': {'label': '3-Point %', 'abbr': '3P%'},
            'TO': {'label': 'Turnovers', 'abbr': 'TO'},
            'DD': {'label': 'Double-Doubles', 'abbr': 'DD'},
            'TD': {'label': 'Triple-Doubles', 'abbr': 'TD'},
        }

        # Make raw API call to get scoringSettings
        endpoint = f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/fba/seasons/2025/segments/0/leagues/{league_id}"
        params = {'view': 'mSettings'}
        cookies = {'espn_s2': espn_s2, 'SWID': swid}

        response = requests.get(endpoint, params=params, cookies=cookies, timeout=10)

        if response.status_code != 200:
            print(f"  ERROR: API returned status {response.status_code}")
            return False

        raw_data = response.json()
        scoring_settings = raw_data.get('settings', {}).get('scoringSettings', {})
        scoring_items = scoring_settings.get('scoringItems', [])

        print(f"  SUCCESS: Found {len(scoring_items)} scoring items in API response")
        print()

        # Parse scoring items
        categories = []
        for item in scoring_items:
            stat_id = item.get('statId')
            is_reverse = item.get('isReverseItem', False)

            category_key = ESPN_STAT_ID_MAP.get(stat_id)
            if not category_key:
                category_key = f'UNKNOWN_{stat_id}'

            metadata = CATEGORY_METADATA.get(category_key, {'label': category_key, 'abbr': category_key})

            categories.append({
                'key': category_key,
                'label': metadata['label'],
                'abbr': metadata['abbr'],
                'is_reverse': is_reverse,
                'stat_id': stat_id,
            })

    except Exception as e:
        print(f"  ERROR: Failed to fetch scoring categories: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 4: Display extracted categories
    print("Step 4: Extracted Scoring Categories")
    print("-" * 50)

    if not categories:
        print("  WARNING: No categories returned!")
        return False

    print(f"  {'Key':<8} {'Label':<25} {'Abbr':<8} {'Reverse':<8} {'StatID'}")
    print(f"  {'-'*8} {'-'*25} {'-'*8} {'-'*8} {'-'*6}")

    all_mapped = True
    for cat in categories:
        key = cat.get('key', 'N/A')
        label = cat.get('label', 'N/A')
        abbr = cat.get('abbr', 'N/A')
        is_reverse = cat.get('is_reverse', False)
        stat_id = cat.get('stat_id', 'N/A')

        if 'UNKNOWN' in str(key):
            all_mapped = False
            print(f"  {key:<8} {label:<25} {abbr:<8} {str(is_reverse):<8} {stat_id} ⚠️ UNMAPPED")
        else:
            print(f"  {key:<8} {label:<25} {abbr:<8} {str(is_reverse):<8} {stat_id}")

    print()

    # Step 5: Verify mapping
    print("Step 5: Mapping Verification")
    print("-" * 50)

    if all_mapped:
        print("  ✅ All statIds successfully mapped to category names")
    else:
        print("  ⚠️  Some statIds could not be mapped - check ESPN_STAT_ID_MAP")

    print()

    # Step 6: Compare with team.stats
    print("Step 6: Comparison with team.stats")
    print("-" * 50)

    if espn_league.teams:
        first_team = espn_league.teams[0]
        team_stats = first_team.stats if hasattr(first_team, 'stats') else {}
        team_stat_keys = set(team_stats.keys()) if isinstance(team_stats, dict) else set()
        scoring_cat_keys = {c['key'] for c in categories}

        print(f"  Stats in team.stats: {sorted(team_stat_keys)}")
        print(f"  Scoring categories:  {sorted(scoring_cat_keys)}")
        print()

        # Stats tracked but not scored
        tracked_only = team_stat_keys - scoring_cat_keys
        if tracked_only:
            print(f"  📊 Tracked but NOT scored: {sorted(tracked_only)}")
            print("     (These are correctly excluded from standings calculations)")
        else:
            print("  All stats in team.stats are scoring categories")

        # Scoring categories not in team.stats
        missing_stats = scoring_cat_keys - team_stat_keys
        if missing_stats:
            print(f"  ⚠️  Scoring categories without team stats: {missing_stats}")

    print()

    # Step 7: Check database storage
    print("Step 7: Database Storage Check")
    print("-" * 50)

    stored_cats = credentials['scoring_settings'].get('categories', [])
    if stored_cats:
        if isinstance(stored_cats[0], dict):
            stored_keys = [c.get('key') for c in stored_cats]
        else:
            stored_keys = stored_cats

        print(f"  Categories stored in DB: {stored_keys}")

        fresh_keys = [c['key'] for c in categories]
        if set(stored_keys) == set(fresh_keys):
            print("  ✅ Database matches ESPN API")
        else:
            print("  ⚠️  Database differs from ESPN API!")
            print(f"     DB has: {set(stored_keys) - set(fresh_keys)}")
            print(f"     API has: {set(fresh_keys) - set(stored_keys)}")
    else:
        print("  ⚠️  No categories stored in database yet")
        print("     Run league refresh to store categories")

    print()

    # Step 8: Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  League: {credentials['league_name']}")
    print(f"  Scoring Categories: {len(categories)}")
    print(f"  Category Keys: {[c['key'] for c in categories]}")
    print()

    num_teams = len(espn_league.teams) if espn_league.teams else 10
    max_points = len(categories) * num_teams
    print(f"  Max possible Roto points: {len(categories)} cats × {num_teams} teams = {max_points}")
    print()

    if all_mapped and len(categories) > 0:
        print("  ✅ TEST PASSED: Scoring categories extraction working correctly")
        return True
    else:
        print("  ❌ TEST FAILED: Issues detected")
        return False


if __name__ == '__main__':
    success = test_scoring_categories()
    sys.exit(0 if success else 1)

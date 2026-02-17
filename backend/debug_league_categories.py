#!/usr/bin/env python3
"""
Debug script to investigate ESPN Fantasy Basketball league scoring categories.
This script helps identify where ESPN stores the actual Roto scoring categories
vs. general stats that are just tracked but not scored.
"""

import os
import sys
from pprint import pprint

# Add the backend directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from espn_api.basketball import League


def inspect_object(obj, name="Object", max_depth=2, current_depth=0):
    """Recursively inspect an object's attributes."""
    indent = "  " * current_depth
    print(f"{indent}=== {name} ===")

    if current_depth >= max_depth:
        print(f"{indent}  (max depth reached)")
        return

    # Get all attributes
    attrs = [a for a in dir(obj) if not a.startswith('_')]

    for attr in attrs:
        try:
            value = getattr(obj, attr)
            if callable(value):
                continue  # Skip methods

            value_type = type(value).__name__

            if isinstance(value, (str, int, float, bool, type(None))):
                print(f"{indent}  {attr}: {value} ({value_type})")
            elif isinstance(value, (list, tuple)):
                print(f"{indent}  {attr}: [{len(value)} items] ({value_type})")
                if len(value) > 0 and current_depth < max_depth - 1:
                    print(f"{indent}    First item: {value[0]}")
            elif isinstance(value, dict):
                print(f"{indent}  {attr}: {{{len(value)} keys}} ({value_type})")
                if len(value) > 0:
                    keys = list(value.keys())[:10]
                    print(f"{indent}    Keys: {keys}")
            else:
                print(f"{indent}  {attr}: ({value_type})")
        except Exception as e:
            print(f"{indent}  {attr}: <error: {e}>")


def main():
    # Get credentials from environment or use test values
    league_id = os.getenv('ESPN_LEAGUE_ID')
    espn_s2 = os.getenv('ESPN_S2')
    swid = os.getenv('SWID')
    year = 2025

    if not all([league_id, espn_s2, swid]):
        print("ERROR: Missing ESPN credentials. Set ESPN_LEAGUE_ID, ESPN_S2, and SWID in .env")
        print("\nTrying to load from database...")

        # Try to get from database
        try:
            import sqlite3
            # Check parent directory first (where Flask app typically runs)
            db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'instance', 'fantasy_basketball.db')
            if not os.path.exists(db_path):
                db_path = os.path.join(os.path.dirname(__file__), 'instance', 'fantasy_basketball.db')

            print(f"Looking for database at: {db_path}")

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT espn_league_id, espn_s2_cookie, swid_cookie, league_name FROM leagues LIMIT 1")
            row = cursor.fetchone()
            conn.close()

            if row:
                league_id = row[0]
                espn_s2 = row[1]
                swid = row[2]
                print(f"Found league in database: {row[3]}")
            else:
                print("No leagues found in database")
                return
        except Exception as e:
            print(f"Could not load from database: {e}")
            import traceback
            traceback.print_exc()
            return

    print("\n" + "="*80)
    print("CONNECTING TO ESPN FANTASY BASKETBALL LEAGUE")
    print("="*80)
    print(f"League ID: {league_id}")
    print(f"Year: {year}")

    try:
        league = League(
            league_id=int(league_id),
            year=year,
            espn_s2=espn_s2,
            swid=swid
        )
        print(f"Successfully connected to: {league.settings.name}")
    except Exception as e:
        print(f"ERROR connecting to league: {e}")
        return

    print("\n" + "="*80)
    print("1. LEAGUE OBJECT ATTRIBUTES")
    print("="*80)
    league_attrs = [a for a in dir(league) if not a.startswith('_') and not callable(getattr(league, a))]
    for attr in league_attrs:
        try:
            value = getattr(league, attr)
            print(f"  {attr}: {type(value).__name__}")
        except:
            pass

    print("\n" + "="*80)
    print("2. LEAGUE.SETTINGS - ALL ATTRIBUTES")
    print("="*80)
    settings = league.settings
    settings_attrs = [a for a in dir(settings) if not a.startswith('_') and not callable(getattr(settings, a))]

    for attr in sorted(settings_attrs):
        try:
            value = getattr(settings, attr)
            if isinstance(value, (str, int, float, bool, type(None))):
                print(f"  {attr}: {value}")
            elif isinstance(value, (list, tuple)):
                print(f"  {attr}: [{len(value)} items]")
                if len(value) > 0:
                    for i, item in enumerate(value[:5]):
                        print(f"    [{i}]: {item}")
                    if len(value) > 5:
                        print(f"    ... and {len(value) - 5} more")
            elif isinstance(value, dict):
                print(f"  {attr}: {{{len(value)} keys}}")
                for k, v in list(value.items())[:10]:
                    print(f"    {k}: {v}")
                if len(value) > 10:
                    print(f"    ... and {len(value) - 10} more keys")
            else:
                print(f"  {attr}: {type(value).__name__} object")
        except Exception as e:
            print(f"  {attr}: <error: {e}>")

    print("\n" + "="*80)
    print("3. SCORING/CATEGORY RELATED FIELDS IN SETTINGS")
    print("="*80)

    scoring_keywords = ['scor', 'stat', 'cat', 'point', 'value', 'rank', 'roto']
    for attr in settings_attrs:
        attr_lower = attr.lower()
        if any(kw in attr_lower for kw in scoring_keywords):
            try:
                value = getattr(settings, attr)
                print(f"\n  >>> {attr}:")
                pprint(value, indent=4, width=100)
            except Exception as e:
                print(f"  {attr}: <error: {e}>")

    print("\n" + "="*80)
    print("4. FIRST TEAM'S STATS DICTIONARY")
    print("="*80)

    if league.teams:
        first_team = league.teams[0]
        print(f"Team: {first_team.team_name}")
        print(f"\nTeam attributes:")
        team_attrs = [a for a in dir(first_team) if not a.startswith('_') and not callable(getattr(first_team, a))]
        for attr in team_attrs:
            try:
                value = getattr(first_team, attr)
                if isinstance(value, dict):
                    print(f"  {attr}: {{{len(value)} keys}}")
                elif isinstance(value, (list, tuple)):
                    print(f"  {attr}: [{len(value)} items]")
                else:
                    print(f"  {attr}: {value}")
            except:
                pass

        print(f"\nTeam.stats keys:")
        if hasattr(first_team, 'stats') and first_team.stats:
            stats = first_team.stats
            print(f"  Keys: {list(stats.keys())}")
            print(f"\n  Full stats dict:")
            pprint(stats, indent=4)
        else:
            print("  No stats attribute or empty")

        # Check for other stat-related attributes
        for attr in ['standing', 'points', 'scores', 'values']:
            if hasattr(first_team, attr):
                print(f"\n  Team.{attr}:")
                pprint(getattr(first_team, attr), indent=4)

    print("\n" + "="*80)
    print("5. CHECK FOR STAT_ID MAPPINGS")
    print("="*80)

    # ESPN uses stat IDs internally - try to find mappings
    if hasattr(settings, 'stat_category_ids'):
        print("Found stat_category_ids:")
        pprint(settings.stat_category_ids)

    if hasattr(settings, 'scoring_items'):
        print("Found scoring_items:")
        pprint(settings.scoring_items)

    # Check the raw data if available
    if hasattr(league, '_raw'):
        print("\nRaw league data keys:")
        print(list(league._raw.keys()) if isinstance(league._raw, dict) else "Not a dict")

    print("\n" + "="*80)
    print("6. LEAGUE TYPE AND SCORING TYPE")
    print("="*80)

    print(f"  League type (scoring_type): {getattr(settings, 'scoring_type', 'N/A')}")
    print(f"  Is Roto: {getattr(settings, 'scoring_type', '') == 'ROTO'}")

    # Check for reg_season_count, playoff info
    for attr in ['reg_season_count', 'playoff_team_count', 'playoff_seed_tie_rule']:
        if hasattr(settings, attr):
            print(f"  {attr}: {getattr(settings, attr)}")

    print("\n" + "="*80)
    print("7. ALL TEAMS - CATEGORY RANKS CHECK")
    print("="*80)

    print("Checking if teams have category-specific standings...")
    for team in league.teams[:3]:  # Just first 3 teams
        print(f"\n  {team.team_name}:")
        print(f"    standing: {getattr(team, 'standing', 'N/A')}")
        print(f"    points: {getattr(team, 'points', 'N/A')}")
        print(f"    stats type: {type(getattr(team, 'stats', None))}")

        # Check for any ranking attributes
        for attr in dir(team):
            if 'rank' in attr.lower() or 'stand' in attr.lower():
                try:
                    val = getattr(team, attr)
                    if not callable(val):
                        print(f"    {attr}: {val}")
                except:
                    pass

    print("\n" + "="*80)
    print("8. ATTEMPT TO ACCESS RAW ESPN DATA")
    print("="*80)

    # The espn_api package may store raw response data
    for attr in ['espn_request', '_fetch', 'cookies', 'endpoint']:
        if hasattr(league, attr):
            print(f"  Found: {attr}")

    # Try to find the actual stat categories used for scoring
    print("\n  Attempting to identify SCORED vs TRACKED stats...")

    # Common ESPN stat IDs for fantasy basketball:
    KNOWN_STAT_IDS = {
        0: 'PTS',
        1: 'BLK',
        2: 'STL',
        3: 'AST',
        6: 'REB',
        11: 'TO',
        13: 'FGM',
        14: 'FGA',
        15: 'FTM',
        16: 'FTA',
        17: '3PM',
        19: 'FG%',
        20: 'FT%',
        # There are more but these are common
    }

    print("\n  Known ESPN Stat ID mappings:")
    for stat_id, name in sorted(KNOWN_STAT_IDS.items()):
        print(f"    {stat_id}: {name}")

    # If we have stats, try to correlate
    if league.teams and hasattr(league.teams[0], 'stats'):
        team_stats = league.teams[0].stats
        print(f"\n  Team stats keys: {list(team_stats.keys())}")

        # Check if keys are stat IDs or names
        sample_keys = list(team_stats.keys())[:5]
        print(f"  Sample values:")
        for key in sample_keys:
            print(f"    {key}: {team_stats[key]}")

    print("\n" + "="*80)
    print("9. SETTINGS OBJECT - RAW DUMP")
    print("="*80)

    # Try to get __dict__ of settings
    if hasattr(settings, '__dict__'):
        print("Settings.__dict__:")
        pprint(settings.__dict__, width=120)

    print("\n" + "="*80)
    print("10. DIGGING INTO ESPN_REQUEST FOR RAW DATA")
    print("="*80)

    # Try to access the raw ESPN API response
    if hasattr(league, 'espn_request'):
        espn_req = league.espn_request
        print("espn_request attributes:")
        for attr in dir(espn_req):
            if not attr.startswith('_') and not callable(getattr(espn_req, attr, None)):
                try:
                    val = getattr(espn_req, attr)
                    print(f"  {attr}: {val}")
                except:
                    pass

        # Try to make a direct API call for league settings
        print("\n  Attempting to fetch raw league settings...")
        try:
            # The ESPN API endpoint for settings
            import requests

            league_endpoint = f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/fba/seasons/{year}/segments/0/leagues/{league_id}"
            params = {
                'view': ['mSettings', 'mTeam', 'mRoster', 'mMatchup', 'mStandings']
            }
            cookies = {
                'espn_s2': espn_s2,
                'SWID': swid
            }

            response = requests.get(league_endpoint, params=params, cookies=cookies)
            if response.status_code == 200:
                raw_data = response.json()
                print(f"\n  Raw API response keys: {list(raw_data.keys())}")

                # Check settings
                if 'settings' in raw_data:
                    raw_settings = raw_data['settings']
                    print(f"\n  Raw settings keys: {list(raw_settings.keys())}")

                    # Look for scoring settings
                    if 'scoringSettings' in raw_settings:
                        scoring_settings = raw_settings['scoringSettings']
                        print(f"\n  scoringSettings keys: {list(scoring_settings.keys())}")
                        pprint(scoring_settings, width=120)

                    # Check for stat modifiers or categories
                    for key in raw_settings.keys():
                        if 'stat' in key.lower() or 'scor' in key.lower() or 'cat' in key.lower():
                            print(f"\n  >>> {key}:")
                            pprint(raw_settings[key], width=120)

                # Check if there's a 'status' with category info
                if 'status' in raw_data:
                    print(f"\n  Status keys: {list(raw_data['status'].keys())}")

            else:
                print(f"  API request failed: {response.status_code}")

        except Exception as e:
            print(f"  Error fetching raw data: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*80)
    print("11. CHECK STAT CATEGORIES IN RAW SCORING SETTINGS")
    print("="*80)

    # ESPN stat IDs map - commonly used in fantasy basketball
    ESPN_STAT_MAP = {
        0: 'PTS',
        1: 'BLK',
        2: 'STL',
        3: 'AST',
        4: 'OREB',
        5: 'DREB',
        6: 'REB',
        9: 'PF',
        10: 'TF',  # Technical Fouls
        11: 'TO',
        12: 'EJ',  # Ejections
        13: 'FGM',
        14: 'FGA',
        15: 'FTM',
        16: 'FTA',
        17: '3PM',
        18: '3PA',
        19: 'FG%',
        20: 'FT%',
        21: '3P%',
        22: 'DD',   # Double-Double
        23: 'TD',   # Triple-Double
        24: 'QD',   # Quadruple-Double
        28: 'MPG',
        29: 'MIN',
        40: 'GS',   # Games Started
        41: 'GP',
    }

    print("Full ESPN Stat ID Map:")
    for stat_id, name in sorted(ESPN_STAT_MAP.items()):
        print(f"  {stat_id}: {name}")

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"League: {league.settings.name}")
    print(f"Type: {getattr(settings, 'scoring_type', 'Unknown')}")
    print(f"Teams: {len(league.teams)}")
    if league.teams and hasattr(league.teams[0], 'stats'):
        stats_keys = list(league.teams[0].stats.keys())
        print(f"Stats in team.stats: {stats_keys}")

        # Identify likely scoring categories
        excluded = {'GP', 'FGA', 'FTA', 'MIN', 'MPG', 'GS'}
        scoring_cats = [k for k in stats_keys if k not in excluded]
        print(f"\nLikely SCORING categories: {scoring_cats}")
        print(f"Non-scoring (helper) stats: {[k for k in stats_keys if k in excluded]}")


if __name__ == '__main__':
    main()

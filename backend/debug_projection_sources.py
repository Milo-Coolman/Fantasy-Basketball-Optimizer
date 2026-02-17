#!/usr/bin/env python3
"""
Debug script to find where ESPN projections and previous season data actually exist.

This script examines player stats structure to understand:
1. What keys exist in player.stats
2. Where ESPN projections are stored (e.g., '2026_projected')
3. Where previous season data is stored (e.g., '2025_total')
4. The exact format needed for extraction

Run from project root:
    python backend/debug_projection_sources.py
"""

import sys
import os

# Add project root to path for imports
sys.path.insert(0, '/Users/milo/fantasy-basketball-optimizer')

from dotenv import load_dotenv
load_dotenv()

from backend.models import League
from backend.app import create_app

# Target players to examine
TARGET_PLAYERS = [
    "Trae Young",       # Likely Tier 2 (6-15 games)
    "Jayson Tatum",     # Should have good data
]


def get_league_credentials():
    """Get ESPN credentials from .env file and league info from database."""
    # Get ESPN credentials from .env
    espn_s2 = os.getenv('ESPN_S2')
    swid = os.getenv('SWID')

    if not espn_s2 or not swid:
        print("ERROR: Missing ESPN credentials in .env file!")
        print("Please add ESPN_S2 and SWID to your .env file.")
        sys.exit(1)

    # Get league_id and season from database
    app = create_app()
    with app.app_context():
        league = League.query.first()
        if not league:
            print("ERROR: No league found in database!")
            print("Please add a league via the web interface first.")
            sys.exit(1)

        print(f"League from database:")
        print(f"  League ID: {league.espn_league_id}")
        print(f"  Season: {league.season}")
        print(f"  Name: {league.league_name}")
        print(f"ESPN credentials from .env:")
        print(f"  ESPN_S2: {espn_s2[:20]}...")
        print(f"  SWID: {swid}")

        return {
            'league_id': league.espn_league_id,
            'season': league.season,
            'espn_s2': espn_s2,
            'swid': swid,
        }


def main():
    print("=" * 80)
    print("DEBUG: Finding ESPN Projection and Previous Season Data Sources")
    print("=" * 80)

    # Get credentials from database
    creds = get_league_credentials()
    LEAGUE_ID = creds['league_id']
    SEASON = creds['season']
    ESPN_S2 = creds['espn_s2']
    SWID = creds['swid']

    # Try using espn-api package first
    try:
        from espn_api.basketball import League
        print(f"\nConnecting to ESPN League {LEAGUE_ID}, Season {SEASON}...")
        league = League(league_id=int(LEAGUE_ID), year=SEASON, espn_s2=ESPN_S2, swid=SWID)
        print(f"Connected! League: {league.settings.name}")

        # Get all players from rosters
        all_players = []
        for team in league.teams:
            all_players.extend(team.roster)

        print(f"\nTotal players in league rosters: {len(all_players)}")

        # Find target players
        found_players = []
        for player in all_players:
            if any(target.lower() in player.name.lower() for target in TARGET_PLAYERS):
                found_players.append(player)

        if not found_players:
            print("\nTarget players not found in rosters. Using first 3 players instead.")
            found_players = all_players[:3]

        # Examine each player
        for player in found_players:
            print("\n" + "=" * 80)
            print(f"PLAYER: {player.name}")
            print("=" * 80)

            # Basic info
            print(f"\n--- Basic Info ---")
            print(f"  Player ID: {player.playerId}")
            print(f"  Position: {player.position}")
            print(f"  Pro Team: {player.proTeam}")
            print(f"  Injury Status: {player.injuryStatus}")

            # Check for stats attribute
            if hasattr(player, 'stats'):
                stats = player.stats
                print(f"\n--- player.stats Keys ---")
                print(f"  All keys: {list(stats.keys())}")

                # Specifically check for projection and previous season keys
                projection_key = f"{SEASON}_projected"  # e.g., '2026_projected'
                prev_season_key = f"{SEASON - 1}_total"  # e.g., '2025_total'
                current_season_key = f"{SEASON}_total"   # e.g., '2026_total'

                print(f"\n--- Looking for key: '{projection_key}' (ESPN Projections) ---")
                if projection_key in stats:
                    proj_data = stats[projection_key]
                    print(f"  FOUND! Type: {type(proj_data)}")
                    if isinstance(proj_data, dict):
                        print(f"  Sub-keys: {list(proj_data.keys())}")
                        for sub_key in ['avg', 'average', 'total']:
                            if sub_key in proj_data:
                                print(f"  stats['{projection_key}']['{sub_key}']: {proj_data[sub_key]}")
                else:
                    print(f"  NOT FOUND")

                print(f"\n--- Looking for key: '{prev_season_key}' (Previous Season) ---")
                if prev_season_key in stats:
                    prev_data = stats[prev_season_key]
                    print(f"  FOUND! Type: {type(prev_data)}")
                    if isinstance(prev_data, dict):
                        print(f"  Sub-keys: {list(prev_data.keys())}")
                        for sub_key in ['avg', 'average', 'total']:
                            if sub_key in prev_data:
                                print(f"  stats['{prev_season_key}']['{sub_key}']: {prev_data[sub_key]}")
                else:
                    print(f"  NOT FOUND")

                print(f"\n--- Looking for key: '{current_season_key}' (Current Season) ---")
                if current_season_key in stats:
                    curr_data = stats[current_season_key]
                    print(f"  FOUND! Type: {type(curr_data)}")
                    if isinstance(curr_data, dict):
                        print(f"  Sub-keys: {list(curr_data.keys())}")
                        for sub_key in ['avg', 'average', 'total']:
                            if sub_key in curr_data:
                                print(f"  stats['{current_season_key}']['{sub_key}']: {curr_data[sub_key]}")
                else:
                    print(f"  NOT FOUND")

                # Show ALL keys with their structure
                print(f"\n--- Full stats structure ---")
                for key in sorted(stats.keys()):
                    value = stats[key]
                    if isinstance(value, dict):
                        print(f"  stats['{key}']: dict with keys {list(value.keys())}")
                    else:
                        print(f"  stats['{key}']: {type(value).__name__} = {value}")
            else:
                print("\n  NO 'stats' attribute found!")

            # Check for other stat-related attributes
            print(f"\n--- Other Stat Attributes ---")
            for attr in ['total_points', 'avg_points', 'projected_total_points',
                        'projected_avg_points', 'stats_2025', 'stats_2026']:
                if hasattr(player, attr):
                    print(f"  {attr}: {getattr(player, attr)}")

    except ImportError as e:
        print(f"\nespn-api not available: {e}")
        print("Trying direct API call...")
        examine_via_direct_api(LEAGUE_ID, SEASON, ESPN_S2, SWID)
    except Exception as e:
        print(f"\nError with espn-api: {e}")
        import traceback
        traceback.print_exc()
        print("\nTrying direct API call...")
        examine_via_direct_api(LEAGUE_ID, SEASON, ESPN_S2, SWID)


def examine_via_direct_api(LEAGUE_ID, SEASON, ESPN_S2, SWID):
    """Examine player stats via direct ESPN API call."""
    import requests

    print("\n" + "=" * 80)
    print("DIRECT ESPN API EXAMINATION")
    print("=" * 80)

    endpoint = f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/fba/seasons/{SEASON}/segments/0/leagues/{LEAGUE_ID}"
    params = {'view': ['mTeam', 'mRoster', 'mMatchup', 'kona_player_info']}
    cookies = {'espn_s2': ESPN_S2, 'SWID': SWID}

    print(f"\nFetching from: {endpoint}")

    try:
        response = requests.get(endpoint, params=params, cookies=cookies, timeout=30)
        response.raise_for_status()
        data = response.json()

        print(f"Response received. Top-level keys: {list(data.keys())}")

        # Find target players
        found = False
        for team in data.get('teams', []):
            for entry in team.get('roster', {}).get('entries', []):
                player = entry.get('playerPoolEntry', {}).get('player', {})
                player_name = player.get('fullName', '')

                if any(target.lower() in player_name.lower() for target in TARGET_PLAYERS):
                    found = True
                    print("\n" + "=" * 80)
                    print(f"PLAYER: {player_name}")
                    print("=" * 80)

                    print(f"\n--- Player Object Keys ---")
                    print(f"  {list(player.keys())}")

                    # Examine stats array
                    stats_array = player.get('stats', [])
                    print(f"\n--- player['stats'] (array of {len(stats_array)} items) ---")

                    for i, stat_set in enumerate(stats_array):
                        stat_source_id = stat_set.get('statSourceId', 'N/A')
                        stat_split_type = stat_set.get('statSplitTypeId', 'N/A')
                        season_id = stat_set.get('seasonId', 'N/A')

                        source_name = {0: 'ACTUAL', 1: 'PROJECTED'}.get(stat_source_id, f'SOURCE_{stat_source_id}')
                        split_name = {0: 'TOTAL', 1: 'LAST_7', 2: 'LAST_15', 3: 'LAST_30'}.get(stat_split_type, f'SPLIT_{stat_split_type}')

                        print(f"\n  [{i}] Season={season_id}, Source={source_name}, Split={split_name}")
                        print(f"      statSourceId={stat_source_id}, statSplitTypeId={stat_split_type}")
                        print(f"      Keys: {list(stat_set.keys())}")

                        # Show stats content
                        if 'stats' in stat_set:
                            raw_stats = stat_set['stats']
                            print(f"      stats keys (ESPN IDs): {list(raw_stats.keys())[:10]}...")
                            # Map some key IDs
                            ESPN_STAT_MAP = {'0': 'PTS', '3': 'AST', '6': 'REB', '2': 'STL', '1': 'BLK', '17': '3PM'}
                            sample = {}
                            for stat_id, name in ESPN_STAT_MAP.items():
                                if stat_id in raw_stats:
                                    sample[name] = raw_stats[stat_id]
                            print(f"      Sample totals: {sample}")

                        if 'averageStats' in stat_set:
                            avg_stats = stat_set['averageStats']
                            ESPN_STAT_MAP = {'0': 'PTS', '3': 'AST', '6': 'REB', '2': 'STL', '1': 'BLK', '17': '3PM'}
                            sample = {}
                            for stat_id, name in ESPN_STAT_MAP.items():
                                if stat_id in avg_stats:
                                    sample[name] = round(avg_stats[stat_id], 1)
                            print(f"      Sample averages: {sample}")

                    # Show recommended extraction code
                    print(f"\n--- RECOMMENDED EXTRACTION CODE ---")
                    print("""
    # For this player's stats array:
    for stat_set in player.get('stats', []):
        stat_source_id = stat_set.get('statSourceId', 0)
        season_id = stat_set.get('seasonId', 0)
        stat_split_type = stat_set.get('statSplitTypeId', 0)

        # Current season actual stats (season total)
        if stat_source_id == 0 and season_id == {season} and stat_split_type == 0:
            current_season_total = stat_set.get('stats', {{}})
            current_season_avg = stat_set.get('averageStats', {{}})

        # Current season ESPN projections
        if stat_source_id == 1 and season_id == {season}:
            espn_projection_total = stat_set.get('stats', {{}})
            # Note: projections may not have averageStats, calculate from total/82

        # Previous season actual stats
        if stat_source_id == 0 and season_id == {prev_season} and stat_split_type == 0:
            prev_season_total = stat_set.get('stats', {{}})
            prev_season_avg = stat_set.get('averageStats', {{}})
    """.format(season=SEASON, prev_season=SEASON-1))

        if not found:
            print("\nTarget players not found. Examining first player found...")
            for team in data.get('teams', []):
                for entry in team.get('roster', {}).get('entries', [])[:1]:
                    player = entry.get('playerPoolEntry', {}).get('player', {})
                    print(f"\nFirst player: {player.get('fullName', 'Unknown')}")
                    print(f"Stats array length: {len(player.get('stats', []))}")
                    for i, stat_set in enumerate(player.get('stats', [])[:5]):
                        print(f"  [{i}] seasonId={stat_set.get('seasonId')}, sourceId={stat_set.get('statSourceId')}, splitType={stat_set.get('statSplitTypeId')}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Debug script to find where ESPN projections and previous season data actually exist.
Thoroughly searches for 2025 (previous season) data across all possible locations.

Uses DATABASE credentials for ESPN connection.
"""

import os
import sys

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from espn_api.basketball import League as ESPNLeague
from backend.app import create_app
from backend.models import League


def get_league_credentials():
    """Get all credentials from database."""
    app = create_app()
    with app.app_context():
        league = League.query.first()
        if not league:
            print('ERROR: No league found in database!')
            sys.exit(1)

        return {
            'league_id': league.espn_league_id,
            'season': league.season,
            'espn_s2': league.espn_s2_cookie,
            'swid': league.swid_cookie,
            'name': league.league_name
        }


def search_for_2025_data(player):
    """
    Thoroughly search for 2025/previous season data in all possible locations.
    """
    print(f"\n{'='*80}")
    print(f"SEARCHING FOR 2025 DATA: {player.name}")
    print(f"{'='*80}")

    if not hasattr(player, 'stats'):
        print("  ERROR: Player has no 'stats' attribute!")
        return

    stats_keys = list(player.stats.keys())
    print(f"\n  All stats keys ({len(stats_keys)}): {stats_keys}")

    # ==========================================================================
    # 1) Check ALL keys for 2025 variations
    # ==========================================================================
    print(f"\n--- Checking for 2025 variations ---")

    variations_2025 = ['2025', '2025_total', '2025_projected', '002025',
                       'last_year', 'previous_season', 'prev', 'lastYear',
                       'previousSeason', 'career', 'historical']

    for var in variations_2025:
        if var in player.stats:
            print(f"  FOUND: '{var}'")
            stat_data = player.stats[var]
            if isinstance(stat_data, dict):
                print(f"    Sub-keys: {list(stat_data.keys())}")
                if 'avg' in stat_data:
                    print(f"    avg: {stat_data['avg']}")
                if 'total' in stat_data:
                    print(f"    total: {stat_data['total']}")

    # Check for ANY key containing '2025'
    print(f"\n--- Keys containing '2025' ---")
    found_2025 = False
    for key in stats_keys:
        if '2025' in str(key):
            found_2025 = True
            print(f"  '{key}': {player.stats[key]}")

    if not found_2025:
        print("  No keys containing '2025' found in player.stats")

    # ==========================================================================
    # 2) Show all player attributes
    # ==========================================================================
    print(f"\n--- All player attributes ---")
    player_attrs = [a for a in dir(player) if not a.startswith('_')]
    print(f"  {player_attrs}")

    # ==========================================================================
    # 3) Dump player.__dict__
    # ==========================================================================
    print(f"\n--- player.__dict__ ---")
    if hasattr(player, '__dict__'):
        for key, val in player.__dict__.items():
            if not key.startswith('_'):
                val_str = str(val)
                if len(val_str) > 150:
                    val_str = val_str[:150] + "..."
                print(f"  {key}: {val_str}")

    # ==========================================================================
    # 4) Deep inspect each stats key
    # ==========================================================================
    print(f"\n--- Deep inspection of player.stats ---")
    for key in stats_keys:
        stat_data = player.stats[key]
        print(f"\n  player.stats['{key}']:")

        if isinstance(stat_data, dict):
            print(f"    Sub-keys: {list(stat_data.keys())}")
            if 'avg' in stat_data:
                avg = stat_data['avg']
                if isinstance(avg, dict):
                    # Show key stats
                    pts = avg.get('PTS', 'N/A')
                    reb = avg.get('REB', 'N/A')
                    ast = avg.get('AST', 'N/A')
                    print(f"    avg -> PTS: {pts}, REB: {reb}, AST: {ast}")
            if 'total' in stat_data:
                total = stat_data['total']
                if isinstance(total, dict):
                    gp = total.get('GP', 'N/A')
                    print(f"    total -> GP: {gp}")
        else:
            print(f"    Value: {stat_data}")


def try_2025_season_connection(creds):
    """
    Try to connect to 2025 season to get previous year stats.
    """
    print(f"\n{'='*80}")
    print(f"CONNECTING TO 2025 SEASON DIRECTLY")
    print(f"{'='*80}")

    try:
        league_2025 = ESPNLeague(
            league_id=creds['league_id'],
            year=2025,
            espn_s2=creds['espn_s2'],
            swid=creds['swid']
        )
        print(f"  SUCCESS! Connected to 2025 season: {league_2025.settings.name}")
        return league_2025

    except Exception as e:
        print(f"  ERROR connecting to 2025: {e}")
        print("  (This may be expected if the league didn't exist in 2025)")
        return None


def find_player_by_name(all_players, name):
    """Find a player by name (partial match)."""
    name_lower = name.lower()
    for player in all_players:
        if name_lower in player.name.lower():
            return player
    return None


def main():
    print("=" * 80)
    print("DEBUG PROJECTION SOURCES - 2025 DATA SEARCH")
    print("Using DATABASE credentials for ESPN connection")
    print("=" * 80)

    # ==========================================================================
    # Get credentials from DATABASE
    # ==========================================================================
    creds = get_league_credentials()

    print(f"\n  League: {creds['name']}")
    print(f"  League ID: {creds['league_id']}")
    print(f"  Season: {creds['season']}")
    print(f"  ESPN_S2: {creds['espn_s2'][:30]}..." if creds['espn_s2'] else "  ESPN_S2: None")
    print(f"  SWID: {creds['swid']}")

    # ==========================================================================
    # Connect to current season
    # ==========================================================================
    print(f"\n{'='*80}")
    print(f"CONNECTING TO {creds['season']} SEASON")
    print(f"{'='*80}")

    try:
        league = ESPNLeague(
            league_id=creds['league_id'],
            year=creds['season'],
            espn_s2=creds['espn_s2'],
            swid=creds['swid']
        )
        print(f"  Connected to: {league.settings.name}")
    except Exception as e:
        print(f"  ERROR: {e}")
        sys.exit(1)

    # Gather all players
    all_players = []
    for team in league.teams:
        all_players.extend(team.roster)
    print(f"  Found {len(all_players)} rostered players")

    # ==========================================================================
    # Search for Jayson Tatum's data (Tier 1 - 0 games)
    # ==========================================================================
    print(f"\n{'='*80}")
    print("TIER 1 PLAYER: Jayson Tatum")
    print("=" * 80)

    tatum = find_player_by_name(all_players, "Jayson Tatum")
    if tatum:
        search_for_2025_data(tatum)
    else:
        print("Jayson Tatum not found in rosters!")
        if all_players:
            print(f"Using first player instead: {all_players[0].name}")
            search_for_2025_data(all_players[0])

    # ==========================================================================
    # Search for Trae Young's data (Tier 2 - ~10 games)
    # ==========================================================================
    print(f"\n{'='*80}")
    print("TIER 2 PLAYER: Trae Young")
    print("=" * 80)

    trae = find_player_by_name(all_players, "Trae Young")
    if trae:
        search_for_2025_data(trae)
    else:
        print("Trae Young not found in rosters!")

    # ==========================================================================
    # Try connecting to 2025 season directly
    # ==========================================================================
    league_2025 = try_2025_season_connection(creds)

    if league_2025:
        print(f"\n{'='*80}")
        print("PLAYER DATA FROM 2025 SEASON")
        print("=" * 80)

        # Get all players from 2025
        players_2025 = []
        for team in league_2025.teams:
            players_2025.extend(team.roster)
        print(f"  Found {len(players_2025)} rostered players in 2025")

        # Find Tatum in 2025
        tatum_2025 = find_player_by_name(players_2025, "Jayson Tatum")
        if tatum_2025:
            print(f"\n  Jayson Tatum in 2025:")
            print(f"    Stats keys: {list(tatum_2025.stats.keys())}")

            for key in tatum_2025.stats.keys():
                if '2025' in str(key):
                    print(f"\n    player.stats['{key}']:")
                    stat_data = tatum_2025.stats[key]
                    if isinstance(stat_data, dict):
                        if 'avg' in stat_data:
                            print(f"      avg: {stat_data['avg']}")
                        if 'total' in stat_data:
                            print(f"      total: {stat_data['total']}")

        # Find Trae in 2025
        trae_2025 = find_player_by_name(players_2025, "Trae Young")
        if trae_2025:
            print(f"\n  Trae Young in 2025:")
            print(f"    Stats keys: {list(trae_2025.stats.keys())}")

            for key in trae_2025.stats.keys():
                if '2025' in str(key):
                    print(f"\n    player.stats['{key}']:")
                    stat_data = trae_2025.stats[key]
                    if isinstance(stat_data, dict):
                        if 'avg' in stat_data:
                            print(f"      avg: {stat_data['avg']}")

    # ==========================================================================
    # Summary
    # ==========================================================================
    print(f"\n{'='*80}")
    print("SUMMARY")
    print("=" * 80)
    print("""
KEY FINDINGS:

1. player.stats keys for 2026 season typically include:
   - '2026_total' -> current season actuals (avg + total)
   - '2026_projected' -> ESPN projections (avg + total)

2. The espn-api library does NOT include previous season stats automatically.
   - When connected to 2026, you only get 2026 data
   - When connected to 2025, you only get 2025 data

3. TO GET 2025 (PREVIOUS SEASON) DATA:

   OPTION A: Connect to 2025 season separately
   ```python
   league_2025 = ESPNLeague(league_id=ID, year=2025, espn_s2=S2, swid=SWID)
   # Match players by playerId between seasons
   ```

   OPTION B: Cache 2025 stats in database at end of season

   OPTION C: Scrape Basketball Reference for historical stats

4. EXTRACTION CODE for current season:

   # ESPN projections
   proj = player.stats.get('2026_projected', {})
   proj_avg = proj.get('avg', {})
   pts = proj_avg.get('PTS', 0)

   # Current season actuals
   curr = player.stats.get('2026_total', {})
   curr_avg = curr.get('avg', {})
   games = curr.get('total', {}).get('GP', 0)
""")


if __name__ == "__main__":
    main()

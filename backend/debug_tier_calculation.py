#!/usr/bin/env python3
"""
Debug script to investigate RAW ESPN player stats structure for games_played.

FIXED: Uses STRING KEYS like 'PTS', 'REB', 'GP' not numeric IDs.
"""

import json
import logging
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def print_subsection(title: str):
    """Print a subsection header."""
    print("\n" + "-" * 60)
    print(f" {title}")
    print("-" * 60)


def safe_get(obj, attr, default=None):
    """Safely get an attribute from an object."""
    try:
        return getattr(obj, attr, default)
    except Exception:
        return default


def extract_games_played_v2(raw_stats: dict, season: int) -> tuple:
    """
    Extract games_played from raw ESPN player stats using STRING KEYS.

    Returns:
        Tuple of (games_played, method_used)
    """
    # Key format: '{season}_total'
    season_key = f'{season}_total'
    season_total = raw_stats.get(season_key, {})

    if not season_total or not isinstance(season_total, dict):
        return 0, "no_season_total"

    # Get sub-dicts
    total_dict = season_total.get('total', {})
    avg_dict = season_total.get('avg', {})

    # GP key variants to try (STRING KEYS)
    GP_KEYS = ['GP', 'G', 'gamesPlayed', 'games_played', 'GamesPlayed', 'GAMES_PLAYED']

    # ==========================================================================
    # Method 1 (FASTEST): Direct GP in 'total' dict
    # ==========================================================================
    if total_dict and isinstance(total_dict, dict):
        for gp_key in GP_KEYS:
            gp = total_dict.get(gp_key, 0)
            if gp and float(gp) > 0:
                return int(float(gp)), f"total['{gp_key}']"

    # ==========================================================================
    # Method 2: Direct GP in 'avg' dict
    # ==========================================================================
    if avg_dict and isinstance(avg_dict, dict):
        for gp_key in GP_KEYS:
            gp = avg_dict.get(gp_key, 0)
            if gp and float(gp) > 0:
                return int(float(gp)), f"avg['{gp_key}']"

    # ==========================================================================
    # Method 3: Direct GP in season_total itself (top level)
    # ==========================================================================
    for gp_key in GP_KEYS + ['applied_total', 'appliedTotal']:
        gp = season_total.get(gp_key, 0)
        if gp and float(gp) > 0:
            return int(float(gp)), f"season_total['{gp_key}']"

    # ==========================================================================
    # Method 4 (FALLBACK): Calculate from total['PTS'] / avg['PTS']
    # ==========================================================================
    if avg_dict and total_dict:
        # Try string keys for PTS
        pts_keys = ['PTS', 'pts', 'Points', 'points', 0, '0']

        pts_avg = 0
        pts_total = 0

        for pts_key in pts_keys:
            if pts_key in avg_dict and avg_dict[pts_key]:
                pts_avg = float(avg_dict[pts_key])
                break

        for pts_key in pts_keys:
            if pts_key in total_dict and total_dict[pts_key]:
                pts_total = float(total_dict[pts_key])
                break

        if pts_avg > 0 and pts_total > 0:
            calculated_gp = int(pts_total / pts_avg)
            return calculated_gp, f"calculated: {pts_total:.1f} / {pts_avg:.3f}"

    return 0, "not_found"


def main():
    print_section("RAW ESPN PLAYER STATS STRUCTURE DEBUG (FIXED)")
    print("Using STRING KEYS like 'PTS', 'REB', 'GP' (not numeric IDs)")

    # Initialize Flask app context
    from backend.app import create_app
    app = create_app()

    with app.app_context():
        # Get league credentials
        from backend.models import League
        league = League.query.first()
        if not league:
            print("ERROR: No league found!")
            return 1

        print(f"League: {league.league_name}")
        print(f"Season: {league.season}")
        season = league.season

        # Connect to ESPN using espn-api directly
        print_section("STEP 1: ACCESS RAW ESPN-API LEAGUE")

        from espn_api.basketball import League as ESPNLeague

        raw_league = ESPNLeague(
            league_id=int(league.espn_league_id),
            year=league.season,
            espn_s2=league.espn_s2_cookie,
            swid=league.swid_cookie
        )
        print(f"Connected to: {raw_league.settings.name}")
        print(f"Season: {raw_league.year}")

        # Find Luka and Kyrie in raw rosters
        print_section("STEP 2: FIND RAW PLAYER OBJECTS")

        raw_luka = None
        raw_kyrie = None

        for team in raw_league.teams:
            for player in team.roster:
                name_lower = player.name.lower()
                if 'luka' in name_lower or 'doncic' in name_lower:
                    raw_luka = player
                    print(f"Found Luka: {player.name} on {team.team_name}")
                if 'kyrie' in name_lower or 'irving' in name_lower:
                    raw_kyrie = player
                    print(f"Found Kyrie: {player.name} on {team.team_name}")

        if not raw_luka:
            print("WARNING: Luka not found, using first player as fallback")
            for team in raw_league.teams:
                if team.roster:
                    raw_luka = team.roster[0]
                    print(f"Using fallback: {raw_luka.name}")
                    break

        # Analyze each player
        for player_name, raw_player in [("LUKA DONCIC", raw_luka), ("KYRIE IRVING", raw_kyrie)]:
            if not raw_player:
                print(f"\n{player_name}: Not found, skipping")
                continue

            print_section(f"ANALYZING: {player_name} ({raw_player.name})")

            raw_stats = safe_get(raw_player, 'stats', {})

            if not raw_stats:
                print("  raw_player.stats is empty!")
                continue

            # Show all stat period keys
            print_subsection("Available Stat Periods")
            print(f"  Keys: {list(raw_stats.keys())}")

            # Deep dive into {season}_total
            season_key = f'{season}_total'
            print_subsection(f"Structure of '{season_key}'")

            season_total = raw_stats.get(season_key, {})
            if not season_total:
                print(f"  '{season_key}' NOT FOUND!")
                print(f"  Available keys: {list(raw_stats.keys())}")
                continue

            print(f"  Top-level keys: {list(season_total.keys())}")

            # Show 'total' sub-dict
            total_dict = season_total.get('total', {})
            if total_dict:
                print(f"\n  'total' dict ({len(total_dict)} keys):")
                for k, v in sorted(total_dict.items(), key=lambda x: str(x[0])):
                    print(f"    {repr(k)}: {v}")
            else:
                print("\n  'total' dict: EMPTY or NOT FOUND")

            # Show 'avg' sub-dict
            avg_dict = season_total.get('avg', {})
            if avg_dict:
                print(f"\n  'avg' dict ({len(avg_dict)} keys):")
                for k, v in sorted(avg_dict.items(), key=lambda x: str(x[0])):
                    print(f"    {repr(k)}: {v}")
            else:
                print("\n  'avg' dict: EMPTY or NOT FOUND")

            # Search for GP specifically
            print_subsection("Searching for Games Played (GP)")

            GP_KEYS = ['GP', 'G', 'gamesPlayed', 'games_played', 'GamesPlayed',
                       'applied_total', 'appliedTotal']

            print("  Checking season_total directly:")
            for gp_key in GP_KEYS:
                if gp_key in season_total:
                    print(f"    season_total['{gp_key}'] = {season_total[gp_key]} <-- FOUND!")

            print("\n  Checking total dict:")
            for gp_key in GP_KEYS:
                if gp_key in total_dict:
                    print(f"    total['{gp_key}'] = {total_dict[gp_key]} <-- FOUND!")

            print("\n  Checking avg dict:")
            for gp_key in GP_KEYS:
                if gp_key in avg_dict:
                    print(f"    avg['{gp_key}'] = {avg_dict[gp_key]} <-- FOUND!")

            # Try calculation method
            print_subsection("Calculate GP from PTS")

            pts_avg = None
            pts_total = None

            for pts_key in ['PTS', 'pts', 0, '0']:
                if pts_key in avg_dict and avg_dict[pts_key]:
                    pts_avg = float(avg_dict[pts_key])
                    print(f"  avg[{repr(pts_key)}] = {pts_avg}")
                    break

            for pts_key in ['PTS', 'pts', 0, '0']:
                if pts_key in total_dict and total_dict[pts_key]:
                    pts_total = float(total_dict[pts_key])
                    print(f"  total[{repr(pts_key)}] = {pts_total}")
                    break

            if pts_avg and pts_avg > 0 and pts_total:
                calc_gp = pts_total / pts_avg
                print(f"\n  Calculated GP = {pts_total} / {pts_avg} = {calc_gp:.1f}")
            else:
                print("\n  Cannot calculate (missing PTS data)")

            # Use the fixed extraction function
            print_subsection("EXTRACTION RESULT (Using Fixed Function)")

            games_played, method = extract_games_played_v2(raw_stats, season)

            print(f"  Games Played: {games_played}")
            print(f"  Method Used: {method}")

            # Tier determination
            print_subsection("TIER DETERMINATION")

            GAMES_TIER_1_MAX = 5
            GAMES_TIER_2_MAX = 15
            GAMES_TIER_3_MAX = 35

            if games_played <= GAMES_TIER_1_MAX:
                tier = 1
                weights = "60% ESPN, 30% prev, 0% curr, 10% ML"
            elif games_played <= GAMES_TIER_2_MAX:
                tier = 2
                weights = "35% ESPN, 20% prev, 35% curr, 10% ML"
            elif games_played <= GAMES_TIER_3_MAX:
                tier = 3
                weights = "15% ESPN, 10% prev, 70% curr, 5% ML"
            else:
                tier = 4
                weights = "0% ESPN, 0% prev, 100% curr, 0% ML"

            print(f"  Games Played: {games_played}")
            print(f"  Tier: {tier}")
            print(f"  Weights: {weights}")

            # Validation
            if "LUKA" in player_name.upper():
                if games_played >= 35:
                    print(f"\n  *** SUCCESS: Luka has {games_played} games -> Tier 4 (100% current stats) ***")
                else:
                    print(f"\n  *** WARNING: Expected 40+ games for Luka, got {games_played} ***")

            if "KYRIE" in player_name.upper():
                if games_played == 0:
                    print(f"\n  *** SUCCESS: Kyrie has 0 games -> Tier 1 (60% ESPN projections) ***")
                else:
                    print(f"\n  *** INFO: Kyrie has {games_played} games ***")

        # Print the working extraction code
        print_section("WORKING EXTRACTION CODE FOR BACKEND")

        print('''
# Add this to backend/services/espn_client.py or backend/api/dashboard.py:

def extract_games_played(raw_stats: dict, season: int) -> int:
    """
    Extract games_played from raw ESPN player stats using STRING KEYS.

    Priority order:
    1. Direct GP in 'total' dict
    2. Direct GP in 'avg' dict
    3. Direct GP in season_total top-level
    4. Calculate from total['PTS'] / avg['PTS']

    Args:
        raw_stats: Player's stats dict (player.stats)
        season: Season year (e.g., 2026)

    Returns:
        Games played count (0 if not found)
    """
    season_key = f'{season}_total'
    season_total = raw_stats.get(season_key, {})

    if not season_total or not isinstance(season_total, dict):
        return 0

    total_dict = season_total.get('total', {})
    avg_dict = season_total.get('avg', {})

    # GP key variants (STRING KEYS)
    GP_KEYS = ['GP', 'G', 'gamesPlayed', 'games_played']

    # Method 1: Direct GP in 'total' dict
    if total_dict:
        for gp_key in GP_KEYS:
            gp = total_dict.get(gp_key, 0)
            if gp and float(gp) > 0:
                return int(float(gp))

    # Method 2: Direct GP in 'avg' dict
    if avg_dict:
        for gp_key in GP_KEYS:
            gp = avg_dict.get(gp_key, 0)
            if gp and float(gp) > 0:
                return int(float(gp))

    # Method 3: Direct GP in season_total top-level
    for gp_key in GP_KEYS + ['applied_total', 'appliedTotal']:
        gp = season_total.get(gp_key, 0)
        if gp and float(gp) > 0:
            return int(float(gp))

    # Method 4: Calculate from total['PTS'] / avg['PTS']
    if avg_dict and total_dict:
        for pts_key in ['PTS', 'pts', 0, '0']:
            pts_avg = avg_dict.get(pts_key, 0)
            pts_total = total_dict.get(pts_key, 0)
            if pts_avg and float(pts_avg) > 0 and pts_total:
                return int(float(pts_total) / float(pts_avg))

    return 0


# Usage in _parse_player():
#   games_played = extract_games_played(player.stats, self.year)
#   player_data['games_played'] = games_played
''')

        print("\n" + "=" * 80)
        print(" DEBUG COMPLETE")
        print("=" * 80)

        return 0


if __name__ == '__main__':
    sys.exit(main())

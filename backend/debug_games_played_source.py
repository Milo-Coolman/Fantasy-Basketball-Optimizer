#!/usr/bin/env python3
"""
Debug script to find the TRUE source of games_played in ESPN API data.

Compares a healthy player (Luka Doncic) with an injured player (Kyrie Irving)
to understand where games played data actually lives.
"""

import json
import logging
import sys
import os
from pprint import pprint

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


def safe_get_attr(obj, attr, default="<not found>"):
    """Safely get an attribute from an object."""
    try:
        val = getattr(obj, attr, default)
        if callable(val):
            return "<method>"
        return val
    except Exception as e:
        return f"<error: {e}>"


def explore_object(obj, name="object", max_depth=2, current_depth=0, visited=None):
    """Explore an object's attributes and structure."""
    if visited is None:
        visited = set()

    obj_id = id(obj)
    if obj_id in visited:
        print(f"{'  ' * current_depth}(circular reference)")
        return
    visited.add(obj_id)

    if current_depth >= max_depth:
        print(f"{'  ' * current_depth}... (max depth)")
        return

    indent = "  " * current_depth

    if isinstance(obj, dict):
        print(f"{indent}{name}: <dict with {len(obj)} keys>")
        for k, v in list(obj.items())[:15]:
            if isinstance(v, (dict, list)) and current_depth < max_depth - 1:
                explore_object(v, str(k), max_depth, current_depth + 1, visited)
            else:
                val_str = str(v)[:60] + "..." if len(str(v)) > 60 else str(v)
                print(f"{indent}  {k}: {val_str}")
        if len(obj) > 15:
            print(f"{indent}  ... and {len(obj) - 15} more keys")
    elif isinstance(obj, list):
        print(f"{indent}{name}: <list with {len(obj)} items>")
        if obj and current_depth < max_depth - 1:
            explore_object(obj[0], "[0]", max_depth, current_depth + 1, visited)
    elif hasattr(obj, '__dict__'):
        attrs = [a for a in dir(obj) if not a.startswith('_')]
        print(f"{indent}{name}: <{type(obj).__name__} with {len(attrs)} attrs>")
        for attr in attrs[:20]:
            val = safe_get_attr(obj, attr)
            if val != "<method>":
                val_str = str(val)[:50] + "..." if len(str(val)) > 50 else str(val)
                print(f"{indent}  .{attr}: {val_str}")
    else:
        val_str = str(obj)[:60] + "..." if len(str(obj)) > 60 else str(obj)
        print(f"{indent}{name}: {val_str}")


def find_games_played_keys(d, prefix="", results=None):
    """Recursively search for any key that might contain games played."""
    if results is None:
        results = []

    if isinstance(d, dict):
        for k, v in d.items():
            key_lower = str(k).lower()
            # Look for keys that might indicate games played
            if any(term in key_lower for term in ['game', 'gp', 'played', 'applied', 'total']):
                if isinstance(v, (int, float)) and v >= 0:
                    results.append((f"{prefix}{k}", v))
                elif isinstance(v, dict) and 'applied_total' in v:
                    results.append((f"{prefix}{k}.applied_total", v.get('applied_total')))

            if isinstance(v, dict):
                find_games_played_keys(v, f"{prefix}{k}.", results)

    return results


def main():
    print_section("GAMES PLAYED SOURCE INVESTIGATION")
    print("Comparing healthy player (Luka) vs injured player (Kyrie)")

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

        # Connect to ESPN
        from backend.services.espn_client import ESPNClient
        espn_client = ESPNClient(
            league_id=int(league.espn_league_id),
            year=league.season,
            espn_s2=league.espn_s2_cookie,
            swid=league.swid_cookie
        )

        # Also get raw ESPN league object
        print_section("STEP 1: ACCESS RAW ESPN-API LEAGUE OBJECT")

        raw_league = espn_client._league
        print(f"Raw league type: {type(raw_league)}")
        print(f"Raw league year: {safe_get_attr(raw_league, 'year')}")

        # List available attributes on raw league
        league_attrs = [a for a in dir(raw_league) if not a.startswith('_')]
        print(f"\nRaw league attributes ({len(league_attrs)}):")
        for attr in league_attrs:
            val = safe_get_attr(raw_league, attr)
            if val != "<method>":
                val_str = str(val)[:50] + "..." if len(str(val)) > 50 else str(val)
                print(f"  .{attr}: {val_str}")

        # Step 2: Get rosters and find target players
        print_section("STEP 2: FIND LUKA DONCIC AND KYRIE IRVING")

        all_rosters = espn_client.get_all_rosters()

        luka = None
        kyrie = None

        for team_id, roster in all_rosters.items():
            for player in roster:
                name = player.get('name', '').lower()
                if 'luka' in name or 'doncic' in name:
                    luka = player
                    print(f"Found Luka: {player.get('name')} (Team {team_id})")
                if 'kyrie' in name or 'irving' in name:
                    kyrie = player
                    print(f"Found Kyrie: {player.get('name')} (Team {team_id})")

        if not luka:
            print("WARNING: Luka not found, using first player with games")
            for team_id, roster in all_rosters.items():
                if roster:
                    luka = roster[0]
                    print(f"Using instead: {luka.get('name')}")
                    break

        if not kyrie:
            print("WARNING: Kyrie not found, looking for any injured player")
            for team_id, roster in all_rosters.items():
                for player in roster:
                    if player.get('injury_status') in ['OUT', 'IR', 'INJ_RESERVE', 'INJURED_RESERVE']:
                        kyrie = player
                        print(f"Using injured player: {player.get('name')} ({player.get('injury_status')})")
                        break
                if kyrie:
                    break

        # Step 3: Get raw player objects from ESPN API
        print_section("STEP 3: ACCESS RAW ESPN PLAYER OBJECTS")

        # The espn-api library stores Player objects
        raw_teams = safe_get_attr(raw_league, 'teams', [])
        print(f"Found {len(raw_teams) if isinstance(raw_teams, list) else 'N/A'} raw teams")

        raw_luka = None
        raw_kyrie = None

        if isinstance(raw_teams, list):
            for team in raw_teams:
                roster = safe_get_attr(team, 'roster', [])
                if isinstance(roster, list):
                    for player in roster:
                        pname = safe_get_attr(player, 'name', '').lower()
                        if 'luka' in pname or 'doncic' in pname:
                            raw_luka = player
                            print(f"Found raw Luka: {safe_get_attr(player, 'name')}")
                        if 'kyrie' in pname or 'irving' in pname:
                            raw_kyrie = player
                            print(f"Found raw Kyrie: {safe_get_attr(player, 'name')}")

        # Step 4: Examine raw player objects
        for name, raw_player, processed_player in [
            ("LUKA DONCIC", raw_luka, luka),
            ("KYRIE IRVING", raw_kyrie, kyrie)
        ]:
            if not processed_player:
                continue

            print_section(f"EXAMINING: {name}")

            # 4a: All stats periods available
            print_subsection(f"{name}: Stats Periods Available")

            stats = processed_player.get('stats', {})
            print(f"Stats dict has {len(stats)} keys:")
            for key in sorted(stats.keys()):
                val = stats[key]
                if isinstance(val, dict):
                    # Look for games info inside
                    games_info = []
                    for gk in ['gamesPlayed', 'GP', 'games_played', 'applied_total', 'G']:
                        if gk in val:
                            games_info.append(f"{gk}={val[gk]}")
                    games_str = f" [{', '.join(games_info)}]" if games_info else ""
                    print(f"  '{key}': <dict with {len(val)} keys>{games_str}")
                else:
                    print(f"  '{key}': {val}")

            # 4b: Search ALL nested dicts for games-related keys
            print_subsection(f"{name}: All Games-Related Keys Found")

            games_keys = find_games_played_keys(stats)
            if games_keys:
                for path, value in games_keys:
                    print(f"  stats.{path} = {value}")
            else:
                print("  No games-related keys found in stats!")

            # Also search the player dict itself
            games_keys_player = find_games_played_keys(processed_player, prefix="player.")
            for path, value in games_keys_player:
                if 'stats.' not in path:  # Avoid duplicates
                    print(f"  {path} = {value}")

            # 4c: Check per_game_stats
            print_subsection(f"{name}: per_game_stats Analysis")

            per_game = processed_player.get('per_game_stats', {})
            if per_game:
                print("per_game_stats exists with keys:", list(per_game.keys()))
                print("Values:", per_game)
                pts = per_game.get('pts', per_game.get('PTS', 0))
                if pts and pts > 0:
                    print(f"\n  * pts_per_game = {pts}")
                    print(f"  * If per-game stats exist, player MUST have played games!")
            else:
                print("per_game_stats is empty or missing")

            # 4d: Examine raw ESPN player object
            if raw_player:
                print_subsection(f"{name}: Raw ESPN Player Object")

                # List all attributes
                attrs = [a for a in dir(raw_player) if not a.startswith('_')]
                print(f"Raw player has {len(attrs)} attributes:")

                important_attrs = ['stats', 'total_points', 'avg_points', 'projected_total_points',
                                   'projected_avg_points', 'schedule', 'game_played', 'gamesPlayed']

                for attr in attrs:
                    val = safe_get_attr(raw_player, attr)
                    if val != "<method>":
                        is_important = attr in important_attrs or 'game' in attr.lower()
                        marker = " ***" if is_important else ""
                        val_str = str(val)[:60] + "..." if len(str(val)) > 60 else str(val)
                        print(f"  .{attr}: {val_str}{marker}")

                # Deep dive into .stats if it exists
                raw_stats = safe_get_attr(raw_player, 'stats', None)
                if raw_stats and raw_stats != "<not found>":
                    print(f"\n  Raw player.stats type: {type(raw_stats)}")
                    if isinstance(raw_stats, dict):
                        print(f"  Raw player.stats keys: {list(raw_stats.keys())}")
                        for k, v in raw_stats.items():
                            if isinstance(v, dict):
                                print(f"    [{k}]: {list(v.keys())[:10]}...")
                            else:
                                print(f"    [{k}]: {v}")

            # 4e: Check current_season structure in detail
            print_subsection(f"{name}: current_season Deep Dive")

            current_season = stats.get('current_season', {})
            if current_season:
                print("current_season structure:")
                for k, v in current_season.items():
                    if isinstance(v, dict):
                        print(f"  {k}: <dict> = {v}")
                    else:
                        print(f"  {k}: {v}")
            else:
                print("No current_season in stats")

            # 4f: Check the dynamic season key
            print_subsection(f"{name}: Dynamic Season Key '{season}_total'")

            season_total = stats.get(f'{season}_total', {})
            if season_total:
                print(f"'{season}_total' structure:")
                for k, v in season_total.items():
                    if isinstance(v, dict):
                        print(f"  {k}: <dict with {len(v)} keys>")
                        for sk, sv in list(v.items())[:10]:
                            print(f"    {sk}: {sv}")
                    else:
                        print(f"  {k}: {v}")
            else:
                print(f"No '{season}_total' in stats")

        # Step 5: Check team stats for total games played
        print_section("STEP 5: TEAM-LEVEL GAMES PLAYED")

        teams = espn_client.get_teams()
        for team in teams[:3]:  # First 3 teams
            team_name = team.get('team_name', 'Unknown')
            team_stats = team.get('stats', {})
            print(f"\n{team_name}:")
            print(f"  Team stats keys: {list(team_stats.keys())}")

            # Look for GP or games played
            for key in ['GP', 'gamesPlayed', 'games_played', 'G']:
                if key in team_stats:
                    print(f"  {key}: {team_stats[key]}")

        # Step 6: Try to access box scores or game logs
        print_section("STEP 6: BOX SCORES / GAME LOGS")

        # Check if raw league has box_scores
        box_scores = safe_get_attr(raw_league, 'box_scores', None)
        print(f"raw_league.box_scores: {type(box_scores) if box_scores else 'None'}")

        if box_scores and isinstance(box_scores, list) and len(box_scores) > 0:
            print(f"Found {len(box_scores)} box scores")
            # Examine first box score
            first_box = box_scores[0]
            print(f"First box score type: {type(first_box)}")
            explore_object(first_box, "box_score[0]", max_depth=2)

        # Check for recent_activity or other data sources
        recent = safe_get_attr(raw_league, 'recent_activity', None)
        print(f"\nraw_league.recent_activity: {type(recent) if recent else 'None'}")

        # Step 7: ESPN stat ID mapping check
        print_section("STEP 7: ESPN STAT ID REFERENCE")

        print("ESPN uses numeric stat IDs. Common mappings:")
        print("  '0': PTS (Points)")
        print("  '1': BLK (Blocks)")
        print("  '2': STL (Steals)")
        print("  '3': AST (Assists)")
        print("  '6': REB (Rebounds)")
        print("  '17': 3PM (3-Pointers Made)")
        print("  '19' or '11': TO (Turnovers)")
        print("  '40' or '41': Games Played (GP) - CHECK THIS!")
        print("  'applied_total': Often contains GP")
        print("  'applied_average': Often contains per-game stats")

        # Check if stat IDs exist in player stats
        if luka:
            stats = luka.get('stats', {})
            for period_key, period_val in stats.items():
                if isinstance(period_val, dict):
                    # Check for numeric keys that might be GP
                    numeric_keys = [k for k in period_val.keys() if str(k).isdigit()]
                    if numeric_keys:
                        print(f"\n{period_key} has numeric stat IDs: {numeric_keys[:15]}...")
                        # Check specific IDs that might be GP
                        for gp_id in ['40', '41', '42', 40, 41, 42]:
                            if gp_id in period_val:
                                print(f"  *** Found stat ID {gp_id}: {period_val[gp_id]} ***")

        # Step 8: Summary
        print_section("SUMMARY")

        print("Locations where games_played might be stored:")
        print("  1. player.stats['current_season']['games_played']")
        print("  2. player.stats['{season}_total']['applied_total']")
        print("  3. player.stats['{season}_total']['GP']")
        print("  4. Raw ESPN Player object attributes")
        print("  5. Calculated from total_points / avg_points")
        print("  6. ESPN stat ID '40' or '41' in stats dict")

        print("\nTo fix the tier calculation:")
        print("  - Update get_player_current_stats() to check all these locations")
        print("  - Use the first non-zero value found")
        print("  - Log which source provided the games_played value")

        print("\n" + "=" * 80)
        print(" DEBUG COMPLETE")
        print("=" * 80)

        return 0


if __name__ == '__main__':
    sys.exit(main())

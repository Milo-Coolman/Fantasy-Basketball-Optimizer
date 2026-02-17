#!/usr/bin/env python3
"""
Debug script for examining Roto league data from ESPN API.

This script connects to an ESPN Fantasy Basketball league and prints
detailed information about the league structure, scoring settings,
standings, and team statistics.

Usage:
    python debug_roto_league.py

Make sure to set up your .env file with the league credentials or
update the credentials in this script.
"""

import os
import sys
import json
from pprint import pprint
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from espn_api.basketball import League
except ImportError:
    print("Error: espn_api package not installed.")
    print("Install with: pip install espn-api")
    sys.exit(1)

# Try to load from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def get_credentials():
    """Get ESPN credentials from environment or prompt user."""
    league_id = os.getenv('ESPN_LEAGUE_ID')
    espn_s2 = os.getenv('ESPN_S2')
    swid = os.getenv('SWID')
    year = os.getenv('ESPN_YEAR', '2025')

    if not all([league_id, espn_s2, swid]):
        print("\n" + "=" * 60)
        print("ESPN Credentials Required")
        print("=" * 60)
        print("\nYou can set these in your .env file:")
        print("  ESPN_LEAGUE_ID=your_league_id")
        print("  ESPN_S2=your_espn_s2_cookie")
        print("  SWID=your_swid_cookie")
        print("  ESPN_YEAR=2025")
        print("\nOr enter them now:\n")

        if not league_id:
            league_id = input("League ID: ").strip()
        if not espn_s2:
            espn_s2 = input("ESPN_S2 cookie: ").strip()
        if not swid:
            swid = input("SWID cookie: ").strip()
        if not year:
            year = input("Year (default 2025): ").strip() or "2025"

    return {
        'league_id': int(league_id),
        'espn_s2': espn_s2,
        'swid': swid,
        'year': int(year)
    }


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_subsection(title):
    """Print a formatted subsection header."""
    print(f"\n--- {title} ---")


def safe_getattr(obj, attr, default=None):
    """Safely get an attribute from an object."""
    try:
        value = getattr(obj, attr, default)
        return value if value is not None else default
    except Exception:
        return default


def examine_league(league):
    """Examine and print all relevant league information."""

    # ==========================================================================
    # SECTION 1: Basic League Info
    # ==========================================================================
    print_section("1. BASIC LEAGUE INFORMATION")

    print(f"\nLeague Name: {safe_getattr(league, 'name', 'Unknown')}")
    print(f"League ID: {safe_getattr(league, 'league_id', 'Unknown')}")
    print(f"Year/Season: {safe_getattr(league, 'year', 'Unknown')}")
    print(f"Number of Teams: {len(league.teams) if hasattr(league, 'teams') else 'Unknown'}")

    # Check league type attributes
    print_subsection("League Type Detection")
    scoring_type = safe_getattr(league, 'scoring_type', None)
    print(f"league.scoring_type: {scoring_type}")

    # Try to detect if it's Roto
    settings = safe_getattr(league, 'settings', None)
    if settings:
        print(f"league.settings type: {type(settings)}")
        print(f"league.settings attributes: {dir(settings)}")

    # ==========================================================================
    # SECTION 2: Scoring Settings (RAW)
    # ==========================================================================
    print_section("2. RAW SCORING SETTINGS")

    if settings:
        scoring_settings = safe_getattr(settings, 'scoring_settings', None)
        print_subsection("settings.scoring_settings")
        if scoring_settings:
            print(f"Type: {type(scoring_settings)}")
            if isinstance(scoring_settings, dict):
                print("\nScoring Settings (dict):")
                pprint(scoring_settings)
            elif hasattr(scoring_settings, '__dict__'):
                print("\nScoring Settings (object):")
                pprint(vars(scoring_settings))
            else:
                print(f"Value: {scoring_settings}")
        else:
            print("No scoring_settings found")

        # Try other settings attributes
        print_subsection("Other Settings Attributes")
        for attr in ['stat_categories', 'categories', 'scoring_type', 'reg_season_count']:
            val = safe_getattr(settings, attr, None)
            if val is not None:
                print(f"settings.{attr}: {val}")

    # Try direct league attributes for scoring
    print_subsection("Direct League Scoring Attributes")
    for attr in ['stat_categories', 'scoring_items', 'categories']:
        val = safe_getattr(league, attr, None)
        if val is not None:
            print(f"league.{attr}: {val}")

    # ==========================================================================
    # SECTION 3: Standings Data
    # ==========================================================================
    print_section("3. STANDINGS DATA")

    print_subsection("league.standings() output")
    try:
        standings = league.standings()
        print(f"Type: {type(standings)}")
        print(f"Length: {len(standings) if standings else 0}")

        if standings:
            print("\nFirst team in standings:")
            first_team = standings[0]
            print(f"  Type: {type(first_team)}")
            print(f"  Attributes: {dir(first_team)}")

            if hasattr(first_team, '__dict__'):
                print("\n  Team __dict__:")
                for k, v in vars(first_team).items():
                    if not k.startswith('_'):
                        print(f"    {k}: {v}")
    except Exception as e:
        print(f"Error getting standings: {e}")

    # ==========================================================================
    # SECTION 4: Team Details
    # ==========================================================================
    print_section("4. TEAM DETAILS")

    if hasattr(league, 'teams'):
        for i, team in enumerate(league.teams):
            print_subsection(f"Team {i + 1}: {safe_getattr(team, 'team_name', 'Unknown')}")

            # Basic info
            print(f"  Team ID: {safe_getattr(team, 'team_id', 'Unknown')}")
            print(f"  Team Name: {safe_getattr(team, 'team_name', 'Unknown')}")
            print(f"  Team Abbrev: {safe_getattr(team, 'team_abbrev', 'Unknown')}")

            # Owner info
            owner = safe_getattr(team, 'owner', None)
            owners = safe_getattr(team, 'owners', None)
            print(f"  Owner: {owner}")
            print(f"  Owners: {owners}")

            # Standings info
            print(f"  Standing: {safe_getattr(team, 'standing', 'Unknown')}")
            print(f"  Final Standing: {safe_getattr(team, 'final_standing', 'Unknown')}")
            print(f"  Playoff Pct: {safe_getattr(team, 'playoff_pct', 'Unknown')}")

            # Record info (for H2H)
            wins = safe_getattr(team, 'wins', None)
            losses = safe_getattr(team, 'losses', None)
            if wins is not None or losses is not None:
                print(f"  Record: {wins}-{losses}")

            # Points info (for Roto)
            points = safe_getattr(team, 'points', None)
            points_for = safe_getattr(team, 'points_for', None)
            print(f"  Points: {points}")
            print(f"  Points For: {points_for}")

            # Stats
            stats = safe_getattr(team, 'stats', None)
            if stats:
                print(f"  Stats type: {type(stats)}")
                if isinstance(stats, dict):
                    print("  Stats (dict):")
                    for k, v in stats.items():
                        print(f"    {k}: {v}")
                elif hasattr(stats, '__dict__'):
                    print("  Stats (object):")
                    for k, v in vars(stats).items():
                        if not k.startswith('_'):
                            print(f"    {k}: {v}")

            # Category stats (try various attribute names)
            for attr in ['category_stats', 'roto_stats', 'cumulative_score', 'stats_total']:
                val = safe_getattr(team, attr, None)
                if val is not None:
                    print(f"  {attr}: {val}")

            # Limit output for large leagues
            if i >= 2:
                remaining = len(league.teams) - 3
                if remaining > 0:
                    print(f"\n  ... and {remaining} more teams")
                break

    # ==========================================================================
    # SECTION 5: Raw Team Object Inspection
    # ==========================================================================
    print_section("5. RAW TEAM OBJECT (First Team)")

    if hasattr(league, 'teams') and league.teams:
        first_team = league.teams[0]
        print(f"\nAll attributes of first team:")
        print(f"dir(team): {dir(first_team)}")

        print("\n__dict__ contents:")
        if hasattr(first_team, '__dict__'):
            for k, v in vars(first_team).items():
                v_str = str(v)
                if len(v_str) > 200:
                    v_str = v_str[:200] + "..."
                print(f"  {k}: {v_str}")

    # ==========================================================================
    # SECTION 6: Roster Sample
    # ==========================================================================
    print_section("6. ROSTER SAMPLE (First Team)")

    if hasattr(league, 'teams') and league.teams:
        first_team = league.teams[0]
        roster = safe_getattr(first_team, 'roster', [])

        print(f"\nRoster size: {len(roster) if roster else 0}")

        if roster:
            print("\nFirst 3 players:")
            for i, player in enumerate(roster[:3]):
                print(f"\n  Player {i + 1}: {safe_getattr(player, 'name', 'Unknown')}")
                print(f"    Position: {safe_getattr(player, 'position', 'Unknown')}")
                print(f"    Pro Team: {safe_getattr(player, 'proTeam', 'Unknown')}")

                # Player stats
                player_stats = safe_getattr(player, 'stats', None)
                if player_stats:
                    print(f"    Stats type: {type(player_stats)}")
                    if isinstance(player_stats, dict):
                        for stat_key in list(player_stats.keys())[:3]:
                            print(f"    Stats[{stat_key}]: {player_stats[stat_key]}")

    # ==========================================================================
    # SECTION 7: Free Agents Sample
    # ==========================================================================
    print_section("7. FREE AGENTS SAMPLE")

    try:
        free_agents = league.free_agents(size=5)
        print(f"\nFree agents retrieved: {len(free_agents)}")

        if free_agents:
            print("\nFirst 3 free agents:")
            for i, player in enumerate(free_agents[:3]):
                print(f"\n  Player {i + 1}: {safe_getattr(player, 'name', 'Unknown')}")
                print(f"    Position: {safe_getattr(player, 'position', 'Unknown')}")
                print(f"    Percent Owned: {safe_getattr(player, 'percent_owned', 'Unknown')}")
    except Exception as e:
        print(f"Error getting free agents: {e}")

    # ==========================================================================
    # SECTION 8: Summary
    # ==========================================================================
    print_section("8. SUMMARY")

    print(f"""
League: {safe_getattr(league, 'name', 'Unknown')}
Teams: {len(league.teams) if hasattr(league, 'teams') else 'Unknown'}
Year: {safe_getattr(league, 'year', 'Unknown')}
Scoring Type: {safe_getattr(league, 'scoring_type', 'Unknown')}

Debug completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Use this information to:
1. Identify the correct attribute names for category stats
2. Understand the standings data structure
3. Map ESPN data to our frontend components
""")


def main():
    """Main entry point."""
    print("\n" + "=" * 70)
    print("  ESPN Fantasy Basketball - Roto League Debug Script")
    print("=" * 70)

    try:
        # Get credentials
        creds = get_credentials()

        print(f"\nConnecting to league {creds['league_id']} for {creds['year']}...")

        # Connect to ESPN
        league = League(
            league_id=creds['league_id'],
            year=creds['year'],
            espn_s2=creds['espn_s2'],
            swid=creds['swid']
        )

        print("Connected successfully!")

        # Examine the league
        examine_league(league)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

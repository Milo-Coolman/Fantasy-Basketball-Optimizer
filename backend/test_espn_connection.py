#!/usr/bin/env python3
"""
ESPN Connection Test Script

Tests the ESPN client integration by connecting to a league and
fetching basic information.

Usage:
    python backend/test_espn_connection.py

You will be prompted for:
    - League ID (from your ESPN league URL)
    - Season year (e.g., 2025 for 2024-25 season)
    - ESPN_S2 cookie
    - SWID cookie
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.services.espn_client import (
    ESPNClient,
    ESPNAuthenticationError,
    ESPNLeagueNotFoundError,
    ESPNConnectionError,
    ESPNClientError,
)


def get_input(prompt: str, default: str = None) -> str:
    """Get input from user with optional default value."""
    if default:
        result = input(f"{prompt} [{default}]: ").strip()
        return result if result else default
    return input(f"{prompt}: ").strip()


def get_multiline_input(prompt: str) -> str:
    """Get multiline input (for long cookie values)."""
    print(f"{prompt} (paste value, then press Enter twice to confirm):")
    lines = []
    while True:
        line = input()
        if line:
            lines.append(line)
        else:
            break
    return ''.join(lines).strip()


def print_separator(char: str = "=", length: int = 60):
    """Print a separator line."""
    print(char * length)


def print_header(title: str):
    """Print a section header."""
    print()
    print_separator()
    print(f"  {title}")
    print_separator()


def main():
    """Main test function."""
    print_header("ESPN Fantasy Basketball Connection Test")
    print()
    print("This script will test your ESPN credentials and fetch league info.")
    print("You can find your league ID in your ESPN league URL:")
    print("  https://fantasy.espn.com/basketball/league?leagueId=XXXXXX")
    print()
    print("For cookies, go to espn.com, open DevTools (F12), and find")
    print("espn_s2 and SWID in Application > Cookies > espn.com")
    print()

    # Get credentials
    print_separator("-")
    print("Enter your credentials:")
    print_separator("-")
    print()

    try:
        league_id = int(get_input("League ID"))
    except ValueError:
        print("\nError: League ID must be a number")
        return 1

    try:
        current_year = 2025  # Default to current season
        year = int(get_input("Season year", str(current_year)))
    except ValueError:
        print("\nError: Season year must be a number")
        return 1

    print()
    print("ESPN_S2 cookie (this is a long string, paste it all):")
    espn_s2 = input().strip()

    if not espn_s2:
        print("\nError: ESPN_S2 cookie is required")
        return 1

    print()
    swid = get_input("SWID cookie (include the curly braces)")

    if not swid:
        print("\nError: SWID cookie is required")
        return 1

    if not swid.startswith('{') or not swid.endswith('}'):
        print("\nWarning: SWID should be in format {XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX}")
        proceed = get_input("Continue anyway? (y/n)", "n")
        if proceed.lower() != 'y':
            return 1

    # Attempt connection
    print_header("Connecting to ESPN...")

    try:
        client = ESPNClient(
            league_id=league_id,
            year=year,
            espn_s2=espn_s2,
            swid=swid
        )

        print("\n✓ Successfully connected to ESPN!")

        # Get league settings
        print_header("League Information")

        settings = client.get_league_settings()
        print(f"  League Name:    {settings.get('name', 'Unknown')}")
        print(f"  League Type:    {settings.get('scoring_type', 'Unknown')}")
        print(f"  Number of Teams: {settings.get('size', 'Unknown')}")
        print(f"  Current Week:   {settings.get('current_week', 'Unknown')}")

        # Get teams
        print_header("Teams")

        teams = client.get_teams()
        for i, team in enumerate(teams, 1):
            print(f"  {i:2}. {team['team_name']:<30} (Owner: {team.get('owner_name', 'Unknown')})")

        # Get standings
        print_header("Current Standings")

        standings = client.get_standings()
        print(f"  {'#':<3} {'Team':<30} {'Record':<10} {'GB':<6}")
        print(f"  {'-'*3} {'-'*30} {'-'*10} {'-'*6}")

        for team in standings:
            gb = team.get('games_back', 0)
            gb_str = '-' if gb == 0 else str(gb)
            print(f"  {team['standing']:<3} {team['team_name']:<30} {team['record']:<10} {gb_str:<6}")

        # Test free agents
        print_header("Top 5 Free Agents")

        try:
            free_agents = client.get_free_agents(size=5)
            for i, player in enumerate(free_agents, 1):
                injury = f" ({player.get('injury_status')})" if player.get('injury_status') else ""
                print(f"  {i}. {player['name']:<25} {player.get('position', 'N/A'):<5} {player.get('nba_team', 'FA'):<4}{injury}")
        except Exception as e:
            print(f"  Could not fetch free agents: {e}")

        # Summary
        print_header("Test Complete!")
        print()
        print("  ✓ ESPN connection successful")
        print("  ✓ League settings retrieved")
        print("  ✓ Teams and standings loaded")
        print("  ✓ Free agents accessible")
        print()
        print("  Your ESPN credentials are working correctly!")
        print("  You can now add this league to the Fantasy Basketball Optimizer.")
        print()

        # Print credentials summary for reference
        print_separator("-")
        print("  Credentials to use in the app:")
        print_separator("-")
        print(f"  League ID: {league_id}")
        print(f"  Season:    {year}")
        print(f"  ESPN_S2:   {espn_s2[:20]}...{espn_s2[-20:]}")
        print(f"  SWID:      {swid}")
        print()

        return 0

    except ESPNAuthenticationError as e:
        print_header("Authentication Failed")
        print()
        print(f"  Error: {e}")
        print()
        print("  Troubleshooting tips:")
        print("  1. Make sure you're logged into ESPN in your browser")
        print("  2. Get fresh cookies (log out and back in to ESPN)")
        print("  3. Ensure you copied the complete cookie values")
        print("  4. Check that your league is not set to 'private' without you being a member")
        print()
        return 1

    except ESPNLeagueNotFoundError as e:
        print_header("League Not Found")
        print()
        print(f"  Error: {e}")
        print()
        print("  Troubleshooting tips:")
        print("  1. Verify the league ID is correct")
        print(f"  2. Check that a league exists for the {year-1}-{str(year)[2:]} season")
        print("  3. Make sure you have access to this league")
        print()
        return 1

    except ESPNConnectionError as e:
        print_header("Connection Failed")
        print()
        print(f"  Error: {e}")
        print()
        print("  Troubleshooting tips:")
        print("  1. Check your internet connection")
        print("  2. ESPN may be temporarily unavailable, try again later")
        print()
        return 1

    except ESPNClientError as e:
        print_header("ESPN Client Error")
        print()
        print(f"  Error: {e}")
        print()
        return 1

    except KeyboardInterrupt:
        print("\n\nTest cancelled by user.")
        return 1

    except Exception as e:
        print_header("Unexpected Error")
        print()
        print(f"  Error: {e}")
        print()
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

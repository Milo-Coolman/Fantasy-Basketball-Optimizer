#!/usr/bin/env python3
"""
Debug script for user team identification.

This script helps debug why the user's team might not be correctly identified
by showing exactly what SWID values are being compared.

Usage:
    python debug_user_team.py
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from espn_api.basketball import League as ESPNLeague
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


def normalize_swid(swid: str) -> str:
    """Normalize SWID for comparison."""
    if not swid:
        return ''
    return swid.strip('{}').lower()


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
        print("\nEnter your credentials:\n")

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


def main():
    print("\n" + "=" * 70)
    print("  User Team Identification Debug")
    print("=" * 70)

    # Get credentials
    creds = get_credentials()

    print(f"\nConnecting to league {creds['league_id']} for {creds['year']}...")

    try:
        league = ESPNLeague(
            league_id=creds['league_id'],
            year=creds['year'],
            espn_s2=creds['espn_s2'],
            swid=creds['swid']
        )
        print(f"Connected to: {league.settings.name}")
    except Exception as e:
        print(f"Error connecting: {e}")
        sys.exit(1)

    # ==========================================================================
    # SWID Analysis
    # ==========================================================================
    print("\n" + "=" * 70)
    print("  SWID ANALYSIS")
    print("=" * 70)

    swid = creds['swid']
    swid_normalized = normalize_swid(swid)

    print(f"\n  Your SWID (raw):        {swid}")
    print(f"  Your SWID (normalized): {swid_normalized}")
    print(f"  SWID length:            {len(swid_normalized)} characters")

    # ==========================================================================
    # Team Owner Analysis
    # ==========================================================================
    print("\n" + "=" * 70)
    print("  TEAM OWNER ANALYSIS")
    print("=" * 70)

    match_found = None
    all_owner_ids = []

    for i, team in enumerate(league.teams, 1):
        print(f"\n--- Team {i}: {team.team_name} ---")
        print(f"  Team ID: {team.team_id}")
        print(f"  Standing: {team.standing}")

        # Check for owners attribute
        owners = getattr(team, 'owners', None)

        if owners is None:
            print("  Owners: None (attribute doesn't exist)")
            # Try alternative attributes
            for attr in ['owner', 'owner_id', 'ownerId', 'primaryOwner']:
                val = getattr(team, attr, None)
                if val:
                    print(f"  Alternative attr '{attr}': {val}")
        elif not owners:
            print("  Owners: [] (empty list)")
        else:
            print(f"  Owners: {len(owners)} owner(s)")

            for j, owner in enumerate(owners):
                print(f"\n  Owner {j + 1}:")

                # Handle dict vs object
                if isinstance(owner, dict):
                    owner_id = owner.get('id', 'N/A')
                    first_name = owner.get('firstName', '')
                    last_name = owner.get('lastName', '')
                    display_name = owner.get('displayName', '')

                    print(f"    Type: dict")
                    print(f"    id: {owner_id}")
                    print(f"    firstName: {first_name}")
                    print(f"    lastName: {last_name}")
                    print(f"    displayName: {display_name}")

                    # Show all keys in the dict
                    print(f"    All keys: {list(owner.keys())}")

                else:
                    # It's an object
                    owner_id = getattr(owner, 'id', 'N/A')
                    print(f"    Type: {type(owner).__name__}")
                    print(f"    id: {owner_id}")

                    # Try to get other attributes
                    for attr in ['firstName', 'lastName', 'displayName', 'email']:
                        val = getattr(owner, attr, None)
                        if val:
                            print(f"    {attr}: {val}")

                    # Show all attributes
                    if hasattr(owner, '__dict__'):
                        print(f"    All attrs: {list(vars(owner).keys())}")

                # Compare owner_id to SWID
                if owner_id and owner_id != 'N/A':
                    all_owner_ids.append((team.team_name, team.team_id, str(owner_id)))
                    owner_id_normalized = normalize_swid(str(owner_id))

                    print(f"\n    --- SWID Comparison ---")
                    print(f"    Owner ID (raw):        {owner_id}")
                    print(f"    Owner ID (normalized): {owner_id_normalized}")
                    print(f"    Your SWID (normalized): {swid_normalized}")

                    # Check match
                    is_match = (owner_id_normalized == swid_normalized)
                    print(f"    MATCH: {'YES!' if is_match else 'No'}")

                    if is_match:
                        match_found = {
                            'team_name': team.team_name,
                            'team_id': team.team_id,
                            'owner_id': owner_id,
                        }

    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    print(f"\n  Total teams in league: {len(league.teams)}")
    print(f"  Total owner IDs found: {len(all_owner_ids)}")

    print(f"\n  Your SWID: {swid}")
    print(f"  Normalized: {swid_normalized}")

    if match_found:
        print(f"\n  MATCH FOUND!")
        print(f"  Your team: {match_found['team_name']}")
        print(f"  Team ID: {match_found['team_id']}")
        print(f"  Owner ID: {match_found['owner_id']}")
    else:
        print(f"\n  NO MATCH FOUND")
        print(f"\n  All owner IDs in league (normalized):")
        for team_name, team_id, owner_id in all_owner_ids:
            normalized = normalize_swid(owner_id)
            matches_partial = swid_normalized in normalized or normalized in swid_normalized
            indicator = " <-- partial match?" if matches_partial else ""
            print(f"    Team {team_id} ({team_name}): {normalized}{indicator}")

        print(f"\n  Possible issues:")
        print(f"    1. SWID cookie might be incorrect or expired")
        print(f"    2. You might not be an owner in this league")
        print(f"    3. The owner ID format might be different than expected")

    # ==========================================================================
    # Raw Data Dump
    # ==========================================================================
    print("\n" + "=" * 70)
    print("  RAW OWNERS DATA (First 3 Teams)")
    print("=" * 70)

    for i, team in enumerate(league.teams[:3]):
        print(f"\n  Team: {team.team_name}")
        owners = getattr(team, 'owners', None)
        print(f"  Raw owners value: {owners}")
        if owners:
            print(f"  Type: {type(owners)}")
            if owners and len(owners) > 0:
                print(f"  First owner type: {type(owners[0])}")
                print(f"  First owner repr: {repr(owners[0])}")

    print("\n" + "=" * 70)
    print("  DEBUG COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

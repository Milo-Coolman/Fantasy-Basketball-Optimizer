#!/usr/bin/env python3
"""
Debug script to investigate ESPN's injury return date data.

Makes raw HTTP requests to ESPN Fantasy API to find where injury return dates
are stored, comparing different API views and examining raw JSON responses.

Usage:
    cd /Users/milo/fantasy-basketball-optimizer
    source venv/bin/activate
    python backend/debug_espn_injury_api.py
"""

import json
import sys
import requests
from typing import Dict, Any, List, Optional

# Add project root to path
sys.path.insert(0, '/Users/milo/fantasy-basketball-optimizer')

from backend.app import create_app
from backend.models import League


def get_league_credentials() -> Dict[str, Any]:
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


def make_espn_request(
    league_id: int,
    season: int,
    espn_s2: str,
    swid: str,
    views: List[str],
    extra_params: Optional[Dict] = None
) -> Dict:
    """Make a raw HTTP request to ESPN Fantasy API."""
    endpoint = (
        f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/fba/"
        f"seasons/{season}/segments/0/leagues/{league_id}"
    )

    params = {'view': views}
    if extra_params:
        params.update(extra_params)

    cookies = {'espn_s2': espn_s2, 'SWID': swid}

    response = requests.get(endpoint, params=params, cookies=cookies, timeout=30)
    response.raise_for_status()
    return response.json()


def find_injury_fields(obj: Any, path: str = "", depth: int = 0) -> List[tuple]:
    """Recursively search for injury-related fields in JSON."""
    results = []
    injury_keywords = ['return', 'injury', 'expected', 'timeline', 'status', 'health', 'out']

    if depth > 10:  # Prevent infinite recursion
        return results

    if isinstance(obj, dict):
        for key, value in obj.items():
            current_path = f"{path}.{key}" if path else key
            # Check if key contains injury-related keywords
            key_lower = key.lower()
            if any(kw in key_lower for kw in injury_keywords):
                results.append((current_path, key, value))
            # Recurse into nested objects
            results.extend(find_injury_fields(value, current_path, depth + 1))
    elif isinstance(obj, list) and len(obj) > 0:
        # Only check first few items to avoid huge output
        for i, item in enumerate(obj[:3]):
            current_path = f"{path}[{i}]"
            results.extend(find_injury_fields(item, current_path, depth + 1))

    return results


def find_player_in_response(data: Dict, player_names: List[str]) -> List[Dict]:
    """Find specific players in the API response."""
    found_players = []

    # Check teams -> roster -> entries -> playerPoolEntry -> player
    for team in data.get('teams', []):
        for entry in team.get('roster', {}).get('entries', []):
            player_pool = entry.get('playerPoolEntry', {})
            player = player_pool.get('player', {})
            full_name = player.get('fullName', '').lower()

            for target_name in player_names:
                if target_name.lower() in full_name:
                    found_players.append({
                        'name': player.get('fullName'),
                        'entry': entry,
                        'playerPoolEntry': player_pool,
                        'player': player,
                        'team_id': team.get('id'),
                    })

    # Also check 'players' array if present
    for player_data in data.get('players', []):
        player = player_data.get('player', {})
        full_name = player.get('fullName', '').lower()

        for target_name in player_names:
            if target_name.lower() in full_name:
                found_players.append({
                    'name': player.get('fullName'),
                    'player_data': player_data,
                    'player': player,
                })

    return found_players


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main():
    print("=" * 80)
    print("ESPN INJURY API INVESTIGATION")
    print("Searching for injury return date fields in raw API responses")
    print("=" * 80)

    # Get credentials
    creds = get_league_credentials()
    print(f"\nLeague: {creds['name']}")
    print(f"League ID: {creds['league_id']}")
    print(f"Season: {creds['season']}")

    # Target players to find
    target_players = ['Trey Murphy', 'Jaren Jackson']
    print(f"\nSearching for players: {target_players}")

    # Different API views to try
    view_combinations = [
        (['mTeam', 'mRoster'], "Basic roster view"),
        (['mTeam', 'mRoster', 'kona_player_info'], "With kona_player_info"),
        (['mTeam', 'mRoster', 'players_wl'], "With players_wl (watchlist)"),
        (['mTeam', 'mRoster', 'kona_playercard'], "With kona_playercard"),
        (['mTeam', 'mRoster', 'mMatchup'], "With mMatchup"),
        (['mTeam', 'mRoster', 'kona_player_info', 'mMatchup'], "Combined views"),
    ]

    for views, description in view_combinations:
        print_section(f"API VIEW: {views} ({description})")

        try:
            data = make_espn_request(
                league_id=creds['league_id'],
                season=creds['season'],
                espn_s2=creds['espn_s2'],
                swid=creds['swid'],
                views=views
            )

            # Find target players
            found = find_player_in_response(data, target_players)

            if not found:
                print("  Target players not found in this view")
                continue

            for player_info in found:
                player_name = player_info['name']
                player_obj = player_info.get('player', {})

                print(f"\n  --- {player_name} ---")

                # Show all keys in player object
                print(f"  Player object keys: {list(player_obj.keys())}")

                # Find and display injury-related fields
                injury_fields = find_injury_fields(player_obj)
                if injury_fields:
                    print(f"\n  Injury-related fields found:")
                    for path, key, value in injury_fields:
                        value_str = str(value)
                        if len(value_str) > 100:
                            value_str = value_str[:100] + "..."
                        print(f"    {path}: {value_str}")
                else:
                    print("  No injury-related fields found in player object")

                # Show specific fields we're looking for
                print(f"\n  Key fields:")
                key_fields = [
                    'injuryStatus', 'injured', 'injuryDate', 'returnDate',
                    'expectedReturnDate', 'injuryType', 'injuryNotes',
                    'seasonOutlook', 'status', 'healthStatus', 'outlooksByWeek'
                ]
                for field in key_fields:
                    value = player_obj.get(field)
                    if value is not None:
                        print(f"    {field} = {value}")

                # Check entry and playerPoolEntry levels too
                if 'entry' in player_info:
                    entry = player_info['entry']
                    entry_injury = find_injury_fields(entry)
                    if entry_injury:
                        print(f"\n  Entry-level injury fields:")
                        for path, key, value in entry_injury[:10]:  # Limit output
                            value_str = str(value)[:100]
                            print(f"    {path}: {value_str}")

                if 'playerPoolEntry' in player_info:
                    ppe = player_info['playerPoolEntry']
                    ppe_injury = find_injury_fields(ppe)
                    if ppe_injury:
                        print(f"\n  PlayerPoolEntry-level injury fields:")
                        for path, key, value in ppe_injury[:10]:
                            value_str = str(value)[:100]
                            print(f"    {path}: {value_str}")

        except Exception as e:
            print(f"  ERROR: {e}")

    # Now dump complete JSON for target players with the most comprehensive view
    print_section("COMPLETE JSON DUMP FOR TARGET PLAYERS")

    try:
        data = make_espn_request(
            league_id=creds['league_id'],
            season=creds['season'],
            espn_s2=creds['espn_s2'],
            swid=creds['swid'],
            views=['mTeam', 'mRoster', 'kona_player_info']
        )

        found = find_player_in_response(data, target_players)

        for player_info in found:
            player_name = player_info['name']
            print(f"\n--- COMPLETE DATA FOR: {player_name} ---\n")

            # Pretty print the player object
            player_obj = player_info.get('player', {})
            print("player object:")
            print(json.dumps(player_obj, indent=2, default=str))

            # Also print entry-level data if available
            if 'entry' in player_info:
                print("\n\nentry object (excluding player):")
                entry_copy = {k: v for k, v in player_info['entry'].items()
                             if k != 'playerPoolEntry'}
                print(json.dumps(entry_copy, indent=2, default=str))

    except Exception as e:
        print(f"ERROR dumping complete JSON: {e}")
        import traceback
        traceback.print_exc()

    # Summary
    print_section("SUMMARY")
    print("""
Based on the investigation above, look for:

1. injuryStatus - Current injury status (OUT, IR, etc.)
2. injured - Boolean indicating if player is injured
3. seasonOutlook - Text that may contain return date info
4. outlooksByWeek - Weekly projections that may indicate return
5. Any field containing 'return', 'expected', 'timeline'

ESPN may NOT provide a direct 'expectedReturnDate' field.
Return dates are often embedded in:
- seasonOutlook text (e.g., "Expected to return in 2-3 weeks")
- News/notes fields
- External injury reports

If no explicit return date field exists, we need to parse the
seasonOutlook or injuryNotes text to estimate return dates.
""")


if __name__ == "__main__":
    main()

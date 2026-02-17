#!/usr/bin/env python3
"""
Debug script to find where ESPN stores starts-used-per-position data.

This script explores various parts of the ESPN API response to find
the actual number of starts used per position slot this season.

Usage:
    cd /Users/milo/fantasy-basketball-optimizer
    source venv/bin/activate
    python backend/debug_starts_used.py
"""

import json
import os
import sqlite3
import sys
from pprint import pprint

import requests

# Position mapping
SLOT_ID_TO_POSITION = {
    0: 'PG', 1: 'SG', 2: 'SF', 3: 'PF', 4: 'C',
    5: 'G', 6: 'F', 7: 'SG/SF', 8: 'G/F', 9: 'PF/C',
    10: 'F/C', 11: 'UTIL', 12: 'BE', 13: 'IR', 14: 'IR+',
}


def get_credentials():
    """Get league credentials from database."""
    db_path = 'instance/fantasy_basketball.db'
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT espn_league_id, espn_s2_cookie, swid_cookie, season
        FROM leagues LIMIT 1
    ''')
    row = cursor.fetchone()
    conn.close()
    return row


def fetch_all_views(league_id, espn_s2, swid, season):
    """Fetch ESPN data with all available views."""
    endpoint = (
        f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/fba/"
        f"seasons/{season}/segments/0/leagues/{league_id}"
    )

    views = [
        'mTeam', 'mRoster', 'mMatchup', 'mSchedule',
        'mSettings', 'mStatus', 'mLiveScoring'
    ]

    params = {'view': views}
    cookies = {'espn_s2': espn_s2, 'SWID': swid}

    response = requests.get(endpoint, params=params, cookies=cookies, timeout=30)
    return response.json()


def find_my_team(teams, swid):
    """Find user's team."""
    for team in teams:
        name = team.get('name', '') or team.get('nickname', '')
        if 'doncic' in name.lower():
            return team
    return teams[0] if teams else None


def search_for_stat_by_slot(obj, path="", depth=0, max_depth=6):
    """Recursively search for statBySlot or similar fields."""
    if depth > max_depth:
        return []

    results = []

    if isinstance(obj, dict):
        for key, value in obj.items():
            key_lower = key.lower()

            # Look for slot-related stat tracking
            if any(term in key_lower for term in ['statbyslot', 'slotstat', 'statslot', 'lineupslot']):
                results.append({
                    'path': f"{path}.{key}" if path else key,
                    'key': key,
                    'value': value,
                    'type': type(value).__name__
                })

            # Recurse
            if isinstance(value, (dict, list)):
                results.extend(search_for_stat_by_slot(value, f"{path}.{key}" if path else key, depth + 1, max_depth))

    elif isinstance(obj, list) and len(obj) > 0:
        results.extend(search_for_stat_by_slot(obj[0], f"{path}[0]", depth + 1, max_depth))

    return results


def main():
    print("=" * 80)
    print("DEBUG: FINDING ESPN STARTS-USED-PER-POSITION DATA")
    print("=" * 80)

    # Get credentials
    creds = get_credentials()
    if not creds:
        print("No league found in database")
        return 1

    league_id, espn_s2, swid, season = creds
    print(f"\nLeague ID: {league_id}, Season: {season}")

    # Fetch all data
    print("\nFetching ESPN data with all views...")
    data = fetch_all_views(league_id, espn_s2, swid, season)

    teams = data.get('teams', [])
    schedule = data.get('schedule', [])

    my_team = find_my_team(teams, swid)
    team_id = my_team.get('id')
    team_name = my_team.get('name', '') or my_team.get('nickname', '')
    print(f"Team: {team_name} (ID: {team_id})")

    # =========================================================================
    # 1. Check team-level data for slot stats
    # =========================================================================
    print("\n" + "=" * 80)
    print("1. TEAM-LEVEL DATA")
    print("=" * 80)

    print(f"\nTeam keys: {list(my_team.keys())}")

    # Check valuesByStat
    values_by_stat = my_team.get('valuesByStat', {})
    if values_by_stat:
        print(f"\nvaluesByStat ({len(values_by_stat)} entries):")
        for stat_id, value in sorted(values_by_stat.items(), key=lambda x: int(x[0]))[:20]:
            print(f"  Stat {stat_id}: {value}")

    # Check for any slot-related fields in team
    for key in my_team.keys():
        if 'slot' in key.lower() or 'stat' in key.lower():
            print(f"\nTeam.{key}: {my_team[key]}")

    # =========================================================================
    # 2. Check schedule/matchup data for cumulative scores
    # =========================================================================
    print("\n" + "=" * 80)
    print("2. SCHEDULE/MATCHUP DATA")
    print("=" * 80)

    print(f"\nTotal matchups in schedule: {len(schedule)}")

    # Find completed matchups for our team
    completed_matchups = []
    for matchup in schedule:
        if matchup.get('winner') == 'UNDECIDED':
            continue

        home = matchup.get('home', {})
        away = matchup.get('away', {})

        if home.get('teamId') == team_id or away.get('teamId') == team_id:
            completed_matchups.append(matchup)

    print(f"Completed matchups for {team_name}: {len(completed_matchups)}")

    if completed_matchups:
        # Examine the first completed matchup structure
        matchup = completed_matchups[0]
        print(f"\nFirst matchup keys: {list(matchup.keys())}")

        # Check home/away structure
        our_side = matchup.get('home') if matchup.get('home', {}).get('teamId') == team_id else matchup.get('away')
        print(f"\nOur team's matchup data keys: {list(our_side.keys())}")

        # Check cumulativeScore
        cumulative = our_side.get('cumulativeScore', {})
        if cumulative:
            print(f"\ncumulativeScore keys: {list(cumulative.keys())}")

            # Check for statBySlot
            stat_by_slot = cumulative.get('statBySlot')
            if stat_by_slot:
                print(f"\n*** FOUND statBySlot ***")
                print(f"Type: {type(stat_by_slot)}")
                if isinstance(stat_by_slot, dict):
                    print("Contents:")
                    for slot_id, value in sorted(stat_by_slot.items(), key=lambda x: int(x[0])):
                        pos_name = SLOT_ID_TO_POSITION.get(int(slot_id), f'?{slot_id}')
                        print(f"  {pos_name} (slot {slot_id}): {value}")

            # Check scoreByStat
            score_by_stat = cumulative.get('scoreByStat', {})
            if score_by_stat:
                print(f"\nscoreByStat ({len(score_by_stat)} stats):")
                for stat_id, value in list(score_by_stat.items())[:10]:
                    print(f"  Stat {stat_id}: {value}")

        # Check rosterForCurrentScoringPeriod or similar
        for key in our_side.keys():
            if 'roster' in key.lower() or 'lineup' in key.lower():
                print(f"\nour_side.{key} exists (type: {type(our_side[key]).__name__})")

    # =========================================================================
    # 3. Sum up statBySlot across all completed matchups
    # =========================================================================
    print("\n" + "=" * 80)
    print("3. AGGREGATE statBySlot ACROSS ALL MATCHUPS")
    print("=" * 80)

    total_by_slot = {}
    matchups_with_slot_data = 0

    for matchup in completed_matchups:
        our_side = matchup.get('home') if matchup.get('home', {}).get('teamId') == team_id else matchup.get('away')
        cumulative = our_side.get('cumulativeScore', {})
        stat_by_slot = cumulative.get('statBySlot', {})

        if stat_by_slot:
            matchups_with_slot_data += 1
            for slot_id, value in stat_by_slot.items():
                slot_id = int(slot_id)
                if slot_id not in total_by_slot:
                    total_by_slot[slot_id] = 0
                # Value might be a count of games or a stat total
                if isinstance(value, (int, float)):
                    total_by_slot[slot_id] += value

    print(f"\nMatchups with statBySlot data: {matchups_with_slot_data}")

    if total_by_slot:
        print("\nAggregate statBySlot (sum across matchups):")
        for slot_id, value in sorted(total_by_slot.items()):
            pos_name = SLOT_ID_TO_POSITION.get(slot_id, f'?{slot_id}')
            print(f"  {pos_name} (slot {slot_id}): {value}")

    # =========================================================================
    # 4. Check rosterSettings for any tracking
    # =========================================================================
    print("\n" + "=" * 80)
    print("4. ROSTER SETTINGS")
    print("=" * 80)

    settings = data.get('settings', {})
    roster_settings = settings.get('rosterSettings', {})

    print(f"\nrosterSettings keys: {list(roster_settings.keys())}")

    for key in roster_settings.keys():
        if 'stat' in key.lower() or 'count' in key.lower() or 'limit' in key.lower():
            value = roster_settings[key]
            print(f"\n{key}:")
            if isinstance(value, dict):
                for k, v in sorted(value.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0):
                    if isinstance(v, dict):
                        print(f"  Slot {k}: {v}")
                    else:
                        pos_name = SLOT_ID_TO_POSITION.get(int(k), f'?{k}') if k.isdigit() else k
                        print(f"  {pos_name}: {v}")
            else:
                print(f"  {value}")

    # =========================================================================
    # 5. Search for any field containing "slot" and "stat"
    # =========================================================================
    print("\n" + "=" * 80)
    print("5. DEEP SEARCH FOR SLOT STAT FIELDS")
    print("=" * 80)

    results = search_for_stat_by_slot(data)
    print(f"\nFound {len(results)} potential matches:")
    for r in results[:20]:
        print(f"\n  Path: {r['path']}")
        print(f"  Key: {r['key']}")
        print(f"  Type: {r['type']}")
        if isinstance(r['value'], dict) and len(r['value']) < 20:
            print(f"  Value: {r['value']}")
        elif isinstance(r['value'], list):
            print(f"  Value: list with {len(r['value'])} items")

    # =========================================================================
    # 6. Check if there's a separate scoring period endpoint
    # =========================================================================
    print("\n" + "=" * 80)
    print("6. CHECK SCORING PERIOD DATA")
    print("=" * 80)

    # Try fetching with mLiveScoring which might have detailed slot data
    endpoint = (
        f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/fba/"
        f"seasons/{season}/segments/0/leagues/{league_id}"
    )

    params = {'view': ['mLiveScoring', 'mMatchupScore']}
    cookies = {'espn_s2': espn_s2, 'SWID': swid}

    response = requests.get(endpoint, params=params, cookies=cookies, timeout=30)
    live_data = response.json()

    if 'teams' in live_data:
        for team in live_data['teams']:
            if team.get('id') == team_id:
                print(f"\nmLiveScoring team keys: {list(team.keys())}")

                # Check for slot-specific data
                for key in team.keys():
                    if 'slot' in key.lower() or 'lineup' in key.lower():
                        print(f"  {key}: {team[key]}")

    # =========================================================================
    # 7. Count starts from historical rosters (if available)
    # =========================================================================
    print("\n" + "=" * 80)
    print("7. HISTORICAL ROSTER ANALYSIS")
    print("=" * 80)

    # Check if matchups contain roster snapshots
    if completed_matchups:
        matchup = completed_matchups[-1]  # Most recent
        our_side = matchup.get('home') if matchup.get('home', {}).get('teamId') == team_id else matchup.get('away')

        roster_for_period = our_side.get('rosterForCurrentScoringPeriod', {})
        roster_for_matchup = our_side.get('rosterForMatchupPeriod', {})

        if roster_for_period:
            print("\nrosterForCurrentScoringPeriod structure:")
            print(f"  Keys: {list(roster_for_period.keys())}")
            entries = roster_for_period.get('entries', [])
            if entries:
                print(f"  Entries: {len(entries)} players")
                entry = entries[0]
                print(f"  Entry keys: {list(entry.keys())}")
                print(f"  Sample entry lineupSlotId: {entry.get('lineupSlotId')}")

        if roster_for_matchup:
            print("\nrosterForMatchupPeriod structure:")
            print(f"  Keys: {list(roster_for_matchup.keys())}")

    # =========================================================================
    # 8. Try fetching historical matchups with roster data
    # =========================================================================
    print("\n" + "=" * 80)
    print("8. FETCH HISTORICAL MATCHUP WITH ROSTERS")
    print("=" * 80)

    # Fetch a specific matchup period with roster data
    matchup_period = 1  # First week
    endpoint = (
        f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/fba/"
        f"seasons/{season}/segments/0/leagues/{league_id}"
    )

    params = {
        'view': ['mMatchup', 'mMatchupScore'],
        'scoringPeriodId': matchup_period
    }
    cookies = {'espn_s2': espn_s2, 'SWID': swid}

    response = requests.get(endpoint, params=params, cookies=cookies, timeout=30)
    historical = response.json()

    hist_schedule = historical.get('schedule', [])
    print(f"\nHistorical matchups for period {matchup_period}: {len(hist_schedule)}")

    if hist_schedule:
        for matchup in hist_schedule:
            home = matchup.get('home', {})
            away = matchup.get('away', {})

            if home.get('teamId') == team_id or away.get('teamId') == team_id:
                our_side = home if home.get('teamId') == team_id else away

                print(f"\nMatchup period {matchup_period} - our team data keys:")
                print(f"  {list(our_side.keys())}")

                # Check for roster with slot assignments
                roster = our_side.get('rosterForMatchupPeriod', {}) or our_side.get('rosterForCurrentScoringPeriod', {})
                if roster:
                    entries = roster.get('entries', [])
                    print(f"\n  Roster entries: {len(entries)}")

                    # Count by slot
                    slot_counts = {}
                    for entry in entries:
                        slot_id = entry.get('lineupSlotId', -1)
                        pos_name = SLOT_ID_TO_POSITION.get(slot_id, f'?{slot_id}')
                        slot_counts[pos_name] = slot_counts.get(pos_name, 0) + 1

                    print(f"  Slots used in period {matchup_period}:")
                    for pos, count in sorted(slot_counts.items()):
                        print(f"    {pos}: {count}")
                break

    print("\n" + "=" * 80)
    print("DEBUG COMPLETE")
    print("=" * 80)

    return 0


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
"""
Comprehensive Debug Script for Dashboard Issues

This script investigates three issues:
1. Identical Current vs Projected Standings
2. Zero Starts Used
3. Wrong Drop Decision (Scottie Barnes)

Run with: python3 debug_dashboard_full.py
"""

import os
import sys
import json
from datetime import date, datetime, timedelta
from collections import defaultdict

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

# =============================================================================
# Get Configuration from Database
# =============================================================================

def get_league_config():
    """Get league configuration from database."""
    from backend.app import create_app
    from backend.models import League

    app = create_app()
    with app.app_context():
        # Get the first (or only) league from database
        league = League.query.first()
        if not league:
            print("ERROR: No league found in database!")
            print("Please set up a league first via the web interface.")
            sys.exit(1)

        return {
            'league_id': int(league.espn_league_id),
            'season': league.season,
            'espn_s2': league.espn_s2_cookie,
            'swid': league.swid_cookie,
            'league_name': league.league_name,
        }

def get_team_ids(espn_client):
    """Get your team ID and Team Campbell's ID."""
    teams = espn_client.get_teams()

    your_team_id = None
    campbell_team_id = None

    for team in teams:
        team_name = team.get('team_name', '').lower()

        # Find your team (marked as user's team)
        if team.get('is_user_team'):
            your_team_id = team['espn_team_id']
            print(f"  Found your team: {team.get('team_name')} (ID: {your_team_id})")

        # Find Team Campbell
        if 'campbell' in team_name:
            campbell_team_id = team['espn_team_id']
            print(f"  Found Team Campbell: {team.get('team_name')} (ID: {campbell_team_id})")

    # Fallback: use SWID matching if is_user_team not set
    if your_team_id is None:
        your_team_id = espn_client.get_user_team_id()
        if your_team_id:
            team_name = next((t.get('team_name') for t in teams if t['espn_team_id'] == your_team_id), 'Unknown')
            print(f"  Found your team via SWID: {team_name} (ID: {your_team_id})")

    # If still no your_team_id, use first team
    if your_team_id is None:
        your_team_id = teams[0]['espn_team_id']
        print(f"  Using first team as yours: {teams[0].get('team_name')} (ID: {your_team_id})")

    # If no Campbell, use second team
    if campbell_team_id is None:
        for team in teams:
            if team['espn_team_id'] != your_team_id:
                campbell_team_id = team['espn_team_id']
                print(f"  Using as comparison team: {team.get('team_name')} (ID: {campbell_team_id})")
                break

    return your_team_id, campbell_team_id

# Get config from database
print("Loading league configuration from database...")
config = get_league_config()

LEAGUE_ID = config['league_id']
SEASON = config['season']
ESPN_S2 = config['espn_s2']
SWID = config['swid']

print(f"  League: {config['league_name']}")
print(f"  League ID: {LEAGUE_ID}")
print(f"  Season: {SEASON}")

if not ESPN_S2 or not SWID:
    print("ERROR: ESPN_S2 and SWID not found in database!")
    sys.exit(1)

# =============================================================================
# Helper Functions
# =============================================================================

def print_header(title):
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)

def print_subheader(title):
    print("\n" + "-" * 60)
    print(f" {title}")
    print("-" * 60)

def print_table_row(cols, widths):
    row = ""
    for col, width in zip(cols, widths):
        if isinstance(col, float):
            row += f"{col:>{width}.2f} | "
        else:
            row += f"{str(col):<{width}} | "
    print(row)

# =============================================================================
# ISSUE 1: Identical Current vs Projected Standings
# =============================================================================

def investigate_standings_issue(your_team_id, other_team_id):
    print_header("ISSUE 1: IDENTICAL CURRENT VS PROJECTED STANDINGS")

    from backend.services.espn_client import ESPNClient
    from backend.projections.start_limit_optimizer import StartLimitOptimizer
    from backend.projections.hybrid_engine import HybridProjectionEngine

    # Initialize clients
    print("\n[1.1] Initializing ESPN client and projection engines...")
    espn_client = ESPNClient(
        league_id=LEAGUE_ID,
        year=SEASON,
        espn_s2=ESPN_S2,
        swid=SWID
    )

    try:
        hybrid_engine = HybridProjectionEngine()
        print("  ✓ Hybrid engine initialized")
    except Exception as e:
        print(f"  ✗ Hybrid engine failed: {e}")
        hybrid_engine = None

    try:
        optimizer = StartLimitOptimizer(
            espn_s2=ESPN_S2,
            swid=SWID,
            league_id=LEAGUE_ID,
            season=SEASON,
            verbose=True
        )
        print("  ✓ Start limit optimizer initialized")
    except Exception as e:
        print(f"  ✗ Start limit optimizer failed: {e}")
        optimizer = None

    # Get all teams and rosters
    print("\n[1.2] Fetching teams and rosters...")
    teams = espn_client.get_teams()
    all_rosters = espn_client.get_all_rosters()
    standings = espn_client.get_standings()

    print(f"  Found {len(teams)} teams")
    print(f"  Fetched rosters for {len(all_rosters)} teams")

    # Get scoring categories
    categories = espn_client.get_scoring_categories()
    print(f"  Scoring categories: {[c.get('abbr', c.get('key', '?')) for c in categories]}")

    # Analyze specific teams
    teams_to_analyze = [your_team_id, other_team_id]

    for team_id in teams_to_analyze:
        team = next((t for t in teams if t['espn_team_id'] == team_id), None)
        if not team:
            print(f"\n[ERROR] Team {team_id} not found!")
            continue

        team_name = team.get('team_name', f'Team {team_id}')
        roster = all_rosters.get(team_id, [])

        print_subheader(f"ANALYZING: {team_name} (ID: {team_id})")

        # Current stats from ESPN
        print("\n[1.3] CURRENT STATS FROM ESPN:")
        current_stats = team.get('stats', {})
        print(f"  Raw stats dict keys: {list(current_stats.keys())}")

        for cat in categories:
            stat_key = cat.get('stat_key', cat.get('key', ''))
            abbr = cat.get('abbr', stat_key)
            value = current_stats.get(stat_key, 'N/A')
            print(f"    {abbr}: {value}")

        # Check if roster has stats
        print(f"\n[1.4] ROSTER CHECK ({len(roster)} players):")
        players_with_stats = 0
        players_with_per_game = 0

        for player in roster[:5]:  # Show first 5 players
            name = player.get('name', 'Unknown')
            stats = player.get('stats', {})
            per_game = player.get('per_game_stats', {})
            injury = player.get('injury_status', 'ACTIVE')
            lineup_slot = player.get('lineupSlotId', -1)

            has_stats = bool(stats.get('current_season', {}).get('games_played', 0))
            has_per_game = bool(per_game)

            if has_stats:
                players_with_stats += 1
            if has_per_game:
                players_with_per_game += 1

            print(f"    {name[:25]:<25} | Slot: {lineup_slot:>2} | Injury: {injury or 'ACTIVE':<12} | "
                  f"Has stats: {has_stats} | Has per_game: {has_per_game}")

            if per_game:
                pts = per_game.get('pts', 0)
                reb = per_game.get('reb', 0)
                print(f"      Per-game: PTS={pts:.1f}, REB={reb:.1f}")

        print(f"\n  Total: {players_with_stats}/{len(roster)} players with stats, "
              f"{players_with_per_game}/{len(roster)} with per_game_stats")

        # Run optimizer for this team
        if optimizer and roster:
            print(f"\n[1.5] RUNNING OPTIMIZER FOR {team_name}:")

            # Build roster for optimizer
            optimizer_roster = []
            for player in roster:
                per_game = player.get('per_game_stats', {})
                projected_games = 30  # Estimate

                projected_stats = {}
                for key, val in per_game.items():
                    projected_stats[key.upper()] = val * projected_games

                optimizer_roster.append({
                    'player_id': player.get('espn_player_id', 0),
                    'name': player.get('name', 'Unknown'),
                    'nba_team': player.get('nba_team', 'UNK'),
                    'eligible_slots': player.get('eligible_slots', []),
                    'projected_stats': projected_stats,
                    'projected_games': projected_games,
                    'lineupSlotId': player.get('lineupSlotId', 0),
                    'injury_status': player.get('injury_status', 'ACTIVE'),
                    'droppable': player.get('droppable', True),
                })

            try:
                adjusted_totals, assignments, ir_players = optimizer.optimize_team_projections(
                    team_id=team_id,
                    team_name=team_name,
                    roster=optimizer_roster,
                    categories=categories,
                    include_ir_returns=True
                )

                print("\n  ROS PROJECTIONS (from optimizer):")
                for key, val in sorted(adjusted_totals.items()):
                    if key in ['FG%', 'FT%']:
                        print(f"    {key}: {val:.3f}")
                    else:
                        print(f"    {key}: {val:.1f}")

                print(f"\n  IR Players returning: {len(ir_players)}")
                for ir in ir_players:
                    print(f"    - {ir.player_name}: returns {ir.projected_return_date}, "
                          f"replacing {ir.replacing_player_name}")

                # Calculate EOS totals
                print("\n  END-OF-SEASON TOTALS (Current + ROS):")
                eos_totals = {}
                for cat in categories:
                    stat_key = cat.get('stat_key', cat.get('key', ''))
                    current = current_stats.get(stat_key, 0) or 0
                    ros = adjusted_totals.get(stat_key, 0) or 0
                    eos = current + ros
                    eos_totals[stat_key] = eos

                    if stat_key in ['FG%', 'FT%']:
                        print(f"    {stat_key}: {current:.3f} + {ros:.3f} = {eos:.3f}")
                    else:
                        print(f"    {stat_key}: {current:.0f} + {ros:.0f} = {eos:.0f}")

                # Check if EOS equals current (bad sign!)
                all_equal = True
                for cat in categories:
                    stat_key = cat.get('stat_key', cat.get('key', ''))
                    if stat_key not in ['FG%', 'FT%']:
                        ros = adjusted_totals.get(stat_key, 0) or 0
                        if ros > 0:
                            all_equal = False
                            break

                if all_equal:
                    print("\n  ⚠️  WARNING: ALL ROS PROJECTIONS ARE ZERO!")
                    print("  This means EOS = Current, so projected standings = current standings")

            except Exception as e:
                print(f"  ✗ Optimizer failed: {e}")
                import traceback
                traceback.print_exc()

        print("")

    # Check if optimizer is called for all teams
    print_subheader("CHECKING OPTIMIZER COVERAGE")
    print(f"Total teams: {len(teams)}")
    print(f"Teams with rosters: {len(all_rosters)}")

    missing_rosters = [t['espn_team_id'] for t in teams if t['espn_team_id'] not in all_rosters]
    if missing_rosters:
        print(f"⚠️  Teams missing rosters: {missing_rosters}")
    else:
        print("✓ All teams have rosters")


# =============================================================================
# ISSUE 2: Zero Starts Used
# =============================================================================

def investigate_starts_used_issue(your_team_id):
    print_header("ISSUE 2: ZERO STARTS USED")

    from backend.projections.start_limit_optimizer import StartLimitOptimizer
    import requests

    print("\n[2.1] Initializing optimizer...")
    optimizer = StartLimitOptimizer(
        espn_s2=ESPN_S2,
        swid=SWID,
        league_id=LEAGUE_ID,
        season=SEASON,
        verbose=True
    )

    # Method 1: Using optimizer's fetch_starts_used
    print(f"\n[2.2] Fetching starts_used via optimizer.fetch_starts_used({your_team_id})...")
    starts_used = optimizer.fetch_starts_used(your_team_id)
    print(f"  Result: {starts_used}")

    if not starts_used or all(v == 0 for v in starts_used.values()):
        print("  ⚠️  WARNING: starts_used is empty or all zeros!")
    else:
        print("  ✓ starts_used has non-zero values")

    # Method 2: Direct ESPN API call (same as test_start_limits.py)
    print(f"\n[2.3] Direct ESPN API call for comparison...")

    endpoint = f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/fba/seasons/{SEASON}/segments/0/leagues/{LEAGUE_ID}"
    params = {'view': ['mTeam', 'mRoster']}
    cookies = {'espn_s2': ESPN_S2, 'SWID': SWID}

    try:
        response = requests.get(endpoint, params=params, cookies=cookies, timeout=15)
        response.raise_for_status()
        data = response.json()

        # Find our team
        for team_data in data.get('teams', []):
            if team_data.get('id') == your_team_id:
                team_name = team_data.get('name', f'Team {your_team_id}')
                print(f"\n  Found team: {team_name}")

                # Check for starts/limits in team data
                roster_data = team_data.get('roster', {})
                print(f"\n  Roster keys: {list(roster_data.keys())}")

                # Check for lineupSettings or stat limits
                lineup_settings = team_data.get('lineupSettings', {})
                print(f"  Lineup settings: {lineup_settings}")

                # Look for starts used per position
                entries = roster_data.get('entries', [])
                print(f"\n  Roster entries: {len(entries)} players")

                # Check first few entries for slot info
                for entry in entries[:3]:
                    lineup_slot_id = entry.get('lineupSlotId', -1)
                    player = entry.get('playerPoolEntry', {}).get('player', {})
                    name = player.get('fullName', 'Unknown')
                    print(f"    {name}: lineupSlotId={lineup_slot_id}")

                break

        # Check league settings for stat limits
        settings = data.get('settings', {})
        roster_settings = settings.get('rosterSettings', {})
        lineup_slot_counts = roster_settings.get('lineupSlotCounts', {})
        lineup_slot_stat_limits = roster_settings.get('lineupSlotStatLimits', {})

        print(f"\n  League lineup slot counts: {lineup_slot_counts}")
        print(f"  League stat limits: {lineup_slot_stat_limits}")

        # Check for acquisition data that might have starts info
        for team_data in data.get('teams', []):
            if team_data.get('id') == your_team_id:
                # Look for any field containing 'start' or 'games'
                print(f"\n  Looking for starts data in team object...")
                for key, value in team_data.items():
                    key_lower = key.lower()
                    if 'start' in key_lower or 'game' in key_lower or 'stat' in key_lower:
                        print(f"    {key}: {value}")
                break

    except Exception as e:
        print(f"  ✗ Direct API call failed: {e}")
        import traceback
        traceback.print_exc()

    # Method 3: Check what dashboard is receiving
    print_subheader("CHECKING DASHBOARD'S STARTS_USED FETCH")

    print("\n[2.4] Simulating dashboard's optimizer initialization...")
    print(f"  Using: league_id={LEAGUE_ID}, season={SEASON}")

    # Get slot counts and limits
    slot_counts = optimizer.get_lineup_slot_counts()
    stat_limits = optimizer.get_lineup_slot_stat_limits()

    print(f"\n  Slot counts from optimizer: {slot_counts}")
    print(f"  Stat limits from optimizer: {stat_limits}")

    # Check if starts_used is being calculated correctly
    print("\n[2.5] Analyzing starts calculation...")

    # The issue might be in how starts are being tracked
    # Let's check the actual roster for lineupSlotId values
    from backend.services.espn_client import ESPNClient
    espn_client = ESPNClient(
        league_id=LEAGUE_ID,
        year=SEASON,
        espn_s2=ESPN_S2,
        swid=SWID
    )

    roster = espn_client.get_team_roster(your_team_id)
    print(f"\n  Your roster has {len(roster)} players")

    slot_distribution = defaultdict(int)
    for player in roster:
        slot_id = player.get('lineupSlotId', -1)
        slot_distribution[slot_id] += 1

    print(f"  Current slot distribution: {dict(slot_distribution)}")
    print(f"  (12=Bench, 13=IR)")


# =============================================================================
# ISSUE 3: Wrong Drop Decision (Scottie Barnes)
# =============================================================================

def investigate_drop_decision_issue(your_team_id):
    print_header("ISSUE 3: WRONG DROP DECISION (SCOTTIE BARNES)")

    from backend.services.espn_client import ESPNClient
    from backend.projections.start_limit_optimizer import StartLimitOptimizer

    print("\n[3.1] Initializing services...")
    espn_client = ESPNClient(
        league_id=LEAGUE_ID,
        year=SEASON,
        espn_s2=ESPN_S2,
        swid=SWID
    )

    optimizer = StartLimitOptimizer(
        espn_s2=ESPN_S2,
        swid=SWID,
        league_id=LEAGUE_ID,
        season=SEASON,
        verbose=True
    )

    # Get your roster
    print(f"\n[3.2] Fetching your roster (Team ID: {your_team_id})...")
    roster = espn_client.get_team_roster(your_team_id)
    print(f"  Found {len(roster)} players")

    # Find Scottie Barnes and IR players
    scottie = None
    ir_players = []
    active_players = []

    for player in roster:
        name = player.get('name', 'Unknown')
        lineup_slot = player.get('lineupSlotId', 0)

        if 'scottie' in name.lower() or 'barnes' in name.lower():
            scottie = player

        if lineup_slot == 13:  # IR slot
            ir_players.append(player)
        elif lineup_slot != 12:  # Not bench
            active_players.append(player)
        else:
            active_players.append(player)  # Include bench in active for analysis

    print(f"\n  IR players: {len(ir_players)}")
    for p in ir_players:
        print(f"    - {p.get('name')}: injury={p.get('injury_status')}")

    if scottie:
        print(f"\n  Scottie Barnes found:")
        print(f"    Position: {scottie.get('position')}")
        print(f"    Lineup slot: {scottie.get('lineupSlotId')}")
        print(f"    Injury status: {scottie.get('injury_status')}")

        per_game = scottie.get('per_game_stats', {})
        print(f"    Per-game stats:")
        for key, val in per_game.items():
            print(f"      {key}: {val:.2f}")

        current_season = scottie.get('stats', {}).get('current_season', {})
        games = current_season.get('games_played', 0)
        print(f"    Games played: {games}")
    else:
        print("\n  ⚠️  Scottie Barnes NOT FOUND in roster!")

    # Analyze all potential drop candidates
    print_subheader("DROP CANDIDATE ANALYSIS")

    categories = espn_client.get_scoring_categories()
    print(f"Scoring categories: {[c.get('abbr', c.get('key')) for c in categories]}")

    # Build optimizer roster
    print("\n[3.3] Building roster for optimizer...")
    optimizer_roster = []

    for player in roster:
        per_game = player.get('per_game_stats', {})
        games_played = player.get('stats', {}).get('current_season', {}).get('games_played', 0)

        # Estimate remaining games
        if player.get('lineupSlotId') == 13:  # IR
            projected_games = 20  # Assume some games after return
        else:
            projected_games = 30  # Estimate for active players

        projected_stats = {}
        for key, val in per_game.items():
            projected_stats[key.upper()] = val * projected_games

        optimizer_roster.append({
            'player_id': player.get('espn_player_id', 0),
            'name': player.get('name', 'Unknown'),
            'nba_team': player.get('nba_team', 'UNK'),
            'eligible_slots': player.get('eligible_slots', []),
            'projected_stats': projected_stats,
            'projected_games': projected_games,
            'per_game_stats': per_game,
            'lineupSlotId': player.get('lineupSlotId', 0),
            'injury_status': player.get('injury_status', 'ACTIVE'),
            'droppable': player.get('droppable', True),
        })

    # Show projections for each player
    print("\n[3.4] PLAYER PROJECTIONS (sorted by projected PTS):")
    print(f"  {'Name':<25} | {'Slot':>4} | {'GP':>4} | {'PTS':>8} | {'REB':>8} | {'AST':>8} | {'3PM':>8}")
    print("  " + "-" * 85)

    sorted_roster = sorted(optimizer_roster,
                          key=lambda x: x.get('projected_stats', {}).get('PTS', 0),
                          reverse=True)

    for p in sorted_roster:
        name = p['name'][:25]
        slot = p['lineupSlotId']
        pg = p.get('per_game_stats', {})
        proj = p.get('projected_stats', {})

        print(f"  {name:<25} | {slot:>4} | {p['projected_games']:>4} | "
              f"{proj.get('PTS', 0):>8.1f} | {proj.get('REB', 0):>8.1f} | "
              f"{proj.get('AST', 0):>8.1f} | {proj.get('3PM', 0):>8.1f}")

    # Check Scottie's projections specifically
    print_subheader("SCOTTIE BARNES PROJECTION ANALYSIS")

    scottie_opt = next((p for p in optimizer_roster if 'scottie' in p['name'].lower() or 'barnes' in p['name'].lower()), None)

    if scottie_opt:
        print(f"\nScottie Barnes optimizer data:")
        print(f"  Player ID: {scottie_opt['player_id']}")
        print(f"  Lineup Slot: {scottie_opt['lineupSlotId']} ({'IR' if scottie_opt['lineupSlotId'] == 13 else 'Active'})")
        print(f"  Projected games: {scottie_opt['projected_games']}")
        print(f"  Droppable: {scottie_opt['droppable']}")
        print(f"\n  Per-game stats:")
        for key, val in scottie_opt.get('per_game_stats', {}).items():
            print(f"    {key}: {val:.2f}")
        print(f"\n  Projected totals (per_game × games):")
        for key, val in scottie_opt.get('projected_stats', {}).items():
            print(f"    {key}: {val:.1f}")

        # Check if projections are zero
        total_proj = sum(scottie_opt.get('projected_stats', {}).values())
        if total_proj == 0:
            print("\n  ⚠️  WARNING: SCOTTIE'S PROJECTIONS ARE ALL ZERO!")
            print("  This would make him appear as lowest value!")

    # Run drop decision analysis if there are IR players
    if ir_players and optimizer:
        print_subheader("DROP DECISION SIMULATION")

        print(f"\n[3.5] Running optimizer with IR returns enabled...")

        try:
            adjusted_totals, assignments, ir_results = optimizer.optimize_team_projections(
                team_id=your_team_id,
                team_name="Your Team",
                roster=optimizer_roster,
                categories=categories,
                include_ir_returns=True
            )

            print(f"\n  IR players processed: {len(ir_results)}")
            for ir in ir_results:
                print(f"\n  IR Player: {ir.player_name}")
                print(f"    Will return: {ir.will_return_before_season_end}")
                print(f"    Return date: {ir.projected_return_date}")
                print(f"    Games after return: {ir.games_after_return}")
                print(f"    Replacing: {ir.replacing_player_name}")
                print(f"    Marginal value: {getattr(ir, 'marginal_value', 'N/A')}")

            # Show assignments
            print("\n  Player assignments after optimization:")
            for a in assignments:
                if isinstance(a, dict):
                    name = a.get('name', 'Unknown')[:25]
                    is_dropped = a.get('is_dropped', False)
                    is_ir = a.get('is_ir_return', False)
                    starts = a.get('actual_games_to_start', 0)

                    status = "DROPPED" if is_dropped else ("IR RETURN" if is_ir else "Active")
                    print(f"    {name:<25} | {status:<12} | Starts: {starts}")

        except Exception as e:
            print(f"  ✗ Optimizer failed: {e}")
            import traceback
            traceback.print_exc()

    # Verify all 8 categories are being used
    print_subheader("CATEGORY VERIFICATION")

    print(f"\n[3.6] Checking all 8 categories are used in calculations...")
    expected_cats = ['PTS', 'REB', 'AST', 'STL', 'BLK', '3PM', 'FG%', 'FT%']
    actual_cats = [c.get('stat_key', c.get('key', '')) for c in categories]

    print(f"  Expected: {expected_cats}")
    print(f"  Actual:   {actual_cats}")

    missing = set(expected_cats) - set(actual_cats)
    extra = set(actual_cats) - set(expected_cats)

    if missing:
        print(f"  ⚠️  MISSING categories: {missing}")
    if extra:
        print(f"  Additional categories: {extra}")
    if not missing:
        print("  ✓ All 8 standard categories present")


# =============================================================================
# Main
# =============================================================================

def main():
    from backend.services.espn_client import ESPNClient

    print("\n" + "=" * 80)
    print(" COMPREHENSIVE DASHBOARD DEBUG SCRIPT")
    print(f" League ID: {LEAGUE_ID} | Season: {SEASON}")
    print("=" * 80)

    # Initialize ESPN client to get team IDs
    print("\nIdentifying teams...")
    espn_client = ESPNClient(
        league_id=LEAGUE_ID,
        year=SEASON,
        espn_s2=ESPN_S2,
        swid=SWID
    )

    your_team_id, campbell_team_id = get_team_ids(espn_client)

    print(f"\n Your Team ID: {your_team_id}")
    print(f" Comparison Team ID (Campbell): {campbell_team_id}")
    print("=" * 80)

    try:
        investigate_standings_issue(your_team_id, campbell_team_id)
    except Exception as e:
        print(f"\n[ERROR] Issue 1 investigation failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        investigate_starts_used_issue(your_team_id)
    except Exception as e:
        print(f"\n[ERROR] Issue 2 investigation failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        investigate_drop_decision_issue(your_team_id)
    except Exception as e:
        print(f"\n[ERROR] Issue 3 investigation failed: {e}")
        import traceback
        traceback.print_exc()

    print_header("DEBUG COMPLETE")
    print("\nReview the output above to identify:")
    print("1. Whether ROS projections are being calculated (non-zero)")
    print("2. Whether starts_used is being fetched correctly")
    print("3. Whether Scottie Barnes has valid projections")
    print("4. Which step in the process is failing")


if __name__ == "__main__":
    main()

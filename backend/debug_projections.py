#!/usr/bin/env python3
"""
Debug script for testing the hybrid projection engine.

This script tests:
1. Connection to ESPN league
2. Fetching your team roster
3. Hybrid projection engine initialization
4. ML model loading
5. Statistical model functioning
6. Player projections with detailed output

Usage:
    cd backend
    source ../venv/bin/activate
    python debug_projections.py
"""

import os
import sys
import sqlite3
import json
from pprint import pprint

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def get_league_credentials():
    """Get league credentials from database."""
    db_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'instance',
        'fantasy_basketball.db'
    )

    if not os.path.exists(db_path):
        db_path = os.path.join(
            os.path.dirname(__file__),
            'instance',
            'fantasy_basketball.db'
        )

    print(f"Database path: {db_path}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT espn_league_id, espn_s2_cookie, swid_cookie, league_name
        FROM leagues LIMIT 1
    ''')
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    return {
        'espn_league_id': row[0],
        'espn_s2': row[1],
        'swid': row[2],
        'league_name': row[3],
    }


def main():
    print("=" * 70)
    print("HYBRID PROJECTION ENGINE DEBUG SCRIPT")
    print("=" * 70)
    print()

    # Step 1: Get credentials
    print("STEP 1: Loading league credentials from database...")
    print("-" * 50)
    credentials = get_league_credentials()

    if not credentials:
        print("ERROR: No league found in database")
        print("Please add a league first via the web interface")
        return 1

    print(f"  League: {credentials['league_name']}")
    print(f"  ESPN League ID: {credentials['espn_league_id']}")
    print()

    # Step 2: Connect to ESPN
    print("STEP 2: Connecting to ESPN...")
    print("-" * 50)
    try:
        from espn_api.basketball import League as ESPNLeague

        league_id = int(credentials['espn_league_id'])
        espn_s2 = credentials['espn_s2']
        swid = credentials['swid']

        # Get the season year from the database
        conn2 = sqlite3.connect(db_path)
        cursor2 = conn2.cursor()
        cursor2.execute('SELECT season FROM leagues WHERE espn_league_id = ?', (credentials['espn_league_id'],))
        season_row = cursor2.fetchone()
        conn2.close()
        season_year = season_row[0] if season_row else 2026

        espn_league = ESPNLeague(
            league_id=league_id,
            year=season_year,
            espn_s2=espn_s2,
            swid=swid
        )
        print(f"  SUCCESS: Connected to ESPN league: {espn_league.settings.name}")
        print()

    except Exception as e:
        print(f"  ERROR: Failed to connect to ESPN: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Step 3: Find your team (Doncic Donuts)
    print("STEP 3: Finding your team...")
    print("-" * 50)
    my_team = None
    for team in espn_league.teams:
        print(f"  Team: {team.team_name} (ID: {team.team_id})")
        if 'doncic' in team.team_name.lower() or 'donut' in team.team_name.lower():
            my_team = team
            print(f"    ^ This is your team!")

    if not my_team:
        print("  WARNING: Could not find 'Doncic Donuts', using first team")
        my_team = espn_league.teams[0]

    print(f"\n  Selected team: {my_team.team_name}")
    print(f"  Roster size: {len(my_team.roster)} players")
    print()

    # Step 4: Test hybrid projection engine initialization
    print("STEP 4: Initializing Hybrid Projection Engine...")
    print("-" * 50)

    hybrid_engine = None
    simple_engine = None

    # Try hybrid engine first
    try:
        from projections.hybrid_engine import (
            HybridProjectionEngine,
            HybridProjection,
            LeagueScoringSettings,
            LeagueType,
        )
        print("  Hybrid imports successful")

        hybrid_engine = HybridProjectionEngine()
        print("  HybridProjectionEngine initialized")

    except ImportError as e:
        print(f"  Hybrid engine import failed: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"  Hybrid engine initialization failed: {e}")
        import traceback
        traceback.print_exc()

    # Also try simple projection engine
    try:
        from projections.simple_projection import SimpleProjectionEngine
        simple_engine = SimpleProjectionEngine()
        print("  SimpleProjectionEngine initialized as fallback")
    except ImportError as e:
        print(f"  Simple engine import failed: {e}")
    except Exception as e:
        print(f"  Simple engine initialization failed: {e}")

    print()

    # Step 5: Check ML model status
    print("STEP 5: Checking ML Model Status...")
    print("-" * 50)

    if hybrid_engine:
        ml_model = hybrid_engine.ml_model
        print(f"  ML Model object: {type(ml_model)}")

        # Check counting models
        counting_models = getattr(ml_model, 'counting_models', {})
        print(f"  Counting models loaded: {len(counting_models)}")
        if counting_models:
            print(f"    Models: {list(counting_models.keys())}")

        # Check shooting models
        shooting_models = getattr(ml_model, 'shooting_models', {})
        print(f"  Shooting models loaded: {len(shooting_models)}")
        if shooting_models:
            print(f"    Models: {list(shooting_models.keys())}")

        # Check for trained models directory
        models_dir = getattr(ml_model, 'models_dir', None)
        print(f"  Models directory: {models_dir}")
        if models_dir and os.path.exists(models_dir):
            model_files = os.listdir(models_dir)
            print(f"    Files in directory: {model_files}")
        else:
            print(f"    Directory does not exist or not set")
    else:
        print("  Skipped - hybrid engine not available")

    print()

    # Step 6: Check statistical model
    print("STEP 6: Checking Statistical Model...")
    print("-" * 50)

    if hybrid_engine:
        stat_engine = hybrid_engine.stat_engine
        print(f"  Statistical engine object: {type(stat_engine)}")
        print(f"  Available attributes: {[a for a in dir(stat_engine) if not a.startswith('_')][:10]}...")
    else:
        print("  Skipped - hybrid engine not available")

    print()

    # Step 7: Test projections for roster players
    print("STEP 7: Testing Player Projections...")
    print("-" * 50)

    # ESPN stat name mapping
    ESPN_TO_ENGINE_MAP = {
        'PTS': 'pts', 'REB': 'trb', 'AST': 'ast', 'STL': 'stl',
        'BLK': 'blk', '3PM': '3p', 'FG%': 'fg_pct', 'FT%': 'ft_pct',
        'TO': 'tov', 'FGM': 'fgm', 'FTM': 'ftm', 'FGA': 'fga', 'FTA': 'fta',
        'MIN': 'mp', 'GP': 'g',
    }

    successful_projections = 0
    failed_projections = 0
    first_detailed_projection = None

    for i, player in enumerate(my_team.roster):
        player_name = player.name
        player_id = str(player.playerId)
        injury_status = getattr(player, 'injuryStatus', 'ACTIVE') or 'ACTIVE'

        print(f"\n  [{i+1}/{len(my_team.roster)}] {player_name}")
        print(f"      Player ID: {player_id}")
        print(f"      Injury Status: {injury_status}")

        # Get player stats
        player_stats = getattr(player, 'stats', {})
        print(f"      Stats periods available: {list(player_stats.keys())}")

        # Find current season stats
        season_stats = None
        games_played = 0
        for key in ['2026_total', '2025_total', 'total', '002026', '002025']:
            if key in player_stats and isinstance(player_stats[key], dict):
                stat_data = player_stats[key]
                if 'avg' in stat_data:
                    season_stats = stat_data['avg']
                    # Check for games played in multiple locations
                    games_played = (
                        stat_data.get('GP', 0) or
                        stat_data.get('applied_total', 0) or
                        season_stats.get('GP', 0) or
                        season_stats.get('G', 0) or
                        0
                    )
                    print(f"      Using stats from: {key}")
                    break

        if not season_stats:
            print(f"      WARNING: No season stats found")
            failed_projections += 1
            continue

        print(f"      Games played: {games_played}")
        print(f"      Sample stats: PTS={season_stats.get('PTS', 'N/A')}, REB={season_stats.get('REB', 'N/A')}, AST={season_stats.get('AST', 'N/A')}")

        # Convert to engine format
        engine_stats = {}
        for espn_name, engine_name in ESPN_TO_ENGINE_MAP.items():
            if espn_name in season_stats and season_stats[espn_name] is not None:
                engine_stats[engine_name] = float(season_stats[espn_name])

        print(f"      Converted stats: {list(engine_stats.keys())}")

        # Attempt projection - try hybrid first, then simple
        projection = None
        projection_method = None

        if hybrid_engine:
            try:
                player_data = {
                    'player_id': player_id,
                    'name': player_name,
                    'team': getattr(player, 'proTeam', 'UNK'),
                    'position': getattr(player, 'position', 'N/A'),
                    'games_played': int(games_played) if games_played else 0,
                }

                projection = hybrid_engine.project_player(
                    player_id=player_id,
                    player_data=player_data,
                    season_stats=engine_stats,
                    injury_status=injury_status,
                )
                projection_method = 'hybrid'

            except Exception as e:
                print(f"      Hybrid projection failed: {e}")

        # Fallback to simple projection
        if projection is None and simple_engine:
            try:
                projection = simple_engine.project_player(
                    player_id=player_id,
                    player_name=player_name,
                    current_stats=season_stats,  # Use original ESPN format
                    games_played=int(games_played) if games_played else 0,
                    injury_status=injury_status,
                )
                projection_method = 'simple'

            except Exception as e:
                print(f"      Simple projection failed: {e}")

        if projection:
            print(f"      ✓ PROJECTION SUCCESS ({projection_method})")

            # Handle different projection types
            if hasattr(projection, 'games_projected'):
                print(f"        Games projected ROS: {projection.games_projected}")
            if hasattr(projection, 'confidence_score'):
                print(f"        Confidence score: {projection.confidence_score:.1f}")
            if hasattr(projection, 'season_phase'):
                print(f"        Season phase/tier: {projection.season_phase}")

            # Show key projected stats
            if hasattr(projection, 'projected_stats'):
                proj_stats = projection.projected_stats
            elif hasattr(projection, 'projected_per_game'):
                proj_stats = projection.projected_per_game
            else:
                proj_stats = {}

            if proj_stats:
                print(f"        Projected per-game:")
                for stat in ['pts', 'trb', 'ast', 'stl', 'blk', '3p']:
                    if stat in proj_stats:
                        print(f"          {stat.upper()}: {proj_stats[stat]:.2f}")

            # Show ROS totals
            ros_totals = projection.ros_totals if hasattr(projection, 'ros_totals') else {}
            if ros_totals:
                print(f"        ROS totals:")
                for stat in ['pts', 'trb', 'ast', 'stl', 'blk', '3p']:
                    if stat in ros_totals:
                        print(f"          {stat.upper()}: {ros_totals[stat]:.1f}")

            successful_projections += 1

            # Save first projection for detailed output
            if first_detailed_projection is None:
                first_detailed_projection = {
                    'player_name': player_name,
                    'projection': projection,
                    'input_stats': engine_stats,
                    'games_played': games_played,
                    'method': projection_method,
                }

        else:
            print(f"      ✗ PROJECTION FAILED - no engine available")
            failed_projections += 1

    print()

    # Step 8: Detailed projection output
    print("STEP 8: Detailed Projection Analysis...")
    print("-" * 50)

    if first_detailed_projection:
        proj = first_detailed_projection['projection']
        player_name = first_detailed_projection['player_name']
        method = first_detailed_projection.get('method', 'unknown')

        print(f"\n  Player: {player_name}")
        print(f"  Projection method: {method}")
        print(f"  Games played this season: {first_detailed_projection['games_played']}")
        print()

        print("  Input Stats (per-game):")
        for stat, value in sorted(first_detailed_projection['input_stats'].items()):
            print(f"    {stat}: {value:.3f}")
        print()

        # Get projected stats (different attribute name for simple vs hybrid)
        if hasattr(proj, 'projected_stats'):
            proj_stats = proj.projected_stats
        elif hasattr(proj, 'projected_per_game'):
            proj_stats = proj.projected_per_game
        else:
            proj_stats = {}

        print("  Projected Stats (per-game):")
        for stat, value in sorted(proj_stats.items()):
            if isinstance(value, (int, float)):
                print(f"    {stat}: {value:.3f}")
        print()

        print("  ROS Totals:")
        ros_totals = proj.ros_totals if hasattr(proj, 'ros_totals') else {}
        for stat, value in sorted(ros_totals.items()):
            if isinstance(value, (int, float)):
                print(f"    {stat}: {value:.1f}")
        print()

        # Hybrid-specific details
        if hasattr(proj, 'ml_weight'):
            print("  Projection Weights Used:")
            print(f"    ML weight: {proj.ml_weight:.2%}")
            print(f"    Statistical weight: {proj.statistical_weight:.2%}")
            if hasattr(proj, 'season_phase'):
                print(f"    Season phase/tier: {proj.season_phase}")
            print()

            print("  ML Contribution (weighted):")
            if hasattr(proj, 'ml_contribution') and proj.ml_contribution:
                for stat, value in sorted(proj.ml_contribution.items())[:5]:
                    print(f"    {stat}: {value:.3f}")
            else:
                print("    (No ML contribution - models may not be loaded)")
            print()

            print("  Statistical Contribution (weighted):")
            if hasattr(proj, 'statistical_contribution') and proj.statistical_contribution:
                for stat, value in sorted(proj.statistical_contribution.items())[:5]:
                    print(f"    {stat}: {value:.3f}")
            else:
                print("    (No statistical contribution)")
            print()

            print("  Confidence Intervals:")
            if hasattr(proj, 'confidence_intervals') and proj.confidence_intervals:
                for stat, interval in list(proj.confidence_intervals.items())[:5]:
                    if isinstance(interval, tuple) and len(interval) == 2:
                        print(f"    {stat}: [{interval[0]:.2f} - {interval[1]:.2f}]")
            print()

            print("  Fantasy Value:")
            if hasattr(proj, 'fantasy_points'):
                print(f"    Fantasy points/game: {proj.fantasy_points:.2f}")
            if hasattr(proj, 'fantasy_value_rank'):
                print(f"    Value rank: {proj.fantasy_value_rank}")
            print()

            print("  Category Values:")
            if hasattr(proj, 'category_values') and proj.category_values:
                for cat, val in sorted(proj.category_values.items(), key=lambda x: -x[1]):
                    print(f"    {cat}: {val:.2f}")
        else:
            print("  (Simple projection - limited details available)")
            print(f"  Confidence score: {proj.confidence_score if hasattr(proj, 'confidence_score') else 'N/A'}")

    else:
        print("  No successful projection to analyze in detail")

    print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  League: {credentials['league_name']}")
    print(f"  Team: {my_team.team_name}")
    print(f"  Roster size: {len(my_team.roster)} players")
    print()
    print(f"  Hybrid engine available: {hybrid_engine is not None}")
    if hybrid_engine:
        ml_model = hybrid_engine.ml_model
        print(f"  ML models loaded: {len(getattr(ml_model, 'counting_models', {})) + len(getattr(ml_model, 'shooting_models', {}))}")
    print()
    print(f"  Projections attempted: {len(my_team.roster)}")
    print(f"  Successful: {successful_projections}")
    print(f"  Failed: {failed_projections}")
    print()

    if successful_projections > 0:
        print("  ✓ Projection system is WORKING")
        if not (hybrid_engine and (getattr(hybrid_engine.ml_model, 'counting_models', {}) or getattr(hybrid_engine.ml_model, 'shooting_models', {}))):
            print("    Note: ML models not loaded - using statistical projections only")
    else:
        print("  ✗ Projection system has ISSUES")
        print("    Check the error messages above for details")

    print()
    return 0 if successful_projections > 0 else 1


if __name__ == '__main__':
    sys.exit(main())

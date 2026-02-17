#!/usr/bin/env python3
"""
Fantasy Basketball Projections Test Script

Tests the hybrid projection engine by:
1. Connecting to your ESPN league
2. Fetching your team's roster
3. Running projections for each player (with tiered weighting system)
4. Displaying projected stats and fantasy values

Projection sources by games played:
- 0-5 games:   60% ESPN proj, 30% prev season, 0% current, 10% ML
- 6-15 games:  35% ESPN proj, 20% prev season, 35% current, 10% ML
- 16-35 games: 15% ESPN proj, 10% prev season, 70% current, 5% ML
- 35+ games:   100% current season stats

Usage:
    python backend/test_projections.py

You will be prompted for your ESPN credentials (league ID, season, cookies).
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.services.espn_client import (
    ESPNClient,
    ESPNAuthenticationError,
    ESPNLeagueNotFoundError,
    ESPNConnectionError,
    ESPNClientError,
)
from backend.projections.hybrid_engine import (
    HybridProjectionEngine,
    LeagueScoringSettings,
    LeagueType,
)

# Try to import season stats service for database lookup
try:
    from backend.services.season_stats_service import SeasonStatsService
    SEASON_STATS_AVAILABLE = True
except ImportError:
    SEASON_STATS_AVAILABLE = False


# =============================================================================
# Display Helpers
# =============================================================================

def print_banner(text: str, char: str = '=', width: int = 80):
    """Print a banner."""
    print()
    print(char * width)
    print(f"  {text}")
    print(char * width)


def print_section(text: str, char: str = '-', width: int = 60):
    """Print a section header."""
    print()
    print(f"{char * 3} {text} {char * 3}")


def get_input(prompt: str, default: str = None) -> str:
    """Get input from user with optional default value."""
    if default:
        result = input(f"{prompt} [{default}]: ").strip()
        return result if result else default
    return input(f"{prompt}: ").strip()


def format_stat(value, decimals: int = 1) -> str:
    """Format a stat value for display."""
    if value is None:
        return "N/A"
    if isinstance(value, float):
        if abs(value) < 1 and value != 0:
            return f"{value:.3f}"  # For percentages
        return f"{value:.{decimals}f}"
    return str(value)


# =============================================================================
# Main Test Function
# =============================================================================

def main():
    """Run projections test."""
    print_banner("FANTASY BASKETBALL PROJECTION TEST")
    print()
    print("This script will:")
    print("  1. Connect to your ESPN league")
    print("  2. Find your team")
    print("  3. Run projections for each player on your roster")
    print("  4. Display projected stats and fantasy values")
    print()
    print("You'll need:")
    print("  - Your ESPN league ID (from the URL)")
    print("  - ESPN_S2 cookie")
    print("  - SWID cookie")
    print()
    print("To find cookies: ESPN.com → DevTools (F12) → Application → Cookies")
    print()

    # Get credentials
    print_section("Enter ESPN Credentials")
    print()

    try:
        league_id = int(get_input("League ID"))
    except ValueError:
        print("\nError: League ID must be a number")
        return 1

    try:
        current_year = 2025
        year = int(get_input("Season year", str(current_year)))
    except ValueError:
        print("\nError: Season year must be a number")
        return 1

    print()
    print("ESPN_S2 cookie (paste the entire value):")
    espn_s2 = input().strip()
    if not espn_s2:
        print("\nError: ESPN_S2 cookie is required")
        return 1

    print()
    swid = get_input("SWID cookie (include curly braces)")
    if not swid:
        print("\nError: SWID cookie is required")
        return 1

    # Connect to ESPN
    print_section("Connecting to ESPN")

    try:
        client = ESPNClient(
            league_id=league_id,
            year=year,
            espn_s2=espn_s2,
            swid=swid
        )
        print("  ✓ Connected successfully!")

        # Get league info
        settings = client.get_league_settings()
        print(f"  League: {settings.get('name', 'Unknown')}")
        print(f"  Type: {settings.get('scoring_type', 'Unknown')}")

    except ESPNAuthenticationError as e:
        print(f"\n  ✗ Authentication failed: {e}")
        print("  Check your ESPN_S2 and SWID cookies")
        return 1
    except ESPNLeagueNotFoundError as e:
        print(f"\n  ✗ League not found: {e}")
        return 1
    except ESPNConnectionError as e:
        print(f"\n  ✗ Connection error: {e}")
        return 1

    # List teams and ask user to select theirs
    print_section("Select Your Team")

    teams = client.get_teams()
    for i, team in enumerate(teams, 1):
        print(f"  {i:2}. {team['team_name']:<30} ({team.get('owner_name', 'Unknown')})")

    print()
    try:
        team_num = int(get_input("Enter your team number (1-" + str(len(teams)) + ")"))
        if team_num < 1 or team_num > len(teams):
            print("Invalid team number")
            return 1
        my_team = teams[team_num - 1]
    except ValueError:
        print("Invalid input")
        return 1

    print(f"\n  Selected: {my_team['team_name']}")

    # Get roster
    print_section("Fetching Roster")

    roster = client.get_team_roster(my_team['espn_team_id'])
    print(f"  Found {len(roster)} players on roster")

    # Initialize projection engine
    print_section("Loading Projection Engine")

    try:
        engine = HybridProjectionEngine()
        print("  ✓ Hybrid projection engine loaded")
        print(f"  ✓ ML models: {len(engine.ml_model.counting_models)} counting + {len(engine.ml_model.shooting_models)} shooting")
    except Exception as e:
        print(f"  ⚠ Could not load projection engine: {e}")
        print("  Will use simplified projections")
        engine = None

    # Initialize season stats service for previous season lookup
    season_stats_service = None
    if SEASON_STATS_AVAILABLE:
        try:
            season_stats_service = SeasonStatsService()
            available_seasons = season_stats_service.get_available_seasons()
            if available_seasons:
                print(f"  ✓ Previous season stats available: {available_seasons}")
            else:
                print("  ⚠ No previous season stats in database")
        except Exception as e:
            print(f"  ⚠ Could not initialize season stats service: {e}")

    # Create league scoring settings
    scoring_type = settings.get('scoring_type', 'H2H_CATEGORY')
    if 'POINTS' in scoring_type.upper():
        league_settings = LeagueScoringSettings.default_points()
        is_points_league = True
    else:
        league_settings = LeagueScoringSettings.default_h2h_category()
        is_points_league = False

    # Run projections for each player
    print_banner("ROSTER PROJECTIONS")

    projections = []

    for player in roster:
        player_name = player['name']
        player_id = str(player['espn_player_id'])
        position = player.get('position', 'N/A')
        nba_team = player.get('nba_team', 'FA')
        injury_status = player.get('injury_status') or 'ACTIVE'
        injury_notes = player.get('injury_notes')  # May contain return date info

        # Extract stats from ESPN player data
        # ESPN format: stats['2026_total']['avg'] contains per-game averages
        current_season_avg = {}
        previous_season_avg = {}
        espn_projection_avg = {}

        if player.get('stats'):
            stats_data = player['stats']

            # Current season stats (2026 or current year)
            for period_key in ['2026_total', '2025_total', 'total']:
                if period_key in stats_data:
                    period_data = stats_data[period_key]
                    if isinstance(period_data, dict) and period_data.get('avg'):
                        current_season_avg = period_data['avg']
                        break

            # Previous season stats (2025 or previous year)
            for period_key in ['2025_total', '2024_total']:
                if period_key in stats_data and period_key not in ['2026_total']:
                    period_data = stats_data[period_key]
                    if isinstance(period_data, dict) and period_data.get('avg'):
                        # Only use if different from current season
                        if period_data['avg'] != current_season_avg:
                            previous_season_avg = period_data['avg']
                        break

            # ESPN projections
            for period_key in ['2026_projected', '2025_projected', 'projected']:
                if period_key in stats_data:
                    period_data = stats_data[period_key]
                    if isinstance(period_data, dict) and period_data.get('avg'):
                        espn_projection_avg = period_data['avg']
                        break

        # Build player data dict for projection engine
        games_played = current_season_avg.get('GP', 0) or 0
        player_data = {
            'player_id': player_id,
            'name': player_name,
            'team': nba_team,
            'position': position,
            'games_played': games_played,
            'age': 25,  # ESPN doesn't provide age in roster data
        }

        # Helper function to map ESPN stats to our format
        def map_espn_stats(avg_stats):
            if not avg_stats:
                return None
            return {
                'pts': avg_stats.get('PTS', avg_stats.get('PPG', 0)) or 0,
                'trb': avg_stats.get('REB', avg_stats.get('RPG', 0)) or 0,
                'ast': avg_stats.get('AST', avg_stats.get('APG', 0)) or 0,
                'stl': avg_stats.get('STL', avg_stats.get('SPG', 0)) or 0,
                'blk': avg_stats.get('BLK', avg_stats.get('BPG', 0)) or 0,
                'tov': avg_stats.get('TO', avg_stats.get('TOPG', 0)) or 0,
                '3p': avg_stats.get('3PM', avg_stats.get('3PG', 0)) or 0,
                'fg_pct': avg_stats.get('FG%', 0) or 0,
                'ft_pct': avg_stats.get('FT%', 0) or 0,
                'fta': avg_stats.get('FTA', avg_stats.get('FTAPG', 0)) or 0,
                'mp': avg_stats.get('MIN', avg_stats.get('MPG', 0)) or 0,
                'g': avg_stats.get('GP', 0) or 0,
            }

        # Map all stat sources
        season_stats = map_espn_stats(current_season_avg)
        previous_season_stats = map_espn_stats(previous_season_avg)
        espn_projection = map_espn_stats(espn_projection_avg)

        # If ESPN doesn't have previous season stats, try database
        if not previous_season_stats and season_stats_service:
            try:
                db_prev_stats = season_stats_service.get_previous_season_stats(
                    espn_player_id=int(player_id) if player_id.isdigit() else None,
                    player_name=player_name,
                    current_season=year
                )
                if db_prev_stats:
                    previous_season_stats = db_prev_stats
            except Exception:
                pass  # Silently fail - database lookup is optional

        # Run projection
        try:
            if engine:
                projection = engine.project_player(
                    player_id=player_id,
                    league_id=league_id,
                    player_data=player_data,
                    season_stats=season_stats,
                    previous_season_stats=previous_season_stats,
                    espn_projection=espn_projection,
                    injury_status=injury_status,
                    injury_notes=injury_notes,
                    league_settings=league_settings
                )
                projections.append({
                    'player': player,
                    'projection': projection,
                    'has_ml': True
                })
            else:
                # Fallback: just use current stats as projection
                projections.append({
                    'player': player,
                    'projection': None,
                    'season_stats': season_stats,
                    'has_ml': False
                })

        except Exception as e:
            print(f"  ⚠ Error projecting {player_name}: {e}")
            projections.append({
                'player': player,
                'projection': None,
                'season_stats': season_stats,
                'has_ml': False
            })

    # Display results
    print_section("Projected Stats (Per Game)")
    print()

    # Header
    if is_points_league:
        print(f"{'Player':<25} {'Pos':<5} {'Team':<4} {'PTS':<6} {'REB':<5} {'AST':<5} {'STL':<4} {'BLK':<4} {'3PM':<4} {'FPTS':<8}")
        print("-" * 80)
    else:
        print(f"{'Player':<25} {'Pos':<5} {'Team':<4} {'PTS':<6} {'REB':<5} {'AST':<5} {'STL':<4} {'BLK':<4} {'3PM':<4} {'FG%':<6} {'FT%':<6} {'Value':<7}")
        print("-" * 95)

    total_fpts = 0.0

    for proj_data in projections:
        player = proj_data['player']
        projection = proj_data['projection']
        player_name = player['name'][:24]
        position = player.get('position', 'N/A')[:4]
        nba_team = player.get('nba_team', 'FA')[:3]
        injury = player.get('injury_status')

        if injury and injury != 'ACTIVE':
            injury_marker = f" ({injury[:3]})"
            player_name = player_name[:20] + injury_marker

        if projection:
            stats = projection.projected_stats
            pts = format_stat(stats.get('pts'))
            reb = format_stat(stats.get('trb'))
            ast = format_stat(stats.get('ast'))
            stl = format_stat(stats.get('stl'))
            blk = format_stat(stats.get('blk'))
            tpm = format_stat(stats.get('3p'))
            fg_pct = format_stat(stats.get('fg_pct'), 3)
            ft_pct = format_stat(stats.get('ft_pct'), 3)
            fpts = projection.fantasy_points
            total_fpts += fpts

            if is_points_league:
                print(f"{player_name:<25} {position:<5} {nba_team:<4} {pts:<6} {reb:<5} {ast:<5} {stl:<4} {blk:<4} {tpm:<4} {fpts:<8.1f}")
            else:
                # Show category value for non-points leagues
                cat_value = sum(projection.category_values.values()) if projection.category_values else 0
                print(f"{player_name:<25} {position:<5} {nba_team:<4} {pts:<6} {reb:<5} {ast:<5} {stl:<4} {blk:<4} {tpm:<4} {fg_pct:<6} {ft_pct:<6} {cat_value:<7.1f}")
        else:
            # Fallback to season stats
            stats = proj_data.get('season_stats', {})
            pts = format_stat(stats.get('pts'))
            reb = format_stat(stats.get('trb'))
            ast = format_stat(stats.get('ast'))
            stl = format_stat(stats.get('stl'))
            blk = format_stat(stats.get('blk'))
            tpm = format_stat(stats.get('3p'))
            fg_pct = format_stat(stats.get('fg_pct'), 3)
            ft_pct = format_stat(stats.get('ft_pct'), 3)

            if is_points_league:
                print(f"{player_name:<25} {position:<5} {nba_team:<4} {pts:<6} {reb:<5} {ast:<5} {stl:<4} {blk:<4} {tpm:<4} {'N/A':<8}")
            else:
                print(f"{player_name:<25} {position:<5} {nba_team:<4} {pts:<6} {reb:<5} {ast:<5} {stl:<4} {blk:<4} {tpm:<4} {fg_pct:<6} {ft_pct:<6} {'N/A':<7}")

    # Summary
    print()
    print("-" * 80)
    if is_points_league:
        print(f"{'TEAM TOTAL FPTS/GAME:':<60} {total_fpts:.1f}")

    # Show detailed projections for top players
    print_section("Detailed Player Projections")

    # Sort by fantasy points
    sorted_projections = sorted(
        [p for p in projections if p['projection']],
        key=lambda x: x['projection'].fantasy_points,
        reverse=True
    )

    for i, proj_data in enumerate(sorted_projections[:5], 1):
        projection = proj_data['projection']
        player = proj_data['player']
        games_played = player_data.get('games_played', 0)

        print(f"\n  {i}. {projection.player_name}")
        print(f"     Position: {projection.position} | Team: {projection.team}")
        print(f"     Weight Tier: {projection.season_phase}")
        print(f"     Current Season Weight: {projection.statistical_weight*100:.0f}%")
        print(f"     ML Weight: {projection.ml_weight*100:.0f}%")
        print(f"     Confidence: {projection.confidence_score:.0f}%")
        print(f"     Games Remaining (Est): {projection.games_remaining}")
        print(f"     Games Projected: {projection.games_projected}")
        print()
        print(f"     Per-Game Projections:")
        # Stats to exclude from per-game display (not per-game metrics)
        exclude_stats = {'g', 'gs', 'mp'}
        for stat, value in sorted(projection.projected_stats.items()):
            if stat in exclude_stats:
                continue
            if value and value != 0:
                ci = projection.confidence_intervals.get(stat)
                if ci:
                    print(f"       {stat.upper():<6}: {value:.2f}  (90% CI: {ci[0]:.2f} - {ci[1]:.2f})")
                else:
                    print(f"       {stat.upper():<6}: {value:.2f}")
        print()
        print(f"     Fantasy Points/Game: {projection.fantasy_points:.1f}")
        if projection.ros_totals:
            ros_pts = projection.ros_totals.get('pts', 0) or 0
            ros_reb = projection.ros_totals.get('trb', 0) or 0
            ros_ast = projection.ros_totals.get('ast', 0) or 0
            print(f"     ROS Totals: {ros_pts:.0f} PTS, {ros_reb:.0f} REB, {ros_ast:.0f} AST")

    # Final summary
    print_banner("PROJECTION SUMMARY")
    print()
    print(f"  Team: {my_team['team_name']}")
    print(f"  Players: {len(roster)}")
    print(f"  Successfully Projected: {len([p for p in projections if p['projection']])}")
    print(f"  League Type: {settings.get('scoring_type', 'Unknown')}")
    if is_points_league:
        print(f"  Projected Team FPTS/Game: {total_fpts:.1f}")
    print()
    print("  ✓ Projection test complete!")
    print()

    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nCancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

#!/usr/bin/env python3
"""
End-of-Season Stats Capture Script.

Run this script at the end of each NBA regular season (mid-April) to capture
final player statistics for use in next season's projections.

Usage:
    python backend/scripts/capture_season_stats.py --season 2025

    # With custom minimum games threshold
    python backend/scripts/capture_season_stats.py --season 2025 --min-games 20

    # Check available seasons
    python backend/scripts/capture_season_stats.py --list-seasons

    # View season summary
    python backend/scripts/capture_season_stats.py --summary 2025
"""

import argparse
import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from datetime import datetime


def setup_app():
    """Set up Flask app context for database access."""
    from backend.app import create_app
    app = create_app()
    return app


def capture_stats(season: int, min_games: int = 10):
    """Capture end-of-season stats."""
    from backend.services.season_stats_service import SeasonStatsService

    print(f"\n{'='*60}")
    print(f"  END-OF-SEASON STATS CAPTURE")
    print(f"{'='*60}")
    print(f"\n  Season: {season}")
    print(f"  Minimum games: {min_games}")
    print(f"  Source: Basketball Reference")
    print(f"\n  Starting capture...")
    print()

    service = SeasonStatsService()

    try:
        result = service.capture_end_of_season_stats(
            season=season,
            min_games=min_games
        )

        print(f"\n{'='*60}")
        print(f"  CAPTURE COMPLETE")
        print(f"{'='*60}")
        print(f"\n  Stats captured: {result['stats_captured']}")
        print(f"  Players created: {result['players_created']}")
        print(f"  Players updated: {result['players_updated']}")

        if result['errors']:
            print(f"\n  Errors ({len(result['errors'])}):")
            for err in result['errors'][:10]:
                print(f"    - {err['player']}: {err['error']}")
            if len(result['errors']) > 10:
                print(f"    ... and {len(result['errors']) - 10} more")

        print(f"\n  Captured at: {result['captured_at']}")
        print()

        return 0

    except Exception as e:
        print(f"\n  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


def list_seasons():
    """List available seasons in database."""
    from backend.services.season_stats_service import SeasonStatsService

    service = SeasonStatsService()
    seasons = service.get_available_seasons()

    print(f"\n{'='*60}")
    print(f"  AVAILABLE SEASONS")
    print(f"{'='*60}")

    if not seasons:
        print("\n  No seasons found in database.")
        print("  Run capture_season_stats.py --season YYYY to add data.")
    else:
        print()
        for season in seasons:
            summary = service.get_season_summary(season)
            print(f"  {season}: {summary['total_players']} players")
            for source, count in summary['by_source'].items():
                print(f"         {source}: {count}")

    print()
    return 0


def show_summary(season: int):
    """Show summary for a specific season."""
    from backend.services.season_stats_service import SeasonStatsService

    service = SeasonStatsService()
    summary = service.get_season_summary(season)

    print(f"\n{'='*60}")
    print(f"  SEASON {season} SUMMARY")
    print(f"{'='*60}")
    print(f"\n  Total players: {summary['total_players']}")
    print(f"\n  By source:")
    for source, count in summary['by_source'].items():
        print(f"    {source}: {count}")
    print()

    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Capture end-of-season NBA player statistics.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Capture 2025 season stats
    python backend/scripts/capture_season_stats.py --season 2025

    # Capture with higher games threshold
    python backend/scripts/capture_season_stats.py --season 2025 --min-games 20

    # List available seasons
    python backend/scripts/capture_season_stats.py --list-seasons
        """
    )

    parser.add_argument(
        '--season', '-s',
        type=int,
        help='Season year to capture (e.g., 2025 for 2024-25 season)'
    )
    parser.add_argument(
        '--min-games', '-m',
        type=int,
        default=10,
        help='Minimum games played to include player (default: 10)'
    )
    parser.add_argument(
        '--list-seasons', '-l',
        action='store_true',
        help='List available seasons in database'
    )
    parser.add_argument(
        '--summary',
        type=int,
        metavar='SEASON',
        help='Show summary for a specific season'
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.season and not args.list_seasons and not args.summary:
        parser.print_help()
        print("\nError: Must specify --season, --list-seasons, or --summary")
        return 1

    # Set up Flask app context
    app = setup_app()

    with app.app_context():
        if args.list_seasons:
            return list_seasons()
        elif args.summary:
            return show_summary(args.summary)
        else:
            return capture_stats(args.season, args.min_games)


if __name__ == '__main__':
    sys.exit(main())

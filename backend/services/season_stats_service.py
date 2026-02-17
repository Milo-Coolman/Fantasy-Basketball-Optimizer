"""
Season Stats Service for Fantasy Basketball Optimizer.

This service handles:
- Capturing end-of-season stats for all NBA players
- Storing stats in the database for use in next season's projections
- Retrieving previous season stats for the tiered projection system

Usage:
    # Capture end-of-season stats
    from backend.services.season_stats_service import SeasonStatsService
    service = SeasonStatsService()
    service.capture_end_of_season_stats(season=2025)

    # Retrieve previous season stats for projection
    prev_stats = service.get_previous_season_stats(espn_player_id=12345, current_season=2026)
"""

import logging
from datetime import datetime, date
from typing import Any, Dict, List, Optional

from backend.extensions import db
from backend.models import Player, PlayerStats
from backend.scrapers.basketball_reference import BasketballReferenceScraper

logger = logging.getLogger(__name__)


class SeasonStatsService:
    """
    Service for managing season-end player statistics.

    Captures stats at end of each NBA season and provides retrieval
    for use in next season's projections.
    """

    def __init__(self):
        """Initialize the season stats service."""
        self.scraper = BasketballReferenceScraper()

    # =========================================================================
    # End of Season Capture
    # =========================================================================

    def capture_end_of_season_stats(
        self,
        season: int,
        source: str = 'BASKETBALL_REFERENCE',
        min_games: int = 10
    ) -> Dict[str, Any]:
        """
        Capture end-of-season stats for all NBA players.

        Should be run at the end of the NBA regular season (mid-April)
        to capture final stats for use in next season's projections.

        Args:
            season: Season year (e.g., 2025 for 2024-25 season)
            source: Data source ('BASKETBALL_REFERENCE' or 'ESPN')
            min_games: Minimum games played to include player

        Returns:
            Summary dict with counts of players captured
        """
        logger.info(f"Capturing end-of-season stats for {season} season...")

        stats_captured = 0
        players_created = 0
        players_updated = 0
        errors = []

        try:
            # Fetch all player stats from Basketball Reference
            if source == 'BASKETBALL_REFERENCE':
                all_stats = self._fetch_bbref_season_stats(season)
            else:
                raise ValueError(f"Unsupported source: {source}")

            logger.info(f"Fetched {len(all_stats)} player records from {source}")

            for player_stats in all_stats:
                try:
                    # Skip players with too few games
                    games = player_stats.get('g', 0) or 0
                    if games < min_games:
                        continue

                    # Get or create player
                    player, created = self._get_or_create_player(player_stats)
                    if created:
                        players_created += 1
                    else:
                        players_updated += 1

                    # Save season stats
                    self._save_player_season_stats(player, season, player_stats, source)
                    stats_captured += 1

                except Exception as e:
                    player_name = player_stats.get('player', 'Unknown')
                    logger.warning(f"Error processing {player_name}: {e}")
                    errors.append({'player': player_name, 'error': str(e)})

            db.session.commit()
            logger.info(f"Successfully captured {stats_captured} player season stats")

        except Exception as e:
            db.session.rollback()
            logger.error(f"Failed to capture season stats: {e}")
            raise

        return {
            'season': season,
            'source': source,
            'stats_captured': stats_captured,
            'players_created': players_created,
            'players_updated': players_updated,
            'errors': errors,
            'captured_at': datetime.utcnow().isoformat()
        }

    def _fetch_bbref_season_stats(self, season: int) -> List[Dict[str, Any]]:
        """Fetch season stats from Basketball Reference."""
        try:
            # Get per-game stats
            per_game_stats = self.scraper.get_per_game_stats(season)

            if not per_game_stats:
                logger.warning(f"No per-game stats found for {season}")
                return []

            return per_game_stats

        except Exception as e:
            logger.error(f"Error fetching BBRef stats: {e}")
            raise

    def _get_or_create_player(
        self,
        player_stats: Dict[str, Any]
    ) -> tuple:
        """
        Get existing player or create new one.

        Returns:
            Tuple of (Player, created_bool)
        """
        player_name = player_stats.get('player', '').strip()
        team = player_stats.get('tm', player_stats.get('team', ''))
        position = player_stats.get('pos', player_stats.get('position', ''))

        # Try to find by name (BBRef doesn't have ESPN IDs)
        player = Player.query.filter_by(name=player_name).first()

        if player:
            # Update team/position if changed
            if team and player.team != team:
                player.team = team
            if position and player.position != position:
                player.position = position
            player.last_updated = datetime.utcnow()
            return player, False

        # Create new player (generate a temporary ID based on name hash)
        # In production, you'd want to match with ESPN IDs
        import hashlib
        name_hash = int(hashlib.md5(player_name.encode()).hexdigest()[:8], 16)

        player = Player(
            espn_player_id=name_hash,  # Temporary ID
            name=player_name,
            team=team,
            position=position,
            last_updated=datetime.utcnow()
        )
        db.session.add(player)
        db.session.flush()  # Get the ID

        return player, True

    def _save_player_season_stats(
        self,
        player: Player,
        season: int,
        stats: Dict[str, Any],
        source: str
    ) -> PlayerStats:
        """Save or update player's season stats."""
        # Check for existing stats
        existing = PlayerStats.query.filter_by(
            player_id=player.id,
            season=season,
            source=source
        ).first()

        stat_date = date(season, 4, 15)  # End of regular season

        if existing:
            # Update existing record
            player_stats = existing
        else:
            # Create new record
            player_stats = PlayerStats(
                player_id=player.id,
                season=season,
                source=source,
                stat_date=stat_date
            )
            db.session.add(player_stats)

        # Map stats
        player_stats.games_played = stats.get('g', 0)
        player_stats.minutes_per_game = stats.get('mp', 0)
        player_stats.points = stats.get('pts', 0)
        player_stats.rebounds = stats.get('trb', stats.get('reb', 0))
        player_stats.assists = stats.get('ast', 0)
        player_stats.steals = stats.get('stl', 0)
        player_stats.blocks = stats.get('blk', 0)
        player_stats.turnovers = stats.get('tov', stats.get('to', 0))
        player_stats.field_goal_pct = stats.get('fg_pct', stats.get('fg%', 0))
        player_stats.free_throw_pct = stats.get('ft_pct', stats.get('ft%', 0))
        player_stats.three_pointers_made = stats.get('3p', stats.get('3pm', 0))
        player_stats.stat_date = stat_date

        return player_stats

    # =========================================================================
    # Previous Season Stats Retrieval
    # =========================================================================

    def get_previous_season_stats(
        self,
        espn_player_id: Optional[int] = None,
        player_name: Optional[str] = None,
        current_season: Optional[int] = None
    ) -> Optional[Dict[str, float]]:
        """
        Get previous season stats for a player.

        Used by the projection engine to get previous season data
        for the tiered weighting system.

        Args:
            espn_player_id: ESPN player ID
            player_name: Player name (fallback if no ESPN ID)
            current_season: Current season year (defaults to current year)

        Returns:
            Dict of per-game stats or None if not found
        """
        if current_season is None:
            current_season = datetime.now().year
            # Adjust for NBA season spanning two years
            if datetime.now().month < 10:
                current_season = current_season
            else:
                current_season = current_season + 1

        previous_season = current_season - 1

        # Find player
        player = None
        if espn_player_id:
            player = Player.query.filter_by(espn_player_id=espn_player_id).first()
        if not player and player_name:
            player = Player.query.filter_by(name=player_name).first()

        if not player:
            logger.debug(f"Player not found: {espn_player_id or player_name}")
            return None

        # Get previous season stats
        stats = PlayerStats.query.filter_by(
            player_id=player.id,
            season=previous_season
        ).order_by(PlayerStats.stat_date.desc()).first()

        if not stats:
            logger.debug(f"No {previous_season} stats for {player.name}")
            return None

        # Convert to projection format
        return {
            'pts': float(stats.points) if stats.points else 0,
            'trb': float(stats.rebounds) if stats.rebounds else 0,
            'ast': float(stats.assists) if stats.assists else 0,
            'stl': float(stats.steals) if stats.steals else 0,
            'blk': float(stats.blocks) if stats.blocks else 0,
            'tov': float(stats.turnovers) if stats.turnovers else 0,
            '3p': float(stats.three_pointers_made) if stats.three_pointers_made else 0,
            'fg_pct': float(stats.field_goal_pct) if stats.field_goal_pct else 0,
            'ft_pct': float(stats.free_throw_pct) if stats.free_throw_pct else 0,
            'mp': float(stats.minutes_per_game) if stats.minutes_per_game else 0,
            'g': stats.games_played or 0,
        }

    def get_previous_season_stats_bulk(
        self,
        player_identifiers: List[Dict[str, Any]],
        current_season: Optional[int] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Get previous season stats for multiple players.

        Args:
            player_identifiers: List of dicts with 'espn_player_id' and/or 'name'
            current_season: Current season year

        Returns:
            Dict mapping player_id (str) to stats dict
        """
        results = {}

        for identifier in player_identifiers:
            player_id = str(identifier.get('espn_player_id', identifier.get('name', '')))
            stats = self.get_previous_season_stats(
                espn_player_id=identifier.get('espn_player_id'),
                player_name=identifier.get('name'),
                current_season=current_season
            )
            if stats:
                results[player_id] = stats

        return results

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_available_seasons(self) -> List[int]:
        """Get list of seasons with stored stats."""
        seasons = db.session.query(PlayerStats.season).distinct().all()
        return sorted([s[0] for s in seasons], reverse=True)

    def get_season_summary(self, season: int) -> Dict[str, Any]:
        """Get summary of stored stats for a season."""
        total_players = PlayerStats.query.filter_by(season=season).count()

        sources = db.session.query(
            PlayerStats.source,
            db.func.count(PlayerStats.id)
        ).filter_by(season=season).group_by(PlayerStats.source).all()

        return {
            'season': season,
            'total_players': total_players,
            'by_source': {source: count for source, count in sources}
        }

    def delete_season_stats(self, season: int) -> int:
        """
        Delete all stats for a season.

        Use with caution - primarily for testing or data cleanup.

        Returns:
            Number of records deleted
        """
        deleted = PlayerStats.query.filter_by(season=season).delete()
        db.session.commit()
        logger.info(f"Deleted {deleted} stat records for season {season}")
        return deleted


# =============================================================================
# Convenience Functions
# =============================================================================

def capture_season_stats(season: int, min_games: int = 10) -> Dict[str, Any]:
    """
    Convenience function to capture end-of-season stats.

    Args:
        season: Season year
        min_games: Minimum games to include player

    Returns:
        Summary of capture results
    """
    service = SeasonStatsService()
    return service.capture_end_of_season_stats(season=season, min_games=min_games)


def get_player_previous_season(
    espn_player_id: int = None,
    player_name: str = None
) -> Optional[Dict[str, float]]:
    """
    Convenience function to get previous season stats.

    Args:
        espn_player_id: ESPN player ID
        player_name: Player name

    Returns:
        Stats dict or None
    """
    service = SeasonStatsService()
    return service.get_previous_season_stats(
        espn_player_id=espn_player_id,
        player_name=player_name
    )

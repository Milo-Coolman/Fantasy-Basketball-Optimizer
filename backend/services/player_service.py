"""
Player Database Service.

CRUD operations for players and their statistics.
Used by API endpoints to manage player data.
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, date
from decimal import Decimal

from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy import or_, func

from backend.extensions import db
from backend.models import Player, PlayerStats, Projection, Roster

logger = logging.getLogger(__name__)


# =============================================================================
# Custom Exceptions
# =============================================================================

class PlayerServiceError(Exception):
    """Base exception for player service errors."""
    pass


class PlayerNotFoundError(PlayerServiceError):
    """Raised when a player is not found."""
    pass


class PlayerAlreadyExistsError(PlayerServiceError):
    """Raised when trying to create a duplicate player."""
    pass


class StatsError(PlayerServiceError):
    """Raised for stats-related errors."""
    pass


# =============================================================================
# Player Create Operations
# =============================================================================

def create_player(
    espn_player_id: int,
    name: str,
    position: Optional[str] = None,
    nba_team: Optional[str] = None,
    injury_status: Optional[str] = None
) -> Player:
    """
    Create a new player.

    Args:
        espn_player_id: ESPN's player ID
        name: Player name
        position: Primary position (PG, SG, SF, PF, C)
        nba_team: NBA team abbreviation
        injury_status: Injury status if any

    Returns:
        Created Player object

    Raises:
        PlayerAlreadyExistsError: If player already exists
    """
    try:
        existing = Player.query.filter_by(espn_player_id=espn_player_id).first()
        if existing:
            raise PlayerAlreadyExistsError(
                f"Player with ESPN ID {espn_player_id} already exists"
            )

        player = Player(
            espn_player_id=espn_player_id,
            name=name,
            position=position,
            nba_team=nba_team,
            injury_status=injury_status
        )

        db.session.add(player)
        db.session.commit()

        logger.info(f"Created player {player.id}: {name}")
        return player

    except PlayerAlreadyExistsError:
        raise
    except IntegrityError:
        db.session.rollback()
        raise PlayerAlreadyExistsError(f"Player {espn_player_id} already exists")
    except SQLAlchemyError as e:
        db.session.rollback()
        logger.error(f"Error creating player: {e}")
        raise PlayerServiceError(f"Failed to create player: {e}")


def create_or_update_player(
    espn_player_id: int,
    name: str,
    position: Optional[str] = None,
    nba_team: Optional[str] = None,
    injury_status: Optional[str] = None
) -> Player:
    """
    Create a player or update if already exists.

    Args:
        espn_player_id: ESPN's player ID
        name: Player name
        position: Primary position
        nba_team: NBA team abbreviation
        injury_status: Injury status

    Returns:
        Player object (created or updated)
    """
    try:
        player = Player.query.filter_by(espn_player_id=espn_player_id).first()

        if player:
            # Update existing player
            player.name = name
            if position:
                player.position = position
            if nba_team is not None:
                player.nba_team = nba_team
            player.injury_status = injury_status
            logger.debug(f"Updated player {player.id}: {name}")
        else:
            # Create new player
            player = Player(
                espn_player_id=espn_player_id,
                name=name,
                position=position,
                nba_team=nba_team,
                injury_status=injury_status
            )
            db.session.add(player)
            logger.debug(f"Created new player: {name}")

        db.session.commit()
        return player

    except SQLAlchemyError as e:
        db.session.rollback()
        logger.error(f"Error creating/updating player: {e}")
        raise PlayerServiceError(f"Failed to create/update player: {e}")


def bulk_create_players(players_data: List[Dict[str, Any]]) -> List[Player]:
    """
    Create or update multiple players at once.

    Args:
        players_data: List of player data dictionaries

    Returns:
        List of Player objects
    """
    players = []

    try:
        for data in players_data:
            player = create_or_update_player(
                espn_player_id=data['espn_player_id'],
                name=data['name'],
                position=data.get('position'),
                nba_team=data.get('nba_team'),
                injury_status=data.get('injury_status')
            )
            players.append(player)

        return players

    except Exception as e:
        logger.error(f"Error bulk creating players: {e}")
        raise PlayerServiceError(f"Failed to bulk create players: {e}")


# =============================================================================
# Player Read Operations
# =============================================================================

def get_player_by_id(player_id: int) -> Optional[Player]:
    """
    Get a player by internal ID.

    Args:
        player_id: Player ID

    Returns:
        Player object or None
    """
    return Player.query.get(player_id)


def get_player_or_404(player_id: int) -> Player:
    """
    Get a player by ID or raise an error.

    Args:
        player_id: Player ID

    Returns:
        Player object

    Raises:
        PlayerNotFoundError: If player doesn't exist
    """
    player = get_player_by_id(player_id)
    if not player:
        raise PlayerNotFoundError(f"Player {player_id} not found")
    return player


def get_player_by_espn_id(espn_player_id: int) -> Optional[Player]:
    """
    Get a player by ESPN ID.

    Args:
        espn_player_id: ESPN player ID

    Returns:
        Player object or None
    """
    return Player.query.filter_by(espn_player_id=espn_player_id).first()


def search_players(
    query: str,
    position: Optional[str] = None,
    team: Optional[str] = None,
    limit: int = 20
) -> List[Player]:
    """
    Search for players by name.

    Args:
        query: Search query (partial name match)
        position: Filter by position
        team: Filter by NBA team
        limit: Maximum results

    Returns:
        List of matching Player objects
    """
    filters = [Player.name.ilike(f"%{query}%")]

    if position:
        filters.append(Player.position == position)
    if team:
        filters.append(Player.nba_team == team)

    return Player.query.filter(*filters).limit(limit).all()


def get_players_by_team(nba_team: str) -> List[Player]:
    """
    Get all players on an NBA team.

    Args:
        nba_team: NBA team abbreviation

    Returns:
        List of Player objects
    """
    return Player.query.filter_by(nba_team=nba_team).all()


def get_players_by_position(position: str) -> List[Player]:
    """
    Get all players at a position.

    Args:
        position: Position (PG, SG, SF, PF, C)

    Returns:
        List of Player objects
    """
    return Player.query.filter_by(position=position).all()


def get_injured_players() -> List[Player]:
    """
    Get all players with injury status.

    Returns:
        List of injured Player objects
    """
    return Player.query.filter(
        Player.injury_status.isnot(None),
        Player.injury_status != ''
    ).all()


def get_all_players(limit: int = 500, offset: int = 0) -> List[Player]:
    """
    Get all players with pagination.

    Args:
        limit: Maximum results
        offset: Starting offset

    Returns:
        List of Player objects
    """
    return Player.query.order_by(Player.name).offset(offset).limit(limit).all()


# =============================================================================
# Player Update Operations
# =============================================================================

def update_player(player_id: int, **kwargs) -> Player:
    """
    Update player properties.

    Args:
        player_id: Player ID
        **kwargs: Fields to update

    Returns:
        Updated Player object
    """
    player = get_player_or_404(player_id)

    allowed_fields = {'name', 'position', 'nba_team', 'injury_status'}

    try:
        for key, value in kwargs.items():
            if key in allowed_fields:
                setattr(player, key, value)

        db.session.commit()
        logger.info(f"Updated player {player_id}")
        return player

    except SQLAlchemyError as e:
        db.session.rollback()
        logger.error(f"Error updating player {player_id}: {e}")
        raise PlayerServiceError(f"Failed to update player: {e}")


def update_player_injury(espn_player_id: int, injury_status: Optional[str]) -> Player:
    """
    Update a player's injury status.

    Args:
        espn_player_id: ESPN player ID
        injury_status: New injury status (None for healthy)

    Returns:
        Updated Player object
    """
    player = get_player_by_espn_id(espn_player_id)
    if not player:
        raise PlayerNotFoundError(f"Player with ESPN ID {espn_player_id} not found")

    try:
        player.injury_status = injury_status
        db.session.commit()
        return player

    except SQLAlchemyError as e:
        db.session.rollback()
        logger.error(f"Error updating injury status: {e}")
        raise PlayerServiceError(f"Failed to update injury status: {e}")


# =============================================================================
# Player Delete Operations
# =============================================================================

def delete_player(player_id: int) -> bool:
    """
    Delete a player and associated data.

    Args:
        player_id: Player ID

    Returns:
        True if deleted
    """
    player = get_player_or_404(player_id)

    try:
        # Delete associated stats
        PlayerStats.query.filter_by(player_id=player_id).delete()

        # Delete from rosters
        Roster.query.filter_by(player_id=player_id).delete()

        # Delete projections
        Projection.query.filter_by(player_id=player_id).delete()

        db.session.delete(player)
        db.session.commit()

        logger.info(f"Deleted player {player_id}")
        return True

    except SQLAlchemyError as e:
        db.session.rollback()
        logger.error(f"Error deleting player {player_id}: {e}")
        raise PlayerServiceError(f"Failed to delete player: {e}")


# =============================================================================
# Player Stats Operations
# =============================================================================

def add_player_stats(
    player_id: int,
    season: int,
    source: str = 'ESPN',
    stat_date: Optional[date] = None,
    **stats
) -> PlayerStats:
    """
    Add or update stats for a player.

    Args:
        player_id: Player ID
        season: Season year
        source: Data source (ESPN, BASKETBALL_REFERENCE)
        stat_date: Date of stats (None for season totals)
        **stats: Stat values (games_played, pts, reb, ast, etc.)

    Returns:
        PlayerStats object
    """
    try:
        # Check for existing stats entry
        existing = PlayerStats.query.filter_by(
            player_id=player_id,
            season=season,
            source=source,
            stat_date=stat_date
        ).first()

        if existing:
            # Update existing stats
            for key, value in stats.items():
                if hasattr(existing, key) and value is not None:
                    setattr(existing, key, value)
            db.session.commit()
            return existing

        # Create new stats entry
        stat_entry = PlayerStats(
            player_id=player_id,
            season=season,
            source=source,
            stat_date=stat_date,
            **{k: v for k, v in stats.items() if v is not None}
        )

        db.session.add(stat_entry)
        db.session.commit()

        logger.debug(f"Added stats for player {player_id}, season {season}")
        return stat_entry

    except SQLAlchemyError as e:
        db.session.rollback()
        logger.error(f"Error adding player stats: {e}")
        raise StatsError(f"Failed to add stats: {e}")


def get_player_stats(
    player_id: int,
    season: Optional[int] = None,
    source: Optional[str] = None
) -> List[PlayerStats]:
    """
    Get stats for a player.

    Args:
        player_id: Player ID
        season: Optional season filter
        source: Optional source filter

    Returns:
        List of PlayerStats objects
    """
    query = PlayerStats.query.filter_by(player_id=player_id)

    if season:
        query = query.filter_by(season=season)
    if source:
        query = query.filter_by(source=source)

    return query.order_by(PlayerStats.season.desc()).all()


def get_player_season_stats(player_id: int, season: int) -> Optional[PlayerStats]:
    """
    Get season total stats for a player.

    Args:
        player_id: Player ID
        season: Season year

    Returns:
        PlayerStats object or None
    """
    return PlayerStats.query.filter_by(
        player_id=player_id,
        season=season,
        stat_date=None  # Season totals have no specific date
    ).first()


def get_player_game_stats(
    player_id: int,
    season: int,
    limit: int = 10
) -> List[PlayerStats]:
    """
    Get recent game stats for a player.

    Args:
        player_id: Player ID
        season: Season year
        limit: Number of games

    Returns:
        List of PlayerStats objects (most recent first)
    """
    return PlayerStats.query.filter(
        PlayerStats.player_id == player_id,
        PlayerStats.season == season,
        PlayerStats.stat_date.isnot(None)
    ).order_by(PlayerStats.stat_date.desc()).limit(limit).all()


def bulk_add_stats(stats_data: List[Dict[str, Any]]) -> int:
    """
    Bulk add stats for multiple players.

    Args:
        stats_data: List of stats dictionaries with player_id

    Returns:
        Number of stats entries created/updated
    """
    count = 0

    try:
        for data in stats_data:
            player_id = data.pop('player_id')
            season = data.pop('season')
            source = data.pop('source', 'ESPN')
            stat_date = data.pop('stat_date', None)

            add_player_stats(
                player_id=player_id,
                season=season,
                source=source,
                stat_date=stat_date,
                **data
            )
            count += 1

        return count

    except Exception as e:
        logger.error(f"Error bulk adding stats: {e}")
        raise StatsError(f"Failed to bulk add stats: {e}")


# =============================================================================
# Projection Operations
# =============================================================================

def add_projection(
    player_id: int,
    league_id: int,
    season: int,
    projection_type: str = 'HYBRID',
    confidence: Optional[float] = None,
    **projected_stats
) -> Projection:
    """
    Add or update a projection for a player.

    Args:
        player_id: Player ID
        league_id: League ID
        season: Season year
        projection_type: Type (STATISTICAL, ML, HYBRID)
        confidence: Confidence score (0-1)
        **projected_stats: Projected stat values

    Returns:
        Projection object
    """
    try:
        existing = Projection.query.filter_by(
            player_id=player_id,
            league_id=league_id,
            season=season,
            projection_type=projection_type
        ).first()

        if existing:
            for key, value in projected_stats.items():
                if hasattr(existing, key) and value is not None:
                    setattr(existing, key, value)
            if confidence is not None:
                existing.confidence = Decimal(str(confidence))
            existing.created_at = datetime.utcnow()
            db.session.commit()
            return existing

        projection = Projection(
            player_id=player_id,
            league_id=league_id,
            season=season,
            projection_type=projection_type,
            confidence=Decimal(str(confidence)) if confidence else None,
            **{k: v for k, v in projected_stats.items() if v is not None}
        )

        db.session.add(projection)
        db.session.commit()

        logger.debug(f"Added projection for player {player_id}")
        return projection

    except SQLAlchemyError as e:
        db.session.rollback()
        logger.error(f"Error adding projection: {e}")
        raise PlayerServiceError(f"Failed to add projection: {e}")


def get_player_projections(
    player_id: int,
    league_id: Optional[int] = None,
    season: Optional[int] = None
) -> List[Projection]:
    """
    Get projections for a player.

    Args:
        player_id: Player ID
        league_id: Optional league filter
        season: Optional season filter

    Returns:
        List of Projection objects
    """
    query = Projection.query.filter_by(player_id=player_id)

    if league_id:
        query = query.filter_by(league_id=league_id)
    if season:
        query = query.filter_by(season=season)

    return query.all()


def get_league_projections(league_id: int, season: int) -> List[Dict[str, Any]]:
    """
    Get all player projections for a league.

    Args:
        league_id: League ID
        season: Season year

    Returns:
        List of projection data with player info
    """
    projections = Projection.query.filter_by(
        league_id=league_id,
        season=season
    ).all()

    result = []
    for proj in projections:
        player = proj.player
        result.append({
            'player_id': proj.player_id,
            'player_name': player.name if player else 'Unknown',
            'position': player.position if player else None,
            'projection_type': proj.projection_type,
            'projected_pts': float(proj.projected_pts) if proj.projected_pts else None,
            'projected_reb': float(proj.projected_reb) if proj.projected_reb else None,
            'projected_ast': float(proj.projected_ast) if proj.projected_ast else None,
            'projected_stl': float(proj.projected_stl) if proj.projected_stl else None,
            'projected_blk': float(proj.projected_blk) if proj.projected_blk else None,
            'fantasy_value': float(proj.fantasy_value) if proj.fantasy_value else None,
            'confidence': float(proj.confidence) if proj.confidence else None,
        })

    return result


# =============================================================================
# Utility Functions
# =============================================================================

def get_player_count() -> int:
    """Get total number of players in database."""
    return Player.query.count()


def get_players_without_stats(season: int) -> List[Player]:
    """
    Get players who don't have stats for a season.

    Args:
        season: Season year

    Returns:
        List of Player objects without stats
    """
    subquery = db.session.query(PlayerStats.player_id).filter(
        PlayerStats.season == season
    ).subquery()

    return Player.query.filter(
        ~Player.id.in_(subquery)
    ).all()


def get_player_with_stats(player_id: int, season: int) -> Dict[str, Any]:
    """
    Get a player with their stats.

    Args:
        player_id: Player ID
        season: Season year

    Returns:
        Dictionary with player info and stats
    """
    player = get_player_or_404(player_id)
    stats = get_player_season_stats(player_id, season)

    return {
        'player': player.to_dict(),
        'stats': stats.to_dict() if stats else None
    }

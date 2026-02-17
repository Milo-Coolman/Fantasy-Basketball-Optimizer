"""
League Database Service.

CRUD operations for leagues and league settings.
Used by API endpoints to manage user leagues.
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from backend.extensions import db
from backend.models import League, Team, User

logger = logging.getLogger(__name__)


# =============================================================================
# Custom Exceptions
# =============================================================================

class LeagueServiceError(Exception):
    """Base exception for league service errors."""
    pass


class LeagueNotFoundError(LeagueServiceError):
    """Raised when a league is not found."""
    pass


class LeagueAlreadyExistsError(LeagueServiceError):
    """Raised when trying to create a duplicate league."""
    pass


class LeagueAccessDeniedError(LeagueServiceError):
    """Raised when user doesn't have access to a league."""
    pass


# =============================================================================
# Create Operations
# =============================================================================

def create_league(
    user_id: int,
    espn_league_id: int,
    season: int,
    espn_s2: str,
    swid: str,
    league_name: Optional[str] = None,
    league_type: str = 'H2H_CATEGORY',
    num_teams: Optional[int] = None,
    roster_settings: Optional[Dict] = None,
    scoring_settings: Optional[Dict] = None
) -> League:
    """
    Create a new league for a user.

    Args:
        user_id: User ID who owns this league
        espn_league_id: ESPN's league ID
        season: Season year (e.g., 2025)
        espn_s2: ESPN_S2 cookie for authentication
        swid: SWID cookie for authentication
        league_name: Optional custom league name
        league_type: League type (H2H_CATEGORY, H2H_POINTS, ROTO)
        num_teams: Number of teams in the league
        roster_settings: JSON roster configuration
        scoring_settings: JSON scoring configuration

    Returns:
        Created League object

    Raises:
        LeagueAlreadyExistsError: If league already exists for this user/season
        LeagueServiceError: For other database errors
    """
    try:
        # Check for existing league
        existing = League.query.filter_by(
            user_id=user_id,
            espn_league_id=espn_league_id,
            season=season
        ).first()

        if existing:
            raise LeagueAlreadyExistsError(
                f"League {espn_league_id} for season {season} already exists"
            )

        league = League(
            user_id=user_id,
            espn_league_id=espn_league_id,
            season=season,
            league_name=league_name or f"League {espn_league_id}",
            espn_s2_cookie=espn_s2,
            swid_cookie=swid,
            league_type=league_type,
            num_teams=num_teams,
            roster_settings=roster_settings or {},
            scoring_settings=scoring_settings or {},
            last_updated=datetime.utcnow()
        )

        db.session.add(league)
        db.session.commit()

        logger.info(f"Created league {league.id} for user {user_id}")
        return league

    except LeagueAlreadyExistsError:
        raise
    except IntegrityError as e:
        db.session.rollback()
        logger.error(f"Integrity error creating league: {e}")
        raise LeagueAlreadyExistsError("League already exists")
    except SQLAlchemyError as e:
        db.session.rollback()
        logger.error(f"Database error creating league: {e}")
        raise LeagueServiceError(f"Failed to create league: {e}")


# =============================================================================
# Read Operations
# =============================================================================

def get_league_by_id(league_id: int, user_id: Optional[int] = None) -> Optional[League]:
    """
    Get a league by its ID.

    Args:
        league_id: League ID
        user_id: Optional user ID to verify ownership

    Returns:
        League object or None if not found

    Raises:
        LeagueAccessDeniedError: If user doesn't own the league
    """
    league = League.query.get(league_id)

    if league and user_id and league.user_id != user_id:
        raise LeagueAccessDeniedError("You don't have access to this league")

    return league


def get_league_or_404(league_id: int, user_id: Optional[int] = None) -> League:
    """
    Get a league by ID or raise an error.

    Args:
        league_id: League ID
        user_id: Optional user ID to verify ownership

    Returns:
        League object

    Raises:
        LeagueNotFoundError: If league doesn't exist
        LeagueAccessDeniedError: If user doesn't own the league
    """
    league = get_league_by_id(league_id, user_id)

    if not league:
        raise LeagueNotFoundError(f"League {league_id} not found")

    return league


def get_user_leagues(user_id: int) -> List[League]:
    """
    Get all leagues for a user.

    Args:
        user_id: User ID

    Returns:
        List of League objects
    """
    return League.query.filter_by(user_id=user_id).order_by(League.season.desc()).all()


def get_league_by_espn_id(
    user_id: int,
    espn_league_id: int,
    season: int
) -> Optional[League]:
    """
    Get a league by ESPN ID and season.

    Args:
        user_id: User ID
        espn_league_id: ESPN league ID
        season: Season year

    Returns:
        League object or None
    """
    return League.query.filter_by(
        user_id=user_id,
        espn_league_id=espn_league_id,
        season=season
    ).first()


def get_leagues_needing_refresh(hours_since_update: int = 24) -> List[League]:
    """
    Get leagues that haven't been updated recently.

    Args:
        hours_since_update: Hours since last update threshold

    Returns:
        List of leagues needing refresh
    """
    from datetime import timedelta

    cutoff = datetime.utcnow() - timedelta(hours=hours_since_update)

    return League.query.filter(
        (League.last_updated < cutoff) | (League.last_updated.is_(None))
    ).all()


# =============================================================================
# Update Operations
# =============================================================================

def update_league(
    league_id: int,
    user_id: int,
    **kwargs
) -> League:
    """
    Update league properties.

    Args:
        league_id: League ID
        user_id: User ID (for ownership verification)
        **kwargs: Fields to update

    Returns:
        Updated League object

    Raises:
        LeagueNotFoundError: If league doesn't exist
        LeagueAccessDeniedError: If user doesn't own the league
    """
    league = get_league_or_404(league_id, user_id)

    # Allowed fields to update
    allowed_fields = {
        'league_name', 'league_type', 'num_teams',
        'roster_settings', 'scoring_settings',
        'espn_s2_cookie', 'swid_cookie', 'refresh_schedule'
    }

    try:
        for key, value in kwargs.items():
            if key in allowed_fields and value is not None:
                setattr(league, key, value)

        db.session.commit()
        logger.info(f"Updated league {league_id}")
        return league

    except SQLAlchemyError as e:
        db.session.rollback()
        logger.error(f"Error updating league {league_id}: {e}")
        raise LeagueServiceError(f"Failed to update league: {e}")


def update_league_settings(
    league_id: int,
    user_id: int,
    league_name: Optional[str] = None,
    league_type: Optional[str] = None,
    num_teams: Optional[int] = None,
    roster_settings: Optional[Dict] = None,
    scoring_settings: Optional[Dict] = None
) -> League:
    """
    Update league settings from ESPN data.

    Args:
        league_id: League ID
        user_id: User ID
        league_name: League name
        league_type: League type
        num_teams: Number of teams
        roster_settings: Roster configuration
        scoring_settings: Scoring configuration

    Returns:
        Updated League object
    """
    league = get_league_or_404(league_id, user_id)

    try:
        if league_name:
            league.league_name = league_name
        if league_type:
            league.league_type = league_type
        if num_teams:
            league.num_teams = num_teams
        if roster_settings:
            league.roster_settings = roster_settings
        if scoring_settings:
            league.scoring_settings = scoring_settings

        league.last_updated = datetime.utcnow()
        db.session.commit()

        logger.info(f"Updated settings for league {league_id}")
        return league

    except SQLAlchemyError as e:
        db.session.rollback()
        logger.error(f"Error updating league settings: {e}")
        raise LeagueServiceError(f"Failed to update league settings: {e}")


def update_league_cookies(
    league_id: int,
    user_id: int,
    espn_s2: str,
    swid: str
) -> League:
    """
    Update ESPN authentication cookies for a league.

    Args:
        league_id: League ID
        user_id: User ID
        espn_s2: New ESPN_S2 cookie
        swid: New SWID cookie

    Returns:
        Updated League object
    """
    league = get_league_or_404(league_id, user_id)

    try:
        league.espn_s2_cookie = espn_s2
        league.swid_cookie = swid
        db.session.commit()

        logger.info(f"Updated cookies for league {league_id}")
        return league

    except SQLAlchemyError as e:
        db.session.rollback()
        logger.error(f"Error updating league cookies: {e}")
        raise LeagueServiceError(f"Failed to update cookies: {e}")


def mark_league_updated(league_id: int) -> None:
    """
    Mark a league as recently updated.

    Args:
        league_id: League ID
    """
    try:
        league = League.query.get(league_id)
        if league:
            league.last_updated = datetime.utcnow()
            db.session.commit()
    except SQLAlchemyError as e:
        db.session.rollback()
        logger.error(f"Error marking league updated: {e}")


# =============================================================================
# Delete Operations
# =============================================================================

def delete_league(league_id: int, user_id: int) -> bool:
    """
    Delete a league and all associated data.

    Args:
        league_id: League ID
        user_id: User ID (for ownership verification)

    Returns:
        True if deleted successfully

    Raises:
        LeagueNotFoundError: If league doesn't exist
        LeagueAccessDeniedError: If user doesn't own the league
    """
    league = get_league_or_404(league_id, user_id)

    try:
        # Delete associated teams (cascade should handle this, but be explicit)
        Team.query.filter_by(league_id=league_id).delete()

        db.session.delete(league)
        db.session.commit()

        logger.info(f"Deleted league {league_id}")
        return True

    except SQLAlchemyError as e:
        db.session.rollback()
        logger.error(f"Error deleting league {league_id}: {e}")
        raise LeagueServiceError(f"Failed to delete league: {e}")


# =============================================================================
# Utility Functions
# =============================================================================

def get_league_with_teams(league_id: int, user_id: int) -> Dict[str, Any]:
    """
    Get a league with all its teams.

    Args:
        league_id: League ID
        user_id: User ID

    Returns:
        Dictionary with league and teams data
    """
    league = get_league_or_404(league_id, user_id)

    return {
        'league': league.to_dict(),
        'teams': [team.to_dict() for team in league.teams]
    }


def get_league_credentials(league_id: int, user_id: int) -> Dict[str, str]:
    """
    Get ESPN credentials for a league.

    Args:
        league_id: League ID
        user_id: User ID

    Returns:
        Dictionary with espn_s2 and swid
    """
    league = get_league_or_404(league_id, user_id)

    return {
        'espn_league_id': league.espn_league_id,
        'season': league.season,
        'espn_s2': league.espn_s2_cookie,
        'swid': league.swid_cookie
    }


def count_user_leagues(user_id: int) -> int:
    """
    Count the number of leagues for a user.

    Args:
        user_id: User ID

    Returns:
        Number of leagues
    """
    return League.query.filter_by(user_id=user_id).count()

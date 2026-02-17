"""
Team Database Service.

CRUD operations for teams and rosters.
Used by API endpoints to manage fantasy teams.
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
from decimal import Decimal

from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from backend.extensions import db
from backend.models import Team, Roster, Player, League

logger = logging.getLogger(__name__)


# =============================================================================
# Custom Exceptions
# =============================================================================

class TeamServiceError(Exception):
    """Base exception for team service errors."""
    pass


class TeamNotFoundError(TeamServiceError):
    """Raised when a team is not found."""
    pass


class TeamAlreadyExistsError(TeamServiceError):
    """Raised when trying to create a duplicate team."""
    pass


class RosterError(TeamServiceError):
    """Raised for roster-related errors."""
    pass


# =============================================================================
# Team Create Operations
# =============================================================================

def create_team(
    league_id: int,
    espn_team_id: int,
    team_name: str,
    owner_name: Optional[str] = None,
    current_record: Optional[str] = None,
    current_standing: Optional[int] = None
) -> Team:
    """
    Create a new team in a league.

    Args:
        league_id: League ID
        espn_team_id: ESPN's team ID
        team_name: Team name
        owner_name: Owner's display name
        current_record: Current record (e.g., "10-5-0")
        current_standing: Current standing position

    Returns:
        Created Team object

    Raises:
        TeamAlreadyExistsError: If team already exists
        TeamServiceError: For other database errors
    """
    try:
        # Check for existing team
        existing = Team.query.filter_by(
            league_id=league_id,
            espn_team_id=espn_team_id
        ).first()

        if existing:
            raise TeamAlreadyExistsError(
                f"Team {espn_team_id} already exists in league {league_id}"
            )

        team = Team(
            league_id=league_id,
            espn_team_id=espn_team_id,
            team_name=team_name,
            owner_name=owner_name or "Unknown",
            current_record=current_record,
            current_standing=current_standing
        )

        db.session.add(team)
        db.session.commit()

        logger.info(f"Created team {team.id} in league {league_id}")
        return team

    except TeamAlreadyExistsError:
        raise
    except IntegrityError as e:
        db.session.rollback()
        logger.error(f"Integrity error creating team: {e}")
        raise TeamAlreadyExistsError("Team already exists")
    except SQLAlchemyError as e:
        db.session.rollback()
        logger.error(f"Database error creating team: {e}")
        raise TeamServiceError(f"Failed to create team: {e}")


def create_or_update_team(
    league_id: int,
    espn_team_id: int,
    team_name: str,
    owner_name: Optional[str] = None,
    current_record: Optional[str] = None,
    current_standing: Optional[int] = None,
    projected_standing: Optional[int] = None,
    win_probability: Optional[float] = None
) -> Team:
    """
    Create a team or update if it already exists.

    Args:
        league_id: League ID
        espn_team_id: ESPN's team ID
        team_name: Team name
        owner_name: Owner's display name
        current_record: Current record
        current_standing: Current standing
        projected_standing: Projected final standing
        win_probability: Probability of winning league

    Returns:
        Team object (created or updated)
    """
    try:
        team = Team.query.filter_by(
            league_id=league_id,
            espn_team_id=espn_team_id
        ).first()

        if team:
            # Update existing team
            team.team_name = team_name
            if owner_name:
                team.owner_name = owner_name
            if current_record:
                team.current_record = current_record
            if current_standing:
                team.current_standing = current_standing
            if projected_standing is not None:
                team.projected_standing = projected_standing
            if win_probability is not None:
                team.win_probability = Decimal(str(win_probability))

            logger.debug(f"Updated team {team.id}")
        else:
            # Create new team
            team = Team(
                league_id=league_id,
                espn_team_id=espn_team_id,
                team_name=team_name,
                owner_name=owner_name or "Unknown",
                current_record=current_record,
                current_standing=current_standing,
                projected_standing=projected_standing,
                win_probability=Decimal(str(win_probability)) if win_probability else None
            )
            db.session.add(team)
            logger.debug(f"Created new team for ESPN ID {espn_team_id}")

        db.session.commit()
        return team

    except SQLAlchemyError as e:
        db.session.rollback()
        logger.error(f"Error creating/updating team: {e}")
        raise TeamServiceError(f"Failed to create/update team: {e}")


def bulk_create_teams(league_id: int, teams_data: List[Dict[str, Any]]) -> List[Team]:
    """
    Create multiple teams at once.

    Args:
        league_id: League ID
        teams_data: List of team data dictionaries

    Returns:
        List of created Team objects
    """
    created_teams = []

    try:
        for data in teams_data:
            team = create_or_update_team(
                league_id=league_id,
                espn_team_id=data['espn_team_id'],
                team_name=data['team_name'],
                owner_name=data.get('owner_name'),
                current_record=data.get('current_record'),
                current_standing=data.get('current_standing')
            )
            created_teams.append(team)

        return created_teams

    except Exception as e:
        logger.error(f"Error bulk creating teams: {e}")
        raise TeamServiceError(f"Failed to bulk create teams: {e}")


# =============================================================================
# Team Read Operations
# =============================================================================

def get_team_by_id(team_id: int) -> Optional[Team]:
    """
    Get a team by its ID.

    Args:
        team_id: Team ID

    Returns:
        Team object or None
    """
    return Team.query.get(team_id)


def get_team_or_404(team_id: int) -> Team:
    """
    Get a team by ID or raise an error.

    Args:
        team_id: Team ID

    Returns:
        Team object

    Raises:
        TeamNotFoundError: If team doesn't exist
    """
    team = get_team_by_id(team_id)
    if not team:
        raise TeamNotFoundError(f"Team {team_id} not found")
    return team


def get_team_by_espn_id(league_id: int, espn_team_id: int) -> Optional[Team]:
    """
    Get a team by ESPN ID within a league.

    Args:
        league_id: League ID
        espn_team_id: ESPN team ID

    Returns:
        Team object or None
    """
    return Team.query.filter_by(
        league_id=league_id,
        espn_team_id=espn_team_id
    ).first()


def get_league_teams(league_id: int) -> List[Team]:
    """
    Get all teams in a league.

    Args:
        league_id: League ID

    Returns:
        List of Team objects ordered by standing
    """
    return Team.query.filter_by(league_id=league_id)\
        .order_by(Team.current_standing).all()


def get_league_standings(league_id: int) -> List[Dict[str, Any]]:
    """
    Get current standings for a league.

    Args:
        league_id: League ID

    Returns:
        List of team standings data
    """
    teams = get_league_teams(league_id)

    return [{
        'team_id': team.id,
        'espn_team_id': team.espn_team_id,
        'team_name': team.team_name,
        'owner_name': team.owner_name,
        'record': team.current_record,
        'standing': team.current_standing,
        'projected_standing': team.projected_standing,
        'win_probability': float(team.win_probability) if team.win_probability else None
    } for team in teams]


# =============================================================================
# Team Update Operations
# =============================================================================

def update_team(team_id: int, **kwargs) -> Team:
    """
    Update team properties.

    Args:
        team_id: Team ID
        **kwargs: Fields to update

    Returns:
        Updated Team object
    """
    team = get_team_or_404(team_id)

    allowed_fields = {
        'team_name', 'owner_name', 'current_record',
        'current_standing', 'projected_standing', 'win_probability'
    }

    try:
        for key, value in kwargs.items():
            if key in allowed_fields and value is not None:
                if key == 'win_probability' and value is not None:
                    value = Decimal(str(value))
                setattr(team, key, value)

        db.session.commit()
        logger.info(f"Updated team {team_id}")
        return team

    except SQLAlchemyError as e:
        db.session.rollback()
        logger.error(f"Error updating team {team_id}: {e}")
        raise TeamServiceError(f"Failed to update team: {e}")


def update_team_projections(
    team_id: int,
    projected_standing: int,
    win_probability: float
) -> Team:
    """
    Update team projection data.

    Args:
        team_id: Team ID
        projected_standing: Projected final standing
        win_probability: Win probability (0-1)

    Returns:
        Updated Team object
    """
    return update_team(
        team_id,
        projected_standing=projected_standing,
        win_probability=win_probability
    )


def update_team_standings(league_id: int, standings_data: List[Dict[str, Any]]) -> None:
    """
    Bulk update standings for all teams in a league.

    Args:
        league_id: League ID
        standings_data: List of dicts with espn_team_id, record, standing
    """
    try:
        for data in standings_data:
            team = get_team_by_espn_id(league_id, data['espn_team_id'])
            if team:
                team.current_record = data.get('record', team.current_record)
                team.current_standing = data.get('standing', team.current_standing)

        db.session.commit()
        logger.info(f"Updated standings for league {league_id}")

    except SQLAlchemyError as e:
        db.session.rollback()
        logger.error(f"Error updating standings: {e}")
        raise TeamServiceError(f"Failed to update standings: {e}")


# =============================================================================
# Team Delete Operations
# =============================================================================

def delete_team(team_id: int) -> bool:
    """
    Delete a team and its roster.

    Args:
        team_id: Team ID

    Returns:
        True if deleted successfully
    """
    team = get_team_or_404(team_id)

    try:
        # Delete roster entries
        Roster.query.filter_by(team_id=team_id).delete()

        db.session.delete(team)
        db.session.commit()

        logger.info(f"Deleted team {team_id}")
        return True

    except SQLAlchemyError as e:
        db.session.rollback()
        logger.error(f"Error deleting team {team_id}: {e}")
        raise TeamServiceError(f"Failed to delete team: {e}")


def delete_league_teams(league_id: int) -> int:
    """
    Delete all teams for a league.

    Args:
        league_id: League ID

    Returns:
        Number of teams deleted
    """
    try:
        # Get team IDs for roster cleanup
        team_ids = [t.id for t in Team.query.filter_by(league_id=league_id).all()]

        # Delete rosters
        if team_ids:
            Roster.query.filter(Roster.team_id.in_(team_ids)).delete(synchronize_session=False)

        # Delete teams
        count = Team.query.filter_by(league_id=league_id).delete()
        db.session.commit()

        logger.info(f"Deleted {count} teams from league {league_id}")
        return count

    except SQLAlchemyError as e:
        db.session.rollback()
        logger.error(f"Error deleting league teams: {e}")
        raise TeamServiceError(f"Failed to delete teams: {e}")


# =============================================================================
# Roster Operations
# =============================================================================

def add_player_to_roster(
    team_id: int,
    player_id: int,
    roster_slot: str = 'BENCH',
    acquisition_type: str = 'DRAFT'
) -> Roster:
    """
    Add a player to a team's roster.

    Args:
        team_id: Team ID
        player_id: Player ID
        roster_slot: Position slot (PG, SG, SF, PF, C, G, F, UTIL, BENCH, IR)
        acquisition_type: How player was acquired (DRAFT, TRADE, WAIVER, FREE_AGENT)

    Returns:
        Created Roster entry
    """
    try:
        # Check if player already on this roster
        existing = Roster.query.filter_by(
            team_id=team_id,
            player_id=player_id
        ).first()

        if existing:
            # Update existing entry
            existing.roster_slot = roster_slot
            existing.acquisition_type = acquisition_type
            db.session.commit()
            return existing

        roster_entry = Roster(
            team_id=team_id,
            player_id=player_id,
            roster_slot=roster_slot,
            acquisition_type=acquisition_type
        )

        db.session.add(roster_entry)
        db.session.commit()

        logger.debug(f"Added player {player_id} to team {team_id}")
        return roster_entry

    except SQLAlchemyError as e:
        db.session.rollback()
        logger.error(f"Error adding player to roster: {e}")
        raise RosterError(f"Failed to add player to roster: {e}")


def remove_player_from_roster(team_id: int, player_id: int) -> bool:
    """
    Remove a player from a team's roster.

    Args:
        team_id: Team ID
        player_id: Player ID

    Returns:
        True if removed, False if not found
    """
    try:
        result = Roster.query.filter_by(
            team_id=team_id,
            player_id=player_id
        ).delete()

        db.session.commit()

        if result:
            logger.debug(f"Removed player {player_id} from team {team_id}")

        return result > 0

    except SQLAlchemyError as e:
        db.session.rollback()
        logger.error(f"Error removing player from roster: {e}")
        raise RosterError(f"Failed to remove player: {e}")


def get_team_roster(team_id: int) -> List[Dict[str, Any]]:
    """
    Get the full roster for a team.

    Args:
        team_id: Team ID

    Returns:
        List of roster entries with player info
    """
    roster_entries = Roster.query.filter_by(team_id=team_id).all()

    result = []
    for entry in roster_entries:
        player = entry.player
        result.append({
            'roster_id': entry.id,
            'player_id': entry.player_id,
            'player_name': player.name if player else 'Unknown',
            'position': player.position if player else None,
            'nba_team': player.nba_team if player else None,
            'roster_slot': entry.roster_slot,
            'acquisition_type': entry.acquisition_type,
            'acquired_date': entry.acquired_date.isoformat() if entry.acquired_date else None
        })

    return result


def update_team_roster(team_id: int, roster_data: List[Dict[str, Any]]) -> None:
    """
    Update a team's entire roster (clear and replace).

    Args:
        team_id: Team ID
        roster_data: List of player roster data
    """
    try:
        # Clear existing roster
        Roster.query.filter_by(team_id=team_id).delete()

        # Add new roster entries
        for data in roster_data:
            roster_entry = Roster(
                team_id=team_id,
                player_id=data['player_id'],
                roster_slot=data.get('roster_slot', 'BENCH'),
                acquisition_type=data.get('acquisition_type', 'UNKNOWN')
            )
            db.session.add(roster_entry)

        db.session.commit()
        logger.info(f"Updated roster for team {team_id} with {len(roster_data)} players")

    except SQLAlchemyError as e:
        db.session.rollback()
        logger.error(f"Error updating roster: {e}")
        raise RosterError(f"Failed to update roster: {e}")


def move_player_slot(team_id: int, player_id: int, new_slot: str) -> Roster:
    """
    Move a player to a different roster slot.

    Args:
        team_id: Team ID
        player_id: Player ID
        new_slot: New roster slot

    Returns:
        Updated Roster entry
    """
    try:
        entry = Roster.query.filter_by(
            team_id=team_id,
            player_id=player_id
        ).first()

        if not entry:
            raise RosterError(f"Player {player_id} not on team {team_id} roster")

        entry.roster_slot = new_slot
        db.session.commit()

        return entry

    except RosterError:
        raise
    except SQLAlchemyError as e:
        db.session.rollback()
        logger.error(f"Error moving player slot: {e}")
        raise RosterError(f"Failed to move player: {e}")


def get_roster_by_slot(team_id: int, slot: str) -> List[Roster]:
    """
    Get roster entries for a specific slot.

    Args:
        team_id: Team ID
        slot: Roster slot to filter

    Returns:
        List of Roster entries
    """
    return Roster.query.filter_by(
        team_id=team_id,
        roster_slot=slot
    ).all()

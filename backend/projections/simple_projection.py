#!/usr/bin/env python3
"""
Simple Projection Engine for Fantasy Basketball.

This module provides a simple, reliable fallback projection engine that:
1. Takes current season per-game averages
2. Estimates games remaining based on season progress
3. Projects rest-of-season totals by multiplying averages by games remaining
4. Handles all 8 standard scoring categories

This is used when ML models aren't trained or when a quick projection is needed.

Usage:
    engine = SimpleProjectionEngine()
    projection = engine.project_player(
        player_id='12345',
        player_name='LeBron James',
        current_stats={'pts': 25.0, 'trb': 8.0, 'ast': 6.0, ...},
        games_played=40,
        injury_status='ACTIVE'
    )
"""

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# NBA season constants
NBA_SEASON_GAMES = 82

# Standard fantasy stat categories
# Using both internal names (lowercase) and ESPN names (uppercase)
COUNTING_STATS = ['pts', 'trb', 'ast', 'stl', 'blk', '3p', 'tov', 'fgm', 'fga', 'ftm', 'fta']
RATE_STATS = ['fg_pct', 'ft_pct', '3p_pct']

# ESPN to internal stat name mapping
ESPN_STAT_MAP = {
    'PTS': 'pts',
    'REB': 'trb',
    'AST': 'ast',
    'STL': 'stl',
    'BLK': 'blk',
    '3PM': '3p',
    'TO': 'tov',
    'FG%': 'fg_pct',
    'FT%': 'ft_pct',
    'FGM': 'fgm',
    'FGA': 'fga',
    'FTM': 'ftm',
    'FTA': 'fta',
}

# Internal to ESPN stat name mapping
INTERNAL_TO_ESPN = {v: k for k, v in ESPN_STAT_MAP.items()}

# Injury adjustments for games projection
INJURY_FACTORS = {
    'ACTIVE': 1.0,
    'PROBABLE': 0.95,
    'QUESTIONABLE': 0.80,
    'DOUBTFUL': 0.50,
    'DAY_TO_DAY': 0.85,
    'DAY': 0.85,
    'GTD': 0.85,
    'OUT': 0.30,  # Short-term, not season-ending
    'INJ_RESERVE': 0.0,
    'INJURED_RESERVE': 0.0,
    'IR': 0.0,
    'SUSPENSION': 0.5,
}


@dataclass
class SimpleProjection:
    """Simple projection result."""
    player_id: str
    player_name: str

    # Per-game projections (same as current for simple model)
    projected_per_game: Dict[str, float]

    # Rest-of-season totals
    ros_totals: Dict[str, float]

    # Games info
    games_played: int
    games_remaining: int
    games_projected: int

    # Confidence (lower for simple projection)
    confidence_score: float = 60.0

    # Metadata
    projection_method: str = 'simple_current_average'
    projection_date: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'player_id': self.player_id,
            'player_name': self.player_name,
            'projected_per_game': self.projected_per_game,
            'ros_totals': self.ros_totals,
            'games_played': self.games_played,
            'games_remaining': self.games_remaining,
            'games_projected': self.games_projected,
            'confidence_score': self.confidence_score,
            'projection_method': self.projection_method,
            'projection_date': self.projection_date.isoformat(),
        }

    def get_espn_format_totals(self) -> Dict[str, float]:
        """Get ROS totals in ESPN naming format."""
        result = {}
        for internal_name, value in self.ros_totals.items():
            espn_name = INTERNAL_TO_ESPN.get(internal_name, internal_name.upper())
            result[espn_name] = value
        return result


class SimpleProjectionEngine:
    """
    Simple projection engine using current season averages.

    Projection formula:
    - ROS Total = Per Game Average × Games Projected
    - Games Projected = Games Remaining × Injury Factor × Historical Game Rate

    For percentages:
    - FG% = Projected FGM / Projected FGA
    - FT% = Projected FTM / Projected FTA
    """

    def __init__(self):
        """Initialize the simple projection engine."""
        # Use 2025-26 season dates
        self.season_start = date(2025, 10, 22)  # 2025-26 season start
        self.season_end = date(2026, 4, 13)     # 2025-26 season end
        logger.debug("SimpleProjectionEngine initialized")

    def _get_season_progress(self) -> float:
        """Get current season progress (0.0 to 1.0)."""
        today = date.today()

        if today < self.season_start:
            return 0.0
        if today > self.season_end:
            return 1.0

        total_days = (self.season_end - self.season_start).days
        elapsed_days = (today - self.season_start).days

        return elapsed_days / total_days

    def _estimate_games_remaining(self) -> int:
        """Estimate games remaining in season."""
        progress = self._get_season_progress()
        games_elapsed = int(NBA_SEASON_GAMES * progress)
        return max(0, NBA_SEASON_GAMES - games_elapsed)

    def _normalize_stats(self, stats: Dict[str, float]) -> Dict[str, float]:
        """Normalize stat names to internal format."""
        normalized = {}

        for key, value in stats.items():
            if value is None:
                continue

            # Check if it's an ESPN name
            if key in ESPN_STAT_MAP:
                normalized[ESPN_STAT_MAP[key]] = float(value)
            else:
                # Assume it's already internal format or lowercase it
                normalized[key.lower()] = float(value)

        return normalized

    def project_player(
        self,
        player_id: str,
        player_name: str,
        current_stats: Dict[str, float],
        games_played: int,
        injury_status: str = 'ACTIVE',
        team_games_remaining: Optional[int] = None
    ) -> SimpleProjection:
        """
        Generate a simple projection based on current season averages.

        Args:
            player_id: Player identifier
            player_name: Player name
            current_stats: Current season per-game averages
            games_played: Games played this season
            injury_status: Current injury status
            team_games_remaining: Optional team-specific games remaining

        Returns:
            SimpleProjection with ROS totals
        """
        logger.debug(f"Projecting {player_name} (GP: {games_played}, status: {injury_status})")

        # Normalize stats to internal format
        stats = self._normalize_stats(current_stats)

        # Calculate games remaining
        if team_games_remaining is not None:
            games_remaining = team_games_remaining
        else:
            games_remaining = self._estimate_games_remaining()

        # Calculate player's game rate (what % of team games do they play?)
        progress = self._get_season_progress()
        expected_team_games = int(NBA_SEASON_GAMES * progress)

        if expected_team_games > 0 and games_played > 0:
            game_rate = min(1.0, games_played / expected_team_games)
        else:
            game_rate = 0.85  # Default: assume player plays 85% of games

        # Apply injury adjustment
        injury_factor = INJURY_FACTORS.get(injury_status.upper(), 0.7)

        # Calculate projected games
        games_projected = int(games_remaining * game_rate * injury_factor)
        games_projected = max(0, games_projected)

        logger.debug(f"  Games remaining: {games_remaining}, rate: {game_rate:.2f}, "
                    f"injury factor: {injury_factor:.2f}, projected: {games_projected}")

        # Calculate ROS totals for counting stats
        ros_totals = {}
        for stat in COUNTING_STATS:
            per_game = stats.get(stat, 0.0)
            ros_totals[stat] = per_game * games_projected

        # Calculate percentage stats from projected makes/attempts
        if ros_totals.get('fga', 0) > 0:
            ros_totals['fg_pct'] = ros_totals.get('fgm', 0) / ros_totals['fga']
        else:
            ros_totals['fg_pct'] = stats.get('fg_pct', 0.45)

        if ros_totals.get('fta', 0) > 0:
            ros_totals['ft_pct'] = ros_totals.get('ftm', 0) / ros_totals['fta']
        else:
            ros_totals['ft_pct'] = stats.get('ft_pct', 0.75)

        # Calculate confidence based on games played
        if games_played >= 40:
            confidence = 75.0
        elif games_played >= 20:
            confidence = 65.0
        elif games_played >= 10:
            confidence = 55.0
        else:
            confidence = 45.0

        # Reduce confidence for injured players
        if injury_factor < 1.0:
            confidence *= injury_factor

        return SimpleProjection(
            player_id=player_id,
            player_name=player_name,
            projected_per_game=stats,
            ros_totals=ros_totals,
            games_played=games_played,
            games_remaining=games_remaining,
            games_projected=games_projected,
            confidence_score=round(confidence, 1),
        )

    def project_roster(
        self,
        players: List[Dict[str, Any]],
        team_games_remaining: Optional[int] = None
    ) -> List[SimpleProjection]:
        """
        Project an entire roster.

        Args:
            players: List of player dicts with 'player_id', 'name', 'stats',
                    'games_played', 'injury_status'
            team_games_remaining: Optional games remaining for the fantasy team

        Returns:
            List of SimpleProjection objects
        """
        projections = []

        for player in players:
            try:
                projection = self.project_player(
                    player_id=str(player.get('player_id', player.get('espn_player_id', ''))),
                    player_name=player.get('name', 'Unknown'),
                    current_stats=player.get('stats', {}),
                    games_played=player.get('games_played', 0),
                    injury_status=player.get('injury_status', 'ACTIVE'),
                    team_games_remaining=team_games_remaining,
                )
                projections.append(projection)
            except Exception as e:
                logger.warning(f"Failed to project {player.get('name', 'Unknown')}: {e}")

        return projections

    def aggregate_team_totals(
        self,
        projections: List[SimpleProjection],
        categories: List[str]
    ) -> Dict[str, float]:
        """
        Aggregate projected totals for a team.

        Args:
            projections: List of player projections
            categories: Scoring categories to aggregate (ESPN format)

        Returns:
            Dictionary of category totals
        """
        totals = {cat: 0.0 for cat in categories}
        totals['FGM'] = 0.0
        totals['FGA'] = 0.0
        totals['FTM'] = 0.0
        totals['FTA'] = 0.0

        for proj in projections:
            espn_totals = proj.get_espn_format_totals()

            for cat in categories:
                if cat in ['FG%', 'FT%']:
                    continue  # Calculate after summing makes/attempts
                totals[cat] = totals.get(cat, 0.0) + espn_totals.get(cat, 0.0)

            # Track makes/attempts for percentage calculation
            totals['FGM'] += espn_totals.get('FGM', 0.0)
            totals['FGA'] += espn_totals.get('FGA', 0.0)
            totals['FTM'] += espn_totals.get('FTM', 0.0)
            totals['FTA'] += espn_totals.get('FTA', 0.0)

        # Calculate percentages
        if totals['FGA'] > 0:
            totals['FG%'] = totals['FGM'] / totals['FGA']
        else:
            totals['FG%'] = 0.0

        if totals['FTA'] > 0:
            totals['FT%'] = totals['FTM'] / totals['FTA']
        else:
            totals['FT%'] = 0.0

        return totals


# Convenience function
def simple_project_player(
    player_name: str,
    current_stats: Dict[str, float],
    games_played: int,
    injury_status: str = 'ACTIVE'
) -> SimpleProjection:
    """
    Quick single-player projection.

    Args:
        player_name: Player name
        current_stats: Per-game averages
        games_played: Games played this season
        injury_status: Injury status

    Returns:
        SimpleProjection object
    """
    engine = SimpleProjectionEngine()
    return engine.project_player(
        player_id=player_name.lower().replace(' ', '_'),
        player_name=player_name,
        current_stats=current_stats,
        games_played=games_played,
        injury_status=injury_status,
    )


if __name__ == '__main__':
    # Demo/test
    logging.basicConfig(level=logging.DEBUG)

    print("=" * 60)
    print("Simple Projection Engine Demo")
    print("=" * 60)

    engine = SimpleProjectionEngine()

    # Example player stats
    sample_stats = {
        'PTS': 28.5,
        'REB': 7.2,
        'AST': 8.1,
        'STL': 1.5,
        'BLK': 0.6,
        '3PM': 3.2,
        'TO': 3.5,
        'FG%': 0.505,
        'FT%': 0.890,
        'FGM': 10.5,
        'FGA': 20.8,
        'FTM': 6.2,
        'FTA': 7.0,
    }

    projection = engine.project_player(
        player_id='demo_star',
        player_name='Demo Star',
        current_stats=sample_stats,
        games_played=45,
        injury_status='ACTIVE',
    )

    print(f"\nPlayer: {projection.player_name}")
    print(f"Games played: {projection.games_played}")
    print(f"Games remaining: {projection.games_remaining}")
    print(f"Games projected: {projection.games_projected}")
    print(f"Confidence: {projection.confidence_score}")
    print()

    print("ROS Totals (Internal format):")
    for stat in ['pts', 'trb', 'ast', 'stl', 'blk', '3p', 'tov']:
        print(f"  {stat}: {projection.ros_totals.get(stat, 0):.1f}")

    print()
    print("ROS Totals (ESPN format):")
    espn_totals = projection.get_espn_format_totals()
    for stat in ['PTS', 'REB', 'AST', 'STL', 'BLK', '3PM', 'TO', 'FG%', 'FT%']:
        value = espn_totals.get(stat, 0)
        if stat in ['FG%', 'FT%']:
            print(f"  {stat}: {value:.3f}")
        else:
            print(f"  {stat}: {value:.1f}")

"""
SQLAlchemy database models for Fantasy Basketball Optimizer.

This module defines all database models as specified in PRD Section 4:
- Users: User authentication and account management
- Leagues: ESPN Fantasy Basketball league configurations
- Teams: Fantasy teams within leagues
- Players: NBA player information
- Rosters: Player assignments to fantasy teams
- PlayerStats: Historical and current player statistics
- Projections: ML and statistical player projections
- TradeHistory: Analyzed trade history
- WaiverRecommendations: Waiver wire pickup suggestions

All models include proper relationships, constraints, and indexes
for optimal query performance.
"""

from datetime import datetime
from decimal import Decimal
from flask_login import UserMixin

from backend.extensions import db, bcrypt


# =============================================================================
# User Model (PRD 4.1)
# =============================================================================

class User(UserMixin, db.Model):
    """
    User model for authentication and league ownership.

    Attributes:
        id: Primary key
        email: Unique email address for login
        password_hash: Bcrypt hashed password
        created_at: Account creation timestamp
        last_login: Most recent login timestamp
    """

    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)

    # Relationships
    leagues = db.relationship(
        'League',
        backref='user',
        lazy='dynamic',
        cascade='all, delete-orphan'
    )

    def set_password(self, password: str) -> None:
        """Hash and set the user's password using bcrypt."""
        self.password_hash = bcrypt.generate_password_hash(password).decode('utf-8')

    def check_password(self, password: str) -> bool:
        """Verify password against stored hash."""
        return bcrypt.check_password_hash(self.password_hash, password)

    def update_last_login(self) -> None:
        """Update the last login timestamp to now."""
        self.last_login = datetime.utcnow()

    def to_dict(self) -> dict:
        """Convert user to dictionary for API responses."""
        return {
            'id': self.id,
            'email': self.email,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None
        }

    def __repr__(self) -> str:
        return f'<User {self.email}>'


# =============================================================================
# League Model (PRD 4.2)
# =============================================================================

class League(db.Model):
    """
    ESPN Fantasy Basketball league configuration.

    Stores league settings, ESPN authentication cookies, and metadata.
    Supports H2H Categories, H2H Points, and Rotisserie formats.

    Attributes:
        id: Primary key
        user_id: Foreign key to owning user
        espn_league_id: ESPN's league identifier
        season: Season year (e.g., 2025)
        league_name: Display name for the league
        espn_s2_cookie: ESPN_S2 authentication cookie (encrypted at rest)
        swid_cookie: SWID authentication cookie
        league_type: H2H_CATEGORY, H2H_POINTS, or ROTO
        num_teams: Number of teams in the league
        roster_settings: JSON of roster position configuration
        scoring_settings: JSON of scoring category rules
        playoff_settings: JSON of playoff structure
        last_updated: Last data refresh timestamp
        refresh_schedule: Daily refresh time
        created_at: Record creation timestamp
    """

    __tablename__ = 'leagues'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(
        db.Integer,
        db.ForeignKey('users.id', ondelete='CASCADE'),
        nullable=False,
        index=True
    )
    espn_league_id = db.Column(db.Integer, nullable=False, index=True)
    season = db.Column(db.Integer, nullable=False)
    league_name = db.Column(db.String(255))

    # ESPN authentication (should be encrypted at rest in production)
    espn_s2_cookie = db.Column(db.Text, nullable=False)
    swid_cookie = db.Column(db.String(255), nullable=False)

    # League settings
    league_type = db.Column(db.String(50))  # H2H_CATEGORY, H2H_POINTS, ROTO
    num_teams = db.Column(db.Integer)
    roster_settings = db.Column(db.JSON)    # Roster positions configuration
    scoring_settings = db.Column(db.JSON)   # Scoring rules
    playoff_settings = db.Column(db.JSON)   # Playoff structure

    # Metadata
    last_updated = db.Column(db.DateTime)
    refresh_schedule = db.Column(
        db.Time,
        default=datetime.strptime('03:00:00', '%H:%M:%S').time()
    )
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Game projection settings
    # 'adaptive': Uses tiered game_rate based on games played (default)
    # 'flat_rate': Uses a fixed percentage for all players
    projection_method = db.Column(db.String(20), default='adaptive')
    flat_game_rate = db.Column(db.Float, default=0.85)  # Used when projection_method='flat_rate'

    # Relationships
    teams = db.relationship(
        'Team',
        backref='league',
        lazy='dynamic',
        cascade='all, delete-orphan'
    )
    projections = db.relationship(
        'Projection',
        backref='league',
        lazy='dynamic',
        cascade='all, delete-orphan'
    )
    trade_history = db.relationship(
        'TradeHistory',
        backref='league',
        lazy='dynamic',
        cascade='all, delete-orphan'
    )
    waiver_recommendations = db.relationship(
        'WaiverRecommendation',
        backref='league',
        lazy='dynamic',
        cascade='all, delete-orphan'
    )

    # Unique constraint: one league per user per season
    __table_args__ = (
        db.UniqueConstraint(
            'user_id', 'espn_league_id', 'season',
            name='unique_user_league_season'
        ),
        db.Index('idx_league_user_season', 'user_id', 'season'),
    )

    @property
    def is_h2h(self) -> bool:
        """Check if league uses head-to-head format."""
        return self.league_type in ('H2H_CATEGORY', 'H2H_POINTS')

    @property
    def is_roto(self) -> bool:
        """Check if league uses rotisserie format."""
        return self.league_type == 'ROTO'

    def to_dict(self, include_settings: bool = True) -> dict:
        """
        Convert league to dictionary for API responses.

        Args:
            include_settings: Whether to include roster/scoring settings
        """
        data = {
            'id': self.id,
            'espn_league_id': self.espn_league_id,
            'season': self.season,
            'league_name': self.league_name,
            'league_type': self.league_type,
            'num_teams': self.num_teams,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'projection_method': self.projection_method or 'adaptive',
            'flat_game_rate': self.flat_game_rate or 0.85,
        }
        if include_settings:
            data['roster_settings'] = self.roster_settings
            data['scoring_settings'] = self.scoring_settings
            data['playoff_settings'] = self.playoff_settings
        return data

    def __repr__(self) -> str:
        return f'<League {self.league_name} ({self.season})>'


# =============================================================================
# Team Model (PRD 4.3)
# =============================================================================

class Team(db.Model):
    """
    Fantasy team within a league.

    Attributes:
        id: Primary key
        league_id: Foreign key to parent league
        espn_team_id: ESPN's team identifier
        team_name: Team display name
        owner_name: Team owner's name
        current_record: Current W-L-T record (e.g., "10-5-2")
        current_standing: Current league standing (1-N)
        projected_standing: Projected end-of-season standing
        win_probability: Projected win/playoff probability (0-100)
        last_updated: Last data refresh timestamp
    """

    __tablename__ = 'teams'

    id = db.Column(db.Integer, primary_key=True)
    league_id = db.Column(
        db.Integer,
        db.ForeignKey('leagues.id', ondelete='CASCADE'),
        nullable=False,
        index=True
    )
    espn_team_id = db.Column(db.Integer, nullable=False)
    team_name = db.Column(db.String(255))
    owner_name = db.Column(db.String(255))

    # Standings
    current_record = db.Column(db.String(50))   # "10-5-2" for H2H
    current_standing = db.Column(db.Integer)
    projected_standing = db.Column(db.Integer)
    win_probability = db.Column(db.Numeric(5, 2))  # 0.00 to 100.00

    last_updated = db.Column(db.DateTime)

    # Relationships
    rosters = db.relationship(
        'Roster',
        backref='team',
        lazy='dynamic',
        cascade='all, delete-orphan'
    )

    __table_args__ = (
        db.UniqueConstraint('league_id', 'espn_team_id', name='unique_league_team'),
        db.Index('idx_team_league_standing', 'league_id', 'current_standing'),
    )

    @property
    def wins(self) -> int:
        """Extract wins from current record."""
        if self.current_record:
            parts = self.current_record.split('-')
            return int(parts[0]) if parts else 0
        return 0

    @property
    def losses(self) -> int:
        """Extract losses from current record."""
        if self.current_record:
            parts = self.current_record.split('-')
            return int(parts[1]) if len(parts) > 1 else 0
        return 0

    def to_dict(self) -> dict:
        """Convert team to dictionary for API responses."""
        return {
            'id': self.id,
            'league_id': self.league_id,
            'espn_team_id': self.espn_team_id,
            'team_name': self.team_name,
            'owner_name': self.owner_name,
            'current_record': self.current_record,
            'current_standing': self.current_standing,
            'projected_standing': self.projected_standing,
            'win_probability': float(self.win_probability) if self.win_probability else None,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None
        }

    def __repr__(self) -> str:
        return f'<Team {self.team_name}>'


# =============================================================================
# Player Model (PRD 4.4)
# =============================================================================

class Player(db.Model):
    """
    NBA player information.

    Central player entity referenced by rosters, stats, and projections.

    Attributes:
        id: Primary key
        espn_player_id: ESPN's unique player identifier
        name: Player's full name
        team: NBA team abbreviation (e.g., 'LAL', 'BOS')
        position: Player position(s) (e.g., 'PG', 'SG/SF')
        injury_status: Current injury status (e.g., 'OUT', 'DTD', 'HEALTHY')
        last_updated: Last data refresh timestamp
    """

    __tablename__ = 'players'

    id = db.Column(db.Integer, primary_key=True)
    espn_player_id = db.Column(db.Integer, unique=True, nullable=False, index=True)
    name = db.Column(db.String(255), nullable=False, index=True)
    team = db.Column(db.String(10))     # NBA team abbreviation
    position = db.Column(db.String(20))
    injury_status = db.Column(db.String(50))
    last_updated = db.Column(db.DateTime)

    # Relationships
    stats = db.relationship(
        'PlayerStats',
        backref='player',
        lazy='dynamic',
        cascade='all, delete-orphan'
    )
    projections = db.relationship(
        'Projection',
        backref='player',
        lazy='dynamic',
        cascade='all, delete-orphan'
    )

    @property
    def is_injured(self) -> bool:
        """Check if player has an injury status."""
        return self.injury_status is not None and self.injury_status not in ('HEALTHY', 'ACTIVE', '')

    @property
    def positions_list(self) -> list:
        """Get list of eligible positions."""
        if self.position:
            return [p.strip() for p in self.position.split('/')]
        return []

    def to_dict(self, include_stats: bool = False) -> dict:
        """
        Convert player to dictionary for API responses.

        Args:
            include_stats: Whether to include current season stats
        """
        data = {
            'id': self.id,
            'espn_player_id': self.espn_player_id,
            'name': self.name,
            'team': self.team,
            'position': self.position,
            'positions': self.positions_list,
            'injury_status': self.injury_status,
            'is_injured': self.is_injured,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None
        }
        return data

    def __repr__(self) -> str:
        return f'<Player {self.name}>'


# =============================================================================
# Roster Model (PRD 4.5)
# =============================================================================

class Roster(db.Model):
    """
    Player roster assignment to a fantasy team.

    Links players to fantasy teams with acquisition metadata.

    Attributes:
        id: Primary key
        team_id: Foreign key to fantasy team
        player_id: Foreign key to player
        acquisition_type: How player was acquired (DRAFT, TRADE, WAIVER, FA)
        acquisition_date: When player was acquired
        roster_slot: Current roster position (PG, SG, SF, PF, C, G, F, UTIL, BE, IR)
    """

    __tablename__ = 'rosters'

    id = db.Column(db.Integer, primary_key=True)
    team_id = db.Column(
        db.Integer,
        db.ForeignKey('teams.id', ondelete='CASCADE'),
        nullable=False,
        index=True
    )
    player_id = db.Column(
        db.Integer,
        db.ForeignKey('players.id'),
        nullable=False,
        index=True
    )
    acquisition_type = db.Column(db.String(50))  # DRAFT, TRADE, WAIVER, FA
    acquisition_date = db.Column(db.Date)
    roster_slot = db.Column(db.String(20))       # PG, SG, BE, IR, etc.

    # Relationship to player
    player = db.relationship('Player', backref='roster_entries')

    __table_args__ = (
        db.UniqueConstraint('team_id', 'player_id', name='unique_team_player'),
    )

    @property
    def is_active(self) -> bool:
        """Check if player is in an active roster slot (not bench/IR)."""
        return self.roster_slot not in ('BE', 'IR', 'IL')

    def to_dict(self, include_player: bool = True) -> dict:
        """Convert roster entry to dictionary for API responses."""
        data = {
            'id': self.id,
            'team_id': self.team_id,
            'player_id': self.player_id,
            'acquisition_type': self.acquisition_type,
            'acquisition_date': self.acquisition_date.isoformat() if self.acquisition_date else None,
            'roster_slot': self.roster_slot,
            'is_active': self.is_active
        }
        if include_player and self.player:
            data['player'] = self.player.to_dict()
        return data

    def __repr__(self) -> str:
        return f'<Roster team={self.team_id} player={self.player_id} slot={self.roster_slot}>'


# =============================================================================
# PlayerStats Model (PRD 4.6)
# =============================================================================

class PlayerStats(db.Model):
    """
    Player statistics for a given period.

    Stores per-game averages from ESPN or Basketball Reference.

    Attributes:
        id: Primary key
        player_id: Foreign key to player
        season: Season year
        games_played: Number of games played
        minutes_per_game: Average minutes per game
        points, rebounds, assists, steals, blocks, turnovers: Per-game averages
        field_goal_pct, free_throw_pct: Shooting percentages
        three_pointers_made: 3PM per game
        stat_date: Date of stats snapshot
        source: Data source (ESPN, BASKETBALL_REFERENCE)
    """

    __tablename__ = 'player_stats'

    id = db.Column(db.Integer, primary_key=True)
    player_id = db.Column(
        db.Integer,
        db.ForeignKey('players.id'),
        nullable=False,
        index=True
    )
    season = db.Column(db.Integer, nullable=False, index=True)
    games_played = db.Column(db.Integer)
    minutes_per_game = db.Column(db.Numeric(5, 2))

    # Counting stats (per game averages)
    points = db.Column(db.Numeric(5, 2))
    rebounds = db.Column(db.Numeric(5, 2))
    assists = db.Column(db.Numeric(5, 2))
    steals = db.Column(db.Numeric(5, 2))
    blocks = db.Column(db.Numeric(5, 2))
    turnovers = db.Column(db.Numeric(5, 2))

    # Shooting stats
    field_goal_pct = db.Column(db.Numeric(5, 3))   # 0.000 to 1.000
    free_throw_pct = db.Column(db.Numeric(5, 3))   # 0.000 to 1.000
    three_pointers_made = db.Column(db.Numeric(5, 2))

    # Metadata
    stat_date = db.Column(db.Date, index=True)
    source = db.Column(db.String(50))  # ESPN, BASKETBALL_REFERENCE

    __table_args__ = (
        db.UniqueConstraint(
            'player_id', 'season', 'stat_date', 'source',
            name='unique_player_stats'
        ),
        db.Index('idx_stats_player_season', 'player_id', 'season'),
    )

    def to_dict(self) -> dict:
        """Convert player stats to dictionary for API responses."""
        return {
            'id': self.id,
            'player_id': self.player_id,
            'season': self.season,
            'games_played': self.games_played,
            'minutes_per_game': float(self.minutes_per_game) if self.minutes_per_game else None,
            'points': float(self.points) if self.points else None,
            'rebounds': float(self.rebounds) if self.rebounds else None,
            'assists': float(self.assists) if self.assists else None,
            'steals': float(self.steals) if self.steals else None,
            'blocks': float(self.blocks) if self.blocks else None,
            'turnovers': float(self.turnovers) if self.turnovers else None,
            'field_goal_pct': float(self.field_goal_pct) if self.field_goal_pct else None,
            'free_throw_pct': float(self.free_throw_pct) if self.free_throw_pct else None,
            'three_pointers_made': float(self.three_pointers_made) if self.three_pointers_made else None,
            'stat_date': self.stat_date.isoformat() if self.stat_date else None,
            'source': self.source
        }

    def __repr__(self) -> str:
        return f'<PlayerStats player={self.player_id} season={self.season} source={self.source}>'


# =============================================================================
# Projection Model (PRD 4.7)
# =============================================================================

class Projection(db.Model):
    """
    Player projections for rest of season.

    Stores statistical, ML, or hybrid projections for player performance.

    Attributes:
        id: Primary key
        player_id: Foreign key to player
        league_id: Foreign key to league (for league-specific projections)
        season: Season year
        projection_type: STATISTICAL, ML, or HYBRID
        games_remaining: Projected games remaining in season
        projected_*: Projected per-game stats
        fantasy_value: Calculated fantasy value based on league scoring
        confidence: Projection confidence score (0-100)
        created_at: When projection was generated
    """

    __tablename__ = 'projections'

    id = db.Column(db.Integer, primary_key=True)
    player_id = db.Column(
        db.Integer,
        db.ForeignKey('players.id'),
        nullable=False,
        index=True
    )
    league_id = db.Column(
        db.Integer,
        db.ForeignKey('leagues.id', ondelete='CASCADE'),
        nullable=False,
        index=True
    )
    season = db.Column(db.Integer, nullable=False)
    projection_type = db.Column(db.String(50))  # STATISTICAL, ML, HYBRID

    games_remaining = db.Column(db.Integer)

    # Projected per-game stats
    projected_points = db.Column(db.Numeric(5, 2))
    projected_rebounds = db.Column(db.Numeric(5, 2))
    projected_assists = db.Column(db.Numeric(5, 2))
    projected_steals = db.Column(db.Numeric(5, 2))
    projected_blocks = db.Column(db.Numeric(5, 2))
    projected_turnovers = db.Column(db.Numeric(5, 2))
    projected_fg_pct = db.Column(db.Numeric(5, 3))
    projected_ft_pct = db.Column(db.Numeric(5, 3))
    projected_threes = db.Column(db.Numeric(5, 2))

    # Calculated values
    fantasy_value = db.Column(db.Numeric(8, 2))
    confidence = db.Column(db.Numeric(5, 2))  # 0-100

    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)

    __table_args__ = (
        db.Index('idx_projection_player_league', 'player_id', 'league_id', 'season'),
        db.Index('idx_projection_type', 'projection_type', 'season'),
    )

    def to_dict(self) -> dict:
        """Convert projection to dictionary for API responses."""
        return {
            'id': self.id,
            'player_id': self.player_id,
            'league_id': self.league_id,
            'season': self.season,
            'projection_type': self.projection_type,
            'games_remaining': self.games_remaining,
            'projected_stats': {
                'points': float(self.projected_points) if self.projected_points else None,
                'rebounds': float(self.projected_rebounds) if self.projected_rebounds else None,
                'assists': float(self.projected_assists) if self.projected_assists else None,
                'steals': float(self.projected_steals) if self.projected_steals else None,
                'blocks': float(self.projected_blocks) if self.projected_blocks else None,
                'turnovers': float(self.projected_turnovers) if self.projected_turnovers else None,
                'fg_pct': float(self.projected_fg_pct) if self.projected_fg_pct else None,
                'ft_pct': float(self.projected_ft_pct) if self.projected_ft_pct else None,
                'threes': float(self.projected_threes) if self.projected_threes else None,
            },
            'fantasy_value': float(self.fantasy_value) if self.fantasy_value else None,
            'confidence': float(self.confidence) if self.confidence else None,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

    def __repr__(self) -> str:
        return f'<Projection player={self.player_id} type={self.projection_type}>'


# =============================================================================
# TradeHistory Model (PRD 4.8)
# =============================================================================

class TradeHistory(db.Model):
    """
    History of analyzed trades.

    Logs all trade analyses for tracking and model improvement.

    Attributes:
        id: Primary key
        league_id: Foreign key to league
        team1_id, team2_id: Teams involved in trade
        team1_players, team2_players: JSON arrays of player IDs
        analyzed_at: When trade was analyzed
        value_differential: Net value difference between teams
        was_suggested: Whether trade was AI-suggested
        was_accepted: Whether trade was actually executed
    """

    __tablename__ = 'trade_history'

    id = db.Column(db.Integer, primary_key=True)
    league_id = db.Column(
        db.Integer,
        db.ForeignKey('leagues.id', ondelete='CASCADE'),
        nullable=False,
        index=True
    )
    team1_id = db.Column(db.Integer, db.ForeignKey('teams.id'), index=True)
    team2_id = db.Column(db.Integer, db.ForeignKey('teams.id'), index=True)
    team1_players = db.Column(db.JSON)   # Array of player IDs
    team2_players = db.Column(db.JSON)   # Array of player IDs
    analyzed_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    value_differential = db.Column(db.Numeric(8, 2))
    was_suggested = db.Column(db.Boolean, default=False)
    was_accepted = db.Column(db.Boolean)

    # Relationships to teams
    team1 = db.relationship('Team', foreign_keys=[team1_id], backref='trades_as_team1')
    team2 = db.relationship('Team', foreign_keys=[team2_id], backref='trades_as_team2')

    __table_args__ = (
        db.Index('idx_trade_league_date', 'league_id', 'analyzed_at'),
    )

    @property
    def is_fair(self) -> bool:
        """Check if trade is considered fair (value differential within 10%)."""
        if self.value_differential is None:
            return True
        return abs(float(self.value_differential)) <= 10.0

    def to_dict(self, include_teams: bool = False) -> dict:
        """Convert trade history to dictionary for API responses."""
        data = {
            'id': self.id,
            'league_id': self.league_id,
            'team1_id': self.team1_id,
            'team2_id': self.team2_id,
            'team1_players': self.team1_players,
            'team2_players': self.team2_players,
            'analyzed_at': self.analyzed_at.isoformat() if self.analyzed_at else None,
            'value_differential': float(self.value_differential) if self.value_differential else None,
            'is_fair': self.is_fair,
            'was_suggested': self.was_suggested,
            'was_accepted': self.was_accepted
        }
        if include_teams:
            data['team1'] = self.team1.to_dict() if self.team1 else None
            data['team2'] = self.team2.to_dict() if self.team2 else None
        return data

    def __repr__(self) -> str:
        return f'<TradeHistory league={self.league_id} teams={self.team1_id}<->{self.team2_id}>'


# =============================================================================
# WaiverRecommendation Model (PRD 4.9)
# =============================================================================

class WaiverRecommendation(db.Model):
    """
    Waiver wire pickup recommendations.

    AI-generated suggestions for free agent pickups.

    Attributes:
        id: Primary key
        league_id: Foreign key to league
        team_id: Foreign key to team receiving recommendation
        player_id: Recommended player to add
        impact_score: Projected impact (0-100)
        suggested_drop_player_id: Suggested player to drop
        recommendation_date: Date recommendation was generated
        was_followed: Whether user followed the recommendation
    """

    __tablename__ = 'waiver_recommendations'

    id = db.Column(db.Integer, primary_key=True)
    league_id = db.Column(
        db.Integer,
        db.ForeignKey('leagues.id', ondelete='CASCADE'),
        nullable=False,
        index=True
    )
    team_id = db.Column(
        db.Integer,
        db.ForeignKey('teams.id'),
        index=True
    )
    player_id = db.Column(
        db.Integer,
        db.ForeignKey('players.id'),
        nullable=False,
        index=True
    )
    impact_score = db.Column(db.Numeric(5, 2))  # 0-100
    suggested_drop_player_id = db.Column(
        db.Integer,
        db.ForeignKey('players.id')
    )
    recommendation_date = db.Column(db.Date, index=True)
    was_followed = db.Column(db.Boolean)

    # Relationships
    player = db.relationship('Player', foreign_keys=[player_id], backref='waiver_recommendations')
    suggested_drop = db.relationship('Player', foreign_keys=[suggested_drop_player_id])
    team = db.relationship('Team', backref='waiver_recommendations')

    __table_args__ = (
        db.UniqueConstraint(
            'league_id', 'team_id', 'player_id', 'recommendation_date',
            name='unique_waiver_recommendation'
        ),
        db.Index('idx_waiver_league_date', 'league_id', 'recommendation_date'),
        db.Index('idx_waiver_impact', 'league_id', 'impact_score'),
    )

    @property
    def impact_tier(self) -> str:
        """Get impact tier based on score."""
        if self.impact_score is None:
            return 'unknown'
        score = float(self.impact_score)
        if score >= 80:
            return 'high'
        elif score >= 50:
            return 'medium'
        else:
            return 'low'

    def to_dict(self, include_players: bool = True) -> dict:
        """Convert waiver recommendation to dictionary for API responses."""
        data = {
            'id': self.id,
            'league_id': self.league_id,
            'team_id': self.team_id,
            'player_id': self.player_id,
            'impact_score': float(self.impact_score) if self.impact_score else None,
            'impact_tier': self.impact_tier,
            'suggested_drop_player_id': self.suggested_drop_player_id,
            'recommendation_date': self.recommendation_date.isoformat() if self.recommendation_date else None,
            'was_followed': self.was_followed
        }
        if include_players:
            data['player'] = self.player.to_dict() if self.player else None
            data['suggested_drop'] = self.suggested_drop.to_dict() if self.suggested_drop else None
        return data

    def __repr__(self) -> str:
        return f'<WaiverRecommendation player={self.player_id} score={self.impact_score}>'

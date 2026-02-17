# Scrapers Package

from backend.scrapers.basketball_reference import (
    BasketballReferenceScraper,
    create_scraper,
    normalize_player_name,
    create_player_id,
    get_current_season,
    ScraperError,
    RateLimitError,
    PageNotFoundError,
    ParseError,
)

from backend.scrapers.nba_schedule import (
    NBASchedule,
    NBA_TEAMS,
    TEAM_ABBR_ALIASES,
    get_player_game_dates,
)

__all__ = [
    # Basketball Reference
    'BasketballReferenceScraper',
    'create_scraper',
    'normalize_player_name',
    'create_player_id',
    'get_current_season',
    'ScraperError',
    'RateLimitError',
    'PageNotFoundError',
    'ParseError',
    # NBA Schedule
    'NBASchedule',
    'NBA_TEAMS',
    'TEAM_ABBR_ALIASES',
    'get_player_game_dates',
]

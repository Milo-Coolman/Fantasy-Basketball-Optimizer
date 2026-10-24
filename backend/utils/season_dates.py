"""
Centralized NBA season date calculator.

This module provides a single source of truth for all season date calculations,
eliminating hardcoded dates throughout the codebase.

NBA Season Convention:
- Season spans two calendar years (e.g., 2026-27)
- ESPN uses ending year as identifier (2026-27 = "2027")
- Regular season: late October to mid-April
- All-Star break: mid-February (3rd weekend)
"""

from datetime import date, timedelta
from typing import Tuple, Optional


class NBASeasonDates:
    """
    Centralized NBA season date calculator.

    All season-related date calculations should go through this class
    to ensure consistency across the application.
    """

    # Typical season dates (may vary by 1-2 days year to year)
    DEFAULT_START_MONTH = 10  # October
    DEFAULT_START_DAY = 22
    DEFAULT_END_MONTH = 4     # April
    DEFAULT_END_DAY = 13

    # Season constants
    GAMES_PER_SEASON = 82

    @classmethod
    def get_season_from_date(cls, d: date = None) -> int:
        """
        Determine ESPN season year from a date.

        NBA seasons span two calendar years. ESPN identifies seasons
        by the ending year (e.g., the 2026-27 season is "2027").

        Args:
            d: Date to check. Defaults to today.

        Returns:
            ESPN season year (e.g., 2027 for the 2026-27 season)

        Examples:
            - October 15, 2026 -> 2027 (2026-27 season)
            - February 1, 2027 -> 2027 (2026-27 season)
            - July 1, 2027 -> 2027 (offseason, next season not started)
        """
        if d is None:
            d = date.today()

        if d.month >= 10:  # October-December: new season
            return d.year + 1
        else:  # January-September: current season year
            return d.year

    @classmethod
    def get_season_start(cls, season: int) -> date:
        """
        Get season start date.

        Args:
            season: ESPN season year (e.g., 2027)

        Returns:
            Start date (October of previous calendar year)
        """
        return date(season - 1, cls.DEFAULT_START_MONTH, cls.DEFAULT_START_DAY)

    @classmethod
    def get_season_end(cls, season: int) -> date:
        """
        Get season end date (regular season).

        Args:
            season: ESPN season year (e.g., 2027)

        Returns:
            End date (April of season year)
        """
        return date(season, cls.DEFAULT_END_MONTH, cls.DEFAULT_END_DAY)

    @classmethod
    def get_all_star_break(cls, season: int) -> Tuple[date, date]:
        """
        Calculate All-Star break dates for a season.

        All-Star weekend is typically the 3rd weekend of February.
        The break spans Friday through Monday (All-Star Sunday game).

        Args:
            season: ESPN season year (e.g., 2027)

        Returns:
            Tuple of (start_date, end_date) for the All-Star break
        """
        # Find the third Monday of February
        feb_first = date(season, 2, 1)

        # Calculate days until first Monday (weekday: 0=Mon, 6=Sun)
        days_until_monday = (7 - feb_first.weekday()) % 7
        if feb_first.weekday() == 0:  # If Feb 1 is Monday
            days_until_monday = 0

        first_monday = feb_first + timedelta(days=days_until_monday)
        third_monday = first_monday + timedelta(weeks=2)

        # All-Star break: Friday before to Monday after (inclusive)
        # Game is Sunday, but Monday is typically off as well
        all_star_start = third_monday - timedelta(days=3)  # Friday
        all_star_end = third_monday  # Monday

        return all_star_start, all_star_end

    @classmethod
    def get_season_progress(cls, season: int, d: date = None) -> float:
        """
        Get season progress as a float from 0.0 to 1.0.

        Args:
            season: ESPN season year
            d: Date to check. Defaults to today.

        Returns:
            Progress value:
            - Negative if before season start
            - 0.0 to 1.0 during season
            - > 1.0 if after season end
        """
        if d is None:
            d = date.today()

        start = cls.get_season_start(season)
        end = cls.get_season_end(season)

        if d < start:
            return -1.0
        if d > end:
            return 1.0

        total_days = (end - start).days
        elapsed_days = (d - start).days

        return elapsed_days / total_days if total_days > 0 else 0.0

    @classmethod
    def is_season_complete(cls, season: int, d: date = None) -> bool:
        """
        Check if regular season is complete.

        Args:
            season: ESPN season year
            d: Date to check. Defaults to today.

        Returns:
            True if the regular season has ended
        """
        if d is None:
            d = date.today()
        return d > cls.get_season_end(season)

    @classmethod
    def is_offseason(cls, d: date = None) -> bool:
        """
        Check if currently in offseason (May-September).

        The offseason is after playoffs end (typically mid-June)
        until the next season starts (late October). For simplicity,
        we define it as May-September.

        Args:
            d: Date to check. Defaults to today.

        Returns:
            True if in offseason period
        """
        if d is None:
            d = date.today()
        return 5 <= d.month <= 9

    @classmethod
    def estimate_games_remaining(cls, season: int, d: date = None) -> int:
        """
        Estimate games remaining based on season progress.

        This is a rough estimate based on linear progress through
        the season. For more accurate counts, use the NBA schedule.

        Args:
            season: ESPN season year
            d: Date to check. Defaults to today.

        Returns:
            Estimated number of games remaining (0-82)
        """
        progress = cls.get_season_progress(season, d)

        if progress < 0:
            return cls.GAMES_PER_SEASON  # Full season ahead
        if progress >= 1:
            return 0  # Season over

        return max(0, int(cls.GAMES_PER_SEASON * (1 - progress)))

    @classmethod
    def get_season_dates_dict(cls, season: int) -> dict:
        """
        Get all season dates as a dictionary.

        Useful for replacing the old SEASON_XXXX constants.

        Args:
            season: ESPN season year

        Returns:
            Dictionary with start, end, and all_star_break dates
        """
        all_star_start, all_star_end = cls.get_all_star_break(season)
        return {
            'start': cls.get_season_start(season),
            'end': cls.get_season_end(season),
            'all_star_break_start': all_star_start,
            'all_star_break_end': all_star_end,
        }

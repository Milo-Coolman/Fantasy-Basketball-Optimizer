"""
Scheduler Service for Fantasy Basketball Optimizer.

Manages scheduled tasks using APScheduler, including:
- Daily league data refresh from ESPN
- Database maintenance tasks
- Cache cleanup

The scheduler integrates with the Flask application and can be
configured via environment variables.

Reference: PRD Section 3.2 - Daily automatic data refresh
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Optional, List, Callable

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR, EVENT_JOB_MISSED

logger = logging.getLogger(__name__)


# =============================================================================
# Scheduler Configuration
# =============================================================================

class SchedulerConfig:
    """Configuration for the scheduler service."""

    # Default refresh time (3 AM)
    DEFAULT_REFRESH_HOUR = 3
    DEFAULT_REFRESH_MINUTE = 0

    # Environment variable names
    ENV_REFRESH_HOUR = 'SCHEDULER_REFRESH_HOUR'
    ENV_REFRESH_MINUTE = 'SCHEDULER_REFRESH_MINUTE'
    ENV_SCHEDULER_ENABLED = 'SCHEDULER_ENABLED'
    ENV_TIMEZONE = 'SCHEDULER_TIMEZONE'

    @classmethod
    def get_refresh_hour(cls) -> int:
        """Get the hour for daily refresh (0-23)."""
        try:
            return int(os.getenv(cls.ENV_REFRESH_HOUR, cls.DEFAULT_REFRESH_HOUR))
        except (ValueError, TypeError):
            return cls.DEFAULT_REFRESH_HOUR

    @classmethod
    def get_refresh_minute(cls) -> int:
        """Get the minute for daily refresh (0-59)."""
        try:
            return int(os.getenv(cls.ENV_REFRESH_MINUTE, cls.DEFAULT_REFRESH_MINUTE))
        except (ValueError, TypeError):
            return cls.DEFAULT_REFRESH_MINUTE

    @classmethod
    def is_enabled(cls) -> bool:
        """Check if scheduler is enabled."""
        enabled = os.getenv(cls.ENV_SCHEDULER_ENABLED, 'true').lower()
        return enabled in ('true', '1', 'yes', 'on')

    @classmethod
    def get_timezone(cls) -> str:
        """Get timezone for scheduler."""
        return os.getenv(cls.ENV_TIMEZONE, 'America/New_York')


# =============================================================================
# Scheduler Service
# =============================================================================

class SchedulerService:
    """
    Background task scheduler for the application.

    Manages scheduled jobs including daily league refresh,
    cache cleanup, and other maintenance tasks.

    Usage:
        scheduler = SchedulerService()
        scheduler.init_app(app)
        scheduler.start()
    """

    def __init__(self):
        """Initialize the scheduler service."""
        self._scheduler: Optional[BackgroundScheduler] = None
        self._app = None
        self._initialized = False

    def init_app(self, app):
        """
        Initialize scheduler with Flask app context.

        Args:
            app: Flask application instance
        """
        self._app = app
        self._scheduler = BackgroundScheduler(
            timezone=SchedulerConfig.get_timezone(),
            job_defaults={
                'coalesce': True,  # Combine missed runs into one
                'max_instances': 1,  # Only one instance at a time
                'misfire_grace_time': 3600  # 1 hour grace period
            }
        )

        # Add event listeners
        self._scheduler.add_listener(
            self._job_executed_listener,
            EVENT_JOB_EXECUTED
        )
        self._scheduler.add_listener(
            self._job_error_listener,
            EVENT_JOB_ERROR
        )
        self._scheduler.add_listener(
            self._job_missed_listener,
            EVENT_JOB_MISSED
        )

        self._initialized = True
        logger.info("Scheduler service initialized")

    def start(self):
        """Start the scheduler and register default jobs."""
        if not self._initialized:
            raise RuntimeError("Scheduler not initialized. Call init_app() first.")

        if not SchedulerConfig.is_enabled():
            logger.info("Scheduler is disabled via configuration")
            return

        if self._scheduler.running:
            logger.warning("Scheduler is already running")
            return

        # Register default jobs
        self._register_default_jobs()

        # Start the scheduler
        self._scheduler.start()
        logger.info(
            f"Scheduler started. Daily refresh scheduled at "
            f"{SchedulerConfig.get_refresh_hour():02d}:{SchedulerConfig.get_refresh_minute():02d} "
            f"({SchedulerConfig.get_timezone()})"
        )

    def shutdown(self, wait: bool = True):
        """
        Shutdown the scheduler.

        Args:
            wait: Whether to wait for running jobs to complete
        """
        if self._scheduler and self._scheduler.running:
            self._scheduler.shutdown(wait=wait)
            logger.info("Scheduler shutdown complete")

    def _register_default_jobs(self):
        """Register the default scheduled jobs."""
        # Daily league refresh
        self._scheduler.add_job(
            func=self._daily_league_refresh_job,
            trigger=CronTrigger(
                hour=SchedulerConfig.get_refresh_hour(),
                minute=SchedulerConfig.get_refresh_minute(),
                timezone=SchedulerConfig.get_timezone()
            ),
            id='daily_league_refresh',
            name='Daily League Refresh',
            replace_existing=True
        )

        # Cache cleanup (every 6 hours)
        self._scheduler.add_job(
            func=self._cache_cleanup_job,
            trigger=IntervalTrigger(hours=6),
            id='cache_cleanup',
            name='Cache Cleanup',
            replace_existing=True
        )

        logger.info("Default scheduled jobs registered")

    # =========================================================================
    # Job Functions
    # =========================================================================

    def _daily_league_refresh_job(self):
        """
        Job: Refresh all active leagues from ESPN.

        This job runs daily to fetch updated data for all leagues.
        """
        logger.info("Starting daily league refresh job")
        start_time = datetime.utcnow()

        results = {
            'total_leagues': 0,
            'successful': 0,
            'failed': 0,
            'errors': []
        }

        try:
            with self._app.app_context():
                results = refresh_all_leagues()

        except Exception as e:
            logger.exception(f"Critical error in daily refresh job: {e}")
            results['errors'].append(f"Critical error: {str(e)}")

        elapsed = (datetime.utcnow() - start_time).total_seconds()

        logger.info(
            f"Daily league refresh completed in {elapsed:.1f}s. "
            f"Total: {results['total_leagues']}, "
            f"Success: {results['successful']}, "
            f"Failed: {results['failed']}"
        )

        if results['errors']:
            logger.warning(f"Refresh errors: {results['errors']}")

        return results

    def _cache_cleanup_job(self):
        """
        Job: Clean up expired cache entries.
        """
        logger.debug("Running cache cleanup job")

        try:
            with self._app.app_context():
                from backend.services.cache_service import get_cache
                cache = get_cache()
                removed = cache.cleanup()
                logger.debug(f"Cache cleanup removed {removed} expired entries")

        except Exception as e:
            logger.error(f"Error in cache cleanup job: {e}")

    # =========================================================================
    # Event Listeners
    # =========================================================================

    def _job_executed_listener(self, event):
        """Handle job execution completion."""
        logger.debug(f"Job {event.job_id} executed successfully")

    def _job_error_listener(self, event):
        """Handle job execution errors."""
        logger.error(
            f"Job {event.job_id} failed with exception: {event.exception}",
            exc_info=event.traceback
        )

    def _job_missed_listener(self, event):
        """Handle missed job executions."""
        logger.warning(f"Job {event.job_id} missed its scheduled run time")

    # =========================================================================
    # Manual Job Triggers
    # =========================================================================

    def trigger_league_refresh(self, league_id: Optional[int] = None):
        """
        Manually trigger a league refresh.

        Args:
            league_id: Optional specific league to refresh. If None, refreshes all.
        """
        if league_id:
            self._scheduler.add_job(
                func=lambda: refresh_single_league(league_id),
                id=f'manual_refresh_{league_id}',
                name=f'Manual Refresh League {league_id}',
                replace_existing=True
            )
            logger.info(f"Triggered manual refresh for league {league_id}")
        else:
            self._scheduler.add_job(
                func=self._daily_league_refresh_job,
                id='manual_refresh_all',
                name='Manual Refresh All Leagues',
                replace_existing=True
            )
            logger.info("Triggered manual refresh for all leagues")

    # =========================================================================
    # Job Management
    # =========================================================================

    def get_jobs(self) -> List[dict]:
        """
        Get information about all scheduled jobs.

        Returns:
            List of job info dictionaries
        """
        if not self._scheduler:
            return []

        jobs = []
        for job in self._scheduler.get_jobs():
            jobs.append({
                'id': job.id,
                'name': job.name,
                'next_run': job.next_run_time.isoformat() if job.next_run_time else None,
                'trigger': str(job.trigger)
            })

        return jobs

    def pause_job(self, job_id: str):
        """Pause a scheduled job."""
        if self._scheduler:
            self._scheduler.pause_job(job_id)
            logger.info(f"Paused job: {job_id}")

    def resume_job(self, job_id: str):
        """Resume a paused job."""
        if self._scheduler:
            self._scheduler.resume_job(job_id)
            logger.info(f"Resumed job: {job_id}")

    def remove_job(self, job_id: str):
        """Remove a scheduled job."""
        if self._scheduler:
            self._scheduler.remove_job(job_id)
            logger.info(f"Removed job: {job_id}")

    @property
    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self._scheduler is not None and self._scheduler.running


# =============================================================================
# League Refresh Functions
# =============================================================================

def refresh_all_leagues() -> dict:
    """
    Refresh data for all active leagues.

    Returns:
        Dictionary with refresh results
    """
    from backend.services.league_service import get_leagues_needing_refresh
    from backend.models import League

    results = {
        'total_leagues': 0,
        'successful': 0,
        'failed': 0,
        'errors': [],
        'refreshed_leagues': []
    }

    try:
        # Get all leagues (or those needing refresh)
        leagues = League.query.all()
        results['total_leagues'] = len(leagues)

        logger.info(f"Found {len(leagues)} leagues to refresh")

        for league in leagues:
            try:
                success = refresh_single_league(league.id)
                if success:
                    results['successful'] += 1
                    results['refreshed_leagues'].append(league.id)
                else:
                    results['failed'] += 1

            except Exception as e:
                results['failed'] += 1
                error_msg = f"League {league.id}: {str(e)}"
                results['errors'].append(error_msg)
                logger.error(f"Failed to refresh league {league.id}: {e}")

    except Exception as e:
        logger.exception(f"Error getting leagues for refresh: {e}")
        results['errors'].append(f"Failed to get leagues: {str(e)}")

    return results


def refresh_single_league(league_id: int) -> bool:
    """
    Refresh data for a single league.

    Args:
        league_id: League ID to refresh

    Returns:
        True if successful, False otherwise
    """
    from backend.services.espn_client import (
        ESPNClient,
        ESPNClientError,
        ESPNAuthenticationError
    )
    from backend.services.league_service import (
        get_league_by_id,
        update_league_settings,
        mark_league_updated
    )
    from backend.services.team_service import create_or_update_team
    from backend.services.player_service import create_or_update_player
    from backend.services.cache_service import get_cache

    logger.info(f"Refreshing league {league_id}")

    try:
        league = get_league_by_id(league_id)
        if not league:
            logger.warning(f"League {league_id} not found")
            return False

        # Connect to ESPN
        espn_client = ESPNClient(
            league_id=league.espn_league_id,
            year=league.season,
            espn_s2=league.espn_s2_cookie,
            swid=league.swid_cookie
        )

        # Update league settings
        settings = espn_client.get_league_settings()
        update_league_settings(
            league_id=league.id,
            user_id=league.user_id,
            league_name=settings.get('name'),
            league_type=settings.get('scoring_type'),
            num_teams=settings.get('size'),
            roster_settings=settings.get('roster_settings'),
            scoring_settings=settings.get('scoring_settings')
        )

        # Update teams and standings
        teams = espn_client.get_teams()
        standings = espn_client.get_standings()
        standings_map = {s['espn_team_id']: s for s in standings}

        for team_data in teams:
            espn_team_id = team_data['espn_team_id']
            standing_data = standings_map.get(espn_team_id, {})

            create_or_update_team(
                league_id=league.id,
                espn_team_id=espn_team_id,
                team_name=team_data['team_name'],
                owner_name=team_data.get('owner_name'),
                current_record=standing_data.get('record'),
                current_standing=standing_data.get('standing')
            )

        # Update players from rosters
        all_rosters = espn_client.get_all_rosters()
        player_ids_seen = set()

        for team_id, roster in all_rosters.items():
            for player_data in roster:
                espn_player_id = player_data['espn_player_id']

                if espn_player_id not in player_ids_seen:
                    create_or_update_player(
                        espn_player_id=espn_player_id,
                        name=player_data['name'],
                        position=player_data.get('position'),
                        nba_team=player_data.get('nba_team'),
                        injury_status=player_data.get('injury_status')
                    )
                    player_ids_seen.add(espn_player_id)

        # Mark league as updated
        mark_league_updated(league.id)

        # Invalidate cache
        cache = get_cache()
        cache.invalidate_league(league.id)

        logger.info(
            f"Successfully refreshed league {league_id}: "
            f"{len(teams)} teams, {len(player_ids_seen)} players"
        )
        return True

    except ESPNAuthenticationError as e:
        logger.warning(f"ESPN auth failed for league {league_id}: {e}")
        return False

    except ESPNClientError as e:
        logger.error(f"ESPN error refreshing league {league_id}: {e}")
        return False

    except Exception as e:
        logger.exception(f"Unexpected error refreshing league {league_id}: {e}")
        return False


# =============================================================================
# Global Scheduler Instance
# =============================================================================

# Singleton scheduler instance
_scheduler_instance: Optional[SchedulerService] = None


def get_scheduler() -> SchedulerService:
    """
    Get the global scheduler instance.

    Returns:
        SchedulerService instance
    """
    global _scheduler_instance
    if _scheduler_instance is None:
        _scheduler_instance = SchedulerService()
    return _scheduler_instance


def init_scheduler(app) -> SchedulerService:
    """
    Initialize and start the scheduler with a Flask app.

    Args:
        app: Flask application instance

    Returns:
        Initialized SchedulerService
    """
    scheduler = get_scheduler()
    scheduler.init_app(app)

    # Only start in the main process (avoid duplicate schedulers in reloader)
    if not app.debug or os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
        scheduler.start()

    return scheduler


def shutdown_scheduler():
    """Shutdown the global scheduler."""
    global _scheduler_instance
    if _scheduler_instance:
        _scheduler_instance.shutdown()
        _scheduler_instance = None

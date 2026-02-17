"""
In-Memory Caching Service for Fantasy Basketball Optimizer.

This module provides a simple TTL-based caching layer for ESPN API responses
to minimize API calls and improve performance.

Features:
- Store data with configurable TTL (time-to-live)
- Automatic expiration of stale entries
- Cache invalidation (specific keys or patterns)
- Thread-safe operations
- Statistics tracking for monitoring

The implementation uses Python dictionaries with timestamps.
Can be upgraded to Redis for production use if needed.

Reference: PRD Section 7.2 - Cache responses with appropriate TTL
"""

import logging
import threading
import time
from datetime import datetime
from typing import Optional, Any, Dict, List, Callable
from functools import wraps

# Set up logging
logger = logging.getLogger(__name__)


# =============================================================================
# Default TTL Values (in seconds)
# =============================================================================

class CacheTTL:
    """Default TTL values for different data types."""

    # League settings rarely change
    LEAGUE_SETTINGS = 3600  # 1 hour

    # Standings update after games
    STANDINGS = 900  # 15 minutes

    # Rosters can change with trades/waivers
    ROSTERS = 600  # 10 minutes

    # Free agents change frequently
    FREE_AGENTS = 300  # 5 minutes

    # Matchups update during games
    MATCHUPS = 300  # 5 minutes

    # Player stats update after games
    PLAYER_STATS = 900  # 15 minutes

    # Short-lived cache for rapid requests
    SHORT = 60  # 1 minute

    # Long-lived cache for static data
    LONG = 86400  # 24 hours


# =============================================================================
# Cache Entry
# =============================================================================

class CacheEntry:
    """Represents a single cache entry with metadata."""

    __slots__ = ['value', 'expires_at', 'created_at', 'hits']

    def __init__(self, value: Any, ttl: int):
        """
        Create a cache entry.

        Args:
            value: The data to cache
            ttl: Time-to-live in seconds
        """
        self.value = value
        self.created_at = time.time()
        self.expires_at = self.created_at + ttl
        self.hits = 0

    @property
    def is_expired(self) -> bool:
        """Check if the entry has expired."""
        return time.time() > self.expires_at

    @property
    def ttl_remaining(self) -> float:
        """Get remaining TTL in seconds."""
        remaining = self.expires_at - time.time()
        return max(0, remaining)

    @property
    def age(self) -> float:
        """Get age of entry in seconds."""
        return time.time() - self.created_at


# =============================================================================
# Cache Service
# =============================================================================

class CacheService:
    """
    In-memory cache service with TTL support.

    Thread-safe implementation using a lock for concurrent access.
    Supports automatic cleanup of expired entries.

    Usage:
        cache = CacheService()
        cache.set('my_key', {'data': 'value'}, ttl=300)
        data = cache.get('my_key')
    """

    def __init__(self, default_ttl: int = 300, cleanup_interval: int = 60):
        """
        Initialize the cache service.

        Args:
            default_ttl: Default TTL in seconds for entries without explicit TTL
            cleanup_interval: How often to run cleanup (in seconds)
        """
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._default_ttl = default_ttl
        self._cleanup_interval = cleanup_interval
        self._last_cleanup = time.time()

        # Statistics
        self._stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'expirations': 0,
        }

        logger.info(f"Cache service initialized with default TTL={default_ttl}s")

    # =========================================================================
    # Core Methods
    # =========================================================================

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve a value from the cache.

        Args:
            key: Cache key

        Returns:
            Cached value if exists and not expired, None otherwise
        """
        self._maybe_cleanup()

        with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._stats['misses'] += 1
                logger.debug(f"Cache MISS: {key}")
                return None

            if entry.is_expired:
                # Remove expired entry
                del self._cache[key]
                self._stats['misses'] += 1
                self._stats['expirations'] += 1
                logger.debug(f"Cache EXPIRED: {key}")
                return None

            # Cache hit
            entry.hits += 1
            self._stats['hits'] += 1
            logger.debug(f"Cache HIT: {key} (age={entry.age:.1f}s, hits={entry.hits})")
            return entry.value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Store a value in the cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (uses default if not specified)
        """
        if ttl is None:
            ttl = self._default_ttl

        with self._lock:
            self._cache[key] = CacheEntry(value, ttl)
            self._stats['sets'] += 1
            logger.debug(f"Cache SET: {key} (ttl={ttl}s)")

    def delete(self, key: str) -> bool:
        """
        Remove a specific entry from the cache.

        Args:
            key: Cache key to remove

        Returns:
            True if key existed and was removed, False otherwise
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._stats['deletes'] += 1
                logger.debug(f"Cache DELETE: {key}")
                return True
            return False

    def exists(self, key: str) -> bool:
        """
        Check if a key exists and is not expired.

        Args:
            key: Cache key

        Returns:
            True if key exists and is valid, False otherwise
        """
        with self._lock:
            entry = self._cache.get(key)
            return entry is not None and not entry.is_expired

    def clear(self) -> int:
        """
        Clear all entries from the cache.

        Returns:
            Number of entries cleared
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            logger.info(f"Cache CLEARED: {count} entries removed")
            return count

    # =========================================================================
    # Pattern-Based Operations
    # =========================================================================

    def delete_pattern(self, pattern: str) -> int:
        """
        Delete all keys matching a pattern (prefix match).

        Args:
            pattern: Key prefix to match

        Returns:
            Number of entries deleted
        """
        with self._lock:
            keys_to_delete = [
                key for key in self._cache.keys()
                if key.startswith(pattern)
            ]

            for key in keys_to_delete:
                del self._cache[key]
                self._stats['deletes'] += 1

            if keys_to_delete:
                logger.debug(f"Cache DELETE PATTERN '{pattern}': {len(keys_to_delete)} entries")

            return len(keys_to_delete)

    def get_keys(self, pattern: Optional[str] = None) -> List[str]:
        """
        Get all cache keys, optionally filtered by prefix.

        Args:
            pattern: Optional key prefix to filter

        Returns:
            List of matching cache keys
        """
        with self._lock:
            if pattern:
                return [k for k in self._cache.keys() if k.startswith(pattern)]
            return list(self._cache.keys())

    # =========================================================================
    # League-Specific Cache Keys
    # =========================================================================

    @staticmethod
    def league_key(league_id: int, data_type: str) -> str:
        """
        Generate a cache key for league-specific data.

        Args:
            league_id: ESPN league ID
            data_type: Type of data (e.g., 'settings', 'standings', 'rosters')

        Returns:
            Formatted cache key
        """
        return f"league:{league_id}:{data_type}"

    @staticmethod
    def team_key(league_id: int, team_id: int, data_type: str) -> str:
        """
        Generate a cache key for team-specific data.

        Args:
            league_id: ESPN league ID
            team_id: ESPN team ID
            data_type: Type of data

        Returns:
            Formatted cache key
        """
        return f"league:{league_id}:team:{team_id}:{data_type}"

    @staticmethod
    def player_key(player_id: int, data_type: str) -> str:
        """
        Generate a cache key for player-specific data.

        Args:
            player_id: ESPN player ID
            data_type: Type of data

        Returns:
            Formatted cache key
        """
        return f"player:{player_id}:{data_type}"

    def invalidate_league(self, league_id: int) -> int:
        """
        Invalidate all cached data for a specific league.

        Args:
            league_id: ESPN league ID

        Returns:
            Number of entries invalidated
        """
        pattern = f"league:{league_id}:"
        count = self.delete_pattern(pattern)
        logger.info(f"Invalidated {count} cache entries for league {league_id}")
        return count

    # =========================================================================
    # Cache Decorator
    # =========================================================================

    def cached(self, ttl: Optional[int] = None, key_prefix: str = ""):
        """
        Decorator to cache function results.

        Args:
            ttl: Time-to-live for cached results
            key_prefix: Prefix for cache keys

        Usage:
            @cache.cached(ttl=300, key_prefix="standings")
            def get_standings(league_id):
                ...
        """
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Build cache key from function name and arguments
                key_parts = [key_prefix or func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = ":".join(key_parts)

                # Try to get from cache
                cached_value = self.get(cache_key)
                if cached_value is not None:
                    return cached_value

                # Call function and cache result
                result = func(*args, **kwargs)
                self.set(cache_key, result, ttl)
                return result

            return wrapper
        return decorator

    # =========================================================================
    # Cleanup and Maintenance
    # =========================================================================

    def _maybe_cleanup(self) -> None:
        """Run cleanup if enough time has passed since last cleanup."""
        current_time = time.time()
        if current_time - self._last_cleanup > self._cleanup_interval:
            self._cleanup_expired()
            self._last_cleanup = current_time

    def _cleanup_expired(self) -> int:
        """
        Remove all expired entries from the cache.

        Returns:
            Number of entries removed
        """
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired
            ]

            for key in expired_keys:
                del self._cache[key]
                self._stats['expirations'] += 1

            if expired_keys:
                logger.debug(f"Cleanup removed {len(expired_keys)} expired entries")

            return len(expired_keys)

    def cleanup(self) -> int:
        """
        Manually trigger cleanup of expired entries.

        Returns:
            Number of entries removed
        """
        return self._cleanup_expired()

    # =========================================================================
    # Statistics and Monitoring
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            total_requests = self._stats['hits'] + self._stats['misses']
            hit_rate = (self._stats['hits'] / total_requests * 100) if total_requests > 0 else 0

            return {
                'entries': len(self._cache),
                'hits': self._stats['hits'],
                'misses': self._stats['misses'],
                'hit_rate': round(hit_rate, 2),
                'sets': self._stats['sets'],
                'deletes': self._stats['deletes'],
                'expirations': self._stats['expirations'],
            }

    def get_entry_info(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a cache entry.

        Args:
            key: Cache key

        Returns:
            Dictionary with entry info or None if not found
        """
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None

            return {
                'key': key,
                'is_expired': entry.is_expired,
                'ttl_remaining': round(entry.ttl_remaining, 1),
                'age': round(entry.age, 1),
                'hits': entry.hits,
                'created_at': datetime.fromtimestamp(entry.created_at).isoformat(),
                'expires_at': datetime.fromtimestamp(entry.expires_at).isoformat(),
            }

    def __len__(self) -> int:
        """Return the number of entries in the cache."""
        with self._lock:
            return len(self._cache)

    def __contains__(self, key: str) -> bool:
        """Check if a key is in the cache (and not expired)."""
        return self.exists(key)


# =============================================================================
# Global Cache Instance
# =============================================================================

# Singleton cache instance for the application
_cache_instance: Optional[CacheService] = None


def get_cache() -> CacheService:
    """
    Get the global cache instance (singleton pattern).

    Returns:
        The global CacheService instance
    """
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = CacheService()
    return _cache_instance


def reset_cache() -> None:
    """Reset the global cache instance (useful for testing)."""
    global _cache_instance
    if _cache_instance is not None:
        _cache_instance.clear()
    _cache_instance = None

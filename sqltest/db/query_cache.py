"""Intelligent query result caching with TTL and cache management."""

import hashlib
import logging
import pickle
import time
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from threading import Lock, Timer
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import weakref

from sqltest.db.base import QueryResult

logger = logging.getLogger(__name__)


class CacheStrategy(str, Enum):
    """Cache invalidation strategies."""
    TTL = "ttl"                    # Time-to-live based
    ADAPTIVE = "adaptive"          # Based on query patterns
    FREQUENCY = "frequency"        # Based on execution frequency
    SIZE_BASED = "size_based"      # Based on result size
    SMART = "smart"               # Combination of strategies


class CacheEvictionPolicy(str, Enum):
    """Cache eviction policies when memory is full."""
    LRU = "lru"                   # Least Recently Used
    LFU = "lfu"                   # Least Frequently Used
    FIFO = "fifo"                 # First In, First Out
    SIZE = "size"                 # Largest results first
    TTL = "ttl"                   # Shortest TTL first
    SMART = "smart"               # Intelligent combination


@dataclass
class CacheEntry:
    """Represents a cached query result."""
    query_hash: str
    query_text: str
    result_data: QueryResult
    created_at: datetime
    expires_at: datetime
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    size_bytes: int = 0
    database_name: str = ""
    table_dependencies: Set[str] = field(default_factory=set)
    parameters_hash: Optional[str] = None
    cache_strategy: CacheStrategy = CacheStrategy.TTL


@dataclass
class CacheStats:
    """Cache performance statistics."""
    total_queries: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_invalidations: int = 0
    cache_evictions: int = 0
    total_entries: int = 0
    total_size_bytes: int = 0
    avg_ttl_seconds: float = 0.0
    hit_rate: float = 0.0
    memory_usage_mb: float = 0.0

    def update_hit_rate(self):
        """Update hit rate based on current stats."""
        if self.total_queries > 0:
            self.hit_rate = self.cache_hits / self.total_queries
        else:
            self.hit_rate = 0.0


class QueryResultCache:
    """Intelligent query result cache with multiple strategies and policies."""

    def __init__(
        self,
        max_size_mb: int = 100,
        default_ttl_seconds: int = 300,
        max_entries: int = 1000,
        eviction_policy: CacheEvictionPolicy = CacheEvictionPolicy.SMART,
        cleanup_interval_seconds: int = 60
    ):
        """Initialize query result cache.

        Args:
            max_size_mb: Maximum cache size in megabytes
            default_ttl_seconds: Default TTL for cached results
            max_entries: Maximum number of cache entries
            eviction_policy: Policy for evicting entries when cache is full
            cleanup_interval_seconds: Interval for automatic cleanup
        """
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.default_ttl_seconds = default_ttl_seconds
        self.max_entries = max_entries
        self.eviction_policy = eviction_policy

        # Thread-safe storage
        self._lock = Lock()
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._stats = CacheStats()

        # Access tracking for LFU/LRU
        self._access_times: Dict[str, datetime] = {}
        self._access_counts: Dict[str, int] = defaultdict(int)

        # Table dependency tracking for intelligent invalidation
        self._table_dependencies: Dict[str, Set[str]] = defaultdict(set)

        # Cleanup timer
        self._cleanup_timer: Optional[Timer] = None
        self._cleanup_interval = cleanup_interval_seconds
        self._start_cleanup_timer()

    def _start_cleanup_timer(self):
        """Start the automatic cleanup timer."""
        if self._cleanup_timer is not None:
            self._cleanup_timer.cancel()

        self._cleanup_timer = Timer(self._cleanup_interval, self._periodic_cleanup)
        self._cleanup_timer.daemon = True
        self._cleanup_timer.start()

    def _periodic_cleanup(self):
        """Periodic cleanup of expired entries."""
        try:
            self._cleanup_expired_entries()
            self._start_cleanup_timer()  # Restart timer
        except Exception as e:
            logger.error(f"Error in periodic cache cleanup: {e}")

    def _generate_cache_key(self, query: str, parameters: Optional[Dict[str, Any]] = None,
                          database_name: str = "") -> str:
        """Generate cache key for query and parameters.

        Args:
            query: SQL query text
            parameters: Query parameters
            database_name: Database name

        Returns:
            Cache key string
        """
        # Normalize query
        normalized_query = self._normalize_query(query)

        # Create key components
        key_data = {
            'query': normalized_query,
            'database': database_name,
            'parameters': parameters or {}
        }

        # Generate hash
        key_string = pickle.dumps(key_data, protocol=pickle.HIGHEST_PROTOCOL)
        return hashlib.sha256(key_string).hexdigest()

    def _normalize_query(self, query: str) -> str:
        """Normalize query for caching consistency."""
        # Remove extra whitespace
        normalized = ' '.join(query.split())

        # Convert to uppercase for consistency
        return normalized.upper()

    def _calculate_result_size(self, result: QueryResult) -> int:
        """Calculate approximate size of query result in bytes."""
        try:
            # Serialize the result to estimate size
            return len(pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL))
        except Exception:
            # Fallback estimation
            if hasattr(result, 'data') and result.data:
                return len(str(result.data)) * 2  # Rough estimate
            return 1024  # Default size

    def _determine_ttl(self, query: str, result: QueryResult, strategy: CacheStrategy) -> int:
        """Determine TTL based on caching strategy.

        Args:
            query: SQL query text
            result: Query result
            strategy: Caching strategy

        Returns:
            TTL in seconds
        """
        if strategy == CacheStrategy.TTL:
            return self.default_ttl_seconds

        elif strategy == CacheStrategy.ADAPTIVE:
            # Adaptive TTL based on query complexity and result size
            base_ttl = self.default_ttl_seconds

            # Longer TTL for expensive queries
            if 'JOIN' in query.upper():
                base_ttl *= 2
            if 'GROUP BY' in query.upper() or 'ORDER BY' in query.upper():
                base_ttl *= 1.5

            # Shorter TTL for large results
            result_size = self._calculate_result_size(result)
            if result_size > 1024 * 1024:  # > 1MB
                base_ttl = int(base_ttl * 0.5)

            return max(60, min(base_ttl, 3600))  # Between 1 minute and 1 hour

        elif strategy == CacheStrategy.FREQUENCY:
            # TTL based on how frequently this query is executed
            query_hash = hashlib.md5(query.encode()).hexdigest()
            frequency = self._access_counts.get(query_hash, 0)

            if frequency > 100:  # Very frequent
                return self.default_ttl_seconds * 3
            elif frequency > 50:  # Frequent
                return self.default_ttl_seconds * 2
            elif frequency > 10:  # Moderate
                return self.default_ttl_seconds
            else:  # Infrequent
                return self.default_ttl_seconds // 2

        elif strategy == CacheStrategy.SIZE_BASED:
            # TTL based on result size
            result_size = self._calculate_result_size(result)
            if result_size < 1024:  # < 1KB
                return self.default_ttl_seconds * 2
            elif result_size < 1024 * 100:  # < 100KB
                return self.default_ttl_seconds
            else:  # Large results
                return self.default_ttl_seconds // 2

        elif strategy == CacheStrategy.SMART:
            # Combination of strategies
            adaptive_ttl = self._determine_ttl(query, result, CacheStrategy.ADAPTIVE)
            frequency_ttl = self._determine_ttl(query, result, CacheStrategy.FREQUENCY)
            size_ttl = self._determine_ttl(query, result, CacheStrategy.SIZE_BASED)

            # Weighted average
            return int((adaptive_ttl * 0.4 + frequency_ttl * 0.4 + size_ttl * 0.2))

        return self.default_ttl_seconds

    def _extract_table_dependencies(self, query: str) -> Set[str]:
        """Extract table names that this query depends on.

        Args:
            query: SQL query text

        Returns:
            Set of table names
        """
        import re

        # Simple regex to extract table names from common SQL patterns
        patterns = [
            r'\bFROM\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            r'\bJOIN\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            r'\bUPDATE\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            r'\bINTO\s+([a-zA-Z_][a-zA-Z0-9_]*)',
        ]

        tables = set()
        query_upper = query.upper()

        for pattern in patterns:
            matches = re.findall(pattern, query_upper)
            tables.update(matches)

        return tables

    def get(self, query: str, parameters: Optional[Dict[str, Any]] = None,
           database_name: str = "") -> Optional[QueryResult]:
        """Get cached query result.

        Args:
            query: SQL query text
            parameters: Query parameters
            database_name: Database name

        Returns:
            Cached QueryResult or None if not found/expired
        """
        cache_key = self._generate_cache_key(query, parameters, database_name)

        with self._lock:
            self._stats.total_queries += 1

            if cache_key not in self._cache:
                self._stats.cache_misses += 1
                self._stats.update_hit_rate()
                return None

            entry = self._cache[cache_key]

            # Check if expired
            if datetime.now() >= entry.expires_at:
                self._remove_entry(cache_key)
                self._stats.cache_misses += 1
                self._stats.update_hit_rate()
                return None

            # Update access tracking
            entry.access_count += 1
            entry.last_accessed = datetime.now()
            self._access_times[cache_key] = datetime.now()
            self._access_counts[cache_key] += 1

            # Move to end for LRU
            self._cache.move_to_end(cache_key)

            self._stats.cache_hits += 1
            self._stats.update_hit_rate()

            logger.debug(f"Cache hit for query: {query[:50]}...")
            return entry.result_data

    def put(self, query: str, result: QueryResult,
           parameters: Optional[Dict[str, Any]] = None,
           database_name: str = "",
           ttl_seconds: Optional[int] = None,
           strategy: CacheStrategy = CacheStrategy.SMART) -> bool:
        """Cache query result.

        Args:
            query: SQL query text
            result: Query result to cache
            parameters: Query parameters
            database_name: Database name
            ttl_seconds: Custom TTL, or None to use strategy-based TTL
            strategy: Caching strategy to use

        Returns:
            True if cached successfully, False otherwise
        """
        if not result or not hasattr(result, 'data'):
            return False

        cache_key = self._generate_cache_key(query, parameters, database_name)
        result_size = self._calculate_result_size(result)

        # Determine TTL
        if ttl_seconds is None:
            ttl_seconds = self._determine_ttl(query, result, strategy)

        expires_at = datetime.now() + timedelta(seconds=ttl_seconds)

        # Extract table dependencies
        table_deps = self._extract_table_dependencies(query)

        with self._lock:
            # Check if we need to make room
            if not self._ensure_capacity(result_size):
                logger.warning(f"Cannot cache query result - insufficient capacity")
                return False

            # Create cache entry
            entry = CacheEntry(
                query_hash=cache_key,
                query_text=query,
                result_data=result,
                created_at=datetime.now(),
                expires_at=expires_at,
                size_bytes=result_size,
                database_name=database_name,
                table_dependencies=table_deps,
                parameters_hash=hashlib.md5(str(parameters or {}).encode()).hexdigest(),
                cache_strategy=strategy
            )

            # Remove existing entry if present
            if cache_key in self._cache:
                self._remove_entry(cache_key)

            # Add new entry
            self._cache[cache_key] = entry

            # Update table dependencies
            for table in table_deps:
                self._table_dependencies[table].add(cache_key)

            # Update statistics
            self._stats.total_entries = len(self._cache)
            self._stats.total_size_bytes += result_size
            self._stats.memory_usage_mb = self._stats.total_size_bytes / (1024 * 1024)

            logger.debug(f"Cached query result: {query[:50]}... (TTL: {ttl_seconds}s)")
            return True

    def _ensure_capacity(self, required_size: int) -> bool:
        """Ensure cache has capacity for new entry.

        Args:
            required_size: Size in bytes required

        Returns:
            True if capacity is available, False otherwise
        """
        # Check entry count limit
        while len(self._cache) >= self.max_entries:
            if not self._evict_entry():
                return False

        # Check size limit
        while (self._stats.total_size_bytes + required_size) > self.max_size_bytes:
            if not self._evict_entry():
                return False

        return True

    def _evict_entry(self) -> bool:
        """Evict an entry based on eviction policy.

        Returns:
            True if an entry was evicted, False if cache is empty
        """
        if not self._cache:
            return False

        if self.eviction_policy == CacheEvictionPolicy.LRU:
            # Remove least recently used (first in OrderedDict)
            cache_key = next(iter(self._cache))

        elif self.eviction_policy == CacheEvictionPolicy.LFU:
            # Remove least frequently used
            cache_key = min(self._cache.keys(),
                          key=lambda k: self._cache[k].access_count)

        elif self.eviction_policy == CacheEvictionPolicy.FIFO:
            # Remove oldest entry
            cache_key = min(self._cache.keys(),
                          key=lambda k: self._cache[k].created_at)

        elif self.eviction_policy == CacheEvictionPolicy.SIZE:
            # Remove largest entry
            cache_key = max(self._cache.keys(),
                          key=lambda k: self._cache[k].size_bytes)

        elif self.eviction_policy == CacheEvictionPolicy.TTL:
            # Remove entry with shortest remaining TTL
            now = datetime.now()
            cache_key = min(self._cache.keys(),
                          key=lambda k: (self._cache[k].expires_at - now).total_seconds())

        elif self.eviction_policy == CacheEvictionPolicy.SMART:
            # Smart eviction based on multiple factors
            cache_key = self._smart_eviction()

        else:
            cache_key = next(iter(self._cache))  # Default to FIFO

        self._remove_entry(cache_key)
        self._stats.cache_evictions += 1
        return True

    def _smart_eviction(self) -> str:
        """Smart eviction algorithm considering multiple factors."""
        if not self._cache:
            return next(iter(self._cache))

        now = datetime.now()
        scores = {}

        for cache_key, entry in self._cache.items():
            # Calculate eviction score (higher = more likely to evict)
            score = 0

            # Age factor (older = higher score)
            age_seconds = (now - entry.created_at).total_seconds()
            score += age_seconds / 3600  # Hours

            # Size factor (larger = higher score)
            score += entry.size_bytes / (1024 * 1024)  # MB

            # Access frequency factor (less frequent = higher score)
            access_freq = entry.access_count / max(1, age_seconds / 3600)
            score += max(0, 10 - access_freq)

            # TTL factor (shorter remaining TTL = higher score)
            remaining_ttl = (entry.expires_at - now).total_seconds()
            score += max(0, 10 - remaining_ttl / 60)  # Minutes

            scores[cache_key] = score

        # Return key with highest eviction score
        return max(scores.keys(), key=lambda k: scores[k])

    def _remove_entry(self, cache_key: str):
        """Remove entry from cache and cleanup references."""
        if cache_key not in self._cache:
            return

        entry = self._cache[cache_key]

        # Update statistics
        self._stats.total_size_bytes -= entry.size_bytes

        # Remove from table dependencies
        for table in entry.table_dependencies:
            self._table_dependencies[table].discard(cache_key)
            if not self._table_dependencies[table]:
                del self._table_dependencies[table]

        # Remove from access tracking
        self._access_times.pop(cache_key, None)
        self._access_counts.pop(cache_key, None)

        # Remove from cache
        del self._cache[cache_key]

        # Update statistics
        self._stats.total_entries = len(self._cache)
        self._stats.memory_usage_mb = self._stats.total_size_bytes / (1024 * 1024)

    def invalidate_by_table(self, table_name: str):
        """Invalidate all cached queries that depend on a specific table.

        Args:
            table_name: Name of the table that was modified
        """
        with self._lock:
            if table_name.upper() in self._table_dependencies:
                cache_keys_to_remove = list(self._table_dependencies[table_name.upper()])

                for cache_key in cache_keys_to_remove:
                    self._remove_entry(cache_key)
                    self._stats.cache_invalidations += 1

                logger.info(f"Invalidated {len(cache_keys_to_remove)} cache entries for table: {table_name}")

    def invalidate_by_pattern(self, pattern: str):
        """Invalidate cached queries matching a pattern.

        Args:
            pattern: SQL pattern to match (case-insensitive)
        """
        with self._lock:
            cache_keys_to_remove = []
            pattern_upper = pattern.upper()

            for cache_key, entry in self._cache.items():
                if pattern_upper in entry.query_text.upper():
                    cache_keys_to_remove.append(cache_key)

            for cache_key in cache_keys_to_remove:
                self._remove_entry(cache_key)
                self._stats.cache_invalidations += 1

            logger.info(f"Invalidated {len(cache_keys_to_remove)} cache entries matching pattern: {pattern}")

    def _cleanup_expired_entries(self):
        """Clean up expired cache entries."""
        with self._lock:
            now = datetime.now()
            expired_keys = []

            for cache_key, entry in self._cache.items():
                if now >= entry.expires_at:
                    expired_keys.append(cache_key)

            for cache_key in expired_keys:
                self._remove_entry(cache_key)

            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

    def clear(self):
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            self._table_dependencies.clear()
            self._access_times.clear()
            self._access_counts.clear()

            # Reset statistics
            self._stats = CacheStats()

            logger.info("Cache cleared")

    def get_statistics(self) -> Dict[str, Any]:
        """Get cache performance statistics.

        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            now = datetime.now()

            # Calculate average TTL
            total_ttl = 0
            active_entries = 0

            for entry in self._cache.values():
                remaining_ttl = (entry.expires_at - now).total_seconds()
                if remaining_ttl > 0:
                    total_ttl += remaining_ttl
                    active_entries += 1

            avg_ttl = total_ttl / active_entries if active_entries > 0 else 0

            return {
                'total_queries': self._stats.total_queries,
                'cache_hits': self._stats.cache_hits,
                'cache_misses': self._stats.cache_misses,
                'hit_rate': self._stats.hit_rate,
                'cache_invalidations': self._stats.cache_invalidations,
                'cache_evictions': self._stats.cache_evictions,
                'total_entries': len(self._cache),
                'max_entries': self.max_entries,
                'total_size_mb': self._stats.total_size_bytes / (1024 * 1024),
                'max_size_mb': self.max_size_bytes / (1024 * 1024),
                'memory_usage_percent': (self._stats.total_size_bytes / self.max_size_bytes) * 100,
                'avg_ttl_seconds': avg_ttl,
                'eviction_policy': self.eviction_policy.value,
                'tracked_tables': len(self._table_dependencies),
                'most_accessed_queries': self._get_top_accessed_queries(5),
                'largest_entries': self._get_largest_entries(5)
            }

    def _get_top_accessed_queries(self, limit: int) -> List[Dict[str, Any]]:
        """Get most frequently accessed cached queries."""
        sorted_entries = sorted(self._cache.items(),
                              key=lambda x: x[1].access_count,
                              reverse=True)

        return [
            {
                'query_hash': entry.query_hash[:12],
                'query_preview': entry.query_text[:100] + "..." if len(entry.query_text) > 100 else entry.query_text,
                'access_count': entry.access_count,
                'size_mb': entry.size_bytes / (1024 * 1024),
                'created_at': entry.created_at.isoformat()
            }
            for _, entry in sorted_entries[:limit]
        ]

    def _get_largest_entries(self, limit: int) -> List[Dict[str, Any]]:
        """Get largest cached entries by size."""
        sorted_entries = sorted(self._cache.items(),
                              key=lambda x: x[1].size_bytes,
                              reverse=True)

        return [
            {
                'query_hash': entry.query_hash[:12],
                'query_preview': entry.query_text[:100] + "..." if len(entry.query_text) > 100 else entry.query_text,
                'size_mb': entry.size_bytes / (1024 * 1024),
                'access_count': entry.access_count,
                'created_at': entry.created_at.isoformat()
            }
            for _, entry in sorted_entries[:limit]
        ]

    def shutdown(self):
        """Shutdown cache and cleanup resources."""
        if self._cleanup_timer:
            self._cleanup_timer.cancel()
            self._cleanup_timer = None

        self.clear()
        logger.info("Query cache shutdown complete")
"""Query performance analysis and optimization engine."""

import hashlib
import logging
import re
import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from threading import Lock
from typing import Dict, List, Optional, Tuple, Any, Set, Union

from sqlalchemy import text
from sqlalchemy.sql.expression import TextClause

logger = logging.getLogger(__name__)


class QueryComplexity(str, Enum):
    """Query complexity levels."""
    SIMPLE = "simple"          # SELECT with basic WHERE
    MODERATE = "moderate"      # JOINs, subqueries, basic aggregations
    COMPLEX = "complex"        # Multiple JOINs, CTEs, window functions
    VERY_COMPLEX = "very_complex"  # Heavy nested queries, multiple CTEs


class QueryCategory(str, Enum):
    """Query operation categories."""
    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    DDL = "ddl"              # CREATE, ALTER, DROP
    TRANSACTION = "transaction"  # BEGIN, COMMIT, ROLLBACK
    UTILITY = "utility"      # ANALYZE, VACUUM, etc.


@dataclass
class QueryMetrics:
    """Detailed metrics for a query execution."""
    query_hash: str
    sql_text: str
    category: QueryCategory
    complexity: QueryComplexity
    execution_time_ms: float
    rows_affected: int
    rows_examined: Optional[int] = None
    database_name: str = ""
    table_names: List[str] = field(default_factory=list)
    index_usage: Dict[str, Any] = field(default_factory=dict)
    execution_plan: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)
    connection_id: Optional[str] = None
    user_context: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


@dataclass
class QueryPerformanceStats:
    """Aggregated performance statistics for queries."""
    query_hash: str
    sql_pattern: str
    category: QueryCategory
    complexity: QueryComplexity
    execution_count: int = 0
    total_execution_time_ms: float = 0.0
    min_execution_time_ms: float = float('inf')
    max_execution_time_ms: float = 0.0
    avg_execution_time_ms: float = 0.0
    median_execution_time_ms: float = 0.0
    p95_execution_time_ms: float = 0.0
    p99_execution_time_ms: float = 0.0
    total_rows_affected: int = 0
    error_count: int = 0
    last_executed: Optional[datetime] = None
    first_executed: Optional[datetime] = None
    frequent_tables: List[str] = field(default_factory=list)
    optimization_suggestions: List[str] = field(default_factory=list)


@dataclass
class QueryOptimizationSuggestion:
    """Query optimization suggestion."""
    query_hash: str
    suggestion_type: str
    description: str
    potential_impact: str  # "high", "medium", "low"
    implementation_effort: str  # "easy", "moderate", "complex"
    details: Dict[str, Any] = field(default_factory=dict)


class QueryAnalyzer:
    """Analyzes and tracks query performance with optimization suggestions."""

    def __init__(self, max_history_size: int = 10000, analysis_window_hours: int = 24):
        """Initialize query analyzer.

        Args:
            max_history_size: Maximum number of query executions to keep in memory
            analysis_window_hours: Time window for performance analysis
        """
        self.max_history_size = max_history_size
        self.analysis_window = timedelta(hours=analysis_window_hours)

        # Thread-safe storage
        self._lock = Lock()
        self._query_history: deque = deque(maxlen=max_history_size)
        self._query_stats: Dict[str, QueryPerformanceStats] = {}
        self._recent_executions: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

        # Pattern matching for query analysis
        self._query_patterns = self._compile_query_patterns()

    def _compile_query_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for query analysis."""
        return {
            'select': re.compile(r'\bSELECT\b', re.IGNORECASE),
            'insert': re.compile(r'\bINSERT\b', re.IGNORECASE),
            'update': re.compile(r'\bUPDATE\b', re.IGNORECASE),
            'delete': re.compile(r'\bDELETE\b', re.IGNORECASE),
            'join': re.compile(r'\b(INNER|LEFT|RIGHT|FULL|CROSS)\s+JOIN\b', re.IGNORECASE),
            'subquery': re.compile(r'\(\s*SELECT\b', re.IGNORECASE),
            'cte': re.compile(r'\bWITH\s+\w+\s+AS\s*\(', re.IGNORECASE),
            'window_function': re.compile(r'\b(ROW_NUMBER|RANK|DENSE_RANK|LAG|LEAD|FIRST_VALUE|LAST_VALUE)\s*\(', re.IGNORECASE),
            'aggregate': re.compile(r'\b(COUNT|SUM|AVG|MIN|MAX|GROUP_CONCAT)\s*\(', re.IGNORECASE),
            'table_name': re.compile(r'\b(?:FROM|JOIN|UPDATE|INTO)\s+([a-zA-Z_][a-zA-Z0-9_]*)', re.IGNORECASE),
        }

    def analyze_query(self, sql: str, execution_time_ms: float, rows_affected: int = 0,
                     database_name: str = "", connection_id: Optional[str] = None,
                     parameters: Optional[Dict[str, Any]] = None) -> QueryMetrics:
        """Analyze a query execution and return metrics.

        Args:
            sql: SQL query text
            execution_time_ms: Execution time in milliseconds
            rows_affected: Number of rows affected/returned
            database_name: Database name where query was executed
            connection_id: Connection identifier
            parameters: Query parameters

        Returns:
            QueryMetrics instance with analysis results
        """
        # Normalize and hash the query
        normalized_sql = self._normalize_query(sql)
        query_hash = self._hash_query(normalized_sql)

        # Categorize the query
        category = self._categorize_query(sql)
        complexity = self._assess_complexity(sql)
        table_names = self._extract_table_names(sql)

        # Create metrics
        metrics = QueryMetrics(
            query_hash=query_hash,
            sql_text=sql,
            category=category,
            complexity=complexity,
            execution_time_ms=execution_time_ms,
            rows_affected=rows_affected,
            database_name=database_name,
            table_names=table_names,
            connection_id=connection_id,
            parameters=parameters
        )

        # Update statistics
        with self._lock:
            self._update_query_stats(metrics)
            self._query_history.append(metrics)
            self._recent_executions[query_hash].append(metrics)

        return metrics

    def _normalize_query(self, sql: str) -> str:
        """Normalize SQL query for pattern matching."""
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', sql.strip())

        # Replace string literals with placeholders
        normalized = re.sub(r"'[^']*'", "'?'", normalized)

        # Replace numeric literals with placeholders
        normalized = re.sub(r'\b\d+\b', '?', normalized)

        # Replace parameter placeholders
        normalized = re.sub(r':\w+', '?', normalized)
        normalized = re.sub(r'\$\d+', '?', normalized)

        return normalized.upper()

    def _hash_query(self, normalized_sql: str) -> str:
        """Generate hash for normalized query."""
        return hashlib.md5(normalized_sql.encode()).hexdigest()[:12]

    def _categorize_query(self, sql: str) -> QueryCategory:
        """Categorize query by operation type."""
        sql_upper = sql.upper().strip()

        if sql_upper.startswith('SELECT'):
            return QueryCategory.SELECT
        elif sql_upper.startswith('INSERT'):
            return QueryCategory.INSERT
        elif sql_upper.startswith('UPDATE'):
            return QueryCategory.UPDATE
        elif sql_upper.startswith('DELETE'):
            return QueryCategory.DELETE
        elif any(sql_upper.startswith(cmd) for cmd in ['CREATE', 'ALTER', 'DROP']):
            return QueryCategory.DDL
        elif any(sql_upper.startswith(cmd) for cmd in ['BEGIN', 'COMMIT', 'ROLLBACK']):
            return QueryCategory.TRANSACTION
        else:
            return QueryCategory.UTILITY

    def _assess_complexity(self, sql: str) -> QueryComplexity:
        """Assess query complexity based on SQL features."""
        complexity_score = 0

        # Check for various SQL features
        features = {
            'join': 2,
            'subquery': 3,
            'cte': 4,
            'window_function': 5,
            'aggregate': 1,
        }

        for feature, score in features.items():
            matches = len(self._query_patterns[feature].findall(sql))
            complexity_score += matches * score

        # Classify based on score
        if complexity_score == 0:
            return QueryComplexity.SIMPLE
        elif complexity_score <= 5:
            return QueryComplexity.MODERATE
        elif complexity_score <= 15:
            return QueryComplexity.COMPLEX
        else:
            return QueryComplexity.VERY_COMPLEX

    def _extract_table_names(self, sql: str) -> List[str]:
        """Extract table names from SQL query."""
        matches = self._query_patterns['table_name'].findall(sql)
        # Remove duplicates and common SQL keywords
        tables = []
        sql_keywords = {'SELECT', 'WHERE', 'ORDER', 'GROUP', 'HAVING', 'LIMIT', 'OFFSET'}

        for match in matches:
            table_name = match.upper().strip()
            if table_name not in sql_keywords and table_name not in tables:
                tables.append(table_name)

        return tables[:10]  # Limit to 10 tables

    def _update_query_stats(self, metrics: QueryMetrics):
        """Update aggregated statistics for a query."""
        query_hash = metrics.query_hash

        if query_hash not in self._query_stats:
            self._query_stats[query_hash] = QueryPerformanceStats(
                query_hash=query_hash,
                sql_pattern=metrics.sql_text[:200] + "..." if len(metrics.sql_text) > 200 else metrics.sql_text,
                category=metrics.category,
                complexity=metrics.complexity,
                first_executed=metrics.timestamp
            )

        stats = self._query_stats[query_hash]

        # Update execution statistics
        stats.execution_count += 1
        stats.total_execution_time_ms += metrics.execution_time_ms
        stats.total_rows_affected += metrics.rows_affected
        stats.last_executed = metrics.timestamp

        # Update min/max execution times
        stats.min_execution_time_ms = min(stats.min_execution_time_ms, metrics.execution_time_ms)
        stats.max_execution_time_ms = max(stats.max_execution_time_ms, metrics.execution_time_ms)

        # Calculate average
        stats.avg_execution_time_ms = stats.total_execution_time_ms / stats.execution_count

        # Update percentiles (simplified calculation)
        recent_times = [m.execution_time_ms for m in self._recent_executions[query_hash]]
        if recent_times:
            recent_times.sort()
            n = len(recent_times)
            stats.median_execution_time_ms = recent_times[n // 2]
            stats.p95_execution_time_ms = recent_times[int(n * 0.95)] if n > 1 else recent_times[0]
            stats.p99_execution_time_ms = recent_times[int(n * 0.99)] if n > 1 else recent_times[0]

        # Update table usage
        for table in metrics.table_names:
            if table not in stats.frequent_tables:
                stats.frequent_tables.append(table)

    def get_query_statistics(self, query_hash: Optional[str] = None,
                           category: Optional[QueryCategory] = None,
                           limit: int = 100) -> List[QueryPerformanceStats]:
        """Get query performance statistics.

        Args:
            query_hash: Specific query hash to get stats for
            category: Filter by query category
            limit: Maximum number of results

        Returns:
            List of QueryPerformanceStats sorted by total execution time
        """
        with self._lock:
            if query_hash:
                stats = self._query_stats.get(query_hash)
                return [stats] if stats else []

            # Filter by category if specified
            filtered_stats = []
            for stats in self._query_stats.values():
                if category is None or stats.category == category:
                    filtered_stats.append(stats)

            # Sort by total execution time (descending)
            filtered_stats.sort(key=lambda s: s.total_execution_time_ms, reverse=True)

            return filtered_stats[:limit]

    def get_slow_queries(self, threshold_ms: float = 1000, limit: int = 50) -> List[QueryPerformanceStats]:
        """Get queries that exceed the execution time threshold.

        Args:
            threshold_ms: Execution time threshold in milliseconds
            limit: Maximum number of results

        Returns:
            List of slow queries sorted by average execution time
        """
        with self._lock:
            slow_queries = []
            for stats in self._query_stats.values():
                if stats.avg_execution_time_ms > threshold_ms:
                    slow_queries.append(stats)

            # Sort by average execution time (descending)
            slow_queries.sort(key=lambda s: s.avg_execution_time_ms, reverse=True)

            return slow_queries[:limit]

    def get_frequent_queries(self, min_executions: int = 10, limit: int = 50) -> List[QueryPerformanceStats]:
        """Get frequently executed queries.

        Args:
            min_executions: Minimum execution count threshold
            limit: Maximum number of results

        Returns:
            List of frequent queries sorted by execution count
        """
        with self._lock:
            frequent_queries = []
            for stats in self._query_stats.values():
                if stats.execution_count >= min_executions:
                    frequent_queries.append(stats)

            # Sort by execution count (descending)
            frequent_queries.sort(key=lambda s: s.execution_count, reverse=True)

            return frequent_queries[:limit]

    def analyze_query_patterns(self) -> Dict[str, Any]:
        """Analyze query execution patterns and trends.

        Returns:
            Dictionary with pattern analysis results
        """
        with self._lock:
            # Get recent queries (last 24 hours)
            cutoff_time = datetime.now() - self.analysis_window
            recent_queries = [q for q in self._query_history if q.timestamp > cutoff_time]

            if not recent_queries:
                return {'message': 'No recent queries to analyze'}

            # Category distribution
            category_counts = defaultdict(int)
            category_times = defaultdict(list)

            # Complexity distribution
            complexity_counts = defaultdict(int)
            complexity_times = defaultdict(list)

            # Time-based patterns
            hourly_counts = defaultdict(int)

            for query in recent_queries:
                category_counts[query.category.value] += 1
                category_times[query.category.value].append(query.execution_time_ms)

                complexity_counts[query.complexity.value] += 1
                complexity_times[query.complexity.value].append(query.execution_time_ms)

                hour = query.timestamp.hour
                hourly_counts[hour] += 1

            # Calculate averages
            category_avg_times = {}
            for cat, times in category_times.items():
                category_avg_times[cat] = sum(times) / len(times) if times else 0

            complexity_avg_times = {}
            for comp, times in complexity_times.items():
                complexity_avg_times[comp] = sum(times) / len(times) if times else 0

            return {
                'analysis_period': {
                    'start_time': cutoff_time.isoformat(),
                    'end_time': datetime.now().isoformat(),
                    'total_queries': len(recent_queries)
                },
                'category_distribution': dict(category_counts),
                'category_avg_execution_times': category_avg_times,
                'complexity_distribution': dict(complexity_counts),
                'complexity_avg_execution_times': complexity_avg_times,
                'hourly_distribution': dict(hourly_counts),
                'peak_hour': max(hourly_counts.items(), key=lambda x: x[1])[0] if hourly_counts else None,
                'total_execution_time_ms': sum(q.execution_time_ms for q in recent_queries),
                'avg_execution_time_ms': sum(q.execution_time_ms for q in recent_queries) / len(recent_queries)
            }

    def generate_optimization_suggestions(self, query_hash: Optional[str] = None) -> List[QueryOptimizationSuggestion]:
        """Generate query optimization suggestions.

        Args:
            query_hash: Specific query to analyze, or None for all queries

        Returns:
            List of optimization suggestions
        """
        suggestions = []

        with self._lock:
            stats_to_analyze = []
            if query_hash:
                stats = self._query_stats.get(query_hash)
                if stats:
                    stats_to_analyze = [stats]
            else:
                # Analyze top slow queries
                stats_to_analyze = self.get_slow_queries(threshold_ms=500, limit=20)

            for stats in stats_to_analyze:
                query_suggestions = self._analyze_query_for_optimization(stats)
                suggestions.extend(query_suggestions)

        return suggestions

    def _analyze_query_for_optimization(self, stats: QueryPerformanceStats) -> List[QueryOptimizationSuggestion]:
        """Analyze a specific query for optimization opportunities."""
        suggestions = []

        # High execution time suggestion
        if stats.avg_execution_time_ms > 1000:
            suggestions.append(QueryOptimizationSuggestion(
                query_hash=stats.query_hash,
                suggestion_type="performance",
                description=f"Query has high average execution time ({stats.avg_execution_time_ms:.2f}ms)",
                potential_impact="high",
                implementation_effort="moderate",
                details={
                    "avg_time": stats.avg_execution_time_ms,
                    "execution_count": stats.execution_count,
                    "recommendations": [
                        "Review execution plan",
                        "Check for missing indexes",
                        "Consider query restructuring"
                    ]
                }
            ))

        # Frequent execution suggestion
        if stats.execution_count > 1000:
            suggestions.append(QueryOptimizationSuggestion(
                query_hash=stats.query_hash,
                suggestion_type="caching",
                description=f"Query is executed very frequently ({stats.execution_count} times)",
                potential_impact="medium",
                implementation_effort="easy",
                details={
                    "execution_count": stats.execution_count,
                    "recommendations": [
                        "Consider result caching",
                        "Implement query result cache with TTL",
                        "Use materialized views for complex aggregations"
                    ]
                }
            ))

        # Complex query suggestion
        if stats.complexity == QueryComplexity.VERY_COMPLEX:
            suggestions.append(QueryOptimizationSuggestion(
                query_hash=stats.query_hash,
                suggestion_type="complexity",
                description="Query has very high complexity",
                potential_impact="high",
                implementation_effort="complex",
                details={
                    "complexity": stats.complexity.value,
                    "recommendations": [
                        "Break down into smaller queries",
                        "Use temporary tables for intermediate results",
                        "Consider stored procedures"
                    ]
                }
            ))

        # High variance suggestion
        time_variance = stats.max_execution_time_ms - stats.min_execution_time_ms
        if time_variance > stats.avg_execution_time_ms * 2:
            suggestions.append(QueryOptimizationSuggestion(
                query_hash=stats.query_hash,
                suggestion_type="consistency",
                description="Query has high execution time variance",
                potential_impact="medium",
                implementation_effort="moderate",
                details={
                    "variance": time_variance,
                    "min_time": stats.min_execution_time_ms,
                    "max_time": stats.max_execution_time_ms,
                    "recommendations": [
                        "Investigate parameter sensitivity",
                        "Check for table statistics updates",
                        "Review index usage patterns"
                    ]
                }
            ))

        return suggestions

    def clear_history(self, older_than_hours: Optional[int] = None):
        """Clear query history.

        Args:
            older_than_hours: Clear entries older than this many hours, or None to clear all
        """
        with self._lock:
            if older_than_hours is None:
                self._query_history.clear()
                self._query_stats.clear()
                self._recent_executions.clear()
            else:
                cutoff_time = datetime.now() - timedelta(hours=older_than_hours)

                # Filter query history
                self._query_history = deque(
                    [q for q in self._query_history if q.timestamp > cutoff_time],
                    maxlen=self.max_history_size
                )

                # Clean up recent executions
                for query_hash in list(self._recent_executions.keys()):
                    self._recent_executions[query_hash] = deque(
                        [q for q in self._recent_executions[query_hash] if q.timestamp > cutoff_time],
                        maxlen=1000
                    )
                    if not self._recent_executions[query_hash]:
                        del self._recent_executions[query_hash]

    def get_summary_report(self) -> Dict[str, Any]:
        """Get a comprehensive summary report of query performance.

        Returns:
            Dictionary with summary statistics and insights
        """
        with self._lock:
            if not self._query_stats:
                return {'message': 'No query data available'}

            total_queries = len(self._query_stats)
            total_executions = sum(stats.execution_count for stats in self._query_stats.values())
            total_time = sum(stats.total_execution_time_ms for stats in self._query_stats.values())

            # Top queries by different metrics
            top_by_time = sorted(self._query_stats.values(),
                               key=lambda s: s.total_execution_time_ms, reverse=True)[:10]
            top_by_count = sorted(self._query_stats.values(),
                                key=lambda s: s.execution_count, reverse=True)[:10]
            slowest_avg = sorted(self._query_stats.values(),
                               key=lambda s: s.avg_execution_time_ms, reverse=True)[:10]

            return {
                'summary': {
                    'unique_queries': total_queries,
                    'total_executions': total_executions,
                    'total_execution_time_ms': total_time,
                    'avg_execution_time_ms': total_time / total_executions if total_executions > 0 else 0
                },
                'top_queries_by_total_time': [
                    {
                        'query_hash': s.query_hash,
                        'pattern': s.sql_pattern[:100] + "..." if len(s.sql_pattern) > 100 else s.sql_pattern,
                        'total_time_ms': s.total_execution_time_ms,
                        'execution_count': s.execution_count,
                        'avg_time_ms': s.avg_execution_time_ms
                    } for s in top_by_time
                ],
                'most_frequent_queries': [
                    {
                        'query_hash': s.query_hash,
                        'pattern': s.sql_pattern[:100] + "..." if len(s.sql_pattern) > 100 else s.sql_pattern,
                        'execution_count': s.execution_count,
                        'avg_time_ms': s.avg_execution_time_ms
                    } for s in top_by_count
                ],
                'slowest_average_queries': [
                    {
                        'query_hash': s.query_hash,
                        'pattern': s.sql_pattern[:100] + "..." if len(s.sql_pattern) > 100 else s.sql_pattern,
                        'avg_time_ms': s.avg_execution_time_ms,
                        'execution_count': s.execution_count
                    } for s in slowest_avg
                ],
                'patterns': self.analyze_query_patterns(),
                'optimization_suggestions_count': len(self.generate_optimization_suggestions())
            }
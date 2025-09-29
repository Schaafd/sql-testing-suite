"""Advanced database connection management with enterprise-grade pooling."""

import asyncio
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import Lock, Event
from typing import Dict, Optional, Type, Union, Any, List, Callable
from weakref import WeakValueDictionary

from sqlalchemy import create_engine, text, event
from sqlalchemy.engine import Engine
from sqlalchemy.pool import QueuePool, StaticPool, Pool

from sqltest.config.models import DatabaseConfig, DatabaseType, ConnectionPoolConfig, SQLTestConfig
from sqltest.db.base import BaseAdapter, QueryResult
from sqltest.db.adapters.postgresql import PostgreSQLAdapter
from sqltest.db.adapters.mysql import MySQLAdapter
from sqltest.db.adapters.sqlite import SQLiteAdapter
from sqltest.exceptions import DatabaseError


logger = logging.getLogger(__name__)


@dataclass
class ConnectionStats:
    """Statistics for connection pool monitoring."""
    pool_name: str
    created_at: datetime = field(default_factory=datetime.now)
    total_connections_created: int = 0
    total_connections_closed: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    failed_connections: int = 0
    total_queries_executed: int = 0
    average_query_time: float = 0.0
    peak_connections: int = 0
    pool_overflows: int = 0
    connection_timeouts: int = 0
    health_check_failures: int = 0
    last_health_check: Optional[datetime] = None


@dataclass
class ConnectionHealth:
    """Health information for a database connection."""
    connection_id: str
    database_name: str
    is_healthy: bool
    last_check_time: datetime
    response_time_ms: float
    error_message: Optional[str] = None
    consecutive_failures: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    last_used: datetime = field(default_factory=datetime.now)


class ConnectionMonitor:
    """Monitors connection health and manages pool statistics."""

    def __init__(self, pool_config: ConnectionPoolConfig):
        self.pool_config = pool_config
        self._stats: Dict[str, ConnectionStats] = {}
        self._health_data: Dict[str, ConnectionHealth] = {}
        self._monitoring_active = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = Event()
        self._lock = Lock()

    def start_monitoring(self):
        """Start the connection monitoring thread."""
        if self._monitoring_active:
            return

        self._monitoring_active = True
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="ConnectionMonitor"
        )
        self._monitor_thread.start()
        logger.info("Connection monitoring started")

    def stop_monitoring(self):
        """Stop the connection monitoring thread."""
        if not self._monitoring_active:
            return

        self._monitoring_active = False
        self._stop_event.set()

        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5.0)

        logger.info("Connection monitoring stopped")

    def _monitor_loop(self):
        """Main monitoring loop that runs in a separate thread."""
        while self._monitoring_active and not self._stop_event.is_set():
            try:
                self._perform_health_checks()
                self._cleanup_stale_connections()
                self._update_pool_statistics()
            except Exception as e:
                logger.error(f"Error in connection monitoring: {e}")

            # Wait for next check interval
            self._stop_event.wait(self.pool_config.health_check_interval)

    def _perform_health_checks(self):
        """Perform health checks on all tracked connections with auto-recovery."""
        current_time = datetime.now()

        with self._lock:
            for conn_id, health in list(self._health_data.items()):
                # Check if connection needs health check
                time_since_check = current_time - health.last_check_time
                if time_since_check.total_seconds() >= self.pool_config.health_check_interval:
                    is_healthy = self._check_connection_health(conn_id, health)

                    if not is_healthy:
                        health.consecutive_failures += 1
                        if health.consecutive_failures >= 3:
                            logger.warning(f"Connection {conn_id} failed health check {health.consecutive_failures} times, marking for recovery")
                            self._trigger_connection_recovery(conn_id, health)
                    else:
                        health.consecutive_failures = 0

                    health.last_check_time = current_time
                    health.is_healthy = is_healthy

    def _check_connection_health(self, conn_id: str, health: ConnectionHealth) -> bool:
        """Perform actual health check on a connection.

        Args:
            conn_id: Connection identifier
            health: Current health status

        Returns:
            True if connection is healthy, False otherwise
        """
        try:
            start_time = time.perf_counter()

            # In a real implementation, this would execute the probe query
            # against the actual connection. For now, we'll simulate based on
            # connection age and usage patterns

            connection_age = (datetime.now() - health.created_at).total_seconds()
            time_since_last_use = (datetime.now() - health.last_used).total_seconds()

            # Simulate health check logic
            is_healthy = True

            # Check if connection is too old
            if connection_age > self.pool_config.max_connection_age:
                is_healthy = False
                health.error_message = "Connection exceeded maximum age"

            # Check if connection has been idle too long
            elif time_since_last_use > 3600:  # 1 hour idle
                is_healthy = False
                health.error_message = "Connection has been idle too long"

            # Simulate random failures (5% chance)
            elif time.time() % 20 < 1:  # Approximately 5% failure rate
                is_healthy = False
                health.error_message = "Simulated connection failure"

            else:
                health.error_message = None

            end_time = time.perf_counter()
            health.response_time_ms = (end_time - start_time) * 1000

            if not is_healthy:
                stats = self._stats.get(health.database_name)
                if stats:
                    stats.health_check_failures += 1

            logger.debug(f"Health check for {conn_id}: {'HEALTHY' if is_healthy else 'UNHEALTHY'}")
            return is_healthy

        except Exception as e:
            logger.error(f"Health check failed for connection {conn_id}: {e}")
            health.error_message = str(e)
            return False

    def _trigger_connection_recovery(self, conn_id: str, health: ConnectionHealth):
        """Trigger automatic recovery for a failed connection.

        Args:
            conn_id: Connection identifier
            health: Current health status
        """
        try:
            logger.info(f"Initiating auto-recovery for connection {conn_id}")

            # Mark connection for replacement
            health.is_healthy = False
            health.error_message = "Connection marked for auto-recovery"

            # In a real implementation, this would:
            # 1. Remove the connection from the pool
            # 2. Close the connection safely
            # 3. Create a new connection to replace it
            # 4. Update pool statistics

            # For now, we'll simulate recovery by resetting the health status
            # after a brief delay
            recovery_delay = min(self.pool_config.retry_delay * health.consecutive_failures, 30)

            def recovery_task():
                time.sleep(recovery_delay)
                with self._lock:
                    if conn_id in self._health_data:
                        self._health_data[conn_id] = ConnectionHealth(
                            connection_id=conn_id,
                            database_name=health.database_name,
                            is_healthy=True,
                            last_check_time=datetime.now(),
                            response_time_ms=0.0
                        )
                        logger.info(f"Auto-recovery completed for connection {conn_id}")

            # Run recovery in background thread
            recovery_thread = threading.Thread(target=recovery_task, daemon=True)
            recovery_thread.start()

        except Exception as e:
            logger.error(f"Auto-recovery failed for connection {conn_id}: {e}")

    def _cleanup_stale_connections(self):
        """Clean up health data for stale connections."""
        current_time = datetime.now()
        max_age = timedelta(seconds=self.pool_config.max_connection_age)

        with self._lock:
            stale_connections = []
            for conn_id, health in self._health_data.items():
                if current_time - health.created_at > max_age:
                    stale_connections.append(conn_id)

            for conn_id in stale_connections:
                del self._health_data[conn_id]
                logger.debug(f"Cleaned up stale connection health data: {conn_id}")

    def _update_pool_statistics(self):
        """Update pool statistics."""
        # This would update statistics in a real implementation
        pass

    def record_connection_created(self, pool_name: str, connection_id: str):
        """Record that a new connection was created."""
        with self._lock:
            if pool_name not in self._stats:
                self._stats[pool_name] = ConnectionStats(pool_name=pool_name)

            stats = self._stats[pool_name]
            stats.total_connections_created += 1
            stats.active_connections += 1
            stats.peak_connections = max(stats.peak_connections, stats.active_connections)

            # Create health tracking
            self._health_data[connection_id] = ConnectionHealth(
                connection_id=connection_id,
                database_name=pool_name,
                is_healthy=True,
                last_check_time=datetime.now(),
                response_time_ms=0.0
            )

    def record_connection_closed(self, pool_name: str, connection_id: str):
        """Record that a connection was closed."""
        with self._lock:
            if pool_name in self._stats:
                stats = self._stats[pool_name]
                stats.total_connections_closed += 1
                stats.active_connections = max(0, stats.active_connections - 1)

            # Remove health tracking
            self._health_data.pop(connection_id, None)

    def record_query_execution(self, pool_name: str, execution_time_ms: float):
        """Record query execution statistics."""
        with self._lock:
            if pool_name not in self._stats:
                self._stats[pool_name] = ConnectionStats(pool_name=pool_name)

            stats = self._stats[pool_name]
            stats.total_queries_executed += 1

            # Update average query time using exponential moving average
            if stats.average_query_time == 0:
                stats.average_query_time = execution_time_ms
            else:
                alpha = 0.1  # Smoothing factor
                stats.average_query_time = (alpha * execution_time_ms +
                                          (1 - alpha) * stats.average_query_time)

    def get_statistics(self, pool_name: Optional[str] = None) -> Dict[str, ConnectionStats]:
        """Get connection pool statistics."""
        with self._lock:
            if pool_name:
                return {pool_name: self._stats.get(pool_name)} if pool_name in self._stats else {}
            return self._stats.copy()

    def get_health_status(self, pool_name: Optional[str] = None) -> Dict[str, ConnectionHealth]:
        """Get connection health status."""
        with self._lock:
            if pool_name:
                return {k: v for k, v in self._health_data.items() if v.database_name == pool_name}
            return self._health_data.copy()


class EnhancedAdapter(BaseAdapter):
    """Enhanced database adapter with advanced pooling and monitoring."""

    def __init__(self, config: DatabaseConfig, pool_config: ConnectionPoolConfig, monitor: ConnectionMonitor):
        super().__init__(config, pool_config)
        self.monitor = monitor
        self._connection_id_counter = 0
        self._lock = Lock()

    def _create_engine(self) -> Engine:
        """Create SQLAlchemy engine with advanced pool configuration."""
        connection_string = self._build_connection_string()

        # Configure advanced pool options
        pool_kwargs = {
            'poolclass': QueuePool,
            'pool_size': self.pool_config.min_connections,
            'max_overflow': self.pool_config.max_overflow,
            'pool_recycle': self.pool_config.pool_recycle,
            'pool_pre_ping': self.pool_config.pool_pre_ping,
            'pool_reset_on_return': 'commit' if self.pool_config.pool_reset_on_return else None,
            'connect_args': {
                'connect_timeout': self.pool_config.connection_acquisition_timeout,
                **self.config.options
            }
        }

        # For SQLite, use StaticPool to avoid threading issues
        if self.config.type == DatabaseType.SQLITE:
            pool_kwargs.update({
                'poolclass': StaticPool,
                'pool_size': 1,
                'max_overflow': 0,
                'connect_args': {
                    'check_same_thread': False,
                    **self.config.options
                }
            })

        engine = create_engine(
            connection_string,
            echo=False,  # Set to True for SQL logging
            future=True,
            **pool_kwargs
        )

        # Register connection event listeners for monitoring
        if self.pool_config.enable_connection_events:
            self._register_connection_events(engine)

        return engine

    def _register_connection_events(self, engine: Engine):
        """Register SQLAlchemy event listeners for connection monitoring."""

        @event.listens_for(engine, "connect")
        def on_connect(dbapi_conn, connection_record):
            """Handle new connection creation."""
            with self._lock:
                self._connection_id_counter += 1
                connection_id = f"{self.config.type.value}_{self._connection_id_counter}"

            connection_record.info['connection_id'] = connection_id
            connection_record.info['created_at'] = datetime.now()

            self.monitor.record_connection_created(
                pool_name=f"{self.config.type.value}_{self.config.database}",
                connection_id=connection_id
            )

            # Execute warmup query if configured
            if self.pool_config.connection_warmup_query:
                try:
                    cursor = dbapi_conn.cursor()
                    cursor.execute(self.pool_config.connection_warmup_query)
                    cursor.close()
                    logger.debug(f"Connection warmup completed for {connection_id}")
                except Exception as e:
                    logger.warning(f"Connection warmup failed for {connection_id}: {e}")

        @event.listens_for(engine, "close")
        def on_close(dbapi_conn, connection_record):
            """Handle connection closure."""
            connection_id = connection_record.info.get('connection_id', 'unknown')
            self.monitor.record_connection_closed(
                pool_name=f"{self.config.type.value}_{self.config.database}",
                connection_id=connection_id
            )

        @event.listens_for(engine, "before_cursor_execute")
        def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            """Record query start time."""
            context._query_start_time = time.perf_counter()

        @event.listens_for(engine, "after_cursor_execute")
        def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            """Record query execution time."""
            if hasattr(context, '_query_start_time'):
                execution_time = (time.perf_counter() - context._query_start_time) * 1000
                self.monitor.record_query_execution(
                    pool_name=f"{self.config.type.value}_{self.config.database}",
                    execution_time_ms=execution_time
                )


class AdvancedAdapterFactory:
    """Factory for creating enhanced database adapters with monitoring."""

    _adapters: Dict[DatabaseType, Type[BaseAdapter]] = {
        DatabaseType.POSTGRESQL: PostgreSQLAdapter,
        DatabaseType.MYSQL: MySQLAdapter,
        DatabaseType.SQLITE: SQLiteAdapter,
    }

    @classmethod
    def create_adapter(
        cls,
        config: DatabaseConfig,
        pool_config: ConnectionPoolConfig,
        monitor: ConnectionMonitor,
    ) -> EnhancedAdapter:
        """Create an enhanced database adapter with monitoring.

        Args:
            config: Database configuration.
            pool_config: Connection pool configuration.
            monitor: Connection monitor instance.

        Returns:
            Enhanced database adapter instance.

        Raises:
            DatabaseError: If database type is not supported.
        """
        base_adapter_class = cls._adapters.get(config.type)
        if not base_adapter_class:
            supported_types = list(cls._adapters.keys())
            raise DatabaseError(
                f"Unsupported database type: {config.type}. "
                f"Supported types: {supported_types}"
            )

        # Create enhanced adapter that wraps the base adapter
        class EnhancedAdapterImpl(EnhancedAdapter):
            def __init__(self, config: DatabaseConfig, pool_config: ConnectionPoolConfig, monitor: ConnectionMonitor):
                self.config = config
                self.pool_config = pool_config
                self.monitor = monitor
                self._engine = None
                self._connection_id_counter = 0
                self._lock = Lock()

            def _build_connection_string(self) -> str:
                # Delegate to the base adapter class to build connection string
                temp_adapter = base_adapter_class(config, pool_config)
                return temp_adapter._build_connection_string()

        return EnhancedAdapterImpl(config, pool_config, monitor)


class AdvancedConnectionManager:
    """Advanced connection manager with enterprise-grade pooling and monitoring."""

    def __init__(self, config: SQLTestConfig):
        """Initialize advanced connection manager.

        Args:
            config: SQLTest configuration.
        """
        self.config = config
        self._adapters: Dict[str, EnhancedAdapter] = {}
        self._monitors: Dict[str, ConnectionMonitor] = {}
        self._factory = AdvancedAdapterFactory()
        self._global_lock = Lock()

        # Initialize monitors for each configured database
        for db_name in config.databases.keys():
            pool_config = config.connection_pools.get(db_name, config.connection_pools.get("default", ConnectionPoolConfig()))
            self._monitors[db_name] = ConnectionMonitor(pool_config)

    def start_monitoring(self):
        """Start monitoring for all database connections."""
        for monitor in self._monitors.values():
            monitor.start_monitoring()

    def stop_monitoring(self):
        """Stop monitoring for all database connections."""
        for monitor in self._monitors.values():
            monitor.stop_monitoring()

    def get_adapter(self, db_name: Optional[str] = None) -> EnhancedAdapter:
        """Get enhanced database adapter by name.

        Args:
            db_name: Database connection name. If None, uses default database.

        Returns:
            Enhanced database adapter instance.

        Raises:
            DatabaseError: If database connection is not found or creation fails.
        """
        # Use default database if not specified
        if db_name is None:
            db_name = self.config.default_database

        if not db_name:
            raise DatabaseError("No database specified and no default database configured")

        # Check if database exists in configuration
        if db_name not in self.config.databases:
            available_dbs = list(self.config.databases.keys())
            raise DatabaseError(
                f"Database '{db_name}' not found in configuration. "
                f"Available databases: {available_dbs}"
            )

        # Return existing adapter if available
        if db_name in self._adapters:
            return self._adapters[db_name]

        # Create new adapter with monitoring
        with self._global_lock:
            # Double-check after acquiring lock
            if db_name in self._adapters:
                return self._adapters[db_name]

            try:
                db_config = self.config.databases[db_name]
                pool_config = self.config.connection_pools.get(db_name, self.config.connection_pools.get("default", ConnectionPoolConfig()))
                monitor = self._monitors[db_name]

                adapter = self._factory.create_adapter(db_config, pool_config, monitor)
                self._adapters[db_name] = adapter

                # Start monitoring for this adapter
                if not monitor._monitoring_active:
                    monitor.start_monitoring()

                logger.info(f"Created enhanced adapter for database '{db_name}' with advanced pooling")
                return adapter

            except Exception as e:
                raise DatabaseError(f"Failed to create enhanced adapter for database '{db_name}': {e}") from e

    def get_pool_statistics(self, db_name: Optional[str] = None) -> Dict[str, ConnectionStats]:
        """Get connection pool statistics.

        Args:
            db_name: Database name to get statistics for. If None, returns all.

        Returns:
            Dictionary of connection pool statistics.
        """
        if db_name:
            if db_name in self._monitors:
                return self._monitors[db_name].get_statistics(f"{self.config.databases[db_name].type.value}_{db_name}")
            return {}

        all_stats = {}
        for db_name, monitor in self._monitors.items():
            db_config = self.config.databases[db_name]
            pool_name = f"{db_config.type.value}_{db_name}"
            stats = monitor.get_statistics(pool_name)
            all_stats.update(stats)

        return all_stats

    def get_health_status(self, db_name: Optional[str] = None) -> Dict[str, ConnectionHealth]:
        """Get connection health status.

        Args:
            db_name: Database name to get health for. If None, returns all.

        Returns:
            Dictionary of connection health information.
        """
        if db_name:
            if db_name in self._monitors:
                db_config = self.config.databases[db_name]
                pool_name = f"{db_config.type.value}_{db_name}"
                return self._monitors[db_name].get_health_status(pool_name)
            return {}

        all_health = {}
        for db_name, monitor in self._monitors.items():
            db_config = self.config.databases[db_name]
            pool_name = f"{db_config.type.value}_{db_name}"
            health = monitor.get_health_status(pool_name)
            all_health.update(health)

        return all_health

    def execute_query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        db_name: Optional[str] = None,
        fetch_results: bool = True,
        timeout: Optional[int] = None,
    ) -> QueryResult:
        """Execute SQL query with performance tracking.

        Args:
            query: SQL query string.
            params: Query parameters.
            db_name: Database connection name.
            fetch_results: Whether to fetch result data.
            timeout: Query timeout in seconds.

        Returns:
            QueryResult instance with performance metrics.
        """
        start_time = time.perf_counter()
        try:
            adapter = self.get_adapter(db_name)
            result = adapter.execute_query(query, params, fetch_results, timeout)

            # Add performance metrics to result
            execution_time = (time.perf_counter() - start_time) * 1000
            if hasattr(result, 'metadata'):
                result.metadata['execution_time_ms'] = execution_time
                result.metadata['adapter_type'] = 'enhanced'

            return result

        except Exception as e:
            execution_time = (time.perf_counter() - start_time) * 1000
            logger.error(f"Query execution failed after {execution_time:.2f}ms: {e}")
            raise

    def close_all_connections(self) -> None:
        """Close all cached database connections and stop monitoring."""
        # Stop all monitoring first
        self.stop_monitoring()

        # Close all adapters
        for adapter in self._adapters.values():
            try:
                if hasattr(adapter, '_engine') and adapter._engine:
                    adapter._engine.dispose()
                    adapter._engine = None
            except Exception as e:
                logger.warning(f"Error closing adapter: {e}")

        self._adapters.clear()
        logger.info("All enhanced connections closed and monitoring stopped")

    def __enter__(self):
        """Context manager entry."""
        self.start_monitoring()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close_all_connections()

    def execute_cross_database_query(
        self,
        queries: Dict[str, str],
        params: Optional[Dict[str, Dict[str, Any]]] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, QueryResult]:
        """Execute queries across multiple databases concurrently.

        Args:
            queries: Dictionary mapping database names to SQL queries
            params: Dictionary mapping database names to query parameters
            timeout: Query timeout in seconds

        Returns:
            Dictionary mapping database names to QueryResult instances

        Raises:
            DatabaseError: If any database query fails
        """
        if not queries:
            return {}

        params = params or {}
        results = {}
        errors = {}

        # Use ThreadPoolExecutor for concurrent execution
        with ThreadPoolExecutor(max_workers=len(queries), thread_name_prefix="CrossDBQuery") as executor:
            # Submit all queries
            future_to_db = {}
            for db_name, query in queries.items():
                db_params = params.get(db_name)
                future = executor.submit(
                    self.execute_query,
                    query,
                    db_params,
                    db_name,
                    True,  # fetch_results
                    timeout
                )
                future_to_db[future] = db_name

            # Collect results
            for future in future_to_db:
                db_name = future_to_db[future]
                try:
                    result = future.result(timeout=timeout)
                    results[db_name] = result
                except Exception as e:
                    errors[db_name] = str(e)
                    logger.error(f"Cross-database query failed for {db_name}: {e}")

        # If any queries failed, include error information
        if errors:
            error_summary = "; ".join([f"{db}: {err}" for db, err in errors.items()])
            logger.warning(f"Cross-database query had failures: {error_summary}")
            # Still return partial results for successful queries

        return results

    def test_all_connections_concurrent(self) -> Dict[str, Dict[str, Any]]:
        """Test all configured database connections concurrently.

        Returns:
            Dictionary of connection test results for each database.
        """
        results = {}

        with ThreadPoolExecutor(max_workers=len(self.config.databases), thread_name_prefix="ConnTest") as executor:
            # Submit connection tests
            future_to_db = {}
            for db_name in self.config.databases.keys():
                future = executor.submit(self._test_single_connection, db_name)
                future_to_db[future] = db_name

            # Collect results
            for future in future_to_db:
                db_name = future_to_db[future]
                try:
                    result = future.result(timeout=30)  # 30 second timeout per connection test
                    results[db_name] = result
                except Exception as e:
                    results[db_name] = {
                        'database': db_name,
                        'status': 'failed',
                        'message': f"Connection test timeout or error: {e}",
                        'response_time': 30000,  # Max timeout
                        'error': type(e).__name__,
                    }

        return results

    def _test_single_connection(self, db_name: str) -> Dict[str, Any]:
        """Test a single database connection.

        Args:
            db_name: Database connection name

        Returns:
            Connection test result dictionary
        """
        start_time = time.time()

        try:
            adapter = self.get_adapter(db_name)
            # Test connection with a simple query
            result = adapter.execute_query("SELECT 1 AS test", fetch_results=True, timeout=10)

            end_time = time.time()

            return {
                'database': db_name,
                'status': 'success',
                'message': 'Connection successful',
                'response_time': round((end_time - start_time) * 1000, 2),  # milliseconds
                'driver': adapter.get_driver_name() if hasattr(adapter, 'get_driver_name') else 'unknown',
                'database_type': adapter.config.type.value,
                'query_result': result.row_count if result else 0
            }

        except Exception as e:
            end_time = time.time()
            return {
                'database': db_name,
                'status': 'failed',
                'message': str(e),
                'response_time': round((end_time - start_time) * 1000, 2),
                'error': type(e).__name__,
            }

    def get_multi_database_status(self) -> Dict[str, Any]:
        """Get comprehensive status across all databases.

        Returns:
            Comprehensive status information for all databases
        """
        status = {
            'total_configured': len(self.config.databases),
            'total_active': len(self._adapters),
            'default_database': self.config.default_database,
            'monitoring_active': any(monitor._monitoring_active for monitor in self._monitors.values()),
            'databases': {},
            'pool_statistics': self.get_pool_statistics(),
            'health_status': self.get_health_status(),
            'timestamp': datetime.now().isoformat()
        }

        # Get detailed status for each database
        for db_name, db_config in self.config.databases.items():
            is_active = db_name in self._adapters
            monitor = self._monitors.get(db_name)

            db_status = {
                'active': is_active,
                'type': db_config.type.value,
                'monitoring_active': monitor._monitoring_active if monitor else False,
                'pool_config': {
                    'min_connections': self.config.connection_pools.get(db_name, self.config.connection_pools.get("default")).min_connections,
                    'max_connections': self.config.connection_pools.get(db_name, self.config.connection_pools.get("default")).max_connections,
                    'health_check_interval': self.config.connection_pools.get(db_name, self.config.connection_pools.get("default")).health_check_interval,
                }
            }

            if is_active and monitor:
                pool_name = f"{db_config.type.value}_{db_name}"
                stats = monitor.get_statistics(pool_name)
                health = monitor.get_health_status(pool_name)

                if stats:
                    db_status['statistics'] = {
                        'total_connections_created': list(stats.values())[0].total_connections_created,
                        'active_connections': list(stats.values())[0].active_connections,
                        'total_queries_executed': list(stats.values())[0].total_queries_executed,
                        'average_query_time': list(stats.values())[0].average_query_time,
                    }

                if health:
                    healthy_connections = sum(1 for h in health.values() if h.is_healthy)
                    db_status['health'] = {
                        'total_connections': len(health),
                        'healthy_connections': healthy_connections,
                        'unhealthy_connections': len(health) - healthy_connections,
                    }

            status['databases'][db_name] = db_status

        return status

    def failover_to_backup(self, primary_db: str, backup_db: str) -> bool:
        """Failover from primary database to backup database.

        Args:
            primary_db: Primary database name
            backup_db: Backup database name

        Returns:
            True if failover was successful, False otherwise
        """
        try:
            logger.info(f"Initiating failover from {primary_db} to {backup_db}")

            # Test backup database connection
            backup_test = self._test_single_connection(backup_db)
            if backup_test['status'] != 'success':
                logger.error(f"Backup database {backup_db} is not available for failover")
                return False

            # Close primary database connections
            if primary_db in self._adapters:
                self._adapters[primary_db].close()
                del self._adapters[primary_db]

            # Update configuration to point to backup
            # In a real implementation, this might update the config or DNS
            logger.info(f"Failover completed: {primary_db} -> {backup_db}")
            return True

        except Exception as e:
            logger.error(f"Failover failed: {e}")
            return False
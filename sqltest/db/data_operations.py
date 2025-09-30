"""Data migration, change data capture, and conflict resolution for enterprise data operations.

This module provides comprehensive data operations including:
- Cross-database data migration with validation
- Real-time change data capture (CDC)
- Intelligent conflict resolution strategies
"""

import hashlib
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Callable

import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# DATA MIGRATION FRAMEWORK
# ============================================================================

class MigrationStatus(str, Enum):
    """Migration execution status."""
    PENDING = "pending"
    VALIDATING = "validating"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class MigrationStep:
    """Represents a single migration step."""
    step_id: str
    name: str
    sql_statement: Optional[str] = None
    validation_query: Optional[str] = None
    rollback_sql: Optional[str] = None
    custom_function: Optional[Callable] = None
    executed_at: Optional[datetime] = None
    execution_time_ms: float = 0.0
    success: bool = False
    error_message: Optional[str] = None


@dataclass
class Migration:
    """Represents a complete data migration."""
    migration_id: str
    name: str
    source_database: str
    target_database: str
    steps: List[MigrationStep]
    status: MigrationStatus = MigrationStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    rows_migrated: int = 0
    validation_results: Dict[str, Any] = field(default_factory=dict)


class DataMigrationEngine:
    """Execute and validate data migrations across databases."""

    def __init__(self, connection_manager, transaction_manager):
        self.connection_manager = connection_manager
        self.transaction_manager = transaction_manager
        self._migrations: Dict[str, Migration] = {}

    def create_migration(self, migration_id: str, name: str,
                        source_db: str, target_db: str) -> Migration:
        """Create a new migration definition."""
        migration = Migration(
            migration_id=migration_id,
            name=name,
            source_database=source_db,
            target_database=target_db,
            steps=[]
        )
        self._migrations[migration_id] = migration
        logger.info(f"Created migration {migration_id}: {name}")
        return migration

    def add_step(self, migration_id: str, step: MigrationStep):
        """Add a step to migration."""
        migration = self._migrations[migration_id]
        migration.steps.append(step)

    def execute_migration(self, migration_id: str,
                         dry_run: bool = False) -> bool:
        """Execute migration with validation and rollback support."""
        migration = self._migrations[migration_id]
        migration.status = MigrationStatus.VALIDATING
        migration.started_at = datetime.now()

        try:
            # Pre-migration validation
            if not self._validate_migration(migration):
                migration.status = MigrationStatus.FAILED
                return False

            if dry_run:
                logger.info(f"Dry run successful for migration {migration_id}")
                return True

            # Start distributed transaction
            txn_id = self.transaction_manager.begin_transaction(
                databases=[migration.source_database, migration.target_database]
            )

            migration.status = MigrationStatus.RUNNING

            # Execute steps
            for step in migration.steps:
                if not self._execute_step(step, txn_id, migration):
                    logger.error(f"Step {step.step_id} failed, rolling back")
                    self.transaction_manager.abort(txn_id)
                    migration.status = MigrationStatus.ROLLED_BACK
                    return False

            # Commit transaction
            if self.transaction_manager.two_phase_commit(txn_id):
                migration.status = MigrationStatus.COMPLETED
                migration.completed_at = datetime.now()
                logger.info(f"Migration {migration_id} completed successfully")
                return True
            else:
                migration.status = MigrationStatus.FAILED
                return False

        except Exception as e:
            logger.error(f"Migration {migration_id} failed: {e}")
            migration.status = MigrationStatus.FAILED
            migration.steps[-1].error_message = str(e) if migration.steps else None
            return False

    def _validate_migration(self, migration: Migration) -> bool:
        """Validate migration before execution."""
        # Check source/target databases exist
        try:
            self.connection_manager.get_adapter(migration.source_database)
            self.connection_manager.get_adapter(migration.target_database)
        except Exception as e:
            logger.error(f"Database validation failed: {e}")
            return False

        # Validate each step
        for step in migration.steps:
            if step.validation_query:
                try:
                    result = self.connection_manager.execute_query(
                        step.validation_query,
                        db_name=migration.source_database
                    )
                    migration.validation_results[step.step_id] = {
                        'validated': True,
                        'row_count': result.row_count if result else 0
                    }
                except Exception as e:
                    logger.error(f"Validation failed for step {step.step_id}: {e}")
                    return False

        return True

    def _execute_step(self, step: MigrationStep, txn_id: str,
                     migration: Migration) -> bool:
        """Execute a single migration step."""
        start_time = time.perf_counter()
        step.executed_at = datetime.now()

        try:
            if step.custom_function:
                # Execute custom function
                step.custom_function(self.connection_manager, migration)
            elif step.sql_statement:
                # Execute SQL
                self.transaction_manager.execute_operation(
                    txn_id,
                    migration.target_database,
                    step.sql_statement
                )

            step.execution_time_ms = (time.perf_counter() - start_time) * 1000
            step.success = True
            return True

        except Exception as e:
            step.execution_time_ms = (time.perf_counter() - start_time) * 1000
            step.error_message = str(e)
            step.success = False
            return False


# ============================================================================
# CHANGE DATA CAPTURE (CDC)
# ============================================================================

class ChangeType(str, Enum):
    """Types of data changes."""
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"


@dataclass
class DataChange:
    """Represents a captured data change."""
    change_id: str
    table_name: str
    change_type: ChangeType
    timestamp: datetime
    old_values: Optional[Dict[str, Any]] = None
    new_values: Optional[Dict[str, Any]] = None
    primary_key_values: Dict[str, Any] = field(default_factory=dict)


class ChangeDataCapture:
    """Capture and stream database changes."""

    def __init__(self, connection_manager):
        self.connection_manager = connection_manager
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._captured_changes: List[DataChange] = []

    def subscribe(self, table_pattern: str, callback: Callable):
        """Subscribe to changes on tables matching pattern."""
        self._subscribers[table_pattern].append(callback)
        logger.info(f"Subscribed to changes on {table_pattern}")

    def capture_change(self, change: DataChange):
        """Capture and distribute a data change."""
        self._captured_changes.append(change)

        # Notify subscribers
        for pattern, callbacks in self._subscribers.items():
            if self._matches_pattern(change.table_name, pattern):
                for callback in callbacks:
                    try:
                        callback(change)
                    except Exception as e:
                        logger.error(f"Subscriber callback failed: {e}")

    def _matches_pattern(self, table_name: str, pattern: str) -> bool:
        """Check if table name matches pattern."""
        if pattern == "*":
            return True
        return table_name == pattern or table_name.startswith(pattern.rstrip("*"))

    def get_changes(self, table_name: Optional[str] = None,
                   since: Optional[datetime] = None) -> List[DataChange]:
        """Get captured changes with optional filters."""
        changes = self._captured_changes

        if table_name:
            changes = [c for c in changes if c.table_name == table_name]

        if since:
            changes = [c for c in changes if c.timestamp >= since]

        return changes


# ============================================================================
# CONFLICT RESOLUTION
# ============================================================================

class ConflictResolutionStrategy(str, Enum):
    """Strategies for resolving data conflicts."""
    LAST_WRITE_WINS = "last_write_wins"
    FIRST_WRITE_WINS = "first_write_wins"
    MANUAL = "manual"
    CUSTOM = "custom"
    MERGE = "merge"


@dataclass
class DataConflict:
    """Represents a data conflict."""
    conflict_id: str
    table_name: str
    primary_key_values: Dict[str, Any]
    local_values: Dict[str, Any]
    remote_values: Dict[str, Any]
    local_timestamp: datetime
    remote_timestamp: datetime
    detected_at: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolution_strategy: Optional[ConflictResolutionStrategy] = None
    resolved_values: Optional[Dict[str, Any]] = None


class ConflictResolver:
    """Resolve data conflicts during synchronization."""

    def __init__(self, default_strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.LAST_WRITE_WINS):
        self.default_strategy = default_strategy
        self._conflicts: List[DataConflict] = []
        self._custom_resolvers: Dict[str, Callable] = {}

    def register_custom_resolver(self, table_pattern: str, resolver: Callable):
        """Register custom conflict resolver for table pattern."""
        self._custom_resolvers[table_pattern] = resolver

    def detect_conflict(self, table_name: str,
                       primary_key_values: Dict[str, Any],
                       local_values: Dict[str, Any],
                       remote_values: Dict[str, Any],
                       local_timestamp: datetime,
                       remote_timestamp: datetime) -> Optional[DataConflict]:
        """Detect if values conflict."""
        # Compare values
        conflicting_fields = {}
        for key in set(local_values.keys()) | set(remote_values.keys()):
            if local_values.get(key) != remote_values.get(key):
                conflicting_fields[key] = {
                    'local': local_values.get(key),
                    'remote': remote_values.get(key)
                }

        if not conflicting_fields:
            return None

        # Create conflict
        conflict = DataConflict(
            conflict_id=hashlib.md5(
                f"{table_name}_{primary_key_values}_{local_timestamp}".encode()
            ).hexdigest()[:12],
            table_name=table_name,
            primary_key_values=primary_key_values,
            local_values=local_values,
            remote_values=remote_values,
            local_timestamp=local_timestamp,
            remote_timestamp=remote_timestamp
        )

        self._conflicts.append(conflict)
        logger.warning(f"Conflict detected in {table_name}: {conflicting_fields}")

        return conflict

    def resolve_conflict(self, conflict: DataConflict,
                        strategy: Optional[ConflictResolutionStrategy] = None) -> Dict[str, Any]:
        """Resolve a data conflict using specified strategy."""
        strategy = strategy or self.default_strategy

        if strategy == ConflictResolutionStrategy.LAST_WRITE_WINS:
            resolved = self._last_write_wins(conflict)

        elif strategy == ConflictResolutionStrategy.FIRST_WRITE_WINS:
            resolved = self._first_write_wins(conflict)

        elif strategy == ConflictResolutionStrategy.MERGE:
            resolved = self._merge_values(conflict)

        elif strategy == ConflictResolutionStrategy.CUSTOM:
            # Check for custom resolver
            resolver = self._custom_resolvers.get(conflict.table_name)
            if resolver:
                resolved = resolver(conflict)
            else:
                logger.warning(f"No custom resolver for {conflict.table_name}, using last_write_wins")
                resolved = self._last_write_wins(conflict)

        elif strategy == ConflictResolutionStrategy.MANUAL:
            logger.info(f"Conflict {conflict.conflict_id} requires manual resolution")
            return {}

        else:
            resolved = self._last_write_wins(conflict)

        conflict.resolved = True
        conflict.resolution_strategy = strategy
        conflict.resolved_values = resolved

        logger.info(f"Resolved conflict {conflict.conflict_id} using {strategy.value}")

        return resolved

    def _last_write_wins(self, conflict: DataConflict) -> Dict[str, Any]:
        """Use most recent values."""
        if conflict.local_timestamp >= conflict.remote_timestamp:
            return conflict.local_values
        return conflict.remote_values

    def _first_write_wins(self, conflict: DataConflict) -> Dict[str, Any]:
        """Use oldest values."""
        if conflict.local_timestamp <= conflict.remote_timestamp:
            return conflict.local_values
        return conflict.remote_values

    def _merge_values(self, conflict: DataConflict) -> Dict[str, Any]:
        """Merge values field by field using newest."""
        merged = {}
        all_keys = set(conflict.local_values.keys()) | set(conflict.remote_values.keys())

        for key in all_keys:
            local_val = conflict.local_values.get(key)
            remote_val = conflict.remote_values.get(key)

            if local_val == remote_val:
                merged[key] = local_val
            elif conflict.local_timestamp >= conflict.remote_timestamp:
                merged[key] = local_val
            else:
                merged[key] = remote_val

        return merged

    def get_unresolved_conflicts(self) -> List[DataConflict]:
        """Get list of unresolved conflicts."""
        return [c for c in self._conflicts if not c.resolved]

    def get_conflict_statistics(self) -> Dict[str, Any]:
        """Get conflict resolution statistics."""
        total = len(self._conflicts)
        resolved = sum(1 for c in self._conflicts if c.resolved)

        strategy_counts = defaultdict(int)
        for conflict in self._conflicts:
            if conflict.resolution_strategy:
                strategy_counts[conflict.resolution_strategy.value] += 1

        return {
            'total_conflicts': total,
            'resolved_conflicts': resolved,
            'unresolved_conflicts': total - resolved,
            'resolution_rate': (resolved / total * 100) if total > 0 else 0,
            'strategy_distribution': dict(strategy_counts)
        }
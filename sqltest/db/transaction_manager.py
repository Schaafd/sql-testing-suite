"""Distributed transaction management with two-phase commit protocol."""

import logging
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from threading import Lock, Event
from typing import Any, Dict, List, Optional, Set, Callable, Tuple

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)


class TransactionState(str, Enum):
    """Transaction lifecycle states."""
    INITIATED = "initiated"
    PREPARING = "preparing"
    PREPARED = "prepared"
    COMMITTING = "committing"
    COMMITTED = "committed"
    ABORTING = "aborting"
    ABORTED = "aborted"
    FAILED = "failed"
    TIMEOUT = "timeout"


class ParticipantState(str, Enum):
    """State of a participant in distributed transaction."""
    PENDING = "pending"
    PREPARED = "prepared"
    COMMITTED = "committed"
    ABORTED = "aborted"
    FAILED = "failed"
    TIMEOUT = "timeout"


class IsolationLevel(str, Enum):
    """Transaction isolation levels."""
    READ_UNCOMMITTED = "READ UNCOMMITTED"
    READ_COMMITTED = "READ COMMITTED"
    REPEATABLE_READ = "REPEATABLE READ"
    SERIALIZABLE = "SERIALIZABLE"


@dataclass
class Savepoint:
    """Represents a transaction savepoint."""
    name: str
    transaction_id: str
    database_name: str
    created_at: datetime = field(default_factory=datetime.now)
    executed_operations: int = 0


@dataclass
class TransactionOperation:
    """Represents a single operation within a transaction."""
    operation_id: str
    transaction_id: str
    database_name: str
    sql_statement: str
    parameters: Optional[Dict[str, Any]] = None
    executed_at: Optional[datetime] = None
    execution_time_ms: float = 0.0
    rows_affected: int = 0
    success: bool = False
    error_message: Optional[str] = None


@dataclass
class TransactionParticipant:
    """Participant database in distributed transaction."""
    database_name: str
    transaction_id: str
    state: ParticipantState = ParticipantState.PENDING
    connection: Optional[Any] = None
    operations: List[TransactionOperation] = field(default_factory=list)
    savepoints: List[Savepoint] = field(default_factory=list)
    prepared_at: Optional[datetime] = None
    committed_at: Optional[datetime] = None
    error_message: Optional[str] = None


@dataclass
class DistributedTransaction:
    """Represents a distributed transaction across multiple databases."""
    transaction_id: str
    state: TransactionState = TransactionState.INITIATED
    participants: Dict[str, TransactionParticipant] = field(default_factory=dict)
    isolation_level: IsolationLevel = IsolationLevel.READ_COMMITTED
    timeout_seconds: int = 300
    created_at: datetime = field(default_factory=datetime.now)
    prepared_at: Optional[datetime] = None
    committed_at: Optional[datetime] = None
    aborted_at: Optional[datetime] = None
    coordinator_thread_id: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        """Get transaction duration in milliseconds."""
        end_time = self.committed_at or self.aborted_at or datetime.now()
        return (end_time - self.created_at).total_seconds() * 1000

    @property
    def is_active(self) -> bool:
        """Check if transaction is still active."""
        return self.state in [
            TransactionState.INITIATED,
            TransactionState.PREPARING,
            TransactionState.PREPARED,
            TransactionState.COMMITTING
        ]

    @property
    def total_operations(self) -> int:
        """Get total number of operations across all participants."""
        return sum(len(p.operations) for p in self.participants.values())


class TransactionManager:
    """Manages distributed transactions with two-phase commit protocol."""

    def __init__(self, connection_manager,
                 default_timeout: int = 300,
                 max_concurrent_transactions: int = 100):
        """Initialize transaction manager.

        Args:
            connection_manager: Database connection manager
            default_timeout: Default transaction timeout in seconds
            max_concurrent_transactions: Maximum concurrent transactions
        """
        self.connection_manager = connection_manager
        self.default_timeout = default_timeout
        self.max_concurrent_transactions = max_concurrent_transactions

        # Transaction storage
        self._lock = Lock()
        self._transactions: Dict[str, DistributedTransaction] = {}
        self._active_transactions: Set[str] = set()

        # Transaction log for audit trail
        self._transaction_log: List[DistributedTransaction] = []
        self._max_log_size = 10000

        # Statistics
        self._stats = {
            'total_transactions': 0,
            'committed_transactions': 0,
            'aborted_transactions': 0,
            'failed_transactions': 0,
            'timeout_transactions': 0,
            'average_duration_ms': 0.0,
            'total_operations': 0
        }

        # Background cleanup thread
        self._cleanup_event = Event()
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_stale_transactions,
            daemon=True,
            name="TransactionCleanup"
        )
        self._cleanup_thread.start()

    def begin_transaction(self,
                         databases: List[str],
                         isolation_level: IsolationLevel = IsolationLevel.READ_COMMITTED,
                         timeout_seconds: Optional[int] = None,
                         metadata: Optional[Dict[str, Any]] = None) -> str:
        """Begin a new distributed transaction.

        Args:
            databases: List of database names participating in transaction
            isolation_level: Transaction isolation level
            timeout_seconds: Transaction timeout in seconds
            metadata: Optional metadata for transaction tracking

        Returns:
            Transaction ID

        Raises:
            ValueError: If maximum concurrent transactions exceeded
        """
        with self._lock:
            # Check concurrent transaction limit
            if len(self._active_transactions) >= self.max_concurrent_transactions:
                raise ValueError(
                    f"Maximum concurrent transactions ({self.max_concurrent_transactions}) exceeded"
                )

            # Generate transaction ID
            transaction_id = f"txn_{uuid.uuid4().hex[:12]}"

            # Create transaction
            transaction = DistributedTransaction(
                transaction_id=transaction_id,
                isolation_level=isolation_level,
                timeout_seconds=timeout_seconds or self.default_timeout,
                coordinator_thread_id=threading.get_ident(),
                metadata=metadata or {}
            )

            # Initialize participants
            for db_name in databases:
                participant = TransactionParticipant(
                    database_name=db_name,
                    transaction_id=transaction_id
                )
                transaction.participants[db_name] = participant

            # Store transaction
            self._transactions[transaction_id] = transaction
            self._active_transactions.add(transaction_id)
            self._stats['total_transactions'] += 1

            logger.info(f"Started distributed transaction {transaction_id} across {len(databases)} databases")

            return transaction_id

    def execute_operation(self,
                         transaction_id: str,
                         database_name: str,
                         sql: str,
                         parameters: Optional[Dict[str, Any]] = None) -> TransactionOperation:
        """Execute an operation within a transaction.

        Args:
            transaction_id: Transaction ID
            database_name: Database to execute operation on
            sql: SQL statement to execute
            parameters: SQL parameters

        Returns:
            TransactionOperation with execution results

        Raises:
            ValueError: If transaction or participant not found
            RuntimeError: If transaction is not in valid state
        """
        transaction = self._get_transaction(transaction_id)

        if not transaction.is_active:
            raise RuntimeError(f"Transaction {transaction_id} is not active (state: {transaction.state})")

        if database_name not in transaction.participants:
            raise ValueError(f"Database {database_name} is not a participant in transaction {transaction_id}")

        participant = transaction.participants[database_name]

        # Create operation
        operation = TransactionOperation(
            operation_id=f"op_{uuid.uuid4().hex[:8]}",
            transaction_id=transaction_id,
            database_name=database_name,
            sql_statement=sql,
            parameters=parameters
        )

        start_time = time.perf_counter()

        try:
            # Get or create connection for participant
            if participant.connection is None:
                adapter = self.connection_manager.get_adapter(database_name)
                participant.connection = adapter._engine.connect()

                # Begin transaction on database
                trans = participant.connection.begin()
                participant.connection._sqltest_transaction = trans

                # Set isolation level
                participant.connection.execute(
                    text(f"SET TRANSACTION ISOLATION LEVEL {transaction.isolation_level.value}")
                )

            # Execute operation
            result = participant.connection.execute(text(sql), parameters or {})

            execution_time = (time.perf_counter() - start_time) * 1000

            # Update operation
            operation.executed_at = datetime.now()
            operation.execution_time_ms = execution_time
            operation.rows_affected = result.rowcount if hasattr(result, 'rowcount') else 0
            operation.success = True

            # Add to participant operations
            participant.operations.append(operation)
            self._stats['total_operations'] += 1

            logger.debug(f"Executed operation {operation.operation_id} in transaction {transaction_id}")

            return operation

        except Exception as e:
            execution_time = (time.perf_counter() - start_time) * 1000

            operation.executed_at = datetime.now()
            operation.execution_time_ms = execution_time
            operation.success = False
            operation.error_message = str(e)

            participant.operations.append(operation)

            logger.error(f"Operation failed in transaction {transaction_id}: {e}")
            raise

    def create_savepoint(self, transaction_id: str, database_name: str,
                        savepoint_name: Optional[str] = None) -> Savepoint:
        """Create a savepoint within a transaction.

        Args:
            transaction_id: Transaction ID
            database_name: Database to create savepoint on
            savepoint_name: Optional savepoint name (auto-generated if not provided)

        Returns:
            Created Savepoint

        Raises:
            ValueError: If transaction or participant not found
        """
        transaction = self._get_transaction(transaction_id)
        participant = transaction.participants.get(database_name)

        if not participant:
            raise ValueError(f"Database {database_name} is not a participant in transaction {transaction_id}")

        if not participant.connection:
            raise RuntimeError(f"No active connection for database {database_name}")

        # Generate savepoint name if not provided
        if not savepoint_name:
            savepoint_name = f"sp_{len(participant.savepoints) + 1}"

        # Create savepoint
        savepoint = Savepoint(
            name=savepoint_name,
            transaction_id=transaction_id,
            database_name=database_name,
            executed_operations=len(participant.operations)
        )

        # Execute SAVEPOINT command
        try:
            participant.connection.execute(text(f"SAVEPOINT {savepoint_name}"))
            participant.savepoints.append(savepoint)

            logger.debug(f"Created savepoint {savepoint_name} in transaction {transaction_id}")

            return savepoint

        except Exception as e:
            logger.error(f"Failed to create savepoint {savepoint_name}: {e}")
            raise

    def rollback_to_savepoint(self, transaction_id: str, database_name: str,
                              savepoint_name: str) -> bool:
        """Rollback to a specific savepoint.

        Args:
            transaction_id: Transaction ID
            database_name: Database to rollback
            savepoint_name: Savepoint name to rollback to

        Returns:
            True if rollback successful

        Raises:
            ValueError: If savepoint not found
        """
        transaction = self._get_transaction(transaction_id)
        participant = transaction.participants.get(database_name)

        if not participant or not participant.connection:
            raise ValueError(f"No active connection for database {database_name}")

        # Find savepoint
        savepoint = next((sp for sp in participant.savepoints if sp.name == savepoint_name), None)
        if not savepoint:
            raise ValueError(f"Savepoint {savepoint_name} not found")

        try:
            # Execute ROLLBACK TO SAVEPOINT command
            participant.connection.execute(text(f"ROLLBACK TO SAVEPOINT {savepoint_name}"))

            # Remove operations after savepoint
            participant.operations = participant.operations[:savepoint.executed_operations]

            # Remove savepoints created after this one
            savepoint_index = participant.savepoints.index(savepoint)
            participant.savepoints = participant.savepoints[:savepoint_index + 1]

            logger.info(f"Rolled back to savepoint {savepoint_name} in transaction {transaction_id}")

            return True

        except Exception as e:
            logger.error(f"Failed to rollback to savepoint {savepoint_name}: {e}")
            raise

    def prepare(self, transaction_id: str) -> bool:
        """Prepare phase of two-phase commit.

        Args:
            transaction_id: Transaction ID

        Returns:
            True if all participants prepared successfully

        Raises:
            RuntimeError: If transaction is not in valid state
        """
        transaction = self._get_transaction(transaction_id)

        if transaction.state != TransactionState.INITIATED:
            raise RuntimeError(f"Cannot prepare transaction in state {transaction.state}")

        transaction.state = TransactionState.PREPARING
        logger.info(f"Starting prepare phase for transaction {transaction_id}")

        all_prepared = True
        failed_participants = []

        for db_name, participant in transaction.participants.items():
            try:
                # Flush any pending operations
                if participant.connection:
                    participant.connection.execute(text("SELECT 1"))  # Test connection

                participant.state = ParticipantState.PREPARED
                participant.prepared_at = datetime.now()

                logger.debug(f"Participant {db_name} prepared for transaction {transaction_id}")

            except Exception as e:
                participant.state = ParticipantState.FAILED
                participant.error_message = str(e)
                all_prepared = False
                failed_participants.append(db_name)

                logger.error(f"Failed to prepare participant {db_name}: {e}")

        if all_prepared:
            transaction.state = TransactionState.PREPARED
            transaction.prepared_at = datetime.now()
            logger.info(f"Transaction {transaction_id} prepared successfully")
        else:
            transaction.state = TransactionState.FAILED
            logger.error(f"Transaction {transaction_id} prepare failed: {failed_participants}")

        return all_prepared

    def commit(self, transaction_id: str) -> bool:
        """Commit phase of two-phase commit.

        Args:
            transaction_id: Transaction ID

        Returns:
            True if commit successful

        Raises:
            RuntimeError: If transaction is not prepared
        """
        transaction = self._get_transaction(transaction_id)

        if transaction.state != TransactionState.PREPARED:
            raise RuntimeError(f"Cannot commit transaction in state {transaction.state}")

        transaction.state = TransactionState.COMMITTING
        logger.info(f"Starting commit phase for transaction {transaction_id}")

        all_committed = True
        failed_participants = []

        for db_name, participant in transaction.participants.items():
            try:
                if participant.connection and hasattr(participant.connection, '_sqltest_transaction'):
                    # Commit the transaction
                    participant.connection._sqltest_transaction.commit()
                    participant.connection.close()
                    participant.connection = None

                participant.state = ParticipantState.COMMITTED
                participant.committed_at = datetime.now()

                logger.debug(f"Participant {db_name} committed for transaction {transaction_id}")

            except Exception as e:
                participant.state = ParticipantState.FAILED
                participant.error_message = str(e)
                all_committed = False
                failed_participants.append(db_name)

                logger.error(f"Failed to commit participant {db_name}: {e}")

        if all_committed:
            transaction.state = TransactionState.COMMITTED
            transaction.committed_at = datetime.now()
            self._stats['committed_transactions'] += 1

            # Remove from active transactions
            with self._lock:
                self._active_transactions.discard(transaction_id)
                self._archive_transaction(transaction)

            logger.info(f"Transaction {transaction_id} committed successfully in {transaction.duration_ms:.2f}ms")
        else:
            transaction.state = TransactionState.FAILED
            self._stats['failed_transactions'] += 1
            logger.error(f"Transaction {transaction_id} commit failed: {failed_participants}")

        return all_committed

    def abort(self, transaction_id: str, reason: Optional[str] = None) -> bool:
        """Abort a transaction and rollback all participants.

        Args:
            transaction_id: Transaction ID
            reason: Optional reason for abort

        Returns:
            True if abort successful
        """
        transaction = self._get_transaction(transaction_id)

        transaction.state = TransactionState.ABORTING
        logger.info(f"Aborting transaction {transaction_id}: {reason or 'manual abort'}")

        for db_name, participant in transaction.participants.items():
            try:
                if participant.connection:
                    if hasattr(participant.connection, '_sqltest_transaction'):
                        participant.connection._sqltest_transaction.rollback()
                    participant.connection.close()
                    participant.connection = None

                participant.state = ParticipantState.ABORTED

                logger.debug(f"Participant {db_name} aborted for transaction {transaction_id}")

            except Exception as e:
                logger.error(f"Error aborting participant {db_name}: {e}")

        transaction.state = TransactionState.ABORTED
        transaction.aborted_at = datetime.now()
        self._stats['aborted_transactions'] += 1

        # Remove from active transactions
        with self._lock:
            self._active_transactions.discard(transaction_id)
            self._archive_transaction(transaction)

        logger.info(f"Transaction {transaction_id} aborted")

        return True

    def two_phase_commit(self, transaction_id: str) -> bool:
        """Execute complete two-phase commit protocol.

        Args:
            transaction_id: Transaction ID

        Returns:
            True if transaction committed successfully

        Raises:
            RuntimeError: If 2PC protocol fails
        """
        try:
            # Phase 1: Prepare
            if not self.prepare(transaction_id):
                logger.error(f"Prepare phase failed for transaction {transaction_id}, aborting")
                self.abort(transaction_id, "Prepare phase failed")
                return False

            # Phase 2: Commit
            if not self.commit(transaction_id):
                logger.error(f"Commit phase failed for transaction {transaction_id}")
                # Note: At this point, some participants may have committed
                # This is a partial commit scenario that needs manual intervention
                return False

            return True

        except Exception as e:
            logger.error(f"Two-phase commit failed for transaction {transaction_id}: {e}")
            try:
                self.abort(transaction_id, f"2PC error: {e}")
            except Exception as abort_error:
                logger.error(f"Failed to abort transaction after 2PC error: {abort_error}")
            return False

    def _get_transaction(self, transaction_id: str) -> DistributedTransaction:
        """Get transaction by ID.

        Args:
            transaction_id: Transaction ID

        Returns:
            DistributedTransaction

        Raises:
            ValueError: If transaction not found
        """
        transaction = self._transactions.get(transaction_id)
        if not transaction:
            raise ValueError(f"Transaction {transaction_id} not found")
        return transaction

    def _archive_transaction(self, transaction: DistributedTransaction):
        """Archive completed transaction to log."""
        self._transaction_log.append(transaction)

        # Trim log if too large
        if len(self._transaction_log) > self._max_log_size:
            self._transaction_log = self._transaction_log[-self._max_log_size:]

        # Update statistics
        if transaction.state == TransactionState.COMMITTED:
            total_duration = sum(
                t.duration_ms for t in self._transaction_log
                if t.state == TransactionState.COMMITTED
            )
            self._stats['average_duration_ms'] = total_duration / self._stats['committed_transactions']

    def _cleanup_stale_transactions(self):
        """Background thread to cleanup stale transactions."""
        while not self._cleanup_event.wait(60):  # Check every minute
            try:
                now = datetime.now()
                stale_transactions = []

                with self._lock:
                    for txn_id, transaction in list(self._transactions.items()):
                        if not transaction.is_active:
                            continue

                        # Check timeout
                        age = (now - transaction.created_at).total_seconds()
                        if age > transaction.timeout_seconds:
                            stale_transactions.append(txn_id)

                # Abort stale transactions
                for txn_id in stale_transactions:
                    logger.warning(f"Aborting stale transaction {txn_id} due to timeout")
                    try:
                        self.abort(txn_id, "Transaction timeout")
                        self._stats['timeout_transactions'] += 1
                    except Exception as e:
                        logger.error(f"Failed to abort stale transaction {txn_id}: {e}")

            except Exception as e:
                logger.error(f"Error in transaction cleanup: {e}")

    def get_transaction_status(self, transaction_id: str) -> Dict[str, Any]:
        """Get detailed status of a transaction.

        Args:
            transaction_id: Transaction ID

        Returns:
            Dictionary with transaction status
        """
        transaction = self._get_transaction(transaction_id)

        return {
            'transaction_id': transaction.transaction_id,
            'state': transaction.state.value,
            'isolation_level': transaction.isolation_level.value,
            'created_at': transaction.created_at.isoformat(),
            'duration_ms': transaction.duration_ms,
            'total_operations': transaction.total_operations,
            'participants': {
                db_name: {
                    'state': p.state.value,
                    'operations_count': len(p.operations),
                    'savepoints_count': len(p.savepoints),
                    'prepared_at': p.prepared_at.isoformat() if p.prepared_at else None,
                    'committed_at': p.committed_at.isoformat() if p.committed_at else None,
                    'error_message': p.error_message
                }
                for db_name, p in transaction.participants.items()
            },
            'metadata': transaction.metadata
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get transaction manager statistics.

        Returns:
            Dictionary with statistics
        """
        with self._lock:
            return {
                **self._stats,
                'active_transactions': len(self._active_transactions),
                'max_concurrent_transactions': self.max_concurrent_transactions,
                'transaction_log_size': len(self._transaction_log)
            }

    def shutdown(self):
        """Shutdown transaction manager."""
        logger.info("Shutting down transaction manager")

        # Stop cleanup thread
        self._cleanup_event.set()
        if self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5)

        # Abort all active transactions
        with self._lock:
            for txn_id in list(self._active_transactions):
                try:
                    self.abort(txn_id, "Transaction manager shutdown")
                except Exception as e:
                    logger.error(f"Failed to abort transaction {txn_id} during shutdown: {e}")

        logger.info("Transaction manager shutdown complete")
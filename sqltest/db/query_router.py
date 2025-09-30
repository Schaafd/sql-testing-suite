"""Intelligent query routing with read/write splitting and load balancing."""

import hashlib
import logging
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from threading import Lock
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class QueryType(str, Enum):
    """Types of SQL queries for routing."""
    READ = "read"
    WRITE = "write"
    DDL = "ddl"
    TRANSACTION = "transaction"
    UNKNOWN = "unknown"


class RoutingStrategy(str, Enum):
    """Routing strategies for read queries."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"
    RANDOM = "random"
    WEIGHTED = "weighted"


@dataclass
class DatabaseNode:
    """Represents a database node (primary or replica)."""
    node_id: str
    database_name: str
    is_primary: bool
    is_available: bool = True
    weight: int = 1
    current_connections: int = 0
    total_queries: int = 0
    total_errors: int = 0
    average_response_time_ms: float = 0.0
    last_health_check: Optional[datetime] = None
    health_check_failures: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingDecision:
    """Represents a routing decision for a query."""
    query_hash: str
    query_type: QueryType
    target_node_id: str
    target_database: str
    routing_strategy: str
    decision_time_ms: float
    timestamp: datetime = field(default_factory=datetime.now)


class QueryRouter:
    """Routes queries to appropriate database nodes with read/write splitting."""

    def __init__(self, connection_manager,
                 default_routing_strategy: RoutingStrategy = RoutingStrategy.LEAST_RESPONSE_TIME,
                 health_check_interval: int = 30):
        """Initialize query router.

        Args:
            connection_manager: Database connection manager
            default_routing_strategy: Default strategy for routing read queries
            health_check_interval: Interval for health checks in seconds
        """
        self.connection_manager = connection_manager
        self.default_routing_strategy = default_routing_strategy
        self.health_check_interval = health_check_interval

        # Node registry
        self._lock = Lock()
        self._nodes: Dict[str, DatabaseNode] = {}
        self._primary_nodes: Dict[str, str] = {}  # database_name -> node_id
        self._replica_nodes: Dict[str, List[str]] = defaultdict(list)  # database_name -> [node_ids]

        # Routing state
        self._round_robin_counters: Dict[str, int] = defaultdict(int)
        self._routing_history: List[RoutingDecision] = []
        self._max_history_size = 10000

        # Query pattern cache
        self._query_patterns = self._compile_query_patterns()

        # Statistics
        self._stats = {
            'total_queries': 0,
            'read_queries': 0,
            'write_queries': 0,
            'routed_to_primary': 0,
            'routed_to_replica': 0,
            'routing_failures': 0,
            'failover_count': 0
        }

    def _compile_query_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for query classification."""
        return {
            # Read operations
            'select': re.compile(r'^\s*SELECT\b', re.IGNORECASE),
            'show': re.compile(r'^\s*SHOW\b', re.IGNORECASE),
            'describe': re.compile(r'^\s*(DESCRIBE|DESC)\b', re.IGNORECASE),
            'explain': re.compile(r'^\s*EXPLAIN\b', re.IGNORECASE),

            # Write operations
            'insert': re.compile(r'^\s*INSERT\b', re.IGNORECASE),
            'update': re.compile(r'^\s*UPDATE\b', re.IGNORECASE),
            'delete': re.compile(r'^\s*DELETE\b', re.IGNORECASE),
            'replace': re.compile(r'^\s*REPLACE\b', re.IGNORECASE),

            # DDL operations
            'create': re.compile(r'^\s*CREATE\b', re.IGNORECASE),
            'alter': re.compile(r'^\s*ALTER\b', re.IGNORECASE),
            'drop': re.compile(r'^\s*DROP\b', re.IGNORECASE),
            'truncate': re.compile(r'^\s*TRUNCATE\b', re.IGNORECASE),

            # Transaction operations
            'begin': re.compile(r'^\s*(BEGIN|START\s+TRANSACTION)\b', re.IGNORECASE),
            'commit': re.compile(r'^\s*COMMIT\b', re.IGNORECASE),
            'rollback': re.compile(r'^\s*ROLLBACK\b', re.IGNORECASE),

            # Locking hints (should go to primary)
            'for_update': re.compile(r'\bFOR\s+UPDATE\b', re.IGNORECASE),
            'lock_in_share_mode': re.compile(r'\bLOCK\s+IN\s+SHARE\s+MODE\b', re.IGNORECASE),
        }

    def register_primary(self, node_id: str, database_name: str,
                        metadata: Optional[Dict[str, Any]] = None) -> DatabaseNode:
        """Register a primary database node.

        Args:
            node_id: Unique node identifier
            database_name: Database name
            metadata: Optional node metadata

        Returns:
            Registered DatabaseNode
        """
        with self._lock:
            node = DatabaseNode(
                node_id=node_id,
                database_name=database_name,
                is_primary=True,
                metadata=metadata or {}
            )

            self._nodes[node_id] = node
            self._primary_nodes[database_name] = node_id

            logger.info(f"Registered primary node {node_id} for database {database_name}")

            return node

    def register_replica(self, node_id: str, database_name: str,
                        weight: int = 1,
                        metadata: Optional[Dict[str, Any]] = None) -> DatabaseNode:
        """Register a read replica node.

        Args:
            node_id: Unique node identifier
            database_name: Database name
            weight: Weight for weighted routing (higher = more traffic)
            metadata: Optional node metadata

        Returns:
            Registered DatabaseNode
        """
        with self._lock:
            node = DatabaseNode(
                node_id=node_id,
                database_name=database_name,
                is_primary=False,
                weight=weight,
                metadata=metadata or {}
            )

            self._nodes[node_id] = node
            self._replica_nodes[database_name].append(node_id)

            logger.info(f"Registered replica node {node_id} for database {database_name} (weight={weight})")

            return node

    def unregister_node(self, node_id: str) -> bool:
        """Unregister a database node.

        Args:
            node_id: Node identifier

        Returns:
            True if node was unregistered
        """
        with self._lock:
            node = self._nodes.get(node_id)
            if not node:
                return False

            # Remove from primary nodes
            if node.is_primary and node.database_name in self._primary_nodes:
                if self._primary_nodes[node.database_name] == node_id:
                    del self._primary_nodes[node.database_name]

            # Remove from replica nodes
            if not node.is_primary and node.database_name in self._replica_nodes:
                self._replica_nodes[node.database_name].remove(node_id)
                if not self._replica_nodes[node.database_name]:
                    del self._replica_nodes[node.database_name]

            del self._nodes[node_id]

            logger.info(f"Unregistered node {node_id}")

            return True

    def classify_query(self, sql: str) -> QueryType:
        """Classify a SQL query by type.

        Args:
            sql: SQL query string

        Returns:
            QueryType classification
        """
        # Remove comments
        sql_clean = re.sub(r'--.*$', '', sql, flags=re.MULTILINE)
        sql_clean = re.sub(r'/\*.*?\*/', '', sql_clean, flags=re.DOTALL)
        sql_clean = sql_clean.strip()

        # Check for locking hints in SELECT (should go to primary)
        if self._query_patterns['select'].match(sql_clean):
            if (self._query_patterns['for_update'].search(sql_clean) or
                self._query_patterns['lock_in_share_mode'].search(sql_clean)):
                return QueryType.WRITE  # Treat as write due to locking

        # Check query patterns
        if any(pattern.match(sql_clean) for name, pattern in self._query_patterns.items()
               if name in ['select', 'show', 'describe', 'explain']):
            return QueryType.READ

        if any(pattern.match(sql_clean) for name, pattern in self._query_patterns.items()
               if name in ['insert', 'update', 'delete', 'replace']):
            return QueryType.WRITE

        if any(pattern.match(sql_clean) for name, pattern in self._query_patterns.items()
               if name in ['create', 'alter', 'drop', 'truncate']):
            return QueryType.DDL

        if any(pattern.match(sql_clean) for name, pattern in self._query_patterns.items()
               if name in ['begin', 'commit', 'rollback']):
            return QueryType.TRANSACTION

        return QueryType.UNKNOWN

    def route_query(self, sql: str, database_name: str,
                   force_primary: bool = False,
                   routing_strategy: Optional[RoutingStrategy] = None) -> Tuple[str, str]:
        """Route a query to appropriate database node.

        Args:
            sql: SQL query string
            database_name: Target database name
            force_primary: Force routing to primary node
            routing_strategy: Override default routing strategy for read queries

        Returns:
            Tuple of (node_id, actual_database_name)

        Raises:
            ValueError: If no suitable node found
        """
        start_time = time.perf_counter()

        # Classify query
        query_type = self.classify_query(sql)
        query_hash = hashlib.md5(sql.encode()).hexdigest()[:12]

        # Determine target node
        if force_primary or query_type in [QueryType.WRITE, QueryType.DDL, QueryType.TRANSACTION]:
            # Route to primary
            node_id = self._route_to_primary(database_name)
            strategy_used = "primary_only"
            self._stats['routed_to_primary'] += 1

        elif query_type == QueryType.READ:
            # Route to replica (or primary if no replicas available)
            strategy = routing_strategy or self.default_routing_strategy
            node_id = self._route_to_replica(database_name, strategy)
            strategy_used = strategy.value
            self._stats['routed_to_replica'] += 1

        else:
            # Unknown query type - route to primary for safety
            node_id = self._route_to_primary(database_name)
            strategy_used = "primary_fallback"
            self._stats['routed_to_primary'] += 1

        decision_time = (time.perf_counter() - start_time) * 1000

        # Get node details
        node = self._nodes[node_id]

        # Record routing decision
        decision = RoutingDecision(
            query_hash=query_hash,
            query_type=query_type,
            target_node_id=node_id,
            target_database=node.database_name,
            routing_strategy=strategy_used,
            decision_time_ms=decision_time
        )

        with self._lock:
            self._routing_history.append(decision)
            if len(self._routing_history) > self._max_history_size:
                self._routing_history = self._routing_history[-self._max_history_size:]

            self._stats['total_queries'] += 1
            if query_type == QueryType.READ:
                self._stats['read_queries'] += 1
            elif query_type == QueryType.WRITE:
                self._stats['write_queries'] += 1

        logger.debug(f"Routed {query_type.value} query to {node_id} using {strategy_used}")

        return node_id, node.database_name

    def _route_to_primary(self, database_name: str) -> str:
        """Route to primary node for database.

        Args:
            database_name: Database name

        Returns:
            Primary node ID

        Raises:
            ValueError: If no primary node found
        """
        with self._lock:
            if database_name not in self._primary_nodes:
                raise ValueError(f"No primary node registered for database {database_name}")

            node_id = self._primary_nodes[database_name]
            node = self._nodes[node_id]

            if not node.is_available:
                raise ValueError(f"Primary node {node_id} is not available")

            return node_id

    def _route_to_replica(self, database_name: str, strategy: RoutingStrategy) -> str:
        """Route to replica node using specified strategy.

        Args:
            database_name: Database name
            strategy: Routing strategy

        Returns:
            Selected replica node ID (or primary if no replicas available)
        """
        with self._lock:
            # Get available replicas
            replica_ids = self._replica_nodes.get(database_name, [])
            available_replicas = [
                node_id for node_id in replica_ids
                if self._nodes[node_id].is_available
            ]

            # If no replicas available, fallback to primary
            if not available_replicas:
                logger.debug(f"No replicas available for {database_name}, routing to primary")
                return self._route_to_primary(database_name)

            # Apply routing strategy
            if strategy == RoutingStrategy.ROUND_ROBIN:
                return self._route_round_robin(database_name, available_replicas)

            elif strategy == RoutingStrategy.LEAST_CONNECTIONS:
                return self._route_least_connections(available_replicas)

            elif strategy == RoutingStrategy.LEAST_RESPONSE_TIME:
                return self._route_least_response_time(available_replicas)

            elif strategy == RoutingStrategy.RANDOM:
                return random.choice(available_replicas)

            elif strategy == RoutingStrategy.WEIGHTED:
                return self._route_weighted(available_replicas)

            else:
                # Default to round robin
                return self._route_round_robin(database_name, available_replicas)

    def _route_round_robin(self, database_name: str, replica_ids: List[str]) -> str:
        """Round-robin routing strategy."""
        counter = self._round_robin_counters[database_name]
        selected = replica_ids[counter % len(replica_ids)]
        self._round_robin_counters[database_name] = counter + 1
        return selected

    def _route_least_connections(self, replica_ids: List[str]) -> str:
        """Route to replica with least active connections."""
        return min(replica_ids, key=lambda nid: self._nodes[nid].current_connections)

    def _route_least_response_time(self, replica_ids: List[str]) -> str:
        """Route to replica with best average response time."""
        return min(replica_ids, key=lambda nid: self._nodes[nid].average_response_time_ms)

    def _route_weighted(self, replica_ids: List[str]) -> str:
        """Weighted random routing based on node weights."""
        nodes = [self._nodes[nid] for nid in replica_ids]
        weights = [node.weight for node in nodes]
        return random.choices(replica_ids, weights=weights)[0]

    def update_node_metrics(self, node_id: str,
                           response_time_ms: Optional[float] = None,
                           connection_delta: int = 0,
                           success: bool = True):
        """Update node performance metrics.

        Args:
            node_id: Node identifier
            response_time_ms: Query response time in milliseconds
            connection_delta: Change in connection count (+1 or -1)
            success: Whether query succeeded
        """
        with self._lock:
            node = self._nodes.get(node_id)
            if not node:
                return

            node.current_connections = max(0, node.current_connections + connection_delta)
            node.total_queries += 1

            if not success:
                node.total_errors += 1

            if response_time_ms is not None:
                # Update rolling average
                if node.average_response_time_ms == 0:
                    node.average_response_time_ms = response_time_ms
                else:
                    alpha = 0.2  # Smoothing factor
                    node.average_response_time_ms = (
                        alpha * response_time_ms +
                        (1 - alpha) * node.average_response_time_ms
                    )

    def mark_node_unavailable(self, node_id: str, reason: Optional[str] = None):
        """Mark a node as unavailable.

        Args:
            node_id: Node identifier
            reason: Optional reason for unavailability
        """
        with self._lock:
            node = self._nodes.get(node_id)
            if not node:
                return

            node.is_available = False
            node.health_check_failures += 1

            logger.warning(f"Node {node_id} marked unavailable: {reason}")

            # If primary node failed, trigger failover
            if node.is_primary:
                self._attempt_failover(node.database_name)

    def mark_node_available(self, node_id: str):
        """Mark a node as available.

        Args:
            node_id: Node identifier
        """
        with self._lock:
            node = self._nodes.get(node_id)
            if not node:
                return

            node.is_available = True
            node.health_check_failures = 0
            node.last_health_check = datetime.now()

            logger.info(f"Node {node_id} marked available")

    def _attempt_failover(self, database_name: str):
        """Attempt to failover to a replica when primary fails.

        Args:
            database_name: Database name
        """
        logger.warning(f"Attempting failover for database {database_name}")

        # Get available replicas
        replica_ids = self._replica_nodes.get(database_name, [])
        available_replicas = [
            node_id for node_id in replica_ids
            if self._nodes[node_id].is_available
        ]

        if not available_replicas:
            logger.error(f"No available replicas for failover of database {database_name}")
            return

        # Promote best replica to primary
        # Use replica with best response time
        new_primary_id = min(available_replicas,
                           key=lambda nid: self._nodes[nid].average_response_time_ms)

        new_primary = self._nodes[new_primary_id]
        new_primary.is_primary = True

        self._primary_nodes[database_name] = new_primary_id
        self._replica_nodes[database_name].remove(new_primary_id)

        self._stats['failover_count'] += 1

        logger.info(f"Failover complete: {new_primary_id} promoted to primary for {database_name}")

    def get_node_status(self, node_id: Optional[str] = None) -> Dict[str, Any]:
        """Get status of one or all nodes.

        Args:
            node_id: Optional node ID (returns all if None)

        Returns:
            Dictionary with node status information
        """
        with self._lock:
            if node_id:
                node = self._nodes.get(node_id)
                if not node:
                    return {}

                return self._node_to_dict(node)

            return {
                nid: self._node_to_dict(node)
                for nid, node in self._nodes.items()
            }

    def _node_to_dict(self, node: DatabaseNode) -> Dict[str, Any]:
        """Convert node to dictionary representation."""
        return {
            'node_id': node.node_id,
            'database_name': node.database_name,
            'is_primary': node.is_primary,
            'is_available': node.is_available,
            'weight': node.weight,
            'current_connections': node.current_connections,
            'total_queries': node.total_queries,
            'total_errors': node.total_errors,
            'error_rate': node.total_errors / node.total_queries if node.total_queries > 0 else 0,
            'average_response_time_ms': node.average_response_time_ms,
            'last_health_check': node.last_health_check.isoformat() if node.last_health_check else None,
            'health_check_failures': node.health_check_failures
        }

    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get routing statistics.

        Returns:
            Dictionary with routing statistics
        """
        with self._lock:
            recent_decisions = self._routing_history[-1000:] if self._routing_history else []

            query_type_distribution = defaultdict(int)
            strategy_distribution = defaultdict(int)
            avg_decision_time = 0

            for decision in recent_decisions:
                query_type_distribution[decision.query_type.value] += 1
                strategy_distribution[decision.routing_strategy] += 1
                avg_decision_time += decision.decision_time_ms

            if recent_decisions:
                avg_decision_time /= len(recent_decisions)

            return {
                **self._stats,
                'total_nodes': len(self._nodes),
                'total_primary_nodes': len(self._primary_nodes),
                'total_replica_nodes': sum(len(replicas) for replicas in self._replica_nodes.values()),
                'query_type_distribution': dict(query_type_distribution),
                'strategy_distribution': dict(strategy_distribution),
                'avg_routing_decision_time_ms': avg_decision_time,
                'routing_history_size': len(self._routing_history)
            }
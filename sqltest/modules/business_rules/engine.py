"""Business rule execution engine for SQLTest Pro."""

import asyncio
import copy
import hashlib
import inspect
import logging
import time
import uuid
from collections import Counter, OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeoutError
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set

import pandas as pd

from ...db.connection import ConnectionManager
from ...exceptions import ValidationError, DatabaseError
from .models import (
    BusinessRule,
    RuleSet,
    RuleResult,
    RuleViolation,
    RuleStatus,
    RuleBatch,
    RuleSeverity,
    ValidationContext,
    ValidationSummary,
    RuleType,
    ValidationScope,
)

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """Lightweight metrics collector for rule execution."""

    def __init__(self) -> None:
        self.counters: Counter[str] = Counter()
        self.values: Dict[str, List[float]] = defaultdict(list)
        self.execution_events: List[Dict[str, Any]] = []

    def increment_counter(self, name: str, value: int = 1) -> None:
        self.counters[name] += value

    def record_value(self, name: str, value: float) -> None:
        self.values[name].append(value)

    def record_execution_time(self, name: str, duration_ms: float) -> None:
        self.execution_events.append({"name": name, "duration_ms": duration_ms})

    def get_all_stats(self) -> Dict[str, Any]:
        value_stats = {
            key: {
                "count": len(values),
                "avg": sum(values) / len(values) if values else 0.0,
                "min": min(values) if values else 0.0,
                "max": max(values) if values else 0.0,
            }
            for key, values in self.values.items()
        }

        return {
            "counters": dict(self.counters),
            "values": value_stats,
            "execution_events": list(self.execution_events),
        }


class CacheManager:
    """Simple in-memory cache for rule results."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self._store: "OrderedDict[str, Any]" = OrderedDict()
        self.max_size: Optional[int] = self.config.get("l1_max_size")
        self.ttl_seconds: Optional[float] = self.config.get("l1_ttl_seconds")

    def build_key(self, rule: BusinessRule, context: ValidationContext) -> str:
        parts = [
            rule.name,
            rule.sql_query,
            getattr(context, "database_name", None),
            getattr(context, "schema_name", None),
            getattr(context, "table_name", None),
            getattr(context, "query", None),
            tuple(sorted((k, repr(v)) for k, v in (context.parameters or {}).items())),
        ]
        return "|".join(str(part) for part in parts if part is not None)

    def get(self, key: str) -> Optional[RuleResult]:
        entry = self._store.get(key)
        if not entry:
            return None

        value, expires_at = entry
        if expires_at and expires_at < time.time():
            self._store.pop(key, None)
            return None

        self._store.move_to_end(key)
        return copy.deepcopy(value)

    def set(self, key: str, value: RuleResult) -> None:
        expires_at = None
        if self.ttl_seconds:
            expires_at = time.time() + self.ttl_seconds

        if self.max_size and len(self._store) >= self.max_size:
            self._store.popitem(last=False)

        self._store[key] = (copy.deepcopy(value), expires_at)

    def invalidate(self, pattern: Optional[str] = None) -> None:
        if pattern is None:
            self._store.clear()
            return

        keys_to_delete = [key for key in self._store if pattern in key]
        for key in keys_to_delete:
            self._store.pop(key, None)

    def get_cache_stats(self) -> Dict[str, Any]:
        return {
            "l1_size": len(self._store),
            "config": dict(self.config),
        }


class RetryManager:
    """Retry helper with exponential backoff."""

    def __init__(self, max_retries: int = 0, base_delay: float = 0.0) -> None:
        self.max_retries = max(0, max_retries)
        self.base_delay = max(0.0, base_delay)

    def execute_with_retry(self, func: Callable, *args, **kwargs):
        call_kwargs = dict(kwargs)
        call_kwargs.pop("use_cache", None)

        attempt = 0
        while True:
            try:
                return func(*args, **call_kwargs)
            except Exception:
                if attempt >= self.max_retries:
                    raise
                delay = self.base_delay * (2 ** attempt)
                if delay > 0:
                    time.sleep(delay)
                attempt += 1
class BusinessRuleEngine:
    """Core engine for executing business rules and validations."""

    def __init__(
        self,
        connection_manager: ConnectionManager,
        max_workers: int = 5,
        *,
        enable_caching: bool = False,
        enable_metrics: bool = False,
        cache_config: Optional[Dict[str, Any]] = None,
        retry_config: Optional[Dict[str, Any]] = None,
        enable_batching: bool = False,
        batch_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the business rule engine."""

        self.connection_manager = connection_manager
        self.max_workers = max_workers
        self._rule_cache: Dict[str, Any] = {}
        self._dependency_graph: Dict[str, Set[str]] = {}

        # Advanced feature flags
        self.enable_batching = enable_batching
        batch_settings = batch_config or {}
        self.max_batch_size = max(1, int(batch_settings.get("max_batch_size", 5)))

        self._cache_config = cache_config or {}
        self.cache_manager = CacheManager(self._cache_config) if enable_caching else None
        self.metrics = PerformanceMetrics() if enable_metrics else None

        retry_settings = retry_config or {}
        self.retry_manager = RetryManager(
            max_retries=retry_settings.get("max_retries", 0),
            base_delay=retry_settings.get("base_delay", 0.0),
        )

        # ensure property toggles work even when tests mutate internal flag
        self._cache_enabled = enable_caching

    def execute_rule(
        self,
        rule: BusinessRule,
        context: ValidationContext,
        timeout_override: Optional[float] = None
    ) -> RuleResult:
        """Execute a single business rule.
        
        Args:
            rule: Business rule to execute
            context: Validation context
            timeout_override: Optional timeout override in seconds
            
        Returns:
            RuleResult with execution details and violations
            
        Raises:
            ValidationError: If rule execution fails
            TimeoutError: If rule execution times out
        """
        start_time = time.time()
        timeout = timeout_override or rule.timeout_seconds

        cache_key = None
        if self.cache_manager:
            cache_key = self.cache_manager.build_key(rule, context)
            cached_result = self.cache_manager.get(cache_key)
            if cached_result is not None:
                if self.metrics:
                    self.metrics.increment_counter("cache_hits")
                return cached_result
            if self.metrics:
                self.metrics.increment_counter("cache_misses")

        logger.info(f"Executing rule: {rule.name}")

        try:
            # Check if rule is enabled
            if not rule.enabled:
                return RuleResult(
                    rule_name=rule.name,
                    rule_type=rule.rule_type,
                    status=RuleStatus.SKIPPED,
                    severity=rule.severity,
                    scope=rule.scope,
                    passed=True,
                    message="Rule is disabled"
                )
            
            # Execute the rule based on its type
            if rule.custom_function:
                execute_fn = lambda: self._execute_custom_rule(rule, context, timeout)
            elif rule.sql_query:
                execute_fn = lambda: self._execute_sql_rule(rule, context, timeout)
            else:
                raise ValidationError(f"Rule '{rule.name}' has neither SQL query nor custom function")

            if self.retry_manager and self.retry_manager.max_retries > 0:
                result = self.retry_manager.execute_with_retry(execute_fn)
            else:
                result = execute_fn()
            
            # Calculate execution time
            end_time = time.time()
            result.execution_time_ms = (end_time - start_time) * 1000
            
            logger.info(f"Rule '{rule.name}' completed: {result.status} ({result.execution_time_ms:.2f}ms)")
            
            if self.metrics:
                self.metrics.increment_counter("rules_executed")
                self.metrics.record_value("rule_execution_ms", result.execution_time_ms)

            if self.cache_manager and cache_key and getattr(rule, "cacheable", True):
                self.cache_manager.set(cache_key, result)
                if self.metrics:
                    self.metrics.increment_counter("cache_writes")

            return result
            
        except TimeoutError:
            logger.warning(f"Rule '{rule.name}' timed out after {timeout} seconds")
            return RuleResult(
                rule_name=rule.name,
                rule_type=rule.rule_type,
                status=RuleStatus.ERROR,
                severity=rule.severity,
                scope=rule.scope,
                passed=False,
                message=f"Rule execution timed out after {timeout} seconds",
                execution_time_ms=(time.time() - start_time) * 1000
            )
            
        except Exception as e:
            logger.error(f"Rule '{rule.name}' failed with error: {str(e)}")
            return RuleResult(
                rule_name=rule.name,
                rule_type=rule.rule_type,
                status=RuleStatus.ERROR,
                severity=rule.severity,
                scope=rule.scope,
                passed=False,
                message=f"Rule execution failed: {str(e)}",
                execution_time_ms=(time.time() - start_time) * 1000,
                context={"error": str(e), "error_type": type(e).__name__}
            )
    
    def execute_rule_set(
        self,
        rule_set: RuleSet,
        context: ValidationContext,
        parallel: Optional[bool] = None,
        fail_fast: bool = False,
        enable_batching: Optional[bool] = None,
    ) -> ValidationSummary:
        """Execute a complete rule set.
        
        Args:
            rule_set: Rule set to execute
            context: Validation context
            parallel: Override parallel execution setting
            fail_fast: Stop execution on first critical failure
            
        Returns:
            ValidationSummary with complete execution results
        """
        start_time = datetime.now()
        logger.info(f"Executing rule set: {rule_set.name}")
        
        # Determine execution mode
        execute_parallel = parallel if parallel is not None else rule_set.parallel_execution
        batching_enabled = enable_batching if enable_batching is not None else self.enable_batching
        enabled_rules = rule_set.get_enabled_rules()

        # Build dependency graph
        self._build_dependency_graph(enabled_rules)

        # Execute rules
        if execute_parallel and len(enabled_rules) > 1:
            if batching_enabled and self.max_batch_size > 1:
                results = self._execute_rules_batched_parallel(
                    enabled_rules,
                    context,
                    rule_set.max_concurrent_rules,
                    fail_fast,
                )
            else:
                results = self._execute_rules_parallel(
                    enabled_rules,
                    context,
                    rule_set.max_concurrent_rules,
                    fail_fast,
                )
        else:
            results = self._execute_rules_sequential(enabled_rules, context, fail_fast)
        
        end_time = datetime.now()
        
        # Build summary
        summary = self._build_validation_summary(
            rule_set=rule_set,
            context=context,
            results=results,
            start_time=start_time,
            end_time=end_time
        )
        
        logger.info(f"Rule set '{rule_set.name}' completed: {summary.rules_passed}/{summary.total_rules} passed")
        
        return summary
    
    def _execute_sql_rule(
        self,
        rule: BusinessRule,
        context: ValidationContext,
        timeout: float
    ) -> RuleResult:
        """Execute a SQL-based business rule."""
        try:
            # Get database adapter
            adapter = self.connection_manager.get_adapter(context.database_name)
            if not adapter:
                raise ValidationError(f"Database adapter not found: {context.database_name}")
            
            # Execute the rule query with streaming support
            start_time = time.time()
            chunk_size = int(rule.parameters.get('chunk_size', 5000)) if rule.parameters else 5000
            result = adapter.execute_query(
                rule.sql_query,
                timeout=timeout,
                stream_results=True,
                chunk_size=chunk_size,
            )
            query_time = time.time() - start_time

            violations: List[RuleViolation] = []
            rows_evaluated = 0

            data_frame = getattr(result, "data", None)
            iterator = None
            close_fn = None

            if isinstance(data_frame, pd.DataFrame):
                iterator = [data_frame]
            elif data_frame is not None:
                iterator = [pd.DataFrame(data_frame)]
            elif hasattr(result, "iter_chunks"):
                potential_iterator = result.iter_chunks()
                # Some mocks return another Mock which is not iterable; guard against that
                try:
                    iter(potential_iterator)
                    iterator = potential_iterator
                    close_fn = getattr(iterator, "close", None)
                except TypeError:
                    iterator = []
            else:
                iterator = []

            try:
                for chunk in iterator:
                    if not isinstance(chunk, pd.DataFrame):
                        chunk = pd.DataFrame(chunk)
                    if chunk.empty:
                        continue
                    rows_evaluated += len(chunk)
                    violations.extend(self._process_sql_results(rule, chunk, context))
            finally:
                if callable(close_fn):
                    try:
                        close_fn()
                    except Exception:
                        logger.debug("Failed to close result iterator for rule %s", rule.name, exc_info=True)
            
            # Determine rule status
            passed = len(violations) == 0
            
            # Check against expected violation count
            if rule.expected_violation_count is not None:
                total_violations = sum(v.violation_count for v in violations)
                if total_violations != rule.expected_violation_count:
                    passed = False
                    violations.append(RuleViolation(
                        rule_name=rule.name,
                        violation_id=str(uuid.uuid4()),
                        severity=rule.severity,
                        message=f"Expected {rule.expected_violation_count} violations but found {total_violations}",
                        context={"expected": rule.expected_violation_count, "actual": total_violations}
                    ))
            
            # Check against max violation count
            if rule.max_violation_count is not None:
                total_violations = sum(v.violation_count for v in violations)
                if total_violations > rule.max_violation_count:
                    passed = False
                    violations.append(RuleViolation(
                        rule_name=rule.name,
                        violation_id=str(uuid.uuid4()),
                        severity=rule.severity,
                        message=f"Violations exceed maximum allowed ({total_violations} > {rule.max_violation_count})",
                        context={"max_allowed": rule.max_violation_count, "actual": total_violations}
                    ))
            
            return RuleResult(
                rule_name=rule.name,
                rule_type=rule.rule_type,
                status=RuleStatus.PASSED if passed else RuleStatus.FAILED,
                severity=rule.severity,
                scope=rule.scope,
                passed=passed,
                message=f"Rule executed successfully" if passed else f"Rule failed with {len(violations)} violations",
                violations=violations,
                rows_evaluated=rows_evaluated,
                context={
                    "query_execution_time_ms": query_time * 1000,
                    "result_rows": rows_evaluated
                }
            )
            
        except Exception as e:
            raise ValidationError(f"SQL rule execution failed: {str(e)}") from e
    
    def _execute_custom_rule(
        self,
        rule: BusinessRule,
        context: ValidationContext,
        timeout: float
    ) -> RuleResult:
        """Execute a custom function-based business rule."""
        try:
            # For custom rules, we might need to fetch data first
            data_df = pd.DataFrame()
            
            if context.query:
                adapter = self.connection_manager.get_adapter(context.database_name)
                if adapter:
                    result = adapter.execute_query(context.query, timeout=timeout)
                    data_df = result.data
            
            # Execute custom function with timeout
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(rule.custom_function, data_df, context)
                try:
                    result = future.result(timeout=timeout)
                    if isinstance(result, RuleResult):
                        return result
                    else:
                        # Custom function should return RuleResult, but handle boolean return
                        passed = bool(result) if result is not None else False
                        return RuleResult(
                            rule_name=rule.name,
                            rule_type=rule.rule_type,
                            status=RuleStatus.PASSED if passed else RuleStatus.FAILED,
                            severity=rule.severity,
                            scope=rule.scope,
                            passed=passed,
                            message="Custom rule executed" if passed else "Custom rule failed",
                            rows_evaluated=len(data_df)
                        )
                except FutureTimeoutError:
                    raise TimeoutError(f"Custom rule '{rule.name}' timed out")
                    
        except Exception as e:
            raise ValidationError(f"Custom rule execution failed: {str(e)}") from e
    
    def _process_sql_results(
        self,
        rule: BusinessRule,
        result_df: pd.DataFrame,
        context: ValidationContext
    ) -> List[RuleViolation]:
        """Process SQL query results to extract violations."""
        violations = []
        
        if len(result_df) == 0:
            # No results means no violations found
            return violations
        
        for _, row in result_df.iterrows():
            # Extract violation details from the result row
            violation_count = 1
            message = "Violation found"
            table_name = None
            column_name = None
            sample_values = []
            
            # Try to extract standard columns
            if "violation_count" in result_df.columns:
                violation_count = int(row.get("violation_count", 1))

            if "message" in result_df.columns:
                message = str(row.get("message", "Violation found"))
            
            if "table_name" in result_df.columns:
                table_name = row.get("table_name")
            
            if "column_name" in result_df.columns:
                column_name = row.get("column_name")
            
            # Extract sample values (any columns not matching standard names)
            standard_columns = {"violation_count", "message", "table_name", "column_name"}
            for col in result_df.columns:
                if col not in standard_columns:
                    sample_values.append(row[col])

            if violation_count <= 0 and not sample_values:
                # Treat rows with zero violations as passing results
                continue

            violation = RuleViolation(
                rule_name=rule.name,
                violation_id=str(uuid.uuid4()),
                severity=rule.severity,
                message=message,
                table_name=table_name,
                column_name=column_name,
                violation_count=violation_count,
                sample_values=sample_values,
                context=dict(row)  # Include all row data as context
            )
            
            violations.append(violation)
        
        return violations
    
    def _execute_rules_sequential(
        self,
        rules: List[BusinessRule],
        context: ValidationContext,
        fail_fast: bool
    ) -> List[RuleResult]:
        """Execute rules sequentially respecting dependencies."""
        results = []
        executed_rules = set()
        
        # Topological sort for dependency order
        ordered_rules = self._topological_sort(rules)
        
        for rule in ordered_rules:
            # Check if dependencies are satisfied
            if not self._dependencies_satisfied(rule, executed_rules):
                results.append(RuleResult(
                    rule_name=rule.name,
                    rule_type=rule.rule_type,
                    status=RuleStatus.SKIPPED,
                    severity=rule.severity,
                    scope=rule.scope,
                    passed=False,
                    message="Dependencies not satisfied"
                ))
                continue
            
            # Execute the rule
            result = self.execute_rule(rule, context)
            results.append(result)
            executed_rules.add(rule.name)
            
            # Check fail-fast condition
            if fail_fast and not result.passed and result.severity in [RuleSeverity.CRITICAL, RuleSeverity.ERROR]:
                logger.warning(f"Stopping rule execution due to fail-fast and critical/error failure in rule: {rule.name}")
                break
        
        return results
    
    def _execute_rules_parallel(
        self,
        rules: List[BusinessRule],
        context: ValidationContext,
        max_concurrent: int,
        fail_fast: bool
    ) -> List[RuleResult]:
        """Execute rules in parallel with dependency management."""
        results = []
        executed_rules = set()
        pending_rules = {rule.name: rule for rule in rules}
        
        with ThreadPoolExecutor(max_workers=min(max_concurrent, len(rules))) as executor:
            futures = {}
            
            while pending_rules or futures:
                # Submit rules whose dependencies are satisfied
                ready_rules = [
                    rule for rule in pending_rules.values()
                    if self._dependencies_satisfied(rule, executed_rules)
                ]
                
                for rule in ready_rules:
                    if len(futures) < max_concurrent:
                        future = executor.submit(self.execute_rule, rule, context)
                        futures[future] = rule
                        del pending_rules[rule.name]
                    else:
                        break
                
                # Process completed futures
                if futures:
                    completed = as_completed(futures, timeout=1.0)
                    try:
                        for future in completed:
                            rule = futures[future]
                            try:
                                result = future.result()
                                results.append(result)
                                executed_rules.add(rule.name)
                                
                                # Check fail-fast condition
                                if fail_fast and not result.passed and result.severity in [RuleSeverity.CRITICAL, RuleSeverity.ERROR]:
                                    logger.warning(f"Stopping parallel execution due to fail-fast and critical/error failure in rule: {rule.name}")
                                    # Cancel remaining futures
                                    for f in futures:
                                        if f != future:
                                            f.cancel()
                                    return results
                                    
                            except Exception as e:
                                logger.error(f"Rule '{rule.name}' failed with exception: {str(e)}")
                                error_result = RuleResult(
                                    rule_name=rule.name,
                                    rule_type=rule.rule_type,
                                    status=RuleStatus.ERROR,
                                    severity=rule.severity,
                                    scope=rule.scope,
                                    passed=False,
                                    message=f"Execution failed: {str(e)}"
                                )
                                results.append(error_result)
                                executed_rules.add(rule.name)
                            
                            del futures[future]
                    except Exception:
                        # Timeout or other exception - continue with next iteration
                        pass
        
        # Mark remaining rules as skipped
        for rule_name in pending_rules:
            rule = pending_rules[rule_name]
            results.append(RuleResult(
                rule_name=rule.name,
                rule_type=rule.rule_type,
                status=RuleStatus.SKIPPED,
                severity=rule.severity,
                scope=rule.scope,
                passed=False,
                message="Rule skipped due to unsatisfied dependencies"
            ))
        
        return results
    
    def _build_dependency_graph(self, rules: List[BusinessRule]) -> None:
        """Build dependency graph for rules."""
        self._dependency_graph.clear()
        
        rule_names = {rule.name for rule in rules}
        
        for rule in rules:
            self._dependency_graph[rule.name] = set()
            for dependency in rule.dependencies:
                if dependency in rule_names:
                    self._dependency_graph[rule.name].add(dependency)
                else:
                    logger.warning(f"Rule '{rule.name}' has undefined dependency: '{dependency}'")
    
    def _topological_sort(self, rules: List[BusinessRule]) -> List[BusinessRule]:
        """Perform topological sort of rules based on dependencies."""
        rule_dict = {rule.name: rule for rule in rules}
        visited = set()
        temp_visited = set()
        result = []
        
        def visit(rule_name: str):
            if rule_name in temp_visited:
                logger.warning(f"Circular dependency detected involving rule: {rule_name}")
                return
            if rule_name in visited:
                return
            
            temp_visited.add(rule_name)
            
            for dependency in self._dependency_graph.get(rule_name, set()):
                if dependency in rule_dict:
                    visit(dependency)
            
            temp_visited.remove(rule_name)
            visited.add(rule_name)
            
            if rule_name in rule_dict:
                result.append(rule_dict[rule_name])
        
        for rule in rules:
            if rule.name not in visited:
                visit(rule.name)
        
        return result
    
    def _dependencies_satisfied(self, rule: BusinessRule, executed_rules: Set[str]) -> bool:
        """Check if rule dependencies are satisfied."""
        return all(dep in executed_rules for dep in rule.dependencies)
    
    def _build_validation_summary(
        self,
        rule_set: RuleSet,
        context: ValidationContext,
        results: List[RuleResult],
        start_time: datetime,
        end_time: datetime
    ) -> ValidationSummary:
        """Build validation summary from results."""
        # Count results by status
        status_counts = {status: 0 for status in RuleStatus}
        for result in results:
            status_counts[result.status] += 1
        
        # Count violations by severity
        violation_counts = {severity: 0 for severity in RuleSeverity}
        total_violations = 0
        
        for result in results:
            for violation in result.violations:
                violation_counts[violation.severity] += violation.violation_count
                total_violations += violation.violation_count
        
        return ValidationSummary(
            validation_name=f"{rule_set.name}_validation",
            rule_set_name=rule_set.name,
            validation_context=context,
            start_time=start_time,
            end_time=end_time,
            total_rules=len(rule_set.rules),
            rules_executed=status_counts[RuleStatus.PASSED] + status_counts[RuleStatus.FAILED] + status_counts[RuleStatus.ERROR],
            rules_passed=status_counts[RuleStatus.PASSED],
            rules_failed=status_counts[RuleStatus.FAILED],
            rules_error=status_counts[RuleStatus.ERROR],
            rules_skipped=status_counts[RuleStatus.SKIPPED],
            total_violations=total_violations,
            critical_violations=violation_counts[RuleSeverity.CRITICAL],
            error_violations=violation_counts[RuleSeverity.ERROR],
            warning_violations=violation_counts[RuleSeverity.WARNING],
            info_violations=violation_counts[RuleSeverity.INFO],
            results=results
        )
    
    def validate_table(
        self,
        database_name: str,
        table_name: str,
        rules: List[BusinessRule],
        schema_name: Optional[str] = None
    ) -> ValidationSummary:
        """Convenience method to validate a specific table."""
        context = ValidationContext(
            database_name=database_name,
            schema_name=schema_name,
            table_name=table_name
        )

        # Create temporary rule set
        rule_set = RuleSet(
            name=f"{table_name}_validation",
            description=f"Validation rules for table {table_name}",
            rules=rules
        )

        return self.execute_rule_set(rule_set, context)

    def _create_rule_batches(self, rules: List[BusinessRule]) -> List[RuleBatch]:
        """Create optimized batches of rules for parallel execution."""
        if not self.enable_batching:
            return [RuleBatch([rule]) for rule in rules]

        batches = []
        remaining_rules = list(rules)

        while remaining_rules:
            # Start a new batch with the first remaining rule
            current_batch = RuleBatch([remaining_rules.pop(0)])

            # Try to add compatible rules to the batch
            i = 0
            while i < len(remaining_rules) and current_batch.size < self.max_batch_size:
                rule = remaining_rules[i]

                # Check if rule can be added to current batch
                if self._can_batch_rule(current_batch.rules[0], rule):
                    current_batch.rules.append(remaining_rules.pop(i))
                    current_batch.size += 1
                else:
                    i += 1

            batches.append(current_batch)

        logger.info(f"Created {len(batches)} rule batches from {len(rules)} rules")
        return batches

    def _can_batch_rule(self, base_rule: BusinessRule, candidate_rule: BusinessRule) -> bool:
        """Check if two rules can be batched together."""
        # Rules can be batched if they:
        # 1. Have no dependencies on each other
        # 2. Target the same database/schema
        # 3. Are both SQL-based or both custom function based
        # 4. Have similar execution characteristics

        if candidate_rule.name in base_rule.dependencies or base_rule.name in candidate_rule.dependencies:
            return False

        if base_rule.rule_type != candidate_rule.rule_type:
            return False

        # Check if they're both marked as batch-compatible
        if not getattr(base_rule, 'batch_compatible', True) or not getattr(candidate_rule, 'batch_compatible', True):
            return False

        return True

    def _execute_rules_batched_parallel(
        self,
        rules: List[BusinessRule],
        context: ValidationContext,
        max_concurrent: int,
        fail_fast: bool
    ) -> List[RuleResult]:
        """Execute rules in optimized batches with parallel processing."""
        all_results = []
        executed_rules = set()

        # Create rule batches
        batches = self._create_rule_batches(rules)
        pending_batches = {batch.batch_id: batch for batch in batches}

        with ThreadPoolExecutor(max_workers=min(max_concurrent, len(batches))) as executor:
            futures = {}

            while pending_batches or futures:
                # Submit batches whose dependencies are satisfied
                ready_batches = [
                    batch for batch in pending_batches.values()
                    if all(self._dependencies_satisfied(rule, executed_rules) for rule in batch.rules)
                ]

                for batch in ready_batches:
                    if len(futures) < max_concurrent:
                        future = executor.submit(self._execute_rule_batch, batch, context)
                        futures[future] = batch
                        del pending_batches[batch.batch_id]
                    else:
                        break

                # Process completed futures
                if futures:
                    completed = as_completed(futures, timeout=1.0)
                    try:
                        for future in completed:
                            batch = futures[future]
                            try:
                                batch_results = future.result()
                                all_results.extend(batch_results)

                                # Mark all rules in batch as executed
                                for rule in batch.rules:
                                    executed_rules.add(rule.name)

                                # Check fail-fast condition
                                if fail_fast:
                                    for result in batch_results:
                                        if not result.passed and result.severity in [RuleSeverity.CRITICAL, RuleSeverity.ERROR]:
                                            logger.warning(f"Stopping batch execution due to fail-fast and critical/error failure")
                                            # Cancel remaining futures
                                            for f in futures:
                                                if f != future:
                                                    f.cancel()
                                            return all_results

                            except Exception as e:
                                logger.error(f"Batch execution failed: {str(e)}")
                                # Create error results for all rules in the batch
                                for rule in batch.rules:
                                    error_result = RuleResult(
                                        rule_name=rule.name,
                                        rule_type=rule.rule_type,
                                        status=RuleStatus.ERROR,
                                        severity=rule.severity,
                                        scope=rule.scope,
                                        passed=False,
                                        message=f"Batch execution failed: {str(e)}"
                                    )
                                    all_results.append(error_result)
                                    executed_rules.add(rule.name)

                            del futures[future]
                    except Exception:
                        # Timeout or other exception - continue with next iteration
                        pass

        # Mark remaining rules as skipped
        for batch in pending_batches.values():
            for rule in batch.rules:
                all_results.append(RuleResult(
                    rule_name=rule.name,
                    rule_type=rule.rule_type,
                    status=RuleStatus.SKIPPED,
                    severity=rule.severity,
                    scope=rule.scope,
                    passed=False,
                    message="Rule skipped due to unsatisfied dependencies"
                ))

        return all_results

    def _execute_rule_batch(self, batch: RuleBatch, context: ValidationContext) -> List[RuleResult]:
        """Execute a batch of rules."""
        results = []

        logger.debug(f"Executing batch {batch.batch_id} with {batch.size} rules")

        if self.metrics:
            self.metrics.increment_counter('batches_executed')
            self.metrics.record_value('batch_size', batch.size)

        # For now, execute rules in batch sequentially
        # TODO: Implement true batch SQL execution for compatible rules
        for rule in batch.rules:
            try:
                result = self.execute_rule(rule, context)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to execute rule {rule.name} in batch: {e}")
                error_result = RuleResult(
                    rule_name=rule.name,
                    rule_type=rule.rule_type,
                    status=RuleStatus.ERROR,
                    severity=rule.severity,
                    scope=rule.scope,
                    passed=False,
                    message=f"Rule execution failed: {str(e)}"
                )
                results.append(error_result)

        return results

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = {}

        if self.metrics:
            stats['execution_metrics'] = self.metrics.get_all_stats()

        if self.cache_manager:
            stats['cache_stats'] = self.cache_manager.get_cache_stats()

        stats['engine_config'] = {
            'max_workers': self.max_workers,
            'enable_batching': self.enable_batching,
            'max_batch_size': self.max_batch_size,
            'cache_enabled': self.cache_manager is not None,
            'metrics_enabled': self.metrics is not None
        }

        return stats

    def invalidate_cache(self, pattern: str = None):
        """Invalidate cached results."""
        if self.cache_manager:
            self.cache_manager.invalidate(pattern)
            logger.info(f"Cache invalidated{f' (pattern: {pattern})' if pattern else ''}")

    def reset_metrics(self):
        """Reset performance metrics."""
        if self.metrics:
            self.metrics = PerformanceMetrics()
            logger.info("Performance metrics reset")

    @property
    def _cache_enabled(self) -> bool:
        """Compatibility shim for tests toggling cache via private attribute."""
        return self.cache_manager is not None

    @_cache_enabled.setter
    def _cache_enabled(self, value: bool) -> None:
        if value:
            if self.cache_manager is None:
                self.cache_manager = CacheManager(self._cache_config)
        else:
            if self.cache_manager is not None:
                self.cache_manager = None
    
    def validate_query(
        self,
        database_name: str,
        query: str,
        rules: List[BusinessRule],
        query_name: str = "custom_query"
    ) -> ValidationSummary:
        """Convenience method to validate query results."""
        context = ValidationContext(
            database_name=database_name,
            query=query
        )
        
        # Create temporary rule set
        rule_set = RuleSet(
            name=f"{query_name}_validation",
            description=f"Validation rules for query: {query_name}",
            rules=rules
        )
        
        return self.execute_rule_set(rule_set, context)

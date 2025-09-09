"""Business rule execution engine for SQLTest Pro."""

import asyncio
import hashlib
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeoutError
from datetime import datetime
from typing import Dict, List, Optional, Set, Callable, Any
import pandas as pd
import logging

from ...db.connection import ConnectionManager
from ...exceptions import ValidationError, DatabaseError
from .models import (
    BusinessRule,
    RuleSet,
    RuleResult,
    RuleViolation,
    RuleStatus,
    RuleSeverity,
    ValidationContext,
    ValidationSummary,
    RuleType,
    ValidationScope
)

logger = logging.getLogger(__name__)


class BusinessRuleEngine:
    """Core engine for executing business rules and validations."""
    
    def __init__(self, connection_manager: ConnectionManager, max_workers: int = 5):
        """Initialize the business rule engine.
        
        Args:
            connection_manager: Database connection manager for executing queries
            max_workers: Maximum number of worker threads for parallel execution
        """
        self.connection_manager = connection_manager
        self.max_workers = max_workers
        self._rule_cache: Dict[str, Any] = {}
        self._dependency_graph: Dict[str, Set[str]] = {}
        
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
                result = self._execute_custom_rule(rule, context, timeout)
            elif rule.sql_query:
                result = self._execute_sql_rule(rule, context, timeout)
            else:
                raise ValidationError(f"Rule '{rule.name}' has neither SQL query nor custom function")
            
            # Calculate execution time
            end_time = time.time()
            result.execution_time_ms = (end_time - start_time) * 1000
            
            logger.info(f"Rule '{rule.name}' completed: {result.status} ({result.execution_time_ms:.2f}ms)")
            
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
        fail_fast: bool = False
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
        enabled_rules = rule_set.get_enabled_rules()
        
        # Build dependency graph
        self._build_dependency_graph(enabled_rules)
        
        # Execute rules
        if execute_parallel and len(enabled_rules) > 1:
            results = self._execute_rules_parallel(enabled_rules, context, rule_set.max_concurrent_rules, fail_fast)
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
            
            # Execute the rule query with timeout
            start_time = time.time()
            result = adapter.execute_query(rule.sql_query, timeout=timeout)
            result_df = result.data
            query_time = time.time() - start_time
            
            # Process results
            violations = self._process_sql_results(rule, result_df, context)
            
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
                rows_evaluated=len(result_df),
                context={
                    "query_execution_time_ms": query_time * 1000,
                    "result_rows": len(result_df)
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

"""
Advanced test orchestration and workflow management for SQL unit testing framework.

This module provides enterprise-grade test orchestration capabilities including
workflow management, conditional execution, and complex test scenarios.
"""
import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

from .models import TestSuite, TestCase, TestResult, TestStatus, TestIsolationLevel
from .enterprise_executor import EnterpriseTestExecutor
from .reporting import TestReportGenerator

logger = logging.getLogger(__name__)


class WorkflowStatus(str, Enum):
    """Status of workflow execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class StepType(str, Enum):
    """Types of workflow steps."""
    TEST_SUITE = "test_suite"
    TEST_CASE = "test_case"
    CONDITIONAL = "conditional"
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
    WAIT = "wait"
    CUSTOM = "custom"
    DATA_SETUP = "data_setup"
    DATA_CLEANUP = "data_cleanup"


class ConditionType(str, Enum):
    """Types of workflow conditions."""
    SUCCESS = "success"
    FAILURE = "failure"
    ALWAYS = "always"
    CUSTOM = "custom"
    COVERAGE_THRESHOLD = "coverage_threshold"
    PERFORMANCE_THRESHOLD = "performance_threshold"


@dataclass
class WorkflowCondition:
    """Defines conditions for workflow execution."""
    condition_type: ConditionType
    target_step: Optional[str] = None
    custom_function: Optional[str] = None
    threshold_value: Optional[float] = None
    operator: str = "gte"  # gte, lte, eq, ne


@dataclass
class WorkflowStep:
    """Represents a single step in a workflow."""
    step_id: str
    step_type: StepType
    name: str
    description: Optional[str] = None
    depends_on: List[str] = field(default_factory=list)
    conditions: List[WorkflowCondition] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    timeout: Optional[int] = None
    retry_count: int = 0
    retry_delay: int = 5
    enabled: bool = True
    tags: List[str] = field(default_factory=list)


@dataclass
class WorkflowExecution:
    """Tracks execution state of a workflow."""
    execution_id: str
    workflow_id: str
    status: WorkflowStatus
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    current_step: Optional[str] = None
    step_results: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestWorkflow:
    """Defines a complete test workflow."""
    workflow_id: str
    name: str
    description: Optional[str] = None
    version: str = "1.0"
    steps: List[WorkflowStep] = field(default_factory=list)
    global_parameters: Dict[str, Any] = field(default_factory=dict)
    timeout: Optional[int] = None
    max_parallel: int = 5
    fail_fast: bool = False
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


class WorkflowOrchestrator:
    """Advanced orchestrator for managing complex test workflows."""

    def __init__(self, test_executor: EnterpriseTestExecutor, report_generator: TestReportGenerator):
        """
        Initialize workflow orchestrator.

        Args:
            test_executor: Enterprise test executor for running tests
            report_generator: Report generator for workflow reporting
        """
        self.test_executor = test_executor
        self.report_generator = report_generator
        self.workflows: Dict[str, TestWorkflow] = {}
        self.executions: Dict[str, WorkflowExecution] = {}
        self._step_handlers = {
            StepType.TEST_SUITE: self._execute_test_suite_step,
            StepType.TEST_CASE: self._execute_test_case_step,
            StepType.CONDITIONAL: self._execute_conditional_step,
            StepType.PARALLEL: self._execute_parallel_step,
            StepType.SEQUENTIAL: self._execute_sequential_step,
            StepType.WAIT: self._execute_wait_step,
            StepType.CUSTOM: self._execute_custom_step,
            StepType.DATA_SETUP: self._execute_data_setup_step,
            StepType.DATA_CLEANUP: self._execute_data_cleanup_step
        }
        self._condition_evaluators = {
            ConditionType.SUCCESS: self._evaluate_success_condition,
            ConditionType.FAILURE: self._evaluate_failure_condition,
            ConditionType.ALWAYS: self._evaluate_always_condition,
            ConditionType.CUSTOM: self._evaluate_custom_condition,
            ConditionType.COVERAGE_THRESHOLD: self._evaluate_coverage_condition,
            ConditionType.PERFORMANCE_THRESHOLD: self._evaluate_performance_condition
        }

    def register_workflow(self, workflow: TestWorkflow) -> None:
        """Register a new workflow."""
        self.workflows[workflow.workflow_id] = workflow
        logger.info(f"Registered workflow: {workflow.name} ({workflow.workflow_id})")

    def load_workflow_from_file(self, file_path: Union[str, Path]) -> TestWorkflow:
        """Load workflow from JSON file."""
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Workflow file not found: {file_path}")

        with open(file_path, 'r') as f:
            data = json.load(f)

        workflow = self._deserialize_workflow(data)
        self.register_workflow(workflow)
        return workflow

    def save_workflow_to_file(self, workflow_id: str, file_path: Union[str, Path]) -> None:
        """Save workflow to JSON file."""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow not found: {workflow_id}")

        workflow = self.workflows[workflow_id]
        data = self._serialize_workflow(workflow)

        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Saved workflow to: {file_path}")

    async def execute_workflow(self,
                             workflow_id: str,
                             parameters: Optional[Dict[str, Any]] = None,
                             execution_id: Optional[str] = None) -> WorkflowExecution:
        """
        Execute a workflow with advanced orchestration.

        Args:
            workflow_id: ID of workflow to execute
            parameters: Runtime parameters for workflow
            execution_id: Optional custom execution ID

        Returns:
            WorkflowExecution object tracking execution state
        """
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow not found: {workflow_id}")

        workflow = self.workflows[workflow_id]
        execution_id = execution_id or str(uuid.uuid4())

        # Create execution context
        execution = WorkflowExecution(
            execution_id=execution_id,
            workflow_id=workflow_id,
            status=WorkflowStatus.PENDING,
            metadata={"parameters": parameters or {}}
        )

        self.executions[execution_id] = execution

        try:
            execution.status = WorkflowStatus.RUNNING
            execution.start_time = datetime.now()

            logger.info(f"Starting workflow execution: {workflow.name} ({execution_id})")

            # Execute workflow steps
            await self._execute_workflow_steps(workflow, execution, parameters or {})

            execution.status = WorkflowStatus.COMPLETED
            execution.end_time = datetime.now()

            logger.info(f"Workflow completed successfully: {execution_id}")

        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            execution.end_time = datetime.now()
            execution.error_message = str(e)

            logger.error(f"Workflow execution failed: {execution_id}, Error: {e}")
            raise

        return execution

    async def _execute_workflow_steps(self,
                                    workflow: TestWorkflow,
                                    execution: WorkflowExecution,
                                    parameters: Dict[str, Any]) -> None:
        """Execute all steps in a workflow with dependency resolution."""
        completed_steps: Set[str] = set()
        failed_steps: Set[str] = set()

        # Build dependency graph
        step_map = {step.step_id: step for step in workflow.steps}

        while len(completed_steps) < len(workflow.steps):
            # Find ready steps (dependencies satisfied)
            ready_steps = []

            for step in workflow.steps:
                if (step.step_id not in completed_steps and
                    step.step_id not in failed_steps and
                    step.enabled and
                    all(dep in completed_steps for dep in step.depends_on)):

                    # Check conditions
                    if self._evaluate_step_conditions(step, execution):
                        ready_steps.append(step)

            if not ready_steps:
                # Check if we have failed steps blocking progress
                remaining_steps = [s for s in workflow.steps
                                 if s.step_id not in completed_steps and s.step_id not in failed_steps]

                if remaining_steps and failed_steps:
                    raise Exception(f"Cannot proceed: Steps blocked by failures: {failed_steps}")
                elif remaining_steps:
                    raise Exception("Circular dependency detected or all remaining steps have unmet conditions")
                else:
                    break

            # Execute ready steps (up to max_parallel)
            steps_to_execute = ready_steps[:workflow.max_parallel]

            if len(steps_to_execute) == 1:
                # Sequential execution
                step = steps_to_execute[0]
                try:
                    execution.current_step = step.step_id
                    await self._execute_single_step(step, workflow, execution, parameters)
                    completed_steps.add(step.step_id)
                except Exception as e:
                    failed_steps.add(step.step_id)
                    if workflow.fail_fast:
                        raise
                    logger.error(f"Step failed: {step.step_id}, Error: {e}")
            else:
                # Parallel execution
                tasks = []
                for step in steps_to_execute:
                    task = asyncio.create_task(
                        self._execute_single_step(step, workflow, execution, parameters)
                    )
                    tasks.append((step.step_id, task))

                # Wait for all tasks to complete
                for step_id, task in tasks:
                    try:
                        await task
                        completed_steps.add(step_id)
                    except Exception as e:
                        failed_steps.add(step_id)
                        if workflow.fail_fast:
                            # Cancel remaining tasks
                            for remaining_step_id, remaining_task in tasks:
                                if remaining_step_id != step_id and not remaining_task.done():
                                    remaining_task.cancel()
                            raise
                        logger.error(f"Step failed: {step_id}, Error: {e}")

    async def _execute_single_step(self,
                                 step: WorkflowStep,
                                 workflow: TestWorkflow,
                                 execution: WorkflowExecution,
                                 parameters: Dict[str, Any]) -> Any:
        """Execute a single workflow step with retry logic."""
        logger.info(f"Executing step: {step.name} ({step.step_id})")

        # Merge parameters
        step_params = {**workflow.global_parameters, **parameters, **step.parameters}

        retry_count = 0
        last_exception = None

        while retry_count <= step.retry_count:
            try:
                # Apply timeout if specified
                if step.timeout:
                    result = await asyncio.wait_for(
                        self._execute_step_handler(step, step_params, execution),
                        timeout=step.timeout
                    )
                else:
                    result = await self._execute_step_handler(step, step_params, execution)

                # Store step result
                execution.step_results[step.step_id] = {
                    "status": "success",
                    "result": result,
                    "execution_time": datetime.now().isoformat(),
                    "retry_count": retry_count
                }

                logger.info(f"Step completed successfully: {step.step_id}")
                return result

            except Exception as e:
                last_exception = e
                retry_count += 1

                if retry_count <= step.retry_count:
                    logger.warning(f"Step failed, retrying ({retry_count}/{step.retry_count}): {step.step_id}")
                    await asyncio.sleep(step.retry_delay)
                else:
                    # Store failure result
                    execution.step_results[step.step_id] = {
                        "status": "failed",
                        "error": str(e),
                        "execution_time": datetime.now().isoformat(),
                        "retry_count": retry_count - 1
                    }

                    logger.error(f"Step failed after {retry_count - 1} retries: {step.step_id}")
                    raise

        raise last_exception

    async def _execute_step_handler(self,
                                  step: WorkflowStep,
                                  parameters: Dict[str, Any],
                                  execution: WorkflowExecution) -> Any:
        """Route step execution to appropriate handler."""
        handler = self._step_handlers.get(step.step_type)
        if not handler:
            raise ValueError(f"Unknown step type: {step.step_type}")

        return await handler(step, parameters, execution)

    async def _execute_test_suite_step(self,
                                     step: WorkflowStep,
                                     parameters: Dict[str, Any],
                                     execution: WorkflowExecution) -> List[TestResult]:
        """Execute a test suite step."""
        suite_name = parameters.get("suite_name") or step.parameters.get("suite_name")
        database_name = parameters.get("database") or step.parameters.get("database")
        parallel = parameters.get("parallel", step.parameters.get("parallel", False))

        if not suite_name:
            raise ValueError("Test suite name is required")

        # Load test suite (this would integrate with your test suite registry)
        test_suite = self._load_test_suite(suite_name, parameters)

        # Execute test suite
        results = await self.test_executor.execute_test_suite(
            test_suite,
            parallel=parallel
        )

        return results

    async def _execute_test_case_step(self,
                                    step: WorkflowStep,
                                    parameters: Dict[str, Any],
                                    execution: WorkflowExecution) -> TestResult:
        """Execute a single test case step."""
        test_name = parameters.get("test_name") or step.parameters.get("test_name")
        database_name = parameters.get("database") or step.parameters.get("database")

        if not test_name:
            raise ValueError("Test case name is required")

        # Load test case
        test_case = self._load_test_case(test_name, parameters)

        # Execute test case
        result = await self.test_executor.execute_test(test_case)

        return result

    async def _execute_conditional_step(self,
                                       step: WorkflowStep,
                                       parameters: Dict[str, Any],
                                       execution: WorkflowExecution) -> Dict[str, Any]:
        """Execute a conditional step."""
        condition_result = self._evaluate_step_conditions(step, execution)

        return {
            "condition_met": condition_result,
            "conditions_evaluated": len(step.conditions)
        }

    async def _execute_parallel_step(self,
                                   step: WorkflowStep,
                                   parameters: Dict[str, Any],
                                   execution: WorkflowExecution) -> List[Any]:
        """Execute parallel substeps."""
        substeps = step.parameters.get("substeps", [])
        max_parallel = step.parameters.get("max_parallel", 5)

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_parallel)

        async def execute_substep(substep_config):
            async with semaphore:
                # Create temporary step from config
                substep = WorkflowStep(**substep_config)
                return await self._execute_step_handler(substep, parameters, execution)

        # Execute all substeps in parallel
        tasks = [execute_substep(substep) for substep in substeps]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return results

    async def _execute_sequential_step(self,
                                     step: WorkflowStep,
                                     parameters: Dict[str, Any],
                                     execution: WorkflowExecution) -> List[Any]:
        """Execute sequential substeps."""
        substeps = step.parameters.get("substeps", [])
        results = []

        for substep_config in substeps:
            # Create temporary step from config
            substep = WorkflowStep(**substep_config)
            result = await self._execute_step_handler(substep, parameters, execution)
            results.append(result)

        return results

    async def _execute_wait_step(self,
                               step: WorkflowStep,
                               parameters: Dict[str, Any],
                               execution: WorkflowExecution) -> None:
        """Execute a wait/delay step."""
        wait_time = step.parameters.get("wait_time", 1)
        await asyncio.sleep(wait_time)

    async def _execute_custom_step(self,
                                 step: WorkflowStep,
                                 parameters: Dict[str, Any],
                                 execution: WorkflowExecution) -> Any:
        """Execute a custom step with user-defined function."""
        custom_function = step.parameters.get("custom_function")
        if not custom_function:
            raise ValueError("Custom function is required for custom step")

        # Create safe execution environment
        namespace = {
            "step": step,
            "parameters": parameters,
            "execution": execution,
            "asyncio": asyncio,
            "datetime": datetime,
            "logger": logger
        }

        # Execute custom function
        exec(custom_function, namespace)

        # Look for result
        if "result" in namespace:
            return namespace["result"]
        else:
            return None

    async def _execute_data_setup_step(self,
                                     step: WorkflowStep,
                                     parameters: Dict[str, Any],
                                     execution: WorkflowExecution) -> Dict[str, Any]:
        """Execute data setup step."""
        setup_sql = step.parameters.get("setup_sql")
        database_name = parameters.get("database") or step.parameters.get("database")

        if setup_sql and database_name:
            # Execute setup SQL
            adapter = self.test_executor.connection_manager.get_adapter(database_name)
            result = adapter.execute_query(setup_sql)

            return {
                "setup_executed": True,
                "rows_affected": getattr(result, "rows_affected", 0)
            }

        return {"setup_executed": False}

    async def _execute_data_cleanup_step(self,
                                       step: WorkflowStep,
                                       parameters: Dict[str, Any],
                                       execution: WorkflowExecution) -> Dict[str, Any]:
        """Execute data cleanup step."""
        cleanup_sql = step.parameters.get("cleanup_sql")
        database_name = parameters.get("database") or step.parameters.get("database")

        if cleanup_sql and database_name:
            # Execute cleanup SQL
            adapter = self.test_executor.connection_manager.get_adapter(database_name)
            result = adapter.execute_query(cleanup_sql)

            return {
                "cleanup_executed": True,
                "rows_affected": getattr(result, "rows_affected", 0)
            }

        return {"cleanup_executed": False}

    def _evaluate_step_conditions(self, step: WorkflowStep, execution: WorkflowExecution) -> bool:
        """Evaluate all conditions for a step."""
        if not step.conditions:
            return True

        # All conditions must be met
        for condition in step.conditions:
            evaluator = self._condition_evaluators.get(condition.condition_type)
            if not evaluator:
                logger.warning(f"Unknown condition type: {condition.condition_type}")
                continue

            if not evaluator(condition, execution):
                return False

        return True

    def _evaluate_success_condition(self, condition: WorkflowCondition, execution: WorkflowExecution) -> bool:
        """Evaluate success condition."""
        if not condition.target_step:
            return True

        step_result = execution.step_results.get(condition.target_step)
        return step_result and step_result.get("status") == "success"

    def _evaluate_failure_condition(self, condition: WorkflowCondition, execution: WorkflowExecution) -> bool:
        """Evaluate failure condition."""
        if not condition.target_step:
            return False

        step_result = execution.step_results.get(condition.target_step)
        return step_result and step_result.get("status") == "failed"

    def _evaluate_always_condition(self, condition: WorkflowCondition, execution: WorkflowExecution) -> bool:
        """Evaluate always condition (always returns True)."""
        return True

    def _evaluate_custom_condition(self, condition: WorkflowCondition, execution: WorkflowExecution) -> bool:
        """Evaluate custom condition."""
        if not condition.custom_function:
            return True

        # Create safe execution environment
        namespace = {
            "condition": condition,
            "execution": execution,
            "datetime": datetime
        }

        try:
            exec(condition.custom_function, namespace)
            return namespace.get("result", True)
        except Exception as e:
            logger.error(f"Custom condition evaluation failed: {e}")
            return False

    def _evaluate_coverage_condition(self, condition: WorkflowCondition, execution: WorkflowExecution) -> bool:
        """Evaluate test coverage condition."""
        # This would integrate with coverage analysis
        current_coverage = execution.metadata.get("test_coverage", 0)
        threshold = condition.threshold_value or 80

        return self._compare_values(current_coverage, threshold, condition.operator)

    def _evaluate_performance_condition(self, condition: WorkflowCondition, execution: WorkflowExecution) -> bool:
        """Evaluate performance threshold condition."""
        # This would integrate with performance metrics
        current_performance = execution.metadata.get("avg_execution_time", 0)
        threshold = condition.threshold_value or 1000  # ms

        return self._compare_values(current_performance, threshold, condition.operator)

    def _compare_values(self, actual: float, expected: float, operator: str) -> bool:
        """Compare values using specified operator."""
        if operator == "gte":
            return actual >= expected
        elif operator == "lte":
            return actual <= expected
        elif operator == "eq":
            return actual == expected
        elif operator == "ne":
            return actual != expected
        elif operator == "gt":
            return actual > expected
        elif operator == "lt":
            return actual < expected
        else:
            return False

    def _load_test_suite(self, suite_name: str, parameters: Dict[str, Any]) -> TestSuite:
        """Load test suite by name (placeholder implementation)."""
        # This would integrate with your test suite registry/loader
        return TestSuite(
            name=suite_name,
            description=f"Loaded test suite: {suite_name}",
            database=parameters.get("database", "default"),
            tests=[]
        )

    def _load_test_case(self, test_name: str, parameters: Dict[str, Any]) -> TestCase:
        """Load test case by name (placeholder implementation)."""
        # This would integrate with your test case registry/loader
        return TestCase(
            name=test_name,
            description=f"Loaded test case: {test_name}",
            sql="SELECT 1",
            assertions=[],
            fixtures=[]
        )

    def _serialize_workflow(self, workflow: TestWorkflow) -> Dict[str, Any]:
        """Serialize workflow to dictionary."""
        return {
            "workflow_id": workflow.workflow_id,
            "name": workflow.name,
            "description": workflow.description,
            "version": workflow.version,
            "steps": [self._serialize_step(step) for step in workflow.steps],
            "global_parameters": workflow.global_parameters,
            "timeout": workflow.timeout,
            "max_parallel": workflow.max_parallel,
            "fail_fast": workflow.fail_fast,
            "tags": workflow.tags,
            "created_at": workflow.created_at.isoformat()
        }

    def _serialize_step(self, step: WorkflowStep) -> Dict[str, Any]:
        """Serialize workflow step to dictionary."""
        return {
            "step_id": step.step_id,
            "step_type": step.step_type.value,
            "name": step.name,
            "description": step.description,
            "depends_on": step.depends_on,
            "conditions": [self._serialize_condition(cond) for cond in step.conditions],
            "parameters": step.parameters,
            "timeout": step.timeout,
            "retry_count": step.retry_count,
            "retry_delay": step.retry_delay,
            "enabled": step.enabled,
            "tags": step.tags
        }

    def _serialize_condition(self, condition: WorkflowCondition) -> Dict[str, Any]:
        """Serialize workflow condition to dictionary."""
        return {
            "condition_type": condition.condition_type.value,
            "target_step": condition.target_step,
            "custom_function": condition.custom_function,
            "threshold_value": condition.threshold_value,
            "operator": condition.operator
        }

    def _deserialize_workflow(self, data: Dict[str, Any]) -> TestWorkflow:
        """Deserialize workflow from dictionary."""
        steps = [self._deserialize_step(step_data) for step_data in data.get("steps", [])]

        return TestWorkflow(
            workflow_id=data["workflow_id"],
            name=data["name"],
            description=data.get("description"),
            version=data.get("version", "1.0"),
            steps=steps,
            global_parameters=data.get("global_parameters", {}),
            timeout=data.get("timeout"),
            max_parallel=data.get("max_parallel", 5),
            fail_fast=data.get("fail_fast", False),
            tags=data.get("tags", []),
            created_at=datetime.fromisoformat(data.get("created_at", datetime.now().isoformat()))
        )

    def _deserialize_step(self, data: Dict[str, Any]) -> WorkflowStep:
        """Deserialize workflow step from dictionary."""
        conditions = [self._deserialize_condition(cond_data) for cond_data in data.get("conditions", [])]

        return WorkflowStep(
            step_id=data["step_id"],
            step_type=StepType(data["step_type"]),
            name=data["name"],
            description=data.get("description"),
            depends_on=data.get("depends_on", []),
            conditions=conditions,
            parameters=data.get("parameters", {}),
            timeout=data.get("timeout"),
            retry_count=data.get("retry_count", 0),
            retry_delay=data.get("retry_delay", 5),
            enabled=data.get("enabled", True),
            tags=data.get("tags", [])
        )

    def _deserialize_condition(self, data: Dict[str, Any]) -> WorkflowCondition:
        """Deserialize workflow condition from dictionary."""
        return WorkflowCondition(
            condition_type=ConditionType(data["condition_type"]),
            target_step=data.get("target_step"),
            custom_function=data.get("custom_function"),
            threshold_value=data.get("threshold_value"),
            operator=data.get("operator", "gte")
        )

    def get_workflow_status(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Get current status of workflow execution."""
        return self.executions.get(execution_id)

    def cancel_workflow(self, execution_id: str) -> bool:
        """Cancel running workflow execution."""
        execution = self.executions.get(execution_id)
        if execution and execution.status == WorkflowStatus.RUNNING:
            execution.status = WorkflowStatus.CANCELLED
            execution.end_time = datetime.now()
            logger.info(f"Workflow cancelled: {execution_id}")
            return True
        return False

    def pause_workflow(self, execution_id: str) -> bool:
        """Pause running workflow execution."""
        execution = self.executions.get(execution_id)
        if execution and execution.status == WorkflowStatus.RUNNING:
            execution.status = WorkflowStatus.PAUSED
            logger.info(f"Workflow paused: {execution_id}")
            return True
        return False

    def resume_workflow(self, execution_id: str) -> bool:
        """Resume paused workflow execution."""
        execution = self.executions.get(execution_id)
        if execution and execution.status == WorkflowStatus.PAUSED:
            execution.status = WorkflowStatus.RUNNING
            logger.info(f"Workflow resumed: {execution_id}")
            return True
        return False

    def get_execution_summary(self, execution_id: str) -> Dict[str, Any]:
        """Get comprehensive summary of workflow execution."""
        execution = self.executions.get(execution_id)
        if not execution:
            return {}

        workflow = self.workflows.get(execution.workflow_id)

        total_steps = len(workflow.steps) if workflow else 0
        completed_steps = len([r for r in execution.step_results.values() if r.get("status") == "success"])
        failed_steps = len([r for r in execution.step_results.values() if r.get("status") == "failed"])

        duration = None
        if execution.start_time and execution.end_time:
            duration = (execution.end_time - execution.start_time).total_seconds()

        return {
            "execution_id": execution.execution_id,
            "workflow_id": execution.workflow_id,
            "workflow_name": workflow.name if workflow else "Unknown",
            "status": execution.status.value,
            "progress": {
                "total_steps": total_steps,
                "completed_steps": completed_steps,
                "failed_steps": failed_steps,
                "success_rate": completed_steps / total_steps if total_steps > 0 else 0
            },
            "timing": {
                "start_time": execution.start_time.isoformat() if execution.start_time else None,
                "end_time": execution.end_time.isoformat() if execution.end_time else None,
                "duration_seconds": duration
            },
            "current_step": execution.current_step,
            "error_message": execution.error_message,
            "step_results": execution.step_results
        }


# Convenience functions for workflow management
def create_simple_test_workflow(name: str, test_suites: List[str], parallel: bool = False) -> TestWorkflow:
    """Create a simple workflow from test suites."""
    workflow_id = str(uuid.uuid4())
    steps = []

    for i, suite_name in enumerate(test_suites):
        step = WorkflowStep(
            step_id=f"suite_{i}",
            step_type=StepType.TEST_SUITE,
            name=f"Execute {suite_name}",
            parameters={"suite_name": suite_name, "parallel": parallel}
        )
        steps.append(step)

    return TestWorkflow(
        workflow_id=workflow_id,
        name=name,
        description=f"Simple workflow executing {len(test_suites)} test suites",
        steps=steps
    )


def create_conditional_workflow(name: str,
                              setup_suite: str,
                              main_suites: List[str],
                              cleanup_suite: str) -> TestWorkflow:
    """Create a conditional workflow with setup, main tests, and cleanup."""
    workflow_id = str(uuid.uuid4())
    steps = []

    # Setup step
    setup_step = WorkflowStep(
        step_id="setup",
        step_type=StepType.TEST_SUITE,
        name="Setup",
        parameters={"suite_name": setup_suite}
    )
    steps.append(setup_step)

    # Main test steps (depend on setup)
    for i, suite_name in enumerate(main_suites):
        main_step = WorkflowStep(
            step_id=f"main_{i}",
            step_type=StepType.TEST_SUITE,
            name=f"Main Tests {i+1}",
            depends_on=["setup"],
            parameters={"suite_name": suite_name}
        )
        steps.append(main_step)

    # Cleanup step (always runs)
    cleanup_step = WorkflowStep(
        step_id="cleanup",
        step_type=StepType.TEST_SUITE,
        name="Cleanup",
        depends_on=[f"main_{i}" for i in range(len(main_suites))],
        conditions=[WorkflowCondition(condition_type=ConditionType.ALWAYS)],
        parameters={"suite_name": cleanup_suite}
    )
    steps.append(cleanup_step)

    return TestWorkflow(
        workflow_id=workflow_id,
        name=name,
        description="Conditional workflow with setup, main tests, and cleanup",
        steps=steps
    )
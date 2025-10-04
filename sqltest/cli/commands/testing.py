"""SQL unit testing command implementation for SQLTest CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import click
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

from sqltest import __version__
from sqltest.cli.utils import console
from sqltest.config import get_config
from sqltest.db import get_connection_manager
from sqltest.exceptions import ConfigurationError


@click.command(name="test")
@click.option("--config", type=click.Path(exists=True), help="Path to test configuration file")
@click.option("--directory", "-d", type=click.Path(exists=True), help="Directory containing test files")
@click.option("--pattern", default="*test*.yaml", help="File pattern for test discovery")
@click.option("--group", help="Specific test group/tag to run")
@click.option("--test-name", help="Specific test name to run")
@click.option("--database", help="Database to run tests against")
@click.option("--parallel", is_flag=True, help="Enable parallel test execution")
@click.option("--coverage", is_flag=True, help="Generate coverage report")
@click.option("--fail-fast", is_flag=True, help="Stop on first test failure")
@click.option("--verbose", "-v", is_flag=True, help="Verbose test output")
@click.option("--output", "-o", type=click.Path(), help="Export results to JSON file")
@click.pass_context
def test_command(
    ctx: click.Context,
    config: Optional[str],
    directory: Optional[str],
    pattern: str,
    group: Optional[str],
    test_name: Optional[str],
    database: Optional[str],
    parallel: bool,
    coverage: bool,
    fail_fast: bool,
    verbose: bool,
    output: Optional[str],
) -> None:
    """Execute SQL unit tests with reporting and coverage output."""
    try:
        from sqltest.modules.sql_testing import SQLTestRunner

        app_config = get_config(ctx.obj.get("config"))
        manager = get_connection_manager(app_config)
        db_name = database or ctx.obj.get("db") or app_config.default_database

        console.print(f"[bold magenta]SQL Unit Tests[/bold magenta]")
        console.print(f"Database: [cyan]{db_name}[/cyan]")
        if group:
            console.print(f"Group/Tag Filter: [cyan]{group}[/cyan]")
        if test_name:
            console.print(f"Test Filter: [cyan]{test_name}[/cyan]")
        console.print()

        test_runner = SQLTestRunner(manager)
        test_results: List = []

        if config:
            test_results = _run_from_config(
                test_runner,
                config=config,
                db_name=db_name,
                group=group,
                test_name=test_name,
                parallel=parallel,
                coverage=coverage,
                fail_fast=fail_fast,
                verbose=verbose or ctx.obj.get("verbose", False),
            )
        elif directory:
            test_results = _run_from_directory(
                test_runner,
                directory=directory,
                pattern=pattern,
                db_name=db_name,
                group=group,
                parallel=parallel,
                coverage=coverage,
                fail_fast=fail_fast,
                verbose=verbose or ctx.obj.get("verbose", False),
            )
        else:
            test_results = _run_auto_discovery(
                test_runner,
                pattern=pattern,
                db_name=db_name,
                group=group,
                parallel=parallel,
                coverage=coverage,
                fail_fast=fail_fast,
                verbose=verbose or ctx.obj.get("verbose", False),
            )

        if not test_results:
            console.print("[yellow]No tests were executed.[/yellow]")
            return

        _render_test_summary(test_results, verbose or ctx.obj.get("verbose", False))

        if coverage:
            _render_coverage_report(test_results)

        if output:
            _export_test_results(test_results, output, include_coverage=coverage)
            console.print(f"[green]📄 Test results exported to: {output}[/green]")

        failed_tests = sum(1 for result in test_results if not result.passed)
        if failed_tests:
            console.print(f"\n[red]❌ {failed_tests}/{len(test_results)} tests failed[/red]")
            raise SystemExit(1)

        console.print(f"\n[green]✅ All {len(test_results)} tests passed[/green]")

    except ConfigurationError as exc:
        console.print(f"[red]Configuration Error: {exc}[/red]")
        raise SystemExit(1) from exc
    except Exception as exc:  # pragma: no cover - defensive programming
        if ctx.obj.get("verbose") or verbose:
            import traceback

            console.print(f"[red]Error: {exc}[/red]")
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
        else:
            console.print(f"[red]Error: {exc}[/red]")
        raise SystemExit(1) from exc


def _run_from_config(
    runner,
    *,
    config: str,
    db_name: str,
    group: Optional[str],
    test_name: Optional[str],
    parallel: bool,
    coverage: bool,
    fail_fast: bool,
    verbose: bool,
):
    console.print(f"[blue]📋 Loading test configuration: {config}[/blue]")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("[green]Loading test configuration...", total=100)
        try:
            if test_name:
                result = runner.run_test(config, test_name, db_name, fail_fast=fail_fast)
                results = [result] if result else []
            else:
                results = runner.run_test_suite(
                    config,
                    db_name,
                    tag_filter=group,
                    parallel=parallel,
                    fail_fast=fail_fast,
                    enable_coverage=coverage,
                )
        finally:
            progress.update(task, completed=100)
    return results


def _run_from_directory(
    runner,
    *,
    directory: str,
    pattern: str,
    db_name: str,
    group: Optional[str],
    parallel: bool,
    coverage: bool,
    fail_fast: bool,
    verbose: bool,
):
    console.print(f"[blue]🔍 Discovering tests in: {directory}[/blue]")
    console.print(f"Pattern: [cyan]{pattern}[/cyan]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("[green]Discovering and running tests...", total=100)
        try:
            results = runner.run_tests_in_directory(
                directory,
                db_name,
                pattern=pattern,
                tag_filter=group,
                parallel=parallel,
                fail_fast=fail_fast,
                enable_coverage=coverage,
            )
        finally:
            progress.update(task, completed=100)
    return results


def _run_auto_discovery(
    runner,
    *,
    pattern: str,
    db_name: str,
    group: Optional[str],
    parallel: bool,
    coverage: bool,
    fail_fast: bool,
    verbose: bool,
):
    from pathlib import Path

    current_dir = Path('.')
    test_files = list(current_dir.glob(pattern)) + list(current_dir.glob('tests/*.yaml'))
    if not test_files:
        console.print("[yellow]No test files found. Use --config or --directory to specify tests.[/yellow]")
        console.print("\n[dim]Expected patterns:[/dim]")
        console.print(f"  [dim]• {pattern}[/dim]")
        console.print("  [dim]• tests/*.yaml[/dim]")
        return []

    console.print(f"[blue]🔍 Found test files in current directory:[/blue]")
    for test_file in test_files:
        console.print(f"  [green]• {test_file}[/green]")
    console.print()

    return _run_from_directory(
        runner,
        directory=str(current_dir),
        pattern=pattern,
        db_name=db_name,
        group=group,
        parallel=parallel,
        coverage=coverage,
        fail_fast=fail_fast,
        verbose=verbose,
    )


def _render_test_summary(test_results: List, verbose: bool) -> None:
    console.print("\n" + "=" * 80)
    console.print("[bold magenta]TEST RESULTS[/bold magenta]")
    console.print("=" * 80)

    for result in test_results:
        _render_single_test(result, verbose)
        console.print("\n" + "-" * 60 + "\n")

    if len(test_results) > 1:
        _render_test_statistics(test_results)


def _render_single_test(result, verbose: bool) -> None:
    status = "✅" if result.passed else "❌"
    console.print(f"[bold cyan]{status} {result.test_name}[/bold cyan]")
    console.print(f"Execution Time: {result.execution_time_ms:.2f}ms")
    if result.error_message:
        console.print(f"Error: [red]{result.error_message}[/red]")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Assertion", style="cyan")
    table.add_column("Passed", style="green")
    table.add_column("Message", style="white")

    for assertion in result.assertion_results:
        table.add_row(
            assertion.name,
            "✅" if assertion.passed else "❌",
            assertion.message or "",
        )

    console.print(table)

    if verbose and not result.passed:
        _render_test_details([result])


def _render_test_statistics(test_results: List) -> None:
    total_tests = len(test_results)
    passed_tests = sum(1 for result in test_results if result.passed)
    failed_tests = total_tests - passed_tests
    total_assertions = sum(len(result.assertion_results) for result in test_results)
    passed_assertions = sum(
        sum(1 for assertion in result.assertion_results if assertion.passed)
        for result in test_results
    )

    total_time = sum(result.execution_time_ms for result in test_results)
    avg_time = total_time / total_tests if total_tests else 0

    console.print("[bold blue]📊 Test Statistics[/bold blue]")
    stats_table = Table(show_header=True, header_style="bold magenta")
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Count", style="white", justify="right")
    stats_table.add_column("Percentage", style="green", justify="right")

    stats_table.add_row("Total Tests", str(total_tests), "100%")
    stats_table.add_row("✅ Passed", str(passed_tests), f"{(passed_tests/total_tests*100):.1f}%" if total_tests else "0.0%")
    stats_table.add_row("❌ Failed", str(failed_tests), f"{(failed_tests/total_tests*100):.1f}%" if total_tests else "0.0%")
    stats_table.add_row("", "", "")
    stats_table.add_row("Total Assertions", str(total_assertions), "100%")
    stats_table.add_row(
        "✅ Passed",
        str(passed_assertions),
        f"{(passed_assertions/total_assertions*100):.1f}%" if total_assertions else "0.0%",
    )
    stats_table.add_row(
        "❌ Failed",
        str(total_assertions - passed_assertions),
        f"{((total_assertions - passed_assertions)/total_assertions*100):.1f}%" if total_assertions else "0.0%",
    )
    stats_table.add_row("", "", "")
    stats_table.add_row("Total Time", f"{total_time:.2f}ms", "")
    stats_table.add_row("Average Time", f"{avg_time:.2f}ms", "")

    console.print(stats_table)

    fastest_test = min(test_results, key=lambda t: t.execution_time_ms)
    slowest_test = max(test_results, key=lambda t: t.execution_time_ms)
    console.print("\n[bold yellow]⚡ Performance Insights[/bold yellow]")
    console.print(f"Fastest: [green]{fastest_test.test_name}[/green] ({fastest_test.execution_time_ms:.2f}ms)")
    console.print(f"Slowest: [red]{slowest_test.test_name}[/red] ({slowest_test.execution_time_ms:.2f}ms)")


def _render_test_details(test_results: List) -> None:
    from rich.table import Table as RichTable

    for result in test_results:
        if result.passed:
            continue

        console.print(f"\n[bold red]❌ Test Failed: {result.test_name}[/bold red]")
        if result.error_message:
            console.print(f"Error: {result.error_message}")

        detail_table = RichTable(show_header=True, header_style="bold magenta")
        detail_table.add_column("Assertion", style="cyan")
        detail_table.add_column("Passed", style="green")
        detail_table.add_column("Message", style="white")

        for assertion in result.assertion_results:
            detail_table.add_row(
                assertion.name,
                "✅" if assertion.passed else "❌",
                assertion.message or "",
            )

        console.print(detail_table)


def _render_coverage_report(test_results: List) -> None:
    console.print("\n[bold green]📈 Coverage Report[/bold green]")

    covered_tables = set()
    covered_queries = set()

    for result in test_results:
        coverage_info = getattr(result, "coverage_info", {}) or {}
        covered_tables.update(coverage_info.get("tables", []))
        covered_queries.update(coverage_info.get("queries", []))

    if not covered_tables and not covered_queries:
        console.print("[yellow]No coverage information available in test results[/yellow]")
        console.print("[dim]Coverage tracking may need to be enabled in test configuration[/dim]")
        return

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Type", style="cyan")
    table.add_column("Items Covered", style="white", justify="right")

    if covered_tables:
        table.add_row("Tables", str(len(covered_tables)))
    if covered_queries:
        table.add_row("Custom Queries", str(len(covered_queries)))

    console.print(table)
    if covered_tables:
        console.print(f"[dim]Tables: {', '.join(sorted(covered_tables))}[/dim]")
    if covered_queries:
        console.print(f"[dim]Queries: {', '.join(sorted(covered_queries))}[/dim]")


def _export_test_results(test_results: List, output_path: str, *, include_coverage: bool = False) -> None:
    export_data = {
        "export_info": {
            "timestamp": __version__,
            "sqltest_version": __version__,
            "test_count": len(test_results),
            "include_coverage": include_coverage,
        },
        "summary": {
            "total_tests": len(test_results),
            "passed_tests": sum(1 for r in test_results if r.passed),
            "failed_tests": sum(1 for r in test_results if not r.passed),
            "success_rate": (
                sum(1 for r in test_results if r.passed) / len(test_results) * 100
            ) if test_results else 0,
            "total_execution_time_ms": sum(r.execution_time_ms for r in test_results),
            "average_execution_time_ms": (
                sum(r.execution_time_ms for r in test_results) / len(test_results)
            ) if test_results else 0,
        },
        "test_results": [],
    }

    for result in test_results:
        test_payload = {
            "test_name": result.test_name,
            "passed": result.passed,
            "execution_time_ms": result.execution_time_ms,
            "setup_time_ms": result.setup_time_ms,
            "teardown_time_ms": result.teardown_time_ms,
            "tags": getattr(result, "tags", []),
            "error_message": result.error_message,
            "assertions": [
                {
                    "name": assertion.name,
                    "passed": assertion.passed,
                    "message": assertion.message,
                }
                for assertion in result.assertion_results
            ],
        }
        if include_coverage and hasattr(result, "coverage_info"):
            test_payload["coverage"] = result.coverage_info
        export_data["test_results"].append(test_payload)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(export_data, indent=2))

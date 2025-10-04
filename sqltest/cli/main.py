"""Main CLI entry point for SQLTest Pro."""

import sys
from pathlib import Path
from typing import Optional

import click
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich import print as rprint

from sqltest import __version__
from sqltest.cli.commands.configuration import config_group
from sqltest.cli.commands.database import db_group
from sqltest.cli.commands.init import init_command
from sqltest.cli.commands.profile import profile_command
from sqltest.cli.commands.report import report_command
from sqltest.cli.commands.testing import test_command
from sqltest.cli.commands.validate import validate_command
from sqltest.cli.utils import console, print_exception
from sqltest.config import get_config
from sqltest.db import get_connection_manager
from sqltest.exceptions import ConfigurationError, DatabaseError


@click.group(invoke_without_command=True)
@click.option("--version", is_flag=True, help="Show version information")
@click.option("--config", type=click.Path(exists=True), help="Path to configuration file")
@click.option("--db", help="Database connection name")
@click.option("--output", type=click.Choice(["table", "json", "html"]), default="table", help="Output format")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.pass_context
def cli(ctx: click.Context, version: bool, config: str, db: str, output: str, verbose: bool) -> None:
    """SQLTest Pro - A comprehensive SQL testing and validation suite.
    
    \b
    🚀 Features:
    • Data profiling and analysis
    • Field and business rule validation  
    • Business logic and integrity validation
    • SQL unit testing with coverage
    • Interactive CLI with progress tracking
    • Multi-database support
    • YAML-based configuration
    """
    ctx.ensure_object(dict)
    ctx.obj.update({
        "config": config,
        "db": db,
        "output": output,
        "verbose": verbose,
    })
    
    if version:
        console.print(f"SQLTest Pro v{__version__}")
        return
    
    if ctx.invoked_subcommand is None:
        # Show main dashboard
        show_dashboard()


cli.add_command(profile_command)
cli.add_command(validate_command)
cli.add_command(test_command)
cli.add_command(db_group)
cli.add_command(config_group)
cli.add_command(init_command)
cli.add_command(report_command)


def show_dashboard() -> None:
    """Display the main interactive dashboard."""
    title = Text("SQLTest Pro", style="bold blue")
    subtitle = Text("A comprehensive SQL testing and validation suite", style="italic")
    
    dashboard_content = Text()
    dashboard_content.append("📊 Profile Data\n", style="bold")
    dashboard_content.append("✓  Field Validations\n", style="bold") 
    dashboard_content.append("🔍 Business Rules\n", style="bold")
    dashboard_content.append("🧪 Execute Unit Tests\n", style="bold")
    dashboard_content.append("📄 Generate Reports\n", style="bold")
    dashboard_content.append("⚙️  Configure Settings\n", style="bold")
    dashboard_content.append("📚 View Documentation\n", style="bold")
    dashboard_content.append("\nRun 'sqltest --help' for available commands", style="dim")
    
    panel = Panel(
        dashboard_content,
        title=title,
        subtitle=subtitle,
        border_style="blue",
        padding=(1, 2)
    )
    
    console.print(panel)









def show_validation_summary(summary) -> None:
    """Display validation summary in a formatted way."""
    from sqltest.modules.business_rules.models import RuleStatus, RuleSeverity
    
    console.print(f"[bold blue]📊 Rule Set: {summary.rule_set_name}[/bold blue]")
    console.print(f"Database: [cyan]{summary.validation_context.database_name}[/cyan]")
    if summary.validation_context.table_name:
        console.print(f"Table: [cyan]{summary.validation_context.table_name}[/cyan]")
    console.print(f"Execution Time: [yellow]{summary.execution_time_ms:.2f}ms[/yellow]")
    console.print()
    
    # Summary statistics
    summary_table = Table(show_header=True, header_style="bold magenta")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Count", style="white", justify="right")
    summary_table.add_column("Percentage", style="green", justify="right")
    
    summary_table.add_row("Total Rules", str(summary.total_rules), "100%")
    summary_table.add_row("✅ Passed", str(summary.rules_passed), f"{summary.success_rate:.1f}%")
    summary_table.add_row("❌ Failed", str(summary.rules_failed), f"{(summary.rules_failed/summary.total_rules*100) if summary.total_rules > 0 else 0:.1f}%")
    summary_table.add_row("🔥 Errors", str(summary.rules_error), f"{(summary.rules_error/summary.total_rules*100) if summary.total_rules > 0 else 0:.1f}%")
    summary_table.add_row("⏭️  Skipped", str(summary.rules_skipped), f"{(summary.rules_skipped/summary.total_rules*100) if summary.total_rules > 0 else 0:.1f}%")
    
    console.print(summary_table)
    console.print()
    
    # Violations summary
    if summary.total_violations > 0:
        violations_table = Table(show_header=True, header_style="bold red")
        violations_table.add_column("Violation Type", style="yellow")
        violations_table.add_column("Count", style="white", justify="right")
        
        if summary.critical_violations > 0:
            violations_table.add_row("🚨 Critical", str(summary.critical_violations))
        if summary.error_violations > 0:
            violations_table.add_row("🔴 Error", str(summary.error_violations))
        if summary.warning_violations > 0:
            violations_table.add_row("🟡 Warning", str(summary.warning_violations))
        if summary.info_violations > 0:
            violations_table.add_row("🔵 Info", str(summary.info_violations))
        
        violations_table.add_row("[bold]Total[/bold]", f"[bold]{summary.total_violations}[/bold]")
        
        console.print("[bold red]📋 Violations Summary[/bold red]")
        console.print(violations_table)
        console.print()
    
    # Show detailed rule results
    if summary.results:
        console.print("[bold cyan]📝 Rule Results Details[/bold cyan]")
        
        details_table = Table(show_header=True, header_style="bold magenta")
        details_table.add_column("Rule", style="cyan", max_width=30)
        details_table.add_column("Type", style="green", width=12)
        details_table.add_column("Status", style="white", width=8)
        details_table.add_column("Violations", style="red", justify="right", width=10)
        details_table.add_column("Time (ms)", style="yellow", justify="right", width=10)
        details_table.add_column("Message", style="white", max_width=40)
        
        for result in summary.results:
            # Status icon
            if result.status == RuleStatus.PASSED:
                status_display = "[green]✅ PASS[/green]"
            elif result.status == RuleStatus.FAILED:
                status_display = "[red]❌ FAIL[/red]"
            elif result.status == RuleStatus.ERROR:
                status_display = "[bright_red]🔥 ERROR[/bright_red]"
            elif result.status == RuleStatus.SKIPPED:
                status_display = "[dim]⏭️  SKIP[/dim]"
            else:
                status_display = f"[dim]{result.status.value.upper()}[/dim]"
            
            # Rule type display
            type_display = result.rule_type.value.replace('_', ' ').title()
            
            # Violation count
            violation_count = str(result.violation_count) if result.violations else "0"
            
            # Truncate message if too long
            message = result.message[:37] + "..." if len(result.message) > 40 else result.message
            
            details_table.add_row(
                result.rule_name,
                type_display,
                status_display,
                violation_count,
                f"{result.execution_time_ms:.1f}",
                message
            )
        
        console.print(details_table)
        
        # Show violation details for failed rules
        failed_rules = [r for r in summary.results if r.violations]
        if failed_rules:
            console.print("\n[bold red]🔍 Violation Details[/bold red]")
            for result in failed_rules[:3]:  # Show details for first 3 failed rules
                console.print(f"\n[red]Rule: {result.rule_name}[/red]")
                for violation in result.violations[:2]:  # Show first 2 violations per rule
                    console.print(f"  • [yellow]{violation.message}[/yellow]")
                    if violation.table_name:
                        console.print(f"    Table: [cyan]{violation.table_name}[/cyan]")
                    if violation.column_name:
                        console.print(f"    Column: [cyan]{violation.column_name}[/cyan]")
                    if violation.violation_count > 1:
                        console.print(f"    Count: [red]{violation.violation_count}[/red]")
                if len(result.violations) > 2:
                    console.print(f"    [dim]... and {len(result.violations) - 2} more violations[/dim]")
            if len(failed_rules) > 3:
                console.print(f"\n[dim]... and {len(failed_rules) - 3} more failed rules[/dim]")


def show_validation_statistics(stats: dict) -> None:
    """Display aggregate validation statistics."""
    console.print("[bold blue]📈 Aggregate Statistics[/bold blue]")
    
    # Overall statistics
    overall_table = Table(show_header=True, header_style="bold magenta")
    overall_table.add_column("Metric", style="cyan")
    overall_table.add_column("Value", style="white", justify="right")
    
    rule_stats = stats['rule_statistics']
    exec_stats = stats['execution_statistics']
    quality_stats = stats['quality_indicators']
    
    overall_table.add_row("Validation Count", str(stats['validation_count']))
    overall_table.add_row("Total Rules", str(rule_stats['total_rules']))
    overall_table.add_row("Overall Success Rate", f"{rule_stats['overall_success_rate']:.1f}%")
    overall_table.add_row("Average Execution Time", f"{exec_stats['average_execution_time_ms']:.1f}ms")
    overall_table.add_row("Clean Validations", str(quality_stats['clean_validations']))
    overall_table.add_row("Validations with Issues", str(quality_stats['validations_with_issues']))
    
    console.print(overall_table)
    
    # Quality indicators
    if quality_stats['has_any_critical_issues'] or quality_stats['has_any_errors']:
        console.print("\n[bold red]⚠️  Quality Issues Detected[/bold red]")
        if quality_stats['has_any_critical_issues']:
            console.print("[red]  • Critical issues found across validations[/red]")
        if quality_stats['has_any_errors']:
            console.print("[yellow]  • Error-level issues found across validations[/yellow]")
    else:
        console.print("\n[bold green]✅ All Validations Clean[/bold green]")


def show_test_result(test_result, verbose: bool = False) -> None:
    """Display individual test result in a formatted way."""
    from sqltest.modules.sql_testing.models import TestStatus
    
    # Test header
    status_icon = "✅" if test_result.passed else "❌"
    console.print(f"{status_icon} [bold]{test_result.test_name}[/bold]")
    
    # Test details
    details_table = Table(show_header=False, box=None, pad_edge=False)
    details_table.add_column("Property", style="cyan", width=15)
    details_table.add_column("Value", style="white")
    
    details_table.add_row("Status:", "[green]PASSED[/green]" if test_result.passed else "[red]FAILED[/red]")
    details_table.add_row("Duration:", f"{test_result.execution_time_ms:.2f}ms")
    details_table.add_row("Assertions:", str(len(test_result.assertion_results)))
    
    if test_result.tags:
        details_table.add_row("Tags:", ", ".join(test_result.tags))
    
    if test_result.setup_time_ms > 0:
        details_table.add_row("Setup Time:", f"{test_result.setup_time_ms:.2f}ms")
    
    if test_result.teardown_time_ms > 0:
        details_table.add_row("Teardown Time:", f"{test_result.teardown_time_ms:.2f}ms")
    
    console.print(details_table)
    
    # Show assertion results
    if verbose or not test_result.passed:
        console.print("\n[bold cyan]Assertion Results:[/bold cyan]")
        
        assertions_table = Table(show_header=True, header_style="bold magenta")
        assertions_table.add_column("#", style="dim", width=3)
        assertions_table.add_column("Type", style="green", width=15)
        assertions_table.add_column("Status", style="white", width=8)
        assertions_table.add_column("Message", style="white")
        
        for i, assertion_result in enumerate(test_result.assertion_results, 1):
            status_display = "[green]✓[/green]" if assertion_result.passed else "[red]✗[/red]"
            assertion_type = assertion_result.assertion_type.replace('_', ' ').title()
            
            # Truncate long messages
            message = assertion_result.message
            if len(message) > 60:
                message = message[:57] + "..."
            
            assertions_table.add_row(
                str(i),
                assertion_type,
                status_display,
                message
            )
        
        console.print(assertions_table)
    
    # Show error details if any
    if test_result.error_message:
        console.print(f"\n[bold red]Error Details:[/bold red]")
        console.print(f"[red]{test_result.error_message}[/red]")
    
    # Show test description if available
    if verbose and hasattr(test_result, 'description') and test_result.description:
        console.print(f"\n[dim]Description: {test_result.description}[/dim]")


def show_test_statistics(test_results: list) -> None:
    """Display aggregate test statistics."""
    total_tests = len(test_results)
    passed_tests = sum(1 for result in test_results if result.passed)
    failed_tests = total_tests - passed_tests
    
    total_assertions = sum(len(result.assertion_results) for result in test_results)
    passed_assertions = sum(sum(1 for assertion in result.assertion_results if assertion.passed) for result in test_results)
    
    total_time = sum(result.execution_time_ms for result in test_results)
    avg_time = total_time / total_tests if total_tests > 0 else 0
    
    console.print("[bold blue]📊 Test Statistics[/bold blue]")
    
    stats_table = Table(show_header=True, header_style="bold magenta")
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Count", style="white", justify="right")
    stats_table.add_column("Percentage", style="green", justify="right")
    
    stats_table.add_row("Total Tests", str(total_tests), "100%")
    stats_table.add_row("✅ Passed", str(passed_tests), f"{(passed_tests/total_tests*100):.1f}%")
    stats_table.add_row("❌ Failed", str(failed_tests), f"{(failed_tests/total_tests*100):.1f}%")
    stats_table.add_row("", "", "")
    stats_table.add_row("Total Assertions", str(total_assertions), "100%")
    stats_table.add_row("✅ Passed", str(passed_assertions), f"{(passed_assertions/total_assertions*100) if total_assertions > 0 else 0:.1f}%")
    stats_table.add_row("❌ Failed", str(total_assertions - passed_assertions), f"{((total_assertions - passed_assertions)/total_assertions*100) if total_assertions > 0 else 0:.1f}%")
    stats_table.add_row("", "", "")
    stats_table.add_row("Total Time", f"{total_time:.2f}ms", "")
    stats_table.add_row("Average Time", f"{avg_time:.2f}ms", "")
    
    console.print(stats_table)
    
    # Performance insights
    if test_results:
        fastest_test = min(test_results, key=lambda t: t.execution_time_ms)
        slowest_test = max(test_results, key=lambda t: t.execution_time_ms)
        
        console.print(f"\n[bold yellow]⚡ Performance Insights[/bold yellow]")
        console.print(f"Fastest: [green]{fastest_test.test_name}[/green] ({fastest_test.execution_time_ms:.2f}ms)")
        console.print(f"Slowest: [red]{slowest_test.test_name}[/red] ({slowest_test.execution_time_ms:.2f}ms)")


def show_coverage_report(test_results: list) -> None:
    """Display test coverage report."""
    console.print("\n[bold green]📈 Coverage Report[/bold green]")
    
    # Extract coverage data from test results
    covered_tables = set()
    covered_queries = set()
    
    for result in test_results:
        # This would need to be enhanced based on actual coverage tracking in test results
        if hasattr(result, 'coverage_info'):
            if result.coverage_info.get('tables'):
                covered_tables.update(result.coverage_info['tables'])
            if result.coverage_info.get('queries'):
                covered_queries.update(result.coverage_info['queries'])
    
    if covered_tables or covered_queries:
        coverage_table = Table(show_header=True, header_style="bold magenta")
        coverage_table.add_column("Type", style="cyan")
        coverage_table.add_column("Items Covered", style="white", justify="right")
        
        if covered_tables:
            coverage_table.add_row("Tables", str(len(covered_tables)))
            console.print(coverage_table)
            console.print(f"\n[dim]Tables: {', '.join(sorted(covered_tables))}[/dim]")
        
        if covered_queries:
            coverage_table.add_row("Custom Queries", str(len(covered_queries)))
    else:
        console.print("[yellow]No coverage information available in test results[/yellow]")
        console.print("[dim]Coverage tracking may need to be enabled in test configuration[/dim]")


def export_test_results(test_results: list, output_path: str, include_coverage: bool = False) -> None:
    """Export test results to JSON file."""
    import json
    from pathlib import Path
    from datetime import datetime
    
    # Prepare export data
    export_data = {
        "export_info": {
            "timestamp": datetime.now().isoformat(),
            "sqltest_version": __version__,
            "test_count": len(test_results),
            "include_coverage": include_coverage
        },
        "summary": {
            "total_tests": len(test_results),
            "passed_tests": sum(1 for r in test_results if r.passed),
            "failed_tests": sum(1 for r in test_results if not r.passed),
            "success_rate": (sum(1 for r in test_results if r.passed) / len(test_results) * 100) if test_results else 0,
            "total_execution_time_ms": sum(r.execution_time_ms for r in test_results),
            "average_execution_time_ms": (sum(r.execution_time_ms for r in test_results) / len(test_results)) if test_results else 0
        },
        "test_results": []
    }
    
    # Convert test results to serializable format
    for result in test_results:
        test_data = {
            "test_name": result.test_name,
            "passed": result.passed,
            "execution_time_ms": result.execution_time_ms,
            "setup_time_ms": result.setup_time_ms,
            "teardown_time_ms": result.teardown_time_ms,
            "tags": result.tags if hasattr(result, 'tags') else [],
            "error_message": result.error_message,
            "assertion_results": [
                {
                    "assertion_type": assertion.assertion_type,
                    "passed": assertion.passed,
                    "message": assertion.message,
                    "expected": str(assertion.expected) if hasattr(assertion, 'expected') else None,
                    "actual": str(assertion.actual) if hasattr(assertion, 'actual') else None
                }
                for assertion in result.assertion_results
            ]
        }
        
        # Add coverage information if available
        if include_coverage and hasattr(result, 'coverage_info'):
            test_data["coverage_info"] = result.coverage_info
        
        export_data["test_results"].append(test_data)
    
    # Add aggregate coverage if requested
    if include_coverage:
        covered_tables = set()
        covered_queries = set()
        
        for result in test_results:
            if hasattr(result, 'coverage_info'):
                if result.coverage_info.get('tables'):
                    covered_tables.update(result.coverage_info['tables'])
                if result.coverage_info.get('queries'):
                    covered_queries.update(result.coverage_info['queries'])
        
        export_data["coverage_summary"] = {
            "tables_covered": list(covered_tables),
            "queries_covered": list(covered_queries),
            "table_count": len(covered_tables),
            "query_count": len(covered_queries)
        }
    
    # Write to file
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path_obj, 'w') as f:
        json.dump(export_data, f, indent=2, default=str)




# Field validation display functions
def show_field_validation_result(result, verbose: bool = False) -> None:
    """Display field validation result in a formatted way."""
    from sqltest.modules.field_validator import ValidationLevel
    
    console.print(f"[bold blue]📋 Table: {result.table_name}[/bold blue]")
    console.print(f"Database: [cyan]{result.database_name}[/cyan]")
    console.print(f"Validation Time: [yellow]{result.validation_timestamp.strftime('%Y-%m-%d %H:%M:%S')}[/yellow]")
    console.print()
    
    # Overall summary
    summary_table = Table(show_header=True, header_style="bold magenta")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="white", justify="right")
    
    summary_table.add_row("Fields Validated", str(len(result.field_results)))
    summary_table.add_row("Total Rules", str(result.total_rules))
    summary_table.add_row("✅ Passed", str(result.passed_rules))
    summary_table.add_row("❌ Failed", str(result.failed_rules))
    summary_table.add_row("⚠️  Warnings", str(result.warnings))
    summary_table.add_row("Success Rate", f"{result.overall_success_rate:.1f}%")
    
    console.print(summary_table)
    console.print()
    
    # Field-level results
    if result.field_results:
        console.print("[bold cyan]📝 Field Validation Details[/bold cyan]")
        
        field_table = Table(show_header=True, header_style="bold magenta")
        field_table.add_column("Field", style="cyan", max_width=20)
        field_table.add_column("Rows", style="white", justify="right", width=8)
        field_table.add_column("Rules", style="green", justify="right", width=6)
        field_table.add_column("✅ Pass", style="green", justify="right", width=6)
        field_table.add_column("❌ Fail", style="red", justify="right", width=6)
        field_table.add_column("⚠️  Warn", style="yellow", justify="right", width=6)
        field_table.add_column("Rate", style="blue", justify="right", width=8)
        field_table.add_column("Status", style="white", width=8)
        
        for field_result in result.field_results:
            # Status display
            if field_result.has_errors:
                status_display = "[red]ERROR[/red]"
            elif field_result.has_warnings:
                status_display = "[yellow]WARN[/yellow]"
            else:
                status_display = "[green]PASS[/green]"
            
            field_table.add_row(
                field_result.column_name,
                f"{field_result.total_rows:,}",
                str(len(field_result.validation_results)),
                str(field_result.passed_rules),
                str(field_result.failed_rules),
                str(field_result.warnings),
                f"{field_result.success_rate:.1f}%",
                status_display
            )
        
        console.print(field_table)
    
    # Show detailed errors if any failures or verbose mode
    failed_fields = [fr for fr in result.field_results if fr.has_errors or fr.has_warnings]
    if failed_fields and (verbose or result.has_errors):
        console.print("\n[bold red]🔍 Validation Issues[/bold red]")
        
        for field_result in failed_fields[:5]:  # Show first 5 fields with issues
            console.print(f"\n[yellow]Field: {field_result.column_name}[/yellow]")
            
            error_results = [vr for vr in field_result.validation_results 
                           if not vr.passed and vr.level in [ValidationLevel.ERROR, ValidationLevel.WARNING]]
            
            for validation_result in error_results[:3]:  # Show first 3 issues per field
                level_icon = "🔴" if validation_result.level == ValidationLevel.ERROR else "🟡"
                console.print(f"  {level_icon} [dim]Rule:[/dim] {validation_result.rule_name}")
                console.print(f"    [red]{validation_result.message}[/red]")
                if validation_result.value is not None:
                    value_str = str(validation_result.value)[:50] + "..." if len(str(validation_result.value)) > 50 else str(validation_result.value)
                    console.print(f"    [dim]Value:[/dim] {value_str}")
                if validation_result.row_number:
                    console.print(f"    [dim]Row:[/dim] {validation_result.row_number}")
            
            if len(error_results) > 3:
                console.print(f"    [dim]... and {len(error_results) - 3} more issues[/dim]")
        
        if len(failed_fields) > 5:
            console.print(f"\n[dim]... and {len(failed_fields) - 5} more fields with issues[/dim]")


def show_field_validation_summary(results: list) -> None:
    """Display aggregate field validation summary."""
    console.print("[bold blue]📊 Aggregate Field Validation Summary[/bold blue]")
    
    total_tables = len(results)
    total_fields = sum(len(result.field_results) for result in results)
    total_rules = sum(result.total_rules for result in results)
    total_passed = sum(result.passed_rules for result in results)
    total_failed = sum(result.failed_rules for result in results)
    total_warnings = sum(result.warnings for result in results)
    
    tables_with_errors = sum(1 for result in results if result.has_errors)
    tables_with_warnings = sum(1 for result in results if result.has_warnings)
    
    # Aggregate statistics
    agg_table = Table(show_header=True, header_style="bold magenta")
    agg_table.add_column("Metric", style="cyan")
    agg_table.add_column("Count", style="white", justify="right")
    agg_table.add_column("Percentage", style="green", justify="right")
    
    agg_table.add_row("Tables Validated", str(total_tables), "100%")
    agg_table.add_row("Total Fields", str(total_fields), "")
    agg_table.add_row("Total Rules", str(total_rules), "100%")
    agg_table.add_row("✅ Passed Rules", str(total_passed), f"{(total_passed/total_rules*100) if total_rules > 0 else 0:.1f}%")
    agg_table.add_row("❌ Failed Rules", str(total_failed), f"{(total_failed/total_rules*100) if total_rules > 0 else 0:.1f}%")
    agg_table.add_row("⚠️  Warning Rules", str(total_warnings), f"{(total_warnings/total_rules*100) if total_rules > 0 else 0:.1f}%")
    agg_table.add_row("🔴 Tables with Errors", str(tables_with_errors), f"{(tables_with_errors/total_tables*100) if total_tables > 0 else 0:.1f}%")
    agg_table.add_row("🟡 Tables with Warnings", str(tables_with_warnings), f"{(tables_with_warnings/total_tables*100) if total_tables > 0 else 0:.1f}%")
    agg_table.add_row("✅ Clean Tables", str(total_tables - tables_with_errors - tables_with_warnings), f"{((total_tables - tables_with_errors - tables_with_warnings)/total_tables*100) if total_tables > 0 else 0:.1f}%")
    
    console.print(agg_table)
    
    # Overall assessment
    if tables_with_errors > 0:
        console.print(f"\n[red]❌ {tables_with_errors} table(s) have validation errors[/red]")
    if tables_with_warnings > 0:
        console.print(f"\n[yellow]⚠️  {tables_with_warnings} table(s) have validation warnings[/yellow]")
    if tables_with_errors == 0 and tables_with_warnings == 0:
        console.print(f"\n[green]✅ All tables passed field validation[/green]")


def export_field_validation_results(results: list, output_path: str) -> None:
    """Export field validation results to JSON file."""
    import json
    from pathlib import Path
    from datetime import datetime
    
    # Prepare export data
    export_data = {
        "export_info": {
            "timestamp": datetime.now().isoformat(),
            "sqltest_version": __version__,
            "validation_count": len(results)
        },
        "summary": {
            "total_tables": len(results),
            "total_fields": sum(len(result.field_results) for result in results),
            "total_rules": sum(result.total_rules for result in results),
            "passed_rules": sum(result.passed_rules for result in results),
            "failed_rules": sum(result.failed_rules for result in results),
            "warnings": sum(result.warnings for result in results),
            "tables_with_errors": sum(1 for result in results if result.has_errors),
            "tables_with_warnings": sum(1 for result in results if result.has_warnings),
            "overall_success_rate": (sum(result.passed_rules for result in results) / sum(result.total_rules for result in results) * 100) if sum(result.total_rules for result in results) > 0 else 0
        },
        "validation_results": []
    }
    
    # Convert validation results to serializable format
    for result in results:
        table_data = {
            "table_name": result.table_name,
            "database_name": result.database_name,
            "validation_timestamp": result.validation_timestamp.isoformat(),
            "total_rules": result.total_rules,
            "passed_rules": result.passed_rules,
            "failed_rules": result.failed_rules,
            "warnings": result.warnings,
            "overall_success_rate": result.overall_success_rate,
            "has_errors": result.has_errors,
            "has_warnings": result.has_warnings,
            "field_results": []
        }
        
        # Add field-level results
        for field_result in result.field_results:
            field_data = {
                "column_name": field_result.column_name,
                "total_rows": field_result.total_rows,
                "passed_rules": field_result.passed_rules,
                "failed_rules": field_result.failed_rules,
                "warnings": field_result.warnings,
                "success_rate": field_result.success_rate,
                "has_errors": field_result.has_errors,
                "has_warnings": field_result.has_warnings,
                "validation_results": [
                    {
                        "rule_name": vr.rule_name,
                        "passed": vr.passed,
                        "level": vr.level.value,
                        "message": vr.message,
                        "value": str(vr.value) if vr.value is not None else None,
                        "row_number": vr.row_number,
                        "validation_timestamp": vr.validation_timestamp.isoformat()
                    }
                    for vr in field_result.validation_results
                ]
            }
            table_data["field_results"].append(field_data)
        
        export_data["validation_results"].append(table_data)
    
    # Write to file
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path_obj, 'w') as f:
        json.dump(export_data, f, indent=2, default=str)


@cli.command()
@click.option("--rule-set", "-r", help="Rule set name or path to YAML configuration")
@click.option("--directory", "-d", type=click.Path(exists=True), help="Directory containing rule set YAML files")
@click.option("--table", "-t", help="Specific table to validate")
@click.option("--query", "-q", help="Custom SQL query to validate")
@click.option("--database", help="Database to validate (default: default database)")
@click.option("--schema", "-s", help="Schema name (database-specific)")
@click.option("--tags", help="Filter rules by tags (comma-separated)")
@click.option("--parallel/--sequential", default=None, help="Override parallel execution setting")
@click.option("--fail-fast", is_flag=True, help="Stop on first critical failure")
@click.option("--max-workers", type=int, default=5, help="Maximum number of worker threads")
@click.option("--output", "-o", type=click.Path(), help="Export results to JSON file")
@click.option("--format", "output_format", type=click.Choice(["table", "json"]), default="table", help="Output format")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed violation information")
@click.pass_context
def business_rules(ctx: click.Context, rule_set: str, directory: str, table: str, query: str,
                  database: str, schema: str, tags: str, parallel: bool, fail_fast: bool,
                  max_workers: int, output: str, output_format: str, verbose: bool) -> None:
    """🔍 Execute business rule validation with comprehensive reporting.
    
    Validates data integrity, business logic, and quality rules including:
    • Data quality checks (completeness, accuracy, consistency)
    • Referential integrity validation
    • Business logic enforcement
    • Custom SQL-based rules
    • Parallel execution with dependency management
    • Rich violation reporting and recommendations
    """
    if not rule_set and not directory:
        console.print("[red]Error: Either --rule-set or --directory must be specified[/red]")
        console.print("\n[dim]Examples:[/dim]")
        console.print("  [cyan]sqltest business-rules --rule-set my_rules.yaml --database prod[/cyan]")
        console.print("  [cyan]sqltest business-rules --directory rules/ --table customers[/cyan]")
        console.print("  [cyan]sqltest business-rules --rule-set ecommerce_rules --tags data_quality[/cyan]")
        return
    
    try:
        from sqltest.modules.business_rules import BusinessRuleValidator
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
        import time as time_module
        from pathlib import Path
        
        config = get_config(ctx.obj.get('config'))
        manager = get_connection_manager(config)
        db_name = database or ctx.obj.get('db') or config.default_database
        
        console.print(f"[bold blue]🔍 Business Rule Validation - Database: {db_name}[/bold blue]\n")
        
        # Initialize validator
        validator = BusinessRuleValidator(manager, max_workers=max_workers)
        
        # Load rule sets
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            load_task = progress.add_task("[green]Loading business rules...", total=100)
            
            loaded_rule_sets = []
            
            try:
                if rule_set:
                    # Load specific rule set
                    if Path(rule_set).exists():
                        # Load from file
                        rule_set_name = validator.load_rule_set_from_file(rule_set)
                        loaded_rule_sets.append(rule_set_name)
                        console.print(f"📄 Loaded rule set from file: [green]{rule_set}[/green]")
                    else:
                        # Assume it's a rule set name already loaded
                        if rule_set in validator.list_rule_sets():
                            loaded_rule_sets.append(rule_set)
                            console.print(f"📋 Using rule set: [green]{rule_set}[/green]")
                        else:
                            console.print(f"[red]❌ Rule set '{rule_set}' not found[/red]")
                            return
                
                if directory:
                    # Load from directory
                    loaded_names = validator.load_rule_sets_from_directory(directory, recursive=True)
                    loaded_rule_sets.extend(loaded_names)
                    console.print(f"📁 Loaded {len(loaded_names)} rule sets from directory: [green]{directory}[/green]")
                
                progress.update(load_task, completed=100)
                
            except Exception as e:
                progress.update(load_task, completed=100)
                console.print(f"[red]❌ Failed to load rule sets: {e}[/red]")
                if ctx.obj.get('verbose') or verbose:
                    import traceback
                    console.print(f"[dim]{traceback.format_exc()}[/dim]")
                sys.exit(1)
        
        # Show loaded rule sets
        if loaded_rule_sets:
            console.print(f"\n📋 Available rule sets:")
            for rule_set_name in loaded_rule_sets:
                rule_set_obj = validator.get_rule_set(rule_set_name)
                enabled_count = len(rule_set_obj.get_enabled_rules())
                total_count = len(rule_set_obj.rules)
                console.print(f"  • [cyan]{rule_set_name}[/cyan]: {enabled_count}/{total_count} enabled rules")
        else:
            console.print("[yellow]⚠️  No rule sets loaded[/yellow]")
            return
        
        # Parse tag filter
        tag_filter = None
        if tags:
            tag_filter = {tag.strip() for tag in tags.split(',')}
            console.print(f"🏷️  Tag filter: [cyan]{', '.join(tag_filter)}[/cyan]")
        
        console.print()
        
        # Execute validation for each rule set
        all_results = []
        
        for rule_set_name in loaded_rule_sets:
            console.print(f"[bold green]▶️  Executing rule set: {rule_set_name}[/bold green]")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
                transient=True
            ) as progress:
                exec_task = progress.add_task(f"[green]Running business rules...", total=100)
                
                try:
                    # Start timing
                    start_time = time_module.time()
                    
                    if table:
                        # Table-specific validation (not implemented for business rules directly)
                        # We'll validate with rule set but note the table context
                        console.print(f"   📊 Table focus: [cyan]{table}[/cyan]")
                    
                    # Execute validation
                    summary = validator.validate_with_rule_set(
                        rule_set_name=rule_set_name,
                        database_name=db_name,
                        schema_name=schema,
                        parallel=parallel,
                        fail_fast=fail_fast,
                        tags=tag_filter
                    )
                    
                    execution_time = time_module.time() - start_time
                    all_results.append(summary)
                    
                    progress.update(exec_task, completed=100)
                    
                    # Show immediate results
                    success_icon = "✅" if not summary.has_errors and not summary.has_critical_issues else "❌"
                    console.print(f"   {success_icon} Completed in {execution_time:.2f}s - {summary.rules_passed}/{summary.total_rules} rules passed")
                    
                except Exception as e:
                    progress.update(exec_task, completed=100)
                    console.print(f"   [red]❌ Failed to execute rule set {rule_set_name}: {e}[/red]")
                    if ctx.obj.get('verbose') or verbose:
                        import traceback
                        console.print(f"[dim]{traceback.format_exc()}[/dim]")
                    continue
        
        # Display comprehensive results
        if not all_results:
            console.print("[yellow]No validation results to display[/yellow]")
            return
        
        console.print("\n" + "="*80)
        console.print("[bold blue]🔍 BUSINESS RULE VALIDATION RESULTS[/bold blue]")
        console.print("="*80)
        
        if output_format == "json":
            display_business_rule_results_json(all_results)
        else:
            display_business_rule_results_table(all_results, verbose)
        
        # Export results if requested
        if output:
            export_business_rule_results(all_results, output)
            console.print(f"\n💾 Results exported to: [cyan]{output}[/cyan]")
        
        # Summary assessment
        total_critical = sum(s.critical_violations for s in all_results)
        total_errors = sum(s.error_violations for s in all_results) 
        total_warnings = sum(s.warning_violations for s in all_results)
        
        if total_critical > 0:
            console.print(f"\n[red]🔴 {total_critical} critical violation(s) found - immediate attention required![/red]")
            sys.exit(1)
        elif total_errors > 0:
            console.print(f"\n[red]❌ {total_errors} error violation(s) found[/red]")
            sys.exit(1)
        elif total_warnings > 0:
            console.print(f"\n[yellow]⚠️  {total_warnings} warning(s) found[/yellow]")
        else:
            console.print(f"\n[green]✅ All business rules passed successfully![/green]")
        
    except ConfigurationError as e:
        console.print(f"[red]Configuration Error: {e}[/red]")
        sys.exit(1)
    except DatabaseError as e:
        console.print(f"[red]Database Error: {e}[/red]")
        sys.exit(1)
    except Exception as e:
        if ctx.obj.get('verbose') or verbose:
            import traceback
            console.print(f"[red]Error: {e}[/red]")
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
        else:
            console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


def display_business_rule_results_table(results: list, verbose: bool = False) -> None:
    """Display business rule validation results in table format."""
    for i, summary in enumerate(results, 1):
        if len(results) > 1:
            console.print(f"\n[bold cyan]📋 Rule Set {i}: {summary.rule_set_name}[/bold cyan]")
        
        # Summary table
        summary_table = Table(show_header=True, header_style="bold magenta")
        summary_table.add_column("Metric", style="cyan", width=25)
        summary_table.add_column("Count", style="white", justify="right", width=10)
        summary_table.add_column("Percentage", style="green", justify="right", width=12)
        
        summary_table.add_row("Execution Time", f"{summary.execution_time_ms:.2f}ms", "")
        summary_table.add_row("Total Rules", str(summary.total_rules), "100%")
        summary_table.add_row("✅ Passed", str(summary.rules_passed), f"{summary.success_rate:.1f}%")
        summary_table.add_row("❌ Failed", str(summary.rules_failed), f"{(summary.rules_failed/summary.total_rules*100) if summary.total_rules > 0 else 0:.1f}%")
        summary_table.add_row("🔥 Errors", str(summary.rules_error), f"{(summary.rules_error/summary.total_rules*100) if summary.total_rules > 0 else 0:.1f}%")
        summary_table.add_row("⏭️  Skipped", str(summary.rules_skipped), f"{(summary.rules_skipped/summary.total_rules*100) if summary.total_rules > 0 else 0:.1f}%")
        
        console.print(summary_table)
        
        # Violations summary
        if summary.total_violations > 0:
            violations_table = Table(show_header=True, header_style="bold red")
            violations_table.add_column("Violation Level", style="cyan")
            violations_table.add_column("Count", style="white", justify="right")
            violations_table.add_column("Percentage", style="red", justify="right")
            
            violations_table.add_row("🔴 Critical", str(summary.critical_violations), f"{(summary.critical_violations/summary.total_violations*100) if summary.total_violations > 0 else 0:.1f}%")
            violations_table.add_row("🟠 Errors", str(summary.error_violations), f"{(summary.error_violations/summary.total_violations*100) if summary.total_violations > 0 else 0:.1f}%")
            violations_table.add_row("🟡 Warnings", str(summary.warning_violations), f"{(summary.warning_violations/summary.total_violations*100) if summary.total_violations > 0 else 0:.1f}%")
            violations_table.add_row("🔵 Info", str(summary.info_violations), f"{(summary.info_violations/summary.total_violations*100) if summary.total_violations > 0 else 0:.1f}%")
            violations_table.add_row("📊 Total", str(summary.total_violations), "100%")
            
            console.print("\n[bold red]🚨 Violations Summary[/bold red]")
            console.print(violations_table)
        
        # Detailed rule results
        if verbose or summary.has_errors or summary.has_critical_issues:
            console.print("\n[bold yellow]📋 Detailed Rule Results[/bold yellow]")
            
            # Show failed rules first, then others
            failed_results = [r for r in summary.results if not r.passed]
            passed_results = [r for r in summary.results if r.passed]
            
            results_to_show = failed_results + (passed_results if verbose else [])
            
            for result in results_to_show[:10]:  # Limit to first 10 rules
                status_icon = "✅" if result.passed else "❌"
                severity_icon = {
                    "critical": "🔴",
                    "error": "🟠", 
                    "warning": "🟡",
                    "info": "🔵"
                }.get(result.severity.value, "⚪")
                
                console.print(f"\n{status_icon} [bold]{result.rule_name}[/bold] {severity_icon}")
                console.print(f"   [dim]Type:[/dim] {result.rule_type.value} | [dim]Status:[/dim] {result.status.value}")
                console.print(f"   [dim]Message:[/dim] {result.message}")
                console.print(f"   [dim]Execution:[/dim] {result.execution_time_ms:.2f}ms | [dim]Rows:[/dim] {result.rows_evaluated:,}")
                
                if result.violations:
                    console.print(f"   [red]Violations ({len(result.violations)}):[/red]")
                    for violation in result.violations[:3]:  # Show first 3 violations
                        console.print(f"     • {violation.message}")
                        if violation.sample_values:
                            sample_display = ", ".join(str(v) for v in violation.sample_values[:3])
                            console.print(f"       [dim]Sample:[/dim] {sample_display}")
                        if violation.table_name:
                            location = violation.table_name
                            if violation.column_name:
                                location += f".{violation.column_name}"
                            console.print(f"       [dim]Location:[/dim] {location}")
                    
                    if len(result.violations) > 3:
                        console.print(f"     [dim]... and {len(result.violations) - 3} more violations[/dim]")
            
            if len(results_to_show) > 10:
                console.print(f"\n[dim]... and {len(results_to_show) - 10} more rules (use --verbose to see all)[/dim]")


def display_business_rule_results_json(results: list) -> None:
    """Display business rule validation results in JSON format."""
    import json
    from datetime import datetime
    
    output_data = {
        "export_info": {
            "timestamp": datetime.now().isoformat(),
            "sqltest_version": __version__,
            "validation_count": len(results)
        },
        "validation_results": []
    }
    
    for summary in results:
        result_data = {
            "rule_set_name": summary.rule_set_name,
            "validation_name": summary.validation_name,
            "execution_time_ms": summary.execution_time_ms,
            "start_time": summary.start_time.isoformat(),
            "end_time": summary.end_time.isoformat(),
            "total_rules": summary.total_rules,
            "rules_executed": summary.rules_executed,
            "rules_passed": summary.rules_passed,
            "rules_failed": summary.rules_failed,
            "rules_error": summary.rules_error,
            "rules_skipped": summary.rules_skipped,
            "success_rate": summary.success_rate,
            "has_critical_issues": summary.has_critical_issues,
            "has_errors": summary.has_errors,
            "total_violations": summary.total_violations,
            "critical_violations": summary.critical_violations,
            "error_violations": summary.error_violations,
            "warning_violations": summary.warning_violations,
            "info_violations": summary.info_violations,
            "validation_context": {
                "database_name": summary.validation_context.database_name,
                "schema_name": summary.validation_context.schema_name,
                "table_name": summary.validation_context.table_name,
                "query": summary.validation_context.query,
                "validation_timestamp": summary.validation_context.validation_timestamp.isoformat()
            },
            "rule_results": []
        }
        
        # Add individual rule results
        for result in summary.results:
            rule_data = {
                "rule_name": result.rule_name,
                "rule_type": result.rule_type.value,
                "status": result.status.value,
                "severity": result.severity.value,
                "scope": result.scope.value,
                "passed": result.passed,
                "message": result.message,
                "execution_time_ms": result.execution_time_ms,
                "rows_evaluated": result.rows_evaluated,
                "timestamp": result.timestamp.isoformat(),
                "violations": [
                    {
                        "violation_id": violation.violation_id,
                        "severity": violation.severity.value,
                        "message": violation.message,
                        "table_name": violation.table_name,
                        "column_name": violation.column_name,
                        "violation_count": violation.violation_count,
                        "sample_values": violation.sample_values,
                        "timestamp": violation.timestamp.isoformat()
                    }
                    for violation in result.violations
                ]
            }
            result_data["rule_results"].append(rule_data)
        
        output_data["validation_results"].append(result_data)
    
    console.print(json.dumps(output_data, indent=2, default=str))


def export_business_rule_results(results: list, output_path: str) -> None:
    """Export business rule validation results to JSON file."""
    import json
    from pathlib import Path
    from datetime import datetime
    
    output_data = {
        "export_info": {
            "timestamp": datetime.now().isoformat(),
            "sqltest_version": __version__,
            "validation_count": len(results)
        },
        "summary": {
            "total_rule_sets": len(results),
            "total_rules": sum(s.total_rules for s in results),
            "total_passed": sum(s.rules_passed for s in results),
            "total_failed": sum(s.rules_failed for s in results),
            "total_errors": sum(s.rules_error for s in results),
            "total_violations": sum(s.total_violations for s in results),
            "critical_violations": sum(s.critical_violations for s in results),
            "error_violations": sum(s.error_violations for s in results),
            "warning_violations": sum(s.warning_violations for s in results),
            "info_violations": sum(s.info_violations for s in results),
            "overall_success_rate": (sum(s.rules_passed for s in results) / sum(s.total_rules for s in results) * 100) if sum(s.total_rules for s in results) > 0 else 0
        },
        "validation_results": []
    }
    
    # Use the existing validator export functionality
    from sqltest.modules.business_rules import BusinessRuleValidator
    
    # Create temporary validator for export functionality
    temp_validator = BusinessRuleValidator(None)
    
    for summary in results:
        # Export each summary using the built-in method
        temp_file = Path(output_path).parent / f"temp_{summary.rule_set_name}.json"
        temp_validator.export_results_to_json(summary, str(temp_file))
        
        # Read and merge into main export
        with open(temp_file, 'r') as f:
            summary_data = json.load(f)
        
        output_data["validation_results"].append(summary_data)
        
        # Clean up temp file
        temp_file.unlink()
    
    # Write consolidated results
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path_obj, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)


if __name__ == "__main__":
    cli()

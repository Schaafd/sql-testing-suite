"""Business rule validation command implementation for SQLTest CLI."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence, Set

import click
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

from sqltest import __version__
from sqltest.cli.utils import console
from sqltest.config import get_config
from sqltest.db import get_connection_manager
from sqltest.exceptions import ConfigurationError, DatabaseError


@click.command(name="business-rules")
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
def business_rules_command(
    ctx: click.Context,
    rule_set: Optional[str],
    directory: Optional[str],
    table: Optional[str],
    query: Optional[str],
    database: Optional[str],
    schema: Optional[str],
    tags: Optional[str],
    parallel: Optional[bool],
    fail_fast: bool,
    max_workers: int,
    output: Optional[str],
    output_format: str,
    verbose: bool,
) -> None:
    """Execute business rule validation with rich reporting."""
    if not rule_set and not directory:
        console.print("[red]Error: Either --rule-set or --directory must be specified[/red]")
        console.print("\n[dim]Examples:[/dim]")
        console.print("  [cyan]sqltest business-rules --rule-set my_rules.yaml --database prod[/cyan]")
        console.print("  [cyan]sqltest business-rules --directory rules/ --table customers[/cyan]")
        console.print("  [cyan]sqltest business-rules --rule-set ecommerce_rules --tags data_quality[/cyan]")
        raise SystemExit(1)

    try:
        from sqltest.modules.business_rules import BusinessRuleValidator
    except ImportError as exc:  # pragma: no cover - defensive import guard
        console.print(f"[red]Business rule module unavailable: {exc}[/red]")
        raise SystemExit(1) from exc

    try:
        app_config = get_config(ctx.obj.get("config"))
        manager = get_connection_manager(app_config)
        db_name = database or ctx.obj.get("db") or app_config.default_database

        console.print(f"[bold blue]🔍 Business Rule Validation - Database: {db_name}[/bold blue]")
        if table:
            console.print(f"Table focus: [cyan]{table}[/cyan]")
        if query:
            console.print("Custom query override provided.")
        if schema:
            console.print(f"Schema: [cyan]{schema}[/cyan]")
        console.print()

        validator = BusinessRuleValidator(manager, max_workers=max_workers)

        loaded_rule_sets = _load_rule_sets(validator, rule_set, directory, ctx.obj.get("verbose", False) or verbose)
        if not loaded_rule_sets:
            console.print("[yellow]⚠️  No rule sets loaded[/yellow]")
            raise SystemExit(1)

        tag_filter = _parse_tag_filter(tags)
        if tag_filter:
            console.print(f"🏷️  Tag filter: [cyan]{', '.join(sorted(tag_filter))}[/cyan]")

        results = _execute_rule_sets(
            validator=validator,
            rule_sets=loaded_rule_sets,
            db_name=db_name,
            schema=schema,
            parallel=parallel,
            fail_fast=fail_fast,
            tags=tag_filter,
            verbose=ctx.obj.get("verbose", False) or verbose,
        )

        if not results:
            console.print("[yellow]No validation results to display[/yellow]")
            return

        console.print("\n" + "=" * 80)
        console.print("[bold blue]🔍 BUSINESS RULE VALIDATION RESULTS[/bold blue]")
        console.print("=" * 80)

        if output_format == "json":
            _display_results_json(results)
        else:
            _display_results_table(results, verbose=ctx.obj.get("verbose", False) or verbose)

        if output:
            _export_results(results, output)
            console.print(f"\n💾 Results exported to: [cyan]{output}[/cyan]")

        _render_outcome_summary(results)

    except ConfigurationError as exc:
        console.print(f"[red]Configuration Error: {exc}[/red]")
        raise SystemExit(1) from exc
    except DatabaseError as exc:
        console.print(f"[red]Database Error: {exc}[/red]")
        raise SystemExit(1) from exc
    except Exception as exc:  # pragma: no cover - defensive programming
        if ctx.obj.get("verbose") or verbose:
            import traceback

            console.print(f"[red]Error: {exc}[/red]")
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
        else:
            console.print(f"[red]Error: {exc}[/red]")
        raise SystemExit(1) from exc


def _load_rule_sets(
    validator,
    rule_set: Optional[str],
    directory: Optional[str],
    verbose: bool,
) -> Sequence[str]:
    """Load rule sets from provided sources with progress feedback."""
    loaded_rule_sets: list[str] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        load_task = progress.add_task("[green]Loading business rules...", total=100)

        try:
            if rule_set:
                _load_single_rule_set(validator, rule_set, loaded_rule_sets)

            if directory:
                loaded_names = validator.load_rule_sets_from_directory(directory, recursive=True)
                loaded_rule_sets.extend(loaded_names)
                console.print(f"📁 Loaded {len(loaded_names)} rule sets from directory: [green]{directory}[/green]")

            progress.update(load_task, completed=100)

        except Exception as exc:
            progress.update(load_task, completed=100)
            console.print(f"[red]❌ Failed to load rule sets: {exc}[/red]")
            if verbose:
                import traceback

                console.print(f"[dim]{traceback.format_exc()}[/dim]")
            raise SystemExit(1) from exc

    if loaded_rule_sets:
        console.print("\n📋 Available rule sets:")
        for rule_set_name in loaded_rule_sets:
            rule_set_obj = validator.get_rule_set(rule_set_name)
            enabled_count = len(rule_set_obj.get_enabled_rules())
            total_count = len(rule_set_obj.rules)
            console.print(f"  • [cyan]{rule_set_name}[/cyan]: {enabled_count}/{total_count} enabled rules")

    return loaded_rule_sets


def _load_single_rule_set(validator, rule_set: str, loaded_rule_sets: list[str]) -> None:
    from pathlib import Path as _Path

    rule_set_path = _Path(rule_set)
    if rule_set_path.exists():
        rule_set_name = validator.load_rule_set_from_file(str(rule_set_path))
        loaded_rule_sets.append(rule_set_name)
        console.print(f"📄 Loaded rule set from file: [green]{rule_set}[/green]")
        return

    if rule_set in validator.list_rule_sets():
        loaded_rule_sets.append(rule_set)
        console.print(f"📋 Using rule set: [green]{rule_set}[/green]")
        return

    console.print(f"[red]❌ Rule set '{rule_set}' not found[/red]")
    raise SystemExit(1)


def _parse_tag_filter(tags: Optional[str]) -> Optional[Set[str]]:
    if not tags:
        return None
    tag_filter = {tag.strip() for tag in tags.split(',') if tag.strip()}
    return tag_filter or None


def _execute_rule_sets(
    *,
    validator,
    rule_sets: Sequence[str],
    db_name: str,
    schema: Optional[str],
    parallel: Optional[bool],
    fail_fast: bool,
    tags: Optional[Set[str]],
    verbose: bool,
) -> list:
    results: list = []

    if not rule_sets:
        return results

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
        transient=True,
    ) as progress:
        for rule_set_name in rule_sets:
            task = progress.add_task(f"[green]Running {rule_set_name}...", total=100)
            try:
                start_time = time.time()
                summary = validator.validate_with_rule_set(
                    rule_set_name=rule_set_name,
                    database_name=db_name,
                    schema_name=schema,
                    parallel=parallel,
                    fail_fast=fail_fast,
                    tags=tags,
                )
                execution_time = time.time() - start_time
                progress.update(task, completed=100)
                results.append(summary)

                success_icon = "✅" if not summary.has_errors and not summary.has_critical_issues else "❌"
                console.print(
                    f"   {success_icon} Completed {rule_set_name} in {execution_time:.2f}s - "
                    f"{summary.rules_passed}/{summary.total_rules} rules passed"
                )

            except Exception as exc:
                progress.update(task, completed=100)
                console.print(f"   [red]❌ Failed to execute rule set {rule_set_name}: {exc}[/red]")
                if verbose:
                    import traceback

                    console.print(f"[dim]{traceback.format_exc()}[/dim]")
                if fail_fast:
                    raise SystemExit(1) from exc

    return results


def _display_results_table(results: Sequence, verbose: bool) -> None:
    for index, summary in enumerate(results, 1):
        if len(results) > 1:
            console.print(f"\n[bold cyan]📋 Rule Set {index}: {summary.rule_set_name}[/bold cyan]")

        summary_table = Table(show_header=True, header_style="bold magenta")
        summary_table.add_column("Metric", style="cyan", width=25)
        summary_table.add_column("Count", style="white", justify="right", width=10)
        summary_table.add_column("Percentage", style="green", justify="right", width=12)

        summary_table.add_row("Total Rules", str(summary.total_rules), "100%")
        summary_table.add_row("✅ Passed", str(summary.rules_passed), f"{summary.success_rate:.1f}%")
        summary_table.add_row(
            "❌ Failed",
            str(summary.rules_failed),
            f"{(summary.rules_failed / summary.total_rules * 100) if summary.total_rules else 0:.1f}%",
        )
        summary_table.add_row(
            "🔥 Errors",
            str(summary.rules_error),
            f"{(summary.rules_error / summary.total_rules * 100) if summary.total_rules else 0:.1f}%",
        )
        summary_table.add_row(
            "⏭️  Skipped",
            str(summary.rules_skipped),
            f"{(summary.rules_skipped / summary.total_rules * 100) if summary.total_rules else 0:.1f}%",
        )

        console.print(summary_table)

        if summary.total_violations:
            violations_table = Table(show_header=True, header_style="bold red")
            violations_table.add_column("Violation Type", style="yellow")
            violations_table.add_column("Count", style="white", justify="right")
            violations_table.add_column("Percentage", style="green", justify="right")

            violations_table.add_row(
                "🔴 Critical",
                str(summary.critical_violations),
                f"{(summary.critical_violations / summary.total_violations * 100) if summary.total_violations else 0:.1f}%",
            )
            violations_table.add_row(
                "🟠 Errors",
                str(summary.error_violations),
                f"{(summary.error_violations / summary.total_violations * 100) if summary.total_violations else 0:.1f}%",
            )
            violations_table.add_row(
                "🟡 Warnings",
                str(summary.warning_violations),
                f"{(summary.warning_violations / summary.total_violations * 100) if summary.total_violations else 0:.1f}%",
            )
            violations_table.add_row(
                "🔵 Info",
                str(summary.info_violations),
                f"{(summary.info_violations / summary.total_violations * 100) if summary.total_violations else 0:.1f}%",
            )
            violations_table.add_row("📊 Total", str(summary.total_violations), "100%")

            console.print("\n[bold red]🚨 Violations Summary[/bold red]")
            console.print(violations_table)

        if verbose or summary.has_errors or summary.has_critical_issues:
            _display_rule_details(summary, verbose)


def _display_rule_details(summary, verbose: bool) -> None:
    console.print("\n[bold yellow]📋 Detailed Rule Results[/bold yellow]")

    failed_results = [result for result in summary.results if not result.passed]
    passed_results = [result for result in summary.results if result.passed]
    results_to_show = failed_results + (passed_results if verbose else [])

    for result in results_to_show[:10]:
        status_icon = "✅" if result.passed else "❌"
        severity_icon = {
            "critical": "🔴",
            "error": "🟠",
            "warning": "🟡",
            "info": "🔵",
        }.get(result.severity.value, "⚪")

        console.print(f"\n{status_icon} [bold]{result.rule_name}[/bold] {severity_icon}")
        console.print(f"   [dim]Type:[/dim] {result.rule_type.value} | [dim]Status:[/dim] {result.status.value}")
        console.print(f"   [dim]Message:[/dim] {result.message}")
        console.print(
            f"   [dim]Execution:[/dim] {result.execution_time_ms:.2f}ms | "
            f"[dim]Rows:[/dim] {result.rows_evaluated:,}"
        )

        if result.violations:
            console.print(f"   [red]Violations ({len(result.violations)}):[/red]")
            for violation in result.violations[:3]:
                console.print(f"     • {violation.message}")
                if violation.sample_values:
                    sample_display = ", ".join(str(value) for value in violation.sample_values[:3])
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


def _display_results_json(results: Sequence) -> None:
    output_data = {
        "export_info": {
            "timestamp": datetime.now().isoformat(),
            "sqltest_version": __version__,
            "validation_count": len(results),
        },
        "validation_results": [],
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
                "validation_timestamp": summary.validation_context.validation_timestamp.isoformat(),
            },
            "rule_results": [],
        }

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
                        "timestamp": violation.timestamp.isoformat(),
                    }
                    for violation in result.violations
                ],
            }
            result_data["rule_results"].append(rule_data)

        output_data["validation_results"].append(result_data)

    console.print(json.dumps(output_data, indent=2, default=str))


def _export_results(results: Sequence, output_path: str) -> None:
    from sqltest.modules.business_rules import BusinessRuleValidator

    output_data = {
        "export_info": {
            "timestamp": datetime.now().isoformat(),
            "sqltest_version": __version__,
            "validation_count": len(results),
        },
        "summary": {
            "total_rule_sets": len(results),
            "total_rules": sum(summary.total_rules for summary in results),
            "total_passed": sum(summary.rules_passed for summary in results),
            "total_failed": sum(summary.rules_failed for summary in results),
            "total_errors": sum(summary.rules_error for summary in results),
            "total_violations": sum(summary.total_violations for summary in results),
            "critical_violations": sum(summary.critical_violations for summary in results),
            "error_violations": sum(summary.error_violations for summary in results),
            "warning_violations": sum(summary.warning_violations for summary in results),
            "info_violations": sum(summary.info_violations for summary in results),
            "overall_success_rate": (
                sum(summary.rules_passed for summary in results)
                / sum(summary.total_rules for summary in results)
                * 100
            )
            if sum(summary.total_rules for summary in results)
            else 0,
        },
        "validation_results": [],
    }

    temp_validator = BusinessRuleValidator(None)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    for summary in results:
        temp_file = destination.parent / f".{summary.rule_set_name}.tmp.json"
        temp_validator.export_results_to_json(summary, str(temp_file))

        with temp_file.open("r", encoding="utf-8") as handle:
            output_data["validation_results"].append(json.load(handle))

        temp_file.unlink(missing_ok=True)

    with destination.open("w", encoding="utf-8") as handle:
        json.dump(output_data, handle, indent=2, default=str)


def _render_outcome_summary(results: Sequence) -> None:
    total_critical = sum(summary.critical_violations for summary in results)
    total_errors = sum(summary.error_violations for summary in results)
    total_warnings = sum(summary.warning_violations for summary in results)

    if total_critical:
        console.print(f"\n[red]🔴 {total_critical} critical violation(s) found - immediate attention required![/red]")
        raise SystemExit(1)
    if total_errors:
        console.print(f"\n[red]❌ {total_errors} error violation(s) found[/red]")
        raise SystemExit(1)
    if total_warnings:
        console.print(f"\n[yellow]⚠️  {total_warnings} warning(s) found[/yellow]")
    else:
        console.print("\n[green]✅ All business rules passed successfully![/green]")

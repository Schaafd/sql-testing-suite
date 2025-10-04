"""Field validation command implementation for SQLTest CLI."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Iterable, Optional

import click
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

from sqltest.cli.utils import console
from sqltest.config import get_config
from sqltest.db import get_connection_manager
from sqltest.exceptions import ConfigurationError, DatabaseError, ValidationError


@click.command(name="validate")
@click.option("--config", type=click.Path(exists=True), help="Path to field validation configuration")
@click.option("--rule-set", help="Name of specific rule set to run")
@click.option("--database", "-d", help="Database to validate (default: default database)")
@click.option("--table", help="Specific table to validate")
@click.option("--schema", "-s", help="Schema to validate (database-specific)")
@click.option("--columns", help="Specific columns to validate (comma-separated)")
@click.option("--fail-fast", is_flag=True, help="Stop on first critical failure")
@click.option("--sample-size", type=int, help="Limit number of rows to validate")
@click.option("--output", "-o", type=click.Path(), help="Export results to JSON file")
@click.option("--generate", is_flag=True, help="Create sample field validation configuration")
@click.pass_context
def validate_command(
    ctx: click.Context,
    config: Optional[str],
    rule_set: Optional[str],
    database: Optional[str],
    table: Optional[str],
    schema: Optional[str],
    columns: Optional[str],
    fail_fast: bool,
    sample_size: Optional[int],
    output: Optional[str],
    generate: bool,
) -> None:
    """Run field validation against configured rule sets."""
    try:
        from sqltest.modules.field_validator import (
            NOT_NULL_RULE,
            TableFieldValidator,
            create_sample_config,
            save_sample_config,
        )
        from sqltest.modules.field_validator.models import ValidationRuleSet

        app_config = get_config(ctx.obj.get("config"))
        manager = get_connection_manager(app_config)
        db_name = database or ctx.obj.get("db") or app_config.default_database

        console.print(f"[bold green]🔍 Field Validation[/bold green]")
        console.print(f"Database: [cyan]{db_name}[/cyan]")
        if schema:
            console.print(f"Schema: [cyan]{schema}[/cyan]")
        if table:
            console.print(f"Table: [cyan]{table}[/cyan]")
        if columns:
            console.print(f"Columns: [cyan]{columns}[/cyan]")
        if sample_size:
            console.print(f"Sample Size: [yellow]{sample_size}[/yellow] rows")
        console.print()

        if generate:
            output_file = output or "field_validation_rules.yaml"
            output_path = Path(output_file)
            if output_path.exists() and not click.confirm(
                f"File '{output_file}' exists. Overwrite?"
            ):
                return
            save_sample_config(output_file)
            console.print(f"[green]✅ Sample field validation configuration created: {output_file}[/green]")
            console.print("\n[yellow]Next steps:[/yellow]")
            console.print("1. Edit the configuration file to match your validation needs")
            console.print(f"2. Run validation: [cyan]sqltest validate --config {output_file} --table your_table[/cyan]")
            return

        if not config and not table:
            console.print(
                "[red]Error: Either --config or --table must be specified (or use --generate for sample config)[/red]"
            )
            console.print("Use [cyan]sqltest validate --generate[/cyan] to create a sample configuration.")
            return

        if not table:
            console.print("[red]Error: --table must be specified for validation[/red]")
            return

        validator = TableFieldValidator(manager, strict_mode=fail_fast)

        if config:
            console.print(f"[blue]📋 Loading validation rules from: {config}[/blue]")
            try:
                validator.load_validation_rules(config)
            except Exception as exc:
                console.print(f"[red]❌ Error loading configuration: {exc}[/red]")
                return
        else:
            basic_rules = ValidationRuleSet(
                name="basic_validation",
                description="Basic data quality validation",
                rules=[NOT_NULL_RULE],
            )
            validator.add_rule_set(basic_rules)

        target_rule_sets = _resolve_rule_sets(validator, rule_set)
        column_list = [col.strip() for col in columns.split(",")] if columns else None

        results = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
            transient=False,
        ) as progress:
            for name in target_rule_sets:
                task = progress.add_task(f"[green]Validating with {name}...", total=100)
                try:
                    result = validator.validate_table_data(
                        table_name=table,
                        rule_set_name=name,
                        database_name=db_name,
                        sample_rows=sample_size,
                    )
                    results.append(result)
                except ValidationError as exc:
                    console.print(f"[red]❌ Validation failed for {name}: {exc}[/red]")
                    if fail_fast:
                        raise
                except Exception as exc:
                    console.print(f"[red]❌ Validation failed for {name}: {exc}[/red]")
                    if fail_fast:
                        raise
                finally:
                    progress.update(task, completed=100)

        if not results:
            console.print("[yellow]No validation results to display[/yellow]")
            return

        console.print("\n" + "=" * 80)
        console.print("[bold blue]🔍 FIELD VALIDATION RESULTS[/bold blue]")
        console.print("=" * 80)

        for result in results:
            _render_field_validation_result(result, verbose=ctx.obj.get("verbose", False))
            console.print("\n" + "-" * 60 + "\n")

        if len(results) > 1:
            _render_field_validation_summary(results)

        if output:
            _export_field_validation_results(results, output)

        if any(result.has_errors for result in results):
            console.print("\n[red]❌ Validation completed with errors[/red]")
            if fail_fast:
                raise SystemExit(1)
        else:
            console.print("\n[green]✅ All validations passed[/green]")

    except ConfigurationError as exc:
        console.print(f"[red]Configuration Error: {exc}[/red]")
        raise SystemExit(1) from exc
    except DatabaseError as exc:
        console.print(f"[red]Database Error: {exc}[/red]")
        raise SystemExit(1) from exc
    except Exception as exc:  # pragma: no cover - defensive programming
        if ctx.obj.get("verbose"):
            import traceback

            console.print(f"[red]Error: {exc}[/red]")
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
        else:
            console.print(f"[red]Error: {exc}[/red]")
        raise SystemExit(1) from exc


def _resolve_rule_sets(validator, rule_set: Optional[str]) -> list[str]:
    if rule_set:
        if rule_set not in validator.rule_sets:
            available = ", ".join(validator.rule_sets.keys())
            console.print(f"[red]Rule set '{rule_set}' not found. Available: {available}[/red]")
            raise SystemExit(1)
        return [rule_set]
    return list(validator.rule_sets.keys())


def _render_field_validation_result(result, verbose: bool) -> None:
    console.print(f"[bold cyan]📋 Validation Result: {result.table_name}[/bold cyan]")
    console.print(f"Database: [cyan]{result.database_name}[/cyan]")
    console.print(
        f"Total Rules: {result.total_rules} | Passed: [green]{result.passed_rules}[/green] | "
        f"Failed: [red]{result.failed_rules}[/red] | Warnings: [yellow]{result.warnings}[/yellow]"
    )

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Column", style="cyan")
    table.add_column("Passed", justify="right", style="green")
    table.add_column("Failed", justify="right", style="red")
    table.add_column("Warnings", justify="right", style="yellow")
    table.add_column("Success %", justify="right", style="blue")

    for field_result in result.field_results:
        table.add_row(
            field_result.column_name,
            str(field_result.passed_rules),
            str(field_result.failed_rules),
            str(field_result.warnings),
            f"{field_result.success_rate:.1f}%",
        )

    console.print(table)

    if verbose:
        _render_field_validation_details(result)


def _render_field_validation_details(result) -> None:
    from rich.table import Table as RichTable

    for field_result in result.field_results:
        if not field_result.has_errors and not field_result.has_warnings:
            continue

        console.print(f"\n[bold yellow]Details for column: {field_result.column_name}[/bold yellow]")
        detail_table = RichTable(show_header=True, header_style="bold magenta")
        detail_table.add_column("Rule", style="cyan")
        detail_table.add_column("Level", style="yellow")
        detail_table.add_column("Message", style="white")
        detail_table.add_column("Value", style="green")
        detail_table.add_column("Row", style="blue")

        for validation in field_result.validation_results:
            if validation.passed:
                continue
            detail_table.add_row(
                validation.rule_name,
                validation.level.value,
                validation.message,
                str(validation.value),
                str(validation.row_number or "-"),
            )

        console.print(detail_table)


def _render_field_validation_summary(results: Iterable) -> None:
    summary_table = Table(show_header=True, header_style="bold magenta")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Count", style="white", justify="right")
    summary_table.add_column("Percentage", style="green", justify="right")

    total_rules = sum(result.total_rules for result in results)
    failed_rules = sum(result.failed_rules for result in results)
    warnings = sum(result.warnings for result in results)

    summary_table.add_row("Total Rules", str(total_rules), "100%")
    summary_table.add_row(
        "✅ Passed",
        str(total_rules - failed_rules),
        f"{((total_rules - failed_rules)/total_rules*100) if total_rules else 0:.1f}%",
    )
    summary_table.add_row(
        "❌ Failed",
        str(failed_rules),
        f"{(failed_rules/total_rules*100) if total_rules else 0:.1f}%",
    )
    summary_table.add_row("⚠️  Warnings", str(warnings), "-")

    console.print(summary_table)


def _export_field_validation_results(results, output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = [dataclasses.asdict(result) for result in results]
    output.write_text(json.dumps(payload, indent=2, default=str))
    console.print(f"[green]📄 Results exported to: {output}[/green]")

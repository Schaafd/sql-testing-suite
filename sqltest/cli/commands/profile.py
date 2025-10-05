"""Profile command implementation for SQLTest CLI."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Iterable, Optional

import click
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from sqltest.cli.utils import console
from sqltest.config import get_config
from sqltest.db import get_connection_manager
from sqltest.exceptions import ConfigurationError, DatabaseError


@click.command(name="profile")
@click.option("--table", help="Table name to profile")
@click.option("--query", help="Custom SQL query to profile")
@click.option("--columns", help="Specific columns to profile (comma-separated)")
@click.option("--sample", type=int, default=10000, help="Sample size for analysis")
@click.option("--database", "-d", help="Database to use (default: default database)")
@click.option("--output", "-o", type=click.Path(), help="Export profile results to JSON file")
@click.option("--schema", "-s", help="Schema name (database-specific)")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json"]),
    default="table",
    help="Output format",
)
@click.pass_context
def profile_command(
    ctx: click.Context,
    table: Optional[str],
    query: Optional[str],
    columns: Optional[str],
    sample: int,
    database: Optional[str],
    output: Optional[str],
    schema: Optional[str],
    output_format: str,
) -> None:
    """Profile data for a table or adhoc query to capture quality metrics."""
    if not table and not query:
        console.print("[red]Error: Either --table or --query must be specified[/red]")
        return

    try:
        from sqltest.modules.profiler import DataProfiler

        config = get_config(ctx.obj.get("config"))
        manager = get_connection_manager(config)
        db_name = database or ctx.obj.get("db") or config.default_database

        console.print(f"[bold blue]📊 Data Profiling - Database: {db_name}[/bold blue]\n")

        profiler = DataProfiler(manager, sample_size=sample)

        if table:
            _profile_table(
                profiler,
                db_name=db_name,
                table=table,
                columns=columns,
                sample=sample,
                schema=schema,
                output=output,
                output_format=output_format,
            )
        else:
            _profile_query(
                profiler,
                db_name=db_name,
                query=query or "",
                sample=sample,
                output=output,
                output_format=output_format,
            )

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


def _profile_table(
    profiler,
    *,
    db_name: str,
    table: str,
    columns: Optional[str],
    sample: int,
    schema: Optional[str],
    output: Optional[str],
    output_format: str,
) -> None:
    console.print(f"📊 Profiling table: [green]{table}[/green]")
    if columns:
        console.print(f"Columns: [cyan]{columns}[/cyan]")
    console.print(f"Sample Size: [cyan]{sample:,}[/cyan]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("[green]Analyzing table data...", total=100)
        column_list = [col.strip() for col in columns.split(",")] if columns else None

        try:
            profile = profiler.profile_table(
                table_name=table,
                database_name=db_name,
                schema_name=schema,
                columns=column_list,
                sample_rows=sample,
            )
        finally:
            progress.update(task, completed=100)

    _render_profile(profile, output_format)

    if output:
        _export_profile(profile, output)


def _profile_query(
    profiler,
    *,
    db_name: str,
    query: str,
    sample: int,
    output: Optional[str],
    output_format: str,
) -> None:
    preview = query[:60] + ("..." if len(query) > 60 else "")
    console.print(f"📊 Profiling query: [green]{preview}[/green]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("[green]Executing and analyzing query...", total=100)

        try:
            profile = profiler.profile_query(query, database_name=db_name)
        finally:
            progress.update(task, completed=100)

    _render_query_profile(profile, output_format)

    if output:
        _export_query_profile(profile, output)


def _render_profile(profile, output_format: str) -> None:
    if output_format == "json":
        _print_json(profile)
        return

    console.print("\n[bold cyan]📋 TABLE OVERVIEW[/bold cyan]")
    overview_table = Table(show_header=False, box=None)
    overview_table.add_column("Property", style="yellow", width=20)
    overview_table.add_column("Value", style="white")
    overview_table.add_row("Table Name:", profile.table_name)
    overview_table.add_row("Database:", profile.database_name)
    if profile.schema_name:
        overview_table.add_row("Schema:", profile.schema_name)
    overview_table.add_row("Total Rows:", f"{profile.total_rows:,}")
    overview_table.add_row("Total Columns:", str(profile.total_columns))
    overview_table.add_row("Analysis Time:", f"{profile.execution_time:.2f}s")
    console.print(overview_table)

    console.print("\n[bold cyan]📈 DATA QUALITY SCORES[/bold cyan]")
    score_table = Table(show_header=False, box=None)
    score_table.add_column("Metric", style="yellow", width=20)
    score_table.add_column("Score", style="green")
    score_table.add_row("Completeness:", f"{profile.completeness_score:.1f}%")
    score_table.add_row("Uniqueness:", f"{profile.uniqueness_score:.1f}%")
    score_table.add_row("Validity:", f"{profile.validity_score:.1f}%")
    score_table.add_row("Consistency:", f"{profile.consistency_score:.1f}%")
    console.print(score_table)

    if profile.columns:
        console.print("\n[bold cyan]📊 COLUMN ANALYSIS SUMMARY[/bold cyan]")
        column_table = Table(show_header=True, header_style="bold magenta")
        column_table.add_column("Column", style="cyan")
        column_table.add_column("Type", style="green")
        column_table.add_column("Null %", style="yellow", justify="right")
        column_table.add_column("Unique %", style="blue", justify="right")
        column_table.add_column("Quality", style="white", justify="right")

        for column_name, stats in profile.columns.items():
            quality = max(0.0, 100.0 - stats.null_percentage)
            column_table.add_row(
                column_name,
                stats.data_type,
                f"{stats.null_percentage:.1f}%",
                f"{stats.unique_percentage:.1f}%",
                f"{quality:.1f}%",
            )

        console.print(column_table)

    patterns = _collect_patterns(profile.columns.values())
    if patterns:
        pattern_table = Table(show_header=True, header_style="bold magenta")
        pattern_table.add_column("Column", style="cyan")
        pattern_table.add_column("Pattern", style="green")
        pattern_table.add_column("Match %", style="yellow", justify="right")
        for column_name, name, confidence in patterns[:5]:
            pattern_table.add_row(column_name, name, f"{confidence:.1f}%")
        console.print("\n[bold cyan]🔍 TOP PATTERNS DETECTED[/bold cyan]")
        console.print(pattern_table)

    if profile.warnings or profile.recommendations:
        console.print("\n[bold cyan]⚠️  INSIGHTS & RECOMMENDATIONS[/bold cyan]")
        for warning in (profile.warnings or [])[:3]:
            console.print(f"[bold yellow]Warning:[/bold yellow] {warning}")
        for recommendation in (profile.recommendations or [])[:3]:
            console.print(f"[bold green]Recommendation:[/bold green] {recommendation}")


def _collect_patterns(columns: Iterable) -> list[tuple[str, str, float]]:
    patterns = []
    for column in columns:
        for pattern in (column.patterns or [])[:1]:
            patterns.append(
                (
                    column.column_name,
                    pattern.get("pattern_name", ""),
                    float(pattern.get("match_percentage", 0.0)),
                )
            )
    return patterns


def _render_query_profile(profile, output_format: str) -> None:
    if output_format == "json":
        _print_json(profile)
        return

    console.print("\n[bold cyan]📅 QUERY PROFILE[/bold cyan]")
    overview_table = Table(show_header=False, box=None)
    overview_table.add_column("Property", style="yellow", width=20)
    overview_table.add_column("Value", style="white")
    overview_table.add_row("Execution Time:", f"{profile.execution_time:.3f}s")
    overview_table.add_row("Rows Returned:", f"{profile.rows_returned:,}")
    overview_table.add_row("Columns:", str(profile.columns_returned))
    overview_table.add_row("Query Hash:", profile.query_hash[:16] + "...")
    console.print(overview_table)

    if profile.columns:
        column_table = Table(show_header=True, header_style="bold magenta")
        column_table.add_column("Column", style="cyan")
        column_table.add_column("Type", style="green")
        column_table.add_column("Nulls", style="yellow", justify="right")
        column_table.add_column("Unique", style="blue", justify="right")
        for column_name, stats in profile.columns.items():
            column_table.add_row(
                column_name,
                stats.data_type,
                f"{stats.null_percentage:.1f}%",
                f"{stats.unique_percentage:.1f}%",
            )
        console.print("\n[bold cyan]📝 RESULT COLUMNS ANALYSIS[/bold cyan]")
        console.print(column_table)


def _print_json(data) -> None:
    payload = dataclasses.asdict(data)
    console.print(json.dumps(payload, indent=2, default=str))


def _export_profile(profile, output_path: str) -> None:
    _export_json(profile, output_path, "Profile")


def _export_query_profile(profile, output_path: str) -> None:
    _export_json(profile, output_path, "Query profile")


def _export_json(data, output_path: str, label: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = dataclasses.asdict(data)
    output.write_text(json.dumps(payload, indent=2, default=str))
    console.print(f"\n[green]✓ {label} results exported to: {output}[/green]")

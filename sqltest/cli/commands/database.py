"""Database management CLI commands."""

from __future__ import annotations

from typing import Optional

import click
from rich.table import Table

from sqltest.cli.utils import console
from sqltest.config import get_config
from sqltest.db import get_connection_manager
from sqltest.exceptions import ConfigurationError, DatabaseError


@click.group(name="db")
@click.pass_context
def db_group(ctx: click.Context) -> None:
    """🗄️  Database connection management."""
    pass


@db_group.command(name="test")
@click.option("--database", "-d", help="Specific database to test (default: all)")
@click.pass_context
def test_connection_command(ctx: click.Context, database: Optional[str]) -> None:
    """Test database connections."""
    try:
        config = get_config(ctx.obj.get('config'))
        manager = get_connection_manager(config)

        console.print("[bold blue]Testing Database Connections[/bold blue]\n")

        if database:
            result = manager.test_connection(database)
            _show_connection_result(result)
        else:
            results = manager.test_all_connections()
            for db_name, result in results.items():
                _show_connection_result(result)
                console.print()
    except ConfigurationError as exc:
        console.print(f"[red]Configuration Error: {exc}[/red]")
        raise SystemExit(1) from exc
    except Exception as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise SystemExit(1) from exc


@db_group.command(name="status")
@click.pass_context
def status_command(ctx: click.Context) -> None:
    """Show database connection status."""
    try:
        config = get_config(ctx.obj.get('config'))
        manager = get_connection_manager(config)

        console.print("[bold blue]Database Connection Status[/bold blue]\n")

        status_info = manager.get_connection_status()
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Database", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Status", style="yellow")
        table.add_column("Default", style="blue")

        for db_name, conn_info in status_info['connections'].items():
            status_icon = "🟢 Active" if conn_info['active'] else "⚪ Inactive"
            is_default = "✓" if db_name == status_info['default_database'] else ""
            table.add_row(db_name, conn_info['type'], status_icon, is_default)

        console.print(table)
        console.print(
            f"\nTotal: {status_info['total_active']} active / {status_info['total_configured']} configured"
        )
    except ConfigurationError as exc:
        console.print(f"[red]Configuration Error: {exc}[/red]")
        raise SystemExit(1) from exc
    except Exception as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise SystemExit(1) from exc


@db_group.command(name="info")
@click.option("--database", "-d", help="Database to inspect (default: default database)")
@click.pass_context
def info_command(ctx: click.Context, database: Optional[str]) -> None:
    """Show database information."""
    try:
        config = get_config(ctx.obj.get('config'))
        manager = get_connection_manager(config)

        db_name = database or config.default_database
        console.print(f"[bold blue]Database Information: {db_name}[/bold blue]\n")

        info = manager.get_database_info(db_name)
        table = Table(show_header=False, box=None)
        table.add_column("Property", style="cyan", width=15)
        table.add_column("Value", style="green")
        for key, value in info.items():
            if value is not None:
                table.add_row(f"{key.replace('_', ' ').title()}:", str(value))
        console.print(table)
    except ConfigurationError as exc:
        console.print(f"[red]Configuration Error: {exc}[/red]")
        raise SystemExit(1) from exc
    except Exception as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise SystemExit(1) from exc


@db_group.command(name="tables")
@click.option("--database", "-d", help="Database to list tables from (default: default database)")
@click.option("--schema", "-s", help="Schema to list tables from (database-specific)")
@click.pass_context
def tables_command(ctx: click.Context, database: Optional[str], schema: Optional[str]) -> None:
    """List tables in database."""
    try:
        config = get_config(ctx.obj.get('config'))
        manager = get_connection_manager(config)

        db_name = database or config.default_database
        adapter = manager.get_adapter(db_name)
        schema_info = f" (schema: {schema})" if schema else ""
        console.print(f"[bold blue]Tables in {db_name}{schema_info}[/bold blue]\n")

        if hasattr(adapter, 'get_table_names'):
            tables_list = adapter.get_table_names(schema)
            if tables_list:
                table = Table(show_header=True, header_style="bold magenta")
                table.add_column("#", style="dim", width=4)
                table.add_column("Table Name", style="cyan")
                for i, table_name in enumerate(sorted(tables_list), start=1):
                    table.add_row(str(i), table_name)
                console.print(table)
                console.print(f"\n[dim]Total: {len(tables_list)} table(s)[/dim]")
            else:
                console.print("[yellow]No tables found[/yellow]")
        else:
            console.print(f"[yellow]Table listing not supported for {adapter.get_driver_name()}[/yellow]")
    except ConfigurationError as exc:
        console.print(f"[red]Configuration Error: {exc}[/red]")
        raise SystemExit(1) from exc
    except Exception as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise SystemExit(1) from exc


@db_group.command(name="views")
@click.option("--database", "-d", help="Database to list views from (default: default database)")
@click.option("--schema", "-s", help="Schema to list views from (database-specific)")
@click.pass_context
def views_command(ctx: click.Context, database: Optional[str], schema: Optional[str]) -> None:
    """List views in database."""
    try:
        config = get_config(ctx.obj.get('config'))
        manager = get_connection_manager(config)

        db_name = database or config.default_database
        adapter = manager.get_adapter(db_name)
        schema_info = f" (schema: {schema})" if schema else ""
        console.print(f"[bold blue]Views in {db_name}{schema_info}[/bold blue]\n")

        if hasattr(adapter, 'get_view_names'):
            views_list = adapter.get_view_names(schema)
            if views_list:
                table = Table(show_header=True, header_style="bold magenta")
                table.add_column("#", style="dim", width=4)
                table.add_column("View Name", style="cyan")
                for i, view_name in enumerate(sorted(views_list), start=1):
                    table.add_row(str(i), view_name)
                console.print(table)
                console.print(f"\n[dim]Total: {len(views_list)} view(s)[/dim]")
            else:
                console.print("[yellow]No views found[/yellow]")
        else:
            console.print(f"[yellow]View listing not supported for {adapter.get_driver_name()}[/yellow]")
    except ConfigurationError as exc:
        console.print(f"[red]Configuration Error: {exc}[/red]")
        raise SystemExit(1) from exc
    except Exception as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise SystemExit(1) from exc


@db_group.command(name="describe")
@click.argument('table_name')
@click.option("--database", "-d", help="Database to use (default: default database)")
@click.option("--schema", "-s", help="Schema name (database-specific)")
@click.pass_context
def describe_command(
    ctx: click.Context,
    table_name: str,
    database: Optional[str],
    schema: Optional[str],
) -> None:
    """Describe table structure and column metadata."""
    try:
        config = get_config(ctx.obj.get('config'))
        manager = get_connection_manager(config)

        db_name = database or config.default_database
        adapter = manager.get_adapter(db_name)
        table_info = adapter.get_table_info(table_name, schema)

        console.print(f"[bold blue]Table Structure: {table_info['table_name']}[/bold blue]")
        summary_parts = [f"Database: [cyan]{db_name}[/cyan]"]
        if table_info.get('schema'):
            summary_parts.append(f"Schema: [cyan]{table_info['schema']}[/cyan]")
        summary_parts.append(f"Rows: [green]{table_info.get('row_count', 0)}[/green]")
        console.print(" ".join(summary_parts) + "\n")

        console.print("[bold cyan]Column Details[/bold cyan]")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Name", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Nullable", style="yellow")
        table.add_column("Default", style="white")
        table.add_column("Primary Key", style="blue")

        for column in table_info.get('columns', []):
            table.add_row(
                str(column.get('column_name', '')),
                str(column.get('data_type', '')),
                "Yes" if str(column.get('is_nullable', '')).upper() in {'YES', 'TRUE', '1'} else "No",
                str(column.get('column_default', '') or ""),
                "✓" if column.get('primary_key') else "",
            )

        console.print(table)

    except ConfigurationError as exc:
        console.print(f"[red]Configuration Error: {exc}[/red]")
        raise SystemExit(1) from exc
    except DatabaseError as exc:
        console.print(f"[red]Database Error: {exc}[/red]")
        raise SystemExit(1) from exc
    except Exception as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise SystemExit(1) from exc


def _show_connection_result(result: dict) -> None:
    status_color = "green" if result['status'] == 'success' else "red"
    console.print(f"Database: [cyan]{result['database']}[/cyan]")
    console.print(f"Status: [{status_color}]{result['status'].upper()}[/{status_color}]")
    console.print(f"Message: {result.get('message', 'No message provided')}")
    console.print(f"Response Time: {result.get('response_time', 0)} ms")

"""Main CLI entry point for SQLTest Pro."""

import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich import print as rprint

from sqltest import __version__
from sqltest.config import get_config, create_sample_config
from sqltest.db import get_connection_manager
from sqltest.exceptions import ConfigurationError, DatabaseError

console = Console()


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


def show_dashboard() -> None:
    """Display the main interactive dashboard."""
    title = Text("SQLTest Pro", style="bold blue")
    subtitle = Text("A comprehensive SQL testing and validation suite", style="italic")
    
    dashboard_content = Text()
    dashboard_content.append("📊 Profile Data\n", style="bold")
    dashboard_content.append("✓  Run Validations\n", style="bold") 
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


@cli.command()
@click.option("--table", help="Table name to profile")
@click.option("--query", help="Custom SQL query to profile")
@click.option("--columns", help="Specific columns to profile (comma-separated)")
@click.option("--sample", type=int, default=10000, help="Sample size for analysis")
@click.option("--database", "-d", help="Database to use (default: default database)")
@click.pass_context
def profile(ctx: click.Context, table: str, query: str, columns: str, sample: int, database: str) -> None:
    """📊 Profile data in tables or queries."""
    if not table and not query:
        console.print("[red]Error: Either --table or --query must be specified[/red]")
        return
    
    try:
        config = get_config(ctx.obj.get('config'))
        manager = get_connection_manager(config)
        db_name = database or ctx.obj.get('db') or config.default_database
        
        console.print(f"[bold blue]Data Profiling - Database: {db_name}[/bold blue]\n")
        
        if table:
            profile_table(manager, db_name, table, columns, sample)
        elif query:
            profile_query(manager, db_name, query, sample)
            
    except ConfigurationError as e:
        console.print(f"[red]Configuration Error: {e}[/red]")
        sys.exit(1)
    except DatabaseError as e:
        console.print(f"[red]Database Error: {e}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


def profile_table(manager, db_name: str, table: str, columns: str, sample: int) -> None:
    """Profile a specific table."""
    console.print(f"📊 Profiling table: [green]{table}[/green]\n")
    
    # Get table information
    try:
        table_info = manager.get_table_info(table, db_name=db_name)
        
        # Show table overview
        console.print(f"[bold cyan]Table Overview[/bold cyan]")
        overview_table = Table(show_header=False, box=None)
        overview_table.add_column("Property", style="yellow", width=15)
        overview_table.add_column("Value", style="white")
        
        overview_table.add_row("Table Name:", table_info['table_name'])
        overview_table.add_row("Schema:", table_info.get('schema', 'N/A'))
        overview_table.add_row("Total Rows:", f"{table_info['row_count']:,}")
        overview_table.add_row("Total Columns:", str(len(table_info['columns'])))
        
        console.print(overview_table)
        console.print()
        
        # Show column information
        if table_info['columns']:
            console.print(f"[bold cyan]Column Information[/bold cyan]")
            
            columns_table = Table(show_header=True, header_style="bold magenta")
            columns_table.add_column("Column", style="cyan")
            columns_table.add_column("Type", style="green")
            columns_table.add_column("Nullable", style="yellow")
            columns_table.add_column("Default", style="blue")
            
            # Filter columns if specified
            column_filter = [c.strip() for c in columns.split(',')] if columns else None
            
            for col in table_info['columns']:
                col_name = col.get('column_name', col.get('name', 'Unknown'))
                
                if column_filter and col_name not in column_filter:
                    continue
                    
                data_type = col.get('data_type', col.get('type', 'Unknown'))
                nullable = col.get('is_nullable', 'Unknown')
                default = col.get('column_default', col.get('dflt_value', 'None'))
                
                # Format values
                nullable_display = "✓" if nullable == 'YES' or nullable == 1 else "✗" if nullable == 'NO' or nullable == 0 else str(nullable)
                default_display = str(default) if default else "None"
                
                columns_table.add_row(
                    col_name,
                    data_type,
                    nullable_display,
                    default_display[:30] + "..." if len(default_display) > 30 else default_display
                )
            
            console.print(columns_table)
            
        # Sample data preview
        if table_info['row_count'] > 0:
            console.print(f"\n[bold cyan]Sample Data (first 5 rows)[/bold cyan]")
            try:
                sample_query = f"SELECT * FROM {table} LIMIT 5"
                result = manager.execute_query(sample_query, db_name=db_name)
                
                if not result.is_empty:
                    data_table = Table(show_header=True, header_style="bold magenta")
                    
                    # Add columns
                    for col in result.columns:
                        data_table.add_column(col, style="white", max_width=20)
                    
                    # Add rows
                    for row in result.data.to_dict('records'):
                        formatted_row = []
                        for col in result.columns:
                            value = str(row.get(col, 'NULL'))
                            # Truncate long values
                            if len(value) > 18:
                                value = value[:15] + "..."
                            formatted_row.append(value)
                        data_table.add_row(*formatted_row)
                    
                    console.print(data_table)
                else:
                    console.print("[yellow]No sample data available[/yellow]")
            except Exception as e:
                console.print(f"[yellow]Could not fetch sample data: {e}[/yellow]")
        else:
            console.print("\n[yellow]Table is empty - no sample data available[/yellow]")
            
    except Exception as e:
        raise DatabaseError(f"Failed to profile table '{table}': {e}")


def profile_query(manager, db_name: str, query: str, sample: int) -> None:
    """Profile a custom query."""
    console.print(f"📊 Profiling query: [green]{query[:50]}{'...' if len(query) > 50 else ''}[/green]\n")
    
    try:
        # Execute the query with limit for sampling
        if sample > 0 and not query.upper().strip().startswith('SELECT'):
            console.print("[yellow]Warning: Sample size only applies to SELECT queries[/yellow]")
        
        # Add LIMIT if it's a SELECT query and doesn't already have one
        executed_query = query
        if (sample > 0 and query.upper().strip().startswith('SELECT') and 
            'LIMIT' not in query.upper() and 'TOP ' not in query.upper()):
            executed_query = f"{query.rstrip(';')} LIMIT {sample}"
        
        result = manager.execute_query(executed_query, db_name=db_name)
        
        # Show query results overview
        console.print(f"[bold cyan]Query Results Overview[/bold cyan]")
        overview_table = Table(show_header=False, box=None)
        overview_table.add_column("Property", style="yellow", width=15)
        overview_table.add_column("Value", style="white")
        
        overview_table.add_row("Rows Returned:", f"{result.row_count:,}")
        overview_table.add_row("Columns:", str(len(result.columns)))
        overview_table.add_row("Execution Time:", f"{result.execution_time:.3f}s")
        
        console.print(overview_table)
        console.print()
        
        # Show column information
        if result.columns:
            console.print(f"[bold cyan]Result Columns[/bold cyan]")
            
            columns_info = ", ".join([f"[cyan]{col}[/cyan]" for col in result.columns])
            console.print(columns_info)
            console.print()
        
        # Show sample results
        if not result.is_empty:
            console.print(f"[bold cyan]Query Results (first 10 rows)[/bold cyan]")
            
            data_table = Table(show_header=True, header_style="bold magenta")
            
            # Add columns
            for col in result.columns:
                data_table.add_column(col, style="white", max_width=25)
            
            # Add up to 10 rows
            for i, row in enumerate(result.data.to_dict('records')):
                if i >= 10:  # Limit to 10 rows for display
                    break
                    
                formatted_row = []
                for col in result.columns:
                    value = str(row.get(col, 'NULL'))
                    # Truncate long values
                    if len(value) > 22:
                        value = value[:19] + "..."
                    formatted_row.append(value)
                data_table.add_row(*formatted_row)
            
            console.print(data_table)
            
            if result.row_count > 10:
                console.print(f"\n[dim]... and {result.row_count - 10} more rows[/dim]")
        else:
            console.print("[yellow]Query returned no results[/yellow]")
            
    except Exception as e:
        raise DatabaseError(f"Failed to execute query: {e}")


@cli.command()
@click.option("--config", type=click.Path(exists=True), help="Path to validation configuration")
@click.option("--type", "validation_type", help="Type of validation to run")
@click.option("--table", help="Specific table to validate")
@click.pass_context
def validate(ctx: click.Context, config: str, validation_type: str, table: str) -> None:
    """✓ Run validation rules."""
    console.print("[bold green]Data Validation[/bold green]")
    
    if config:
        console.print(f"Using configuration: [green]{config}[/green]")
    if validation_type:
        console.print(f"Validation type: [green]{validation_type}[/green]")
    if table:
        console.print(f"Target table: [green]{table}[/green]")
    
    console.print("[yellow]⚠️  Validation module not yet implemented[/yellow]")


@cli.command()
@click.option("--config", type=click.Path(exists=True), help="Path to test configuration")
@click.option("--group", help="Specific test group to run")
@click.option("--coverage", is_flag=True, help="Generate coverage report")
@click.pass_context
def test(ctx: click.Context, config: str, group: str, coverage: bool) -> None:
    """🧪 Execute unit tests."""
    console.print("[bold magenta]SQL Unit Tests[/bold magenta]")
    
    if config:
        console.print(f"Using configuration: [green]{config}[/green]")
    if group:
        console.print(f"Test group: [green]{group}[/green]")
    if coverage:
        console.print("Coverage reporting: [green]enabled[/green]")
    
    console.print("[yellow]⚠️  Unit testing module not yet implemented[/yellow]")


@cli.command()
@click.option("--type", "report_type", type=click.Choice(["coverage", "validation", "profiling"]), help="Type of report to generate")
@click.option("--format", "output_format", type=click.Choice(["html", "json", "csv"]), default="html", help="Report format")
@click.option("--output", type=click.Path(), help="Output file path")
@click.pass_context
def report(ctx: click.Context, report_type: str, output_format: str, output: str) -> None:
    """📄 Generate reports."""
    console.print("[bold cyan]Report Generation[/bold cyan]")
    
    if report_type:
        console.print(f"Report type: [green]{report_type}[/green]")
    console.print(f"Format: [green]{output_format}[/green]")
    if output:
        console.print(f"Output: [green]{output}[/green]")
    
    console.print("[yellow]⚠️  Reporting module not yet implemented[/yellow]")


@cli.command()
@click.argument("project_name")
@click.option("--template", type=click.Choice(["basic", "advanced"]), default="basic", help="Project template")
def init(project_name: str, template: str) -> None:
    """🚀 Initialize a new SQLTest Pro project."""
    console.print(f"[bold green]Initializing project: {project_name}[/bold green]")
    console.print(f"Using template: [green]{template}[/green]")
    
    project_path = Path(project_name)
    if project_path.exists():
        console.print(f"[red]Error: Directory '{project_name}' already exists[/red]")
        return
    
    try:
        # Create project directory
        project_path.mkdir(parents=True)
        console.print(f"📁 Created project directory: [green]{project_path}[/green]")
        
        # Create sample configuration
        config_path = project_path / "sqltest.yaml"
        create_sample_config(config_path)
        console.print(f"⚙️  Created sample configuration: [green]{config_path}[/green]")
        
        # Create directories
        (project_path / "reports").mkdir()
        (project_path / "tests").mkdir()
        console.print("📂 Created project directories")
        
        console.print(f"\n[bold green]✅ Project '{project_name}' initialized successfully![/bold green]")
        console.print(f"\nNext steps:")
        console.print(f"1. Edit [cyan]{config_path}[/cyan] to configure your databases")
        console.print(f"2. Test your configuration: [cyan]sqltest config validate[/cyan]")
        console.print(f"3. Test database connection: [cyan]sqltest db test[/cyan]")
        
    except Exception as e:
        console.print(f"[red]Error creating project: {e}[/red]")
        

# Database management commands
@cli.group()
def db():
    """🗄️  Database connection management."""
    pass


@db.command()
@click.option("--database", "-d", help="Specific database to test (default: all)")
@click.pass_context
def test(ctx: click.Context, database: Optional[str]) -> None:
    """Test database connections."""
    try:
        config = get_config(ctx.obj.get('config'))
        manager = get_connection_manager(config)
        
        console.print("[bold blue]Testing Database Connections[/bold blue]\n")
        
        if database:
            # Test specific database
            result = manager.test_connection(database)
            show_connection_result(result)
        else:
            # Test all databases
            results = manager.test_all_connections()
            for db_name, result in results.items():
                show_connection_result(result)
                console.print()  # Empty line between results
                
    except ConfigurationError as e:
        console.print(f"[red]Configuration Error: {e}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@db.command()
@click.pass_context  
def status(ctx: click.Context) -> None:
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
        console.print(f"\nTotal: {status_info['total_active']} active / {status_info['total_configured']} configured")
        
    except ConfigurationError as e:
        console.print(f"[red]Configuration Error: {e}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@db.command()
@click.option("--database", "-d", help="Database to inspect (default: default database)")
@click.pass_context
def info(ctx: click.Context, database: Optional[str]) -> None:
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
        
    except ConfigurationError as e:
        console.print(f"[red]Configuration Error: {e}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@db.command()
@click.option("--database", "-d", help="Database to list tables from (default: default database)")
@click.option("--schema", "-s", help="Schema to list tables from (database-specific)")
@click.pass_context
def tables(ctx: click.Context, database: Optional[str], schema: Optional[str]) -> None:
    """List tables in database."""
    try:
        config = get_config(ctx.obj.get('config'))
        manager = get_connection_manager(config)
        
        db_name = database or config.default_database
        adapter = manager.get_adapter(db_name)
        
        schema_info = f" (schema: {schema})" if schema else ""
        console.print(f"[bold blue]Tables in {db_name}{schema_info}[/bold blue]\n")
        
        # Get table names using adapter-specific methods
        if hasattr(adapter, 'get_table_names'):
            tables_list = adapter.get_table_names(schema)
            
            if tables_list:
                table = Table(show_header=True, header_style="bold magenta")
                table.add_column("#", style="dim", width=4)
                table.add_column("Table Name", style="cyan")
                
                for i, table_name in enumerate(sorted(tables_list), 1):
                    table.add_row(str(i), table_name)
                
                console.print(table)
                console.print(f"\n[dim]Total: {len(tables_list)} table(s)[/dim]")
            else:
                console.print("[yellow]No tables found[/yellow]")
        else:
            console.print(f"[yellow]Table listing not supported for {adapter.get_driver_name()}[/yellow]")
        
    except ConfigurationError as e:
        console.print(f"[red]Configuration Error: {e}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@db.command()
@click.option("--database", "-d", help="Database to list views from (default: default database)")
@click.option("--schema", "-s", help="Schema to list views from (database-specific)")
@click.pass_context
def views(ctx: click.Context, database: Optional[str], schema: Optional[str]) -> None:
    """List views in database."""
    try:
        config = get_config(ctx.obj.get('config'))
        manager = get_connection_manager(config)
        
        db_name = database or config.default_database
        adapter = manager.get_adapter(db_name)
        
        schema_info = f" (schema: {schema})" if schema else ""
        console.print(f"[bold blue]Views in {db_name}{schema_info}[/bold blue]\n")
        
        # Get view names using adapter-specific methods
        if hasattr(adapter, 'get_view_names'):
            views_list = adapter.get_view_names(schema)
            
            if views_list:
                table = Table(show_header=True, header_style="bold magenta")
                table.add_column("#", style="dim", width=4)
                table.add_column("View Name", style="cyan")
                
                for i, view_name in enumerate(sorted(views_list), 1):
                    table.add_row(str(i), view_name)
                
                console.print(table)
                console.print(f"\n[dim]Total: {len(views_list)} view(s)[/dim]")
            else:
                console.print("[yellow]No views found[/yellow]")
        else:
            console.print(f"[yellow]View listing not supported for {adapter.get_driver_name()}[/yellow]")
        
    except ConfigurationError as e:
        console.print(f"[red]Configuration Error: {e}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@db.command()
@click.argument("table_name")
@click.option("--database", "-d", help="Database containing the table (default: default database)")
@click.option("--schema", "-s", help="Schema containing the table (database-specific)")
@click.pass_context
def describe(ctx: click.Context, table_name: str, database: Optional[str], schema: Optional[str]) -> None:
    """Describe table structure."""
    try:
        config = get_config(ctx.obj.get('config'))
        manager = get_connection_manager(config)
        
        db_name = database or config.default_database
        
        console.print(f"[bold blue]Table Structure: {table_name}[/bold blue]")
        console.print(f"Database: [cyan]{db_name}[/cyan]")
        if schema:
            console.print(f"Schema: [cyan]{schema}[/cyan]")
        console.print()
        
        table_info = manager.get_table_info(table_name, schema, db_name)
        
        # Show table overview
        overview_table = Table(show_header=False, box=None)
        overview_table.add_column("Property", style="yellow", width=15)
        overview_table.add_column("Value", style="white")
        
        overview_table.add_row("Table Name:", table_info['table_name'])
        overview_table.add_row("Schema:", table_info.get('schema', 'N/A'))
        overview_table.add_row("Total Rows:", f"{table_info['row_count']:,}")
        overview_table.add_row("Total Columns:", str(len(table_info['columns'])))
        
        console.print(overview_table)
        console.print()
        
        # Show column details
        if table_info['columns']:
            console.print(f"[bold cyan]Column Details[/bold cyan]")
            
            columns_table = Table(show_header=True, header_style="bold magenta")
            columns_table.add_column("Column", style="cyan")
            columns_table.add_column("Type", style="green")
            columns_table.add_column("Nullable", style="yellow")
            columns_table.add_column("Default", style="blue")
            columns_table.add_column("Extra", style="dim")
            
            for col in table_info['columns']:
                col_name = col.get('column_name', col.get('name', 'Unknown'))
                data_type = col.get('data_type', col.get('type', 'Unknown'))
                nullable = col.get('is_nullable', 'Unknown')
                default = col.get('column_default', col.get('dflt_value', 'None'))
                
                # Format values
                nullable_display = "✓" if nullable == 'YES' or nullable == 1 else "✗" if nullable == 'NO' or nullable == 0 else str(nullable)
                default_display = str(default) if default else "None"
                
                # Extra info (primary key, etc.)
                extra_info = []
                if col.get('primary_key'):
                    extra_info.append('PK')
                if col.get('column_key') == 'PRI':
                    extra_info.append('PK')
                if col.get('extra'):
                    extra_info.append(col['extra'])
                    
                extra_display = ", ".join(extra_info) if extra_info else ""
                
                columns_table.add_row(
                    col_name,
                    data_type,
                    nullable_display,
                    default_display[:20] + "..." if len(default_display) > 20 else default_display,
                    extra_display
                )
            
            console.print(columns_table)
        else:
            console.print("[yellow]No column information available[/yellow]")
        
    except ConfigurationError as e:
        console.print(f"[red]Configuration Error: {e}[/red]")
        sys.exit(1)
    except DatabaseError as e:
        console.print(f"[red]Database Error: {e}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


# Configuration management commands  
@cli.group()
def config_cmd():
    """⚙️  Configuration management."""
    pass

# Rename to avoid conflict with click context config
cli.add_command(config_cmd, name='config')


@config_cmd.command()
@click.argument("config_file", type=click.Path(exists=True))
def validate(config_file: str) -> None:
    """Validate configuration file."""
    try:
        config = get_config(config_file)
        console.print(f"[green]✅ Configuration file '{config_file}' is valid[/green]")
        console.print(f"Found {len(config.databases)} database(s): {', '.join(config.databases.keys())}")
        console.print(f"Default database: [cyan]{config.default_database}[/cyan]")
        
    except ConfigurationError as e:
        console.print(f"[red]❌ Configuration validation failed: {e}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@config_cmd.command()
@click.argument("output_file", type=click.Path())
def sample(output_file: str) -> None:
    """Create sample configuration file."""
    try:
        output_path = Path(output_file)
        if output_path.exists():
            click.confirm(f"File '{output_file}' exists. Overwrite?", abort=True)
            
        create_sample_config(output_path)
        console.print(f"[green]✅ Sample configuration created: {output_file}[/green]")
        console.print("\n[yellow]Next steps:[/yellow]")
        console.print("1. Edit the configuration file to match your database settings")
        console.print("2. Set required environment variables (e.g., DEV_DB_PASSWORD)")
        console.print(f"3. Validate: [cyan]sqltest config validate {output_file}[/cyan]")
        
    except Exception as e:
        console.print(f"[red]Error creating sample configuration: {e}[/red]")
        sys.exit(1)


def show_connection_result(result: dict) -> None:
    """Display connection test result in a formatted way."""
    db_name = result['database']
    status = result['status']
    
    if status == 'success':
        console.print(f"[green]✅ {db_name}[/green] - {result['message']}")
        console.print(f"   Type: [cyan]{result['database_type']}[/cyan]")
        console.print(f"   Driver: [blue]{result['driver']}[/blue]")
        console.print(f"   Response Time: [yellow]{result['response_time']}ms[/yellow]")
    else:
        console.print(f"[red]❌ {db_name}[/red] - {result['message']}")
        if 'error' in result:
            console.print(f"   Error: [red]{result['error']}[/red]")
        console.print(f"   Response Time: [yellow]{result['response_time']}ms[/yellow]")


if __name__ == "__main__":
    cli()

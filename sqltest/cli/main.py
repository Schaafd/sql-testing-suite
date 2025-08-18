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
@click.option("--config", type=click.Path(exists=True), help="Path to validation rule set configuration")
@click.option("--rule-set", help="Name of specific rule set to run")
@click.option("--database", "-d", help="Database to validate (default: default database)")
@click.option("--table", help="Specific table to validate")
@click.option("--schema", "-s", help="Schema to validate (database-specific)")
@click.option("--parallel", is_flag=True, help="Enable parallel rule execution")
@click.option("--fail-fast", is_flag=True, help="Stop on first critical failure")
@click.option("--tags", help="Filter rules by tags (comma-separated)")
@click.option("--output", "-o", type=click.Path(), help="Export results to JSON file")
@click.option("--generate", is_flag=True, help="Auto-generate basic data quality rules")
@click.pass_context
def validate(ctx: click.Context, config: str, rule_set: str, database: str, table: str, 
             schema: str, parallel: bool, fail_fast: bool, tags: str, output: str, 
             generate: bool) -> None:
    """✓ Run business rule validation.
    
    Validates data using business rules defined in YAML configuration files.
    Supports data quality, referential integrity, and custom business logic rules.
    """
    try:
        from sqltest.modules.business_rules import BusinessRuleValidator, quick_validate_table_data_quality
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
        import time as time_module
        
        # Get configuration and database manager
        app_config = get_config(ctx.obj.get('config'))
        manager = get_connection_manager(app_config)
        db_name = database or ctx.obj.get('db') or app_config.default_database
        
        console.print(f"[bold green]Business Rule Validation[/bold green]")
        console.print(f"Database: [cyan]{db_name}[/cyan]")
        if schema:
            console.print(f"Schema: [cyan]{schema}[/cyan]")
        if table:
            console.print(f"Table: [cyan]{table}[/cyan]")
        console.print()
        
        # Handle auto-generation mode
        if generate:
            if not table:
                console.print("[red]Error: --table must be specified when using --generate[/red]")
                return
            
            console.print("[yellow]🔧 Auto-generating data quality rules...[/yellow]")
            
            summary = quick_validate_table_data_quality(
                db_manager=manager,
                database_name=db_name,
                table_name=table,
                schema_name=schema
            )
            
            show_validation_summary(summary)
            
            if output:
                validator = BusinessRuleValidator(manager)
                validator.export_results_to_json(summary, output)
                console.print(f"[green]Results exported to: {output}[/green]")
            
            return
        
        # Load rule sets
        validator = BusinessRuleValidator(manager)
        
        if config:
            # Load from specific file
            console.print(f"[blue]📋 Loading rule set from: {config}[/blue]")
            rule_set_name = validator.load_rule_set_from_file(config)
            console.print(f"[green]✅ Loaded rule set: {rule_set_name}[/green]\n")
        else:
            # Try to find rule sets in current directory
            from pathlib import Path
            rule_files = list(Path('.').glob('*rules*.yaml')) + list(Path('.').glob('*validation*.yaml'))
            
            if rule_files:
                console.print(f"[blue]📋 Found rule configuration files:[/blue]")
                for file in rule_files:
                    try:
                        rule_set_name = validator.load_rule_set_from_file(file)
                        console.print(f"  [green]✅ {file} -> {rule_set_name}[/green]")
                    except Exception as e:
                        console.print(f"  [red]❌ {file} -> Error: {e}[/red]")
                console.print()
            else:
                console.print("[yellow]No rule configuration files found. Use --config to specify a file or --generate for auto-rules.[/yellow]")
                return
        
        # Get available rule sets
        available_rule_sets = validator.list_rule_sets()
        
        if not available_rule_sets:
            console.print("[red]No rule sets loaded. Please provide a valid configuration file.[/red]")
            return
        
        # Determine which rule set to run
        if rule_set:
            if rule_set not in available_rule_sets:
                console.print(f"[red]Rule set '{rule_set}' not found. Available: {', '.join(available_rule_sets)}[/red]")
                return
            target_rule_sets = [rule_set]
        else:
            target_rule_sets = available_rule_sets
        
        # Parse tags filter
        tags_set = None
        if tags:
            tags_set = {tag.strip() for tag in tags.split(',')}
            console.print(f"[blue]🏷️  Filtering rules by tags: {', '.join(tags_set)}[/blue]\n")
        
        # Run validation for each rule set
        all_summaries = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
            transient=True
        ) as progress:
            
            for rs_name in target_rule_sets:
                task = progress.add_task(f"[green]Validating with {rs_name}...", total=100)
                
                try:
                    # Run validation
                    summary = validator.validate_with_rule_set(
                        rule_set_name=rs_name,
                        database_name=db_name,
                        schema_name=schema,
                        parallel=parallel,
                        fail_fast=fail_fast,
                        tags=tags_set
                    )
                    
                    all_summaries.append(summary)
                    progress.update(task, completed=100)
                    
                except Exception as e:
                    progress.update(task, completed=100)
                    console.print(f"[red]❌ Validation failed for {rs_name}: {e}[/red]")
                    if fail_fast:
                        break
        
        console.print("\n" + "="*80)
        console.print("[bold blue]VALIDATION RESULTS[/bold blue]")
        console.print("="*80)
        
        # Show results for each rule set
        for summary in all_summaries:
            show_validation_summary(summary)
            console.print("\n" + "-"*60 + "\n")
        
        # Show aggregate statistics if multiple rule sets
        if len(all_summaries) > 1:
            stats = validator.get_validation_statistics(all_summaries)
            show_validation_statistics(stats)
        
        # Export results if requested
        if output:
            if len(all_summaries) == 1:
                validator.export_results_to_json(all_summaries[0], output)
            else:
                # Export aggregate results
                import json
                from pathlib import Path
                aggregate_data = {
                    "validation_count": len(all_summaries),
                    "statistics": stats if len(all_summaries) > 1 else {},
                    "summaries": []
                }
                
                for summary in all_summaries:
                    # Convert summary to dict format (simplified)
                    summary_data = {
                        "rule_set_name": summary.rule_set_name,
                        "execution_time_ms": summary.execution_time_ms,
                        "total_rules": summary.total_rules,
                        "rules_passed": summary.rules_passed,
                        "rules_failed": summary.rules_failed,
                        "success_rate": summary.success_rate,
                        "total_violations": summary.total_violations,
                        "has_critical_issues": summary.has_critical_issues
                    }
                    aggregate_data["summaries"].append(summary_data)
                
                output_path = Path(output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_path, 'w') as f:
                    json.dump(aggregate_data, f, indent=2, default=str)
            
            console.print(f"[green]📄 Results exported to: {output}[/green]")
        
        # Exit with error code if there were critical issues
        has_critical_issues = any(s.has_critical_issues for s in all_summaries)
        if has_critical_issues:
            console.print("\n[red]❌ Validation completed with critical issues[/red]")
            sys.exit(1)
        elif any(s.has_errors for s in all_summaries):
            console.print("\n[yellow]⚠️  Validation completed with errors[/yellow]")
        else:
            console.print("\n[green]✅ All validations passed successfully[/green]")
        
    except ConfigurationError as e:
        console.print(f"[red]Configuration Error: {e}[/red]")
        sys.exit(1)
    except Exception as e:
        if ctx.obj.get('verbose'):
            import traceback
            console.print(f"[red]Error: {e}[/red]")
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
        else:
            console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@cli.command()
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
def test(ctx: click.Context, config: str, directory: str, pattern: str, group: str, 
         test_name: str, database: str, parallel: bool, coverage: bool, fail_fast: bool, 
         verbose: bool, output: str) -> None:
    """🧪 Execute SQL unit tests.
    
    Runs SQL unit tests with fixtures, assertions, and coverage reporting.
    Supports test discovery from directories or specific configuration files.
    """
    try:
        from sqltest.modules.sql_testing import SQLTestRunner
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
        import time as time_module
        
        # Get configuration and database manager
        app_config = get_config(ctx.obj.get('config'))
        manager = get_connection_manager(app_config)
        db_name = database or ctx.obj.get('db') or app_config.default_database
        
        console.print(f"[bold magenta]SQL Unit Tests[/bold magenta]")
        console.print(f"Database: [cyan]{db_name}[/cyan]")
        if group:
            console.print(f"Group/Tag Filter: [cyan]{group}[/cyan]")
        if test_name:
            console.print(f"Test Filter: [cyan]{test_name}[/cyan]")
        console.print()
        
        # Initialize test runner
        test_runner = SQLTestRunner(manager)
        
        # Determine what to run
        if config:
            # Run specific test configuration file
            console.print(f"[blue]📋 Loading test configuration: {config}[/blue]")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True
            ) as progress:
                task = progress.add_task("[green]Loading test configuration...", total=100)
                
                try:
                    if test_name:
                        # Run single test
                        results = test_runner.run_test(
                            config, 
                            test_name, 
                            db_name, 
                            fail_fast=fail_fast
                        )
                        test_results = [results] if results else []
                    else:
                        # Run test suite
                        test_results = test_runner.run_test_suite(
                            config, 
                            db_name,
                            tag_filter=group,
                            parallel=parallel,
                            fail_fast=fail_fast,
                            enable_coverage=coverage
                        )
                    
                    progress.update(task, completed=100)
                    
                except Exception as e:
                    progress.update(task, completed=100)
                    console.print(f"[red]❌ Failed to run tests from {config}: {e}[/red]")
                    if ctx.obj.get('verbose') or verbose:
                        import traceback
                        console.print(f"[dim]{traceback.format_exc()}[/dim]")
                    sys.exit(1)
        
        elif directory:
            # Run test discovery in directory
            console.print(f"[blue]🔍 Discovering tests in: {directory}[/blue]")
            console.print(f"Pattern: [cyan]{pattern}[/cyan]")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
                transient=True
            ) as progress:
                task = progress.add_task("[green]Discovering and running tests...", total=100)
                
                try:
                    test_results = test_runner.run_tests_in_directory(
                        directory,
                        db_name,
                        pattern=pattern,
                        tag_filter=group,
                        parallel=parallel,
                        fail_fast=fail_fast,
                        enable_coverage=coverage
                    )
                    
                    progress.update(task, completed=100)
                    
                except Exception as e:
                    progress.update(task, completed=100)
                    console.print(f"[red]❌ Failed to run tests from directory {directory}: {e}[/red]")
                    if ctx.obj.get('verbose') or verbose:
                        import traceback
                        console.print(f"[dim]{traceback.format_exc()}[/dim]")
                    sys.exit(1)
        
        else:
            # Try to discover tests in current directory
            from pathlib import Path
            current_dir = Path('.')
            test_files = list(current_dir.glob(pattern)) + list(current_dir.glob('tests/*.yaml'))
            
            if test_files:
                console.print(f"[blue]🔍 Found test files in current directory:[/blue]")
                for test_file in test_files:
                    console.print(f"  [green]• {test_file}[/green]")
                console.print()
                
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    console=console,
                    transient=True
                ) as progress:
                    task = progress.add_task("[green]Running discovered tests...", total=100)
                    
                    try:
                        test_results = test_runner.run_tests_in_directory(
                            str(current_dir),
                            db_name,
                            pattern=pattern,
                            tag_filter=group,
                            parallel=parallel,
                            fail_fast=fail_fast,
                            enable_coverage=coverage
                        )
                        
                        progress.update(task, completed=100)
                        
                    except Exception as e:
                        progress.update(task, completed=100)
                        console.print(f"[red]❌ Failed to run discovered tests: {e}[/red]")
                        if ctx.obj.get('verbose') or verbose:
                            import traceback
                            console.print(f"[dim]{traceback.format_exc()}[/dim]")
                        sys.exit(1)
            else:
                console.print("[yellow]No test files found. Use --config or --directory to specify tests.[/yellow]")
                console.print("\n[dim]Expected patterns:[/dim]")
                console.print(f"  [dim]• {pattern}[/dim]")
                console.print(f"  [dim]• tests/*.yaml[/dim]")
                return
        
        # Display results
        console.print("\n" + "="*80)
        console.print("[bold magenta]TEST RESULTS[/bold magenta]")
        console.print("="*80)
        
        if not test_results:
            console.print("[yellow]No test results to display[/yellow]")
            return
        
        # Show individual test results
        for test_result in test_results:
            show_test_result(test_result, verbose or ctx.obj.get('verbose', False))
            console.print("\n" + "-"*60 + "\n")
        
        # Show aggregate statistics
        if len(test_results) > 1:
            show_test_statistics(test_results)
        
        # Show coverage report if enabled
        if coverage and test_results:
            show_coverage_report(test_results)
        
        # Export results if requested
        if output:
            export_test_results(test_results, output, coverage)
            console.print(f"[green]📄 Test results exported to: {output}[/green]")
        
        # Exit with appropriate code
        failed_tests = sum(1 for result in test_results if not result.passed)
        total_tests = len(test_results)
        
        if failed_tests > 0:
            console.print(f"\n[red]❌ {failed_tests}/{total_tests} tests failed[/red]")
            sys.exit(1)
        else:
            console.print(f"\n[green]✅ All {total_tests} tests passed[/green]")
        
    except ConfigurationError as e:
        console.print(f"[red]Configuration Error: {e}[/red]")
        sys.exit(1)
    except Exception as e:
        if ctx.obj.get('verbose') or verbose:
            import traceback
            console.print(f"[red]Error: {e}[/red]")
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
        else:
            console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


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


if __name__ == "__main__":
    cli()

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


@cli.command()
@click.option("--table", help="Table name to profile")
@click.option("--query", help="Custom SQL query to profile")
@click.option("--columns", help="Specific columns to profile (comma-separated)")
@click.option("--sample", type=int, default=10000, help="Sample size for analysis")
@click.option("--database", "-d", help="Database to use (default: default database)")
@click.option("--output", "-o", type=click.Path(), help="Export profile results to JSON file")
@click.option("--schema", "-s", help="Schema name (database-specific)")
@click.option("--format", "output_format", type=click.Choice(["table", "json"]), default="table", help="Output format")
@click.pass_context
def profile(ctx: click.Context, table: str, query: str, columns: str, sample: int, database: str,
           output: str, schema: str, output_format: str) -> None:
    """📊 Profile data in tables or queries with comprehensive statistical analysis.
    
    Generates detailed data profiles including:
    • Data quality scores (completeness, uniqueness, validity, consistency)
    • Statistical analysis (mean, median, quartiles, outliers)
    • Pattern detection (emails, phones, URLs, etc.)
    • Data type inference and validation
    • Value frequency analysis
    • Recommendations for data quality improvements
    """
    if not table and not query:
        console.print("[red]Error: Either --table or --query must be specified[/red]")
        return
    
    try:
        from sqltest.modules.profiler import DataProfiler
        from rich.progress import Progress, SpinnerColumn, TextColumn
        import json
        from pathlib import Path
        
        config = get_config(ctx.obj.get('config'))
        manager = get_connection_manager(config)
        db_name = database or ctx.obj.get('db') or config.default_database
        
        console.print(f"[bold blue]📊 Data Profiling - Database: {db_name}[/bold blue]\n")
        
        # Initialize profiler
        profiler = DataProfiler(manager, sample_size=sample)
        
        if table:
            profile_table_advanced(profiler, db_name, table, columns, sample, schema, output, output_format)
        elif query:
            profile_query_advanced(profiler, db_name, query, sample, output, output_format)
            
    except ConfigurationError as e:
        console.print(f"[red]Configuration Error: {e}[/red]")
        sys.exit(1)
    except DatabaseError as e:
        console.print(f"[red]Database Error: {e}[/red]")
        sys.exit(1)
    except Exception as e:
        if ctx.obj.get('verbose'):
            import traceback
            console.print(f"[red]Error: {e}[/red]")
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
        else:
            console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


def profile_table_advanced(profiler, db_name: str, table: str, columns: str, sample: int, 
                          schema: str, output: str, output_format: str) -> None:
    """Profile a specific table using advanced DataProfiler."""
    from rich.progress import Progress, SpinnerColumn, TextColumn
    import json
    from pathlib import Path
    import dataclasses
    
    console.print(f"📊 Profiling table: [green]{table}[/green]")
    if columns:
        console.print(f"Columns: [cyan]{columns}[/cyan]")
    console.print(f"Sample Size: [cyan]{sample:,}[/cyan]")
    console.print()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task("[green]Analyzing table data...", total=100)
        
        try:
            # Parse column filter
            column_list = [c.strip() for c in columns.split(',')] if columns else None
            
            # Run profiling
            profile = profiler.profile_table(
                table_name=table,
                database_name=db_name,
                schema_name=schema,
                columns=column_list,
                sample_rows=sample
            )
            
            progress.update(task, completed=100)
            
        except Exception as e:
            progress.update(task, completed=100)
            raise e
    
    # Display results based on format
    if output_format == "json":
        display_profile_json(profile)
    else:
        display_profile_table(profile)
    
    # Export to file if requested
    if output:
        export_profile_results(profile, output)


def profile_query_advanced(profiler, db_name: str, query: str, sample: int, 
                          output: str, output_format: str) -> None:
    """Profile a custom query using advanced DataProfiler."""
    from rich.progress import Progress, SpinnerColumn, TextColumn
    
    console.print(f"📊 Profiling query: [green]{query[:60]}{'...' if len(query) > 60 else ''}[/green]")
    console.print()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task("[green]Executing and analyzing query...", total=100)
        
        try:
            # Run query profiling
            profile = profiler.profile_query(query, database_name=db_name)
            
            progress.update(task, completed=100)
            
        except Exception as e:
            progress.update(task, completed=100)
            raise e
    
    # Display results based on format
    if output_format == "json":
        display_query_profile_json(profile)
    else:
        display_query_profile_table(profile)
    
    # Export to file if requested
    if output:
        export_query_profile_results(profile, output)


def display_profile_table(profile) -> None:
    """Display table profile results in Rich table format."""
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    
    # Table overview
    console.print(f"\n[bold cyan]📋 TABLE OVERVIEW[/bold cyan]")
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
    
    # Data quality scores
    console.print(f"\n[bold cyan]📈 DATA QUALITY SCORES[/bold cyan]")
    scores_table = Table(show_header=False, box=None)
    scores_table.add_column("Metric", style="yellow", width=20)
    scores_table.add_column("Score", style="green")
    
    scores_table.add_row("Completeness:", f"{profile.completeness_score:.1f}%")
    scores_table.add_row("Uniqueness:", f"{profile.uniqueness_score:.1f}%")
    scores_table.add_row("Validity:", f"{profile.validity_score:.1f}%")
    scores_table.add_row("Consistency:", f"{profile.consistency_score:.1f}%")
    scores_table.add_row("Overall Quality:", f"[bold]{profile.overall_score:.1f}%[/bold]")
    
    console.print(scores_table)
    
    # Column analysis summary
    console.print(f"\n[bold cyan]📝 COLUMN ANALYSIS SUMMARY[/bold cyan]")
    columns_table = Table(show_header=True, header_style="bold magenta")
    columns_table.add_column("Column", style="cyan")
    columns_table.add_column("Type", style="green")
    columns_table.add_column("Nulls", style="yellow", justify="right")
    columns_table.add_column("Unique", style="blue", justify="right")
    columns_table.add_column("Quality", style="white", justify="right")
    
    for column_name, stats in profile.columns.items():
        # Calculate column quality score (simplified)
        quality = 100 - stats.null_percentage
        quality_display = f"{quality:.1f}%"
        
        columns_table.add_row(
            column_name,
            stats.data_type,
            f"{stats.null_percentage:.1f}%",
            f"{stats.unique_percentage:.1f}%",
            quality_display
        )
    
    console.print(columns_table)
    
    # Show top patterns detected
    patterns_found = []
    for column_name, stats in profile.columns.items():
        if stats.patterns:
            for pattern in stats.patterns[:1]:  # Top pattern per column
                patterns_found.append({
                    'column': column_name,
                    'pattern': pattern['pattern_name'],
                    'confidence': pattern['match_percentage']
                })
    
    if patterns_found:
        console.print(f"\n[bold cyan]🔍 TOP PATTERNS DETECTED[/bold cyan]")
        patterns_table = Table(show_header=True, header_style="bold magenta")
        patterns_table.add_column("Column", style="cyan")
        patterns_table.add_column("Pattern", style="green")
        patterns_table.add_column("Match %", style="yellow", justify="right")
        
        for pattern in patterns_found[:5]:  # Show top 5 patterns
            patterns_table.add_row(
                pattern['column'],
                pattern['pattern'],
                f"{pattern['confidence']:.1f}%"
            )
        
        console.print(patterns_table)
    
    # Warnings and recommendations
    if profile.warnings or profile.recommendations:
        console.print(f"\n[bold cyan]⚠️  INSIGHTS & RECOMMENDATIONS[/bold cyan]")
        
        if profile.warnings:
            console.print("[bold yellow]Warnings:[/bold yellow]")
            for warning in profile.warnings[:3]:  # Show top 3 warnings
                console.print(f"  • {warning}")
        
        if profile.recommendations:
            console.print("[bold green]Recommendations:[/bold green]")
            for rec in profile.recommendations[:3]:  # Show top 3 recommendations
                console.print(f"  • {rec}")


def display_query_profile_table(profile) -> None:
    """Display query profile results in Rich table format."""
    from rich.table import Table
    
    # Query overview
    console.print(f"\n[bold cyan]📅 QUERY PROFILE[/bold cyan]")
    overview_table = Table(show_header=False, box=None)
    overview_table.add_column("Property", style="yellow", width=20)
    overview_table.add_column("Value", style="white")
    
    overview_table.add_row("Execution Time:", f"{profile.execution_time:.3f}s")
    overview_table.add_row("Rows Returned:", f"{profile.rows_returned:,}")
    overview_table.add_row("Columns:", str(profile.columns_returned))
    overview_table.add_row("Query Hash:", profile.query_hash[:16] + "...")
    
    console.print(overview_table)
    
    # Column analysis
    if profile.columns:
        console.print(f"\n[bold cyan]📝 RESULT COLUMNS ANALYSIS[/bold cyan]")
        columns_table = Table(show_header=True, header_style="bold magenta")
        columns_table.add_column("Column", style="cyan")
        columns_table.add_column("Type", style="green")
        columns_table.add_column("Nulls", style="yellow", justify="right")
        columns_table.add_column("Unique", style="blue", justify="right")
        
        for column_name, stats in profile.columns.items():
            columns_table.add_row(
                column_name,
                stats.data_type,
                f"{stats.null_percentage:.1f}%",
                f"{stats.unique_percentage:.1f}%"
            )
        
        console.print(columns_table)


def display_profile_json(profile) -> None:
    """Display profile results as JSON."""
    import json
    import dataclasses
    
    profile_dict = dataclasses.asdict(profile)
    json_output = json.dumps(profile_dict, indent=2, default=str)
    console.print(json_output)


def display_query_profile_json(profile) -> None:
    """Display query profile results as JSON."""
    import json
    import dataclasses
    
    profile_dict = dataclasses.asdict(profile)
    json_output = json.dumps(profile_dict, indent=2, default=str)
    console.print(json_output)


def export_profile_results(profile, output_path: str) -> None:
    """Export profile results to file."""
    import json
    import dataclasses
    from pathlib import Path
    
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    
    profile_dict = dataclasses.asdict(profile)
    
    with open(output, 'w') as f:
        json.dump(profile_dict, f, indent=2, default=str)
    
    console.print(f"\n[green]✓ Profile results exported to: {output}[/green]")


def export_query_profile_results(profile, output_path: str) -> None:
    """Export query profile results to file."""
    import json
    import dataclasses
    from pathlib import Path
    
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    
    profile_dict = dataclasses.asdict(profile)
    
    with open(output, 'w') as f:
        json.dump(profile_dict, f, indent=2, default=str)
    
    console.print(f"\n[green]✓ Query profile results exported to: {output}[/green]")


@cli.command()
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
def validate(ctx: click.Context, config: str, rule_set: str, database: str, table: str, 
             schema: str, columns: str, fail_fast: bool, sample_size: int, output: str, 
             generate: bool) -> None:
    """✓ Run field validation.
    
    Validates data using field validation rules defined in YAML configuration files.
    Supports data quality checks, format validation, range validation, and custom rules.
    """
    try:
        from sqltest.modules.field_validator import (
            TableFieldValidator, save_sample_config, create_sample_config,
            ValidationLevel
        )
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
        import json
        from pathlib import Path
        
        # Get configuration and database manager
        app_config = get_config(ctx.obj.get('config'))
        manager = get_connection_manager(app_config)
        db_name = database or ctx.obj.get('db') or app_config.default_database
        
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
        
        # Handle sample configuration generation
        if generate:
            output_file = output or "field_validation_rules.yaml"
            output_path = Path(output_file)
            
            if output_path.exists():
                if not click.confirm(f"File '{output_file}' exists. Overwrite?"):
                    return
            
            save_sample_config(output_file)
            console.print(f"[green]✅ Sample field validation configuration created: {output_file}[/green]")
            console.print("\n[yellow]Next steps:[/yellow]")
            console.print("1. Edit the configuration file to match your validation needs")
            console.print(f"2. Run validation: [cyan]sqltest validate --config {output_file} --table your_table[/cyan]")
            return
        
        # Require either config file or table for validation
        if not config and not table:
            console.print("[red]Error: Either --config or --table must be specified (or use --generate for sample config)[/red]")
            console.print("Use [cyan]sqltest validate --generate[/cyan] to create a sample configuration.")
            return
        
        # Create field validator
        validator = TableFieldValidator(manager, strict_mode=fail_fast)
        
        # Load validation rules
        if config:
            console.print(f"[blue]📋 Loading validation rules from: {config}[/blue]")
            try:
                validator.load_validation_rules(config)
                rule_sets = list(validator.rule_sets.keys())
                console.print(f"[green]✅ Loaded {len(rule_sets)} rule set(s): {', '.join(rule_sets)}[/green]\n")
            except Exception as e:
                console.print(f"[red]❌ Error loading configuration: {e}[/red]")
                return
        else:
            # No config provided, use basic validation rules for the table
            console.print("[yellow]📋 No configuration file provided, using basic validation rules[/yellow]")
            from sqltest.modules.field_validator.models import ValidationRuleSet
            from sqltest.modules.field_validator import NOT_NULL_RULE
            
            basic_rules = ValidationRuleSet(
                name="basic_validation",
                description="Basic data quality validation",
                rules=[NOT_NULL_RULE]
            )
            validator.add_rule_set(basic_rules)
            rule_sets = ["basic_validation"]
            console.print(f"[green]✅ Using basic validation rules[/green]\n")
        
        # Determine which rule set to use
        if rule_set:
            if rule_set not in validator.rule_sets:
                console.print(f"[red]Rule set '{rule_set}' not found. Available: {', '.join(validator.rule_sets.keys())}[/red]")
                return
            target_rule_sets = [rule_set]
        else:
            target_rule_sets = list(validator.rule_sets.keys())
        
        # Validate that we have a table to work with
        if not table:
            console.print("[red]Error: --table must be specified for validation[/red]")
            return
        
        # Parse columns filter
        column_list = None
        if columns:
            column_list = [col.strip() for col in columns.split(',')]
            console.print(f"[blue]📋 Validating specific columns: {', '.join(column_list)}[/blue]\n")
        
        # Run validation
        all_results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
            transient=False
        ) as progress:
            
            for rule_set_name in target_rule_sets:
                task = progress.add_task(f"[green]Validating with {rule_set_name}...", total=100)
                
                try:
                    # Run field validation
                    result = validator.validate_table_data(
                        table_name=table,
                        rule_set_name=rule_set_name,
                        database_name=db_name,
                        sample_rows=sample_size
                    )
                    
                    all_results.append(result)
                    progress.update(task, completed=100)
                    
                except Exception as e:
                    progress.update(task, completed=100)
                    console.print(f"[red]❌ Validation failed for {rule_set_name}: {e}[/red]")
                    if fail_fast:
                        return
        
        # Display results
        console.print("\n" + "="*80)
        console.print("[bold blue]🔍 FIELD VALIDATION RESULTS[/bold blue]")
        console.print("="*80)
        
        for result in all_results:
            show_field_validation_result(result, verbose=ctx.obj.get('verbose', False))
            console.print("\n" + "-"*60 + "\n")
        
        # Show aggregate summary
        if len(all_results) > 1:
            show_field_validation_summary(all_results)
        
        # Export results if requested
        if output:
            export_field_validation_results(all_results, output)
            console.print(f"[green]📄 Results exported to: {output}[/green]")
        
        # Exit with appropriate code based on validation results
        has_errors = any(result.has_errors for result in all_results)
        if has_errors:
            console.print(f"\n[red]❌ Validation completed with errors[/red]")
            if fail_fast:
                sys.exit(1)
        else:
            console.print(f"\n[green]✅ All validations passed[/green]")
        
    except ConfigurationError as e:
        console.print(f"[red]Configuration Error: {e}[/red]")
        sys.exit(1)
    except DatabaseError as e:
        console.print(f"[red]Database Error: {e}[/red]")
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
@click.option("--type", "report_type", type=click.Choice(["coverage", "validation", "profiling", "testing", "dashboard"]), help="Type of report to generate")
@click.option("--format", "output_format", type=click.Choice(["html", "json", "csv"]), default="html", help="Report format")
@click.option("--input", "input_file", type=click.Path(exists=True), help="Input file (JSON results from previous runs)")
@click.option("--output", type=click.Path(), help="Output file path")
@click.option("--config", type=click.Path(exists=True), help="Configuration file for report settings")
@click.option("--database", "-d", help="Database to generate reports for")
@click.option("--table", help="Specific table to report on")
@click.option("--title", help="Custom report title")
@click.option("--include-charts", is_flag=True, help="Include charts in HTML reports")
@click.option("--template", type=click.Path(exists=True), help="Custom HTML template for reports")
@click.pass_context
def report(ctx: click.Context, report_type: str, output_format: str, input_file: str, 
           output: str, config: str, database: str, table: str, title: str, 
           include_charts: bool, template: str) -> None:
    """📄 Generate comprehensive reports.
    
    Creates detailed reports from validation results, test results, profiling data,
    and coverage information. Supports HTML, JSON, and CSV formats with customizable
    templates and interactive charts.
    """
    try:
        from datetime import datetime
        import json
        from pathlib import Path
        
        # Get configuration and database manager
        app_config = get_config(ctx.obj.get('config'))
        manager = get_connection_manager(app_config)
        db_name = database or ctx.obj.get('db') or app_config.default_database
        
        console.print(f"[bold cyan]📄 Report Generation[/bold cyan]")
        console.print(f"Database: [cyan]{db_name}[/cyan]")
        if report_type:
            console.print(f"Report Type: [green]{report_type}[/green]")
        console.print(f"Format: [green]{output_format}[/green]")
        if input_file:
            console.print(f"Input Source: [blue]{input_file}[/blue]")
        console.print()
        
        # Generate timestamp for default output names
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Determine output file path
        if not output:
            if report_type:
                output = f"sqltest_{report_type}_report_{timestamp}.{output_format}"
            else:
                output = f"sqltest_report_{timestamp}.{output_format}"
        
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate report based on type and format
        if report_type == "dashboard":
            generate_dashboard_report(manager, db_name, output_path, output_format, title, include_charts, template)
        elif report_type == "validation":
            generate_validation_report(manager, db_name, input_file, output_path, output_format, title, include_charts, template)
        elif report_type == "testing":
            generate_testing_report(manager, db_name, input_file, output_path, output_format, title, include_charts, template)
        elif report_type == "coverage":
            generate_coverage_report(manager, db_name, input_file, output_path, output_format, title, include_charts, template)
        elif report_type == "profiling":
            generate_profiling_report(manager, db_name, table, output_path, output_format, title, include_charts, template)
        else:
            # Generate comprehensive report with all available data
            generate_comprehensive_report(manager, db_name, input_file, output_path, output_format, title, include_charts, template)
        
        console.print(f"\n[green]✅ Report generated successfully: {output_path}[/green]")
        
        # Show report summary
        show_report_summary(output_path, report_type, output_format)
        
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
@click.argument("project_name")
@click.option("--template", type=click.Choice(["basic", "advanced", "complete"]), default="basic", help="Project template")
@click.option("--with-validation", is_flag=True, help="Include sample validation rule sets")
@click.option("--with-tests", is_flag=True, help="Include sample test configurations")
@click.option("--with-examples", is_flag=True, help="Include example data and scenarios")
def init(project_name: str, template: str, with_validation: bool, with_tests: bool, with_examples: bool) -> None:
    """🚀 Initialize a new SQLTest Pro project.
    
    Creates a new project directory with sample configurations, validation rules,
    test cases, and documentation based on the selected template.
    """
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
        (project_path / "rules").mkdir()
        (project_path / "templates").mkdir()
        console.print("📂 Created project directories")
        
        # Determine what to include based on template and options
        include_validation = with_validation or template in ["advanced", "complete"]
        include_tests = with_tests or template in ["advanced", "complete"]
        include_examples = with_examples or template == "complete"
        
        # Create validation rule sets
        if include_validation:
            create_validation_templates(project_path, template, include_examples)
            console.print(f"✅ Created validation rule templates")
        
        # Create test configurations
        if include_tests:
            create_test_templates(project_path, template, include_examples)
            console.print(f"🧪 Created test configuration templates")
        
        # Create example data and documentation
        if include_examples:
            create_example_data(project_path)
            console.print(f"📊 Created example data and scenarios")
        
        # Create documentation
        create_project_documentation(project_path, template, include_validation, include_tests, include_examples)
        console.print(f"📚 Created project documentation")
        
        console.print(f"\n[bold green]✅ Project '{project_name}' initialized successfully![/bold green]")
        
        # Show what was created
        show_project_summary(project_path, template, include_validation, include_tests, include_examples)
        
        # Show next steps
        console.print(f"\n[bold yellow]Next Steps:[/bold yellow]")
        console.print(f"1. Edit [cyan]{config_path}[/cyan] to configure your databases")
        console.print(f"2. Set required environment variables (see README.md)")
        console.print(f"3. Test your configuration: [cyan]sqltest config validate[/cyan]")
        console.print(f"4. Test database connection: [cyan]sqltest db test[/cyan]")
        
        if include_validation:
            console.print(f"5. Customize validation rules in [cyan]rules/[/cyan] directory")
            console.print(f"6. Run validation: [cyan]sqltest validate --config rules/data_quality_rules.yaml[/cyan]")
        
        if include_tests:
            console.print(f"5. Customize test cases in [cyan]tests/[/cyan] directory")
            console.print(f"6. Run tests: [cyan]sqltest test --directory tests/[/cyan]")
        
        if include_examples:
            console.print(f"7. Explore example scenarios in [cyan]examples/[/cyan] directory")
        
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


# Report generation functions
def generate_dashboard_report(manager, db_name: str, output_path: Path, output_format: str, 
                            title: str, include_charts: bool, template: str) -> None:
    """Generate a comprehensive dashboard report."""
    from datetime import datetime
    import json
    
    console.print("[yellow]📊 Generating dashboard report...[/yellow]")
    
    # Collect dashboard data
    dashboard_data = {
        "metadata": {
            "title": title or f"SQLTest Pro Dashboard - {db_name}",
            "database": db_name,
            "generated_at": datetime.now().isoformat(),
            "sqltest_version": __version__
        },
        "database_summary": collect_database_summary(manager, db_name),
        "recent_activities": [],  # Could be populated from logs
        "system_health": collect_system_health(manager, db_name)
    }
    
    if output_format == "html":
        generate_html_dashboard(dashboard_data, output_path, include_charts, template)
    elif output_format == "json":
        generate_json_report(dashboard_data, output_path)
    else:
        generate_csv_report(dashboard_data, output_path, "dashboard")


def generate_validation_report(manager, db_name: str, input_file: str, output_path: Path, 
                             output_format: str, title: str, include_charts: bool, template: str) -> None:
    """Generate a validation results report."""
    console.print("[yellow]✅ Generating validation report...[/yellow]")
    
    # Load validation data from input file if provided
    validation_data = load_validation_data(input_file) if input_file else {}
    
    report_data = {
        "metadata": {
            "title": title or f"Validation Report - {db_name}",
            "database": db_name,
            "generated_at": datetime.now().isoformat(),
            "sqltest_version": __version__
        },
        "validation_results": validation_data,
        "summary_statistics": calculate_validation_statistics(validation_data)
    }
    
    if output_format == "html":
        generate_html_validation_report(report_data, output_path, include_charts, template)
    elif output_format == "json":
        generate_json_report(report_data, output_path)
    else:
        generate_csv_report(report_data, output_path, "validation")


def generate_testing_report(manager, db_name: str, input_file: str, output_path: Path, 
                          output_format: str, title: str, include_charts: bool, template: str) -> None:
    """Generate a testing results report."""
    console.print("[yellow]🧪 Generating testing report...[/yellow]")
    
    # Load testing data from input file if provided
    testing_data = load_testing_data(input_file) if input_file else {}
    
    report_data = {
        "metadata": {
            "title": title or f"Testing Report - {db_name}",
            "database": db_name,
            "generated_at": datetime.now().isoformat(),
            "sqltest_version": __version__
        },
        "test_results": testing_data,
        "summary_statistics": calculate_testing_statistics(testing_data)
    }
    
    if output_format == "html":
        generate_html_testing_report(report_data, output_path, include_charts, template)
    elif output_format == "json":
        generate_json_report(report_data, output_path)
    else:
        generate_csv_report(report_data, output_path, "testing")


def generate_coverage_report(manager, db_name: str, input_file: str, output_path: Path, 
                           output_format: str, title: str, include_charts: bool, template: str) -> None:
    """Generate a coverage report."""
    console.print("[yellow]📈 Generating coverage report...[/yellow]")
    
    # Load coverage data from input file if provided
    coverage_data = load_coverage_data(input_file) if input_file else {}
    
    report_data = {
        "metadata": {
            "title": title or f"Coverage Report - {db_name}",
            "database": db_name,
            "generated_at": datetime.now().isoformat(),
            "sqltest_version": __version__
        },
        "coverage_results": coverage_data,
        "summary_statistics": calculate_coverage_statistics(coverage_data)
    }
    
    if output_format == "html":
        generate_html_coverage_report(report_data, output_path, include_charts, template)
    elif output_format == "json":
        generate_json_report(report_data, output_path)
    else:
        generate_csv_report(report_data, output_path, "coverage")


def generate_profiling_report(manager, db_name: str, table: str, output_path: Path, 
                            output_format: str, title: str, include_charts: bool, template: str) -> None:
    """Generate a data profiling report."""
    console.print("[yellow]📊 Generating profiling report...[/yellow]")
    
    # Collect profiling data
    profiling_data = collect_profiling_data(manager, db_name, table)
    
    report_data = {
        "metadata": {
            "title": title or f"Profiling Report - {table or 'Database'}",
            "database": db_name,
            "table": table,
            "generated_at": datetime.now().isoformat(),
            "sqltest_version": __version__
        },
        "profiling_results": profiling_data,
        "summary_statistics": calculate_profiling_statistics(profiling_data)
    }
    
    if output_format == "html":
        generate_html_profiling_report(report_data, output_path, include_charts, template)
    elif output_format == "json":
        generate_json_report(report_data, output_path)
    else:
        generate_csv_report(report_data, output_path, "profiling")


def generate_comprehensive_report(manager, db_name: str, input_file: str, output_path: Path, 
                                output_format: str, title: str, include_charts: bool, template: str) -> None:
    """Generate a comprehensive report with all available data."""
    console.print("[yellow]📋 Generating comprehensive report...[/yellow]")
    
    # Collect all available data
    comprehensive_data = {
        "metadata": {
            "title": title or f"Comprehensive Report - {db_name}",
            "database": db_name,
            "generated_at": datetime.now().isoformat(),
            "sqltest_version": __version__
        },
        "database_summary": collect_database_summary(manager, db_name),
        "validation_results": load_validation_data(input_file) if input_file else {},
        "testing_results": load_testing_data(input_file) if input_file else {},
        "coverage_results": load_coverage_data(input_file) if input_file else {},
        "system_health": collect_system_health(manager, db_name)
    }
    
    if output_format == "html":
        generate_html_comprehensive_report(comprehensive_data, output_path, include_charts, template)
    elif output_format == "json":
        generate_json_report(comprehensive_data, output_path)
    else:
        generate_csv_report(comprehensive_data, output_path, "comprehensive")


# Helper functions for data collection and report generation
def collect_database_summary(manager, db_name: str) -> dict:
    """Collect database summary information."""
    try:
        summary = {
            "database_name": db_name,
            "connection_status": "connected",
            "tables_count": 0,
            "views_count": 0
        }
        
        adapter = manager.get_adapter(db_name)
        
        # Get table count
        if hasattr(adapter, 'get_table_names'):
            tables = adapter.get_table_names()
            summary["tables_count"] = len(tables) if tables else 0
        
        # Get view count
        if hasattr(adapter, 'get_view_names'):
            views = adapter.get_view_names()
            summary["views_count"] = len(views) if views else 0
        
        return summary
    except Exception as e:
        return {
            "database_name": db_name,
            "connection_status": "error",
            "error": str(e),
            "tables_count": 0,
            "views_count": 0
        }


def collect_system_health(manager, db_name: str) -> dict:
    """Collect system health information."""
    try:
        # Test database connection
        result = manager.test_connection(db_name)
        return {
            "database_connection": result['status'],
            "response_time_ms": result.get('response_time', 0),
            "last_check": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "database_connection": "error",
            "error": str(e),
            "last_check": datetime.now().isoformat()
        }


def collect_profiling_data(manager, db_name: str, table: str) -> dict:
    """Collect profiling data for a table."""
    try:
        if table:
            table_info = manager.get_table_info(table, db_name=db_name)
            return {
                "table_name": table,
                "row_count": table_info['row_count'],
                "column_count": len(table_info['columns']),
                "columns": table_info['columns']
            }
        else:
            # Get summary for all tables
            adapter = manager.get_adapter(db_name)
            tables = adapter.get_table_names() if hasattr(adapter, 'get_table_names') else []
            return {
                "total_tables": len(tables),
                "table_names": tables[:20]  # Limit to first 20 for report
            }
    except Exception as e:
        return {"error": str(e)}


def load_validation_data(input_file: str) -> dict:
    """Load validation data from input file."""
    if not input_file:
        return {}
    
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
            # Extract validation-specific data if it's a comprehensive export
            if 'validation_results' in data:
                return data['validation_results']
            return data
    except Exception as e:
        console.print(f"[yellow]Warning: Could not load validation data from {input_file}: {e}[/yellow]")
        return {}


def load_testing_data(input_file: str) -> dict:
    """Load testing data from input file."""
    if not input_file:
        return {}
    
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
            # Extract testing-specific data if it's a comprehensive export
            if 'test_results' in data:
                return data
            return data
    except Exception as e:
        console.print(f"[yellow]Warning: Could not load testing data from {input_file}: {e}[/yellow]")
        return {}


def load_coverage_data(input_file: str) -> dict:
    """Load coverage data from input file."""
    if not input_file:
        return {}
    
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
            # Extract coverage-specific data if it's a comprehensive export
            if 'coverage_summary' in data:
                return data['coverage_summary']
            return data
    except Exception as e:
        console.print(f"[yellow]Warning: Could not load coverage data from {input_file}: {e}[/yellow]")
        return {}


def calculate_validation_statistics(validation_data: dict) -> dict:
    """Calculate validation statistics for reporting."""
    if not validation_data:
        return {}
    
    try:
        if 'summaries' in validation_data:
            # Multiple validation summaries
            total_rules = sum(s.get('total_rules', 0) for s in validation_data['summaries'])
            passed_rules = sum(s.get('rules_passed', 0) for s in validation_data['summaries'])
            return {
                "total_validations": len(validation_data['summaries']),
                "total_rules": total_rules,
                "passed_rules": passed_rules,
                "success_rate": (passed_rules / total_rules * 100) if total_rules > 0 else 0
            }
        else:
            # Single validation summary
            return {
                "total_validations": 1,
                "total_rules": validation_data.get('total_rules', 0),
                "passed_rules": validation_data.get('rules_passed', 0),
                "success_rate": validation_data.get('success_rate', 0)
            }
    except Exception:
        return {}


def calculate_testing_statistics(testing_data: dict) -> dict:
    """Calculate testing statistics for reporting."""
    if not testing_data:
        return {}
    
    try:
        if 'summary' in testing_data:
            return testing_data['summary']
        else:
            # Calculate from test results
            test_results = testing_data.get('test_results', [])
            total_tests = len(test_results)
            passed_tests = sum(1 for t in test_results if t.get('passed', False))
            return {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": total_tests - passed_tests,
                "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0
            }
    except Exception:
        return {}


def calculate_coverage_statistics(coverage_data: dict) -> dict:
    """Calculate coverage statistics for reporting."""
    if not coverage_data:
        return {}
    
    try:
        return {
            "tables_covered": len(coverage_data.get('tables_covered', [])),
            "queries_covered": len(coverage_data.get('queries_covered', [])),
            "total_coverage_items": coverage_data.get('table_count', 0) + coverage_data.get('query_count', 0)
        }
    except Exception:
        return {}


def calculate_profiling_statistics(profiling_data: dict) -> dict:
    """Calculate profiling statistics for reporting."""
    if not profiling_data:
        return {}
    
    try:
        if 'table_name' in profiling_data:
            # Single table profiling
            return {
                "table_name": profiling_data['table_name'],
                "row_count": profiling_data.get('row_count', 0),
                "column_count": profiling_data.get('column_count', 0)
            }
        else:
            # Database-wide profiling
            return {
                "total_tables": profiling_data.get('total_tables', 0)
            }
    except Exception:
        return {}


# Report format generators
def generate_json_report(data: dict, output_path: Path) -> None:
    """Generate JSON format report."""
    import json
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def generate_csv_report(data: dict, output_path: Path, report_type: str) -> None:
    """Generate CSV format report."""
    import csv
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write metadata header
        writer.writerow(['SQLTest Pro Report', report_type.title()])
        writer.writerow(['Generated', data['metadata']['generated_at']])
        writer.writerow(['Database', data['metadata']['database']])
        writer.writerow([])  # Empty row
        
        # Write summary statistics if available
        if 'summary_statistics' in data and data['summary_statistics']:
            writer.writerow(['Summary Statistics'])
            for key, value in data['summary_statistics'].items():
                writer.writerow([key.replace('_', ' ').title(), value])
            writer.writerow([])  # Empty row
        
        # Write type-specific data
        if report_type == "validation" and 'validation_results' in data:
            write_validation_csv_data(writer, data['validation_results'])
        elif report_type == "testing" and 'test_results' in data:
            write_testing_csv_data(writer, data['test_results'])
        elif report_type == "coverage" and 'coverage_results' in data:
            write_coverage_csv_data(writer, data['coverage_results'])
        elif report_type == "profiling" and 'profiling_results' in data:
            write_profiling_csv_data(writer, data['profiling_results'])


def write_validation_csv_data(writer, validation_data: dict) -> None:
    """Write validation data to CSV."""
    if 'summaries' in validation_data:
        writer.writerow(['Rule Set', 'Total Rules', 'Passed Rules', 'Failed Rules', 'Success Rate'])
        for summary in validation_data['summaries']:
            writer.writerow([
                summary.get('rule_set_name', ''),
                summary.get('total_rules', 0),
                summary.get('rules_passed', 0),
                summary.get('rules_failed', 0),
                f"{summary.get('success_rate', 0):.1f}%"
            ])


def write_testing_csv_data(writer, testing_data: dict) -> None:
    """Write testing data to CSV."""
    if 'test_results' in testing_data:
        writer.writerow(['Test Name', 'Status', 'Execution Time (ms)', 'Assertions', 'Error Message'])
        for test in testing_data['test_results']:
            writer.writerow([
                test.get('test_name', ''),
                'PASSED' if test.get('passed', False) else 'FAILED',
                test.get('execution_time_ms', 0),
                len(test.get('assertion_results', [])),
                test.get('error_message', '')
            ])


def write_coverage_csv_data(writer, coverage_data: dict) -> None:
    """Write coverage data to CSV."""
    writer.writerow(['Coverage Type', 'Items'])
    writer.writerow(['Tables', len(coverage_data.get('tables_covered', []))])
    writer.writerow(['Queries', len(coverage_data.get('queries_covered', []))])
    
    if coverage_data.get('tables_covered'):
        writer.writerow([])  # Empty row
        writer.writerow(['Covered Tables'])
        for table in coverage_data['tables_covered']:
            writer.writerow([table])


def write_profiling_csv_data(writer, profiling_data: dict) -> None:
    """Write profiling data to CSV."""
    if 'table_name' in profiling_data:
        writer.writerow(['Property', 'Value'])
        writer.writerow(['Table Name', profiling_data['table_name']])
        writer.writerow(['Row Count', profiling_data.get('row_count', 0)])
        writer.writerow(['Column Count', profiling_data.get('column_count', 0)])
        
        if 'columns' in profiling_data:
            writer.writerow([])  # Empty row
            writer.writerow(['Column Name', 'Data Type', 'Nullable'])
            for col in profiling_data['columns']:
                writer.writerow([
                    col.get('column_name', col.get('name', '')),
                    col.get('data_type', col.get('type', '')),
                    col.get('is_nullable', '')
                ])


# HTML report generators (simplified versions)
def generate_html_dashboard(data: dict, output_path: Path, include_charts: bool, template: str) -> None:
    """Generate HTML dashboard report."""
    html_content = generate_basic_html_report(data, "Dashboard", include_charts)
    with open(output_path, 'w') as f:
        f.write(html_content)


def generate_html_validation_report(data: dict, output_path: Path, include_charts: bool, template: str) -> None:
    """Generate HTML validation report."""
    html_content = generate_basic_html_report(data, "Validation", include_charts)
    with open(output_path, 'w') as f:
        f.write(html_content)


def generate_html_testing_report(data: dict, output_path: Path, include_charts: bool, template: str) -> None:
    """Generate HTML testing report."""
    html_content = generate_basic_html_report(data, "Testing", include_charts)
    with open(output_path, 'w') as f:
        f.write(html_content)


def generate_html_coverage_report(data: dict, output_path: Path, include_charts: bool, template: str) -> None:
    """Generate HTML coverage report."""
    html_content = generate_basic_html_report(data, "Coverage", include_charts)
    with open(output_path, 'w') as f:
        f.write(html_content)


def generate_html_profiling_report(data: dict, output_path: Path, include_charts: bool, template: str) -> None:
    """Generate HTML profiling report."""
    html_content = generate_basic_html_report(data, "Profiling", include_charts)
    with open(output_path, 'w') as f:
        f.write(html_content)


def generate_html_comprehensive_report(data: dict, output_path: Path, include_charts: bool, template: str) -> None:
    """Generate HTML comprehensive report."""
    html_content = generate_basic_html_report(data, "Comprehensive", include_charts)
    with open(output_path, 'w') as f:
        f.write(html_content)


def generate_basic_html_report(data: dict, report_type: str, include_charts: bool) -> str:
    """Generate basic HTML report content."""
    metadata = data.get('metadata', {})
    
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{metadata.get('title', f'{report_type} Report')}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .metadata {{ background-color: #ecf0f1; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        .stat-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
        .stat-card {{ background-color: #f8f9fa; padding: 15px; border-left: 4px solid #3498db; border-radius: 4px; }}
        .stat-card h3 {{ margin: 0; color: #2c3e50; font-size: 14px; }}
        .stat-card .value {{ font-size: 24px; font-weight: bold; color: #3498db; margin: 5px 0; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #3498db; color: white; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .success {{ color: #27ae60; font-weight: bold; }}
        .error {{ color: #e74c3c; font-weight: bold; }}
        .warning {{ color: #f39c12; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{metadata.get('title', f'{report_type} Report')}</h1>
        
        <div class="metadata">
            <strong>Database:</strong> {metadata.get('database', 'N/A')}<br>
            <strong>Generated:</strong> {metadata.get('generated_at', 'N/A')}<br>
            <strong>SQLTest Version:</strong> {metadata.get('sqltest_version', 'N/A')}
        </div>
"""
    
    # Add summary statistics if available
    if 'summary_statistics' in data and data['summary_statistics']:
        html_content += "\n        <h2>Summary Statistics</h2>\n        <div class=\"stat-grid\">\n"
        
        for key, value in data['summary_statistics'].items():
            display_key = key.replace('_', ' ').title()
            html_content += f"""            <div class="stat-card">
                <h3>{display_key}</h3>
                <div class="value">{value}</div>
            </div>\n"""
        
        html_content += "        </div>\n"
    
    # Add type-specific content
    if report_type == "Validation" and 'validation_results' in data:
        html_content += generate_validation_html_section(data['validation_results'])
    elif report_type == "Testing" and 'test_results' in data:
        html_content += generate_testing_html_section(data['test_results'])
    elif report_type == "Coverage" and 'coverage_results' in data:
        html_content += generate_coverage_html_section(data['coverage_results'])
    elif report_type == "Profiling" and 'profiling_results' in data:
        html_content += generate_profiling_html_section(data['profiling_results'])
    elif report_type == "Dashboard":
        html_content += generate_dashboard_html_section(data)
    
    html_content += """
    </div>
</body>
</html>
"""
    
    return html_content


def generate_validation_html_section(validation_data: dict) -> str:
    """Generate HTML section for validation data."""
    if not validation_data:
        return "<p>No validation data available.</p>"
    
    html = "<h2>Validation Results</h2>"
    
    if 'summaries' in validation_data:
        html += """<table>
            <tr><th>Rule Set</th><th>Total Rules</th><th>Passed</th><th>Failed</th><th>Success Rate</th></tr>"""
        
        for summary in validation_data['summaries']:
            success_rate = summary.get('success_rate', 0)
            status_class = 'success' if success_rate >= 90 else 'warning' if success_rate >= 70 else 'error'
            
            html += f"""<tr>
                <td>{summary.get('rule_set_name', '')}</td>
                <td>{summary.get('total_rules', 0)}</td>
                <td class="success">{summary.get('rules_passed', 0)}</td>
                <td class="error">{summary.get('rules_failed', 0)}</td>
                <td class="{status_class}">{success_rate:.1f}%</td>
            </tr>"""
        
        html += "</table>"
    
    return html


def generate_testing_html_section(testing_data: dict) -> str:
    """Generate HTML section for testing data."""
    if not testing_data or 'test_results' not in testing_data:
        return "<p>No testing data available.</p>"
    
    html = "<h2>Test Results</h2>"
    html += """<table>
        <tr><th>Test Name</th><th>Status</th><th>Execution Time</th><th>Assertions</th><th>Error</th></tr>"""
    
    for test in testing_data['test_results']:
        status = 'PASSED' if test.get('passed', False) else 'FAILED'
        status_class = 'success' if test.get('passed', False) else 'error'
        
        html += f"""<tr>
            <td>{test.get('test_name', '')}</td>
            <td class="{status_class}">{status}</td>
            <td>{test.get('execution_time_ms', 0):.2f}ms</td>
            <td>{len(test.get('assertion_results', []))}</td>
            <td class="error">{test.get('error_message', '')[:50]}{'...' if len(test.get('error_message', '')) > 50 else ''}</td>
        </tr>"""
    
    html += "</table>"
    return html


def generate_coverage_html_section(coverage_data: dict) -> str:
    """Generate HTML section for coverage data."""
    if not coverage_data:
        return "<p>No coverage data available.</p>"
    
    html = "<h2>Coverage Results</h2>"
    
    html += f"""<table>
        <tr><th>Coverage Type</th><th>Items Covered</th></tr>
        <tr><td>Tables</td><td>{len(coverage_data.get('tables_covered', []))}</td></tr>
        <tr><td>Queries</td><td>{len(coverage_data.get('queries_covered', []))}</td></tr>
    </table>"""
    
    if coverage_data.get('tables_covered'):
        html += "<h3>Covered Tables</h3><ul>"
        for table in coverage_data['tables_covered'][:20]:  # Limit to first 20
            html += f"<li>{table}</li>"
        if len(coverage_data['tables_covered']) > 20:
            html += f"<li>... and {len(coverage_data['tables_covered']) - 20} more tables</li>"
        html += "</ul>"
    
    return html


def generate_profiling_html_section(profiling_data: dict) -> str:
    """Generate HTML section for profiling data."""
    if not profiling_data:
        return "<p>No profiling data available.</p>"
    
    html = "<h2>Profiling Results</h2>"
    
    if 'table_name' in profiling_data:
        html += f"""<table>
            <tr><th>Property</th><th>Value</th></tr>
            <tr><td>Table Name</td><td>{profiling_data['table_name']}</td></tr>
            <tr><td>Row Count</td><td>{profiling_data.get('row_count', 0):,}</td></tr>
            <tr><td>Column Count</td><td>{profiling_data.get('column_count', 0)}</td></tr>
        </table>"""
        
        if 'columns' in profiling_data and profiling_data['columns']:
            html += "<h3>Column Details</h3><table>"
            html += "<tr><th>Column Name</th><th>Data Type</th><th>Nullable</th></tr>"
            
            for col in profiling_data['columns'][:20]:  # Limit to first 20 columns
                html += f"""<tr>
                    <td>{col.get('column_name', col.get('name', ''))}</td>
                    <td>{col.get('data_type', col.get('type', ''))}</td>
                    <td>{col.get('is_nullable', '')}</td>
                </tr>"""
            
            if len(profiling_data['columns']) > 20:
                html += f"<tr><td colspan='3'>... and {len(profiling_data['columns']) - 20} more columns</td></tr>"
            
            html += "</table>"
    else:
        html += f"""<p>Total tables analyzed: {profiling_data.get('total_tables', 0)}</p>"""
    
    return html


def generate_dashboard_html_section(data: dict) -> str:
    """Generate HTML section for dashboard data."""
    html = "<h2>Database Summary</h2>"
    
    if 'database_summary' in data:
        summary = data['database_summary']
        html += f"""<table>
            <tr><th>Property</th><th>Value</th></tr>
            <tr><td>Connection Status</td><td class="{'success' if summary.get('connection_status') == 'connected' else 'error'}">{summary.get('connection_status', 'unknown').title()}</td></tr>
            <tr><td>Tables</td><td>{summary.get('tables_count', 0)}</td></tr>
            <tr><td>Views</td><td>{summary.get('views_count', 0)}</td></tr>
        </table>"""
    
    if 'system_health' in data:
        health = data['system_health']
        html += "<h2>System Health</h2>"
        html += f"""<table>
            <tr><th>Component</th><th>Status</th><th>Details</th></tr>
            <tr>
                <td>Database Connection</td>
                <td class="{'success' if health.get('database_connection') == 'success' else 'error'}">{health.get('database_connection', 'unknown').title()}</td>
                <td>Response Time: {health.get('response_time_ms', 0)}ms</td>
            </tr>
        </table>"""
    
    return html


def show_report_summary(output_path: Path, report_type: str, output_format: str) -> None:
    """Show a summary of the generated report."""
    file_size = output_path.stat().st_size
    size_display = f"{file_size:,} bytes"
    if file_size > 1024:
        size_display = f"{file_size/1024:.1f} KB"
    if file_size > 1024*1024:
        size_display = f"{file_size/(1024*1024):.1f} MB"
    
    console.print("\n[bold cyan]📊 Report Summary[/bold cyan]")
    
    summary_table = Table(show_header=False, box=None)
    summary_table.add_column("Property", style="yellow", width=15)
    summary_table.add_column("Value", style="white")
    
    summary_table.add_row("Type:", report_type or "Comprehensive")
    summary_table.add_row("Format:", output_format.upper())
    summary_table.add_row("File Size:", size_display)
    summary_table.add_row("Location:", str(output_path))
    
    console.print(summary_table)
    
    # Show next steps
    if output_format == "html":
        console.print(f"\n[green]💡 Open the report in your browser:[/green]")
        console.print(f"[cyan]file://{output_path.absolute()}[/cyan]")
    elif output_format == "json":
        console.print(f"\n[green]💡 Use the JSON file for further processing or API integration[/green]")
    else:
        console.print(f"\n[green]💡 Open the CSV file in Excel or any spreadsheet application[/green]")


# Project template creation functions
def create_validation_templates(project_path: Path, template: str, include_examples: bool) -> None:
    """Create validation rule templates."""
    rules_dir = project_path / "rules"
    
    # Basic data quality rules
    data_quality_rules = {
        "rule_set_name": "Data Quality Rules",
        "description": "Basic data quality validation rules for common data issues",
        "rules": [
            {
                "name": "null_check_required_fields",
                "type": "null_check",
                "severity": "error",
                "description": "Check for null values in required fields",
                "target": {
                    "table": "${TABLE_NAME}",
                    "columns": ["id", "email", "created_at"]
                },
                "tags": ["completeness", "critical"]
            },
            {
                "name": "email_format_validation",
                "type": "pattern_match",
                "severity": "warning",
                "description": "Validate email addresses format",
                "target": {
                    "table": "${TABLE_NAME}",
                    "column": "email"
                },
                "parameters": {
                    "pattern": "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"
                },
                "tags": ["format", "data_quality"]
            },
            {
                "name": "duplicate_check_primary_key",
                "type": "uniqueness",
                "severity": "critical",
                "description": "Check for duplicate values in primary key columns",
                "target": {
                    "table": "${TABLE_NAME}",
                    "columns": ["id"]
                },
                "tags": ["uniqueness", "integrity"]
            },
            {
                "name": "date_range_validation",
                "type": "range",
                "severity": "warning",
                "description": "Validate date ranges are within reasonable bounds",
                "target": {
                    "table": "${TABLE_NAME}",
                    "column": "created_at"
                },
                "parameters": {
                    "min_date": "2020-01-01",
                    "max_date": "${CURRENT_DATE}"
                },
                "tags": ["temporal", "reasonableness"]
            }
        ]
    }
    
    # Write data quality rules
    with open(rules_dir / "data_quality_rules.yaml", 'w') as f:
        yaml.dump(data_quality_rules, f, default_flow_style=False, sort_keys=False)
    
    # Referential integrity rules for advanced templates
    if template in ["advanced", "complete"]:
        referential_rules = {
            "rule_set_name": "Referential Integrity Rules",
            "description": "Rules to validate foreign key relationships and data consistency",
            "rules": [
                {
                    "name": "foreign_key_existence",
                    "type": "foreign_key",
                    "severity": "error",
                    "description": "Validate foreign key references exist",
                    "target": {
                        "table": "orders",
                        "column": "customer_id"
                    },
                    "parameters": {
                        "reference_table": "customers",
                        "reference_column": "id"
                    },
                    "tags": ["referential_integrity", "consistency"]
                },
                {
                    "name": "orphan_record_check",
                    "type": "sql_rule",
                    "severity": "warning",
                    "description": "Check for orphaned records without valid parents",
                    "target": {
                        "sql": """
                        SELECT COUNT(*) as orphan_count 
                        FROM orders o 
                        LEFT JOIN customers c ON o.customer_id = c.id 
                        WHERE c.id IS NULL
                        """
                    },
                    "parameters": {
                        "expected_value": 0,
                        "comparison": "equals"
                    },
                    "tags": ["orphans", "data_integrity"]
                }
            ]
        }
        
        with open(rules_dir / "referential_integrity_rules.yaml", 'w') as f:
            yaml.dump(referential_rules, f, default_flow_style=False, sort_keys=False)
    
    # Business logic rules for complete template
    if template == "complete" or include_examples:
        business_rules = {
            "rule_set_name": "Business Logic Rules",
            "description": "Custom business logic validation rules",
            "rules": [
                {
                    "name": "order_total_calculation",
                    "type": "sql_rule",
                    "severity": "error",
                    "description": "Validate order totals are calculated correctly",
                    "target": {
                        "sql": """
                        SELECT o.id, o.total, 
                               SUM(oi.quantity * oi.unit_price) as calculated_total
                        FROM orders o
                        JOIN order_items oi ON o.id = oi.order_id
                        GROUP BY o.id, o.total
                        HAVING ABS(o.total - SUM(oi.quantity * oi.unit_price)) > 0.01
                        """
                    },
                    "parameters": {
                        "max_violations": 0
                    },
                    "tags": ["business_logic", "calculations"]
                },
                {
                    "name": "inventory_consistency",
                    "type": "sql_rule",
                    "severity": "warning",
                    "description": "Check inventory levels are consistent with sales",
                    "target": {
                        "sql": """
                        SELECT p.id, p.stock_quantity,
                               COALESCE(SUM(oi.quantity), 0) as total_sold
                        FROM products p
                        LEFT JOIN order_items oi ON p.id = oi.product_id
                        GROUP BY p.id, p.stock_quantity
                        HAVING p.stock_quantity < 0
                        """
                    },
                    "parameters": {
                        "max_violations": 0
                    },
                    "tags": ["inventory", "consistency"]
                }
            ]
        }
        
        with open(rules_dir / "business_logic_rules.yaml", 'w') as f:
            yaml.dump(business_rules, f, default_flow_style=False, sort_keys=False)


def create_test_templates(project_path: Path, template: str, include_examples: bool) -> None:
    """Create test configuration templates."""
    tests_dir = project_path / "tests"
    
    # Basic unit test configuration
    unit_tests = {
        "test_suite_name": "Basic Unit Tests",
        "description": "Basic SQL unit tests for database functions and procedures",
        "tests": [
            {
                "name": "test_user_creation",
                "description": "Test user creation with valid data",
                "tags": ["users", "crud", "basic"],
                "setup": {
                    "sql": [
                        "DELETE FROM users WHERE email = 'test@example.com'"
                    ]
                },
                "execute": {
                    "sql": """
                    INSERT INTO users (name, email, created_at) 
                    VALUES ('Test User', 'test@example.com', NOW())
                    RETURNING id, name, email
                    """
                },
                "assertions": [
                    {
                        "type": "row_count",
                        "expected": 1,
                        "message": "Should insert exactly one user"
                    },
                    {
                        "type": "column_value",
                        "column": "name",
                        "expected": "Test User",
                        "message": "Name should match input"
                    },
                    {
                        "type": "not_null",
                        "column": "id",
                        "message": "ID should be generated"
                    }
                ],
                "teardown": {
                    "sql": [
                        "DELETE FROM users WHERE email = 'test@example.com'"
                    ]
                }
            },
            {
                "name": "test_email_uniqueness",
                "description": "Test that duplicate emails are rejected",
                "tags": ["users", "constraints", "validation"],
                "setup": {
                    "sql": [
                        "DELETE FROM users WHERE email = 'duplicate@example.com'",
                        "INSERT INTO users (name, email, created_at) VALUES ('First User', 'duplicate@example.com', NOW())"
                    ]
                },
                "execute": {
                    "sql": """
                    INSERT INTO users (name, email, created_at) 
                    VALUES ('Second User', 'duplicate@example.com', NOW())
                    """,
                    "expect_error": True
                },
                "assertions": [
                    {
                        "type": "error_occurred",
                        "expected": True,
                        "message": "Should raise constraint violation error"
                    }
                ],
                "teardown": {
                    "sql": [
                        "DELETE FROM users WHERE email = 'duplicate@example.com'"
                    ]
                }
            }
        ]
    }
    
    with open(tests_dir / "unit_tests.yaml", 'w') as f:
        yaml.dump(unit_tests, f, default_flow_style=False, sort_keys=False)
    
    # Integration tests for advanced templates
    if template in ["advanced", "complete"]:
        integration_tests = {
            "test_suite_name": "Integration Tests",
            "description": "Integration tests for complex business workflows",
            "fixtures": {
                "customers": {
                    "type": "csv",
                    "data": [
                        {"id": 1, "name": "John Doe", "email": "john@example.com"},
                        {"id": 2, "name": "Jane Smith", "email": "jane@example.com"}
                    ]
                },
                "products": {
                    "type": "csv",
                    "data": [
                        {"id": 1, "name": "Widget A", "price": 10.99, "stock": 100},
                        {"id": 2, "name": "Widget B", "price": 15.50, "stock": 50}
                    ]
                }
            },
            "tests": [
                {
                    "name": "test_order_workflow",
                    "description": "Test complete order creation and processing workflow",
                    "tags": ["orders", "workflow", "integration"],
                    "dependencies": ["customers", "products"],
                    "setup": {
                        "fixtures": ["customers", "products"],
                        "sql": [
                            "DELETE FROM order_items",
                            "DELETE FROM orders"
                        ]
                    },
                    "execute": {
                        "sql": """
                        WITH new_order AS (
                            INSERT INTO orders (customer_id, order_date, status)
                            VALUES (1, NOW(), 'pending')
                            RETURNING id
                        ),
                        order_items AS (
                            INSERT INTO order_items (order_id, product_id, quantity, unit_price)
                            SELECT no.id, 1, 2, 10.99 FROM new_order no
                            UNION ALL
                            SELECT no.id, 2, 1, 15.50 FROM new_order no
                        )
                        SELECT o.id, o.customer_id, o.status,
                               COUNT(oi.id) as item_count,
                               SUM(oi.quantity * oi.unit_price) as total
                        FROM orders o
                        JOIN order_items oi ON o.id = oi.order_id
                        WHERE o.id = (SELECT id FROM new_order)
                        GROUP BY o.id, o.customer_id, o.status
                        """
                    },
                    "assertions": [
                        {
                            "type": "row_count",
                            "expected": 1,
                            "message": "Should create one order"
                        },
                        {
                            "type": "column_value",
                            "column": "item_count",
                            "expected": 2,
                            "message": "Should have 2 order items"
                        },
                        {
                            "type": "column_value",
                            "column": "total",
                            "expected": 37.48,
                            "message": "Total should be calculated correctly"
                        }
                    ],
                    "teardown": {
                        "sql": [
                            "DELETE FROM order_items",
                            "DELETE FROM orders"
                        ]
                    }
                }
            ]
        }
        
        with open(tests_dir / "integration_tests.yaml", 'w') as f:
            yaml.dump(integration_tests, f, default_flow_style=False, sort_keys=False)
    
    # Performance tests for complete template
    if template == "complete" or include_examples:
        performance_tests = {
            "test_suite_name": "Performance Tests",
            "description": "Performance and load testing for database operations",
            "tests": [
                {
                    "name": "test_bulk_insert_performance",
                    "description": "Test performance of bulk insert operations",
                    "tags": ["performance", "bulk_operations"],
                    "timeout_ms": 5000,
                    "setup": {
                        "sql": ["CREATE TEMPORARY TABLE temp_users AS SELECT * FROM users LIMIT 0"]
                    },
                    "execute": {
                        "sql": """
                        INSERT INTO temp_users (name, email, created_at)
                        SELECT 
                            'User ' || generate_series(1, 1000),
                            'user' || generate_series(1, 1000) || '@example.com',
                            NOW()
                        """
                    },
                    "assertions": [
                        {
                            "type": "execution_time",
                            "max_ms": 3000,
                            "message": "Bulk insert should complete within 3 seconds"
                        },
                        {
                            "type": "row_count",
                            "table": "temp_users",
                            "expected": 1000,
                            "message": "Should insert 1000 users"
                        }
                    ]
                }
            ]
        }
        
        with open(tests_dir / "performance_tests.yaml", 'w') as f:
            yaml.dump(performance_tests, f, default_flow_style=False, sort_keys=False)


def create_example_data(project_path: Path) -> None:
    """Create example data and scenarios."""
    examples_dir = project_path / "examples"
    examples_dir.mkdir(exist_ok=True)
    
    # Sample data files
    sample_users_csv = """id,name,email,created_at,status
1,John Doe,john@example.com,2023-01-15 10:30:00,active
2,Jane Smith,jane@example.com,2023-01-16 14:20:00,active
3,Bob Johnson,bob@example.com,2023-01-17 09:15:00,inactive
4,Alice Brown,alice@example.com,2023-01-18 16:45:00,active
5,Charlie Wilson,charlie@example.com,2023-01-19 11:30:00,pending"""
    
    with open(examples_dir / "sample_users.csv", 'w') as f:
        f.write(sample_users_csv)
    
    # Sample schema creation script
    schema_sql = """
-- Example database schema for SQLTest Pro
-- This creates a simple e-commerce-like schema for demonstration

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(20) DEFAULT 'active'
);

-- Products table
CREATE TABLE IF NOT EXISTS products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    price DECIMAL(10, 2) NOT NULL,
    stock_quantity INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Orders table
CREATE TABLE IF NOT EXISTS orders (
    id SERIAL PRIMARY KEY,
    customer_id INTEGER REFERENCES users(id),
    order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(20) DEFAULT 'pending',
    total DECIMAL(10, 2)
);

-- Order items table
CREATE TABLE IF NOT EXISTS order_items (
    id SERIAL PRIMARY KEY,
    order_id INTEGER REFERENCES orders(id),
    product_id INTEGER REFERENCES products(id),
    quantity INTEGER NOT NULL,
    unit_price DECIMAL(10, 2) NOT NULL
);

-- Sample data
INSERT INTO users (name, email, created_at, status) VALUES
('John Doe', 'john@example.com', '2023-01-15 10:30:00', 'active'),
('Jane Smith', 'jane@example.com', '2023-01-16 14:20:00', 'active'),
('Bob Johnson', 'bob@example.com', '2023-01-17 09:15:00', 'inactive')
ON CONFLICT (email) DO NOTHING;

INSERT INTO products (name, description, price, stock_quantity) VALUES
('Widget A', 'A useful widget for various tasks', 10.99, 100),
('Widget B', 'An advanced widget with extra features', 15.50, 50),
('Gadget X', 'A revolutionary gadget', 25.00, 25)
ON CONFLICT DO NOTHING;
"""
    
    with open(examples_dir / "sample_schema.sql", 'w') as f:
        f.write(schema_sql)
    
    # Sample scenarios documentation
    scenarios_md = """
# Example Test Scenarios

This directory contains example data and test scenarios to help you get started with SQLTest Pro.

## Files

- `sample_schema.sql` - Example database schema with tables and sample data
- `sample_users.csv` - Sample user data for testing
- `scenario_*.yaml` - Various testing scenarios

## Scenarios

### 1. Data Quality Testing
- Null value validation
- Format validation (email, phone, etc.)
- Range validation for numeric fields
- Duplicate detection

### 2. Business Rule Testing  
- Order total calculations
- Inventory consistency
- Customer status transitions
- Date range validations

### 3. Integration Testing
- Multi-table operations
- Foreign key constraints
- Transaction integrity
- Workflow validation

### 4. Performance Testing
- Bulk operations
- Query performance
- Index effectiveness
- Concurrent access patterns

## Getting Started

1. Create your test database and run `sample_schema.sql`
2. Update the database configuration in `../sqltest.yaml`
3. Customize the validation rules in `../rules/`
4. Modify the test cases in `../tests/`
5. Run your first validation: `sqltest validate --config ../rules/data_quality_rules.yaml`
6. Run your first tests: `sqltest test --directory ../tests/`

## Customization

Feel free to modify these examples to match your specific:
- Database schema
- Business rules
- Data quality requirements
- Performance expectations
"""
    
    with open(examples_dir / "README.md", 'w') as f:
        f.write(scenarios_md)


def create_project_documentation(project_path: Path, template: str, include_validation: bool, 
                                include_tests: bool, include_examples: bool) -> None:
    """Create project documentation."""
    
    readme_content = f"""
# SQLTest Pro Project

Welcome to your new SQLTest Pro project! This project was created with the `{template}` template.

## 🚀 Quick Start

### 1. Configure Your Database

Edit `sqltest.yaml` to configure your database connections:

```yaml
default_database: dev_db

databases:
  dev_db:
    type: postgresql  # or mysql, sqlite
    host: localhost
    port: 5432
    database: your_database
    username: your_username
    password: ${{DEV_DB_PASSWORD}}  # Use environment variables for passwords
```

### 2. Set Environment Variables

Create a `.env` file or set environment variables:

```bash
export DEV_DB_PASSWORD="your_secure_password"
```

### 3. Test Your Configuration

```bash
# Validate configuration
sqltest config validate sqltest.yaml

# Test database connection
sqltest db test

# List available tables
sqltest db tables
```

## 📊 Features Included

### Data Profiling
Analyze your data structure and quality:

```bash
# Profile a specific table
sqltest profile --table users

# Profile with custom query
sqltest profile --query "SELECT * FROM orders WHERE created_at > '2023-01-01'"
```
"""
    
    if include_validation:
        readme_content += """
### Validation Rules
The `rules/` directory contains validation rule sets:

- `data_quality_rules.yaml` - Basic data quality checks
"""
        if template in ["advanced", "complete"]:
            readme_content += """- `referential_integrity_rules.yaml` - Foreign key and relationship validation
"""
        if template == "complete":
            readme_content += """- `business_logic_rules.yaml` - Custom business rule validation
"""
        
        readme_content += """
```bash
# Run validation rules
sqltest validate --config rules/data_quality_rules.yaml

# Run with specific table focus
sqltest validate --config rules/data_quality_rules.yaml --table users

# Generate automatic data quality rules
sqltest validate --generate --table users
```
"""
    
    if include_tests:
        readme_content += """
### Unit Testing
The `tests/` directory contains test suites:

- `unit_tests.yaml` - Basic unit tests for database operations
"""
        if template in ["advanced", "complete"]:
            readme_content += """- `integration_tests.yaml` - Multi-table integration tests
"""
        if template == "complete":
            readme_content += """- `performance_tests.yaml` - Performance and load tests
"""
        
        readme_content += """
```bash
# Run all tests
sqltest test --directory tests/

# Run specific test file
sqltest test --config tests/unit_tests.yaml

# Run with coverage reporting
sqltest test --directory tests/ --coverage

# Run tests matching specific tags
sqltest test --directory tests/ --group "users,crud"
```
"""
    
    if include_examples:
        readme_content += """
### Example Data
The `examples/` directory contains:

- Sample database schema (`sample_schema.sql`)
- Sample data files (`sample_users.csv`)
- Documentation for various testing scenarios

See `examples/README.md` for detailed information.
"""
    
    readme_content += """
## 📄 Reporting

Generate comprehensive reports:

```bash
# Generate HTML dashboard
sqltest report --type dashboard --format html

# Generate validation report from results
sqltest validate --output validation_results.json
sqltest report --type validation --input validation_results.json --format html

# Generate test coverage report
sqltest test --directory tests/ --coverage --output test_results.json
sqltest report --type testing --input test_results.json --format html
```

## 📚 Directory Structure

```
{project_path.name}/
├── sqltest.yaml              # Main configuration
├── README.md                 # This file
├── .env                      # Environment variables (create this)
├── reports/                  # Generated reports
├── templates/                # Custom report templates
"""
    
    if include_validation:
        readme_content += """├── rules/                    # Validation rule sets
│   ├── data_quality_rules.yaml
"""
        if template in ["advanced", "complete"]:
            readme_content += """│   ├── referential_integrity_rules.yaml
"""
        if template == "complete":
            readme_content += """│   └── business_logic_rules.yaml
"""
        else:
            readme_content = readme_content.rstrip() + "\n"
    
    if include_tests:
        readme_content += """├── tests/                   # Test configurations
│   ├── unit_tests.yaml
"""
        if template in ["advanced", "complete"]:
            readme_content += """│   ├── integration_tests.yaml
"""
        if template == "complete":
            readme_content += """│   └── performance_tests.yaml
"""
        else:
            readme_content = readme_content.rstrip() + "\n"
    
    if include_examples:
        readme_content += """└── examples/               # Example data and scenarios
    ├── README.md
    ├── sample_schema.sql
    └── sample_users.csv
"""
    else:
        readme_content += """└── ...
"""
    
    readme_content += """
```

## 🔧 Customization

### Adding New Validation Rules

1. Edit existing rule files in `rules/` or create new ones
2. Follow the YAML schema for rule definitions
3. Test your rules: `sqltest validate --config rules/your_rules.yaml`

### Creating New Tests

1. Add test configurations to `tests/` directory
2. Define fixtures, setup, execution, and assertions
3. Run your tests: `sqltest test --config tests/your_tests.yaml`

### Custom Reports

1. Create custom HTML templates in `templates/`
2. Use `--template` option when generating reports
3. Customize styling and layout as needed

## 📖 Documentation

- [SQLTest Pro Documentation](https://github.com/your-org/sqltest-pro)
- [Configuration Reference](https://github.com/your-org/sqltest-pro/docs/config)
- [Rule Definition Guide](https://github.com/your-org/sqltest-pro/docs/rules)
- [Testing Framework Guide](https://github.com/your-org/sqltest-pro/docs/testing)

## 🤝 Support

If you need help:

1. Check the documentation
2. Run `sqltest --help` for CLI usage
3. Use `--verbose` flag for detailed error information
4. Create an issue on the project repository

---

**Happy Testing!** 🎉
"""
    
    with open(project_path / "README.md", 'w') as f:
        f.write(readme_content)


def show_project_summary(project_path: Path, template: str, include_validation: bool, 
                        include_tests: bool, include_examples: bool) -> None:
    """Show summary of what was created."""
    console.print(f"\n[bold cyan]📋 Project Summary[/bold cyan]")
    
    summary_table = Table(show_header=True, header_style="bold magenta")
    summary_table.add_column("Component", style="cyan")
    summary_table.add_column("Status", style="green")
    summary_table.add_column("Files Created", style="white")
    
    summary_table.add_row("Configuration", "✅ Created", "sqltest.yaml")
    summary_table.add_row("Documentation", "✅ Created", "README.md")
    summary_table.add_row("Directories", "✅ Created", "reports/, templates/")
    
    if include_validation:
        files = "data_quality_rules.yaml"
        if template in ["advanced", "complete"]:
            files += ", referential_integrity_rules.yaml"
        if template == "complete":
            files += ", business_logic_rules.yaml"
        summary_table.add_row("Validation Rules", "✅ Created", files)
    
    if include_tests:
        files = "unit_tests.yaml"
        if template in ["advanced", "complete"]:
            files += ", integration_tests.yaml"
        if template == "complete":
            files += ", performance_tests.yaml"
        summary_table.add_row("Test Configurations", "✅ Created", files)
    
    if include_examples:
        summary_table.add_row("Example Data", "✅ Created", "sample_schema.sql, sample_users.csv")
    
    console.print(summary_table)


# Import yaml at the top of the file - add this near other imports
import yaml
from datetime import datetime


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

"""Main CLI entry point for SQLTest Pro."""

import click
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from sqltest import __version__

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
@click.pass_context
def profile(ctx: click.Context, table: str, query: str, columns: str, sample: int) -> None:
    """📊 Profile data in tables or queries."""
    console.print("[bold blue]Data Profiling[/bold blue]")
    
    if table:
        console.print(f"Profiling table: [green]{table}[/green]")
    elif query:
        console.print(f"Profiling query: [green]{query[:50]}...[/green]")
    else:
        console.print("[red]Error: Either --table or --query must be specified[/red]")
        return
    
    console.print("[yellow]⚠️  Data profiling module not yet implemented[/yellow]")


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
    console.print("[yellow]⚠️  Project initialization not yet implemented[/yellow]")


if __name__ == "__main__":
    cli()

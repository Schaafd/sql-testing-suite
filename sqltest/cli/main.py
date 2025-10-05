"""Main CLI entry point for SQLTest Pro."""

from __future__ import annotations

import click
from rich.panel import Panel
from rich.text import Text

from sqltest import __version__
from sqltest.cli.commands import register_commands
from sqltest.cli.commands.business_rules import business_rules_command
from sqltest.cli.commands.configuration import config_group
from sqltest.cli.commands.database import db_group
from sqltest.cli.commands.init import init_command
from sqltest.cli.commands.profile import profile_command
from sqltest.cli.commands.report import report_command
from sqltest.cli.commands.testing import test_command
from sqltest.cli.commands.validate import validate_command
from sqltest.cli.utils import console


@click.group(invoke_without_command=True)
@click.option("--version", is_flag=True, help="Show version information")
@click.option("--config", type=click.Path(exists=True), help="Path to configuration file")
@click.option("--db", help="Database connection name")
@click.option("--output", type=click.Choice(["table", "json", "html"]), default="table", help="Output format")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.pass_context
def cli(
    ctx: click.Context,
    version: bool,
    config: str,
    db: str,
    output: str,
    verbose: bool,
) -> None:
    """SQLTest Pro - A comprehensive SQL testing and validation suite."""
    ctx.ensure_object(dict)
    ctx.obj.update(
        {
            "config": config,
            "db": db,
            "output": output,
            "verbose": verbose,
        }
    )

    if version:
        console.print(f"SQLTest Pro v{__version__}")
        return

    if ctx.invoked_subcommand is None:
        show_dashboard()


# Commands are registered in workflow order:
# 1) Profiling and validation, 2) Testing and reporting, 3) Environment tools.
COMMAND_REGISTRY = [
    profile_command,
    validate_command,
    business_rules_command,
    test_command,
    report_command,
    db_group,
    config_group,
    init_command,
]

register_commands(cli, COMMAND_REGISTRY)


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
        padding=(1, 2),
    )

    console.print(panel)


if __name__ == "__main__":
    cli()

"""Configuration management CLI commands."""

from __future__ import annotations

from pathlib import Path

import click

from sqltest.cli.utils import console
from sqltest.config import create_sample_config, get_config
from sqltest.exceptions import ConfigurationError


@click.group(name="config")
def config_group() -> None:
    """⚙️  Configuration management."""
    pass


@config_group.command(name="validate")
@click.argument("config_file", type=click.Path(exists=True))
def validate_command(config_file: str) -> None:
    """Validate configuration file."""
    try:
        config = get_config(config_file)
        console.print(f"[green]✅ Configuration file '{config_file}' is valid[/green]")
        console.print(f"Found {len(config.databases)} database(s): {', '.join(config.databases.keys())}")
        console.print(f"Default database: [cyan]{config.default_database}[/cyan]")
    except ConfigurationError as exc:
        console.print(f"[red]❌ Configuration validation failed: {exc}[/red]")
        raise SystemExit(1) from exc
    except Exception as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise SystemExit(1) from exc


@config_group.command(name="sample")
@click.argument("output_file", type=click.Path())
def sample_command(output_file: str) -> None:
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
    except Exception as exc:
        console.print(f"[red]Error creating sample configuration: {exc}[/red]")
        raise SystemExit(1) from exc

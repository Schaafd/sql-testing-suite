"""Command registration helpers for SQLTest CLI."""

from __future__ import annotations

from typing import Iterable

import click


def register_commands(cli: click.Group, commands: Iterable[click.Command]) -> None:
    """Register the provided click commands with the root CLI group."""
    for command in commands:
        cli.add_command(command)

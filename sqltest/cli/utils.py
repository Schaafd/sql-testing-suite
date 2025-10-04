"""Shared CLI utilities for SQLTest Pro."""

from __future__ import annotations

from rich.console import Console

# Single console instance reused across CLI modules
console = Console()


def print_exception(message: str, error: Exception, verbose: bool = False) -> None:
    """Render a formatted exception message.

    Args:
        message: Friendly context message to display before the exception.
        error: Original exception instance.
        verbose: When True, render the full traceback for debugging.
    """
    from rich import print as rprint

    rprint(f"[red]{message}: {error}[/red]")
    if verbose:
        import traceback

        console.print(f"[dim]{traceback.format_exc()}[/dim]")

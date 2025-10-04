"""Report command implementation for SQLTest CLI."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
from rich.table import Table

from sqltest import __version__
from sqltest.cli.utils import console
from sqltest.config import get_config
from sqltest.db import get_connection_manager
from sqltest.exceptions import ConfigurationError, DatabaseError


@click.command(name="report")
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
def report_command(ctx: click.Context, report_type: str | None, output_format: str, input_file: str | None,
                   output: str | None, config: str | None, database: str | None, table: str | None,
                   title: str | None, include_charts: bool, template: str | None) -> None:
    """Generate comprehensive reports for SQLTest Pro."""
    try:
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

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not output:
            suffix = report_type or 'report'
            output = f"sqltest_{suffix}_{timestamp}.{output_format}"

        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

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
            generate_comprehensive_report(manager, db_name, input_file, output_path, output_format, title, include_charts, template)

        show_report_summary(output_path, report_type or 'comprehensive', output_format)
    except ConfigurationError as exc:
        console.print(f"[red]Configuration Error: {exc}[/red]")
        raise SystemExit(1) from exc
    except DatabaseError as exc:
        console.print(f"[red]Database Error: {exc}[/red]")
        raise SystemExit(1) from exc
    except Exception as exc:
        if ctx.obj.get('verbose'):
            import traceback
            console.print(f"[red]Error: {exc}[/red]")
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
        else:
            console.print(f"[red]Error: {exc}[/red]")
        raise SystemExit(1) from exc


# Report generation functions
def generate_dashboard_report(manager, db_name: str, output_path: Path, output_format: str, 
                            title: Optional[str], include_charts: bool, template: Optional[str]) -> None:
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


def generate_validation_report(manager, db_name: str, input_file: Optional[str], output_path: Path, 
                             output_format: str, title: Optional[str], include_charts: bool, template: Optional[str]) -> None:
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


def generate_testing_report(manager, db_name: str, input_file: Optional[str], output_path: Path, 
                          output_format: str, title: Optional[str], include_charts: bool, template: Optional[str]) -> None:
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


def generate_coverage_report(manager, db_name: str, input_file: Optional[str], output_path: Path, 
                           output_format: str, title: Optional[str], include_charts: bool, template: Optional[str]) -> None:
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


def generate_profiling_report(manager, db_name: str, table: Optional[str], output_path: Path, 
                            output_format: str, title: Optional[str], include_charts: bool, template: Optional[str]) -> None:
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


def generate_comprehensive_report(manager, db_name: str, input_file: Optional[str], output_path: Path, 
                                output_format: str, title: Optional[str], include_charts: bool, template: Optional[str]) -> None:
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

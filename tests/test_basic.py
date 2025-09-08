"""Basic tests for SQLTest Pro package and CLI."""

import pytest
from click.testing import CliRunner

import sqltest
from sqltest.cli.main import cli


class TestPackageBasics:
    """Test basic package functionality."""
    
    def test_package_version(self) -> None:
        """Test that package has a version."""
        assert hasattr(sqltest, '__version__')
        assert isinstance(sqltest.__version__, str)
        assert len(sqltest.__version__) > 0
    
    def test_package_exports(self) -> None:
        """Test that package exports expected classes."""
        assert hasattr(sqltest, 'SQLTestError')
        assert hasattr(sqltest, 'ConfigurationError')
        assert hasattr(sqltest, 'DatabaseError')


class TestCLI:
    """Test CLI functionality."""
    
    def test_cli_help(self) -> None:
        """Test that CLI help works."""
        runner = CliRunner()
        result = runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        assert 'SQLTest Pro' in result.output
    
    def test_cli_version(self) -> None:
        """Test that CLI version flag works."""
        runner = CliRunner()
        result = runner.invoke(cli, ['--version'])
        assert result.exit_code == 0
        assert 'SQLTest Pro' in result.output
        assert sqltest.__version__ in result.output
    
    def test_cli_profile_help(self) -> None:
        """Test profile command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ['profile', '--help'])
        assert result.exit_code == 0
        assert 'Profile data' in result.output
    
    def test_cli_validate_help(self) -> None:
        """Test validate command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ['validate', '--help'])
        assert result.exit_code == 0
        assert 'field validation' in result.output
    
    def test_cli_test_help(self) -> None:
        """Test test command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ['test', '--help'])
        assert result.exit_code == 0
        assert 'unit tests' in result.output
    
    def test_cli_report_help(self) -> None:
        """Test report command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ['report', '--help'])
        assert result.exit_code == 0
        assert 'Generate comprehensive reports' in result.output
    
    def test_cli_init_help(self) -> None:
        """Test init command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ['init', '--help'])
        assert result.exit_code == 0
        assert 'Initialize' in result.output

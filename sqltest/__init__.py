"""SQLTest Pro: A comprehensive testing framework for SQL code.

SQLTest Pro provides:
- Data profiling and analysis
- Field and business rule validation
- SQL unit testing with fixtures and coverage
- Interactive CLI with progress tracking
- Multi-database support
- YAML-based configuration
"""

__version__ = "0.1.0"
__author__ = "David Schaaf"
__email__ = "your.email@example.com"
__license__ = "MIT"

# Core exports
from sqltest.exceptions import SQLTestError, ConfigurationError, DatabaseError

__all__ = [
    "__version__",
    "SQLTestError", 
    "ConfigurationError",
    "DatabaseError",
]

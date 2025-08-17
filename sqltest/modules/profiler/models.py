"""Data profiler models and statistics containers."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import pandas as pd


@dataclass
class ColumnStatistics:
    """Statistics for a single column."""
    
    column_name: str
    data_type: str
    
    # Basic counts
    total_rows: int
    non_null_count: int
    null_count: int
    unique_count: int
    
    # Percentages
    null_percentage: float
    unique_percentage: float
    
    # For numeric columns
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    mean_value: Optional[float] = None
    median_value: Optional[float] = None
    std_deviation: Optional[float] = None
    quartiles: Optional[Dict[str, float]] = None
    
    # For string columns
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    avg_length: Optional[float] = None
    
    # For date columns
    min_date: Optional[datetime] = None
    max_date: Optional[datetime] = None
    
    # Pattern analysis
    patterns: Optional[List[Dict[str, Any]]] = None
    
    # Most/least frequent values
    most_frequent: Optional[List[Dict[str, Any]]] = None
    least_frequent: Optional[List[Dict[str, Any]]] = None
    
    # Outliers
    outliers: Optional[List[Any]] = None
    
    def __post_init__(self):
        """Calculate derived statistics."""
        if self.total_rows > 0:
            self.null_percentage = (self.null_count / self.total_rows) * 100
            self.unique_percentage = (self.unique_count / self.total_rows) * 100
        else:
            self.null_percentage = 0.0
            self.unique_percentage = 0.0


@dataclass
class TableProfile:
    """Complete profile of a database table."""
    
    table_name: str
    schema_name: Optional[str]
    database_name: str
    
    # Table-level statistics
    total_rows: int
    total_columns: int
    
    # Timing
    profile_timestamp: datetime
    execution_time: float  # seconds
    
    # Column profiles
    columns: Dict[str, ColumnStatistics]
    
    # Table-level insights
    primary_keys: List[str]
    foreign_keys: List[Dict[str, str]]
    indexes: List[str]
    
    # Data quality metrics
    completeness_score: float  # 0-100
    uniqueness_score: float   # 0-100
    validity_score: float     # 0-100
    consistency_score: float  # 0-100
    overall_score: float      # 0-100
    
    # Warnings and recommendations
    warnings: List[str]
    recommendations: List[str]
    
    def __post_init__(self):
        """Calculate overall quality scores."""
        if self.columns:
            # Calculate completeness score (average of non-null percentages)
            completeness_scores = [
                (col.non_null_count / col.total_rows) * 100
                for col in self.columns.values()
                if col.total_rows > 0
            ]
            self.completeness_score = sum(completeness_scores) / len(completeness_scores) if completeness_scores else 0
            
            # Calculate uniqueness score (inverse of duplicate percentage)
            uniqueness_scores = [
                col.unique_percentage for col in self.columns.values()
            ]
            self.uniqueness_score = sum(uniqueness_scores) / len(uniqueness_scores) if uniqueness_scores else 0
            
            # For now, set validity and consistency to base values
            # These will be enhanced with actual validation rules
            self.validity_score = 85.0  # Placeholder
            self.consistency_score = 90.0  # Placeholder
            
            # Overall score is weighted average
            self.overall_score = (
                self.completeness_score * 0.3 +
                self.uniqueness_score * 0.2 +
                self.validity_score * 0.3 +
                self.consistency_score * 0.2
            )


@dataclass
class QueryProfile:
    """Profile of a SQL query result."""
    
    query: str
    query_hash: str
    
    # Query execution
    execution_time: float  # seconds
    rows_returned: int
    columns_returned: int
    profile_timestamp: datetime
    
    # Column profiles
    columns: Dict[str, ColumnStatistics]
    
    # Query insights
    estimated_cost: Optional[float] = None
    query_plan: Optional[Dict[str, Any]] = None
    
    # Performance metrics
    rows_per_second: float = 0.0
    
    def __post_init__(self):
        """Calculate performance metrics."""
        if self.execution_time > 0:
            self.rows_per_second = self.rows_returned / self.execution_time


@dataclass
class PatternMatch:
    """Represents a detected pattern in column data."""
    
    pattern_type: str  # email, phone, ssn, credit_card, etc.
    pattern_name: str
    regex_pattern: str
    confidence: float  # 0.0 to 1.0
    match_count: int
    match_percentage: float
    examples: List[str]


@dataclass  
class DataQualityIssue:
    """Represents a data quality issue found during profiling."""
    
    severity: str  # critical, warning, info
    category: str  # completeness, uniqueness, validity, consistency
    column_name: Optional[str]
    description: str
    recommendation: str
    affected_rows: int
    examples: List[Any]


@dataclass
class ComparisonResult:
    """Results from comparing two profiles."""
    
    profile1_name: str
    profile2_name: str
    comparison_timestamp: datetime
    
    # Schema changes
    added_columns: List[str]
    removed_columns: List[str]
    modified_columns: List[Dict[str, Any]]
    
    # Data changes
    row_count_change: int
    row_count_change_percentage: float
    
    # Column-level changes
    column_changes: Dict[str, Dict[str, Any]]
    
    # Overall assessment
    schema_stability: float  # 0-100
    data_stability: float    # 0-100
    overall_stability: float # 0-100

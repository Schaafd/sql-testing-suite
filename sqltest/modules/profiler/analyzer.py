"""Core data analysis and statistics generation."""

import re
import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np

from .models import (
    ColumnStatistics, TableProfile, QueryProfile, PatternMatch, 
    DataQualityIssue, ComparisonResult
)
from ...exceptions import ProfilingError


class DataAnalyzer:
    """Core data analyzer for generating statistics and insights."""
    
    # Common patterns for data type detection
    PATTERNS = {
        'email': {
            'regex': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'name': 'Email Address'
        },
        'phone_us': {
            'regex': r'^\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}$',
            'name': 'US Phone Number'
        },
        'ssn': {
            'regex': r'^\d{3}-?\d{2}-?\d{4}$',
            'name': 'Social Security Number'
        },
        'credit_card': {
            'regex': r'^\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}$',
            'name': 'Credit Card Number'
        },
        'url': {
            'regex': r'^https?://[^\s/$.?#].[^\s]*$',
            'name': 'URL'
        },
        'uuid': {
            'regex': r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
            'name': 'UUID'
        },
        'ip_address': {
            'regex': r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$',
            'name': 'IP Address'
        }
    }
    
    def __init__(self, sample_size: int = 10000):
        """Initialize analyzer with configuration.
        
        Args:
            sample_size: Maximum number of rows to analyze for patterns
        """
        self.sample_size = sample_size
    
    def analyze_column(self, data: pd.Series, column_name: str) -> ColumnStatistics:
        """Analyze a single column and generate statistics.
        
        Args:
            data: Pandas Series containing column data
            column_name: Name of the column
            
        Returns:
            ColumnStatistics object with analysis results
        """
        total_rows = len(data)
        non_null_data = data.dropna()
        non_null_count = len(non_null_data)
        null_count = total_rows - non_null_count
        
        # Basic statistics
        unique_count = data.nunique()
        
        # Initialize statistics object
        stats = ColumnStatistics(
            column_name=column_name,
            data_type=str(data.dtype),
            total_rows=total_rows,
            non_null_count=non_null_count,
            null_count=null_count,
            unique_count=unique_count,
            null_percentage=0.0,  # Will be calculated in __post_init__
            unique_percentage=0.0  # Will be calculated in __post_init__
        )
        
        if non_null_count == 0:
            return stats
        
        # Numeric analysis
        if pd.api.types.is_numeric_dtype(data):
            stats.min_value = float(non_null_data.min())
            stats.max_value = float(non_null_data.max())
            stats.mean_value = float(non_null_data.mean())
            stats.median_value = float(non_null_data.median())
            stats.std_deviation = float(non_null_data.std())
            
            # Quartiles
            quartiles = non_null_data.quantile([0.25, 0.5, 0.75])
            stats.quartiles = {
                'q1': float(quartiles[0.25]),
                'q2': float(quartiles[0.5]),
                'q3': float(quartiles[0.75])
            }
            
            # Detect outliers using IQR method
            q1, q3 = quartiles[0.25], quartiles[0.75]
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = non_null_data[(non_null_data < lower_bound) | (non_null_data > upper_bound)]
            stats.outliers = outliers.tolist() if len(outliers) <= 10 else outliers.head(10).tolist()
        
        # String analysis
        elif pd.api.types.is_string_dtype(data) or data.dtype == 'object':
            str_data = non_null_data.astype(str)
            lengths = str_data.str.len()
            
            stats.min_length = int(lengths.min())
            stats.max_length = int(lengths.max())
            stats.avg_length = float(lengths.mean())
            
            # Pattern detection
            stats.patterns = self._detect_patterns(str_data.head(self.sample_size))
        
        # Date analysis
        elif pd.api.types.is_datetime64_any_dtype(data):
            stats.min_date = pd.to_datetime(non_null_data.min())
            stats.max_date = pd.to_datetime(non_null_data.max())
        
        # Frequency analysis
        value_counts = data.value_counts().head(10)
        stats.most_frequent = [
            {'value': str(val), 'count': int(count), 'percentage': float(count / total_rows * 100)}
            for val, count in value_counts.items()
        ]
        
        # Least frequent (only for non-unique values)
        if unique_count < total_rows:
            least_frequent = data.value_counts().tail(10)
            stats.least_frequent = [
                {'value': str(val), 'count': int(count), 'percentage': float(count / total_rows * 100)}
                for val, count in least_frequent.items()
            ]
        
        return stats
    
    def _detect_patterns(self, str_data: pd.Series) -> List[Dict[str, Any]]:
        """Detect common patterns in string data.
        
        Args:
            str_data: Series of string values to analyze
            
        Returns:
            List of detected patterns with confidence scores
        """
        patterns = []
        total_count = len(str_data)
        
        if total_count == 0:
            return patterns
        
        for pattern_type, pattern_info in self.PATTERNS.items():
            regex = re.compile(pattern_info['regex'], re.IGNORECASE)
            matches = str_data.str.match(regex, na=False)
            match_count = matches.sum()
            
            if match_count > 0:
                confidence = match_count / total_count
                # Only include patterns with reasonable confidence
                if confidence >= 0.1:  # At least 10% match
                    examples = str_data[matches].head(3).tolist()
                    patterns.append({
                        'pattern_type': pattern_type,
                        'pattern_name': pattern_info['name'],
                        'regex_pattern': pattern_info['regex'],
                        'confidence': confidence,
                        'match_count': int(match_count),
                        'match_percentage': float(confidence * 100),
                        'examples': examples
                    })
        
        # Sort by confidence
        return sorted(patterns, key=lambda x: x['confidence'], reverse=True)
    
    def profile_dataframe(
        self, 
        df: pd.DataFrame, 
        table_name: str,
        database_name: str,
        schema_name: Optional[str] = None
    ) -> TableProfile:
        """Profile a complete DataFrame.
        
        Args:
            df: DataFrame to profile
            table_name: Name of the table
            database_name: Name of the database
            schema_name: Optional schema name
            
        Returns:
            TableProfile with complete analysis
        """
        start_time = datetime.now()
        
        # Analyze each column
        column_stats = {}
        for column_name in df.columns:
            try:
                column_stats[column_name] = self.analyze_column(df[column_name], column_name)
            except Exception as e:
                # Log error but continue with other columns
                print(f"Warning: Failed to analyze column '{column_name}': {e}")
                continue
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Generate warnings and recommendations
        warnings, recommendations = self._generate_insights(column_stats, df)
        
        profile = TableProfile(
            table_name=table_name,
            schema_name=schema_name,
            database_name=database_name,
            total_rows=len(df),
            total_columns=len(df.columns),
            profile_timestamp=start_time,
            execution_time=execution_time,
            columns=column_stats,
            primary_keys=[],  # Will be enhanced with actual metadata
            foreign_keys=[],  # Will be enhanced with actual metadata
            indexes=[],       # Will be enhanced with actual metadata
            completeness_score=0.0,  # Calculated in __post_init__
            uniqueness_score=0.0,    # Calculated in __post_init__
            validity_score=0.0,      # Calculated in __post_init__
            consistency_score=0.0,   # Calculated in __post_init__
            overall_score=0.0,       # Calculated in __post_init__
            warnings=warnings,
            recommendations=recommendations
        )
        
        return profile
    
    def profile_query_result(
        self, 
        df: pd.DataFrame, 
        query: str,
        execution_time: float
    ) -> QueryProfile:
        """Profile the results of a SQL query.
        
        Args:
            df: DataFrame containing query results
            query: The SQL query that was executed
            execution_time: Time taken to execute the query
            
        Returns:
            QueryProfile with analysis results
        """
        start_time = datetime.now()
        
        # Generate query hash
        query_hash = hashlib.md5(query.encode()).hexdigest()
        
        # Analyze each column
        column_stats = {}
        for column_name in df.columns:
            try:
                column_stats[column_name] = self.analyze_column(df[column_name], column_name)
            except Exception as e:
                print(f"Warning: Failed to analyze column '{column_name}': {e}")
                continue
        
        profile = QueryProfile(
            query=query,
            query_hash=query_hash,
            execution_time=execution_time,
            rows_returned=len(df),
            columns_returned=len(df.columns),
            profile_timestamp=start_time,
            columns=column_stats
        )
        
        return profile
    
    def _generate_insights(
        self, 
        column_stats: Dict[str, ColumnStatistics], 
        df: pd.DataFrame
    ) -> tuple[List[str], List[str]]:
        """Generate warnings and recommendations based on analysis.
        
        Args:
            column_stats: Dictionary of column statistics
            df: Original DataFrame
            
        Returns:
            Tuple of (warnings, recommendations)
        """
        warnings = []
        recommendations = []
        
        for col_name, stats in column_stats.items():
            # High null percentage warning
            if stats.null_percentage > 50:
                warnings.append(f"Column '{col_name}' has {stats.null_percentage:.1f}% null values")
                recommendations.append(f"Consider investigating missing data in column '{col_name}'")
            
            # Low uniqueness for potential ID columns
            if 'id' in col_name.lower() and stats.unique_percentage < 95:
                warnings.append(f"ID column '{col_name}' has only {stats.unique_percentage:.1f}% unique values")
                recommendations.append(f"Verify data integrity for ID column '{col_name}'")
            
            # Very long strings
            if stats.max_length and stats.max_length > 1000:
                warnings.append(f"Column '{col_name}' has very long text values (max: {stats.max_length} chars)")
                recommendations.append(f"Consider if column '{col_name}' needs text length constraints")
            
            # Potential data quality issues
            if stats.patterns:
                for pattern in stats.patterns:
                    if pattern['confidence'] < 0.8 and pattern['match_count'] > 10:
                        warnings.append(
                            f"Column '{col_name}' has inconsistent {pattern['pattern_name']} formatting "
                            f"({pattern['match_percentage']:.1f}% match)"
                        )
                        recommendations.append(f"Standardize {pattern['pattern_name']} format in column '{col_name}'")
        
        return warnings, recommendations
    
    def compare_profiles(self, profile1: TableProfile, profile2: TableProfile) -> ComparisonResult:
        """Compare two table profiles to detect changes.
        
        Args:
            profile1: First profile (baseline)
            profile2: Second profile (comparison)
            
        Returns:
            ComparisonResult with detailed comparison
        """
        cols1 = set(profile1.columns.keys())
        cols2 = set(profile2.columns.keys())
        
        added_columns = list(cols2 - cols1)
        removed_columns = list(cols1 - cols2)
        common_columns = cols1 & cols2
        
        # Analyze changes in common columns
        modified_columns = []
        column_changes = {}
        
        for col_name in common_columns:
            stats1 = profile1.columns[col_name]
            stats2 = profile2.columns[col_name]
            
            changes = {}
            
            # Check for significant changes
            if abs(stats1.null_percentage - stats2.null_percentage) > 5:
                changes['null_percentage'] = {
                    'before': stats1.null_percentage,
                    'after': stats2.null_percentage,
                    'change': stats2.null_percentage - stats1.null_percentage
                }
            
            if abs(stats1.unique_percentage - stats2.unique_percentage) > 5:
                changes['unique_percentage'] = {
                    'before': stats1.unique_percentage,
                    'after': stats2.unique_percentage,
                    'change': stats2.unique_percentage - stats1.unique_percentage
                }
            
            if changes:
                modified_columns.append({'column': col_name, 'changes': changes})
                column_changes[col_name] = changes
        
        # Calculate row count change
        row_count_change = profile2.total_rows - profile1.total_rows
        row_count_change_percentage = (
            (row_count_change / profile1.total_rows * 100) if profile1.total_rows > 0 else 0
        )
        
        # Calculate stability scores
        schema_changes = len(added_columns) + len(removed_columns)
        schema_stability = max(0, 100 - (schema_changes * 10))  # Penalize schema changes
        
        data_stability = max(0, 100 - abs(row_count_change_percentage))  # Penalize data changes
        
        overall_stability = (schema_stability + data_stability) / 2
        
        return ComparisonResult(
            profile1_name=f"{profile1.table_name}@{profile1.profile_timestamp}",
            profile2_name=f"{profile2.table_name}@{profile2.profile_timestamp}",
            comparison_timestamp=datetime.now(),
            added_columns=added_columns,
            removed_columns=removed_columns,
            modified_columns=modified_columns,
            row_count_change=row_count_change,
            row_count_change_percentage=row_count_change_percentage,
            column_changes=column_changes,
            schema_stability=schema_stability,
            data_stability=data_stability,
            overall_stability=overall_stability
        )

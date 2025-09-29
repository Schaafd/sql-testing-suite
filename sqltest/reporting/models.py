"""Data models for report generation and representation."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field, validator


class ReportFormat(str, Enum):
    """Supported report output formats."""
    JSON = "json"
    HTML = "html"
    CSV = "csv"
    PDF = "pdf"
    EXCEL = "excel"


class ReportType(str, Enum):
    """Types of reports that can be generated."""
    SUMMARY = "summary"
    DETAILED = "detailed"
    EXECUTIVE = "executive"
    TECHNICAL = "technical"
    TREND_ANALYSIS = "trend_analysis"
    PERFORMANCE = "performance"
    COMPLIANCE = "compliance"


class SeverityLevel(str, Enum):
    """Severity levels for findings and issues."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class ReportMetadata:
    """Metadata information for reports."""
    title: str
    description: str
    generated_at: datetime
    generated_by: str
    version: str = "1.0"
    tags: List[str] = field(default_factory=list)
    custom_fields: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataSource:
    """Information about data sources used in the report."""
    name: str
    type: str
    connection_string: Optional[str] = None
    query_count: int = 0
    last_accessed: Optional[datetime] = None
    schema_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionMetrics:
    """Metrics about report execution and performance."""
    execution_time: float
    memory_usage: float
    queries_executed: int
    rows_processed: int
    cache_hit_rate: float = 0.0
    errors_encountered: int = 0


@dataclass
class Finding:
    """Represents a finding or issue discovered during analysis."""
    id: str
    title: str
    description: str
    severity: SeverityLevel
    category: str
    details: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    affected_objects: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ChartData:
    """Data structure for chart generation."""
    chart_type: str
    title: str
    data: Dict[str, Any]
    options: Dict[str, Any] = field(default_factory=dict)
    width: Optional[int] = None
    height: Optional[int] = None


@dataclass
class ReportSection:
    """A section within a report."""
    id: str
    title: str
    content: str
    order: int
    subsections: List['ReportSection'] = field(default_factory=list)
    charts: List[ChartData] = field(default_factory=list)
    tables: List[pd.DataFrame] = field(default_factory=list)
    findings: List[Finding] = field(default_factory=list)


class ReportConfiguration(BaseModel):
    """Configuration for report generation."""
    report_type: ReportType
    format: ReportFormat
    title: str
    description: Optional[str] = None
    output_path: Optional[Path] = None
    template_name: Optional[str] = None
    include_sections: List[str] = Field(default_factory=list)
    exclude_sections: List[str] = Field(default_factory=list)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    styling: Dict[str, Any] = Field(default_factory=dict)

    @validator('output_path', pre=True)
    def convert_output_path(cls, v):
        if v is not None and not isinstance(v, Path):
            return Path(v)
        return v


@dataclass
class ReportData:
    """Main data structure containing all report information."""
    metadata: ReportMetadata
    configuration: ReportConfiguration
    data_sources: List[DataSource] = field(default_factory=list)
    execution_metrics: ExecutionMetrics = field(default_factory=lambda: ExecutionMetrics(0.0, 0.0, 0, 0))
    sections: List[ReportSection] = field(default_factory=list)
    findings: List[Finding] = field(default_factory=list)
    summary_statistics: Dict[str, Any] = field(default_factory=dict)
    raw_data: Dict[str, pd.DataFrame] = field(default_factory=dict)

    def add_section(self, section: ReportSection) -> None:
        """Add a section to the report."""
        self.sections.append(section)
        self.sections.sort(key=lambda x: x.order)

    def get_section(self, section_id: str) -> Optional[ReportSection]:
        """Get a section by ID."""
        for section in self.sections:
            if section.id == section_id:
                return section
        return None

    def add_finding(self, finding: Finding) -> None:
        """Add a finding to the report."""
        self.findings.append(finding)

    def get_findings_by_severity(self, severity: SeverityLevel) -> List[Finding]:
        """Get all findings of a specific severity level."""
        return [f for f in self.findings if f.severity == severity]

    def get_critical_findings(self) -> List[Finding]:
        """Get all critical findings."""
        return self.get_findings_by_severity(SeverityLevel.CRITICAL)


@dataclass
class ReportTemplate:
    """Template configuration for report generation."""
    name: str
    format: ReportFormat
    template_path: Path
    variables: Dict[str, Any] = field(default_factory=dict)
    includes: List[str] = field(default_factory=list)
    custom_functions: Dict[str, Any] = field(default_factory=dict)


class ReportOptions(BaseModel):
    """Options for report generation and formatting."""
    include_charts: bool = True
    include_raw_data: bool = False
    include_executive_summary: bool = True
    max_rows_per_table: int = 1000
    chart_theme: str = "default"
    color_scheme: str = "blue"
    font_family: str = "Arial, sans-serif"
    show_timestamps: bool = True
    include_metadata: bool = True
    compress_output: bool = False

    class Config:
        extra = "allow"


@dataclass
class ReportGenerationResult:
    """Result of report generation process."""
    success: bool
    output_path: Optional[Path] = None
    format: Optional[ReportFormat] = None
    file_size: Optional[int] = None
    generation_time: Optional[float] = None
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
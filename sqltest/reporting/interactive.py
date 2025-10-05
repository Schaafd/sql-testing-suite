"""Interactive report features and dashboard components."""

import json
from datetime import datetime, timedelta, date
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import numpy as np

from .models import (
    ReportData, ReportSection, ChartData, Finding, SeverityLevel
)


@dataclass
class InteractiveWidget:
    """Base class for interactive dashboard widgets."""
    widget_id: str
    title: str
    widget_type: str
    data: Dict[str, Any]
    options: Dict[str, Any] = field(default_factory=dict)
    position: Dict[str, int] = field(default_factory=dict)  # x, y, width, height

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access for compatibility with legacy tests."""
        if key == "id":
            return self.widget_id
        if key == "title":
            return self.title
        if key == "type":
            return self.widget_type
        if key == "data":
            return self.data.get("data", self.data)
        if key == "options":
            return self.options
        if key in self.data:
            return self.data[key]
        if key in self.options:
            return self.options[key]
        if key == "position":
            return self.position
        raise KeyError(key)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "id": self.widget_id,
            "title": self.title,
            "type": self.widget_type,
            "position": self.position,
        }
        payload["options"] = self.options
        payload["data"] = self.data
        payload.update(self.options)
        payload.update(self.data)
        return payload


@dataclass
class FilterConfig:
    """Configuration for report filters."""
    filter_id: str
    label: str
    filter_type: str  # select, range, date, text
    column: Optional[str] = None
    options: List[Any] = field(default_factory=list)
    default_value: Any = None
    extras: Dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.filter_id

    @property
    def type(self) -> str:
        return self.filter_type

    def __getitem__(self, key: str) -> Any:
        mapping = {
            "id": self.filter_id,
            "filter_id": self.filter_id,
            "label": self.label,
            "type": self.filter_type,
            "filter_type": self.filter_type,
            "column": self.column,
            "options": self.options,
            "default_value": self.default_value,
        }
        if key in mapping:
            return mapping[key]
        if key in self.extras:
            return self.extras[key]
        raise KeyError(key)


class InteractiveReportBuilder:
    """Builder for creating interactive dashboard-style reports."""

    def __init__(self):
        """Initialize the interactive report builder."""
        self.widgets: List[InteractiveWidget] = []
        self.filters: List[FilterConfig] = []
        self.layout_config: Dict[str, Any] = {}

    def add_metric_card(
        self,
        widget_id: str,
        title: str,
        value: Union[int, float, str],
        subtitle: Optional[str] = None,
        trend: Optional[float] = None,
        color: str = "primary",
    ) -> 'InteractiveReportBuilder':
        """Add a metric card widget.

        Args:
            widget_id: Unique identifier for the widget
            title: Card title
            value: Main metric value
            subtitle: Optional subtitle
            trend: Optional trend percentage
            color: Card color theme

        Returns:
            Self for method chaining
        """
        widget = InteractiveWidget(
            widget_id=widget_id,
            title=title,
            widget_type="metric",
            data={
                "value": value,
                "subtitle": subtitle,
                "trend": trend,
                "color": color,
            },
            options={
                "display": "metric",
                "trend_direction": "positive" if (trend or 0) >= 0 else "negative",
            },
        )
        self.widgets.append(widget)
        return self

    def add_chart_widget(
        self,
        widget_id: str,
        title: str,
        chart_type: str,
        data: Dict[str, Any],
        options: Optional[Dict[str, Any]] = None,
    ) -> 'InteractiveReportBuilder':
        """Add a chart widget.

        Args:
            widget_id: Unique identifier for the widget
            title: Chart title
            chart_type: Type of chart (bar, line, pie, etc.)
            data: Chart data
            options: Chart options

        Returns:
            Self for method chaining
        """
        widget = InteractiveWidget(
            widget_id=widget_id,
            title=title,
            widget_type="chart",
            data=data,
            options={"chart_type": chart_type, **(options or {})},
        )
        self.widgets.append(widget)
        return self

    def add_table_widget(self, widget_id: str, title: str, dataframe: pd.DataFrame,
                        paginate: bool = True, searchable: bool = True,
                        sortable: bool = True) -> 'InteractiveReportBuilder':
        """Add a data table widget.

        Args:
            widget_id: Unique identifier for the widget
            title: Table title
            dataframe: Data to display
            paginate: Enable pagination
            searchable: Enable search functionality
            sortable: Enable column sorting

        Returns:
            Self for method chaining
        """
        widget = InteractiveWidget(
            widget_id=widget_id,
            title=title,
            widget_type="table",
            data={
                "columns": dataframe.columns.tolist(),
                "data": dataframe.to_dict('records'),
                "total_rows": len(dataframe),
            },
            options={
                "paginate": paginate,
                "searchable": searchable,
                "sortable": sortable,
                "page_size": 25,
            },
        )
        self.widgets.append(widget)
        return self

    def add_data_table_widget(self, widget_id: str, title: str, data: List[Dict[str, Any]],
                             paginate: bool = True, searchable: bool = True,
                             sortable: bool = True) -> 'InteractiveReportBuilder':
        """Add a data table widget from list of dictionaries.

        Args:
            widget_id: Unique identifier for the widget
            title: Table title
            data: List of dictionaries representing table data
            paginate: Enable pagination
            searchable: Enable search functionality
            sortable: Enable column sorting

        Returns:
            Self for method chaining
        """
        if not data:
            columns: List[str] = []
        else:
            columns = list(data[0].keys())

        widget = InteractiveWidget(
            widget_id=widget_id,
            title=title,
            widget_type="table",
            data={
                "columns": columns,
                "data": data,
                "total_rows": len(data),
            },
            options={
                "paginate": paginate,
                "searchable": searchable,
                "sortable": sortable,
                "page_size": 25,
            },
        )
        self.widgets.append(widget)
        return self

    def add_progress_widget(self, widget_id: str, title: str, value: Union[int, float],
                           max_value: Union[int, float] = 100, color: str = "primary") -> 'InteractiveReportBuilder':
        """Add a progress bar widget.

        Args:
            widget_id: Unique identifier for the widget
            title: Progress bar title
            value: Current progress value
            max_value: Maximum value for progress calculation
            color: Progress bar color theme

        Returns:
            Self for method chaining
        """
        widget = InteractiveWidget(
            widget_id=widget_id,
            title=title,
            widget_type="progress",
            data={
                "value": value,
                "max_value": max_value,
            },
            options={"color": color},
        )
        self.widgets.append(widget)
        return self

    def add_kpi_grid(self, widget_id: str, title: str, kpis: List[Dict[str, Any]]) -> 'InteractiveReportBuilder':
        """Add a KPI grid widget.

        Args:
            widget_id: Unique identifier for the widget
            title: Grid title
            kpis: List of KPI dictionaries with keys: name, value, target, unit

        Returns:
            Self for method chaining
        """
        widget = InteractiveWidget(
            widget_id=widget_id,
            title=title,
            widget_type="kpi_grid",
            data={"kpis": kpis}
        )
        self.widgets.append(widget)
        return self

    def add_filter(self, filter_id: str, label: str, filter_type: str,
                  options_or_column: Union[List[Any], str, Dict[str, Any]],
                  default_value: Any = None) -> 'InteractiveReportBuilder':
        """Add a filter control.

        Args:
            filter_id: Unique identifier for the filter
            label: Filter label
            filter_type: Type of filter (select, range, date, text, dropdown, daterange)
            options_or_column: Options list, column name, or config dict depending on filter type
            default_value: Default filter value

        Returns:
            Self for method chaining
        """
        # Convert to expected format for tests
        column: Optional[str] = None
        options: List[Any] = []
        extras: Dict[str, Any] = {}

        if isinstance(options_or_column, list):
            options = options_or_column
        elif isinstance(options_or_column, str):
            column = options_or_column
        elif isinstance(options_or_column, dict):
            extras = dict(options_or_column)
            options = extras.pop('options', extras.get('values', []))
            column = extras.get('column', extras.get('field'))

        filter_obj = FilterConfig(
            filter_id=filter_id,
            label=label,
            filter_type=filter_type,
            column=column,
            options=options,
            default_value=default_value,
            extras=extras,
        )

        self.filters.append(filter_obj)
        return self

    def set_layout(self, layout_type: str = "grid", columns: int = 12,
                  responsive: bool = True) -> 'InteractiveReportBuilder':
        """Set layout configuration.

        Args:
            layout_type: Type of layout (grid, flow)
            columns: Number of grid columns
            responsive: Enable responsive layout

        Returns:
            Self for method chaining
        """
        self.layout_config = {
            "type": layout_type,
            "columns": columns,
            "responsive": responsive
        }
        return self

    def generate_javascript(self) -> str:
        """Generate JavaScript code for interactive features.

        Returns:
            JavaScript code string
        """
        widgets_payload = self._serialize_for_js([widget.to_dict() for widget in self.widgets])
        filters_payload = self._serialize_for_js([
            {
                "id": filter_obj.filter_id,
                "label": filter_obj.label,
                "type": filter_obj.filter_type,
                "column": filter_obj.column,
                "options": filter_obj.options,
                "default_value": filter_obj.default_value,
                **filter_obj.extras,
            }
            for filter_obj in self.filters
        ])

        prefix = (
            "// Interactive Report JavaScript\n"
            f"const INTERACTIVE_WIDGETS = {widgets_payload};\n"
            f"const INTERACTIVE_FILTERS = {filters_payload};\n"
        )

        js_body = """
class InteractiveReport {
    constructor() {
        this.charts = {};
        this.tables = {};
        this.filters = {};
        this.data = {};
        this.init();
    }

    init() {
        this.setupFilters();
        this.setupCharts();
        this.setupTables();
        this.setupEventListeners();
    }

    setupFilters() {
        // Initialize filter controls
        document.querySelectorAll('.report-filter').forEach(filter => {
            const filterId = filter.dataset.filterId;
            const filterType = filter.dataset.filterType;

            this.filters[filterId] = {
                element: filter,
                type: filterType,
                column: filter.dataset.column
            };

            // Add event listeners
            filter.addEventListener('change', (e) => {
                saveFilterPreferences();
                this.applyFilters();
            });
        });
    }

    setupCharts() {
        // Initialize Chart.js charts
        document.querySelectorAll('.chart-widget').forEach(chartElement => {
            const widgetId = chartElement.dataset.widgetId;
            const chartData = JSON.parse(chartElement.dataset.chartData);
            const chartOptions = JSON.parse(chartElement.dataset.chartOptions || '{}');

            const canvas = chartElement.querySelector('canvas');
            if (canvas) {
                this.charts[widgetId] = new Chart(canvas, {
                    type: chartOptions.chart_type || 'bar',
                    data: chartData,
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        ...chartOptions
                    }
                });
            }
        });
    }

    setupTables() {
        // Initialize DataTables
        document.querySelectorAll('.table-widget').forEach(tableElement => {
            const widgetId = tableElement.dataset.widgetId;
            const tableOptions = JSON.parse(tableElement.dataset.tableOptions || '{}');

            const table = tableElement.querySelector('table');
            if (table && typeof $ !== 'undefined' && $.fn.DataTable) {
                this.tables[widgetId] = $(table).DataTable({
                    responsive: true,
                    searching: tableOptions.searchable !== false,
                    paging: tableOptions.paginate !== false,
                    ordering: tableOptions.sortable !== false,
                    pageLength: tableOptions.page_size || 25,
                    dom: '<"row"<"col-sm-12 col-md-6"l><"col-sm-12 col-md-6"f>>' +
                         '<"row"<"col-sm-12"tr>>' +
                         '<"row"<"col-sm-12 col-md-5"i><"col-sm-12 col-md-7"p>>',
                    language: {
                        search: "Search:",
                        lengthMenu: "Show _MENU_ entries",
                        info: "Showing _START_ to _END_ of _TOTAL_ entries",
                        paginate: {
                            first: "First",
                            last: "Last",
                            next: "Next",
                            previous: "Previous"
                        }
                    }
                });
            }
        });
    }

    setupEventListeners() {
        // Refresh button
        document.querySelectorAll('.refresh-widget').forEach(button => {
            button.addEventListener('click', (e) => {
                const widgetId = e.target.dataset.widgetId;
                this.refreshWidget(widgetId);
            });
        });

        // Export buttons
        document.querySelectorAll('.export-widget').forEach(button => {
            button.addEventListener('click', (e) => {
                const widgetId = e.target.dataset.widgetId;
                const format = e.target.dataset.format;
                this.exportWidget(widgetId, format);
            });
        });

        // Full screen toggle
        document.querySelectorAll('.fullscreen-widget').forEach(button => {
            button.addEventListener('click', (e) => {
                const widgetId = e.target.dataset.widgetId;
                this.toggleFullscreen(widgetId);
            });
        });
    }

    applyFilters() {
        // Apply all active filters to widgets
        const activeFilters = {};

        Object.keys(this.filters).forEach(filterId => {
            const filter = this.filters[filterId];
            const value = this.getFilterValue(filter);

            if (value !== null && value !== '') {
                activeFilters[filter.column] = {
                    type: filter.type,
                    value: value
                };
            }
        });

        // Update charts and tables based on filters
        this.updateWidgets(activeFilters);
    }

    getFilterValue(filter) {
        const element = filter.element;

        switch (filter.type) {
            case 'select':
                return element.value;
            case 'range':
                const min = element.querySelector('.range-min').value;
                const max = element.querySelector('.range-max').value;
                return { min: min, max: max };
            case 'date':
                return element.value;
            case 'text':
                return element.value;
            default:
                return element.value;
        }
    }

    updateWidgets(filters) {
        // Update charts
        Object.keys(this.charts).forEach(widgetId => {
            this.updateChartData(widgetId, filters);
        });

        // Update tables
        Object.keys(this.tables).forEach(widgetId => {
            this.updateTableData(widgetId, filters);
        });
    }

    updateChartData(widgetId, filters) {
        // This would typically make an AJAX call to get filtered data
        // For now, we'll just log the action
        console.log(`Updating chart ${widgetId} with filters:`, filters);
    }

    updateTableData(widgetId, filters) {
        // Apply filters to DataTable
        const table = this.tables[widgetId];
        if (table) {
            // Clear existing search
            table.search('').columns().search('').draw();

            // Apply new filters
            Object.keys(filters).forEach(column => {
                const filter = filters[column];
                const columnIndex = table.column(`${column}:name`).index();

                if (columnIndex >= 0) {
                    if (filter.type === 'text') {
                        table.column(columnIndex).search(filter.value);
                    }
                    // Add more filter types as needed
                }
            });

            table.draw();
        }
    }

    refreshWidget(widgetId) {
        console.log(`Refreshing widget: ${widgetId}`);
        // Implement widget refresh logic
        // This would typically reload data from the server
    }

    exportWidget(widgetId, format) {
        console.log(`Exporting widget ${widgetId} as ${format}`);

        if (format === 'csv' && this.tables[widgetId]) {
            // Export table data as CSV
            const table = this.tables[widgetId];
            const data = table.data().toArray();
            this.downloadCSV(data, `${widgetId}_export.csv`);
        } else if (format === 'png' && this.charts[widgetId]) {
            // Export chart as PNG
            const chart = this.charts[widgetId];
            const url = chart.toBase64Image();
            this.downloadImage(url, `${widgetId}_chart.png`);
        }
    }

    downloadCSV(data, filename) {
        const csv = this.arrayToCSV(data);
        const blob = new Blob([csv], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.click();
        window.URL.revokeObjectURL(url);
    }

    downloadImage(dataURL, filename) {
        const a = document.createElement('a');
        a.href = dataURL;
        a.download = filename;
        a.click();
    }

    arrayToCSV(data) {
        if (!data || data.length === 0) return '';

        const headers = Object.keys(data[0]);
        const csvContent = [
            headers.join(','),
            ...data.map(row => headers.map(header => {
                const value = row[header];
                return typeof value === 'string' ? `"${value.replace(/"/g, '""')}"` : value;
            }).join(','))
        ].join('\\n');

        return csvContent;
    }

    toggleFullscreen(widgetId) {
        const widget = document.querySelector(`[data-widget-id="${widgetId}"]`);
        if (widget) {
            widget.classList.toggle('fullscreen-widget-active');

            // Resize charts after fullscreen toggle
            if (this.charts[widgetId]) {
                setTimeout(() => {
                    this.charts[widgetId].resize();
                }, 300);
            }
        }
    }

    // Real-time updates
    startRealTimeUpdates(intervalSeconds = 300) {
        setInterval(() => {
            this.updateAllWidgets();
        }, intervalSeconds * 1000);
    }

    updateAllWidgets() {
        // Refresh all widgets with latest data
        Object.keys(this.charts).forEach(widgetId => {
            this.refreshWidget(widgetId);
        });
    }
}

// Initialize when DOM is ready
function initializeDashboard() {
    window.interactiveReport = new InteractiveReport();
    restoreFilterPreferences();
}

document.addEventListener('DOMContentLoaded', initializeDashboard);

// Utility functions
function formatNumber(value, decimals = 0) {
    return Number(value).toLocaleString(undefined, {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals
    });
}

function formatPercentage(value, decimals = 1) {
    return `${(value * 100).toFixed(decimals)}%`;
}

function formatCurrency(value, currency = 'USD') {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: currency
    }).format(value);
}

function clearAllFilters() {
    document.querySelectorAll('.report-filter').forEach(filter => {
        if (filter.tagName === 'SELECT') {
            filter.selectedIndex = 0;
        } else {
            filter.value = '';
        }
    });

    document.querySelectorAll('.range-min').forEach(input => input.value = '');
    document.querySelectorAll('.range-max').forEach(input => input.value = '');

    saveFilterPreferences();
    if (window.interactiveReport) {
        window.interactiveReport.applyFilters();
    }
}

function saveFilterPreferences() {
    if (!window.localStorage) return;

    const preferences = {};
    document.querySelectorAll('.report-filter').forEach(filter => {
        preferences[filter.dataset.filterId] = filter.value;
    });

    try {
        localStorage.setItem('interactiveDashboardFilters', JSON.stringify(preferences));
    } catch (err) {
        console.warn('Unable to save filter preferences', err);
    }
}

function restoreFilterPreferences() {
    if (!window.localStorage) return;

    try {
        const preferences = JSON.parse(localStorage.getItem('interactiveDashboardFilters') || '{}');
        Object.keys(preferences).forEach(filterId => {
            const element = document.querySelector(`[data-filter-id="${filterId}"]`);
            if (element) {
                element.value = preferences[filterId];
            }
        });
    } catch (err) {
        console.warn('Unable to restore filter preferences', err);
    }
}

function exportDashboard() {
    if (window.interactiveReport) {
        window.interactiveReport.updateAllWidgets();
    }
    window.print();
}

function toggleFullscreenDashboard() {
    const container = document.querySelector('.interactive-dashboard');
    if (container) {
        container.classList.toggle('dashboard-fullscreen');
    }
}
        """

        return prefix + js_body

    def generate_css(self) -> str:
        """Generate CSS styles for interactive features.

        Returns:
            CSS code string
        """
        css_code = """
/* Interactive Report Styles */
.interactive-dashboard {
    padding: 20px;
    background: #f8f9fa;
    min-height: 100vh;
}

.dashboard-fullscreen {
    position: fixed;
    inset: 0;
    z-index: 9998;
    background: #f8f9fa;
    overflow: auto;
    padding: 30px;
}

.dashboard-nav {
    border-radius: 8px;
}

.dashboard-header {
    background: white;
    padding: 20px;
    border-radius: 8px;
    margin-bottom: 20px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.dashboard-filters {
    background: white;
    padding: 15px 20px;
    border-radius: 8px;
    margin-bottom: 20px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.filter-group {
    display: flex;
    gap: 15px;
    flex-wrap: wrap;
    align-items: end;
}

.filter-control {
    flex: 1;
    min-width: 200px;
}

.filter-control label {
    display: block;
    margin-bottom: 5px;
    font-weight: 500;
    color: #495057;
}

.filter-control input,
.filter-control select {
    width: 100%;
    padding: 8px 12px;
    border: 1px solid #ced4da;
    border-radius: 4px;
    font-size: 14px;
}

.widget-container {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 20px;
    margin-bottom: 20px;
}

.widget-card {
    background: white;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    overflow: hidden;
    transition: transform 0.2s, box-shadow 0.2s;
}

.widget-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}

.widget-header {
    padding: 15px 20px;
    border-bottom: 1px solid #e9ecef;
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #f8f9fa;
}

.widget-title {
    font-size: 16px;
    font-weight: 600;
    color: #495057;
    margin: 0;
}

.widget-actions {
    display: flex;
    gap: 8px;
}

.widget-action-btn {
    background: none;
    border: none;
    color: #6c757d;
    cursor: pointer;
    padding: 4px;
    border-radius: 4px;
    transition: color 0.2s, background 0.2s;
}

.widget-action-btn:hover {
    color: #495057;
    background: #e9ecef;
}

.widget-content {
    padding: 20px;
}

/* Metric Card Styles */
.metric-card {
    text-align: center;
}

.metric-value {
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 8px;
    color: #0d6efd;
}

.metric-value.success { color: #198754; }
.metric-value.warning { color: #ffc107; }
.metric-value.danger { color: #dc3545; }
.metric-value.info { color: #0dcaf0; }

.metric-subtitle {
    color: #6c757d;
    font-size: 14px;
    margin-bottom: 10px;
}

.metric-trend {
    font-size: 12px;
    padding: 4px 8px;
    border-radius: 12px;
    display: inline-block;
}

.metric-trend.positive {
    background: #d1e7dd;
    color: #0f5132;
}

.metric-trend.negative {
    background: #f8d7da;
    color: #842029;
}

/* Chart Widget Styles */
.chart-widget .widget-content {
    position: relative;
    height: 300px;
}

.chart-container {
    position: relative;
    width: 100%;
    min-height: 260px;
}

.chart-widget canvas {
    max-height: 100%;
}

/* Table Widget Styles */
.table-widget .widget-content {
    padding: 0;
}

.table-widget table {
    margin: 0;
}

.table-widget .dataTables_wrapper {
    padding: 20px;
}

/* KPI Grid Styles */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 15px;
}

.kpi-item {
    text-align: center;
    padding: 15px;
    border: 1px solid #e9ecef;
    border-radius: 6px;
    background: #f8f9fa;
}

.kpi-name {
    font-size: 12px;
    color: #6c757d;
    margin-bottom: 5px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.kpi-value {
    font-size: 20px;
    font-weight: 600;
    color: #495057;
    margin-bottom: 5px;
}

.kpi-target {
    font-size: 11px;
    color: #6c757d;
}

.kpi-progress {
    width: 100%;
    height: 4px;
    background: #e9ecef;
    border-radius: 2px;
    margin-top: 8px;
    overflow: hidden;
}

.kpi-progress-bar {
    height: 100%;
    background: #0d6efd;
    transition: width 0.3s ease;
}

/* Fullscreen Widget */
.fullscreen-widget-active {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    z-index: 9999;
    background: white;
    overflow: auto;
}

.fullscreen-widget-active .widget-content {
    height: calc(100vh - 80px);
}

/* Responsive Design */
@media (max-width: 768px) {
    .widget-container {
        grid-template-columns: 1fr;
    }

    .filter-group {
        flex-direction: column;
    }

    .filter-control {
        min-width: unset;
    }

    .dashboard-header,
    .dashboard-filters {
        margin-left: -10px;
        margin-right: -10px;
        border-radius: 0;
    }

    .interactive-dashboard {
        padding: 10px;
    }
}

/* Dark Mode Support */
@media (prefers-color-scheme: dark) {
    .interactive-dashboard {
        background: #212529;
        color: #f8f9fa;
    }

    .widget-card,
    .dashboard-header,
    .dashboard-filters {
        background: #343a40;
        color: #f8f9fa;
    }

    .widget-header {
        background: #495057;
        border-bottom-color: #6c757d;
    }

    .filter-control input,
    .filter-control select {
        background: #495057;
        border-color: #6c757d;
        color: #f8f9fa;
    }

    .kpi-item {
        background: #495057;
        border-color: #6c757d;
    }
}

/* Loading States */
.widget-loading {
    opacity: 0.6;
    pointer-events: none;
}

.widget-loading::after {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    width: 20px;
    height: 20px;
    margin: -10px 0 0 -10px;
    border: 2px solid #0d6efd;
    border-top-color: transparent;
    border-radius: 50%;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

/* Print Styles */
@media print {
    .widget-actions,
    .dashboard-filters {
        display: none;
    }

    .widget-card {
        break-inside: avoid;
        box-shadow: none;
        border: 1px solid #ccc;
    }

    .interactive-dashboard {
        background: white;
    }
}
        """

        return css_code

    def _serialize_for_js(self, payload: Any) -> str:
        """Serialize Python objects for embedding into JavaScript."""
        return json.dumps(payload, default=self._json_default)

    @staticmethod
    def _json_default(value: Any) -> Any:
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        if isinstance(value, (set, tuple)):
            return list(value)
        if isinstance(value, np.generic):
            return value.item()
        return value


class TrendAnalyzer:
    """Advanced trend analysis and forecasting for reports."""

    @staticmethod
    def analyze_time_series(df: pd.DataFrame, date_column: str,
                          value_column: str, periods: int = 30) -> Dict[str, Any]:
        """Analyze time series data and provide trend insights.

        Args:
            df: DataFrame containing time series data
            date_column: Name of the date column
            value_column: Name of the value column
            periods: Number of periods to forecast

        Returns:
            Dictionary containing trend analysis results
        """
        # Handle empty DataFrame
        if df.empty or date_column not in df.columns or value_column not in df.columns:
            return {
                "trend": {
                    "direction": "unknown",
                    "slope": 0.0
                },
                "forecast": [],
                "statistics": {
                    "mean": 0.0,
                    "std": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "missing_values": 0
                },
                "seasonality": {
                    "has_seasonality": False,
                    "period": None,
                    "strength": 0.0
                }
            }
        # Convert date column to datetime
        df_copy = df.copy()
        df_copy[date_column] = pd.to_datetime(df_copy[date_column])
        df_copy = df_copy.sort_values(date_column)

        # Calculate basic statistics
        current_value = df_copy[value_column].iloc[-1]
        previous_value = df_copy[value_column].iloc[-2] if len(df_copy) > 1 else current_value
        change = current_value - previous_value
        change_percent = (change / previous_value * 100) if previous_value != 0 else 0

        # Calculate moving averages
        df_copy['ma_7'] = df_copy[value_column].rolling(window=min(7, len(df_copy))).mean()
        df_copy['ma_30'] = df_copy[value_column].rolling(window=min(30, len(df_copy))).mean()

        # Trend direction
        recent_trend = df_copy[value_column].tail(7).corr(pd.Series(range(7)))
        trend_direction = "increasing" if recent_trend > 0.1 else "decreasing" if recent_trend < -0.1 else "stable"

        # Simple linear forecast
        x = np.arange(len(df_copy))
        y = df_copy[value_column].values
        coeffs = np.polyfit(x, y, 1)

        # Forecast future values
        future_x = np.arange(len(df_copy), len(df_copy) + periods)
        forecast = np.polyval(coeffs, future_x)

        return {
            "trend": {
                "direction": "up" if trend_direction == "increasing" else "down" if trend_direction == "decreasing" else "stable",
                "slope": coeffs[0]
            },
            "forecast": forecast.tolist(),
            "statistics": {
                "mean": float(df_copy[value_column].mean()),
                "std": float(df_copy[value_column].std()),
                "min": float(df_copy[value_column].min()),
                "max": float(df_copy[value_column].max()),
                "missing_values": int(df_copy[value_column].isnull().sum())
            },
            "seasonality": {
                "has_seasonality": abs(recent_trend) > 0.5,
                "period": 7,  # Weekly pattern assumption
                "strength": abs(recent_trend)
            },
            "current_value": current_value,
            "previous_value": previous_value,
            "change": change,
            "change_percent": change_percent,
            "trend_direction": trend_direction,
            "trend_strength": abs(recent_trend),
            "moving_averages": {
                "ma_7": df_copy['ma_7'].iloc[-1],
                "ma_30": df_copy['ma_30'].iloc[-1]
            },
            "forecast_dates": pd.date_range(
                start=df_copy[date_column].iloc[-1] + timedelta(days=1),
                periods=periods,
                freq='D'
            ).strftime('%Y-%m-%d').tolist()
        }

    @staticmethod
    def generate_executive_summary(report_data: ReportData) -> Dict[str, Any]:
        """Generate executive summary with key insights.

        Args:
            report_data: Report data to summarize

        Returns:
            Dictionary containing executive summary
        """
        total_rows = sum(len(df) for df in report_data.raw_data.values())
        total_datasets = len(report_data.raw_data)
        execution_time = report_data.execution_metrics.execution_time if report_data.execution_metrics else 0.0
        memory_usage = report_data.execution_metrics.memory_usage if report_data.execution_metrics else 0.0

        missing_values = 0
        numeric_columns = 0
        for df in report_data.raw_data.values():
            if isinstance(df, pd.DataFrame) and not df.empty:
                missing_values += int(df.isnull().sum().sum())
                numeric_columns += len(df.select_dtypes(include=['number']).columns)

        data_quality_score = 100.0
        if total_rows > 0:
            data_quality_score = max(0.0, 100.0 - (missing_values / total_rows) * 100.0)

        performance_score = max(10.0, 100.0 - execution_time * 10.0)
        completeness_penalty = (missing_values / max(1, numeric_columns * (total_rows or 1))) * 100.0
        completeness_score = max(20.0, 100.0 - completeness_penalty)

        critical_findings = [f for f in report_data.findings if f.severity == SeverityLevel.CRITICAL]
        high_findings = [f for f in report_data.findings if f.severity == SeverityLevel.HIGH]

        summary = {
            "overview": {
                "total_records": total_rows,
                "data_sources": total_datasets,
                "time_period": "Generated analysis period",
                "total_datasets": total_datasets,
                "total_rows": total_rows,
                "generation_time": execution_time,
                "timestamp": datetime.now().isoformat()
            },
            "key_metrics": [],
            "key_findings": [],
            "critical_issues": [],
            "recommendations": [],
            "findings_summary": {
                "total_findings": len(report_data.findings),
                "by_severity": {},
                "by_category": {}
            },
            "data_quality": {
                "score": 0,
                "issues": []
            },
            "performance_metrics": {
                "data_quality_score": round(data_quality_score, 1),
                "performance_score": round(performance_score, 1),
                "completeness_score": round(completeness_score, 1)
            }
        }

        # Critical issues and findings
        summary["critical_issues"] = [
            {
                "title": f.title,
                "description": f.description,
                "category": f.category,
                "affected_objects": f.affected_objects
            }
            for f in critical_findings[:5]
        ]

        summary["key_findings"] = [
            {
                "title": f.title,
                "description": f.description,
                "category": f.category
            }
            for f in (critical_findings + high_findings)[:10]
        ]

        severity_counts: Dict[str, int] = {}
        category_counts: Dict[str, int] = {}
        for finding in report_data.findings:
            severity = finding.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            if finding.category:
                category_counts[finding.category] = category_counts.get(finding.category, 0) + 1

        summary["findings_summary"]["by_severity"] = severity_counts
        summary["findings_summary"]["by_category"] = category_counts

        # Recommendations
        recommendations = []
        if critical_findings:
            recommendations.append("Address critical data quality issues immediately")
        if high_findings:
            recommendations.append("Prioritize remediation for high severity findings")
        if not recommendations and summary["findings_summary"]["total_findings"] > 0:
            recommendations.append("Review medium severity findings for improvement opportunities")
        if execution_time > 10:
            recommendations.append("Investigate report performance bottlenecks")
        if not recommendations:
            recommendations.append("Maintain ongoing monitoring of key metrics")
        summary["recommendations"] = recommendations

        # Data quality score adjustments
        severity_penalty = len(critical_findings) * 20 + len(high_findings) * 10
        adjusted_quality = max(0.0, data_quality_score - severity_penalty)
        summary["data_quality"]["score"] = round(adjusted_quality, 1)
        summary["data_quality"]["issues"] = [finding.title for finding in critical_findings[:5]]
        summary["performance_metrics"]["data_quality_score"] = summary["data_quality"]["score"]

        # Key metrics for metric cards
        summary["key_metrics"] = [
            {
                "name": "Total Records",
                "value": f"{total_rows:,}",
                "subtitle": "Across all datasets",
                "trend": summary["performance_metrics"]["data_quality_score"] - 50,
                "color": "info"
            },
            {
                "name": "Findings Logged",
                "value": summary["findings_summary"]["total_findings"],
                "subtitle": "Issues identified",
                "trend": -(len(critical_findings) * 15 + len(high_findings) * 5),
                "color": "warning"
            },
            {
                "name": "Execution Time",
                "value": f"{execution_time:.2f}s",
                "subtitle": "Processing duration",
                "trend": -execution_time,
                "color": "primary"
            },
            {
                "name": "Memory Usage",
                "value": f"{memory_usage:.1f} MB",
                "subtitle": "Peak consumption",
                "trend": -(memory_usage * 0.05),
                "color": "secondary"
            }
        ]

        # Insights
        summary["key_insights"] = TrendAnalyzer.identify_key_insights(report_data.raw_data)

        return summary

    @staticmethod
    def create_dashboard_widgets(report_data: ReportData) -> List[InteractiveWidget]:
        """Create dashboard widgets from report data.

        Args:
            report_data: Report data to create widgets from

        Returns:
            List of interactive widgets
        """
        widgets = []

        def _slugify(value: str) -> str:
            safe = ''.join(ch.lower() if ch.isalnum() else '_' for ch in value)
            return safe.strip('_') or 'metric'

        def _normalize_trend(value: Any) -> Optional[float]:
            if value is None:
                return None
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                lowered = value.lower()
                if lowered in {'up', 'increase', 'positive', 'growth'}:
                    return 5.0
                if lowered in {'down', 'decrease', 'negative', 'decline'}:
                    return -5.0
                if lowered in {'stable', 'flat', 'steady', 'neutral'}:
                    return 0.0
            return None

        # Executive summary metrics
        exec_summary = TrendAnalyzer.generate_executive_summary(report_data)
        overview = exec_summary.get("overview") or {}
        performance_metrics = (
            exec_summary.get("performance_metrics")
            or exec_summary.get("performance")
            or {}
        )

        total_datasets = overview.get("total_datasets")
        if total_datasets is None:
            total_datasets = overview.get("data_sources")
        if total_datasets is None:
            total_datasets = len(report_data.raw_data)
        else:
            try:
                total_datasets = int(total_datasets)
            except (TypeError, ValueError):
                total_datasets = len(report_data.raw_data)

        total_rows = overview.get("total_rows")
        if total_rows is None:
            total_rows = overview.get("total_records")
        if total_rows is None:
            total_rows = sum(
                len(df) for df in report_data.raw_data.values()
                if isinstance(df, pd.DataFrame)
            )
        else:
            try:
                total_rows = int(total_rows)
            except (TypeError, ValueError):
                total_rows = sum(
                    len(df) for df in report_data.raw_data.values()
                    if isinstance(df, pd.DataFrame)
                )

        generation_time = overview.get("generation_time")
        if generation_time is None and report_data.execution_metrics:
            generation_time = report_data.execution_metrics.execution_time or 0.0
        if generation_time is None:
            generation_time = 0.0
        else:
            try:
                generation_time = float(generation_time)
            except (TypeError, ValueError):
                generation_time = 0.0

        data_quality_score = performance_metrics.get("data_quality_score")
        if data_quality_score is None:
            data_quality_score = (
                exec_summary.get("data_quality", {}).get("score")
            )
        if data_quality_score is None:
            data_quality_score = 0.0
        else:
            try:
                data_quality_score = float(data_quality_score)
            except (TypeError, ValueError):
                data_quality_score = 0.0

        # Metric card widgets from key metrics
        for idx, metric in enumerate(exec_summary.get("key_metrics", [])[:4]):
            metric_name = metric.get("name") or f"Metric {idx + 1}"
            raw_value = metric.get("value", "")
            widget_id = f"metric_{_slugify(metric_name)}"
            widgets.append(InteractiveWidget(
                widget_id=widget_id,
                title=metric_name,
                widget_type="metric_card",
                data={
                    "value": raw_value,
                    "subtitle": metric.get("subtitle"),
                    "trend": _normalize_trend(metric.get("trend")),
                    "color": metric.get("color", "primary"),
                },
                options={"display": "metric"},
            ))

        # Data overview widget
        widgets.append(InteractiveWidget(
            widget_id="data_overview",
            title="Data Overview",
            widget_type="kpi_grid",
            data={
                "kpis": [
                    {
                        "name": "Total Datasets",
                        "value": total_datasets,
                        "unit": "datasets"
                    },
                    {
                        "name": "Total Rows",
                        "value": f"{int(total_rows):,}" if total_rows is not None else "0",
                        "unit": "rows"
                    },
                    {
                        "name": "Generation Time",
                        "value": f"{float(generation_time):.2f}",
                        "unit": "seconds"
                    },
                    {
                        "name": "Data Quality Score",
                        "value": f"{float(data_quality_score):.0f}",
                        "unit": "%"
                    }
                ]
            }
        ))

        # Findings by severity chart
        severity_counts = {}
        for finding in report_data.findings:
            severity = finding.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        if severity_counts:
            widgets.append(InteractiveWidget(
                widget_id="findings_by_severity",
                title="Findings by Severity",
                widget_type="chart",
                data={
                    "labels": list(severity_counts.keys()),
                    "datasets": [{
                        "label": "Number of Findings",
                        "data": list(severity_counts.values()),
                        "backgroundColor": [
                            "#dc3545", "#fd7e14", "#ffc107", "#20c997", "#0dcaf0"
                        ][:len(severity_counts)]
                    }]
                },
                options={"chart_type": "doughnut"}
            ))

        # Time series chart if available
        for dataset_name, df in report_data.raw_data.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                datetime_cols = df.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns
                numeric_cols = df.select_dtypes(include=['number']).columns
                if datetime_cols.any() and numeric_cols.any():
                    date_col = datetime_cols[0]
                    value_col = numeric_cols[0]
                    trend = TrendAnalyzer.analyze_time_series(df, date_col, value_col)
                    chart_data = {
                        "labels": df[date_col].dt.strftime('%Y-%m-%d').tolist(),
                        "datasets": [
                            {
                                "label": value_col.replace('_', ' ').title(),
                                "data": df[value_col].tolist(),
                                "borderColor": "#0d6efd",
                                "backgroundColor": "rgba(13, 110, 253, 0.1)",
                                "fill": True,
                            }
                        ],
                    }
                    widgets.append(InteractiveWidget(
                        widget_id=f"trend_{dataset_name}",
                        title=f"{dataset_name.replace('_', ' ').title()} Trend",
                        widget_type="chart",
                        data=chart_data,
                        options={
                            "chart_type": "line",
                            "tension": 0.3,
                            "trend": trend,
                        },
                    ))
                    break

        # Recent findings table
        recent_findings = sorted(report_data.findings,
                               key=lambda x: x.created_at or datetime.min,
                               reverse=True)[:20]

        if recent_findings:
            findings_data = []
            for finding in recent_findings:
                findings_data.append({
                    "Title": finding.title,
                    "Severity": finding.severity.value,
                    "Category": finding.category,
                    "Created": finding.created_at.strftime('%Y-%m-%d %H:%M') if finding.created_at else "N/A"
                })

            findings_df = pd.DataFrame(findings_data)
            widgets.append(InteractiveWidget(
                widget_id="recent_findings",
                title="Recent Findings",
                widget_type="table",
                data={
                    "columns": findings_df.columns.tolist(),
                    "data": findings_df.to_dict('records'),
                    "total_rows": len(findings_df)
                },
                options={
                    "paginate": True,
                    "searchable": True,
                    "sortable": True,
                    "page_size": 10
                }
            ))

        return widgets

    @staticmethod
    def detect_seasonality(df: pd.DataFrame, date_column: str, value_column: str) -> Dict[str, Any]:
        """Detect seasonality in time series data.

        Args:
            df: DataFrame containing time series data
            date_column: Name of the date column
            value_column: Name of the value column

        Returns:
            Dictionary containing seasonality analysis
        """
        # Simple seasonality detection based on autocorrelation
        df_copy = df.copy()
        df_copy[date_column] = pd.to_datetime(df_copy[date_column])
        df_copy = df_copy.sort_values(date_column)

        # Calculate day-of-week variance
        df_copy['day_of_week'] = df_copy[date_column].dt.dayofweek
        weekly_variance = df_copy.groupby('day_of_week')[value_column].var().mean()

        # Basic seasonality detection
        has_seasonality = weekly_variance > df_copy[value_column].var() * 0.1
        period = 7 if has_seasonality else None
        strength = weekly_variance / df_copy[value_column].var() if df_copy[value_column].var() > 0 else 0

        return {
            "has_seasonality": has_seasonality,
            "period": period,
            "strength": float(strength)
        }

    @staticmethod
    def calculate_forecast_confidence(actual: List[float], predicted: List[float]) -> float:
        """Calculate forecast confidence based on actual vs predicted values.

        Args:
            actual: List of actual values
            predicted: List of predicted values

        Returns:
            Confidence score as percentage
        """
        if len(actual) != len(predicted) or len(actual) == 0:
            return 0.0

        # Calculate Mean Absolute Percentage Error (MAPE)
        errors = []
        for a, p in zip(actual, predicted):
            if a != 0:
                errors.append(abs((a - p) / a))

        if not errors:
            return 0.0

        mape = sum(errors) / len(errors)
        confidence = max(0, 100 * (1 - mape))
        return confidence

    @staticmethod
    def identify_key_insights(datasets: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """Identify key insights from datasets.

        Args:
            datasets: Dictionary of DataFrames to analyze

        Returns:
            List of insights
        """
        insights = []

        for name, df in datasets.items():
            if df.empty:
                continue

            insights.append({
                "type": "dataset_overview",
                "description": f"Dataset '{name}' contains {len(df):,} rows and {len(df.columns)} columns",
                "confidence": 0.5
            })

            # High cardinality columns
            for col in df.select_dtypes(include=['object']).columns:
                cardinality = df[col].nunique()
                if cardinality > len(df) * 0.8:
                    insights.append({
                        "type": "high_cardinality",
                        "description": f"Column '{col}' in {name} has very high cardinality ({cardinality} unique values)",
                        "confidence": 0.9
                    })
                else:
                    value_counts = df[col].value_counts(normalize=True)
                    if not value_counts.empty:
                        top_category, top_share = value_counts.index[0], value_counts.iloc[0]
                        if top_share > 0.4:
                            insights.append({
                                "type": "category_dominance",
                                "description": f"Column '{col}' in {name} is dominated by '{top_category}' ({top_share:.0%})",
                                "confidence": 0.7
                            })

            # Missing data patterns
            null_percentages = df.isnull().mean()
            high_null_cols = null_percentages[null_percentages > 0.5].index.tolist()
            if high_null_cols:
                insights.append({
                    "type": "missing_data",
                    "description": f"Dataset '{name}' has columns with >50% missing values: {', '.join(high_null_cols)}",
                    "confidence": 1.0
                })

            # Numeric correlations
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                correlations = df[numeric_cols].corr()
                high_corr = correlations.where(
                    (correlations.abs() > 0.8) & (correlations != 1.0)
                ).dropna(how='all').dropna(axis=1, how='all')

                if not high_corr.empty:
                    insights.append({
                        "type": "correlation",
                        "description": f"Dataset '{name}' shows strong correlations between numeric columns",
                        "confidence": 0.8
                    })

            # Time trend detection for datetime + numeric
            datetime_cols = df.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns
            if len(datetime_cols) > 0 and len(numeric_cols) > 0:
                date_col = datetime_cols[0]
                value_col = numeric_cols[0]
                ordered = df.sort_values(date_col)
                if len(ordered) >= 2:
                    slope = ordered[value_col].iloc[-1] - ordered[value_col].iloc[0]
                    direction = "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
                    insights.append({
                        "type": "trend",
                        "description": f"Metric '{value_col}' in {name} appears to be {direction}",
                        "confidence": 0.6
                    })

        return insights

    @staticmethod
    def generate_recommendations(findings: List[Finding]) -> List[str]:
        """Generate recommendations based on findings.

        Args:
            findings: List of findings to analyze

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Count findings by severity
        critical_count = len([f for f in findings if f.severity == SeverityLevel.CRITICAL])
        high_count = len([f for f in findings if f.severity == SeverityLevel.HIGH])
        medium_count = len([f for f in findings if f.severity == SeverityLevel.MEDIUM])

        if critical_count > 0:
            recommendations.append(f"Immediately address {critical_count} critical issue{'s' if critical_count > 1 else ''}")

        if high_count > 0:
            recommendations.append(f"Review and resolve {high_count} high-priority finding{'s' if high_count > 1 else ''}")

        if medium_count > 5:
            recommendations.append(f"Consider addressing {medium_count} medium-priority findings to improve data quality")

        # Category-based recommendations
        categories = {}
        for finding in findings:
            if finding.category:
                categories[finding.category] = categories.get(finding.category, 0) + 1

        if categories:
            top_category = max(categories, key=categories.get)
            if categories[top_category] > 3:
                recommendations.append(f"Focus attention on {top_category.lower()} issues, which represent the largest category of findings")

        if not recommendations:
            recommendations.append("Data quality appears good with no major issues identified")

        return recommendations

    @staticmethod
    def analyze_data_quality(datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze overall data quality across datasets.

        Args:
            datasets: Dictionary of DataFrames to analyze

        Returns:
            Data quality analysis results
        """
        quality_report = {
            "overall_score": 100,
            "issues": [],
            "completeness": 100,
            "consistency": 100,
            "validity": 100
        }

        total_cells = 0
        missing_cells = 0

        for name, df in datasets.items():
            if df.empty:
                quality_report["issues"].append(f"Dataset '{name}' is empty")
                quality_report["overall_score"] -= 20
                continue

            # Calculate completeness
            dataset_cells = df.shape[0] * df.shape[1]
            dataset_missing = df.isnull().sum().sum()

            total_cells += dataset_cells
            missing_cells += dataset_missing

            if dataset_missing > 0:
                missing_percentage = (dataset_missing / dataset_cells) * 100 if dataset_cells else 0
                quality_report["issues"].append(
                    f"Dataset '{name}' has {missing_percentage:.1f}% missing values"
                )
                quality_report["completeness"] -= min(15, missing_percentage)

            # Check for duplicate rows
            duplicates = df.duplicated().sum()
            if duplicates > 0:
                quality_report["issues"].append(f"Dataset '{name}' has {duplicates} duplicate rows")
                quality_report["consistency"] -= min(20, duplicates / len(df) * 100)

            # Check for all-null columns
            null_cols = df.columns[df.isnull().all()].tolist()
            if null_cols:
                    quality_report["issues"].append(f"Dataset '{name}' has completely empty columns: {', '.join(null_cols)}")
                    quality_report["validity"] -= 15

        # Calculate overall completeness
        if total_cells > 0:
            completeness_percentage = ((total_cells - missing_cells) / total_cells) * 100
            quality_report["completeness"] = completeness_percentage
        else:
            quality_report["completeness"] = 0

        quality_report["completeness"] = max(0, min(100, quality_report["completeness"]))
        quality_report["consistency"] = max(0, min(100, quality_report["consistency"]))
        quality_report["validity"] = max(0, min(100, quality_report["validity"]))

        # Calculate overall score
        quality_report["overall_score"] = min(100, max(0, (
            quality_report["completeness"] * 0.4 +
            quality_report["consistency"] * 0.3 +
            quality_report["validity"] * 0.3
        )))

        return quality_report

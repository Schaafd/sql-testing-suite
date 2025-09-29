"""Tests for the report scheduler."""

import pytest
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

from sqltest.reporting.scheduler import (
    ReportScheduler, ReportScheduleConfig, ScheduleFrequency,
    NotificationMethod, NotificationConfig, EmailConfig,
    ScheduleStatus, ScheduleExecution
)
from sqltest.reporting.models import ReportConfiguration, ReportFormat, ReportType


class TestReportScheduler:
    """Test ReportScheduler class."""

    def test_initialization(self):
        """Test scheduler initialization."""
        scheduler = ReportScheduler()

        assert scheduler.reporting_engine is not None
        assert isinstance(scheduler.schedules, dict)
        assert isinstance(scheduler.executions, list)
        assert scheduler.running is False

    def test_add_schedule_basic(self, temp_output_dir):
        """Test adding a basic schedule."""
        scheduler = ReportScheduler()

        config = ReportScheduleConfig(
            schedule_id="test_schedule",
            name="Test Schedule",
            description="A test schedule",
            frequency=ScheduleFrequency.DAILY,
            time_of_day="09:00",
            report_config=ReportConfiguration(
                report_type=ReportType.SUMMARY,
                format=ReportFormat.JSON,
                title="Daily Report"
            ),
            output_directory=temp_output_dir
        )

        result = scheduler.add_schedule(config)

        assert result is True
        assert config.schedule_id in scheduler.schedules
        assert scheduler.schedules[config.schedule_id] == config

    def test_add_schedule_weekly(self, temp_output_dir):
        """Test adding a weekly schedule."""
        scheduler = ReportScheduler()

        config = ReportScheduleConfig(
            schedule_id="weekly_schedule",
            name="Weekly Report",
            frequency=ScheduleFrequency.WEEKLY,
            day_of_week=1,  # Tuesday
            time_of_day="10:00",
            report_config=ReportConfiguration(
                report_type=ReportType.DETAILED,
                format=ReportFormat.HTML,
                title="Weekly Report"
            )
        )

        result = scheduler.add_schedule(config)
        assert result is True

    def test_add_schedule_monthly(self, temp_output_dir):
        """Test adding a monthly schedule."""
        scheduler = ReportScheduler()

        config = ReportScheduleConfig(
            schedule_id="monthly_schedule",
            name="Monthly Report",
            frequency=ScheduleFrequency.MONTHLY,
            day_of_month=15,
            time_of_day="08:00",
            report_config=ReportConfiguration(
                report_type=ReportType.EXECUTIVE,
                format=ReportFormat.HTML,
                title="Monthly Report"
            )
        )

        result = scheduler.add_schedule(config)
        assert result is True

    def test_add_schedule_validation_errors(self):
        """Test schedule validation errors."""
        scheduler = ReportScheduler()

        # Missing schedule ID
        with pytest.raises(ValueError):
            config = ReportScheduleConfig(
                schedule_id="",
                name="Invalid Schedule",
                frequency=ScheduleFrequency.DAILY,
                report_config=ReportConfiguration(
                    report_type=ReportType.SUMMARY,
                    format=ReportFormat.JSON,
                    title="Test"
                )
            )
            scheduler._validate_schedule_config(config)

        # Weekly without day_of_week
        with pytest.raises(ValueError):
            config = ReportScheduleConfig(
                schedule_id="test",
                name="Invalid Weekly",
                frequency=ScheduleFrequency.WEEKLY,
                report_config=ReportConfiguration(
                    report_type=ReportType.SUMMARY,
                    format=ReportFormat.JSON,
                    title="Test"
                )
            )
            scheduler._validate_schedule_config(config)

        # Monthly without day_of_month
        with pytest.raises(ValueError):
            config = ReportScheduleConfig(
                schedule_id="test",
                name="Invalid Monthly",
                frequency=ScheduleFrequency.MONTHLY,
                report_config=ReportConfiguration(
                    report_type=ReportType.SUMMARY,
                    format=ReportFormat.JSON,
                    title="Test"
                )
            )
            scheduler._validate_schedule_config(config)

    def test_remove_schedule(self, temp_output_dir):
        """Test removing a schedule."""
        scheduler = ReportScheduler()

        config = ReportScheduleConfig(
            schedule_id="test_remove",
            name="Test Remove",
            frequency=ScheduleFrequency.DAILY,
            time_of_day="09:00",
            report_config=ReportConfiguration(
                report_type=ReportType.SUMMARY,
                format=ReportFormat.JSON,
                title="Test"
            )
        )

        # Add schedule
        scheduler.add_schedule(config)
        assert config.schedule_id in scheduler.schedules

        # Remove schedule
        result = scheduler.remove_schedule(config.schedule_id)
        assert result is True
        assert config.schedule_id not in scheduler.schedules

        # Try to remove non-existent schedule
        result = scheduler.remove_schedule("non_existent")
        assert result is False

    def test_enable_disable_schedule(self, temp_output_dir):
        """Test enabling and disabling schedules."""
        scheduler = ReportScheduler()

        config = ReportScheduleConfig(
            schedule_id="test_enable_disable",
            name="Test Enable/Disable",
            frequency=ScheduleFrequency.DAILY,
            time_of_day="09:00",
            enabled=True,
            report_config=ReportConfiguration(
                report_type=ReportType.SUMMARY,
                format=ReportFormat.JSON,
                title="Test"
            )
        )

        scheduler.add_schedule(config)

        # Disable schedule
        result = scheduler.disable_schedule(config.schedule_id)
        assert result is True
        assert scheduler.schedules[config.schedule_id].enabled is False

        # Enable schedule
        result = scheduler.enable_schedule(config.schedule_id)
        assert result is True
        assert scheduler.schedules[config.schedule_id].enabled is True

        # Try to enable non-existent schedule
        result = scheduler.enable_schedule("non_existent")
        assert result is False

    def test_generate_output_path(self, temp_output_dir):
        """Test output path generation."""
        scheduler = ReportScheduler()

        config = ReportScheduleConfig(
            schedule_id="test_path",
            name="Test Path Generation",
            frequency=ScheduleFrequency.DAILY,
            report_config=ReportConfiguration(
                report_type=ReportType.SUMMARY,
                format=ReportFormat.JSON,
                title="Test"
            ),
            output_directory=temp_output_dir,
            filename_template="{report_name}_{timestamp}"
        )

        output_path = scheduler._generate_output_path(config)

        assert output_path.parent == temp_output_dir
        assert output_path.suffix == ".json"
        assert "test_path_generation" in output_path.name

    def test_prepare_report_data(self):
        """Test report data preparation."""
        scheduler = ReportScheduler()

        config = ReportScheduleConfig(
            schedule_id="test_data",
            name="Test Data",
            frequency=ScheduleFrequency.DAILY,
            report_config=ReportConfiguration(
                report_type=ReportType.SUMMARY,
                format=ReportFormat.JSON,
                title="Test"
            ),
            data_source_query="SELECT * FROM test_table"
        )

        # This should return sample data since we don't have actual DB connection
        data = scheduler._prepare_report_data(config)

        assert isinstance(data, dict)
        assert 'scheduled_data' in data
        assert isinstance(data['scheduled_data'], pd.DataFrame)

    @patch('sqltest.reporting.scheduler.smtplib.SMTP')
    def test_email_notification(self, mock_smtp, temp_output_dir):
        """Test email notification sending."""
        scheduler = ReportScheduler()

        # Create a test report file
        report_path = temp_output_dir / "test_report.json"
        report_path.write_text('{"test": "data"}')

        email_config = EmailConfig(
            smtp_server="smtp.test.com",
            smtp_port=587,
            username="test@test.com",
            password="password"
        )

        notification_config = NotificationConfig(
            method=NotificationMethod.EMAIL,
            recipients=["recipient@test.com"],
            email_config=email_config
        )

        config = ReportScheduleConfig(
            schedule_id="test_email",
            name="Test Email",
            frequency=ScheduleFrequency.DAILY,
            report_config=ReportConfiguration(
                report_type=ReportType.SUMMARY,
                format=ReportFormat.JSON,
                title="Test"
            ),
            notifications=[notification_config]
        )

        execution = ScheduleExecution(
            execution_id="test_execution",
            schedule_id="test_email",
            started_at=datetime.now()
        )

        from sqltest.reporting.models import ReportGenerationResult
        result = ReportGenerationResult(
            success=True,
            output_path=report_path,
            format=ReportFormat.JSON
        )

        # Mock SMTP server
        mock_server = Mock()
        mock_smtp.return_value = mock_server

        # Test email sending
        scheduler._send_email_notification(config, notification_config, execution, result)

        # Verify SMTP calls
        mock_smtp.assert_called_once()
        mock_server.send_message.assert_called_once()
        mock_server.quit.assert_called_once()

    def test_file_notification(self, temp_output_dir):
        """Test file notification."""
        scheduler = ReportScheduler()

        # Create a test report file
        report_path = temp_output_dir / "test_report.json"
        report_path.write_text('{"test": "data"}')

        # Create destination path
        dest_dir = temp_output_dir / "destinations"
        dest_dir.mkdir()

        notification_config = NotificationConfig(
            method=NotificationMethod.FILE,
            recipients=[str(dest_dir / "copied_report.json")]
        )

        config = ReportScheduleConfig(
            schedule_id="test_file",
            name="Test File",
            frequency=ScheduleFrequency.DAILY,
            report_config=ReportConfiguration(
                report_type=ReportType.SUMMARY,
                format=ReportFormat.JSON,
                title="Test"
            ),
            notifications=[notification_config]
        )

        execution = ScheduleExecution(
            execution_id="test_execution",
            schedule_id="test_file",
            started_at=datetime.now()
        )

        from sqltest.reporting.models import ReportGenerationResult
        result = ReportGenerationResult(
            success=True,
            output_path=report_path,
            format=ReportFormat.JSON
        )

        # Test file notification
        scheduler._send_file_notification(config, notification_config, execution, result)

        # Verify file was copied
        copied_file = dest_dir / "copied_report.json"
        assert copied_file.exists()
        assert copied_file.read_text() == '{"test": "data"}'

    def test_get_schedule_status(self, temp_output_dir):
        """Test getting schedule status."""
        scheduler = ReportScheduler()

        config = ReportScheduleConfig(
            schedule_id="test_status",
            name="Test Status",
            frequency=ScheduleFrequency.DAILY,
            time_of_day="09:00",
            enabled=True,
            report_config=ReportConfiguration(
                report_type=ReportType.SUMMARY,
                format=ReportFormat.JSON,
                title="Test"
            )
        )

        scheduler.add_schedule(config)

        # Add some execution history
        execution = ScheduleExecution(
            execution_id="test_exec",
            schedule_id="test_status",
            started_at=datetime.now(),
            status=ScheduleStatus.COMPLETED
        )
        scheduler.executions.append(execution)

        status = scheduler.get_schedule_status("test_status")

        assert status is not None
        assert status['schedule_id'] == "test_status"
        assert status['name'] == "Test Status"
        assert status['enabled'] is True
        assert status['frequency'] == "daily"
        assert status['recent_executions'] == 1
        assert status['success_rate'] == 100.0

        # Test non-existent schedule
        status = scheduler.get_schedule_status("non_existent")
        assert status is None

    def test_list_schedules(self, temp_output_dir):
        """Test listing all schedules."""
        scheduler = ReportScheduler()

        # Add multiple schedules
        for i in range(3):
            config = ReportScheduleConfig(
                schedule_id=f"test_schedule_{i}",
                name=f"Test Schedule {i}",
                frequency=ScheduleFrequency.DAILY,
                time_of_day="09:00",
                report_config=ReportConfiguration(
                    report_type=ReportType.SUMMARY,
                    format=ReportFormat.JSON,
                    title=f"Test {i}"
                )
            )
            scheduler.add_schedule(config)

        schedules = scheduler.list_schedules()

        assert len(schedules) == 3
        assert all(isinstance(s, dict) for s in schedules)
        assert all('schedule_id' in s for s in schedules)

    def test_execution_history(self):
        """Test getting execution history."""
        scheduler = ReportScheduler()

        # Add some execution history
        for i in range(5):
            execution = ScheduleExecution(
                execution_id=f"test_exec_{i}",
                schedule_id="test_schedule",
                started_at=datetime.now() - timedelta(hours=i),
                status=ScheduleStatus.COMPLETED if i % 2 == 0 else ScheduleStatus.FAILED
            )
            scheduler.executions.append(execution)

        # Get all history
        history = scheduler.get_execution_history()
        assert len(history) == 5

        # Get history for specific schedule
        history = scheduler.get_execution_history("test_schedule")
        assert len(history) == 5

        # Get limited history
        history = scheduler.get_execution_history(limit=3)
        assert len(history) == 3

        # Get history for non-existent schedule
        history = scheduler.get_execution_history("non_existent")
        assert len(history) == 0

    def test_export_import_config(self, temp_output_dir):
        """Test exporting and importing schedule configuration."""
        scheduler = ReportScheduler()

        config = ReportScheduleConfig(
            schedule_id="test_export",
            name="Test Export",
            frequency=ScheduleFrequency.DAILY,
            time_of_day="09:00",
            report_config=ReportConfiguration(
                report_type=ReportType.SUMMARY,
                format=ReportFormat.JSON,
                title="Test"
            ),
            output_directory=temp_output_dir
        )

        scheduler.add_schedule(config)

        # Export configuration
        exported_json = scheduler.export_schedule_config("test_export")
        assert exported_json is not None
        assert "test_export" in exported_json

        # Import configuration with new ID
        import json
        config_dict = json.loads(exported_json)
        config_dict['schedule_id'] = "test_import"
        config_dict['name'] = "Test Import"

        result = scheduler.import_schedule_config(json.dumps(config_dict))
        assert result is True
        assert "test_import" in scheduler.schedules

        # Test exporting non-existent schedule
        exported = scheduler.export_schedule_config("non_existent")
        assert exported is None

        # Test importing invalid JSON
        result = scheduler.import_schedule_config("invalid json")
        assert result is False

    def test_calculate_success_rate(self):
        """Test success rate calculation."""
        scheduler = ReportScheduler()

        # Test empty executions
        rate = scheduler._calculate_success_rate([])
        assert rate == 0.0

        # Test with mixed results
        executions = [
            ScheduleExecution("1", "test", datetime.now(), status=ScheduleStatus.COMPLETED),
            ScheduleExecution("2", "test", datetime.now(), status=ScheduleStatus.COMPLETED),
            ScheduleExecution("3", "test", datetime.now(), status=ScheduleStatus.FAILED),
            ScheduleExecution("4", "test", datetime.now(), status=ScheduleStatus.COMPLETED)
        ]

        rate = scheduler._calculate_success_rate(executions)
        assert rate == 75.0

    @patch('time.sleep')
    def test_scheduler_thread_operations(self, mock_sleep):
        """Test scheduler thread start/stop operations."""
        scheduler = ReportScheduler()

        # Test starting scheduler
        scheduler.start_scheduler()
        assert scheduler.running is True
        assert scheduler.scheduler_thread is not None

        # Test stopping scheduler
        scheduler.stop_scheduler()
        assert scheduler.running is False

        # Test starting already running scheduler
        scheduler.running = True
        scheduler.start_scheduler()  # Should log warning but not create new thread
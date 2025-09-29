"""Advanced report scheduling and automation system."""

import asyncio
import logging
import smtplib
import schedule
import time
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import threading
import json

import pandas as pd
from pydantic import BaseModel, Field, validator

from .engine import ReportingEngine
from .models import (
    ReportConfiguration, ReportFormat, ReportData, ReportGenerationResult,
    ReportOptions, SeverityLevel
)

logger = logging.getLogger(__name__)


class ScheduleFrequency(str, Enum):
    """Report scheduling frequency options."""
    ONCE = "once"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    CUSTOM = "custom"


class NotificationMethod(str, Enum):
    """Notification delivery methods."""
    EMAIL = "email"
    WEBHOOK = "webhook"
    SLACK = "slack"
    TEAMS = "teams"
    FILE = "file"


class ScheduleStatus(str, Enum):
    """Schedule execution status."""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    DISABLED = "disabled"


@dataclass
class EmailConfig:
    """Email configuration for report distribution."""
    smtp_server: str
    smtp_port: int = 587
    username: str = ""
    password: str = ""
    use_tls: bool = True
    use_ssl: bool = False
    timeout: int = 30


@dataclass
class NotificationConfig:
    """Configuration for report notifications."""
    method: NotificationMethod
    recipients: List[str]
    subject_template: str = "Scheduled Report: {report_title}"
    body_template: str = "Please find the attached report generated on {timestamp}."
    webhook_url: Optional[str] = None
    email_config: Optional[EmailConfig] = None
    custom_headers: Dict[str, str] = field(default_factory=dict)


class ReportScheduleConfig(BaseModel):
    """Configuration for scheduled report generation."""
    schedule_id: str
    name: str
    description: Optional[str] = None
    frequency: ScheduleFrequency
    report_config: ReportConfiguration
    report_options: Optional[ReportOptions] = None
    data_source_query: Optional[str] = None
    data_source_params: Dict[str, Any] = Field(default_factory=dict)
    notifications: List[NotificationConfig] = Field(default_factory=list)

    # Scheduling options
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    time_of_day: str = "09:00"  # HH:MM format
    day_of_week: Optional[int] = None  # 0=Monday, 6=Sunday
    day_of_month: Optional[int] = None  # 1-31
    custom_cron: Optional[str] = None
    timezone: str = "UTC"

    # Execution options
    max_retries: int = 3
    retry_delay: int = 300  # seconds
    timeout: int = 3600  # seconds
    enabled: bool = True

    # Output options
    output_directory: Optional[Path] = None
    filename_template: str = "{report_name}_{timestamp}"
    archive_after_days: Optional[int] = None

    class Config:
        extra = "allow"
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            Path: lambda v: str(v)
        }

    @validator('output_directory', pre=True)
    def convert_output_directory(cls, v):
        if v is not None and not isinstance(v, Path):
            return Path(v)
        return v


@dataclass
class ScheduleExecution:
    """Record of a scheduled report execution."""
    execution_id: str
    schedule_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    status: ScheduleStatus = ScheduleStatus.ACTIVE
    result: Optional[ReportGenerationResult] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    notifications_sent: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ReportScheduler:
    """Advanced report scheduling and automation system."""

    def __init__(self, reporting_engine: Optional[ReportingEngine] = None):
        """Initialize the report scheduler.

        Args:
            reporting_engine: ReportingEngine instance for generating reports
        """
        self.reporting_engine = reporting_engine or ReportingEngine()
        self.schedules: Dict[str, ReportScheduleConfig] = {}
        self.executions: List[ScheduleExecution] = []
        self.running = False
        self.scheduler_thread: Optional[threading.Thread] = None
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Set up logging for the scheduler."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    def add_schedule(self, schedule_config: ReportScheduleConfig) -> bool:
        """Add a new report schedule.

        Args:
            schedule_config: Configuration for the scheduled report

        Returns:
            True if schedule was added successfully
        """
        try:
            # Validate schedule configuration
            self._validate_schedule_config(schedule_config)

            # Store the schedule
            self.schedules[schedule_config.schedule_id] = schedule_config

            # Set up the actual schedule
            if schedule_config.enabled:
                self._setup_schedule(schedule_config)

            logger.info(f"Added schedule '{schedule_config.schedule_id}': {schedule_config.name}")
            return True

        except Exception as e:
            logger.error(f"Failed to add schedule '{schedule_config.schedule_id}': {e}")
            return False

    def remove_schedule(self, schedule_id: str) -> bool:
        """Remove a report schedule.

        Args:
            schedule_id: ID of the schedule to remove

        Returns:
            True if schedule was removed successfully
        """
        try:
            if schedule_id in self.schedules:
                # Cancel the scheduled job
                schedule.clear(schedule_id)

                # Remove from our tracking
                del self.schedules[schedule_id]

                logger.info(f"Removed schedule '{schedule_id}'")
                return True
            else:
                logger.warning(f"Schedule '{schedule_id}' not found")
                return False

        except Exception as e:
            logger.error(f"Failed to remove schedule '{schedule_id}': {e}")
            return False

    def enable_schedule(self, schedule_id: str) -> bool:
        """Enable a disabled schedule.

        Args:
            schedule_id: ID of the schedule to enable

        Returns:
            True if schedule was enabled successfully
        """
        if schedule_id not in self.schedules:
            logger.error(f"Schedule '{schedule_id}' not found")
            return False

        try:
            self.schedules[schedule_id].enabled = True
            self._setup_schedule(self.schedules[schedule_id])
            logger.info(f"Enabled schedule '{schedule_id}'")
            return True

        except Exception as e:
            logger.error(f"Failed to enable schedule '{schedule_id}': {e}")
            return False

    def disable_schedule(self, schedule_id: str) -> bool:
        """Disable an active schedule.

        Args:
            schedule_id: ID of the schedule to disable

        Returns:
            True if schedule was disabled successfully
        """
        if schedule_id not in self.schedules:
            logger.error(f"Schedule '{schedule_id}' not found")
            return False

        try:
            self.schedules[schedule_id].enabled = False
            schedule.clear(schedule_id)
            logger.info(f"Disabled schedule '{schedule_id}'")
            return True

        except Exception as e:
            logger.error(f"Failed to disable schedule '{schedule_id}': {e}")
            return False

    def _validate_schedule_config(self, config: ReportScheduleConfig) -> None:
        """Validate schedule configuration.

        Args:
            config: Schedule configuration to validate

        Raises:
            ValueError: If configuration is invalid
        """
        if not config.schedule_id:
            raise ValueError("Schedule ID is required")

        if config.frequency == ScheduleFrequency.CUSTOM and not config.custom_cron:
            raise ValueError("Custom cron expression required for custom frequency")

        if config.frequency == ScheduleFrequency.WEEKLY and config.day_of_week is None:
            raise ValueError("Day of week required for weekly frequency")

        if config.frequency == ScheduleFrequency.MONTHLY and config.day_of_month is None:
            raise ValueError("Day of month required for monthly frequency")

        # Validate time format
        try:
            datetime.strptime(config.time_of_day, "%H:%M")
        except ValueError:
            raise ValueError("Invalid time format. Use HH:MM")

    def _setup_schedule(self, config: ReportScheduleConfig) -> None:
        """Set up the actual schedule using the schedule library.

        Args:
            config: Schedule configuration
        """
        # Create the job function with bound config
        job_func = lambda: self._execute_scheduled_report(config.schedule_id)

        # Set up schedule based on frequency
        if config.frequency == ScheduleFrequency.DAILY:
            schedule.every().day.at(config.time_of_day).do(job_func).tag(config.schedule_id)

        elif config.frequency == ScheduleFrequency.WEEKLY:
            day_map = {0: 'monday', 1: 'tuesday', 2: 'wednesday', 3: 'thursday',
                      4: 'friday', 5: 'saturday', 6: 'sunday'}
            day_name = day_map.get(config.day_of_week, 'monday')
            getattr(schedule.every(), day_name).at(config.time_of_day).do(job_func).tag(config.schedule_id)

        elif config.frequency == ScheduleFrequency.MONTHLY:
            # For monthly, we'll check daily and execute on the right day
            schedule.every().day.at(config.time_of_day).do(
                lambda: self._execute_monthly_check(config.schedule_id)
            ).tag(config.schedule_id)

        elif config.frequency == ScheduleFrequency.ONCE:
            # For one-time execution, schedule for immediate execution
            schedule.every().day.at(config.time_of_day).do(job_func).tag(config.schedule_id)

    def _execute_monthly_check(self, schedule_id: str) -> None:
        """Check if monthly report should run today.

        Args:
            schedule_id: ID of the schedule to check
        """
        config = self.schedules.get(schedule_id)
        if not config:
            return

        if datetime.now().day == config.day_of_month:
            self._execute_scheduled_report(schedule_id)

    def _execute_scheduled_report(self, schedule_id: str) -> None:
        """Execute a scheduled report.

        Args:
            schedule_id: ID of the schedule to execute
        """
        config = self.schedules.get(schedule_id)
        if not config or not config.enabled:
            return

        execution_id = f"{schedule_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        execution = ScheduleExecution(
            execution_id=execution_id,
            schedule_id=schedule_id,
            started_at=datetime.now()
        )

        logger.info(f"Starting scheduled report execution: {execution_id}")

        try:
            # Prepare data if query is provided
            data = self._prepare_report_data(config)

            # Generate output path
            output_path = self._generate_output_path(config)
            config.report_config.output_path = output_path

            # Generate the report
            result = self.reporting_engine.generate_report(
                data=data,
                config=config.report_config,
                options=config.report_options,
                analyze=True
            )

            execution.result = result
            execution.completed_at = datetime.now()

            if result.success:
                execution.status = ScheduleStatus.COMPLETED
                logger.info(f"Report generated successfully: {result.output_path}")

                # Send notifications
                self._send_notifications(config, execution, result)

                # Handle one-time schedules
                if config.frequency == ScheduleFrequency.ONCE:
                    self.disable_schedule(schedule_id)

            else:
                execution.status = ScheduleStatus.FAILED
                execution.error_message = result.error_message
                logger.error(f"Report generation failed: {result.error_message}")

        except Exception as e:
            execution.status = ScheduleStatus.FAILED
            execution.error_message = str(e)
            execution.completed_at = datetime.now()
            logger.error(f"Scheduled report execution failed: {e}")

        finally:
            self.executions.append(execution)

    def _prepare_report_data(self, config: ReportScheduleConfig) -> Dict[str, pd.DataFrame]:
        """Prepare data for report generation.

        Args:
            config: Schedule configuration

        Returns:
            Dictionary of DataFrames for report generation
        """
        # For now, return sample data
        # In a real implementation, this would execute the data_source_query
        # against the configured database connection

        if config.data_source_query:
            # TODO: Execute actual query against database
            # This would require database connection configuration
            logger.info(f"Would execute query: {config.data_source_query}")

        # Return sample data for demonstration
        return {
            'scheduled_data': pd.DataFrame({
                'timestamp': pd.date_range('2023-01-01', periods=100, freq='D'),
                'value': range(100),
                'category': ['A', 'B', 'C'] * 33 + ['A']
            })
        }

    def _generate_output_path(self, config: ReportScheduleConfig) -> Path:
        """Generate output path for the report.

        Args:
            config: Schedule configuration

        Returns:
            Path for the generated report
        """
        # Use configured directory or default
        output_dir = config.output_directory or Path("reports/scheduled")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename from template
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = config.filename_template.format(
            report_name=config.name.replace(' ', '_').lower(),
            timestamp=timestamp,
            schedule_id=config.schedule_id
        )

        # Add extension based on format
        ext = config.report_config.format.value
        if not filename.endswith(f'.{ext}'):
            filename += f'.{ext}'

        return output_dir / filename

    def _send_notifications(self, config: ReportScheduleConfig,
                          execution: ScheduleExecution,
                          result: ReportGenerationResult) -> None:
        """Send notifications for completed report.

        Args:
            config: Schedule configuration
            execution: Execution record
            result: Report generation result
        """
        for notification in config.notifications:
            try:
                if notification.method == NotificationMethod.EMAIL:
                    self._send_email_notification(config, notification, execution, result)
                elif notification.method == NotificationMethod.FILE:
                    self._send_file_notification(config, notification, execution, result)
                # TODO: Implement webhook, Slack, Teams notifications

                execution.notifications_sent.append(notification.method.value)
                logger.info(f"Sent {notification.method.value} notification for {execution.execution_id}")

            except Exception as e:
                logger.error(f"Failed to send {notification.method.value} notification: {e}")

    def _send_email_notification(self, config: ReportScheduleConfig,
                               notification: NotificationConfig,
                               execution: ScheduleExecution,
                               result: ReportGenerationResult) -> None:
        """Send email notification with report attachment.

        Args:
            config: Schedule configuration
            notification: Notification configuration
            execution: Execution record
            result: Report generation result
        """
        if not notification.email_config:
            raise ValueError("Email configuration required for email notifications")

        email_config = notification.email_config

        # Create message
        msg = MIMEMultipart()
        msg['From'] = email_config.username
        msg['To'] = ', '.join(notification.recipients)
        msg['Subject'] = notification.subject_template.format(
            report_title=config.name,
            timestamp=execution.started_at.strftime('%Y-%m-%d %H:%M:%S')
        )

        # Add body
        body = notification.body_template.format(
            report_title=config.name,
            timestamp=execution.started_at.strftime('%Y-%m-%d %H:%M:%S'),
            execution_id=execution.execution_id
        )
        msg.attach(MIMEText(body, 'plain'))

        # Attach report file
        if result.output_path and result.output_path.exists():
            with open(result.output_path, 'rb') as attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())

            encoders.encode_base64(part)
            part.add_header(
                'Content-Disposition',
                f'attachment; filename= {result.output_path.name}'
            )
            msg.attach(part)

        # Send email
        if email_config.use_ssl:
            server = smtplib.SMTP_SSL(email_config.smtp_server, email_config.smtp_port)
        else:
            server = smtplib.SMTP(email_config.smtp_server, email_config.smtp_port)
            if email_config.use_tls:
                server.starttls()

        if email_config.username and email_config.password:
            server.login(email_config.username, email_config.password)

        server.send_message(msg)
        server.quit()

    def _send_file_notification(self, config: ReportScheduleConfig,
                              notification: NotificationConfig,
                              execution: ScheduleExecution,
                              result: ReportGenerationResult) -> None:
        """Send file-based notification by copying report to specified locations.

        Args:
            config: Schedule configuration
            notification: Notification configuration
            execution: Execution record
            result: Report generation result
        """
        if not result.output_path or not result.output_path.exists():
            raise ValueError("Report file not found for file notification")

        # Copy report to each specified recipient path
        for recipient_path in notification.recipients:
            dest_path = Path(recipient_path)
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            import shutil
            shutil.copy2(result.output_path, dest_path)
            logger.info(f"Copied report to: {dest_path}")

    def start_scheduler(self) -> None:
        """Start the background scheduler thread."""
        if self.running:
            logger.warning("Scheduler is already running")
            return

        self.running = True
        self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()
        logger.info("Report scheduler started")

    def stop_scheduler(self) -> None:
        """Stop the background scheduler thread."""
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        schedule.clear()
        logger.info("Report scheduler stopped")

    def _run_scheduler(self) -> None:
        """Run the scheduler in a background thread."""
        while self.running:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                time.sleep(60)

    def get_schedule_status(self, schedule_id: str) -> Optional[Dict[str, Any]]:
        """Get status information for a schedule.

        Args:
            schedule_id: ID of the schedule

        Returns:
            Dictionary containing schedule status information
        """
        if schedule_id not in self.schedules:
            return None

        config = self.schedules[schedule_id]
        recent_executions = [
            e for e in self.executions[-10:]  # Last 10 executions
            if e.schedule_id == schedule_id
        ]

        return {
            'schedule_id': schedule_id,
            'name': config.name,
            'enabled': config.enabled,
            'frequency': config.frequency.value,
            'next_run': self._get_next_run_time(schedule_id),
            'last_execution': recent_executions[-1] if recent_executions else None,
            'recent_executions': len(recent_executions),
            'success_rate': self._calculate_success_rate(recent_executions)
        }

    def _get_next_run_time(self, schedule_id: str) -> Optional[datetime]:
        """Get the next scheduled run time for a schedule.

        Args:
            schedule_id: ID of the schedule

        Returns:
            Next run time or None if not found
        """
        # This would require extending the schedule library to expose next run times
        # For now, return None
        return None

    def _calculate_success_rate(self, executions: List[ScheduleExecution]) -> float:
        """Calculate success rate for executions.

        Args:
            executions: List of executions to analyze

        Returns:
            Success rate as a percentage
        """
        if not executions:
            return 0.0

        successful = len([e for e in executions if e.status == ScheduleStatus.COMPLETED])
        return (successful / len(executions)) * 100

    def list_schedules(self) -> List[Dict[str, Any]]:
        """List all configured schedules.

        Returns:
            List of schedule information dictionaries
        """
        return [
            self.get_schedule_status(schedule_id)
            for schedule_id in self.schedules.keys()
        ]

    def get_execution_history(self, schedule_id: Optional[str] = None,
                            limit: int = 50) -> List[ScheduleExecution]:
        """Get execution history for schedules.

        Args:
            schedule_id: Optional schedule ID to filter by
            limit: Maximum number of executions to return

        Returns:
            List of execution records
        """
        executions = self.executions

        if schedule_id:
            executions = [e for e in executions if e.schedule_id == schedule_id]

        # Sort by start time, most recent first
        executions.sort(key=lambda x: x.started_at, reverse=True)

        return executions[:limit]

    def export_schedule_config(self, schedule_id: str) -> Optional[str]:
        """Export schedule configuration as JSON.

        Args:
            schedule_id: ID of the schedule to export

        Returns:
            JSON string of the schedule configuration
        """
        if schedule_id not in self.schedules:
            return None

        config = self.schedules[schedule_id]
        return config.model_dump_json(indent=2)

    def import_schedule_config(self, config_json: str) -> bool:
        """Import schedule configuration from JSON.

        Args:
            config_json: JSON string containing schedule configuration

        Returns:
            True if import was successful
        """
        try:
            config_dict = json.loads(config_json)
            config = ReportScheduleConfig(**config_dict)
            return self.add_schedule(config)
        except Exception as e:
            logger.error(f"Failed to import schedule configuration: {e}")
            return False
import json
import logging
import smtplib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Callable, Dict, List, Optional


@dataclass
class Alert:
    """Base class for alerts"""

    severity: str  # 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
    message: str
    timestamp: datetime
    source: str
    alert_id: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary"""
        return {
            "alert_id": self.alert_id,
            "severity": self.severity,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "metadata": self.metadata,
        }


class AlertHandler(ABC):
    """Abstract base class for alert handlers"""

    @abstractmethod
    def handle(self, alert: Alert) -> bool:
        """Handle an alert"""
        pass


class EmailAlertHandler(AlertHandler):
    """Email alert handler"""

    def __init__(self, config: Dict[str, Any]):
        self.smtp_server = config["smtp_server"]
        self.smtp_port = config["smtp_port"]
        self.sender_email = config["sender_email"]
        self.sender_password = config["sender_password"]
        self.recipient_emails = config["recipient_emails"]
        self.logger = logging.getLogger(__name__)

    def handle(self, alert: Alert) -> bool:
        """Send alert via email"""
        try:
            msg = MIMEMultipart()
            msg["From"] = self.sender_email
            msg["To"] = ", ".join(self.recipient_emails)
            msg["Subject"] = f"[{alert.severity}] CLV System Alert: {alert.source}"

            body = self._format_email_body(alert)
            msg.attach(MIMEText(body, "plain"))

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)

            return True

        except Exception as e:
            self.logger.error(f"Failed to send email alert: {str(e)}")
            return False

    def _format_email_body(self, alert: Alert) -> str:
        """Format alert email body"""
        return f"""
Alert Details:
-------------
Severity: {alert.severity}
Source: {alert.source}
Time: {alert.timestamp}
Message: {alert.message}

Additional Information:
----------------------
{json.dumps(alert.metadata, indent=2)}
"""


class SlackAlertHandler(AlertHandler):
    """Slack alert handler"""

    def __init__(self, config: Dict[str, Any]):
        self.webhook_url = config["webhook_url"]
        self.channel = config["channel"]
        self.username = config.get("username", "CLV Alert System")
        self.logger = logging.getLogger(__name__)

    def handle(self, alert: Alert) -> bool:
        """Send alert to Slack"""
        try:
            import requests

            severity_emoji = {
                "INFO": ":information_source:",
                "WARNING": ":warning:",
                "ERROR": ":x:",
                "CRITICAL": ":rotating_light:",
            }

            payload = {
                "channel": self.channel,
                "username": self.username,
                "text": f"{severity_emoji.get(alert.severity, ':bell:')} *{alert.severity} Alert*",
                "attachments": [
                    {
                        "color": self._get_severity_color(alert.severity),
                        "fields": [
                            {"title": "Source", "value": alert.source, "short": True},
                            {
                                "title": "Time",
                                "value": alert.timestamp.isoformat(),
                                "short": True,
                            },
                            {
                                "title": "Message",
                                "value": alert.message,
                                "short": False,
                            },
                            {
                                "title": "Details",
                                "value": f"```{json.dumps(alert.metadata, indent=2)}```",
                                "short": False,
                            },
                        ],
                    }
                ],
            }

            response = requests.post(self.webhook_url, json=payload)
            return response.status_code == 200

        except Exception as e:
            self.logger.error(f"Failed to send Slack alert: {str(e)}")
            return False

    def _get_severity_color(self, severity: str) -> str:
        """Get color for severity level"""
        return {
            "INFO": "#36a64f",
            "WARNING": "#ffcc00",
            "ERROR": "#ff9900",
            "CRITICAL": "#ff0000",
        }.get(severity, "#cccccc")


class AlertManager:
    """Manages system alerts and notifications"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.handlers: List[AlertHandler] = []
        self.alert_history: List[Alert] = []
        self.alert_rules: Dict[str, Dict[str, Any]] = config.get("alert_rules", {})
        self.logger = logging.getLogger(__name__)

        self._setup_handlers()

    def _setup_handlers(self) -> None:
        """Setup alert handlers based on configuration"""
        if self.config.get("email_alerts", {}).get("enabled", False):
            self.handlers.append(EmailAlertHandler(self.config["email_alerts"]))

        if self.config.get("slack_alerts", {}).get("enabled", False):
            self.handlers.append(SlackAlertHandler(self.config["slack_alerts"]))

    def add_handler(self, handler: AlertHandler) -> None:
        """Add new alert handler"""
        self.handlers.append(handler)

    def create_alert(
        self,
        severity: str,
        message: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Alert:
        """Create and process a new alert"""
        alert = Alert(
            severity=severity.upper(),
            message=message,
            timestamp=datetime.now(),
            source=source,
            alert_id=self._generate_alert_id(),
            metadata=metadata or {},
        )

        self._process_alert(alert)
        return alert

    def _process_alert(self, alert: Alert) -> None:
        """Process alert through handlers"""
        self.alert_history.append(alert)

        # Check if alert meets threshold for notification
        if self._should_notify(alert):
            for handler in self.handlers:
                try:
                    handler.handle(alert)
                except Exception as e:
                    self.logger.error(f"Handler failed to process alert: {str(e)}")

    def _should_notify(self, alert: Alert) -> bool:
        """Check if alert should trigger notification"""
        if alert.severity == "CRITICAL":
            return True

        rule = self.alert_rules.get(alert.source, {})
        min_severity = rule.get("min_severity", "ERROR")

        severity_levels = {"INFO": 0, "WARNING": 1, "ERROR": 2, "CRITICAL": 3}

        return severity_levels.get(alert.severity, 0) >= severity_levels.get(
            min_severity, 0
        )

    def _generate_alert_id(self) -> str:
        """Generate unique alert ID"""
        import uuid

        return f"alert_{uuid.uuid4().hex[:8]}"

    def get_alerts(
        self,
        severity: Optional[str] = None,
        source: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[Alert]:
        """Get filtered alerts"""
        filtered_alerts = self.alert_history

        if severity:
            filtered_alerts = [
                alert for alert in filtered_alerts if alert.severity == severity.upper()
            ]

        if source:
            filtered_alerts = [
                alert for alert in filtered_alerts if alert.source == source
            ]

        if start_time:
            filtered_alerts = [
                alert for alert in filtered_alerts if alert.timestamp >= start_time
            ]

        if end_time:
            filtered_alerts = [
                alert for alert in filtered_alerts if alert.timestamp <= end_time
            ]

        return filtered_alerts

    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of alerts"""
        return {
            "total_alerts": len(self.alert_history),
            "by_severity": {
                severity: len(
                    [
                        alert
                        for alert in self.alert_history
                        if alert.severity == severity
                    ]
                )
                for severity in ["INFO", "WARNING", "ERROR", "CRITICAL"]
            },
            "by_source": {
                source: len(
                    [alert for alert in self.alert_history if alert.source == source]
                )
                for source in set(alert.source for alert in self.alert_history)
            },
        }

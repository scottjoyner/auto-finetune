"""Alerting system for pipeline failures and anomalies.

Supports desktop notifications, webhooks (Slack/Discord), and log files.
Integrates with the real-time pipeline to notify on failures.
"""
from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any

from src.config import Config, load
from src.locking import atomic_write_json


@dataclass
class Alert:
    """An alert event."""
    alert_id: str
    level: str  # info, warning, error, critical
    source: str  # component that triggered the alert
    message: str
    timestamp: float
    metadata: dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False


class AlertManager:
    """Manages alerts and notifications."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.alerts_path = os.path.join(cfg.path("analysis_dir"), "alerts.jsonl")
        self.alerts: list[Alert] = []
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.alerts_path):
            try:
                with open(self.alerts_path) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            data = json.loads(line)
                            self.alerts.append(Alert(**data))
            except Exception:
                pass

    def _save_alert(self, alert: Alert) -> None:
        os.makedirs(os.path.dirname(self.alerts_path), exist_ok=True)
        with open(self.alerts_path, "a") as f:
            f.write(json.dumps(asdict(alert)) + "\n")

    def fire(self, level: str, source: str, message: str,
             metadata: dict | None = None) -> str:
        """Fire a new alert."""
        alert_id = f"alert-{int(time.time() * 1000)}"
        alert = Alert(
            alert_id=alert_id,
            level=level,
            source=source,
            message=message,
            timestamp=time.time(),
            metadata=metadata or {},
        )
        self.alerts.append(alert)
        self._save_alert(alert)
        self._notify(alert)
        return alert_id

    def _notify(self, alert: Alert) -> None:
        """Send notifications based on config."""
        # Webhook
        webhook_url = self.cfg.get("notify", "webhook_url", default="")
        if webhook_url:
            try:
                payload = json.dumps({
                    "text": f"[{alert.level.upper()}] {alert.source}: {alert.message}",
                    "alert_id": alert.alert_id,
                    "timestamp": alert.timestamp,
                    "metadata": alert.metadata,
                }).encode()
                req = urllib.request.Request(
                    webhook_url, data=payload,
                    headers={"Content-Type": "application/json"},
                )
                urllib.request.urlopen(req, timeout=5)
            except Exception:
                pass

        # Desktop notification (Linux)
        try:
            import subprocess
            subprocess.run(
                ["notify-send", f"Pipeline {alert.level}", alert.message],
                timeout=5, capture_output=True,
            )
        except Exception:
            pass

        # Log file
        log_path = os.path.join(self.cfg.path("analysis_dir"), "alerts.log")
        try:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, "a") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{alert.level.upper()}] "
                        f"{alert.source}: {alert.message}\n")
        except Exception:
            pass

    def recent(self, level: str | None = None, limit: int = 20) -> list[Alert]:
        """Get recent alerts, optionally filtered by level."""
        alerts = self.alerts
        if level:
            alerts = [a for a in alerts if a.level == level]
        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)[:limit]

    def unacknowledged(self) -> list[Alert]:
        return [a for a in self.alerts if not a.acknowledged]

    def acknowledge(self, alert_id: str) -> bool:
        for a in self.alerts:
            if a.alert_id == alert_id:
                a.acknowledged = True
                self._save_alert(a)
                return True
        return False

    def summary(self) -> dict:
        return {
            "total": len(self.alerts),
            "unacknowledged": len(self.unacknowledged()),
            "by_level": {
                level: len([a for a in self.alerts if a.level == level])
                for level in ["info", "warning", "error", "critical"]
            },
            "recent": [asdict(a) for a in self.recent(limit=5)],
        }


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Alert management")
    parser.add_argument("--list", action="store_true", help="List recent alerts")
    parser.add_argument("--level", default=None, help="Filter by level")
    parser.add_argument("--summary", action="store_true", help="Show summary")
    parser.add_argument("--fire", nargs=3, metavar=("LEVEL", "SOURCE", "MESSAGE"),
                        help="Fire a test alert")
    args = parser.parse_args(argv)

    cfg = load()
    am = AlertManager(cfg)

    if args.fire:
        level, source, message = args.fire
        aid = am.fire(level, source, message)
        print(f"[alert] Fired: {aid}")
    elif args.summary:
        print(json.dumps(am.summary(), indent=2))
    elif args.list:
        for a in am.recent(level=args.level):
            print(f"  [{a.level.upper()}] {a.source}: {a.message} ({a.timestamp:.0f})")
    else:
        parser.print_help()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

"""Notification system for pipeline events.

Sends alerts when training completes, deploys happen, or something fails.
Supports desktop notifications, webhooks, and log files.

Usage:
    python -m src.cli notify --event=training_complete --message="combined won"
    python -m src.cli notify --event=deploy_failed --message="node2 unreachable"
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen
from urllib.error import URLError

from src.config import Config


@dataclass
class Notification:
    """A notification event."""
    event: str  # training_complete, deploy_complete, deploy_failed, harvest_complete, error
    message: str
    timestamp: float
    severity: str  # info, warning, error
    data: dict | None = None


# Event severity mapping
EVENT_SEVERITY = {
    "training_complete": "info",
    "training_started": "info",
    "training_failed": "error",
    "deploy_complete": "info",
    "deploy_failed": "error",
    "rollback_complete": "info",
    "rollback_failed": "error",
    "harvest_complete": "info",
    "harvest_failed": "error",
    "eval_complete": "info",
    "error": "error",
}


def send_desktop(title: str, message: str) -> bool:
    """Send a desktop notification via notify-send (Linux)."""
    try:
        subprocess.run(
            ["notify-send", "-u", "normal", "-i", "dialog-information", title, message],
            capture_output=True, timeout=5,
        )
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def send_webhook(url: str, notification: Notification) -> bool:
    """Send a notification to a webhook (Slack, Discord, etc.)."""
    if not url:
        return False

    payload = json.dumps({
        "text": f"[{notification.severity.upper()}] {notification.event}: {notification.message}",
        "event": notification.event,
        "severity": notification.severity,
        "timestamp": notification.timestamp,
        "data": notification.data,
    }).encode()

    req = Request(url, data=payload, headers={"Content-Type": "application/json"})

    try:
        with urlopen(req, timeout=10) as resp:
            return resp.status < 400
    except (URLError, OSError):
        return False


def send_emailsmtp(
    to: str, subject: str, body: str,
    smtp_host: str = "localhost", smtp_port: int = 25,
) -> bool:
    """Send email notification via SMTP."""
    try:
        import smtplib
        from email.mime.text import MIMEText

        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["To"] = to
        msg["From"] = "auto-finetune@localhost"

        with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as server:
            server.send_message(msg)
        return True
    except Exception:
        return False


def log_notification(notification: Notification, log_dir: str):
    """Append notification to a log file."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "notifications.jsonl")
    with open(log_file, "a") as f:
        f.write(json.dumps({
            "event": notification.event,
            "message": notification.message,
            "timestamp": notification.timestamp,
            "severity": notification.severity,
            "data": notification.data,
        }) + "\n")


def send_notification(
    cfg: Config,
    event: str,
    message: str,
    data: dict | None = None,
) -> dict[str, bool]:
    """Send a notification through all configured channels.

    Returns dict of channel -> success status.
    """
    severity = EVENT_SEVERITY.get(event, "info")
    notification = Notification(
        event=event,
        message=message,
        timestamp=time.time(),
        severity=severity,
        data=data,
    )

    results = {}

    # Desktop notification
    if cfg.get("notify", "desktop", default=True):
        results["desktop"] = send_desktop(f"auto-finetune: {event}", message)

    # Webhook
    webhook_url = cfg.get("notify", "webhook_url", default="")
    if webhook_url:
        results["webhook"] = send_webhook(webhook_url, notification)

    # Email
    email_to = cfg.get("notify", "email_to", default="")
    if email_to:
        smtp_host = cfg.get("notify", "smtp_host", default="localhost")
        smtp_port = cfg.get("notify", "smtp_port", default=25)
        results["email"] = send_emailsmtp(
            email_to, f"[auto-finetune] {event}", message,
            smtp_host=smtp_host, smtp_port=smtp_port,
        )

    # Log file (always)
    log_dir = cfg.get("notify", "log_dir",
                      default=os.path.join(cfg.path("analysis_dir"), "notifications"))
    log_notification(notification, log_dir)
    results["log"] = True

    return results


def get_notification_history(cfg: Config, limit: int = 50) -> list[dict]:
    """Get recent notifications from the log."""
    log_dir = cfg.get("notify", "log_dir",
                      default=os.path.join(cfg.path("analysis_dir"), "notifications"))
    log_file = os.path.join(log_dir, "notifications.jsonl")

    if not os.path.exists(log_file):
        return []

    notifications = []
    with open(log_file) as f:
        for line in f:
            line = line.strip()
            if line:
                notifications.append(json.loads(line))

    return notifications[-limit:]


def main(cfg: Config, argv: list[str]) -> int:
    """CLI handler for notify commands."""
    cmd = argv[1] if len(argv) > 1 else "notify-history"

    event = "info"
    message = ""
    data = None

    for arg in argv:
        if arg.startswith("--event="):
            event = arg.split("=", 1)[1]
        elif arg.startswith("--message="):
            message = arg.split("=", 1)[1]
        elif arg.startswith("--data="):
            try:
                data = json.loads(arg.split("=", 1)[1])
            except json.JSONDecodeError:
                pass

    if cmd == "notify":
        if not message:
            print("[error] notify requires --message=<text>")
            return 2
        results = send_notification(cfg, event, message, data)
        print(f"[notify] {event}: {message}")
        for channel, ok in results.items():
            status = "OK" if ok else "FAIL"
            print(f"  {channel}: {status}")
        return 0

    if cmd == "notify-history":
        limit = 50
        for arg in argv:
            if arg.startswith("--limit="):
                limit = int(arg.split("=", 1)[1])

        history = get_notification_history(cfg, limit)
        if not history:
            print("[notify-history] no notifications")
            return 0

        print(f"[notify-history] {len(history)} recent notifications:")
        for n in history:
            ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(n["timestamp"]))
            print(f"  [{n['severity']}] {ts} {n['event']}: {n['message']}")
        return 0

    print("Commands:")
    print("  notify --event=<name> --message=<text>")
    print("  notify-history [--limit=N]")
    print(f"  Events: {', '.join(EVENT_SEVERITY.keys())}")
    return 0

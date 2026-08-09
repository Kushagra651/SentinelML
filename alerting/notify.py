"""
alerting/notify.py

Sends alerts to Slack, PagerDuty, and/or SMTP email.
Uses stdlib only — no requests, no httpx, no third-party HTTP libraries.

Public API (matches all DAG call sites):
  send_alert(title, message, severity, channel="slack", labels=None) -> dict
  alert_info(title, message, **kwargs)
  alert_warning(title, message, **kwargs)
  alert_critical(title, message, **kwargs)

Channels:
  "slack"     — Slack Incoming Webhook (SLACK_WEBHOOK_URL env)
  "pagerduty" — PagerDuty Events API v2 (PAGERDUTY_ROUTING_KEY env)
  "email"     — SMTP (SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, ALERT_EMAIL_TO env)
  "all"       — all three channels
  "log"       — log only, no external calls (safe fallback / testing)

Severity levels: "info", "warning", "critical"

Design principles:
  - Every send path is wrapped in try/except — a broken alert channel NEVER
    raises into the calling DAG task. Alerting failure is logged, not propagated.
  - All HTTP is done with urllib.request — zero external dependencies.
  - Env vars read at call time, not module import time, so the module imports
    cleanly even if vars aren't set (Airflow imports all DAG files on scheduler
    startup before env vars may be fully available).
  - Returns a dict of {channel: success_bool} so callers can inspect results.
"""

from __future__ import annotations

import json
import logging
import os
import smtplib
import socket
import urllib.request
from datetime import datetime, timezone
from email.mime.text import MIMEText
from typing import Dict, Optional

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

VALID_SEVERITIES = {"info", "warning", "critical"}
VALID_CHANNELS = {"slack", "pagerduty", "email", "all", "log"}

# PagerDuty severity mapping
_PD_SEVERITY = {"info": "info", "warning": "warning", "critical": "critical"}

# HTTP timeout for all outbound webhook calls (seconds)
_HTTP_TIMEOUT = int(os.getenv("ALERT_HTTP_TIMEOUT", "10"))


# ── Internal HTTP helper ──────────────────────────────────────────────────────


def _post_json(url: str, payload: dict) -> tuple[int, str]:
    """
    POST JSON payload to url using stdlib urllib. Returns (status_code, body).
    Raises urllib.error.URLError on network failure — caller must handle.
    """
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "SentinelML-Alerting/1.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as resp:
        body = resp.read().decode("utf-8", errors="replace")
        return resp.status, body


# ── Channel implementations ───────────────────────────────────────────────────


def _send_slack(title: str, message: str, severity: str, labels: dict) -> bool:
    """
    Posts a formatted message to a Slack Incoming Webhook.
    SLACK_WEBHOOK_URL must be set in env. Returns True on success.
    """
    webhook_url = os.getenv("SLACK_WEBHOOK_URL", "")
    if not webhook_url:
        log.warning("SLACK_WEBHOOK_URL not set — skipping Slack alert.")
        return False

    # Emoji prefix by severity for quick visual scanning in Slack
    emoji = {"info": "ℹ️", "warning": "⚠️", "critical": "🔴"}.get(severity, "📢")

    label_str = ""
    if labels:
        label_str = "  |  " + "  ".join(f"`{k}={v}`" for k, v in labels.items())

    payload = {
        "text": f"{emoji} *{title}*{label_str}",
        "attachments": [
            {
                "color": {
                    "info": "#36a64f",
                    "warning": "#ff9900",
                    "critical": "#cc0000",
                }.get(severity, "#cccccc"),
                "text": message,
                "footer": f"SentinelML | {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
            }
        ],
    }

    try:
        status, body = _post_json(webhook_url, payload)
        if status == 200:
            log.info("Slack alert sent: %s", title)
            return True
        else:
            log.error("Slack webhook returned %d: %s", status, body[:200])
            return False
    except Exception as e:
        log.error("Slack alert failed: %s", e)
        return False


def _send_pagerduty(title: str, message: str, severity: str, labels: dict) -> bool:
    """
    Triggers a PagerDuty incident via Events API v2.
    PAGERDUTY_ROUTING_KEY must be set in env. Returns True on success.
    Only fires for warning and critical — info alerts are skipped.
    """
    routing_key = os.getenv("PAGERDUTY_ROUTING_KEY", "")
    if not routing_key:
        log.warning("PAGERDUTY_ROUTING_KEY not set — skipping PagerDuty alert.")
        return False

    if severity == "info":
        log.debug("PagerDuty: skipping info-level alert (not actionable).")
        return True  # Not a failure — deliberate skip

    # Dedup key: title + date, so repeated alerts for same issue don't flood PD
    dedup_key = f"sentinelml-{title.lower().replace(' ', '-')[:60]}-{datetime.now(timezone.utc).strftime('%Y%m%d')}"

    payload = {
        "routing_key": routing_key,
        "event_action": "trigger",
        "dedup_key": dedup_key,
        "payload": {
            "summary": title,
            "severity": _PD_SEVERITY.get(severity, "warning"),
            "source": socket.gethostname(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "custom_details": {
                "message": message,
                **{str(k): str(v) for k, v in (labels or {}).items()},
            },
        },
    }

    try:
        status, body = _post_json("https://events.pagerduty.com/v2/enqueue", payload)
        if status in (200, 202):
            log.info("PagerDuty alert sent: %s", title)
            return True
        else:
            log.error("PagerDuty returned %d: %s", status, body[:200])
            return False
    except Exception as e:
        log.error("PagerDuty alert failed: %s", e)
        return False


def _send_email(title: str, message: str, severity: str, labels: dict) -> bool:
    """
    Sends a plain-text alert email via SMTP.
    Required env: SMTP_HOST, ALERT_EMAIL_TO.
    Optional env: SMTP_PORT (default 587), SMTP_USER, SMTP_PASSWORD, ALERT_EMAIL_FROM.
    Returns True on success.
    """
    smtp_host = os.getenv("SMTP_HOST", "")
    to_addr = os.getenv("ALERT_EMAIL_TO", "")

    if not smtp_host or not to_addr:
        log.warning("SMTP_HOST or ALERT_EMAIL_TO not set — skipping email alert.")
        return False

    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER", "")
    smtp_password = os.getenv("SMTP_PASSWORD", "")
    from_addr = os.getenv("ALERT_EMAIL_FROM", smtp_user or "sentinelml@localhost")

    label_str = "\n".join(f"  {k}: {v}" for k, v in (labels or {}).items())
    body = f"{message}\n\nSeverity: {severity}\n{label_str}\n\nSentinelML | {datetime.now(timezone.utc).isoformat()}"

    msg = MIMEText(body, "plain")
    msg["Subject"] = f"[SentinelML/{severity.upper()}] {title}"
    msg["From"] = from_addr
    msg["To"] = to_addr

    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=_HTTP_TIMEOUT) as server:
            server.ehlo()
            if smtp_port != 25:
                server.starttls()
            if smtp_user and smtp_password:
                server.login(smtp_user, smtp_password)
            server.sendmail(from_addr, [to_addr], msg.as_string())
        log.info("Email alert sent: %s → %s", title, to_addr)
        return True
    except Exception as e:
        log.error("Email alert failed: %s", e)
        return False


def _log_only(title: str, message: str, severity: str, labels: dict) -> bool:
    """Logs the alert without sending anywhere. Always succeeds."""
    log.info(
        "ALERT [%s] %s | %s | labels=%s",
        severity.upper(),
        title,
        message,
        labels,
    )
    return True


# ── Public API ────────────────────────────────────────────────────────────────


def send_alert(
    title: str,
    message: str,
    severity: str = "warning",
    channel: str = "slack",
    labels: Optional[Dict[str, str]] = None,
) -> Dict[str, bool]:
    """
    Sends an alert to the specified channel(s).

    Args:
        title:    Short one-line summary (shown in Slack header / PD title / email subject)
        message:  Full alert body
        severity: "info" | "warning" | "critical"
        channel:  "slack" | "pagerduty" | "email" | "all" | "log"
        labels:   Optional key-value metadata attached to the alert (dag, task, etc.)

    Returns:
        Dict of {channel_name: success_bool} for every channel attempted.
        Never raises — all failures are caught and logged.
    """
    # Sanitise inputs — never let bad caller data blow up alerting
    severity = severity.lower() if severity else "warning"
    if severity not in VALID_SEVERITIES:
        log.warning("Unknown severity '%s', defaulting to 'warning'.", severity)
        severity = "warning"

    channel = channel.lower() if channel else "slack"
    if channel not in VALID_CHANNELS:
        log.warning("Unknown channel '%s', defaulting to 'log'.", channel)
        channel = "log"

    labels = labels or {}

    # Always log every alert regardless of channel — gives a guaranteed audit trail
    log.info("send_alert: channel=%s severity=%s title=%s", channel, severity, title)

    results: Dict[str, bool] = {}

    channels_to_send = (
        ["slack", "pagerduty", "email"] if channel == "all" else [channel]
    )

    _channel_fn = {
        "slack": _send_slack,
        "pagerduty": _send_pagerduty,
        "email": _send_email,
        "log": _log_only,
    }

    for ch in channels_to_send:
        fn = _channel_fn.get(ch, _log_only)
        try:
            results[ch] = fn(title, message, severity, labels)
        except Exception as e:
            # Belt-and-suspenders: individual channel fns already catch exceptions,
            # but if something truly unexpected happens we still never raise out.
            log.error("Unexpected error in channel '%s': %s", ch, e)
            results[ch] = False

    return results


# ── Convenience wrappers ──────────────────────────────────────────────────────


def alert_info(title: str, message: str, **kwargs) -> Dict[str, bool]:
    return send_alert(title=title, message=message, severity="info", **kwargs)


def alert_warning(title: str, message: str, **kwargs) -> Dict[str, bool]:
    return send_alert(title=title, message=message, severity="warning", **kwargs)


def alert_critical(title: str, message: str, **kwargs) -> Dict[str, bool]:
    return send_alert(title=title, message=message, severity="critical", **kwargs)

# notify_team.py
import os
from pathlib import Path
import smtplib
from email.message import EmailMessage

def notify_team(subject, message, to_addrs=None):
    """
    Simple notifier. If SMTP env vars set, send email; else print.
    Env: SMTP_SERVER, SMTP_PORT, SMTP_USER, SMTP_PASS, ALERT_TO
    """
    to_addrs = to_addrs or os.getenv("ALERT_TO", "")
    if not to_addrs:
        print("🔔 ALERT:", subject)
        print(message)
        return

    server = os.getenv("SMTP_SERVER")
    port = int(os.getenv("SMTP_PORT", 587))
    user = os.getenv("SMTP_USER")
    password = os.getenv("SMTP_PASS")
    if not all([server, user, password]):
        print("🔔 SMTP not configured, printing instead.")
        print(subject, message)
        return

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = to_addrs
    msg.set_content(message)

    with smtplib.SMTP(server, port) as s:
        s.starttls()
        s.login(user, password)
        s.send_message(msg)
    print("✅ Notification sent.")

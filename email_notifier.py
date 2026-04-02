"""
Email Alert System - WITH MEDIA ATTACHMENTS
"""

import smtplib
import threading
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime
from typing import Dict, Optional
import cv2
import io
import os


class EmailNotifier:

    def __init__(self, config):

        if isinstance(config, dict):
            self.config = config
        else:
            self.config = {}

        # Enable email
        self.enabled = self.config.get("enabled", True)

        self.cooldown_seconds = self.config.get("cooldown_seconds", 30)

        self._last_sent: Dict[str, float] = {}
        self._lock = threading.Lock()

        print("📧 Email Notifier Started")

        if self.enabled:
            print("Email Enabled")
            print("Sender:", self.config.get("sender_email"))
            print("Receiver:", self.config.get("recipient_email"))
        else:
            print("⚠ Email disabled")

    def send_alert(
        self,
        track_id: int,
        anomaly_type: str,
        severity: str,
        zone: Optional[str],
        description: str,
        frame=None,
        screenshot_path: Optional[str] = None,
        clip_path: Optional[str] = None,
    ) -> bool:

        print("\n================ ALERT ================")
        print("Track:", track_id)
        print("Type:", anomaly_type)
        print("Severity:", severity)
        print("Zone:", zone)
        print("Description:", description)
        print("Time:", datetime.now())
        print("======================================\n")

        if not self.enabled:
            print("⚠ Email disabled")
            return False

        alert_key = f"{track_id}_{anomaly_type}"

        with self._lock:

            if alert_key in self._last_sent:

                diff = time.time() - self._last_sent[alert_key]

                if diff < self.cooldown_seconds:
                    print("⏱ Cooldown active")
                    return False

            self._last_sent[alert_key] = time.time()

        try:

            subject = f"🚨 Surveillance Alert : {anomaly_type}"

            msg = MIMEMultipart()

            sender_email = self.config.get("sender_email")
            sender_password = self.config.get("sender_password")
            receiver_email = self.config.get("recipient_email")

            msg["From"] = sender_email
            msg["To"] = receiver_email
            msg["Subject"] = subject

            body = f"""
AI Surveillance Alert

Time : {datetime.now()}
Track ID : {track_id}
Anomaly : {anomaly_type}
Severity : {severity}
Zone : {zone}
Description : {description}

See attachments for evidence.
"""

            msg.attach(MIMEText(body, "plain"))

            # Attach live frame if screenshot not provided
            if frame is not None and screenshot_path is None:

                _, img = cv2.imencode(".jpg", frame)

                image = MIMEImage(img.tobytes())

                image.add_header(
                    "Content-Disposition", "attachment", filename="frame.jpg"
                )

                msg.attach(image)

                print("📎 Frame attached")

            # Attach screenshot
            if screenshot_path and os.path.exists(screenshot_path):

                with open(screenshot_path, "rb") as f:

                    img = MIMEImage(f.read())

                img.add_header(
                    "Content-Disposition",
                    "attachment",
                    filename=os.path.basename(screenshot_path),
                )

                msg.attach(img)

                print("📎 Screenshot attached")

            # Attach video
            if clip_path and os.path.exists(clip_path):

                with open(clip_path, "rb") as f:

                    video = MIMEBase("application", "octet-stream")

                    video.set_payload(f.read())

                encoders.encode_base64(video)

                video.add_header(
                    "Content-Disposition",
                    "attachment",
                    filename=os.path.basename(clip_path),
                )

                msg.attach(video)

                print("📎 Video attached")

            smtp_server = self.config.get("smtp_server", "smtp.gmail.com")
            smtp_port = self.config.get("smtp_port", 587)

            print("📤 Connecting to SMTP")

            server = smtplib.SMTP(smtp_server, smtp_port)

            server.starttls()

            print("🔐 Logging in")

            server.login(sender_email, sender_password)

            print("📨 Sending email")

            server.sendmail(sender_email, receiver_email, msg.as_string())

            server.quit()

            print("✅ Email Sent Successfully")

            return True

        except Exception as e:

            print("❌ Email sending failed")

            print(e)

            return False
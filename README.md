# Project Title

AI Surveillance System

## One-line tagline

Real-time AI-powered video surveillance for anomaly detection, multi-object tracking, and actionable alerts.

## Hero Image / Dashboard Screenshot

![Dashboard Screenshot](docs/assets/dashboard_screenshot.png)

*(Replace with your actual dashboard or KPI screenshot at docs/assets/dashboard_screenshot.png)*

## Business Problem

Organizations need reliable, real-time monitoring to detect security incidents and anomalous behavior across camera feeds, but manual monitoring is costly, error-prone, and doesn’t scale. Existing systems struggle with false positives, poor multi-object tracking, and limited actionable analytics.

## Objective

Deliver an end-to-end, real-time surveillance platform that detects and tracks people/objects, filters noise, identifies anomalies in behavior, logs events, and notifies stakeholders via automated alerts and an interactive dashboard.

## Solution Overview

AI Surveillance System uses a lightweight YOLOv8 detection model plus ByteTrack and Kalman filtering for robust multi-object tracking. A behavior and anomaly analysis pipeline filters noise, classifies events, and logs structured events to an SQLite database. Alerts are sent via SMTP and operators monitor feeds and event history through a Streamlit dashboard.

## Key Features

- Real-time object and person detection (YOLOv8)
- Multi-object tracking with ByteTrack and Kalman smoothing
- Human behavior classification (standing, walking, running)
- Anomaly detection (Isolation Forest / multivariate scoring)
- Noise and false-positive filtering (confidence + motion validation)
- Multi-zone ROI definitions with per-zone thresholds
- Event logging with associated media (frames/clips)
- Automated email notifications for critical events
- Interactive Streamlit dashboard with live stream and timeline

## Business Metrics / KPIs

- Detection latency: ~50–100 ms per frame (GPU)
- Tracking stability: <5% ID switches (multi-object tracking)
- Anomaly inference: real-time windowed scoring
- Storage growth: ~1 MB per 1000 events (SQLite)
- Dashboard update rate: sub-second for live monitoring
- Alert delivery success rate (SMTP) — monitor via debug_alerts.log

## Dashboard / Visualizations

- Live video feed with annotated bounding boxes and track IDs
- Event timeline and searchable history
- Zone overlays and per-zone event counts
- Anomaly heatmaps and trend charts (events over time)
- System health panel: frame rate, queue length, memory usage
- Event detail view with captured frames/clips and metadata

## Tech Stack

- Language: Python 3.x
- Detection: YOLOv8 (Ultralytics)
- Tracking: ByteTrack + Kalman Filter
- ML: Scikit-learn, NumPy
- Video: OpenCV
- Database: SQLite3
- Dashboard: Streamlit
- Notifications: SMTP (email)

## System Workflow

1. Capture frames (webcam / IP camera) via frame_manager.py
2. Detect objects (detector.py using YOLOv8) and filter persons (person_filter.py)
3. Track objects across frames with bytetrack.py and smooth trajectories using kalman_filter.py
4. Buffer track history in track_buffer.py and apply behavior_analyzer.py for activity classification
5. Run anomaly_engine.py to score and classify unusual patterns; apply noise_detector.py to suppress false positives
6. Log structured events to surveillance_events.db via event_logger.py and store associated media via media_manager.py
7. If thresholds exceeded, send notifications via email_notifier.py
8. Visualize live feed, events, and controls in Streamlit dashboard (main_dashboard.py)

## Project Structure

- Detection & Tracking: detector.py, person_filter.py, bytetrack.py, kalman_filter.py
- Analysis Engines: anomaly_engine.py, behavior_analyzer.py, noise_detector.py
- Pipeline & State: frame_manager.py, track_buffer.py, zone_manager.py, settings.py
- Persistence & Media: event_logger.py, media_manager.py, surveillance_events.db
- Notifications: email_notifier.py, email_config.json
- UI: main_dashboard.py, visualization.py, dashboard.py (legacy)
- Utilities & Setup: init_database.py, requirements.txt

## Installation

Prerequisites:
- Python 3.8+
- (Optional) CUDA-capable GPU for acceleration
- Webcam or IP camera feed
- SMTP email account for notifications

Steps:
1. git clone https://github.com/revan5x/SURVILLENCE_SYS.git
2. cd SURVILLENCE_SYS
3. pip install -r requirements.txt
4. python init_database.py
5. Update settings.py, email_config.json, and zones.json as required

## Usage

- Run full system (engine + dashboard): `python run_full_system.py`
- Run engine only: `python main.py`
- Launch dashboard only: `streamlit run main_dashboard.py`
- Test anomaly engine: `python test_anomaly.py`
- Test email notifications: `python test_email.py`

## Project Outcomes

- Real-time detection and tracking pipeline capable of sub-100ms per-frame latency on GPU
- Robust multi-object tracking with low ID-switch rates and historical track buffers for context
- Operational anomaly detection and per-zone alerting with persistent event logs and media capture
- Interactive dashboard enabling operators to monitor, review, and configure zones and alerts

## Future Improvements

- Multi-GPU and distributed processing for scale
- RTSP/multi-stream aggregation
- Cloud database sync and REST API for integrations
- Facial recognition and vehicle plate recognition modules (privacy/legal considerations required)
- Mobile app for alert management and push notifications
- Advanced crowd behavior modeling and loitering detection

## Repository Structure

- README.md
- main.py — core surveillance engine
- run_full_system.py — orchestrator (engine + dashboard)
- main_dashboard.py — Streamlit UI entrypoint
- detector.py, person_filter.py, bytetrack.py, kalman_filter.py
- anomaly_engine.py, behavior_analyzer.py, noise_detector.py
- frame_manager.py, track_buffer.py, zone_manager.py, event_logger.py, media_manager.py
- email_notifier.py, email_config.json, zones.json, settings.py
- init_database.py, requirements.txt
- tests: test_anomaly.py, test_email.py
- data: surveillance_events.db, yolov8n.pt
- logs: debug_alerts.log

## License

Provided as-is for surveillance and security purposes. Ensure you comply with local laws and privacy regulations before deployment.

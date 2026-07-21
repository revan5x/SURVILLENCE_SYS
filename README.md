# 🛡️ AI Surveillance System

### Intelligent Video Analytics Platform for Real-Time Monitoring and Operational Insights

> An AI-powered surveillance platform that automates object detection, behavior analysis, anomaly detection, and real-time alerting through an interactive analytics dashboard.

---

## 🎯 Business Problem

Traditional surveillance systems depend heavily on manual monitoring, making them difficult to scale and prone to missed incidents, delayed responses, and operator fatigue. Organizations require an intelligent monitoring solution capable of automatically detecting suspicious activities, reducing false alarms, and providing actionable operational insights in real time.

---

## 🎯 Project Objective

Develop an end-to-end intelligent surveillance platform that:

- Detects and tracks people in real time
- Identifies anomalous behavior automatically
- Reduces false positives using AI-based filtering
- Generates real-time alerts for critical events
- Provides operational insights through an interactive analytics dashboard

---

## 💡 Solution Overview

The AI Surveillance System combines computer vision, machine learning, and business intelligence into a single monitoring platform.

Using **YOLOv8** for object detection, **ByteTrack** and **Kalman Filtering** for multi-object tracking, and AI-based anomaly detection, the system continuously analyzes surveillance footage, identifies suspicious activities, records structured event data, and delivers automated email alerts. A **Streamlit dashboard** enables operators to monitor live feeds, review historical events, track operational KPIs, and support faster data-driven security decisions.

## ✨ Key Features

- 🎥 Real-time person and object detection using YOLOv8
- 👥 Multi-object tracking with ByteTrack and Kalman Filtering
- 🚶 Human activity recognition and behavior analysis
- 🚨 AI-powered anomaly detection for unusual events
- 🎯 Zone-based monitoring with configurable alert thresholds
- 📩 Automated email notifications for critical incidents
- 🗄️ Event logging with timestamps and media snapshots
- 📊 Interactive Streamlit dashboard for real-time monitoring and analytics

---

## 📊 Business Impact & KPIs

The platform helps security teams reduce manual monitoring effort while improving incident response through automated detection and operational visibility.

| Metric | Value |
|---------|-------|
| Detection Latency | 50–100 ms/frame (GPU) |
| Multi-Object Tracking | <5% ID Switches |
| Real-Time Alerting | Automated Email Notifications |
| Dashboard Refresh | Sub-second Updates |
| Event Storage | ~1 MB per 1000 Events |
| Operational Monitoring | Live KPIs & Historical Trends |

---

## 📈 Dashboard & Analytics

The Streamlit dashboard enables operators to monitor system performance and security events through:

- 📹 Live video feed with detected objects and tracking IDs
- 📅 Event timeline with searchable incident history
- 📍 Zone-wise event distribution and activity counts
- 📊 Event trend visualizations and anomaly analytics
- ⚙️ System health metrics (FPS, memory usage, processing status)
- 🖼️ Incident details with captured images and metadata

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| **Programming** | Python |
| **Computer Vision** | YOLOv8, OpenCV |
| **Object Tracking** | ByteTrack, Kalman Filter |
| **Machine Learning** | Scikit-learn, NumPy |
| **Database** | SQLite |
| **Dashboard** | Streamlit |
| **Notifications** | SMTP Email |

## ⚙️ System Workflow

```text
Video Input
      │
      ▼
Object Detection (YOLOv8)
      │
      ▼
Multi-Object Tracking (ByteTrack + Kalman Filter)
      │
      ▼
Behavior Analysis & Anomaly Detection
      │
      ▼
Event Logging (SQLite)
      │
      ├────────► Email Alerts
      │
      ▼
Streamlit Dashboard & Analytics
```

### Workflow Overview

1. Capture live video from a webcam or IP camera.
2. Detect people and objects using **YOLOv8**.
3. Track objects across frames with **ByteTrack** and **Kalman Filtering**.
4. Analyze movement patterns and classify human activities.
5. Detect anomalous events while reducing false positives.
6. Store events, timestamps, and media in an SQLite database.
7. Trigger automated email alerts for critical incidents.
8. Display live monitoring, historical events, and operational metrics through a Streamlit dashboard.

---

## 📂 Project Structure

```text
AI_SURVEILLANCE_SYSTEM/
│
├── Detection & Tracking
│   ├── detector.py
│   ├── person_filter.py
│   ├── bytetrack.py
│   └── kalman_filter.py
│
├── Analysis Engine
│   ├── anomaly_engine.py
│   ├── behavior_analyzer.py
│   └── noise_detector.py
│
├── Data & Event Management
│   ├── event_logger.py
│   ├── media_manager.py
│   └── surveillance_events.db
│
├── Dashboard
│   ├── main_dashboard.py
│   └── visualization.py
│
├── Notifications
│   └── email_notifier.py
│
└── Configuration
    ├── settings.py
    ├── zones.json
    └── requirements.txt
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8+
- Webcam or IP Camera
- SMTP Email Account (for alerts)
- CUDA-enabled GPU *(optional for faster inference)*

### Setup

```bash
git clone https://github.com/revan5x/SURVILLENCE_SYS.git
cd SURVILLENCE_SYS

pip install -r requirements.txt

python init_database.py
```

Update the following configuration files before running the project:

- `settings.py`
- `email_config.json`
- `zones.json`

---

## ▶️ Running the Project

| Task | Command |
|------|---------|
| Run complete system | `python run_full_system.py` |
| Run detection engine | `python main.py` |
| Launch dashboard | `streamlit run main_dashboard.py` |
| Test anomaly detection | `python test_anomaly.py` |
| Test email notifications | `python test_email.py` |

## 📌 Project Outcomes

- Developed an end-to-end AI surveillance platform capable of real-time object detection, tracking, and anomaly detection.
- Automated security monitoring by reducing manual intervention through intelligent event detection and email alerts.
- Built an interactive Streamlit dashboard to monitor live feeds, visualize operational metrics, and review historical events.
- Designed a modular and scalable architecture that separates detection, tracking, analytics, event management, and visualization.

---

## 🚀 Future Enhancements

- Support multiple camera streams (RTSP/IP Cameras)
- Cloud-based event storage and analytics
- REST API for third-party integrations
- Mobile application for real-time alert management
- Advanced crowd analytics and loitering detection
- Edge deployment for low-latency inference

---

## 🤝 Contributing

Contributions, suggestions, and feature requests are welcome.

If you'd like to improve the project, feel free to fork the repository and submit a pull request.

---

## 📄 License

This project is intended for educational and research purposes.

Ensure compliance with applicable privacy laws and surveillance regulations before deploying in production environments.

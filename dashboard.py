import streamlit as st
import cv2
import time
import psutil
from datetime import datetime

from ingestion.frame_manager import FrameManager
from vision.detector import PersonDetector
from tracking.bytetrack import ByteTracker
from logic.behavior_analyzer import BehaviorAnalyzer
from logic.anomaly_engine import AnomalyEngine
from utils.visualization import Visualizer

# ---------------- UI THEME ---------------- #
st.set_page_config(page_title="AI Surveillance", layout="wide")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-color: #0A0A0A;
    color: #E5E7EB;
}
[data-testid="stSidebar"] {
    background-color: #111827;
}
.stMetric {
    background-color: #1E1E2E;
    padding: 12px;
    border-radius: 10px;
}
.stButton > button {
    background-color: #7F00FF;
    color: white;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- INIT ---------------- #
if "running" not in st.session_state:
    st.session_state.running = False

# ---------------- SIDEBAR ---------------- #
st.sidebar.markdown("## ⚙️ Control Panel")

if not st.session_state.running:
    if st.sidebar.button("🟢 START SYSTEM"):
        st.session_state.running = True
else:
    if st.sidebar.button("🔴 STOP SYSTEM"):
        st.session_state.running = False

# ---------------- TITLE ---------------- #
st.markdown("## 🔒 AI Surveillance System")
st.caption("Real-Time Edge Monitoring • CPU Optimized • Privacy Preserving")

# ---------------- LAYOUT ---------------- #
video_col, info_col = st.columns([3, 1])

video_placeholder = video_col.empty()

status_text = video_col.empty()
tracks_text = video_col.empty()
fps_text = video_col.empty()

cpu_metric = info_col.empty()
mem_metric = info_col.empty()
events_box = info_col.empty()

# ---------------- SYSTEM INIT ---------------- #
if st.session_state.running:

    frame_manager = FrameManager(source=0)
    detector = PersonDetector()
    tracker = ByteTracker()
    behavior = BehaviorAnalyzer()
    anomaly = AnomalyEngine()
    visualizer = Visualizer()

    frame_manager.start()

    recent_events = []

    last_time = time.time()

    while st.session_state.running:

        frame, _, _ = frame_manager.get_frame()

        if frame is None:
            continue

        # -------- DETECTION -------- #
        detections, inf_time = detector.detect(frame)

        # -------- TRACKING -------- #
        tracks = tracker.update(detections, time.time())

        behavior_profiles = {}

        for t in tracks:
            profile = behavior.analyze(t, None)
            behavior_profiles[t.track_id] = profile

            event = anomaly.evaluate(profile)

            if event:
                recent_events.append(
                    f"{datetime.now().strftime('%H:%M:%S')} | ID {event.track_id} | {event.anomaly_type}"
                )
                recent_events = recent_events[-5:]

        # -------- VISUALIZATION -------- #
        frame = visualizer.draw_tracks(frame, tracks, behavior_profiles)

        # -------- METRICS -------- #
        now = time.time()
        fps = 1 / (now - last_time) if (now - last_time) > 0 else 0
        last_time = now

        cpu = psutil.cpu_percent()
        mem = psutil.virtual_memory().percent

        # -------- DISPLAY -------- #
        video_placeholder.image(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            use_column_width=True
        )

        status_text.metric("System", "ACTIVE")
        tracks_text.metric("Tracks", len(tracks))
        fps_text.metric("FPS", f"{fps:.1f}")

        cpu_metric.metric("CPU", f"{cpu}%")
        mem_metric.metric("Memory", f"{mem}%")

        if recent_events:
            events_box.markdown(
                "### 🚨 Recent Events\n```\n" + "\n".join(reversed(recent_events)) + "\n```"
            )
        else:
            events_box.markdown("### 🚨 Recent Events\nNo alerts")

        time.sleep(0.01)

else:
    st.info("👈 Click START SYSTEM to begin surveillance")

"""
Streamlit Dashboard Interface
Live feed, zone configuration, event analysis
"""

import streamlit as st
import cv2
import numpy as np
import threading
import time
import queue
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import pandas as pd
import sys
import os

# Set page config FIRST (before any other st commands)
st.set_page_config(
    page_title="AI Surveillance System",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class DashboardManager:
    """
    Streamlit-based monitoring dashboard
    """
    
    def __init__(self):
        self._init_session_state()
    
    def _init_session_state(self):
        """Initialize Streamlit session state"""
        defaults = {
            'zones': {},
            'privacy_mode': True,
            'system_running': False,
            'events': [],
            'performance_history': [],
            'frame_count': 0,
            'last_fps': 0.0,
            'active_tracks': 0
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def render_sidebar(self):
        """Render configuration sidebar"""
        st.sidebar.title("🔧 System Configuration")
        
        # Zone Management
        st.sidebar.header("Zone Management")
        
        with st.sidebar.expander("➕ Add New Zone", expanded=False):
            zone_name = st.text_input("Zone Name", key="new_zone_name", value="Zone1")
            is_restricted = st.checkbox("Restricted Zone", key="new_zone_restricted", value=True)
            
            st.write("Define polygon points (x,y):")
            col1, col2 = st.columns(2)
            with col1:
                x_points = st.text_area("X coords", "100,200,200,100", height=100, key="zone_x")
            with col2:
                y_points = st.text_area("Y coords", "100,100,200,200", height=100, key="zone_y")
            
            if st.button("Add Zone", key="add_zone_btn", use_container_width=True):
                try:
                    xs = [int(x.strip()) for x in x_points.split(",") if x.strip()]
                    ys = [int(y.strip()) for y in y_points.split(",") if y.strip()]
                    
                    if len(xs) == len(ys) and len(xs) >= 3:
                        polygon = list(zip(xs, ys))
                        st.session_state.zones[zone_name] = {
                            'polygon': polygon,
                            'restricted': is_restricted,
                            'allowed_directions': ['N', 'S'] if is_restricted else ['N', 'S', 'E', 'W']
                        }
                        st.success(f"✅ Zone '{zone_name}' added!")
                        st.rerun()
                    else:
                        st.error("❌ Need equal x,y pairs (min 3 points)")
                except ValueError as e:
                    st.error(f"❌ Invalid coordinates: {e}")
        
        # Display existing zones
        if st.session_state.zones:
            st.sidebar.subheader("📍 Active Zones")
            for name, zone in list(st.session_state.zones.items()):
                col1, col2 = st.sidebar.columns([4, 1])
                icon = "🔴" if zone.get('restricted') else "🟢"
                col1.write(f"{icon} {name}")
                if col2.button("🗑️", key=f"del_{name}"):
                    del st.session_state.zones[name]
                    st.rerun()
        else:
            st.sidebar.info("No zones defined. Add zones above.")
        
        # Privacy Settings
        st.sidebar.header("🔒 Privacy Controls")
        st.session_state.privacy_mode = st.toggle(
            "Face Blurring (Always On)", 
            value=st.session_state.privacy_mode,
            disabled=True  # Force always on
        )
        
        # System Controls
        st.sidebar.header("▶️ System Control")
        
        if not st.session_state.system_running:
            if st.button("🚀 START SURVEILLANCE", type="primary", use_container_width=True):
                st.session_state.system_running = True
                st.rerun()
        else:
            if st.button("⏹️ STOP SYSTEM", type="secondary", use_container_width=True):
                st.session_state.system_running = False
                st.rerun()
        
        # Status indicator
        status = "🟢 RUNNING" if st.session_state.system_running else "🔴 STOPPED"
        st.sidebar.markdown(f"**Status:** {status}")
        
        # Performance Settings
        st.sidebar.header("⚡ Performance")
        st.session_state.frame_skip = st.slider(
            "Frame Skip (CPU optimization)", 
            1, 5, 2,
            help="Process every Nth frame. Higher = less CPU usage"
        )
    
    def render_main_interface(self):
        """Render main dashboard layout"""
        # Title
        st.title("🔒 AI Surveillance System")
        st.markdown("*Edge-Based Intelligent Surveillance with Privacy Protection*")
        st.divider()
        
        # Create layout
        video_col, stats_col = st.columns([3, 2])
        
        with video_col:
            st.subheader("📹 Live Feed")
            
            # Video placeholder (will be updated by main loop)
            video_placeholder = st.empty()
            
            # Status bar below video
            status_col1, status_col2, status_col3 = st.columns(3)
            with status_col1:
                st.metric("System Status", "Active" if st.session_state.system_running else "Standby")
            with status_col2:
                track_metric = st.metric("Active Tracks", st.session_state.active_tracks)
            with status_col3:
                fps_metric = st.metric("FPS", f"{st.session_state.last_fps:.1f}")
        
        with stats_col:
            st.subheader("📊 Real-time Metrics")
            
            # Performance gauges
            perf_container = st.container()
            with perf_container:
                cpu_placeholder = st.empty()
                mem_placeholder = st.empty()
                latency_placeholder = st.empty()
            
            st.divider()
            
            st.subheader("🚨 Recent Events")
            # Event log placeholder
            events_placeholder = st.empty()
            
            st.divider()
            
            # Quick stats
            st.subheader("📈 Session Statistics")
            stats_placeholder = st.empty()
        
        return {
            'video': video_placeholder,
            'fps': fps_metric,
            'tracks': track_metric,
            'cpu': cpu_placeholder,
            'memory': mem_placeholder,
            'latency': latency_placeholder,
            'events': events_placeholder,
            'stats': stats_placeholder
        }
    
    def render_event_query(self, event_logger=None):
        """Render historical event query interface"""
        st.divider()
        st.header("📊 Event Analysis & Reports")
        
        tab1, tab2 = st.tabs(["🔍 Query Events", "📥 Export Reports"])
        
        with tab1:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                start_date = st.date_input("Start Date", datetime.now() - timedelta(days=1))
            with col2:
                end_date = st.date_input("End Date", datetime.now())
            with col3:
                severity_filter = st.multiselect(
                    "Severity Filter",
                    ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
                    default=["HIGH", "CRITICAL"]
                )
            
            if st.button("🔍 Query Database", type="primary", use_container_width=True):
                if event_logger:
                    try:
                        start_dt = datetime.combine(start_date, datetime.min.time())
                        end_dt = datetime.combine(end_date, datetime.max.time())
                        
                        severity = severity_filter[0] if len(severity_filter) == 1 else None
                        
                        events = event_logger.query_events(
                            start_time=start_dt,
                            end_time=end_dt,
                            severity=severity,
                            limit=100
                        )
                        
                        if events:
                            st.success(f"✅ Found {len(events)} events")
                            df = pd.DataFrame(events)
                            st.dataframe(df, use_container_width=True, height=400)
                            
                            # Download button
                            csv = df.to_csv(index=False)
                            st.download_button(
                                "⬇️ Download CSV",
                                csv,
                                f"events_{start_date}_{end_date}.csv",
                                "text/csv"
                            )
                        else:
                            st.info("ℹ️ No events found for selected criteria")
                    except Exception as e:
                        st.error(f"❌ Query error: {e}")
                else:
                    st.error("❌ Event logger not available")
        
        with tab2:
            st.write("Generate daily activity reports")
            
            report_date = st.date_input("Report Date", datetime.now() - timedelta(days=1))
            
            if st.button("📄 Generate Report", use_container_width=True):
                if event_logger:
                    try:
                        report_path = event_logger.generate_daily_report(
                            datetime.combine(report_date, datetime.min.time())
                        )
                        
                        with open(report_path, 'r') as f:
                            report_content = f.read()
                        
                        st.download_button(
                            "⬇️ Download Report CSV",
                            report_content,
                            f"daily_report_{report_date}.csv",
                            "text/csv"
                        )
                        
                        st.success(f"✅ Report generated: {report_path}")
                    except Exception as e:
                        st.error(f"❌ Report generation failed: {e}")
                else:
                    st.error("❌ Event logger not available")
    
    def update_video(self, placeholder, frame: np.ndarray):
        """Update video display"""
        if frame is not None:
            # Convert BGR to RGB for Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
    
    def update_metrics(self, 
                      placeholders: Dict,
                      performance: Dict,
                      recent_events: List[Dict]):
        """Update all dashboard metrics"""
        
        # Update session state
        st.session_state.last_fps = performance.get('fps', 0)
        st.session_state.active_tracks = performance.get('active_tracks', 0)
        
        # Update performance gauges
        cpu_val = performance.get('cpu_percent', 0)
        cpu_color = "normal" if cpu_val < 70 else "off" if cpu_val < 85 else "inverse"
        placeholders['cpu'].metric(
            "CPU Usage", 
            f"{cpu_val:.1f}%",
            delta=None
        )
        
        mem_val = performance.get('memory_mb', 0)
        placeholders['memory'].metric(
            "Memory", 
            f"{mem_val:.1f} MB"
        )
        
        lat_val = performance.get('inference_ms', 0)
        placeholders['latency'].metric(
            "Inference", 
            f"{lat_val:.1f} ms"
        )
        
        # Update events table
        if recent_events:
            # Format events for display
            display_events = []
            for e in recent_events[-10:]:  # Last 10 events
                display_events.append({
                    'Time': e.get('timestamp', 'N/A')[-8:] if isinstance(e.get('timestamp'), str) else 'Now',
                    'Track': e.get('track_id', 'N/A'),
                    'Type': e.get('anomaly_type', 'N/A')[:20],
                    'Severity': e.get('severity', 'N/A'),
                    'Zone': e.get('zone_name', 'General')[:15]
                })
            
            placeholders['events'].dataframe(
                pd.DataFrame(display_events),
                use_container_width=True,
                height=250
            )
        else:
            placeholders['events'].info("No events in current session")
        
        # Update stats
        total_events = len(recent_events)
        critical = sum(1 for e in recent_events if e.get('severity') == 'CRITICAL')
        high = sum(1 for e in recent_events if e.get('severity') == 'HIGH')
        
        stats_text = f"""
        **Session Events:** {total_events}
        - 🔴 Critical: {critical}
        - 🟠 High: {high}
        - 🟡 Medium/Low: {total_events - critical - high}
        
        **System Health:** {'✅ Optimal' if performance.get('fps', 0) > 20 else '⚠️ Degraded' if performance.get('fps', 0) > 10 else '❌ Critical'}
        """
        placeholders['stats'].markdown(stats_text)


def run_dashboard(system_instance=None):
    """Main entry point for dashboard with live processing"""
    from config.settings import ConfigManager
    from storage.event_logger import EventLogger
    
    dashboard = DashboardManager()
    dashboard.render_sidebar()
    
    # Get placeholders
    placeholders = dashboard.render_main_interface()
    
    # Event query section (only show when not running)
    if not st.session_state.system_running:
        event_logger = EventLogger(ConfigManager.get_db_path()) if system_instance else None
        dashboard.render_event_query(event_logger)
        return dashboard, placeholders, False
    
    return dashboard, placeholders, True  # Signal to run processing


# For standalone testing
if __name__ == "__main__":
    run_dashboard()
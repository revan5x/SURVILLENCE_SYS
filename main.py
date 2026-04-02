"""
AI Surveillance System - Main Execution Pipeline
CORRECTED: Proper integration of all modules
"""

import sys
import os
import time
import threading
import queue
from datetime import datetime
from typing import Optional, Dict, List

import cv2
import numpy as np
import psutil

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import CONFIG, ConfigManager
from ingestion.frame_manager import FrameManager
from vision.detector import PersonDetector, Detection
from tracking.bytetrack import ByteTracker, Track
from logic.zone_manager import ZoneManager, ZoneConfig
from logic.behavior_analyzer import BehaviorAnalyzer, BehaviorType
from logic.anomaly_engine import AnomalyEngine, AnomalyEvent, Severity
from storage.event_logger import EventLogger
from alerts.email_notifier import EmailNotifier
from utils.visualization import Visualizer


class SurveillanceSystem:
    """
    Main system controller
    Single-process, CPU-only, modular pipeline
    """
    
    def __init__(self):
        print("🔒 Initializing AI Surveillance System...")
        
        # Initialize components
        self.config = CONFIG.SYSTEM
        
        # Ingestion
        self.frame_manager = FrameManager(
            source=0,
            width=self.config.FRAME_WIDTH,
            height=self.config.FRAME_HEIGHT,
            fps=self.config.FPS_TARGET,
            frame_skip=self.config.FRAME_SKIP
        )
        
        # Vision
        self.detector = PersonDetector(
            confidence_threshold=self.config.DETECTION_CONFIDENCE,
            iou_threshold=self.config.IOU_THRESHOLD
        )
        
        # Tracking
        self.tracker = ByteTracker(
            high_thresh=self.config.TRACK_HIGH_THRESH,
            low_thresh=self.config.TRACK_LOW_THRESH,
            new_track_thresh=self.config.NEW_TRACK_THRESH,
            match_thresh=self.config.MATCH_THRESH,
            max_lost=self.config.MAX_LOST_FRAMES
        )
        
        # Logic
        self.zone_manager = ZoneManager()
        self.behavior_analyzer = BehaviorAnalyzer(
            velocity_window=self.config.VELOCITY_WINDOW,
            stationary_threshold=self.config.STATIONARY_THRESHOLD,
            loitering_time=self.config.LOITERING_TIME,
            rapid_threshold=200.0
        )
        self.anomaly_engine = AnomalyEngine(
            persistence_time=self.config.ANOMALY_PERSISTENCE,
            alert_cooldown=self.config.ALERT_COOLDOWN
        )
        
        # Storage & Alerts
        self.event_logger = EventLogger(ConfigManager.get_db_path())
        self.email_notifier = EmailNotifier(ConfigManager.get_email_config())
        
        # Visualization
        self.visualizer = Visualizer(privacy_mode=self.config.PRIVACY_MODE)
        
        # Performance monitoring
        self.process = psutil.Process()
        self.performance_history = []
        self.max_history = 100
        
        # State
        self.running = False
        self.frame_count = 0
        self.last_events = []
        
        # Load default zones
        self._load_default_zones()
        
        print("✅ System initialized successfully")
    
    def _load_default_zones(self):
        """Load default zones from config"""
        for name, zone_config in ConfigManager.DEFAULT_ZONES.items():
            self.zone_manager.add_zone(zone_config)
            print(f"   📍 Loaded zone: {name}")
    
    def add_zone(self, name: str, polygon: List[tuple], restricted: bool = False):
        """Add dynamic zone"""
        zone_config = ZoneConfig(
            name=name,
            polygon=polygon,
            is_restricted=restricted
        )
        self.zone_manager.add_zone(zone_config)
        print(f"➕ Added zone: {name}")
    
    def _update_performance(self, 
                         inference_time: float,
                         active_tracks: int,
                         detection_latency: float):
        """Update performance metrics"""
        fps = 1.0 / (time.time() - self._last_frame_time) if hasattr(self, '_last_frame_time') else 0
        self._last_frame_time = time.time()
        
        # Get system stats
        cpu_percent = self.process.cpu_percent() / psutil.cpu_count()
        memory_mb = self.process.memory_info().rss / 1024 / 1024
        
        metrics = {
            'timestamp': datetime.now(),
            'fps': fps,
            'inference_ms': inference_time,
            'cpu_percent': cpu_percent,
            'memory_mb': memory_mb,
            'active_tracks': active_tracks,
            'detection_latency_ms': detection_latency
        }
        
        self.performance_history.append(metrics)
        if len(self.performance_history) > self.max_history:
            self.performance_history.pop(0)
        
        # Log to database every 5 seconds
        if self.frame_count % 150 == 0:
            self.event_logger.log_performance(
                fps=fps,
                inference_time_ms=inference_time,
                cpu_percent=cpu_percent,
                memory_mb=memory_mb,
                active_tracks=active_tracks,
                detection_latency_ms=detection_latency
            )
        
        return metrics
    
    def _process_frame(self, frame: np.ndarray, timestamp: float) -> tuple:
        """
        Process single frame through pipeline
        Returns: (annotated_frame, events_generated)
        """
        events_generated = []
        
        # Step 1: Detection
        det_start = time.perf_counter()
        detections, inference_time = self.detector.detect(frame)
        detection_latency = (time.perf_counter() - det_start) * 1000
        
        # Step 2: Privacy protection (always-on blurring)
        privacy_frame = self.visualizer.apply_privacy_blur(frame, detections)
        
        # Step 3: Tracking
        tracks = self.tracker.update(detections, timestamp)
        active_track_ids = [t.track_id for t in tracks]
        
        # Step 4: Behavior Analysis & Anomaly Detection
        behavior_profiles = {}
        
        for track in tracks:
            # Analyze behavior
            profile = self.behavior_analyzer.analyze(track, self.zone_manager)
            behavior_profiles[track.track_id] = profile
            
            # Evaluate for anomaly
            event = self.anomaly_engine.evaluate(profile)
            
            if event:
                events_generated.append(event)
                
                # Log event
                self.event_logger.log_event(
                    track_id=event.track_id,
                    anomaly_type=event.anomaly_type,
                    severity=event.severity.value,
                    zone=event.zone,
                    velocity=event.velocity,
                    duration=event.duration,
                    direction=profile.direction,
                    metadata={
                        'bbox': track.buffer.observations[-1].bbox if track.buffer.observations else None,
                        'position': profile.current_position
                    }
                )
                
                # Send alert (async)
                self.email_notifier.send_alert(
                    track_id=event.track_id,
                    anomaly_type=event.anomaly_type,
                    severity=event.severity.value,
                    zone=event.zone,
                    description=event.description
                )
                
                print(f"🚨 ANOMALY DETECTED: {event.description} (Severity: {event.severity.value})")
        
        # Cleanup old tracks
        self.behavior_analyzer.cleanup_old_tracks(active_track_ids)
        self.anomaly_engine.cleanup_old_tracks(active_track_ids)
        
        # Step 5: Visualization
        # Draw zones
        display_frame = self.zone_manager.draw_zones(privacy_frame)
        
        # Draw tracks
        display_frame = self.visualizer.draw_tracks(display_frame, tracks, behavior_profiles)
        
        # Update performance
        metrics = self._update_performance(inference_time, len(tracks), detection_latency)
        
        # Draw HUD
        display_frame = self.visualizer.draw_hud(
            display_frame,
            fps=metrics['fps'],
            cpu_percent=metrics['cpu_percent'],
            memory_mb=metrics['memory_mb'],
            active_tracks=metrics['active_tracks'],
            inference_time=metrics['inference_ms'],
            detection_latency=metrics['detection_latency_ms']
        )
        
        # Draw recent events panel
        recent_events = self.event_logger.query_events(limit=5)
        display_frame = self.visualizer.draw_events_panel(display_frame, recent_events)
        
        return display_frame, events_generated
    
    def run(self, display_window: bool = True):
        """
        Main execution loop
        """
        print("🚀 Starting surveillance pipeline...")
        
        # Start frame capture
        if not self.frame_manager.start():
            print("❌ Failed to start video capture")
            return
        
        self.running = True
        self._last_frame_time = time.time()
        
        print("✅ Pipeline running. Press 'Q' to quit.")
        print("   Zones active:", list(self.zone_manager.zones.keys()))
        
        try:
            while self.running:
                # Get frame
                frame, timestamp, should_process = self.frame_manager.get_frame()
                
                if frame is None:
                    continue
                
                self.frame_count += 1
                
                # Skip frames for CPU optimization
                if not should_process:
                    continue
                
                # Process frame
                display_frame, events = self._process_frame(frame, timestamp)
                
                if events:
                    self.last_events.extend(events)
                    # Keep only recent events
                    self.last_events = self.last_events[-50:]
                
                # Display
                if display_window:
                    cv2.imshow('AI Surveillance System', display_frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        
        except KeyboardInterrupt:
            print("\n⏹️ Interrupted by user")
        finally:
            self.shutdown()
    
    def run_headless(self):
        """Run without display window (for servers)"""
        self.run(display_window=False)
    
    def shutdown(self):
        """Clean shutdown"""
        print("🛑 Shutting down system...")
        self.running = False
        self.frame_manager.release()
        cv2.destroyAllWindows()
        print("✅ System shutdown complete")


def main():
    """Entry point"""
    system = SurveillanceSystem()
    
    # Example: Add custom zone if needed
    # system.add_zone("TestZone", [(100,100), (200,100), (200,200), (100,200)], restricted=True)
    
    # Run system
    system.run(display_window=True)


if __name__ == "__main__":
    main()
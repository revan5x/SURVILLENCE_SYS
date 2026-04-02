"""
Media Capture Manager
Handles screenshot and video clip recording on alerts
"""

import cv2
import os
import numpy as np
import threading
import time
from datetime import datetime
from collections import deque
from pathlib import Path
from typing import Optional, Dict, List
import sqlite3


class MediaManager:
    """
    Manages screenshot capture and event video recording
    """
    
    def __init__(self, buffer_seconds: int = 5, fps: int = 30, output_dir: str = "media_output"):
        self.buffer_seconds = buffer_seconds
        self.fps = fps
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "screenshots").mkdir(exist_ok=True)
        (self.output_dir / "clips").mkdir(exist_ok=True)
        
        # Circular buffer for pre-event frames
        self.frame_buffer = deque(maxlen=buffer_seconds * fps)
        self.recording_events: Dict[str, dict] = {}
        self._lock = threading.Lock()
        
        print(f"📸 MediaManager initialized")
        print(f"   Buffer: {buffer_seconds}s, FPS: {fps}")
        print(f"   Output: {self.output_dir.absolute()}")
    
    def add_frame(self, frame):
        """Continuously add frames to buffer"""
        with self._lock:
            self.frame_buffer.append(frame.copy())
            
            # Also add to any active recordings
            for event_id, event_data in self.recording_events.items():
                if event_data['recording']:
                    event_data['post_frames'].append(frame.copy())
    
    def capture_screenshot(self, frame: np.ndarray, alert_data: dict) -> str:
        """
        Capture and save screenshot when alert occurs
        Returns: path to saved screenshot
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{alert_data['anomaly_type']}_track{alert_data['track_id']}_{timestamp}.jpg"
        filepath = self.output_dir / "screenshots" / filename
        
        # Draw alert info on frame before saving
        annotated_frame = self._annotate_frame(frame.copy(), alert_data)
        
        cv2.imwrite(str(filepath), annotated_frame)
        print(f"   📸 Screenshot saved: {filepath.name}")
        
        return str(filepath)
    
    def start_event_recording(self, alert_data: dict, duration: int = 10) -> str:
        """
        Start recording video clip when alert triggers
        Returns: event_id to track this recording
        """
        event_id = f"{alert_data['track_id']}_{alert_data['anomaly_type']}_{time.time()}"
        
        with self._lock:
            self.recording_events[event_id] = {
                'alert_data': alert_data,
                'recording': True,
                'post_frames': [],
                'start_time': time.time(),
                'duration': duration
            }
        
        # Schedule video finalization
        timer = threading.Timer(duration, self._finalize_recording, args=[event_id])
        timer.daemon = True
        timer.start()
        
        print(f"   🎬 Started recording clip: {event_id[:20]}...")
        return event_id
    
    def _finalize_recording(self, event_id: str):
        """Save buffered frames + post-event frames as video"""
        with self._lock:
            if event_id not in self.recording_events:
                return None
            
            event_data = self.recording_events[event_id]
            event_data['recording'] = False
            
            # Combine pre-event buffer + post-event frames
            all_frames = list(self.frame_buffer) + event_data['post_frames']
            
            if not all_frames:
                del self.recording_events[event_id]
                return None
            
            # Generate filename
            alert = event_data['alert_data']
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"clip_{alert['anomaly_type']}_track{alert['track_id']}_{timestamp}.mp4"
            filepath = self.output_dir / "clips" / filename
            
            # Get frame dimensions
            height, width = all_frames[0].shape[:2]
            
            # Use XVID codec for better compatibility
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(filepath), fourcc, self.fps, (width, height))
            
            if out.isOpened():
                for frame in all_frames:
                    out.write(frame)
                out.release()
                print(f"   ✅ Video clip saved: {filepath.name} ({len(all_frames)} frames)")
                result_path = str(filepath)
            else:
                print(f"   ❌ Failed to create video writer")
                result_path = None
            
            # Cleanup
            del self.recording_events[event_id]
            return result_path
    
    def _annotate_frame(self, frame: np.ndarray, alert_data: dict) -> np.ndarray:
        """Draw alert information on frame"""
        height, width = frame.shape[:2]
        
        # Add semi-transparent overlay at bottom
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, height - 80), (width, height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        cv2.putText(frame, f"ALERT: {alert_data['anomaly_type']}", (10, height - 50),
                   font, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, f"Track: {alert_data['track_id']} | Zone: {alert_data.get('zone', 'N/A')}", 
                   (10, height - 30), font, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, timestamp, (10, height - 10), font, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def get_recent_captures(self, limit: int = 10) -> Dict[str, List[str]]:
        """Get list of recent screenshots and clips"""
        screenshots = sorted((self.output_dir / "screenshots").glob("*.jpg"))[-limit:]
        clips = sorted((self.output_dir / "clips").glob("*.mp4"))[-limit:]
        
        return {
            'screenshots': [str(s) for s in screenshots],
            'clips': [str(c) for c in clips]
        }
    
    def cleanup_old_media(self, days: int = 7):
        """Remove media files older than specified days"""
        cutoff = time.time() - (days * 24 * 60 * 60)
        
        for folder in ['screenshots', 'clips']:
            folder_path = self.output_dir / folder
            if folder_path.exists():
                for file_path in folder_path.glob('*'):
                    if file_path.stat().st_mtime < cutoff:
                        file_path.unlink()
                        print(f"   🗑️ Cleaned up: {file_path.name}")


# Global instance
MEDIA_MANAGER = None

def get_media_manager() -> MediaManager:
    """Get or create global media manager instance"""
    global MEDIA_MANAGER
    if MEDIA_MANAGER is None:
        MEDIA_MANAGER = MediaManager()
    return MEDIA_MANAGER
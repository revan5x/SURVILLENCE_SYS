"""
Visualization Utilities
Privacy-compliant rendering
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from tracking.bytetrack import Track
from logic.behavior_analyzer import BehaviorProfile, BehaviorType


class Visualizer:
    """
    Render tracks, zones, and analytics
    Respects privacy settings (always-on blurring)
    """
    
    def __init__(self, privacy_mode: bool = True):
        self.privacy_mode = privacy_mode
        self.colors = self._generate_colors()
    
    def _generate_colors(self) -> Dict[int, Tuple[int, int, int]]:
        """Generate distinct colors for track IDs"""
        np.random.seed(42)
        colors = {}
        for i in range(1, 101):
            colors[i] = tuple(map(int, np.random.randint(0, 255, 3)))
        return colors
    
    def draw_tracks(self, 
                   frame: np.ndarray,
                   tracks: List[Track],
                   profiles: Dict[int, BehaviorProfile]) -> np.ndarray:
        """Draw tracking visualization"""
        result = frame.copy()
        
        for track in tracks:
            tid = track.track_id
            color = self.colors.get(tid % 100, (0, 255, 0))
            
            # Get last observation
            last_pos = track.buffer.get_last_position()
            if last_pos is None:
                continue
            
            # Draw trajectory
            trajectory = track.buffer.get_trajectory()
            if len(trajectory) > 1:
                points = np.array(trajectory, np.int32)
                points = points.reshape((-1, 1, 2))
                cv2.polylines(result, [points], False, color, 2)
            
            # Draw current position
            cv2.circle(result, last_pos, 5, color, -1)
            
            # Draw bounding box if available
            if track.buffer.observations:
                last_obs = track.buffer.observations[-1]
                x1, y1, x2, y2 = last_obs.bbox
                
                # Draw ID label
                label = f"ID:{tid}"
                
                # Add behavior indicator
                if tid in profiles:
                    profile = profiles[tid]
                    if profile.behavior_type != BehaviorType.NORMAL:
                        label += f" [{profile.behavior_type.value[:4]}]"
                        color = self._get_behavior_color(profile.behavior_type)
                        cv2.rectangle(result, (x1, y1), (x2, y2), color, 3)
                
                # Draw label background
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(result, (x1, y1-th-10), (x1+tw, y1), color, -1)
                cv2.putText(result, label, (x1, y1-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return result
    
    def _get_behavior_color(self, behavior: BehaviorType) -> Tuple[int, int, int]:
        """Get color for behavior type"""
        colors = {
            BehaviorType.NORMAL: (0, 255, 0),
            BehaviorType.LOITERING: (0, 165, 255),  # Orange
            BehaviorType.SUSPICIOUS_MOVEMENT: (0, 0, 255),  # Red
            BehaviorType.WRONG_DIRECTION: (255, 0, 255),  # Magenta
            BehaviorType.RAPID_MOVEMENT: (0, 255, 255)  # Yellow
        }
        return colors.get(behavior, (128, 128, 128))
    
    def draw_hud(self,
                frame: np.ndarray,
                fps: float,
                cpu_percent: float,
                memory_mb: float,
                active_tracks: int,
                inference_time: float,
                detection_latency: float) -> np.ndarray:
        """Draw performance HUD"""
        result = frame.copy()
        h, w = result.shape[:2]
        
        # Semi-transparent background
        overlay = result.copy()
        cv2.rectangle(overlay, (10, 10), (350, 140), (0, 0, 0), -1)
        result = cv2.addWeighted(result, 1.0, overlay, 0.7, 0)
        
        # Draw metrics
        metrics = [
            f"FPS: {fps:.1f}",
            f"CPU: {cpu_percent:.1f}%",
            f"Memory: {memory_mb:.1f} MB",
            f"Tracks: {active_tracks}",
            f"Inference: {inference_time:.1f}ms",
            f"Latency: {detection_latency:.1f}ms"
        ]
        
        y_offset = 35
        for metric in metrics:
            cv2.putText(result, metric, (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 18
        
        return result
    
    def draw_events_panel(self,
                         frame: np.ndarray,
                         recent_events: List[Dict]) -> np.ndarray:
        """Draw recent events overlay"""
        result = frame.copy()
        h, w = result.shape[:2]
        
        # Panel on right side
        panel_width = 300
        overlay = result.copy()
        cv2.rectangle(overlay, (w-panel_width, 0), (w, h), (0, 0, 0), -1)
        result = cv2.addWeighted(result, 1.0, overlay, 0.8, 0)
        
        # Title
        cv2.putText(result, "RECENT EVENTS", (w-panel_width+10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Events
        y_offset = 60
        for event in recent_events[:8]:  # Show last 8
            time_str = event['timestamp'][:19] if isinstance(event['timestamp'], str) else "Now"
            text = f"{time_str[-8:]} | ID:{event['track_id']} | {event['anomaly_type'][:15]}"
            color = (0, 0, 255) if event['severity'] in ['HIGH', 'CRITICAL'] else (0, 165, 255)
            
            cv2.putText(result, text, (w-panel_width+10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 25
        
        return result
    
    def apply_privacy_blur(self, 
                          frame: np.ndarray,
                          detections: List) -> np.ndarray:
        """Apply face blurring (always-on)"""
        result = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            
            # Face region: top 25% of person bounding box
            face_height = int((y2 - y1) * 0.25)
            face_y2 = y1 + face_height
            
            if face_y2 > y1 + 5:
                face_roi = result[y1:face_y2, x1:x2]
                if face_roi.size > 0:
                    # Strong blur
                    blurred = cv2.GaussianBlur(face_roi, (51, 51), 50)
                    result[y1:face_y2, x1:x2] = blurred
        
        return result
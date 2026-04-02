"""
Behavioral Analysis Engine - CORRECTED
Fixed stationary detection and zone violation logic
"""

import sys
import os
import time
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tracking.track_buffer import TrackBuffer, TrackState


class BehaviorType(Enum):
    NORMAL = "normal"
    LOITERING = "loitering"
    SUSPICIOUS_MOVEMENT = "suspicious_movement"
    WRONG_DIRECTION = "wrong_direction"
    RAPID_MOVEMENT = "rapid_movement"


@dataclass
class BehaviorProfile:
    track_id: int
    behavior_type: BehaviorType
    confidence: float
    velocity: float
    direction: Optional[str]
    stationary_duration: float
    zone_violations: List[Tuple[str, str]]
    timestamp: float
    current_position: Optional[Tuple[int, int]]


class BehaviorAnalyzer:
    """
    CORRECTED: Proper stationary detection and zone checking
    """
    
    def __init__(self,
                 velocity_window: int = 5,
                 stationary_threshold: float = 5.0,
                 loitering_time: float = 2.0,
                 rapid_threshold: float = 150.0):
        
        self.velocity_window = velocity_window
        self.stationary_threshold = stationary_threshold
        self.loitering_time = loitering_time
        self.rapid_threshold = rapid_threshold
        self.track_zone_history: Dict[int, List[Tuple[str, float]]] = {}
    
    def analyze(self, 
                track,
                zone_manager=None) -> BehaviorProfile:
        """
        FIXED: Proper stationary detection and zone violation checking
        """
        buffer = track.buffer
        track_id = track.track_id
        
        # Get current position
        current_pos = buffer.get_last_position()
        if current_pos is None:
            current_pos = (0, 0)
        
        # Compute kinematics
        vx, vy = buffer.get_velocity(window=self.velocity_window)
        velocity = np.sqrt(vx**2 + vy**2)
        direction = self._calculate_direction(vx, vy)
        
        # FIXED: Proper stationary detection
        is_stationary, stationary_duration = self._check_stationary_fixed(buffer)
        
        # ZONE ANALYSIS - FIXED
        zone_violations = []
        current_zone = None
        
        if zone_manager and current_pos:
            # Check each zone
            for zone_name, zone_config in zone_manager.zones.items():
                if self._point_in_polygon(current_pos, zone_config.polygon):
                    current_zone = zone_name
                    
                    # Check restricted zone
                    if zone_config.is_restricted:
                        zone_violations.append((zone_name, 'RESTRICTED_ZONE_ENTRY'))
                        print(f"  [DEBUG] Restricted zone violation: {zone_name}")
                    
                    # Check direction violation
                    if direction and direction not in zone_config.allowed_directions:
                        zone_violations.append((zone_name, 'WRONG_DIRECTION'))
                        print(f"  [DEBUG] Direction violation: {direction} not in {zone_config.allowed_directions}")
                    
                    # Check dwell time
                    dwell = buffer.get_dwell_time()
                    if dwell > zone_config.max_dwell_time:
                        zone_violations.append((zone_name, 'EXCESSIVE_DWELL'))
                        print(f"  [DEBUG] Excessive dwell: {dwell:.1f}s > {zone_config.max_dwell_time}s")
                    
                    break  # Only check first matching zone
        
        # Determine behavior type - PRIORITY ORDER
        behavior_type = BehaviorType.NORMAL
        confidence = 0.0
        
        # Priority 1: Zone violations
        if zone_violations:
            if any('RESTRICTED' in v[1] for v in zone_violations):
                behavior_type = BehaviorType.SUSPICIOUS_MOVEMENT
                confidence = 0.9
            elif any('WRONG_DIRECTION' in v[1] for v in zone_violations):
                behavior_type = BehaviorType.WRONG_DIRECTION
                confidence = 0.85
            else:
                behavior_type = BehaviorType.SUSPICIOUS_MOVEMENT
                confidence = 0.7
        
        # Priority 2: Loitering - FIXED THRESHOLD CHECK
        elif is_stationary and stationary_duration >= self.loitering_time:
            behavior_type = BehaviorType.LOITERING
            confidence = min(0.95, 0.6 + (stationary_duration / 10.0))
            print(f"  [DEBUG] Loitering detected: {stationary_duration:.2f}s stationary")
        
        # Priority 3: Rapid movement
        elif velocity > self.rapid_threshold:
            behavior_type = BehaviorType.RAPID_MOVEMENT
            confidence = min(0.9, velocity / 400.0)
        
        # Create profile
        profile = BehaviorProfile(
            track_id=track_id,
            behavior_type=behavior_type,
            confidence=confidence,
            velocity=velocity,
            direction=direction,
            stationary_duration=stationary_duration if is_stationary else 0.0,
            zone_violations=zone_violations,
            timestamp=time.time(),
            current_position=current_pos
        )
        
        return profile
    
    def _calculate_direction(self, vx: float, vy: float) -> Optional[str]:
        """Calculate cardinal direction from velocity"""
        speed = np.sqrt(vx**2 + vy**2)
        
        if speed < 10:  # pixels/second threshold
            return None
        
        if abs(vx) > abs(vy):
            return 'E' if vx > 0 else 'W'
        else:
            return 'S' if vy > 0 else 'N'
    
    def _check_stationary_fixed(self, buffer: TrackBuffer) -> Tuple[bool, float]:
        """
        FIXED: Proper stationary detection
        Returns: (is_stationary, stationary_duration)
        """
        if len(buffer.observations) < 2:
            return False, 0.0
        
        # Get time window for loitering check
        latest_time = buffer.observations[-1].timestamp
        earliest_time = buffer.observations[0].timestamp
        
        # Check if track has existed long enough
        total_dwell = latest_time - earliest_time
        
        if total_dwell < self.loitering_time:
            return False, total_dwell
        
        # Check spatial movement in the entire history
        positions = [obs.position for obs in buffer.observations]
        
        if len(positions) < 2:
            return False, total_dwell
        
        # Calculate max movement
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        
        x_range = max(xs) - min(xs)
        y_range = max(ys) - min(ys)
        
        # Check if movement is within threshold
        is_stationary = x_range <= self.stationary_threshold and y_range <= self.stationary_threshold
        
        if is_stationary:
            return True, total_dwell
        else:
            return False, total_dwell
    
    def _point_in_polygon(self, point: Tuple[int, int], polygon: List[Tuple[int, int]]) -> bool:
        """Ray casting algorithm for point-in-polygon test"""
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def cleanup_old_tracks(self, active_track_ids: List[int]):
        """Remove history for deleted tracks"""
        to_remove = [tid for tid in self.track_zone_history if tid not in active_track_ids]
        for tid in to_remove:
            del self.track_zone_history[tid]
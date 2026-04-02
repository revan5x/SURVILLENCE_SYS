"""
Temporal Buffer for Track History
30-frame buffer for behavior analysis
"""

from collections import deque
from typing import List, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from enum import Enum


class TrackState(Enum):
    NEW = "new"
    TRACKED = "tracked"
    LOST = "lost"
    REMOVED = "removed"


@dataclass
class TrackObservation:
    timestamp: float
    position: Tuple[int, int]
    bbox: Tuple[int, int, int, int]
    confidence: float


class TrackBuffer:
    """
    Circular buffer storing 30 frames of track history
    Enables velocity calculation and behavior analysis
    """
    
    def __init__(self, max_size: int = 30):
        self.max_size = max_size
        self.observations: deque = deque(maxlen=max_size)
        self.state = TrackState.NEW
        self.lost_count = 0
    
    def add(self, timestamp: float, 
            position: Tuple[int, int],
            bbox: Tuple[int, int, int, int],
            confidence: float):
        """Add new observation"""
        obs = TrackObservation(timestamp, position, bbox, confidence)
        self.observations.append(obs)
        self.state = TrackState.TRACKED
        self.lost_count = 0
    
    def mark_lost(self):
        """Increment lost counter"""
        self.lost_count += 1
        if self.lost_count > 30:  # Max lost frames
            self.state = TrackState.REMOVED
    
    def get_velocity(self, window: int = 5) -> Tuple[float, float]:
        """
        Calculate average velocity over recent observations
        Returns (vx, vy) in pixels/second
        """
        if len(self.observations) < 2:
            return 0.0, 0.0
        
        recent = list(self.observations)[-window:]
        if len(recent) < 2:
            recent = list(self.observations)
        
        velocities = []
        for i in range(1, len(recent)):
            dt = recent[i].timestamp - recent[i-1].timestamp
            if dt > 0:
                dx = recent[i].position[0] - recent[i-1].position[0]
                dy = recent[i].position[1] - recent[i-1].position[1]
                velocities.append((dx/dt, dy/dt))
        
        if not velocities:
            return 0.0, 0.0
        
        avg_vx = sum(v[0] for v in velocities) / len(velocities)
        avg_vy = sum(v[1] for v in velocities) / len(velocities)
        
        return avg_vx, avg_vy
    
    def get_speed(self) -> float:
        """Get scalar speed"""
        vx, vy = self.get_velocity()
        return np.sqrt(vx**2 + vy**2)
    
    def get_direction(self) -> Optional[str]:
        """
        Get cardinal direction of movement
        Returns: 'N', 'S', 'E', 'W', or None if stationary
        """
        vx, vy = self.get_velocity()
        speed = np.sqrt(vx**2 + vy**2)
        
        if speed < 5:  # Stationary threshold
            return None
        
        # Determine primary direction
        if abs(vx) > abs(vy):
            return 'E' if vx > 0 else 'W'
        else:
            return 'S' if vy > 0 else 'N'
    
    def is_stationary(self, threshold: float = 5.0, duration: float = 2.0) -> bool:
        """
        Check if track has been stationary
        threshold: pixel movement threshold
        duration: minimum time in seconds
        """
        if len(self.observations) < 2:
            return False
        
        # Check movement over duration window
        now = self.observations[-1].timestamp
        cutoff = now - duration
        
        recent = [obs for obs in self.observations if obs.timestamp >= cutoff]
        if len(recent) < 2:
            return False
        
        positions = [obs.position for obs in recent]
        max_dist = 0
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                dist = np.sqrt(
                    (positions[i][0] - positions[j][0])**2 + 
                    (positions[i][1] - positions[j][1])**2
                )
                max_dist = max(max_dist, dist)
        
        return max_dist < threshold
    
    def get_dwell_time(self) -> float:
        """Get total time span of track"""
        if len(self.observations) < 2:
            return 0.0
        return self.observations[-1].timestamp - self.observations[0].timestamp
    
    def get_last_position(self) -> Optional[Tuple[int, int]]:
        """Get most recent position"""
        if not self.observations:
            return None
        return self.observations[-1].position
    
    def get_trajectory(self) -> List[Tuple[int, int]]:
        """Get full position history"""
        return [obs.position for obs in self.observations]
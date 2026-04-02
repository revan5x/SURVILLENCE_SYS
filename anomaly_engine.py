"""
Anomaly Detection Engine - CORRECTED VERSION
Fixed persistence logic and event generation
"""

import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from logic.behavior_analyzer import BehaviorProfile, BehaviorType


class Severity(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class AnomalyEvent:
    timestamp: float
    track_id: int
    anomaly_type: str
    severity: Severity
    zone: Optional[str]
    description: str
    velocity: float
    duration: float


class AnomalyEngine:
    """
    FIXED: Proper 2-second persistence rule implementation
    """
    
    def __init__(self,
                 persistence_time: float = 2.0,
                 alert_cooldown: float = 30.0):
        
        self.persistence_time = persistence_time
        self.alert_cooldown = alert_cooldown
        
        # Track when anomaly started for each track
        self.anomaly_start_time: Dict[int, Tuple[BehaviorType, float]] = {}
        
        # Track which anomalies have been confirmed (to prevent duplicates)
        self.confirmed_anomalies: Dict[int, BehaviorType] = {}
        
        # Alert cooldown tracking
        self.last_alert_time: Dict[str, float] = {}
    
    def evaluate(self, profile: BehaviorProfile) -> Optional[AnomalyEvent]:
        """
        FIXED: Proper state machine for anomaly detection
        """
        current_time = time.time()
        track_id = profile.track_id
        
        # CASE 1: Normal behavior - clear all states
        if profile.behavior_type == BehaviorType.NORMAL:
            if track_id in self.anomaly_start_time:
                del self.anomaly_start_time[track_id]
            if track_id in self.confirmed_anomalies:
                del self.confirmed_anomalies[track_id]
            return None
        
        # CASE 2: Check if we already confirmed this exact anomaly
        if track_id in self.confirmed_anomalies:
            if self.confirmed_anomalies[track_id] == profile.behavior_type:
                # Already confirmed and still ongoing, don't spam
                return None
            else:
                # Behavior type changed, treat as new anomaly
                del self.confirmed_anomalies[track_id]
        
        # CASE 3: New or ongoing anomaly detection
        if track_id not in self.anomaly_start_time:
            # First time seeing this anomaly - start timer
            self.anomaly_start_time[track_id] = (profile.behavior_type, current_time)
            return None
        
        # Get stored info
        stored_type, start_time = self.anomaly_start_time[track_id]
        
        # CASE 4: Behavior changed during pending period
        if stored_type != profile.behavior_type:
            # Reset timer with new type
            self.anomaly_start_time[track_id] = (profile.behavior_type, current_time)
            return None
        
        # CASE 5: Check if persisted long enough
        elapsed = current_time - start_time
        
        if elapsed >= self.persistence_time:
            # CONFIRMED ANOMALY - Generate event
            event = self._create_event(profile, elapsed, current_time)
            
            # Mark as confirmed so we don't generate again
            self.confirmed_anomalies[track_id] = profile.behavior_type
            
            # Keep the start time for duration tracking, but we won't generate
            # another event until behavior changes
            
            return event
        
        # Still waiting for persistence
        return None
    
    def _create_event(self, 
                     profile: BehaviorProfile, 
                     duration: float,
                     current_time: float) -> AnomalyEvent:
        """Create anomaly event with severity calculation"""
        
        # Determine severity
        severity = self._calculate_severity(profile, duration)
        
        # Build zone info
        zone_str = None
        if profile.zone_violations:
            zone_str = profile.zone_violations[0][0]
            violation_detail = profile.zone_violations[0][1]
        else:
            violation_detail = "General Area"
        
        # Build description
        description = f"{profile.behavior_type.value.replace('_', ' ').title()}"
        if zone_str:
            description += f" in {zone_str}"
        if profile.zone_violations:
            description += f" ({violation_detail})"
        
        # Apply cooldown check
        alert_key = f"{profile.track_id}_{profile.behavior_type.value}"
        
        if alert_key in self.last_alert_time:
            time_since_last = current_time - self.last_alert_time[alert_key]
            if time_since_last < self.alert_cooldown:
                # In cooldown - reduce severity
                severity = Severity.LOW
        
        # Update last alert time
        self.last_alert_time[alert_key] = current_time
        
        return AnomalyEvent(
            timestamp=current_time,
            track_id=profile.track_id,
            anomaly_type=profile.behavior_type.value,
            severity=severity,
            zone=zone_str,
            description=description,
            velocity=profile.velocity,
            duration=duration
        )
    
    def _calculate_severity(self, 
                           profile: BehaviorProfile, 
                           duration: float) -> Severity:
        """Calculate severity score 0-100"""
        
        score = 0
        
        # Base behavior score
        if profile.behavior_type == BehaviorType.LOITERING:
            # More dangerous the longer it lasts
            score += min(40, duration * 15)
        elif profile.behavior_type == BehaviorType.RAPID_MOVEMENT:
            score += 30
        elif profile.behavior_type == BehaviorType.WRONG_DIRECTION:
            score += 35
        elif profile.behavior_type == BehaviorType.SUSPICIOUS_MOVEMENT:
            score += 25
        
        # Zone violations add significant score
        if profile.zone_violations:
            for zone_name, violation_type in profile.zone_violations:
                if 'RESTRICTED' in violation_type:
                    score += 45  # Major violation
                elif 'WRONG_DIRECTION' in violation_type:
                    score += 25
                elif 'EXCESSIVE_DWELL' in violation_type:
                    score += 20
        
        # High velocity in restricted zone is worse
        if profile.velocity > 250 and profile.zone_violations:
            score += 15
        
        # Confidence factor
        score *= (0.5 + 0.5 * profile.confidence)
        
        # Map to severity levels
        if score >= 75:
            return Severity.CRITICAL
        elif score >= 55:
            return Severity.HIGH
        elif score >= 35:
            return Severity.MEDIUM
        else:
            return Severity.LOW
    
    def reset_track(self, track_id: int):
        """Reset state when track is lost"""
        if track_id in self.anomaly_start_time:
            del self.anomaly_start_time[track_id]
        if track_id in self.confirmed_anomalies:
            del self.confirmed_anomalies[track_id]
    
    def cleanup_old_tracks(self, active_track_ids: List[int]):
        """Remove state for old tracks"""
        all_track_ids = list(self.anomaly_start_time.keys()) + list(self.confirmed_anomalies.keys())
        to_remove = set(all_track_ids) - set(active_track_ids)
        
        for track_id in to_remove:
            self.reset_track(track_id)
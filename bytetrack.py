"""
ByteTrack Implementation
Multi-object tracking with Kalman filtering
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.optimize import linear_sum_assignment
import time

# Fix: Use absolute imports to avoid circular dependency
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tracking.kalman_filter import TrackKalmanFilter
from tracking.track_buffer import TrackBuffer, TrackState
from vision.detector import Detection


class Track:
    """Single track instance"""
    _id_counter = 0
    
    def __init__(self, detection: Detection, timestamp: float):
        Track._id_counter += 1
        self.track_id = Track._id_counter
        
        self.kf = TrackKalmanFilter()
        self.buffer = TrackBuffer(max_size=30)
        
        # Initialize
        cx, cy = detection.center
        self.kf.init_state(cx, cy)
        self.buffer.add(timestamp, detection.center, detection.bbox, detection.confidence)
        
        self.start_time = timestamp
        self.score = detection.confidence
    
    def predict(self):
        """Kalman prediction step"""
        return self.kf.predict()
    
    def update(self, detection: Detection, timestamp: float):
        """Update with new detection"""
        cx, cy = detection.center
        self.kf.update(cx, cy)
        self.buffer.add(timestamp, detection.center, detection.bbox, detection.confidence)
        self.score = detection.confidence
    
    def mark_lost(self):
        """Mark as lost"""
        self.buffer.mark_lost()
        if self.buffer.state == TrackState.REMOVED:
            return False  # Should be removed
        self.buffer.state = TrackState.LOST
        return True  # Still active
    
    def reactivate(self, detection: Detection, timestamp: float):
        """Reactivate lost track"""
        self.update(detection, timestamp)
        self.buffer.state = TrackState.TRACKED
    
    def get_state(self) -> TrackState:
        return self.buffer.state
    
    def is_confirmed(self) -> bool:
        """Check if track is confirmed (not new)"""
        return len(self.buffer.observations) >= 3 and self.buffer.state != TrackState.REMOVED


class ByteTracker:
    """
    ByteTrack: Multi-object tracking
    Handles detection-to-track association
    """
    
    def __init__(self,
                 high_thresh: float = 0.6,
                 low_thresh: float = 0.1,
                 new_track_thresh: float = 0.6,
                 match_thresh: float = 0.8,
                 max_lost: int = 30):
        
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.new_track_thresh = new_track_thresh
        self.match_thresh = match_thresh
        self.max_lost = max_lost
        
        self.tracks: List[Track] = []
        self.lost_tracks: List[Track] = []
        self.removed_tracks: List[Track] = []
        
        self.frame_count = 0
    
    def update(self, detections: List[Detection], timestamp: float) -> List[Track]:
        """
        Update tracks with new detections
        Returns: List of active confirmed tracks
        """
        self.frame_count += 1
        
        # Separate high/low confidence detections
        high_dets = [d for d in detections if d.confidence >= self.high_thresh]
        low_dets = [d for d in detections if self.low_thresh <= d.confidence < self.high_thresh]
        
        # Predict existing tracks
        for track in self.tracks:
            track.predict()
        
        # First association: High confidence detections with tracked tracks
        matched, unmatched_tracks, unmatched_dets = self._associate(
            self.tracks, high_dets, self.match_thresh
        )
        
        # Update matched tracks
        for track_idx, det_idx in matched:
            self.tracks[track_idx].update(high_dets[det_idx], timestamp)
        
        # Second association: Low confidence detections with unmatched tracks
        if unmatched_tracks and low_dets:
            remaining_tracks = [self.tracks[i] for i in unmatched_tracks]
            matched2, unmatched_tracks2, _ = self._associate(
                remaining_tracks, low_dets, self.match_thresh
            )
            
            for track_idx, det_idx in matched2:
                remaining_tracks[track_idx].update(low_dets[det_idx], timestamp)
            
            # Mark truly unmatched as lost
            for track in [remaining_tracks[i] for i in unmatched_tracks2]:
                if not track.mark_lost():
                    self.lost_tracks.append(track)
                    self.tracks.remove(track)
        else:
            # Mark all unmatched as lost
            for idx in sorted(unmatched_tracks, reverse=True):
                if not self.tracks[idx].mark_lost():
                    self.lost_tracks.append(self.tracks[idx])
                    self.tracks.pop(idx)
        
        # Third association: Unmatched high conf detections with lost tracks
        new_tracks = []
        for det_idx in unmatched_dets:
            det = high_dets[det_idx]
            
            # Try to match with lost tracks
            matched_lost = False
            for lost_track in self.lost_tracks[:]:  # Copy list to allow removal
                if self._iou(det.bbox, lost_track.buffer.get_last_position()) > 0.5:
                    lost_track.reactivate(det, timestamp)
                    self.tracks.append(lost_track)
                    self.lost_tracks.remove(lost_track)
                    matched_lost = True
                    break
            
            if not matched_lost and det.confidence >= self.new_track_thresh:
                # Create new track
                new_tracks.append(Track(det, timestamp))
        
        self.tracks.extend(new_tracks)
        
        # Clean up old lost tracks (older than 30 seconds)
        self.lost_tracks = [t for t in self.lost_tracks 
                          if timestamp - t.start_time < 30]
        
        # Return only confirmed, active tracks
        confirmed_tracks = []
        for track in self.tracks:
            if track.is_confirmed() and track.buffer.state in [TrackState.TRACKED, TrackState.LOST]:
                confirmed_tracks.append(track)
        
        return confirmed_tracks
    
    def _associate(self, 
                   tracks: List[Track], 
                   detections: List[Detection],
                   threshold: float) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Hungarian algorithm for optimal assignment
        Returns: (matched_pairs, unmatched_track_indices, unmatched_det_indices)
        """
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))
        
        # Compute cost matrix (1 - IoU)
        cost_matrix = np.zeros((len(tracks), len(detections)))
        for i, track in enumerate(tracks):
            pred_pos = track.predict()
            for j, det in enumerate(detections):
                cost = 1.0 - self._compute_similarity(track, det, pred_pos)
                cost_matrix[i, j] = cost
        
        # Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matched = []
        unmatched_tracks = []
        unmatched_dets = list(range(len(detections)))
        
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < 1 - threshold:
                matched.append((r, c))
                if c in unmatched_dets:
                    unmatched_dets.remove(c)
            else:
                unmatched_tracks.append(r)
        
        # Add unmatched tracks
        for i in range(len(tracks)):
            if i not in row_ind:
                unmatched_tracks.append(i)
        
        return matched, unmatched_tracks, unmatched_dets
    
    def _compute_similarity(self, 
                           track: Track, 
                           detection: Detection,
                           predicted_pos: Tuple[float, float]) -> float:
        """Compute similarity between track and detection"""
        # Position distance
        det_center = detection.center
        pos_dist = np.sqrt(
            (predicted_pos[0] - det_center[0])**2 + 
            (predicted_pos[1] - det_center[1])**2
        )
        
        # Normalize by detection size
        size = np.sqrt(detection.area)
        if size > 0:
            pos_dist /= size
        
        # Convert to similarity (closer = higher similarity)
        pos_sim = max(0, 1 - pos_dist / 2)
        
        # IoU component
        iou = self._iou_from_center(track.buffer.get_last_position(), det_center, detection.bbox)
        
        # Combined similarity
        return 0.7 * pos_sim + 0.3 * iou
    
    def _iou(self, pos1: Optional[Tuple[int, int]], pos2: Tuple[int, int]) -> float:
        """IoU between two points (approximated)"""
        if pos1 is None:
            return 0.0
        dist = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
        return max(0, 1 - dist / 100)  # Normalize
    
    def _iou_from_center(self, 
                         center: Optional[Tuple[int, int]], 
                         det_center: Tuple[int, int],
                         det_bbox: Tuple[int, int, int, int]) -> float:
        """Approximate IoU using center distance"""
        if center is None:
            return 0.0
        
        w = det_bbox[2] - det_bbox[0]
        h = det_bbox[3] - det_bbox[1]
        
        dx = abs(center[0] - det_center[0]) / w
        dy = abs(center[1] - det_center[1]) / h
        
        return max(0, 1 - (dx + dy) / 2)
    
    def get_active_tracks(self) -> List[Track]:
        """Get all confirmed active tracks"""
        return [t for t in self.tracks if t.is_confirmed()]
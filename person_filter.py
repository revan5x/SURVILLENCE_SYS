#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Person Filter - Ensures only person class is detected, filters out other objects
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any


class PersonFilter:
    """
    Filters detections to keep only person class (typically class 0 in COCO)
    and applies additional person-specific validation
    """
    
    PERSON_CLASS_ID = 0  # COCO dataset person class
    PERSON_CLASS_NAMES = ['person', 'human', 'man', 'woman', 'child', 'people']
    
    def __init__(self, 
                 min_confidence: float = 0.5,
                 min_height_ratio: float = 0.15,  # Min height relative to frame
                 max_height_ratio: float = 0.9,   # Max height relative to frame
                 aspect_ratio_range: Tuple[float, float] = (0.2, 5.0),
                 enable_pose_check: bool = True):
        
        self.min_confidence = min_confidence
        self.min_height_ratio = min_height_ratio
        self.max_height_ratio = max_height_ratio
        self.aspect_ratio_range = aspect_ratio_range
        self.enable_pose_check = enable_pose_check
        
        # Statistics
        self.total_detections = 0
        self.filtered_detections = 0
        self.person_detections = 0
    
    def filter_detections(self, 
                         detections: List[Dict], 
                         frame_shape: Tuple[int, ...]) -> List[Dict]:
        """
        Filter detections to keep only valid persons
        
        Args:
            detections: List of detection dicts with 'class_id', 'confidence', 'bbox'
            frame_shape: Shape of the frame (h, w, ...)
        
        Returns:
            Filtered list of person detections only
        """
        self.total_detections += len(detections)
        frame_h, frame_w = frame_shape[:2]
        
        filtered = []
        
        for det in detections:
            # Check 1: Must be person class
            if not self._is_person_class(det):
                continue
            
            # Check 2: Confidence threshold
            confidence = det.get('confidence', 0)
            if confidence < self.min_confidence:
                continue
            
            bbox = det.get('bbox', [0, 0, 0, 0])
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            
            # Check 3: Size constraints
            height_ratio = h / frame_h
            if height_ratio < self.min_height_ratio or height_ratio > self.max_height_ratio:
                continue
            
            # Check 4: Aspect ratio (persons are typically taller than wide)
            if w > 0:
                aspect_ratio = h / w
                if aspect_ratio < self.aspect_ratio_range[0] or aspect_ratio > self.aspect_ratio_range[1]:
                    continue
            
            # Check 5: Optional pose/validation check
            if self.enable_pose_check and not self._validate_person_shape(det, frame_shape):
                continue
            
            # Add metadata
            det['detection_type'] = 'person'
            det['validation_passed'] = True
            filtered.append(det)
        
        self.filtered_detections += len(filtered)
        self.person_detections += len(filtered)
        
        return filtered
    
    def _is_person_class(self, detection: Dict) -> bool:
        """Check if detection is a person class"""
        class_id = detection.get('class_id', -1)
        class_name = detection.get('class_name', '').lower()
        
        # Check by ID (COCO)
        if class_id == self.PERSON_CLASS_ID:
            return True
        
        # Check by name
        if any(name in class_name for name in self.PERSON_CLASS_NAMES):
            return True
        
        return False
    
    def _validate_person_shape(self, detection: Dict, frame_shape: Tuple) -> bool:
        """Additional validation based on person shape characteristics"""
        bbox = detection.get('bbox', [0, 0, 0, 0])
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        
        # Persons typically have height > width
        if h < w * 0.8:  # Too wide to be a standing person
            return False
        
        # Check if detection has keypoints (pose estimation)
        if 'keypoints' in detection:
            keypoints = detection['keypoints']
            # Ensure minimum keypoints are detected for a valid person
            valid_keypoints = sum(1 for kp in keypoints if kp[2] > 0.5)  # confidence > 0.5
            if valid_keypoints < 5:  # Need at least 5 keypoints
                return False
        
        return True
    
    def get_statistics(self) -> Dict[str, int]:
        """Get filtering statistics"""
        return {
            'total_processed': self.total_detections,
            'person_detections': self.person_detections,
            'filter_rate': (1 - self.person_detections / max(self.total_detections, 1)) * 100
        }
    
    def reset_statistics(self):
        """Reset statistics counters"""
        self.total_detections = 0
        self.filtered_detections = 0
        self.person_detections = 0


class DetectionAggregator:
    """
    Aggregates detections over multiple frames to reduce false positives
    """
    
    def __init__(self, history_size: int = 5, min_hits: int = 3):
        self.history_size = history_size
        self.min_hits = min_hits
        self.detection_history = []
        self.next_id = 0
    
    def update(self, detections: List[Dict]) -> List[Dict]:
        """Update with new detections and return confirmed detections"""
        self.detection_history.append(detections)
        if len(self.detection_history) > self.history_size:
            self.detection_history.pop(0)
        
        # Simple temporal filtering - could be enhanced with IoU matching
        return detections  # Pass through for now, implement temporal consistency if needed
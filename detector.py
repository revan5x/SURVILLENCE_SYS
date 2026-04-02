"""
Vision Layer
YOLOv8-Nano CPU inference, person detection only
"""

import cv2
import numpy as np
import time
from typing import List, Tuple, Optional
from ultralytics import YOLO


class Detection:
    """Detection data structure"""
    def __init__(self, 
                 bbox: Tuple[int, int, int, int],  # x1, y1, x2, y2
                 confidence: float,
                 class_id: int = 0):
        self.bbox = bbox
        self.confidence = confidence
        self.class_id = class_id
        self.center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
        self.area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


class PersonDetector:
    """
    YOLOv8-Nano detector
    CPU-only, inference-only, person class only
    """
    
    def __init__(self,
                 confidence_threshold: float = 0.45,
                 iou_threshold: float = 0.5):
        
        self.conf_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
        # Load model
        self.model = YOLO('yolov8n.pt', verbose=False)
        self.model.fuse()  # Optimize for inference
        
        # Force CPU
        self.model.to('cpu')
        
        # Warm up
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        self.model.predict(dummy, verbose=False)
        
        self._inference_times = []
        self._max_history = 30
    
    def detect(self, frame: np.ndarray) -> Tuple[List[Detection], float]:
        """
        Run detection on frame
        Returns: (detections, inference_time_ms)
        """
        start_time = time.perf_counter()
        
        # Run inference
        results = self.model.predict(
            frame,
            classes=[0],  # Person only
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
            device='cpu'
        )[0]
        
        inference_time = (time.perf_counter() - start_time) * 1000
        
        # Parse detections
        detections = []
        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                detections.append(Detection((x1, y1, x2, y2), conf))
        
        # Track performance
        self._inference_times.append(inference_time)
        if len(self._inference_times) > self._max_history:
            self._inference_times.pop(0)
        
        return detections, inference_time
    
    def get_avg_inference_time(self) -> float:
        """Rolling average inference time"""
        if not self._inference_times:
            return 0.0
        return sum(self._inference_times) / len(self._inference_times)
    
    def blur_faces(self, 
                   frame: np.ndarray, 
                   detections: List[Detection]) -> np.ndarray:
        """
        Privacy-preserving face blurring
        Always-on, no toggle in processing
        """
        result = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            
            # Estimate face region (top 30% of bounding box)
            face_height = int((y2 - y1) * 0.3)
            face_y2 = y1 + face_height
            
            # Ensure valid coordinates
            face_y2 = min(face_y2, y2)
            if face_y2 > y1 + 10:  # Minimum face size
                face_roi = result[y1:face_y2, x1:x2]
                
                # Apply strong Gaussian blur
                if face_roi.size > 0:
                    blurred = cv2.GaussianBlur(face_roi, (35, 35), 30)
                    result[y1:face_y2, x1:x2] = blurred
        
        return result
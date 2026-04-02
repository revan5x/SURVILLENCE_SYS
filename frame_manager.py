"""
Frame Ingestion Layer
CPU-optimized capture with frame skipping
"""

import cv2
import time
import threading
import numpy as np
from typing import Optional, Tuple


class FrameManager:
    """
    Thread-safe frame capture with frame skipping
    """
    
    def __init__(self, 
                 source: int = 0,
                 width: int = 640,
                 height: int = 480,
                 fps: int = 30,
                 frame_skip: int = 2):
        
        self.source = source
        self.width = width
        self.height = height
        self.target_fps = fps
        self.frame_skip = frame_skip
        
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_count = 0
        self._last_frame: Optional[np.ndarray] = None
        self._timestamp: float = 0
        
        # Thread safety
        self._lock = threading.Lock()
        self._running = False
        self._capture_thread: Optional[threading.Thread] = None
        
    def start(self) -> bool:
        """Initialize video capture"""
        print(f"📷 Starting camera (source={self.source})...")
        
        self._cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)  # Use DirectShow on Windows
        
        if not self._cap.isOpened():
            # Try alternative source
            self._cap = cv2.VideoCapture(self.source)
            if not self._cap.isOpened():
                print("❌ Failed to open camera")
                return False
        
        # Configure capture
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        
        # Warm up - read a few frames
        for i in range(10):
            ret, frame = self._cap.read()
            if ret and frame is not None:
                self._last_frame = frame
                self._timestamp = time.time()
                print(f"✅ Camera warmed up (frame {i+1})")
                break
            time.sleep(0.1)
        
        if self._last_frame is None:
            print("❌ Camera warm-up failed")
            return False
        
        self._running = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()
        
        print("✅ Camera started successfully")
        return True
    
    def _capture_loop(self):
        """Background capture thread"""
        frame_interval = 1.0 / self.target_fps
        
        while self._running:
            loop_start = time.time()
            
            if self._cap is None or not self._cap.isOpened():
                break
            
            ret, frame = self._cap.read()
            if ret and frame is not None:
                with self._lock:
                    self._last_frame = frame.copy()
                    self._timestamp = time.time()
                    self._frame_count += 1
            
            # Maintain target FPS
            elapsed = time.time() - loop_start
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
    
    def get_frame(self) -> Tuple[Optional[np.ndarray], float, bool]:
        """
        Get frame with skip logic
        Returns: (frame, timestamp, should_process)
        """
        with self._lock:
            if self._last_frame is None:
                return None, 0.0, False
            
            # Always return a copy to avoid corruption
            frame = self._last_frame.copy()
            timestamp = self._timestamp
            count = self._frame_count
        
        # Determine if this frame should be processed
        should_process = (count % self.frame_skip) == 0
        
        return frame, timestamp, should_process
    
    def release(self):
        """Clean shutdown"""
        print("📷 Releasing camera...")
        self._running = False
        
        if self._capture_thread:
            self._capture_thread.join(timeout=1.0)
        
        if self._cap:
            self._cap.release()
        
        self._last_frame = None
        print("✅ Camera released")
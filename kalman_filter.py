"""
Kalman Filter for Motion Prediction
Handles occlusion and smooth tracking
"""

import numpy as np
from typing import Optional, Tuple
from filterpy.kalman import KalmanFilter


class TrackKalmanFilter:
    """
    Kalman filter for 2D motion prediction
    State: [x, y, vx, vy]
    Measurement: [x, y]
    """
    
    def __init__(self):
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        
        # State transition matrix (constant velocity model)
        self.kf.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Measurement matrix (observe position only)
        self.kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Measurement noise
        self.kf.R *= 10
        
        # Process noise
        self.kf.Q[2:, 2:] *= 0.01
        
        # Initial uncertainty
        self.kf.P *= 100
        
        self._initialized = False
    
    def init_state(self, x: float, y: float):
        """Initialize filter with first observation"""
        self.kf.x = np.array([x, y, 0., 0.])
        self._initialized = True
    
    def predict(self) -> Tuple[float, float]:
        """Predict next state"""
        if not self._initialized:
            return 0.0, 0.0
        
        self.kf.predict()
        return self.kf.x[0], self.kf.x[1]
    
    def update(self, x: float, y: float):
        """Update with measurement"""
        if not self._initialized:
            self.init_state(x, y)
            return
        
        self.kf.update(np.array([x, y]))
    
    def get_velocity(self) -> Tuple[float, float]:
        """Get estimated velocity"""
        return self.kf.x[2], self.kf.x[3]
    
    def get_uncertainty(self) -> float:
        """Get position uncertainty (trace of covariance)"""
        return np.trace(self.kf.P[:2, :2])
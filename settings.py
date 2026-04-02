"""
System Configuration - Single Source of Truth
Frozen Architecture Parameters - DO NOT MODIFY
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class SystemConfig:
    """Immutable system configuration"""
    
    
    # Video Settings
    FRAME_WIDTH: int = 640
    FRAME_HEIGHT: int = 480
    FPS_TARGET: int = 30
    FRAME_SKIP: int = 2  # Process every Nth frame for CPU optimization
    
    # Detection Settings
    DETECTION_CONFIDENCE: float = 0.45
    IOU_THRESHOLD: float = 0.5
    PERSON_CLASS_ID: int = 0  # COCO class for person
    
    # Tracking Settings
    TRACK_BUFFER_SIZE: int = 30  # 30-frame temporal buffer
    TRACK_HIGH_THRESH: float = 0.6
    TRACK_LOW_THRESH: float = 0.1
    NEW_TRACK_THRESH: float = 0.6
    MATCH_THRESH: float = 0.8
    MAX_LOST_FRAMES: int = 30
    
    # Behavior Analysis
    VELOCITY_WINDOW: int = 5  # Frames for velocity calculation
    STATIONARY_THRESHOLD: float = 5.0  # Pixels movement
    LOITERING_TIME: float = 2.0  # Seconds
    ANOMALY_PERSISTENCE: float = 2.0  # Seconds before alert
    DIRECTION_THRESHOLD: float = 0.7  # Consistency ratio
    
    # Alert Settings
    ALERT_COOLDOWN: float = 30.0  # Seconds between similar alerts
    SMTP_TIMEOUT: int = 10
    
    # Privacy
    FACE_BLUR_KERNEL: int = 35  # Must be odd
    PRIVACY_MODE: bool = True  # Always-on blurring
    
    # Performance
    MAX_CPU_PERCENT: float = 80.0


@dataclass
class ZoneConfig:
    """Dynamic zone configuration"""
    name: str
    polygon: List[Tuple[int, int]]  # List of (x, y) points
    is_restricted: bool = False
    allowed_directions: List[str] = None  # ['N', 'S', 'E', 'W']
    max_dwell_time: float = 60.0  # Seconds
    
    def __post_init__(self):
        if self.allowed_directions is None:
            self.allowed_directions = ['N', 'S', 'E', 'W']


# ============================================
# MEDIA CAPTURE CONFIGURATION - ADD THIS
# ============================================
@dataclass(frozen=True)
class MediaConfig:
    """Media capture settings"""
    ENABLED: bool = True
    ENABLE_SCREENSHOTS: bool = True
    ENABLE_VIDEO_CLIPS: bool = True
    CLIP_DURATION: int = 10  # seconds
    BUFFER_SECONDS: int = 5   # pre-event buffer
    OUTPUT_DIR: str = 'media_output'
    MAX_STORAGE_GB: float = 5.0  # auto cleanup limit
    CLEANUP_DAYS: int = 7  # auto delete after 7 days


class ConfigManager:
    """Centralized configuration management"""
    
    SYSTEM = SystemConfig()
    MEDIA = MediaConfig()  # ADD THIS LINE
    
    # Default zones (can be overridden via dashboard)
    DEFAULT_ZONES: Dict[str, ZoneConfig] = {
        'entrance': ZoneConfig(
            name='Entrance',
            polygon=[(50, 400), (200, 400), (200, 480), (50, 480)],
            is_restricted=False,
            allowed_directions=['N', 'S']
        ),
        'restricted': ZoneConfig(
            name='Restricted Zone',
            polygon=[(400, 100), (600, 100), (600, 300), (400, 300)],
            is_restricted=True,
            max_dwell_time=10.0
        )
    }
    
    @classmethod
    def get_email_config(cls) -> Dict[str, str]:
        """Load from environment variables"""
        return {
            'smtp_server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
            'smtp_port': int(os.getenv('SMTP_PORT', '587')),
            'sender_email': os.getenv('SENDER_EMAIL', ''),
            'sender_password': os.getenv('SENDER_PASSWORD', ''),
            'recipient_email': os.getenv('RECIPIENT_EMAIL', ''),
            'enabled': os.getenv('EMAIL_ENABLED', 'false').lower() == 'true'
        }
    
    @classmethod
    def get_db_path(cls) -> str:
        return os.getenv('DB_PATH', 'surveillance_events.db')
    
    @classmethod
    def get_report_path(cls) -> str:
        return os.getenv('REPORT_PATH', 'daily_reports/')


# Global instance - THIS WAS MISSING!
CONFIG = ConfigManager()
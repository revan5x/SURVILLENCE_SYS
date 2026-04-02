#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AI Surveillance System - Dashboard Mode WITH MEDIA CAPTURE AND ALARM SOUNDS
"""

import sys
import os

# Handle __file__ not defined in some environments
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
except NameError:
    pass

import streamlit as st
import cv2
import numpy as np
import psutil
import time
import atexit
import threading
import queue
from datetime import datetime
from enum import Enum
from typing import Optional, Callable, List, Dict, Tuple, Any
import copy
from pathlib import Path

st.set_page_config(
    page_title="AI Surveillance System",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

from config.settings import CONFIG, ConfigManager
from ingestion.frame_manager import FrameManager
from vision.detector import PersonDetector
from tracking.bytetrack import ByteTracker
from logic.zone_manager import ZoneManager, ZoneConfig
from logic.behavior_analyzer import BehaviorAnalyzer, BehaviorType
from logic.anomaly_engine import AnomalyEngine
from storage.event_logger import EventLogger
from alerts.email_notifier import EmailNotifier
from utils.visualization import Visualizer

# Try to import MediaManager, if fails we'll use built-in capture
try:
    from storage.media_manager import MediaManager
    MEDIA_MANAGER_AVAILABLE = True
except ImportError:
    MEDIA_MANAGER_AVAILABLE = False
    print("⚠️ MediaManager not available, using built-in capture")

# ============================================
# AUDIO/ALARM SYSTEM - BUILT-IN
# ============================================
try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("⚠️ sounddevice not installed. Noise detection disabled.")

# Try to import pygame for alarm sounds
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("⚠️ pygame not installed. Alarm sounds will use beep fallback.")

# Try to import winsound for Windows beep fallback
try:
    import winsound
    WINSOUND_AVAILABLE = True
except ImportError:
    WINSOUND_AVAILABLE = False

class NoiseLevel(Enum):
    NORMAL = "NORMAL"
    ELEVATED = "ELEVATED"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class NoiseEvent:
    def __init__(self, level: NoiseLevel, db_level: float, frequency_profile: dict, timestamp: datetime = None):
        self.level = level
        self.db_level = db_level
        self.frequency_profile = frequency_profile
        self.timestamp = timestamp or datetime.now()
        self.description = self._generate_description()
    
    def _generate_description(self) -> str:
        descriptions = {
            NoiseLevel.NORMAL: "Background noise within normal range",
            NoiseLevel.ELEVATED: "Elevated noise levels detected",
            NoiseLevel.HIGH: "High noise levels - possible disturbance",
            NoiseLevel.CRITICAL: "Critical noise levels - immediate attention required"
        }
        return descriptions.get(self.level, "Unknown noise event")

class NoiseDetector:
    def __init__(self, 
                 sample_rate: int = 44100,
                 block_duration: float = 0.5,
                 normal_threshold: float = 60.0,
                 elevated_threshold: float = 75.0,
                 high_threshold: float = 85.0,
                 cooldown_seconds: float = 5.0):
        
        self.sample_rate = sample_rate
        self.block_duration = block_duration
        self.block_size = int(sample_rate * block_duration)
        
        self.thresholds = {
            NoiseLevel.NORMAL: 0,
            NoiseLevel.ELEVATED: normal_threshold,
            NoiseLevel.HIGH: elevated_threshold,
            NoiseLevel.CRITICAL: high_threshold
        }
        
        self.cooldown_seconds = cooldown_seconds
        self.last_alert_time = 0
        
        self.audio_queue = queue.Queue(maxsize=10)
        self.is_running = False
        self.detection_thread = None
        
        self.callback: Optional[Callable[[NoiseEvent], None]] = None
        
        self.freq_bands = {
            'low': (20, 250),
            'mid': (250, 2000),
            'high': (2000, 8000),
            'very_high': (8000, 20000)
        }
        
        self._current_db = 0.0
        self._noise_history = []
        self.max_history = 100
        
    def set_callback(self, callback: Callable[[NoiseEvent], None]):
        self.callback = callback
    
    def start(self) -> bool:
        if not AUDIO_AVAILABLE:
            print("❌ Audio libraries not available")
            return False
        
        try:
            self.is_running = True
            self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
            self.detection_thread.start()
            
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                channels=1,
                dtype=np.float32,
                callback=self._audio_callback
            )
            self.stream.start()
            
            print(f"✅ Noise detector started")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start noise detector: {e}")
            self.is_running = False
            return False
    
    def stop(self):
        self.is_running = False
        if hasattr(self, 'stream'):
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                print(f"Error stopping noise detector: {e}")
    
    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"Audio status: {status}")
        try:
            self.audio_queue.put(indata.copy(), timeout=0.1)
        except queue.Full:
            pass
    
    def _detection_loop(self):
        while self.is_running:
            try:
                audio_data = self.audio_queue.get(timeout=1.0)
                self._process_audio(audio_data)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Noise detection error: {e}")
    
    def _process_audio(self, audio_data: np.ndarray):
        try:
            rms = np.sqrt(np.mean(audio_data**2))
            if rms > 0:
                db = 20 * np.log10(rms) + 94
            else:
                db = 0
            
            self._current_db = db
            self._noise_history.append(db)
            if len(self._noise_history) > self.max_history:
                self._noise_history.pop(0)
            
            freq_profile = self._analyze_frequencies(audio_data)
            level = self._classify_noise(db, freq_profile)
            
            current_time = time.time()
            if level.value in [NoiseLevel.ELEVATED.value, NoiseLevel.HIGH.value, NoiseLevel.CRITICAL.value]:
                if current_time - self.last_alert_time > self.cooldown_seconds:
                    event = NoiseEvent(level, db, freq_profile)
                    self.last_alert_time = current_time
                    
                    if self.callback:
                        threading.Thread(target=self.callback, args=(event,), daemon=True).start()
        except Exception as e:
            print(f"Error processing audio: {e}")
    
    def _analyze_frequencies(self, audio_data: np.ndarray) -> dict:
        try:
            fft = np.fft.fft(audio_data[:, 0])
            freqs = np.fft.fftfreq(len(audio_data), 1/self.sample_rate)
            
            magnitude = np.abs(fft[:len(fft)//2])
            freqs = freqs[:len(freqs)//2]
            
            band_energy = {}
            for band_name, (low, high) in self.freq_bands.items():
                mask = (freqs >= low) & (freqs <= high)
                if np.any(mask):
                    band_energy[band_name] = np.mean(magnitude[mask])
                else:
                    band_energy[band_name] = 0
            
            return band_energy
        except Exception as e:
            return {'low': 0, 'mid': 0, 'high': 0, 'very_high': 0}
    
    def _classify_noise(self, db: float, freq_profile: dict) -> NoiseLevel:
        try:
            total_energy = sum(freq_profile.values()) + 1e-10
            high_freq_ratio = freq_profile.get('high', 0) / total_energy
            
            if db >= self.thresholds[NoiseLevel.CRITICAL] or (db >= self.thresholds[NoiseLevel.HIGH] and high_freq_ratio > 0.5):
                return NoiseLevel.CRITICAL
            elif db >= self.thresholds[NoiseLevel.HIGH]:
                return NoiseLevel.HIGH
            elif db >= self.thresholds[NoiseLevel.ELEVATED]:
                return NoiseLevel.ELEVATED
            else:
                return NoiseLevel.NORMAL
        except Exception as e:
            return NoiseLevel.NORMAL
    
    def get_current_level(self) -> float:
        return self._current_db
    
    def get_average_level(self, seconds: float = 5.0) -> float:
        samples = int(seconds / self.block_duration)
        if len(self._noise_history) >= samples:
            return np.mean(self._noise_history[-samples:])
        return np.mean(self._noise_history) if self._noise_history else 0
    
    def is_available(self) -> bool:
        return AUDIO_AVAILABLE

# ============================================
# ALARM SOUND MANAGER
# ============================================
class AlarmSoundManager:
    """Manages alarm sounds for anomaly detection with cooldown to prevent spamming"""
    
    def __init__(self, cooldown_seconds: float = 5.0):
        self.cooldown_seconds = cooldown_seconds
        self.last_alarm_time = 0
        self.is_initialized = False
        self.alarm_thread = None
        self._lock = threading.Lock()
        
        # Initialize pygame if available
        if PYGAME_AVAILABLE:
            try:
                pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
                self.is_initialized = True
                print("✅ Pygame audio initialized for alarm sounds")
            except Exception as e:
                print(f"⚠️ Failed to initialize pygame audio: {e}")
        
        # Pre-generate alarm sounds
        self.alarm_sounds = {}
        self._generate_alarm_sounds()
    
    def _generate_alarm_sounds(self):
        """Generate different alarm sound frequencies"""
        # Generate a high-pitched alarm tone (1000 Hz)
        sample_rate = 44100
        duration = 0.5  # seconds
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        
        # Create a beeping pattern (2 beeps)
        beep_pattern = np.zeros(int(sample_rate * duration))
        beep_duration = int(sample_rate * 0.1)  # 100ms beep
        gap_duration = int(sample_rate * 0.05)  # 50ms gap
        
        # First beep
        beep1_start = int(sample_rate * 0.05)
        beep1_end = beep1_start + beep_duration
        beep_pattern[beep1_start:beep1_end] = np.sin(2 * np.pi * 1000 * t[beep1_start:beep1_end])
        
        # Second beep
        beep2_start = beep1_end + gap_duration
        beep2_end = beep2_start + beep_duration
        if beep2_end < len(beep_pattern):
            beep_pattern[beep2_start:beep2_end] = np.sin(2 * np.pi * 1000 * t[beep2_start:beep2_end])
        
        # Normalize and convert to 16-bit
        beep_pattern = (beep_pattern * 0.5 * 32767).astype(np.int16)
        
        # Create stereo sound
        stereo_sound = np.column_stack((beep_pattern, beep_pattern))
        
        self.alarm_sounds['standard'] = stereo_sound
        self.alarm_sounds['critical'] = stereo_sound  # Can customize for critical alerts
        
        print("✅ Alarm sounds generated")
    
    def play_alarm(self, severity: str = "CRITICAL", custom_message: str = None):
        """Play alarm sound with cooldown protection"""
        with self._lock:
            current_time = time.time()
            if current_time - self.last_alarm_time < self.cooldown_seconds:
                return False  # Still in cooldown
            self.last_alarm_time = current_time
        
        # Play alarm in separate thread to avoid blocking
        self.alarm_thread = threading.Thread(
            target=self._play_alarm_async,
            args=(severity, custom_message),
            daemon=True
        )
        self.alarm_thread.start()
        return True
    
    def _play_alarm_async(self, severity: str, custom_message: str = None):
        """Internal method to play alarm sound"""
        try:
            if PYGAME_AVAILABLE and self.is_initialized:
                self._play_pygame_alarm(severity)
            elif WINSOUND_AVAILABLE:
                self._play_winsound_alarm(severity)
            else:
                self._play_fallback_alarm(severity)
                
            print(f"🔔 ALARM PLAYED: {severity} - {custom_message or 'Anomaly detected'}")
            
        except Exception as e:
            print(f"❌ Alarm playback error: {e}")
    
    def _play_pygame_alarm(self, severity: str):
        """Play alarm using pygame"""
        try:
            sound_key = 'critical' if severity == 'CRITICAL' else 'standard'
            sound_data = self.alarm_sounds.get(sound_key, self.alarm_sounds['standard'])
            
            # Convert numpy array to pygame Sound
            sound = pygame.sndarray.make_sound(sound_data)
            
            # Play sound
            sound.play()
            pygame.time.wait(int(sound.get_length() * 1000))
            
        except Exception as e:
            print(f"Pygame alarm error: {e}")
            # Fallback to winsound if available
            if WINSOUND_AVAILABLE:
                self._play_winsound_alarm(severity)
    
    def _play_winsound_alarm(self, severity: str):
        """Play alarm using Windows beep (fallback)"""
        try:
            frequency = 1000 if severity == 'CRITICAL' else 800
            duration = 500  # milliseconds
            
            # Play two beeps for pattern
            winsound.Beep(frequency, duration)
            time.sleep(0.1)
            winsound.Beep(frequency, duration)
            
        except Exception as e:
            print(f"Winsound alarm error: {e}")
    
    def _play_fallback_alarm(self, severity: str):
        """Print-based fallback when no audio system available"""
        print(f"🔔 ALARM ALERT: {severity} - Anomaly detected! (Audio not available)")
        # Visual alert in console
        print("\a")  # Bell character
    
    def stop(self):
        """Stop any playing alarm and cleanup"""
        try:
            if PYGAME_AVAILABLE and self.is_initialized:
                pygame.mixer.stop()
        except Exception as e:
            print(f"Error stopping alarm: {e}")
    
    def is_available(self) -> bool:
        """Check if alarm system is available"""
        return PYGAME_AVAILABLE or WINSOUND_AVAILABLE

# ============================================
# BUILT-IN MEDIA CAPTURE (Fallback)
# ============================================
class SimpleMediaCapture:
    """Simple media capture that works without external MediaManager"""
    
    def __init__(self, output_dir="media_output", buffer_seconds=5, fps=30):
        self.output_dir = Path(output_dir)
        self.screenshot_dir = self.output_dir / "screenshots"
        self.clips_dir = self.output_dir / "clips"
        
        # Create directories
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        self.clips_dir.mkdir(parents=True, exist_ok=True)
        
        self.buffer_seconds = buffer_seconds
        self.fps = fps
        self.frame_buffer = []
        self.max_buffer_size = int(fps * buffer_seconds)
        
        self.active_recordings = {}
        
        print(f"✅ SimpleMediaCapture initialized: {self.output_dir}")
    
    def add_frame(self, frame):
        """Add frame to buffer"""
        self.frame_buffer.append(frame.copy())
        if len(self.frame_buffer) > self.max_buffer_size:
            self.frame_buffer.pop(0)
    
    def capture_screenshot(self, frame, alert_data=None):
        """Capture screenshot immediately"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"screenshot_{timestamp}.jpg"
            filepath = self.screenshot_dir / filename
            
            # Ensure frame is valid
            if frame is None or frame.size == 0:
                print("❌ Invalid frame for screenshot")
                return None
            
            # Save with high quality
            success = cv2.imwrite(str(filepath), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            if success and filepath.exists():
                print(f"📸 Screenshot saved: {filepath}")
                return str(filepath)
            else:
                print(f"❌ Failed to save screenshot")
                return None
                
        except Exception as e:
            print(f"❌ Screenshot error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def start_event_recording(self, alert_data, duration=10):
        """Start recording video clip"""
        try:
            event_id = f"clip_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}"
            filename = f"{event_id}.mp4"
            filepath = self.clips_dir / filename
            
            # Store recording info
            self.active_recordings[event_id] = {
                'filepath': str(filepath),
                'frames': [],
                'alert_data': alert_data,
                'duration': duration,
                'start_time': time.time()
            }
            
            # Save pre-event buffer
            self.active_recordings[event_id]['frames'] = [f.copy() for f in self.frame_buffer]
            
            print(f"🎬 Recording started: {event_id} ({len(self.frame_buffer)} buffer frames)")
            return event_id
            
        except Exception as e:
            print(f"❌ Recording start error: {e}")
            return None
    
    def add_frame_to_recording(self, event_id, frame):
        """Add frame to active recording"""
        if event_id in self.active_recordings:
            recording = self.active_recordings[event_id]
            elapsed = time.time() - recording['start_time']
            
            if elapsed < recording['duration']:
                recording['frames'].append(frame.copy())
                return True
            else:
                # Recording complete, save it
                self._save_recording(event_id)
                return False
        return False
    
    def _save_recording(self, event_id):
        """Save recording to file"""
        try:
            if event_id not in self.active_recordings:
                return
            
            recording = self.active_recordings[event_id]
            frames = recording['frames']
            filepath = recording['filepath']
            
            if len(frames) == 0:
                print(f"❌ No frames to save for {event_id}")
                del self.active_recordings[event_id]
                return
            
            # Get frame dimensions
            height, width = frames[0].shape[:2]
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(filepath, fourcc, self.fps, (width, height))
            
            if not out.isOpened():
                print(f"❌ Failed to open video writer for {filepath}")
                del self.active_recordings[event_id]
                return
            
            # Write frames
            for frame in frames:
                out.write(frame)
            
            out.release()
            
            if Path(filepath).exists():
                print(f"🎬 Video saved: {filepath} ({len(frames)} frames)")
            else:
                print(f"❌ Video file not created: {filepath}")
            
            del self.active_recordings[event_id]
            
        except Exception as e:
            print(f"❌ Video save error: {e}")
            import traceback
            traceback.print_exc()
    
    def finalize_all_recordings(self):
        """Save all active recordings"""
        for event_id in list(self.active_recordings.keys()):
            self._save_recording(event_id)

# ============================================
# PERSON FILTER - BUILT-IN
# ============================================
class PersonFilter:
    PERSON_CLASS_ID = 0
    PERSON_CLASS_NAMES = ['person', 'human', 'man', 'woman', 'child', 'people']
    
    def __init__(self, 
                 min_confidence: float = 0.5,
                 min_height_ratio: float = 0.15,
                 max_height_ratio: float = 0.9,
                 aspect_ratio_range: Tuple[float, float] = (0.2, 5.0),
                 enable_pose_check: bool = True):
        
        self.min_confidence = min_confidence
        self.min_height_ratio = min_height_ratio
        self.max_height_ratio = max_height_ratio
        self.aspect_ratio_range = aspect_ratio_range
        self.enable_pose_check = enable_pose_check
        
        self.total_detections = 0
        self.filtered_detections = 0
        self.person_detections = 0
    
    def _get_attr(self, obj: Any, attr: str, default=None):
        if isinstance(obj, dict):
            return obj.get(attr, default)
        return getattr(obj, attr, default)
    
    def filter_detections(self, detections: List[Any], frame_shape: Tuple[int, ...]) -> List[Any]:
        self.total_detections += len(detections)
        frame_h, frame_w = frame_shape[:2]
        
        filtered = []
        
        for det in detections:
            if not self._is_person_class(det):
                continue
            
            confidence = self._get_attr(det, 'confidence', 0)
            if isinstance(confidence, (list, tuple)):
                confidence = confidence[0] if len(confidence) > 0 else 0
            if confidence < self.min_confidence:
                continue
            
            bbox = self._get_bbox(det)
            if bbox is None:
                continue
                
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            
            if w <= 0 or h <= 0:
                continue
            
            height_ratio = h / frame_h
            if height_ratio < self.min_height_ratio or height_ratio > self.max_height_ratio:
                continue
            
            aspect_ratio = h / w
            if aspect_ratio < self.aspect_ratio_range[0] or aspect_ratio > self.aspect_ratio_range[1]:
                continue
            
            if self.enable_pose_check and not self._validate_person_shape(det, frame_shape):
                continue
            
            if isinstance(det, dict):
                det['detection_type'] = 'person'
                det['validation_passed'] = True
            else:
                try:
                    det.detection_type = 'person'
                    det.validation_passed = True
                except:
                    pass
            
            filtered.append(det)
        
        self.filtered_detections += len(filtered)
        self.person_detections += len(filtered)
        
        return filtered
    
    def _get_bbox(self, det: Any) -> Optional[Tuple[float, float, float, float]]:
        bbox = self._get_attr(det, 'bbox', None)
        if bbox is None:
            bbox = self._get_attr(det, 'box', None)
        if bbox is None:
            bbox = self._get_attr(det, 'xyxy', None)
        
        if bbox is None:
            xywh = self._get_attr(det, 'xywh', None)
            if xywh is not None and len(xywh) >= 4:
                x, y, w, h = xywh[:4]
                return (x, y, x + w, y + h)
            return None
        
        if isinstance(bbox, (list, tuple, np.ndarray)) and len(bbox) >= 4:
            return tuple(float(x) for x in bbox[:4])
        
        return None
    
    def _is_person_class(self, det: Any) -> bool:
        class_id = self._get_attr(det, 'class_id', -1)
        if class_id == -1:
            class_id = self._get_attr(det, 'cls', -1)
        
        class_name = self._get_attr(det, 'class_name', '')
        if not class_name:
            class_name = self._get_attr(det, 'name', '')
        
        if isinstance(class_name, str):
            class_name = class_name.lower()
        
        if isinstance(class_id, (int, float, np.integer)):
            if int(class_id) == self.PERSON_CLASS_ID:
                return True
        
        if isinstance(class_name, str):
            if any(name in class_name for name in self.PERSON_CLASS_NAMES):
                return True
        
        return False
    
    def _validate_person_shape(self, det: Any, frame_shape: Tuple) -> bool:
        bbox = self._get_bbox(det)
        if bbox is None:
            return True
        
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        
        if h < w * 0.8:
            return False
        
        keypoints = self._get_attr(det, 'keypoints', None)
        if keypoints is not None:
            if isinstance(keypoints, (list, np.ndarray)):
                valid_count = 0
                for kp in keypoints:
                    if isinstance(kp, (list, tuple, np.ndarray)) and len(kp) >= 3:
                        if kp[2] > 0.5:
                            valid_count += 1
                    else:
                        valid_count += 1
                
                if valid_count < 5:
                    return False
        
        return True

# ============================================
# EMAIL CONFIGURATION
# ============================================
EMAIL_CONFIG = {
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'sender_email': 'gayatrikandula04@gmail.com',
    'sender_password': 'nkhu oihg dhog uzki',
    'recipient_email': 'gayatrikandula04@gmail.com',
    'enabled': True,
    'cooldown_seconds': 30.0
}

def init_session_state():
    defaults = {
        'system_running': False,
        'zones': {},
        'frame_count': 0,
        'camera_initialized': False,
        'components': None,
        'alert_history': [],
        'media_enabled': True,
        'show_media_gallery': False,
        'person_only_mode': True,
        'noise_detection_enabled': False,
        'noise_events': [],
        'last_noise_level': 0.0,
        'noise_alerts': [],
        'email_cooldown': {},
        'capture_status': [],  # Track capture status messages
        'alarm_enabled': True,  # NEW: Alarm sound toggle
        'alarm_cooldown': 5.0,  # NEW: Alarm cooldown in seconds
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def cleanup_camera():
    if st.session_state.get('components'):
        # Stop alarm if running
        if st.session_state.components.get('alarm_manager'):
            try:
                st.session_state.components['alarm_manager'].stop()
            except Exception as e:
                print(f"Error stopping alarm: {e}")
        
        # Finalize all recordings before cleanup
        if st.session_state.components.get('media_manager'):
            try:
                st.session_state.components['media_manager'].finalize_all_recordings()
            except Exception as e:
                print(f"Error finalizing recordings: {e}")
        
        if st.session_state.components.get('frame_manager'):
            try:
                st.session_state.components['frame_manager'].release()
            except Exception as e:
                print(f"Error releasing frame manager: {e}")
        if st.session_state.components.get('noise_detector'):
            try:
                st.session_state.components['noise_detector'].stop()
            except Exception as e:
                print(f"Error stopping noise detector: {e}")
        st.session_state.camera_initialized = False

atexit.register(cleanup_camera)

def render_sidebar():
    st.sidebar.title("🔧 System Configuration")
    st.sidebar.header("Zone Management")
    
    with st.sidebar.expander("➕ Add New Zone", expanded=False):
        zone_name = st.text_input("Zone Name", "Zone1", key="zone_name")
        is_restricted = st.checkbox("Restricted Zone", True, key="zone_restricted")
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            x_points = st.text_area("X coords", "100,200,200,100", height=80, key="zone_x")
        with col2:
            y_points = st.text_area("Y coords", "100,100,200,200", height=80, key="zone_y")
        
        if st.button("Add Zone", key="add_zone_btn", use_container_width=True):
            try:
                xs = [int(x.strip()) for x in x_points.split(",") if x.strip()]
                ys = [int(y.strip()) for y in y_points.split(",") if y.strip()]
                
                if len(xs) == len(ys) and len(xs) >= 3:
                    polygon = list(zip(xs, ys))
                    st.session_state.zones[zone_name] = {
                        'polygon': polygon,
                        'restricted': is_restricted,
                        'allowed_directions': ['N', 'S'] if is_restricted else ['N', 'S', 'E', 'W']
                    }
                    st.success(f"✅ Zone '{zone_name}' added!")
                    st.rerun()
                else:
                    st.error("❌ Need equal x,y pairs (min 3 points)")
            except Exception as e:
                st.error(f"❌ Invalid coordinates: {e}")
    
    if st.session_state.zones:
        st.sidebar.subheader("Active Zones")
        for name in list(st.session_state.zones.keys()):
            col1, col2 = st.sidebar.columns([4, 1])
            icon = "🔴" if st.session_state.zones[name].get('restricted') else "🟢"
            col1.write(f"{icon} {name}")
            if col2.button("🗑️", key=f"del_{name}"):
                del st.session_state.zones[name]
                st.rerun()
    else:
        st.sidebar.info("No zones defined. Add zones above.")
    
    st.sidebar.header("🔒 Privacy Controls")
    st.sidebar.toggle("Face Blurring (Always On)", value=True, disabled=True)
    
    st.sidebar.header("🎯 Detection Settings")
    
    st.session_state.person_only_mode = st.sidebar.toggle(
        "👤 Person-Only Detection", 
        value=st.session_state.person_only_mode,
        key="person_only_toggle"
    )
    
    st.session_state.noise_detection_enabled = st.sidebar.toggle(
        "🔊 Noise Detection", 
        value=st.session_state.noise_detection_enabled,
        key="noise_toggle"
    )
    
    if st.session_state.noise_detection_enabled:
        with st.sidebar.expander("🔊 Noise Settings", expanded=True):
            noise_threshold = st.slider("Elevated Threshold (dB)", 50, 80, 60, key="noise_elevated")
            noise_high = st.slider("High Threshold (dB)", 70, 100, 75, key="noise_high")
            noise_critical = st.slider("Critical Threshold (dB)", 80, 120, 85, key="noise_critical")
            
            st.session_state.noise_config = {
                'elevated': noise_threshold,
                'high': noise_high,
                'critical': noise_critical,
                'cooldown': 5.0
            }
    
    # NEW: Alarm Sound Controls
    st.sidebar.header("🔔 Alarm Settings")
    st.session_state.alarm_enabled = st.sidebar.toggle(
        "🔔 Enable Alarm Sound",
        value=st.session_state.alarm_enabled,
        key="alarm_toggle"
    )
    
    if st.session_state.alarm_enabled:
        st.session_state.alarm_cooldown = st.sidebar.slider(
            "Alarm Cooldown (seconds)",
            min_value=1,
            max_value=30,
            value=int(st.session_state.alarm_cooldown),
            key="alarm_cooldown_slider"
        )
        
        # Test alarm button
        if st.sidebar.button("🔔 Test Alarm", key="test_alarm_btn"):
            test_alarm_manager = AlarmSoundManager(cooldown_seconds=0)  # No cooldown for test
            test_alarm_manager.play_alarm("CRITICAL", "Test alarm")
            st.sidebar.success("✅ Alarm test triggered! Check your speakers.")
    
    st.sidebar.header("📸 Media Capture")
    
    with st.sidebar.expander("⚙️ Capture Settings", expanded=True):
        st.session_state.media_enabled = st.sidebar.toggle(
            "Enable Media Capture", 
            value=st.session_state.media_enabled,
            key="media_toggle"
        )
        
        if st.session_state.media_enabled:
            enable_screenshots = st.checkbox("📸 Screenshots", value=True, key="enable_screenshots")
            enable_clips = st.checkbox("🎬 Video Clips", value=True, key="enable_clips")
            clip_duration = st.slider("Clip Duration (sec)", 5, 30, 10, key="clip_duration")
            buffer_seconds = st.slider("Pre-event Buffer (sec)", 1, 10, 5, key="buffer_seconds")
            
            st.session_state.media_config = {
                'enable_screenshots': enable_screenshots,
                'enable_clips': enable_clips,
                'clip_duration': clip_duration,
                'buffer_seconds': buffer_seconds
            }
        else:
            st.session_state.media_config = None
    
    st.sidebar.header("▶️ System Control")
    
    if not st.session_state.system_running:
        if st.button("🚀 START SURVEILLANCE", type="primary", use_container_width=True):
            st.session_state.system_running = True
            st.rerun()
    else:
        if st.button("⏹️ STOP SYSTEM", type="secondary", use_container_width=True):
            st.session_state.system_running = False
            cleanup_camera()
            st.session_state.components = None
            st.rerun()
    
    status = "🟢 RUNNING" if st.session_state.system_running else "🔴 STOPPED"
    st.sidebar.markdown(f"**Status:** {status}")
    
    st.sidebar.header("⚡ Performance")
    frame_skip = st.sidebar.slider("Frame Skip", 1, 5, 2, key="frame_skip")
    
    # Show recent capture status
    if st.session_state.capture_status:
        st.sidebar.header("📸 Recent Captures")
        for status_msg in st.session_state.capture_status[-3:]:
            st.sidebar.text(status_msg)
    
    return frame_skip

def main():
    init_session_state()
    frame_skip = render_sidebar()
    
    st.title("🔒 AI Surveillance System")
    st.markdown("*Edge-Based Intelligent Surveillance with Privacy Protection*")
    
    if not st.session_state.system_running:
        st.info("👆 Click 'START SURVEILLANCE' in the sidebar to begin")
        
        st.divider()
        tab1, tab2, tab3 = st.tabs(["📊 Event History", "🎥 Media Gallery", "🔊 Noise Events"])
        
        with tab1:
            try:
                event_logger = EventLogger(ConfigManager.get_db_path())
                events = event_logger.query_events(limit=20)
                if events:
                    import pandas as pd
                    df = pd.DataFrame(events)
                    if 'screenshot_path' in df.columns:
                        df['has_media'] = df['screenshot_path'].notna() | df['clip_path'].notna()
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("No events recorded yet")
            except Exception as e:
                st.error(f"Database error: {e}")
        
        with tab2:
            render_media_gallery()
        
        with tab3:
            render_noise_events_history()
        
        return
    
    st.success("✅ System is running")
    
    vid_col, met_col = st.columns([2, 1])
    
    with vid_col:
        st.subheader("📹 Live Feed")
        video_placeholder = st.empty()
        
        c1, c2, c3 = st.columns(3)
        status_text = c1.empty()
        tracks_text = c2.empty()
        fps_text = c3.empty()
    
    with met_col:
        st.subheader("📊 Metrics")
        cpu_metric = st.empty()
        mem_metric = st.empty()
        inf_metric = st.empty()
        
        noise_metric = st.empty()
        
        st.divider()
        
        if st.session_state.noise_detection_enabled:
            st.subheader("🔊 Noise Alerts")
            noise_alerts_placeholder = st.empty()
        
        st.subheader("📋 Recent Events")
        events_text = st.empty()
        
        # NEW: Alarm Status Indicator
        if st.session_state.alarm_enabled:
            st.subheader("🔔 Alarm Status")
            alarm_status_text = st.empty()
            alarm_status_text.success("🔔 Alarm Active - Will sound on anomalies")
        else:
            alarm_status_text = None
        
        # Capture status indicator
        if st.session_state.media_enabled:
            st.subheader("📸 Capture Status")
            capture_status_text = st.empty()
        else:
            capture_status_text = None
    
    try:
        if st.session_state.components is None:
            st.session_state.components = initialize_system(frame_skip)
            atexit.register(cleanup_camera)
        
        system = st.session_state.components
        system['frame_manager'].frame_skip = frame_skip
        
        if not st.session_state.camera_initialized:
            if not system['frame_manager'].start():
                st.error("❌ Failed to start camera!")
                cleanup_camera()
                st.session_state.system_running = False
                st.rerun()
            st.session_state.camera_initialized = True
            st.info("📷 Camera started")
        
        process_frames(system, video_placeholder, status_text, tracks_text, 
                      fps_text, cpu_metric, mem_metric, inf_metric, events_text,
                      noise_metric if st.session_state.noise_detection_enabled else None,
                      noise_alerts_placeholder if st.session_state.noise_detection_enabled else None,
                      capture_status_text,
                      alarm_status_text if st.session_state.alarm_enabled else None)
        
    except Exception as e:
        st.error(f"❌ System error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        cleanup_camera()

def render_noise_events_history():
    if not st.session_state.noise_events:
        st.info("No noise events recorded yet.")
        return
    
    st.subheader("Recent Noise Events")
    
    import pandas as pd
    noise_df = pd.DataFrame(st.session_state.noise_events)
    st.dataframe(noise_df, use_container_width=True)

def render_media_gallery():
    media_dir = Path("media_output")

    if not media_dir.exists():
        st.info("No media captured yet.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📸 Recent Screenshots")
        screenshot_dir = media_dir / "screenshots"

        if screenshot_dir.exists():
            screenshots = sorted(
                screenshot_dir.glob("*.jpg"),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )[:10]

            if screenshots:
                for img_path in screenshots:
                    with st.expander(f"📷 {img_path.name}"):
                        st.image(str(img_path), use_column_width=True)
                        colA, colB = st.columns(2)
                        with colA:
                            with open(img_path, "rb") as f:
                                st.download_button(
                                    "⬇️ Download",
                                    f,
                                    file_name=img_path.name,
                                    key=f"download_{img_path.name}"
                                )
                        with colB:
                            if st.button("🗑️ Delete", key=f"delete_{img_path.name}"):
                                try:
                                    img_path.unlink()
                                    st.success("Deleted")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Delete failed: {e}")
            else:
                st.info("No screenshots yet")
        else:
            st.info("Screenshot folder not found")

    with col2:
        st.subheader("🎬 Recent Video Clips")
        clips_dir = media_dir / "clips"

        if clips_dir.exists():
            clips = sorted(
                clips_dir.glob("*.mp4"),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )[:5]

            if clips:
                for clip_path in clips:
                    with st.expander(f"🎥 {clip_path.name}"):
                        st.video(str(clip_path))
                        colA, colB = st.columns(2)
                        with colA:
                            with open(clip_path, "rb") as f:
                                st.download_button(
                                    "⬇️ Download",
                                    f,
                                    file_name=clip_path.name,
                                    key=f"download_vid_{clip_path.name}"
                                )
                        with colB:
                            if st.button("🗑️ Delete", key=f"delete_vid_{clip_path.name}"):
                                try:
                                    clip_path.unlink()
                                    st.success("Deleted")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Delete failed: {e}")
            else:
                st.info("No video clips yet")
        else:
            st.info("Clips folder not found")

def initialize_system(frame_skip):
    config = CONFIG.SYSTEM
    
    # Initialize alarm manager
    alarm_manager = None
    if st.session_state.alarm_enabled:
        try:
            alarm_manager = AlarmSoundManager(cooldown_seconds=st.session_state.alarm_cooldown)
            print(f"✅ Alarm Sound Manager initialized (cooldown: {st.session_state.alarm_cooldown}s)")
        except Exception as e:
            print(f"⚠️ Failed to initialize alarm manager: {e}")
    
    # Initialize media manager (SimpleMediaCapture as fallback)
    media_manager = None
    if st.session_state.media_enabled and st.session_state.media_config:
        media_config = st.session_state.media_config
        try:
            if MEDIA_MANAGER_AVAILABLE:
                # Try external MediaManager first
                media_manager = MediaManager(
                    buffer_seconds=media_config['buffer_seconds'],
                    fps=config.FPS_TARGET,
                    output_dir="media_output"
                )
                print(f"✅ External MediaManager initialized")
            else:
                # Use built-in SimpleMediaCapture
                media_manager = SimpleMediaCapture(
                    buffer_seconds=media_config['buffer_seconds'],
                    fps=config.FPS_TARGET,
                    output_dir="media_output"
                )
                print(f"✅ SimpleMediaCapture initialized")
        except Exception as e:
            print(f"❌ Failed to initialize media manager: {e}")
            media_manager = None
    
    components = {
        'frame_manager': FrameManager(
            source=0,
            width=config.FRAME_WIDTH,
            height=config.FRAME_HEIGHT,
            fps=config.FPS_TARGET,
            frame_skip=frame_skip
        ),
        'detector': PersonDetector(),
        'tracker': ByteTracker(),
        'zone_manager': ZoneManager(),
        'behavior_analyzer': BehaviorAnalyzer(),
        'anomaly_engine': AnomalyEngine(),
        'event_logger': EventLogger(ConfigManager.get_db_path()),
        'email_notifier': EmailNotifier(EMAIL_CONFIG),
        'visualizer': Visualizer(),
        'media_manager': media_manager,
        'alarm_manager': alarm_manager,  # NEW: Alarm manager
        'person_filter': PersonFilter(
            min_confidence=0.5,
            min_height_ratio=0.1,
            enable_pose_check=True
        ) if st.session_state.person_only_mode else None,
        'noise_detector': None,
        'process': psutil.Process(),
        'recent_events': [],
        'frame_count': 0,
        'active_recordings': {},  # Track active video recordings
    }
    
    # Initialize noise detector if enabled
    if st.session_state.noise_detection_enabled:
        noise_config = st.session_state.get('noise_config', {})
        noise_detector = NoiseDetector(
            normal_threshold=noise_config.get('elevated', 60),
            elevated_threshold=noise_config.get('high', 75),
            high_threshold=noise_config.get('critical', 85),
            cooldown_seconds=noise_config.get('cooldown', 5.0)
        )
        
        def on_noise_event(event: NoiseEvent):
            try:
                timestamp_str = datetime.now().strftime("%H:%M:%S")
                
                alert_data = {
                    'time': timestamp_str,
                    'level': event.level.value,
                    'db': round(event.db_level, 1),
                    'description': event.description,
                    'timestamp': datetime.now().isoformat()
                }
                
                if 'noise_events' not in st.session_state:
                    st.session_state.noise_events = []
                if 'noise_alerts' not in st.session_state:
                    st.session_state.noise_alerts = []
                
                st.session_state.noise_events.append(alert_data)
                st.session_state.noise_events = st.session_state.noise_events[-20:]
                
                st.session_state.noise_alerts.append(alert_data)
                st.session_state.noise_alerts = st.session_state.noise_alerts[-5:]
                
                # Log to database (NO media for noise)
                try:
                    components['event_logger'].log_event(
                        track_id=-1,
                        anomaly_type=f"NOISE_{event.level.value}",
                        severity=event.level.value,
                        zone="AUDIO",
                        velocity=0,
                        duration=0,
                        description=event.description,
                        screenshot_path=None,
                        clip_path=None
                    )
                except Exception as e:
                    print(f"Error logging noise event: {e}")
                
                # Play alarm for HIGH and CRITICAL noise events
                if event.level in [NoiseLevel.HIGH, NoiseLevel.CRITICAL]:
                    if components.get('alarm_manager'):
                        components['alarm_manager'].play_alarm(
                            severity=event.level.value,
                            custom_message=f"Noise Alert: {event.description}"
                        )
                
                # Send email for HIGH and CRITICAL
                if event.level in [NoiseLevel.HIGH, NoiseLevel.CRITICAL]:
                    current_time = time.time()
                    email_key = f"noise_{event.level.value}"
                    
                    if email_key not in st.session_state.email_cooldown:
                        st.session_state.email_cooldown[email_key] = 0
                    
                    if current_time - st.session_state.email_cooldown[email_key] > EMAIL_CONFIG.get('cooldown_seconds', 30):
                        st.session_state.email_cooldown[email_key] = current_time
                        
                        def send_noise_email_async():
                            try:
                                components['email_notifier'].send_alert(
                                    track_id=-1,
                                    anomaly_type=f"NOISE_{event.level.value}",
                                    severity=event.level.value,
                                    zone="AUDIO",
                                    description=f"Noise Alert: {event.description} (Level: {event.db_level:.1f} dB)",
                                    frame=None,
                                    screenshot_path=None,
                                    clip_path=None
                                )
                            except Exception as e:
                                print(f"Noise email error: {e}")
                        
                        threading.Thread(target=send_noise_email_async, daemon=True).start()
                
                print(f"🔊 NOISE EVENT: {event.level.value} at {event.db_level:.1f} dB")
                
            except Exception as e:
                print(f"Error in noise event callback: {e}")
        
        noise_detector.set_callback(on_noise_event)
        if noise_detector.start():
            components['noise_detector'] = noise_detector
            st.session_state.noise_alerts = []
        else:
            st.sidebar.warning("⚠️ Could not start noise detection.")
    
    for name, data in st.session_state.zones.items():
        components['zone_manager'].add_zone(ZoneConfig(
            name=name,
            polygon=data['polygon'],
            is_restricted=data.get('restricted', False),
            allowed_directions=data.get('allowed_directions', ['N', 'S', 'E', 'W'])
        ))
    
    if not components['zone_manager'].zones:
        for name, zone in ConfigManager.DEFAULT_ZONES.items():
            components['zone_manager'].add_zone(zone)
    
    return components

def process_frames(system, video_placeholder, status_text, tracks_text, 
                   fps_text, cpu_metric, mem_metric, inf_metric, events_text,
                   noise_metric=None, noise_alerts_placeholder=None,
                   capture_status_text=None, alarm_status_text=None):
    
    last_time = time.time()
    frame_count = 0
    error_placeholder = st.empty()
    
    media_config = st.session_state.get('media_config', {})
    media_manager = system.get('media_manager')
    alarm_manager = system.get('alarm_manager')  # Get alarm manager
    
    while st.session_state.system_running:
        try:
            if not st.session_state.system_running:
                break
            
            frame, timestamp, should_process = system['frame_manager'].get_frame()
            
            if frame is None:
                error_placeholder.warning("⚠️ No frame received")
                time.sleep(0.05)
                continue
            
            error_placeholder.empty()
            
            # ALWAYS add frame to media buffer first
            if media_manager:
                try:
                    media_manager.add_frame(frame)
                except Exception as e:
                    print(f"⚠️ Media buffer error: {e}")
            
            # Update active video recordings
            if media_manager and hasattr(media_manager, 'active_recordings'):
                for event_id in list(media_manager.active_recordings.keys()):
                    try:
                        media_manager.add_frame_to_recording(event_id, frame)
                    except Exception as e:
                        print(f"⚠️ Recording frame error: {e}")
            
            if not should_process:
                continue
            
            frame_count += 1
            
            # Detection
            try:
                detections, inference_time = system['detector'].detect(frame)
            except Exception as e:
                print(f"Detection error: {e}")
                detections = []
                inference_time = 0
            
            # Filter for persons
            if st.session_state.person_only_mode and system.get('person_filter'):
                original_count = len(detections)
                detections = system['person_filter'].filter_detections(detections, frame.shape)
                if original_count - len(detections) > 0:
                    print(f"🎯 Filtered {original_count - len(detections)} non-person objects")
            
            # Privacy blur
            try:
                privacy_frame = system['visualizer'].apply_privacy_blur(frame, detections)
            except Exception as e:
                privacy_frame = frame
            
            # Tracking
            try:
                tracks = system['tracker'].update(detections, timestamp)
            except Exception as e:
                tracks = []
            
            # Process visual anomalies - CAPTURE MEDIA AND PLAY ALARM
            behavior_profiles = {}
            alarm_triggered_this_frame = False  # Track if alarm was triggered
            
            for track in tracks:
                try:
                    profile = system['behavior_analyzer'].analyze(track, system['zone_manager'])
                    behavior_profiles[track.track_id] = profile
                    
                    event = system['anomaly_engine'].evaluate(profile)
                    
                    if event:
                        print(f"🚨 ANOMALY DETECTED: Track {event.track_id} - {event.anomaly_type}")
                        
                        # NEW: Play alarm sound for visual anomalies
                        if st.session_state.alarm_enabled and alarm_manager and not alarm_triggered_this_frame:
                            alarm_played = alarm_manager.play_alarm(
                                severity=event.severity.value,
                                custom_message=f"Track {event.track_id}: {event.anomaly_type} in {event.zone}"
                            )
                            if alarm_played:
                                alarm_triggered_this_frame = True
                                print(f"🔔 ALARM TRIGGERED for {event.anomaly_type}")
                                # Update alarm status in UI
                                if alarm_status_text:
                                    alarm_status_text.warning(f"🔔 ALARM: {event.anomaly_type} detected!")
                        
                        # CAPTURE SCREENSHOT IMMEDIATELY (in main thread)
                        screenshot_path = None
                        clip_event_id = None
                        
                        if st.session_state.media_enabled and media_manager:
                            try:
                                # Screenshot capture
                                if media_config.get('enable_screenshots', True):
                                    screenshot_path = media_manager.capture_screenshot(
                                        frame.copy(),
                                        alert_data={
                                            'track_id': event.track_id,
                                            'anomaly_type': event.anomaly_type,
                                            'zone': event.zone
                                        }
                                    )
                                    
                                    if screenshot_path:
                                        status_msg = f"✅ Screenshot: {Path(screenshot_path).name}"
                                        st.session_state.capture_status.append(status_msg)
                                        st.session_state.capture_status = st.session_state.capture_status[-5:]
                                        print(f"📸 Screenshot saved: {screenshot_path}")
                                        
                                        if capture_status_text:
                                            capture_status_text.success(status_msg)
                                    else:
                                        print("❌ Screenshot failed")
                                
                                # Video recording start
                                if media_config.get('enable_clips', True):
                                    clip_event_id = media_manager.start_event_recording(
                                        alert_data={
                                            'track_id': event.track_id,
                                            'anomaly_type': event.anomaly_type,
                                            'zone': event.zone
                                        },
                                        duration=media_config.get('clip_duration', 10)
                                    )
                                    
                                    if clip_event_id:
                                        status_msg = f"🎬 Recording: {clip_event_id[:20]}..."
                                        st.session_state.capture_status.append(status_msg)
                                        st.session_state.capture_status = st.session_state.capture_status[-5:]
                                        print(f"🎬 Video recording started: {clip_event_id}")
                                        
                                        # Store in system for frame addition
                                        system['active_recordings'][clip_event_id] = True
                                
                            except Exception as e:
                                print(f"❌ Media capture error: {e}")
                                import traceback
                                traceback.print_exc()
                        
                        # Log to database
                        try:
                            system['event_logger'].log_event(
                                track_id=event.track_id,
                                anomaly_type=event.anomaly_type,
                                severity=event.severity.value,
                                zone=event.zone,
                                velocity=event.velocity,
                                duration=event.duration,
                                screenshot_path=screenshot_path,
                                clip_path=None
                            )
                            print(f"📝 Event logged to database")
                        except Exception as e:
                            print(f"❌ Database logging error: {e}")
                        
                        # Send email (async)
                        try:
                            current_time = time.time()
                            email_key = f"visual_{event.track_id}_{event.anomaly_type}"
                            
                            if email_key not in st.session_state.email_cooldown:
                                st.session_state.email_cooldown[email_key] = 0
                            
                            if current_time - st.session_state.email_cooldown[email_key] > EMAIL_CONFIG.get('cooldown_seconds', 30):
                                st.session_state.email_cooldown[email_key] = current_time
                                
                                def send_email():
                                    try:
                                        system['email_notifier'].send_alert(
                                            track_id=event.track_id,
                                            anomaly_type=event.anomaly_type,
                                            severity=event.severity.value,
                                            zone=event.zone,
                                            description=event.description,
                                            frame=frame if not screenshot_path else None,
                                            screenshot_path=screenshot_path,
                                            clip_path=None
                                        )
                                    except Exception as e:
                                        print(f"Email error: {e}")
                                
                                threading.Thread(target=send_email, daemon=True).start()
                        except Exception as e:
                            print(f"Email setup error: {e}")
                        
                        # Add to recent events
                        event_entry = {
                            'time': datetime.now().strftime("%H:%M:%S"),
                            'track': event.track_id,
                            'type': event.anomaly_type,
                            'severity': event.severity.value
                        }
                        
                        if 'recent_events' not in system:
                            system['recent_events'] = []
                        system['recent_events'].append(event_entry)
                        system['recent_events'] = system['recent_events'][-10:]
                        
                except Exception as e:
                    print(f"Track processing error: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Reset alarm status display after cooldown period
            if alarm_status_text and not alarm_triggered_this_frame:
                # Check if enough time has passed since last alarm
                if alarm_manager and (time.time() - alarm_manager.last_alarm_time > alarm_manager.cooldown_seconds):
                    alarm_status_text.success("🔔 Alarm Active - Will sound on anomalies")
            
            # Cleanup
            try:
                active_ids = [t.track_id for t in tracks]
                system['behavior_analyzer'].cleanup_old_tracks(active_ids)
                system['anomaly_engine'].cleanup_old_tracks(active_ids)
            except Exception as e:
                print(f"Cleanup error: {e}")
            
            # Visualization
            try:
                display_frame = system['zone_manager'].draw_zones(privacy_frame)
                display_frame = system['visualizer'].draw_tracks(display_frame, tracks, behavior_profiles)
            except Exception as e:
                display_frame = privacy_frame
            
            # Metrics
            current_time = time.time()
            fps = 1.0 / (current_time - last_time) if (current_time - last_time) > 0 else 0
            last_time = current_time
            
            try:
                cpu_percent = system['process'].cpu_percent(interval=None) / psutil.cpu_count()
                memory_mb = system['process'].memory_info().rss / 1024 / 1024
            except Exception as e:
                cpu_percent = 0
                memory_mb = 0
            
            # HUD
            try:
                display_frame = system['visualizer'].draw_hud(
                    display_frame, fps, cpu_percent, memory_mb, 
                    len(tracks), inference_time, 0
                )
            except Exception as e:
                pass
            
            # Update UI
            try:
                video_placeholder.image(
                    cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB),
                    channels="RGB",
                    use_column_width=True
                )
                
                status_text.metric("Status", "Active")
                tracks_text.metric("Tracks", len(tracks))
                fps_text.metric("FPS", f"{fps:.1f}")
                
                cpu_metric.metric("CPU", f"{cpu_percent:.1f}%")
                mem_metric.metric("Memory", f"{memory_mb:.1f} MB")
                inf_metric.metric("Inference", f"{inference_time:.1f} ms")
            except Exception as e:
                print(f"UI update error: {e}")
            
            # Noise display
            if st.session_state.noise_detection_enabled and system.get('noise_detector') and noise_metric:
                try:
                    current_db = system['noise_detector'].get_current_level()
                    avg_db = system['noise_detector'].get_average_level(3.0)
                    
                    if current_db >= 85:
                        noise_color = "🔴"
                    elif current_db >= 75:
                        noise_color = "🟠"
                    elif current_db >= 60:
                        noise_color = "🟡"
                    else:
                        noise_color = "🟢"
                    
                    noise_metric.metric(
                        "🔊 Noise Level", 
                        f"{noise_color} {current_db:.1f} dB",
                        f"Avg: {avg_db:.1f} dB"
                    )
                    
                    if noise_alerts_placeholder and st.session_state.get('noise_alerts'):
                        recent_alerts = st.session_state.noise_alerts[-3:]
                        alert_text = "\n".join([
                            f"{a['time']} | {a['level']} | {a['db']} dB"
                            for a in reversed(recent_alerts)
                        ])
                        noise_alerts_placeholder.text(alert_text)
                except Exception as e:
                    print(f"Noise display error: {e}")
            
            # Events display
            try:
                if system.get('recent_events'):
                    events_df = "\n".join([
                        f"{e['time']} | Track {e['track']} | {e['type']} | {e['severity']}"
                        for e in reversed(system['recent_events'][-5:])
                    ])
                    events_text.text(events_df)
                else:
                    events_text.text("No events")
            except Exception as e:
                print(f"Events display error: {e}")
            
            time.sleep(0.001)
            
        except Exception as e:
            error_placeholder.error(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            time.sleep(0.1)
    
    # Finalize all recordings before stopping
    if media_manager and hasattr(media_manager, 'finalize_all_recordings'):
        try:
            media_manager.finalize_all_recordings()
            print("✅ All recordings finalized")
        except Exception as e:
            print(f"Error finalizing recordings: {e}")
    
    cleanup_camera()
    st.info("⏹️ System stopped")

if __name__ == "__main__":
    main()
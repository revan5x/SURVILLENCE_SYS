#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Noise Detection Module - Detects abnormal sounds using microphone input
"""

import numpy as np
import threading
import queue
import time
from datetime import datetime
from enum import Enum
from typing import Optional, Callable
import logging

try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("⚠️ sounddevice not installed. Noise detection disabled.")
    print("   Install with: pip install sounddevice")


class NoiseLevel(Enum):
    NORMAL = "NORMAL"
    ELEVATED = "ELEVATED"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class NoiseEvent:
    def __init__(self, level: NoiseLevel, db_level: float, frequency_profile: dict, timestamp: datetime = None):
        self.level = level
        self.db_level = db_level  # Decibel level
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
                 normal_threshold: float = 60.0,   # dB
                 elevated_threshold: float = 75.0,  # dB
                 high_threshold: float = 85.0,      # dB
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
        
        self.audio_queue = queue.Queue()
        self.is_running = False
        self.detection_thread = None
        
        self.callback: Optional[Callable[[NoiseEvent], None]] = None
        
        # Frequency analysis bands (Hz)
        self.freq_bands = {
            'low': (20, 250),      # Low frequency (rumbling, machinery)
            'mid': (250, 2000),    # Mid frequency (speech, alarms)
            'high': (2000, 8000),   # High frequency (screams, glass breaking)
            'very_high': (8000, 20000)  # Very high frequency
        }
        
        self._current_db = 0.0
        self._noise_history = []
        self.max_history = 100
        
    def set_callback(self, callback: Callable[[NoiseEvent], None]):
        """Set callback function for noise events"""
        self.callback = callback
    
    def start(self) -> bool:
        """Start noise detection"""
        if not AUDIO_AVAILABLE:
            print("❌ Audio libraries not available")
            return False
        
        try:
            self.is_running = True
            self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
            self.detection_thread.start()
            
            # Start audio stream
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                channels=1,
                dtype=np.float32,
                callback=self._audio_callback
            )
            self.stream.start()
            
            print(f"✅ Noise detector started (Thresholds: {self.thresholds[NoiseLevel.ELEVATED]}/{self.thresholds[NoiseLevel.HIGH]}/{self.thresholds[NoiseLevel.CRITICAL]} dB)")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start noise detector: {e}")
            self.is_running = False
            return False
    
    def stop(self):
        """Stop noise detection"""
        self.is_running = False
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        print("⏹️ Noise detector stopped")
    
    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for audio stream"""
        if status:
            print(f"Audio status: {status}")
        self.audio_queue.put(indata.copy())
    
    def _detection_loop(self):
        """Main detection loop"""
        while self.is_running:
            try:
                # Get audio data with timeout
                audio_data = self.audio_queue.get(timeout=1.0)
                self._process_audio(audio_data)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Noise detection error: {e}")
    
    def _process_audio(self, audio_data: np.ndarray):
        """Process audio block and detect noise levels"""
        # Calculate RMS and convert to dB
        rms = np.sqrt(np.mean(audio_data**2))
        if rms > 0:
            db = 20 * np.log10(rms) + 94  # Convert to SPL dB (approximate)
        else:
            db = 0
        
        self._current_db = db
        self._noise_history.append(db)
        if len(self._noise_history) > self.max_history:
            self._noise_history.pop(0)
        
        # Frequency analysis using FFT
        freq_profile = self._analyze_frequencies(audio_data)
        
        # Determine noise level
        level = self._classify_noise(db, freq_profile)
        
        # Trigger alert if above elevated and not in cooldown
        current_time = time.time()
        if level.value in [NoiseLevel.ELEVATED.value, NoiseLevel.HIGH.value, NoiseLevel.CRITICAL.value]:
            if current_time - self.last_alert_time > self.cooldown_seconds:
                event = NoiseEvent(level, db, freq_profile)
                self.last_alert_time = current_time
                
                if self.callback:
                    self.callback(event)
    
    def _analyze_frequencies(self, audio_data: np.ndarray) -> dict:
        """Analyze frequency content of audio"""
        # Perform FFT
        fft = np.fft.fft(audio_data[:, 0])
        freqs = np.fft.fftfreq(len(audio_data), 1/self.sample_rate)
        
        # Get magnitude spectrum (only positive frequencies)
        magnitude = np.abs(fft[:len(fft)//2])
        freqs = freqs[:len(freqs)//2]
        
        # Calculate energy in each band
        band_energy = {}
        for band_name, (low, high) in self.freq_bands.items():
            mask = (freqs >= low) & (freqs <= high)
            if np.any(mask):
                band_energy[band_name] = np.mean(magnitude[mask])
            else:
                band_energy[band_name] = 0
        
        return band_energy
    
    def _classify_noise(self, db: float, freq_profile: dict) -> NoiseLevel:
        """Classify noise level based on dB and frequency content"""
        # Check for specific patterns (e.g., high frequency screams)
        high_freq_ratio = freq_profile.get('high', 0) / (sum(freq_profile.values()) + 1e-10)
        
        if db >= self.thresholds[NoiseLevel.CRITICAL] or (db >= self.thresholds[NoiseLevel.HIGH] and high_freq_ratio > 0.5):
            return NoiseLevel.CRITICAL
        elif db >= self.thresholds[NoiseLevel.HIGH]:
            return NoiseLevel.HIGH
        elif db >= self.thresholds[NoiseLevel.ELEVATED]:
            return NoiseLevel.ELEVATED
        else:
            return NoiseLevel.NORMAL
    
    def get_current_level(self) -> float:
        """Get current dB level"""
        return self._current_db
    
    def get_average_level(self, seconds: float = 5.0) -> float:
        """Get average dB level over last N seconds"""
        samples = int(seconds / self.block_duration)
        if len(self._noise_history) >= samples:
            return np.mean(self._noise_history[-samples:])
        return np.mean(self._noise_history) if self._noise_history else 0
    
    def is_available(self) -> bool:
        """Check if audio is available"""
        return AUDIO_AVAILABLE
"""
Event Logging & Database Management
WITH MEDIA PATHS SUPPORT
"""

import sqlite3
import json
import csv
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import threading


class EventLogger:
    """Thread-safe SQLite event logging with media paths support"""
    
    def __init__(self, db_path: str = None):
        # Handle if db_path is passed as dict (from ConfigManager)
        if isinstance(db_path, dict):
            db_path = db_path.get('db_path', 'surveillance_events.db')
        
        self.db_path = db_path or 'surveillance_events.db'
        self._local = threading.local()
        self._init_database()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Thread-local connection"""
        if not hasattr(self._local, 'connection'):
            self._local.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        return self._local.connection
    
    def _init_database(self):
        """Initialize schema - now includes media paths"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Events table - NOW WITH MEDIA PATHS
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                track_id INTEGER NOT NULL,
                zone_name TEXT,
                anomaly_type TEXT NOT NULL,
                severity TEXT CHECK(severity IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')),
                velocity REAL,
                duration REAL,
                direction TEXT,
                metadata TEXT,
                screenshot_path TEXT,
                clip_path TEXT,
                email_sent BOOLEAN DEFAULT 0,
                resolved BOOLEAN DEFAULT 0
            )
        ''')
        
        # Daily statistics - aggregated only
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_stats (
                date DATE PRIMARY KEY,
                total_tracks INTEGER,
                total_anomalies INTEGER,
                avg_velocity REAL,
                peak_hour INTEGER,
                zone_violations TEXT
            )
        ''')
        
        # Performance metrics
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                fps REAL,
                inference_time_ms REAL,
                cpu_percent REAL,
                memory_mb REAL,
                active_tracks INTEGER,
                detection_latency_ms REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def log_event(self, 
                  track_id: int,
                  anomaly_type: str,
                  severity: str,
                  zone: Optional[str] = None,
                  velocity: float = 0.0,
                  duration: float = 0.0,
                  direction: Optional[str] = None,
                  metadata: Optional[Dict] = None,
                  screenshot_path: Optional[str] = None,
                  clip_path: Optional[str] = None,
                  email_sent: bool = False):
        """Log event with media paths"""
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO events 
            (track_id, zone_name, anomaly_type, severity, velocity, duration, 
             direction, metadata, screenshot_path, clip_path, email_sent)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            track_id,
            zone,
            anomaly_type,
            severity,
            velocity,
            duration,
            direction,
            json.dumps(metadata) if metadata else None,
            screenshot_path,
            clip_path,
            email_sent
        ))
        
        conn.commit()
        return cursor.lastrowid
    
    def update_event_clip_path(self, event_id: int, clip_path: str):
        """Update event with video clip path once recording is complete"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE events SET clip_path = ? WHERE id = ?", 
            (clip_path, event_id)
        )
        conn.commit()
    
    def log_performance(self,
                       fps: float,
                       inference_time_ms: float,
                       cpu_percent: float,
                       memory_mb: float,
                       active_tracks: int,
                       detection_latency_ms: float):
        """Log system performance metrics"""
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO performance_logs 
            (fps, inference_time_ms, cpu_percent, memory_mb, active_tracks, detection_latency_ms)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (fps, inference_time_ms, cpu_percent, memory_mb, active_tracks, detection_latency_ms))
        
        conn.commit()
    
    def query_events(self,
                    start_time: Optional[datetime] = None,
                    end_time: Optional[datetime] = None,
                    zone: Optional[str] = None,
                    severity: Optional[str] = None,
                    keyword: Optional[str] = None,
                    has_media: Optional[bool] = None,
                    limit: int = 100) -> List[Dict]:
        """Searchable event query for dashboard - now includes media info"""
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        query = "SELECT * FROM events WHERE 1=1"
        params = []
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())
        if zone:
            query += " AND zone_name = ?"
            params.append(zone)
        if severity:
            query += " AND severity = ?"
            params.append(severity)
        if keyword:
            query += " AND (anomaly_type LIKE ? OR zone_name LIKE ?)"
            like_term = f"%{keyword}%"
            params.extend([like_term, like_term])
        if has_media is not None:
            if has_media:
                query += " AND (screenshot_path IS NOT NULL OR clip_path IS NOT NULL)"
            else:
                query += " AND screenshot_path IS NULL AND clip_path IS NULL"
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        columns = [desc[0] for desc in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        return results
    
    def generate_daily_report(self, report_date: Optional[datetime] = None) -> str:
        """Generate CSV report - includes media summary"""
        
        if report_date is None:
            report_date = datetime.now()
        
        date_str = report_date.strftime('%Y-%m-%d')
        next_date = (report_date + timedelta(days=1)).strftime('%Y-%m-%d')
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Aggregate statistics
        cursor.execute('''
            SELECT 
                COUNT(DISTINCT track_id) as unique_tracks,
                COUNT(*) as total_anomalies,
                AVG(velocity) as avg_velocity,
                zone_name,
                COUNT(*) as zone_count
            FROM events 
            WHERE timestamp >= ? AND timestamp < ?
            GROUP BY zone_name
        ''', (date_str, next_date))
        
        stats = cursor.fetchall()
        
        # Media statistics
        cursor.execute('''
            SELECT 
                COUNT(screenshot_path) as screenshots,
                COUNT(clip_path) as clips
            FROM events 
            WHERE timestamp >= ? AND timestamp < ?
        ''', (date_str, next_date))
        
        media_stats = cursor.fetchone()
        
        # Hourly distribution
        cursor.execute('''
            SELECT strftime('%H', timestamp) as hour, COUNT(*) as count
            FROM events 
            WHERE timestamp >= ? AND timestamp < ?
            GROUP BY hour
            ORDER BY count DESC
            LIMIT 1
        ''', (date_str, next_date))
        
        peak_hour = cursor.fetchone()
        peak_hour_val = int(peak_hour[0]) if peak_hour else 0
        
        # Generate CSV
        report_dir = 'daily_reports'
        os.makedirs(report_dir, exist_ok=True)
        report_path = os.path.join(report_dir, f'report_{date_str}.csv')
        
        with open(report_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            writer.writerow(['Date', date_str])
            writer.writerow(['Total Anomalies', sum(s[1] for s in stats)])
            writer.writerow(['Unique Tracks', stats[0][0] if stats else 0])
            writer.writerow(['Avg Velocity', f"{stats[0][2]:.2f}" if stats and stats[0][2] else 0])
            writer.writerow(['Peak Hour', peak_hour_val])
            writer.writerow(['Screenshots Captured', media_stats[0] if media_stats else 0])
            writer.writerow(['Video Clips Recorded', media_stats[1] if media_stats else 0])
            writer.writerow([])
            writer.writerow(['Zone', 'Violation Count'])
            for stat in stats:
                writer.writerow([stat[3], stat[4]])
        
        conn.close()
        return report_path
    
    def get_performance_summary(self, hours: int = 24) -> Dict:
        """Get performance metrics for dashboard HUD"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        since = (datetime.now() - timedelta(hours=hours)).isoformat()
        
        cursor.execute('''
            SELECT 
                AVG(fps) as avg_fps,
                AVG(inference_time_ms) as avg_inference,
                AVG(cpu_percent) as avg_cpu,
                AVG(memory_mb) as avg_memory,
                MAX(detection_latency_ms) as max_latency
            FROM performance_logs
            WHERE timestamp >= ?
        ''', (since,))
        
        row = cursor.fetchone()
        conn.close()
        
        return {
            'avg_fps': row[0] or 0,
            'avg_inference_ms': row[1] or 0,
            'avg_cpu_percent': row[2] or 0,
            'avg_memory_mb': row[3] or 0,
            'max_latency_ms': row[4] or 0
        }
    
    def purge_old_data(self, days: int = 30):
        """GDPR-compliant data retention"""
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        
        cursor.execute("DELETE FROM events WHERE timestamp < ?", (cutoff,))
        cursor.execute("DELETE FROM performance_logs WHERE timestamp < ?", (cutoff,))
        
        conn.commit()
    
    def close(self):
        """Close database connection"""
        if hasattr(self._local, 'connection'):
            self._local.connection.close()
            delattr(self._local, 'connection')
"""
Database Initialization Script
Run this first to create the surveillance database
"""

import sqlite3
import os

def init_database(db_path="surveillance_events.db"):
    """Initialize SQLite database with all required tables"""
    
    # Remove existing database if corrupted (optional)
    if os.path.exists(db_path):
        print(f"Database {db_path} already exists. Skipping creation.")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Main events table (anonymized metadata only)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS surveillance_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            track_id INTEGER NOT NULL,
            zone_name TEXT,
            anomaly_type TEXT NOT NULL,
            severity INTEGER NOT NULL,
            duration_seconds REAL,
            velocity REAL,
            position_x REAL,
            position_y REAL,
            metadata TEXT
        )
    """)
    
    # Track lifecycle table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS track_lifecycle (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            track_id INTEGER NOT NULL,
            start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
            end_time DATETIME,
            total_duration REAL,
            zones_visited TEXT,
            anomaly_count INTEGER DEFAULT 0
        )
    """)
    
    # Daily statistics
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS daily_stats (
            date DATE PRIMARY KEY,
            total_tracks INTEGER DEFAULT 0,
            total_anomalies INTEGER DEFAULT 0,
            avg_fps REAL,
            avg_inference_time REAL,
            cpu_usage_peak REAL,
            alerts_sent INTEGER DEFAULT 0
        )
    """)
    
    # Performance metrics
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS performance_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            fps REAL,
            inference_ms REAL,
            cpu_percent REAL,
            memory_mb REAL,
            active_tracks INTEGER,
            detection_latency_ms REAL
        )
    """)
    
    # Alert log
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS alert_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            event_id INTEGER,
            alert_type TEXT,
            recipient TEXT,
            status TEXT
        )
    """)
    
    # Create indexes for fast queries
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON surveillance_events(timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_track_id ON surveillance_events(track_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_performance_timestamp ON performance_metrics(timestamp)")
    
    conn.commit()
    conn.close()
    print(f"✓ Database initialized successfully: {db_path}")

if __name__ == "__main__":
    init_database()
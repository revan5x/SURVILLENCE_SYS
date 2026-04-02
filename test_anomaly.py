"""
Test script to verify anomaly detection - CORRECTED
"""

import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from logic.behavior_analyzer import BehaviorAnalyzer, BehaviorProfile, BehaviorType
from logic.anomaly_engine import AnomalyEngine, Severity
from tracking.track_buffer import TrackBuffer, TrackState


class MockTrack:
    """Mock track for testing"""
    def __init__(self, track_id, start_pos=(100, 100), simulate_loitering=True):
        self.track_id = track_id
        self.buffer = TrackBuffer(max_size=30)
        self.start_time = time.time()
        
        if simulate_loitering:
            # Simulate stationary person - VERY SMALL movement
            for i in range(20):
                # Movement of only 0-2 pixels
                jitter_x = i % 2  # 0 or 1
                jitter_y = i % 2  # 0 or 1
                
                self.buffer.add(
                    timestamp=time.time() - (20-i)*0.15,  # 3 seconds total
                    position=(start_pos[0] + jitter_x, start_pos[1] + jitter_y),
                    bbox=(start_pos[0]+jitter_x, start_pos[1]+jitter_y, 
                          start_pos[0]+jitter_x+50, start_pos[1]+jitter_y+100),
                    confidence=0.8
                )
        else:
            # Simulate movement
            for i in range(10):
                self.buffer.add(
                    timestamp=time.time() - (10-i)*0.1,
                    position=(start_pos[0] + i*20, start_pos[1]),  # Moving fast
                    bbox=(start_pos[0]+i*20, start_pos[1], 
                          start_pos[0]+i*20+50, start_pos[1]+100),
                    confidence=0.8
        )
        
        self.buffer.state = TrackState.TRACKED


def test_loitering_detection():
    print("="*60)
    print("TEST 1: Loitering Detection")
    print("="*60)
    
    analyzer = BehaviorAnalyzer(
        velocity_window=5,
        stationary_threshold=5.0,
        loitering_time=2.0
    )
    
    anomaly_engine = AnomalyEngine(
        persistence_time=2.0,
        alert_cooldown=30.0
    )
    
    # Create stationary track
    track = MockTrack(1, start_pos=(100, 100), simulate_loitering=True)
    
    print(f"\nTrack created:")
    print(f"  Observations: {len(track.buffer.observations)}")
    print(f"  Total dwell: {track.buffer.get_dwell_time():.2f}s")
    print(f"  Last position: {track.buffer.get_last_position()}")
    
    # Check positions range
    positions = [obs.position for obs in track.buffer.observations]
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    print(f"  X range: {min(xs)}-{max(xs)} (span: {max(xs)-min(xs)}px)")
    print(f"  Y range: {min(ys)}-{max(ys)} (span: {max(ys)-min(ys)}px)")
    
    # Velocity check
    vx, vy = track.buffer.get_velocity()
    speed = (vx**2 + vy**2)**0.5
    print(f"  Speed: {speed:.2f} px/s")
    
    print(f"\nRunning analysis...")
    print("-"*60)
    
    anomaly_found = False
    start_test = time.time()
    
    for i in range(25):
        current_time = time.time()
        elapsed = current_time - start_test
        
        profile = analyzer.analyze(track, zone_manager=None)
        event = anomaly_engine.evaluate(profile)
        
        status = "NORMAL"
        if event:
            status = f"🚨 ANOMALY: {event.anomaly_type}"
            anomaly_found = True
        elif profile.behavior_type != BehaviorType.NORMAL:
            status = f"DETECTING: {profile.behavior_type.value}"
        
        print(f"[{elapsed:4.1f}s] {status:25s} "
              f"stationary={profile.stationary_duration:.1f}s "
              f"conf={profile.confidence:.2f}")
        
        if event:
            print(f"\n{'='*60}")
            print("✅ ANOMALY CONFIRMED!")
            print(f"   Type: {event.anomaly_type}")
            print(f"   Severity: {event.severity.value}")
            print(f"   Duration: {event.duration:.2f}s")
            print(f"{'='*60}")
            break
        
        time.sleep(0.2)
    
    if not anomaly_found:
        print(f"\n{'='*60}")
        print("❌ LOITERING NOT DETECTED")
        print("Issues:")
        print(f"  - X movement span: {max(xs)-min(xs)}px (threshold: 5px)")
        print(f"  - Y movement span: {max(ys)-min(ys)}px (threshold: 5px)")
        print(f"  - Need both <= 5px for loitering")
        print(f"{'='*60}")
    
    return anomaly_found


def test_zone_violation():
    print("\n" + "="*60)
    print("TEST 2: Zone Violation Detection")
    print("="*60)
    
    from logic.zone_manager import ZoneManager, ZoneConfig
    
    # Create zone
    zone_mgr = ZoneManager()
    zone_mgr.add_zone(ZoneConfig(
        name="Restricted_Area",
        polygon=[(50, 50), (300, 50), (300, 300), (50, 300)],
        is_restricted=True,
        allowed_directions=['N', 'S']
    ))
    
    print(f"\nZone: Restricted_Area (50,50) to (300,300)")
    
    # Test point at center
    test_pos = (150, 150)
    print(f"Test position: {test_pos}")
    
    # Manual point-in-polygon check
    in_zone = zone_mgr.check_point_in_zone(test_pos, "Restricted_Area")
    print(f"Point in zone: {in_zone}")
    
    # Create track with horizontal movement (wrong direction)
    track = MockTrack(2, start_pos=test_pos, simulate_loitering=False)
    
    # Add more horizontal movement
    for i in range(5):
        track.buffer.add(
            timestamp=time.time() - (5-i)*0.1,
            position=(test_pos[0] + i*15, test_pos[1]),  # Moving EAST
            bbox=(test_pos[0]+i*15, test_pos[1], 
                  test_pos[0]+i*15+50, test_pos[1]+100),
            confidence=0.8
        )
    
    print(f"\nTrack movement: EAST (horizontal)")
    print(f"Zone allows: NORTH, SOUTH only")
    
    vx, vy = track.buffer.get_velocity()
    direction = 'E' if vx > 0 else 'W' if vx < 0 else 'N' if vy < 0 else 'S' if vy > 0 else None
    print(f"Calculated direction: {direction}")
    
    print(f"\nAnalyzing...")
    analyzer = BehaviorAnalyzer()
    profile = analyzer.analyze(track, zone_mgr)
    
    print(f"\nResults:")
    print(f"  Behavior: {profile.behavior_type.value}")
    print(f"  Zone violations: {len(profile.zone_violations)}")
    for v in profile.zone_violations:
        print(f"    - {v[0]}: {v[1]}")
    print(f"  Confidence: {profile.confidence:.2f}")
    
    if profile.zone_violations:
        print(f"\n✅ ZONE VIOLATION DETECTED!")
        return True
    else:
        print(f"\n❌ NO ZONE VIOLATION")
        return False


if __name__ == "__main__":
    print("🔒 AI SURVEILLANCE SYSTEM - ANOMALY DETECTION TEST")
    print("Python:", sys.version)
    
    test1 = test_loitering_detection()
    test2 = test_zone_violation()
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Loitering Detection: {'✅ PASS' if test1 else '❌ FAIL'}")
    print(f"Zone Violation:      {'✅ PASS' if test2 else '❌ FAIL'}")
    
    if test1 and test2:
        print(f"\n🎉 ALL TESTS PASSED!")
    else:
        print(f"\n⚠️  SOME TESTS FAILED")
    print("="*60)
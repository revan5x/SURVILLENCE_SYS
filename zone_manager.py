"""
Zone Management System
Polygon-based spatial analysis for surveillance system
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class ZoneConfig:
    """Zone configuration"""
    name: str
    polygon: List[Tuple[int, int]]  # List of (x, y) points
    is_restricted: bool = False
    allowed_directions: List[str] = None  # ['N', 'S', 'E', 'W']
    max_dwell_time: float = 60.0  # Seconds
    
    def __post_init__(self):
        if self.allowed_directions is None:
            self.allowed_directions = ['N', 'S', 'E', 'W']


class ZoneManager:
    """
    Manages detection zones and spatial queries
    Supports polygon zones with configurable rules
    """
    
    def __init__(self):
        self.zones: Dict[str, ZoneConfig] = {}
    
    def add_zone(self, config: ZoneConfig):
        """Add or update zone"""
        self.zones[config.name] = config
    
    def remove_zone(self, name: str):
        """Remove zone"""
        if name in self.zones:
            del self.zones[name]
    
    def check_point_in_zone(self, 
                           point: Tuple[int, int], 
                           zone_name: str) -> bool:
        """Check if point is inside zone polygon"""
        if zone_name not in self.zones:
            return False
        
        polygon = self.zones[zone_name].polygon
        return self._point_in_polygon(point, polygon)
    
    def get_zone_for_point(self, 
                          point: Tuple[int, int]) -> Optional[str]:
        """Find which zone contains the point"""
        for name, zone in self.zones.items():
            if self._point_in_polygon(point, zone.polygon):
                return name
        return None
    
    def check_violation(self, 
                       point: Tuple[int, int],
                       direction: Optional[str],
                       velocity: float,
                       dwell_time: float) -> List[Tuple[str, str]]:
        """
        Check zone violations
        Returns: List of (zone_name, violation_type)
        """
        violations = []
        
        for name, zone in self.zones.items():
            if not self._point_in_polygon(point, zone.polygon):
                continue
            
            # Check restricted zone
            if zone.is_restricted:
                violations.append((name, 'RESTRICTED_ZONE_ENTRY'))
            
            # Check direction violation
            if direction and direction not in zone.allowed_directions:
                violations.append((name, 'WRONG_DIRECTION'))
            
            # Check dwell time
            if dwell_time > zone.max_dwell_time:
                violations.append((name, 'EXCESSIVE_DWELL'))
        
        return violations
    
    def _point_in_polygon(self, 
                         point: Tuple[int, int], 
                         polygon: List[Tuple[int, int]]) -> bool:
        """Ray casting algorithm for point-in-polygon"""
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def get_zone_color(self, zone_name: str) -> Tuple[int, int, int]:
        """Get visualization color for zone"""
        if zone_name not in self.zones:
            return (128, 128, 128)
        
        zone = self.zones[zone_name]
        if zone.is_restricted:
            return (0, 0, 255)  # Red for restricted
        return (0, 255, 0)  # Green for normal
    
    def draw_zones(self, frame: np.ndarray) -> np.ndarray:
        """Draw all zones on frame"""
        import cv2
        result = frame.copy()
        
        for name, zone in self.zones.items():
            color = self.get_zone_color(name)
            polygon = np.array(zone.polygon, np.int32)
            polygon = polygon.reshape((-1, 1, 2))
            
            # Draw filled with transparency
            overlay = result.copy()
            cv2.fillPoly(overlay, [polygon], color)
            result = cv2.addWeighted(result, 1.0, overlay, 0.3, 0)
            
            # Draw border
            cv2.polylines(result, [polygon], True, color, 2)
            
            # Draw label
            centroid = np.mean(zone.polygon, axis=0).astype(int)
            cv2.putText(result, name, tuple(centroid), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return result
import cv2
import numpy as np
from typing import List, Dict
from logic.behavior_analyzer import BehaviorType


class Visualizer:

    def __init__(self):
        pass

    def apply_privacy_blur(self, frame, detections):
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            face_region = frame[y1:y2, x1:x2]
            if face_region.size > 0:
                face_region = cv2.GaussianBlur(face_region, (35, 35), 30)
                frame[y1:y2, x1:x2] = face_region
        return frame

    def draw_tracks(self, frame, tracks, behavior_profiles: Dict):

        for track in tracks:
            x1, y1, x2, y2 = map(int, track.bbox)
            tid = track.track_id

            behavior = behavior_profiles.get(tid, None)

            color = self._get_behavior_color(behavior.behavior if behavior else BehaviorType.NORMAL)

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"ID:{tid}"

            # Label background (dark)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), (30, 30, 40), -1)

            # Text
            cv2.putText(frame, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return frame

    def _get_behavior_color(self, behavior: BehaviorType):

        if behavior == BehaviorType.NORMAL:
            return (0, 195, 255)   # Blue

        elif behavior == BehaviorType.LOITERING:
            return (127, 0, 255)   # Purple

        elif behavior == BehaviorType.WRONG_DIRECTION:
            return (127, 0, 255)

        elif behavior == BehaviorType.RAPID_MOVEMENT:
            return (127, 0, 255)

        else:
            return (255, 53, 94)   # Red

    def draw_hud(self, frame, fps, cpu, memory, tracks, inference, latency):

        overlay = frame.copy()

        # Background panel
        cv2.rectangle(overlay, (10, 10), (320, 130), (20, 20, 30), -1)
        frame = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)

        color = (0, 195, 255)  # Blue

        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.putText(frame, f"Tracks: {tracks}", (20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.putText(frame, f"CPU: {cpu:.1f}%", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.putText(frame, f"MEM: {memory:.1f}MB", (20, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return frame

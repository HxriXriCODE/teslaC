import cv2
import numpy as np
import config

class TeslaHUDVisualizer:
    """Overlay generator mimicking Tesla Autopilot UI aesthetics."""

    @staticmethod
    def draw_hud(frame: np.ndarray, detections: list, fps: float) -> np.ndarray:
        overlay = frame.copy()
        h, w, _ = frame.shape

        # 1. Draw Bounding Boxes and Ground Polygons
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cls_id = det["class_id"]
            label = det["label"]
            dist = det["distance_m"]
            
            color = config.CLASS_COLORS.get(cls_id, (255, 255, 255))

            # Ground projection polygon (Tesla-style object floor footprint)
            pts = np.array([[x1, y2], [x2, y2], [x2 - 5, y2 + 5], [x1 + 5, y2 + 5]], np.int32)
            cv2.fillPoly(overlay, [pts], color)

            # Modern corner bounding box
            line_len = int(min(x2 - x1, y2 - y1) * 0.2)
            thickness = 2
            
            # Top-Left Corner
            cv2.line(frame, (x1, y1), (x1 + line_len, y1), color, thickness)
            cv2.line(frame, (x1, y1), (x1, y1 + line_len), color, thickness)
            # Bottom-Right Corner
            cv2.line(frame, (x2, y2), (x2 - line_len, y2), color, thickness)
            cv2.line(frame, (x2, y2), (x2, y2 - line_len), color, thickness)

            # Metadata Tag
            tag = f"{label.upper()} | {dist}m"
            cv2.putText(frame, tag, (x1, max(y1 - 8, 15)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # 2. Blend Polygon Overlay
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

        # 3. Top Telemetry Banner
        cv2.rectangle(frame, (0, 0), (w, 40), (10, 10, 10), -1)
        telemetry_text = f"TESLA AUTOPILOT CLONE  |  FPS: {fps:.1f}  |  TRACKED OBJECTS: {len(detections)}"
        cv2.putText(frame, telemetry_text, (20, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2, cv2.LINE_AA)

        return frame

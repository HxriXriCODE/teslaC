import cv2
import numpy as np
from ultralytics import YOLO
import config

class AutopilotPerception:
    """Core Object Detection and Distance Estimation Engine."""
    
    def __init__(self, model_path: str = config.MODEL_WEIGHTS):
        self.model = YOLO(model_path)
        self.target_classes = config.TARGET_CLASSES

    def estimate_distance(self, bbox_height: int, image_height: int) -> float:
        """
        Monocular Distance Estimation heuristic.
        d = (focal_length * real_height) / pixel_height
        """
        # Approximated camera calibration parameter for a standard 1080p dashcam
        focal_length_px = image_height * 1.2
        real_object_height_m = 1.5  # Average car/person height approximation
        
        if bbox_height <= 0:
            return 0.0
        
        distance = (focal_length_px * real_object_height_m) / bbox_height
        return round(distance, 1)

    def process_frame(self, frame: np.ndarray):
        """Runs object detection and returns parsed predictions."""
        results = self.model.predict(
            source=frame,
            conf=config.CONFIDENCE_THRESHOLD,
            iou=config.NMS_IOU_THRESHOLD,
            classes=self.target_classes,
            verbose=False
        )[0]

        detections = []
        img_h, img_w, _ = frame.shape

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            
            bbox_h = y2 - y1
            dist = self.estimate_distance(bbox_h, img_h)

            detections.append({
                "bbox": (x1, y1, x2, y2),
                "class_id": cls_id,
                "label": self.model.names[cls_id],
                "confidence": conf,
                "distance_m": dist
            })

        return detections

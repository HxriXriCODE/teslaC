import os

# Base Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_VIDEO_PATH = os.path.join(BASE_DIR, "data", "dashcam_input.mp4")
OUTPUT_VIDEO_PATH = os.path.join(BASE_DIR, "output", "autopilot_annotated.mp4")

# Model Settings
MODEL_WEIGHTS = "yolov8n.pt"  # Nano model for real-time edge processing
CONFIDENCE_THRESHOLD = 0.40
NMS_IOU_THRESHOLD = 0.45

# Target Autonomous Driving Classes (COCO Indices)
# 0: person, 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck, 9: traffic light, 11: stop sign
TARGET_CLASSES = [0, 1, 2, 3, 5, 7, 9, 11]

# Visual Customizations (BGR Format)
CLASS_COLORS = {
    0: (0, 255, 255),    # Pedestrian - Yellow
    1: (255, 165, 0),    # Bicycle - Orange
    2: (0, 255, 0),      # Car - Neon Green
    3: (255, 100, 0),    # Motorcycle - Deep Orange
    5: (255, 0, 255),    # Bus - Magenta
    7: (255, 0, 128),    # Truck - Purple
    9: (0, 0, 255),      # Traffic Light - Red
    11: (0, 0, 255)      # Stop Sign - Red
}

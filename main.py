import cv2
import time
import os
import config
from detector import AutopilotPerception
from visualizer import TeslaHUDVisualizer

def run_pipeline():
    # Initialize Core Modules
    detector = AutopilotPerception()
    
    if not os.path.exists(config.INPUT_VIDEO_PATH):
        print(f"[Error] Input video not found at: {config.INPUT_VIDEO_PATH}")
        return

    cap = cv2.VideoCapture(config.INPUT_VIDEO_PATH)
    
    # Video Writer Initialization
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    
    os.makedirs(os.path.dirname(config.OUTPUT_VIDEO_PATH), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(config.OUTPUT_VIDEO_PATH, fourcc, fps_in, (width, height))

    prev_time = time.time()

    print("[INFO] Executing Tesla Autopilot Perception Pipeline...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Calculate FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time

        # Inference & Visualization
        detections = detector.process_frame(frame)
        annotated_frame = TeslaHUDVisualizer.draw_hud(frame, detections, fps)

        # Write & Display
        out.write(annotated_frame)
        cv2.imshow("Tesla Autopilot - Perception System", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"[SUCCESS] Pipeline complete. Video saved to {config.OUTPUT_VIDEO_PATH}")

if __name__ == "__main__":
    run_pipeline()

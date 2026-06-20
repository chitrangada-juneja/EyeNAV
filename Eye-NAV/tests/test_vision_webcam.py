import cv2
import time
import sys
import os

# Ensure project root is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.io.camera import CameraSource
from src.perception.detection import ObjectDetector

def test_webcam_detection():
    print("--- Starting Webcam Test (Source=0) ---")
    
    try:
        # 0 = Default Webcam
        camera = CameraSource(source=0)
        
        # Load YOLO - Using the same Large model
        detector = ObjectDetector(conf_threshold=0.4, output_path="data/states/vision_state.json")
    except Exception as e:
        print(f"Failed to initialize: {e}")
        return

    print("Press 'q' to quit.")
    
    prev_time = time.time()
    fps_avg = 0
    
    while True:
        # OPTIMIZATION: Clear buffer to get LATEST frame (Low Latency)
        # If processing is slow (5 FPS), the camera buffer (30 FPS) fills up.
        # We must discard old frames to see "now", not "2 seconds ago".
        # This is CRITICAL for real-time feel.
        if not camera.is_file:
             # Grab multiple times to flush buffer (simple hack)
             # Better way involves a separate thread, but this works for simple loop
             for _ in range(4): 
                 camera.cap.grab()
        
        frame = camera.get_frame()
        if frame is None:
            break
            
        # Inference
        results = detector.detect(frame, visualize=True)
        
        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        fps_avg = 0.9 * fps_avg + 0.1 * fps
        
        print(f"\rFPS: {fps_avg:.1f}  Objects: {len(results['detections'])}", end="")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_webcam_detection()

import os
import cv2
import time
import json
import sys
import numpy as np

# Ensure project root is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.io.camera import CameraSource
from src.perception.detection import ObjectDetector
from src.perception.path_analyzer import PathAnalyzer

def test_video_detection():
    # 1. Setup Logic
    # Robust path finding (handle running from root or tests/)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    video_path = os.path.join(project_root, "data", "validation_videos", "footage_4.mp4")
    if not os.path.exists(video_path):
        # Fallback for relative data/ path
        video_path = "data/validation_videos/footage_4.mp4"
        if not os.path.exists(video_path):
            print(f"Error: Video not found at {video_path}")
            return

    print(f"--- Starting Video Test [DEBUG MODE]: {video_path} ---")
    
    # Initialize Modules
    try:
        # Enable Real-Time Mode
        camera = CameraSource(source=video_path, realtime_mode=True)
        # Using advanced PathAnalyzer with recommended settings
        analyzer = PathAnalyzer(use_com=True, use_weighted_bias=True, use_bev=False)
        detector = ObjectDetector(conf_threshold=0.25, output_path="data/states/test_vision_state.json")
    except Exception as e:
        print(f"Failed to initialize modules: {e}")
        return

    frame_count = 0
    start_time = time.time()

    # 2. Main Loop
    while True:
        frame = camera.get_frame(block=True)
        if frame is None:
            break
            
        frame_count += 1
        
        # Run Detection
        results = detector.detect(frame, visualize=True)
        detections = results['detections']
        masks = results['masks']
        vis_frame = results.get('vis_frame', frame.copy())
        
        # Run Path Analysis
        if masks is not None:
            # Map masks to labels for analyzer
            mask_labels = [None] * len(masks)
            for det in detections:
                if det.get('mask_idx') is not None:
                    mask_labels[det['mask_idx']] = det['label']
            
            instruction = analyzer.get_instruction(masks, mask_labels)
            
            # Overlay instruction on main frame
            cv2.putText(vis_frame, instruction, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Overlay Path Information (NEW)
            if analyzer.use_com:
                midline = analyzer.last_midline
                if midline and len(midline) > 1:
                    pts = np.array(midline, dtype=np.int32)
                    cv2.polylines(vis_frame, [pts], False, (0, 255, 0), 2)
                
                com = analyzer.last_com
                if com:
                    cv2.circle(vis_frame, com, 10, (255, 0, 0), -1)
            else:
                # Overlay Grid lines
                grid = getattr(analyzer, 'last_grid', None)
                if grid:
                    h, w = vis_frame.shape[:2]
                    for gy in grid['rows']:
                        cv2.line(vis_frame, (0, gy), (w, gy), (0, 255, 0), 1)
                    for gx in grid['cols']:
                        cv2.line(vis_frame, (gx, 0), (gx, h), (0, 255, 0), 1)

            # Display Debug Window (CoM, Padding, BEV)
            if analyzer.last_debug_frame is not None:
                cv2.imshow("Path Analysis [CoM + BEV Debug]", analyzer.last_debug_frame)

        # Show main vision loop
        cv2.imshow("Main Vision Loop", vis_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if frame_count % 30 == 0:
            print(f"FPS: {frame_count / (time.time() - start_time):.1f}")

    camera.release()
    cv2.destroyAllWindows()
    print(f"--- Test Complete ---")

if __name__ == "__main__":
    test_video_detection()

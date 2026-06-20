import cv2
import time
import torch
from ultralytics import YOLO
import pandas as pd
import os
import gc

# --- Configuration ---
# You can choose different sizes of the YOLO-World model
# v2_s (small), v2_m (medium), v2_l (large), v2_x (extra-large)
# Start with 's' for the best speed.
MODEL_WEIGHT = "yolov8s-worldv2.pt"

# Define the custom classes (prompts) you want to detect!
# This is the core feature of YOLO-World.
PROMPTS = [ "person", "bicycle", "car", "motorcycle", "bus", "truck",
    "traffic light", "stop sign", "fire hydrant", "bench",
    "pothole", "stairs", "traffic cone", "trash can","walkable path", "pothole", "rough ground", "stone", "stairs", "road","sidwalk","tree"
]

VIDEO_SOURCE_DIR = "../../../data/validation_videos/"
OUTPUT_DIR = "results/"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_VIDEO = False   # Set to True to save annotated video
SHOW_VISUALIZATION = False  # Set to True to see the model in action during benchmark
MAX_WINDOW_WIDTH = 1280

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def evaluate_yolo_world_on_video(model, video_path):
    """Runs YOLO-World on a single video and returns performance metrics."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None

    frame_count = 0
    total_inference_time = 0
    video_start_time = time.perf_counter()

    # Create a video writer to save the output with bounding boxes
    out = None
    if SAVE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video_path = os.path.join(OUTPUT_DIR, f"yoloworld_{os.path.basename(video_path)}")
        out = cv2.VideoWriter(out_video_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        # 3. Perform inference
        inference_start_time = time.perf_counter()
        results = model.predict(frame, conf=0.25, verbose=False, device=DEVICE)
        inference_end_time = time.perf_counter()
        
        total_inference_time += (inference_end_time - inference_start_time)

        # 4. Draw bounding boxes on the frame and save it (optional)
        if (SAVE_VIDEO or SHOW_VISUALIZATION):
            # Show both boxes and masks if the model supports segmentation
            annotated_frame = results[0].plot(boxes=True, masks=True)
            
            if SAVE_VIDEO and out is not None:
                out.write(annotated_frame)
            
            if SHOW_VISUALIZATION:
                display_frame = annotated_frame
                h, w = annotated_frame.shape[:2]
                if w > MAX_WINDOW_WIDTH:
                    scale = MAX_WINDOW_WIDTH / w
                    display_frame = cv2.resize(annotated_frame, (MAX_WINDOW_WIDTH, int(h * scale)))
                
                cv2.imshow("YOLO-World Benchmark", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    video_end_time = time.perf_counter()
    total_video_processing_time = video_end_time - video_start_time
    
    cap.release()
    if out is not None:
        out.release()
    
    if SHOW_VISUALIZATION:
        cv2.destroyAllWindows()

    avg_latency_ms = (total_inference_time / frame_count) * 1000
    fps = frame_count / total_video_processing_time
    
    print(f"Finished processing. FPS: {fps:.2f}, Avg Latency: {avg_latency_ms:.2f} ms")
    
    return {
        "model_name": "YOLO-World-S",
        "video_name": os.path.basename(video_path),
        "fps": fps,
        "avg_latency_ms": avg_latency_ms,
        "device": DEVICE
    }

def main():
    """Main function to run the benchmark."""
    all_results = []
    video_files = [os.path.join(VIDEO_SOURCE_DIR, f) for f in os.listdir(VIDEO_SOURCE_DIR) if f.endswith('.mp4')]

    # 1. Load the YOLO-World model once
    print(f"--- Loading model: {MODEL_WEIGHT} ---")
    model = YOLO(MODEL_WEIGHT).to(DEVICE)
    
    # 2. Set the custom classes using your text prompts
    model.set_classes(PROMPTS)
    
    # --- Warm-up ---
    print(f"Warm-up {MODEL_WEIGHT}...")
    dummy_input = torch.zeros((1, 3, 640, 640)).to(DEVICE)
    for _ in range(10):
        model.predict(dummy_input, verbose=False)
    
    print(f"Set custom classes: {PROMPTS}")

    for video_path in video_files:
        print(f"\nEvaluating YOLO-World on {os.path.basename(video_path)}...")
        result = evaluate_yolo_world_on_video(model, video_path)
        if result:
            all_results.append(result)

    # --- Memory cleanup ---
    print("Cleaning up...")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    if all_results:
        results_df = pd.DataFrame(all_results)
        results_csv_path = os.path.join(OUTPUT_DIR, "yoloworld_results.csv")
        results_df.to_csv(results_csv_path, index=False)
        
        print(f"\n--- YOLO-World Benchmark Complete ---")
        print(f"Results saved to {results_csv_path}")
        print(results_df)

if __name__ == "__main__":
    main()
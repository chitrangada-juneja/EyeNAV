import cv2
import time
import torch
from ultralytics import YOLO, RTDETR
import pandas as pd
import os
import gc

# --- Configuration ---
MODELS_TO_TEST = {
    "YOLOv8n": "yolov8n.pt",
    "YOLOv9c": "yolov9c.pt",
    "YOLOv10n": "yolov10n.pt",
    "YOLOv26n": "yolo26n.pt",
    #"YOLO-NAS-S": "yolo_nas_s.pt", 
    "RT-DETR-L": "rtdetr-l.pt"
}

VIDEO_SOURCE_DIR = "../../../data/validation_videos/"
OUTPUT_DIR = "results/"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SHOW_VISUALIZATION = False  # Set to True to see the model in action
SAVE_VIDEO = False          # Set to True to save annotated videos
MAX_WINDOW_WIDTH = 1280

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_model(model_name, weight_file):
    """Loads a model based on its name."""
    print(f"--- Loading model: {model_name} ---")
    if "RT-DETR" in model_name:
        return RTDETR(weight_file).to(DEVICE)
    # elif "NAS" in model_name:
    #     return YOLO(weight_file).to(DEVICE)
    else:
        # The YOLO class from ultralytics can handle most YOLO versions
        model = YOLO(weight_file).to(DEVICE)
    
    # --- Warm-up ---
    print(f"Warm-up {model_name}...")
    dummy_input = torch.zeros((1, 3, 640, 640)).to(DEVICE)
    for _ in range(10):
        model(dummy_input, verbose=False)
    
    return model

def evaluate_model_on_video(model, model_name, video_path):
    """Runs a model on a single video and returns performance metrics."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None

    frame_count = 0
    total_inference_time = 0
    
    video_start_time = time.perf_counter()

    # Create a video writer (optional)
    out = None
    if SAVE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video_path = os.path.join(OUTPUT_DIR, f"{model_name}_{os.path.basename(video_path)}")
        out = cv2.VideoWriter(out_video_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        # 1. Measure Latency (Inference Time)
        inference_start_time = time.perf_counter()
        results = model(frame, verbose=False) # verbose=False to keep the console clean
        inference_end_time = time.perf_counter()
        
        total_inference_time += (inference_end_time - inference_start_time)

        # Optional: Save/Show visualization
        if SAVE_VIDEO or SHOW_VISUALIZATION:
            annotated_frame = results[0].plot()
            
            if SAVE_VIDEO and out is not None:
                out.write(annotated_frame)
            
            if SHOW_VISUALIZATION:
                display_frame = annotated_frame
                h, w = annotated_frame.shape[:2]
                if w > MAX_WINDOW_WIDTH:
                    scale = MAX_WINDOW_WIDTH / w
                    display_frame = cv2.resize(annotated_frame, (MAX_WINDOW_WIDTH, int(h * scale)))
                
                cv2.imshow(f"Benchmarking: {model_name}", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


    video_end_time = time.perf_counter()
    total_video_processing_time = video_end_time - video_start_time
    
    cap.release()
    if out is not None:
        out.release()
    if SHOW_VISUALIZATION:
        cv2.destroyAllWindows()

    # 2. Calculate Metrics
    avg_latency_ms = (total_inference_time / frame_count) * 1000  # in milliseconds
    fps = frame_count / total_video_processing_time
    
    print(f"Finished processing. FPS: {fps:.2f}, Avg Latency: {avg_latency_ms:.2f} ms")
    
    return {
        "model_name": model_name,
        "video_name": os.path.basename(video_path),
        "fps": fps,
        "avg_latency_ms": avg_latency_ms,
        "device": DEVICE
    }

def main():
    """Main function to run the benchmark."""
    all_results = []
    
    video_files = [os.path.join(VIDEO_SOURCE_DIR, f) for f in os.listdir(VIDEO_SOURCE_DIR) if f.endswith('.mp4')]

    for model_name, weight_file in MODELS_TO_TEST.items():
        try:
            model = load_model(model_name, weight_file)
            for video_path in video_files:
                print(f"\nEvaluating {model_name} on {os.path.basename(video_path)}...")
                result = evaluate_model_on_video(model, model_name, video_path)
                if result:
                    all_results.append(result)
            
            # --- Better memory management ---
            print(f"Cleaning up after {model_name}...")
            del model
            gc.collect() # Force garbage collection
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize() # Wait for kernels to finish

        except Exception as e:
            print(f"Could not load or run model {model_name}. Error: {e}")

    # Save results to a CSV file
    results_df = pd.DataFrame(all_results)
    results_csv_path = os.path.join(OUTPUT_DIR, "benchmark_results.csv")
    results_df.to_csv(results_csv_path, index=False)
    
    print(f"\n--- Benchmark Complete ---")
    print(f"Results saved to {results_csv_path}")
    print(results_df)

if __name__ == "__main__":
    main()
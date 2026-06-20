import cv2
import time
import torch
from ultralytics import YOLO
import pandas as pd
import os
import gc

# --- Configuration ---
MODELS_TO_TEST = {
    "YOLOv8n-Seg": "yolov8n-seg.pt",
    "YOLOv11n-Seg": "yolo11n-seg.pt"
}

VIDEO_SOURCE_DIR = "../../../data/validation_videos/"
OUTPUT_DIR = "results/"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SHOW_VISUALIZATION = False
SAVE_VIDEO = False
MAX_WINDOW_WIDTH = 1280

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def evaluate_segmentation_on_video(model, model_name, video_path):
    """Runs a segmentation model on a single video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None

    frame_count = 0
    total_inference_time = 0
    video_start_time = time.perf_counter()
    
    # Create a video writer to save the output with bounding boxes/masks
    out = None
    if SAVE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video_path = os.path.join(OUTPUT_DIR, f"seg_{model_name}_{os.path.basename(video_path)}")
        out = cv2.VideoWriter(out_video_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        # Inference
        inference_start_time = time.perf_counter()
        # task='segment' is explicit for segmentation models
        results = model.predict(frame, verbose=False, device=DEVICE)
        inference_end_time = time.perf_counter()
        
        total_inference_time += (inference_end_time - inference_start_time)

        # Draw masks/boxes and save
        if SAVE_VIDEO or SHOW_VISUALIZATION:
            # boxes=True is default, but we can be explicit
            annotated_frame = results[0].plot(boxes=True, masks=True)
            
            if SAVE_VIDEO and out is not None:
                out.write(annotated_frame)
            
            if SHOW_VISUALIZATION:
                display_frame = annotated_frame
                h, w = annotated_frame.shape[:2]
                if w > MAX_WINDOW_WIDTH:
                    scale = MAX_WINDOW_WIDTH / w
                    display_frame = cv2.resize(annotated_frame, (MAX_WINDOW_WIDTH, int(h * scale)))
                
                cv2.imshow(f"Segmentation: {model_name}", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    video_end_time = time.perf_counter()
    total_video_processing_time = video_end_time - video_start_time
    
    cap.release()

    avg_latency_ms = (total_inference_time / frame_count) * 1000
    fps = frame_count / total_video_processing_time
    
    print(f"[{model_name}] FPS: {fps:.2f}, Avg Latency: {avg_latency_ms:.2f} ms")
    
    return {
        "model_name": model_name,
        "video_name": os.path.basename(video_path),
        "fps": fps,
        "avg_latency_ms": avg_latency_ms,
        "device": DEVICE,
        "task": "segmentation"
    }

def main():
    all_results = []
    video_files = [os.path.join(VIDEO_SOURCE_DIR, f) for f in os.listdir(VIDEO_SOURCE_DIR) if f.endswith('.mp4')]

    for model_name, weight_file in MODELS_TO_TEST.items():
        try:
            print(f"--- Loading model: {model_name} ---")
            model = YOLO(weight_file).to(DEVICE)
            
            # Warm-up
            print(f"Warm-up {model_name}...")
            dummy_input = torch.zeros((1, 3, 640, 640)).to(DEVICE)
            for _ in range(5):
                model.predict(dummy_input, verbose=False)

            for video_path in video_files:
                print(f"Evaluating {model_name} on {os.path.basename(video_path)}...")
                result = evaluate_segmentation_on_video(model, model_name, video_path)
                if result:
                    all_results.append(result)
            
            # Memory cleanup
            print(f"Cleaning up {model_name}...")
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        except Exception as e:
            print(f"Error with {model_name}: {e}")

    if all_results:
        results_df = pd.DataFrame(all_results)
        results_csv_path = os.path.join(OUTPUT_DIR, "segmentation_results.csv")
        results_df.to_csv(results_csv_path, index=False)
        print(f"\nResults saved to {results_csv_path}")
        print(results_df)

if __name__ == "__main__":
    main()

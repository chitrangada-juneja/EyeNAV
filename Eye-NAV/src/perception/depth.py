import cv2
import torch
import numpy as np
import os
import time
import json
import warnings
from PIL import Image
from accelerate import Accelerator

# Suppress annoying deprecation and Triton warnings
warnings.filterwarnings("ignore")

class DepthEstimator:
    """
    Wrapper for Metric3D v2 (ViT-Small).
    Provides high-speed metric monocular depth estimation.
    """
    def __init__(self, model_id='yvanyin/metric3d', output_path="data/states/depth_state.json", depth_scale=1.0, is_metric=True):
        """Initialize the model."""
        # Using Accelerator to stay consistent with your original code
        self.device = Accelerator().device
        print(f"Loading Metric3D model: {model_id} on {self.device}...")
        
        # Load Metric3D via Torch Hub
        self.model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_small', pretrain=True)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.output_path = output_path
        self.depth_scale = depth_scale
        self.is_metric = is_metric # Metric3D outputs actual meters
        
        self.depth_dir = os.path.join(os.path.dirname(os.path.dirname(output_path)), "frames", "depth")
        os.makedirs(self.depth_dir, exist_ok=True)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        self.frame_history = []
        self.max_history = 5

    def estimate_depth(self, frame):
        """
        Estimate depth for the given frame using Metric3D.
        Returns a numpy array representing depth in meters.
        """
        h, w = frame.shape[:2]
        
        # 1. Pre-process: BGR to RGB and Normalize
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = torch.from_numpy(rgb_frame).float() / 255.0
        
        # 2. Format: [Batch, Channel, Height, Width] and move to GPU
        input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        # 3. Inference with AutoCast (only on GPU)
        with torch.no_grad():
            if self.device.type == 'cuda':
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    # Metric3D expects a dict input
                    pred_depth, confidence, output_dict = self.model.inference({'input': input_tensor})
            else:
                # CPU: Float32 is safer and avoids xformers float16 operators
                pred_depth, confidence, output_dict = self.model.inference({'input': input_tensor})
        
        # 4. Post-process: Extract and scale back to original resolution
        depth_map = pred_depth.squeeze().cpu().float().numpy()
        
        # Resize to original frame dimensions
        if depth_map.shape[0] != h or depth_map.shape[1] != w:
            depth_map = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Apply scaling if necessary
        if self.depth_scale != 1.0:
            depth_map *= self.depth_scale
            
        return depth_map

    def save_state(self, depth_map, timestamp=None):
        """
        Save the depth estimation results/stats using timestamped files and cleanup routine.
        """
        if depth_map is None:
            return
            
        if timestamp is None:
            timestamp = time.time()
            
        # 1. Update latest state (Legacy/Singleton)
        metadata = {
            "timestamp": timestamp,
            "min_depth": float(depth_map.min()),
            "max_depth": float(depth_map.max()),
            "mean_depth": float(depth_map.mean()),
            "median_depth": float(np.median(depth_map)),
            **getattr(self, 'latest_metadata', {})
        }
        
        temp_json = self.output_path + ".tmp"
        with open(temp_json, 'w') as f:
            json.dump(metadata, f, indent=4)
        os.replace(temp_json, self.output_path)
        
        npy_singleton = self.output_path.replace(".json", ".npy")
        np.save(npy_singleton, depth_map)
        
        # 2. Save frame-specific files
        ts_ms = int(timestamp * 1000)
        frame_json = os.path.join(self.depth_dir, f"depth_{ts_ms}.json")
        frame_npy = os.path.join(self.depth_dir, f"depth_{ts_ms}.npy")
        
        with open(frame_json, 'w') as f:
            json.dump(metadata, f, indent=4)
        np.save(frame_npy, depth_map)
        
        # 3. Cleanup logic
        self.frame_history.append(timestamp)
        if len(self.frame_history) > self.max_history:
            old_ts = self.frame_history.pop(0)
            self._cleanup_old_depth(old_ts)

    def _cleanup_old_depth(self, timestamp):
        """Delete old depth files."""
        ts_ms = int(timestamp * 1000)
        json_file = os.path.join(self.depth_dir, f"depth_{ts_ms}.json")
        npy_file = os.path.join(self.depth_dir, f"depth_{ts_ms}.npy")
        
        for f in [json_file, npy_file]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception as e:
                    print(f"Error deleting old depth file {f}: {e}")

    def calculate_object_distance(self, bbox, depth_map, mask=None):
        """
        Calculate the distance of an object given its BBox, the depth map, and optional mask.
        """
        if depth_map is None:
            return None
            
        x1, y1, x2, y2 = bbox
        h, w = depth_map.shape[:2]
        
        # Clamp coordinates to frame boundaries
        ix1, iy1 = max(0, int(x1)), max(0, int(y1))
        ix2, iy2 = min(w, int(x2)), min(h, int(y2))
        
        if ix2 <= ix1 or iy2 <= iy1:
            return None
            
        roi_depth = depth_map[iy1:iy2, ix1:ix2]
        
        if mask is not None:
            mask_h, mask_w = mask.shape[:2]
            if mask_h != h or mask_w != w:
                mx1 = int((ix1 / w) * mask_w)
                my1 = int((iy1 / h) * mask_h)
                mx2 = int((ix2 / w) * mask_w)
                my2 = int((iy2 / h) * mask_h)
                
                roi_mask_small = mask[my1:my2, mx1:mx2]
                roi_h = iy2 - iy1
                roi_w = ix2 - ix1
                
                if roi_w > 0 and roi_h > 0:
                    roi_mask = cv2.resize(roi_mask_small, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
                else:
                    roi_mask = np.zeros((roi_h, roi_w), dtype=mask.dtype)
            else:
                roi_mask = mask[iy1:iy2, ix1:ix2]
                
            mask_valid = roi_mask > 0.5
            roi_valid = roi_depth[mask_valid]
            
            if roi_valid.size < 5: 
                roi_valid = roi_depth 
        else:
            roi_valid = roi_depth
            
        if roi_valid.size == 0:
            return None
            
        roi_filtered = roi_valid[np.isfinite(roi_valid)]
        if roi_filtered.size == 0:
            return None
            
        raw_val = float(np.median(roi_filtered))
        
        # Logic: Metric3D already provides distance in meters
        if not self.is_metric:
            # Fallback if somehow using a non-metric model
            distance = 1.0 / (raw_val + 1e-6)
            print("Error: Not metric values detected!")
        else:
            distance = raw_val
            
        return distance * self.depth_scale
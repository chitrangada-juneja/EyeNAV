import cv2
import torch
import numpy as np
import json
import time
import os
from ultralytics import YOLO

class ObjectDetector:
    """
    Wrapper for YOLO Object Detection and Segmentation.
    Handles model initialization, inference, and result standardization.
    """

    def __init__(self, model_path="yoloe-v8l-seg.pt", conf_threshold=0.1, prompts=None, output_path="data/states/vision_state.json", preprocessing_cfg=None,
                 path_conf_threshold=0.3, obj_conf_threshold=0.5, path_labels=None):
        """
        Initialize the YOLO-World object detector with segmentation.
        
        Args:
            model_path (str): Path to the YOLO-World weights.
            conf_threshold (float): Confidence threshold (lower for open-vocab).
            prompts (list): Optional list of labels/prompts for the model.
            output_path (str): Path to save JSON metadata.
            preprocessing_cfg (dict): Configuration for image enhancement.
        """
        self.preprocessing_cfg = preprocessing_cfg or {
            "auto_enhance": False,
            "brightness": 0.0,
            "contrast": 1.0,
            "gamma": 1.0
        }
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Loading YOLO model: {model_path} on {self.device}...")
        
        try:
            self.model = YOLO(model_path).to(self.device)
        except Exception as e:
            print(f"Error loading model {model_path}: {e}")
            raise
            
        self.conf_threshold     = conf_threshold      # base inference threshold (low)
        self.path_conf_threshold = path_conf_threshold # minimum conf for path surfaces
        self.obj_conf_threshold  = obj_conf_threshold  # minimum conf for all other objects

        # Default path label keywords — anything matching these is treated as a surface
        self.path_label_keywords = set(path_labels or [
            "sidewalk", "footpath", "walkway", "pavement", "concrete", "path",
            "tile", "floor", "marble", "carpet", "hallway", "corridor", "walkable",
            "asphalt", "road", "rough ground", "gravel", "grass", "dirt",
            "hallway floor", "indoor corridor", "corridor floor"
        ])
        self.output_path = output_path
        
        # Base data directories
        self.data_dir = os.path.dirname(os.path.dirname(output_path)) # data/
        self.det_dir = os.path.join(self.data_dir, "frames", "detections")
        self.seg_dir = os.path.join(self.data_dir, "frames", "segmentations")
        
        # Ensure directories exist
        os.makedirs(self.det_dir, exist_ok=True)
        os.makedirs(self.seg_dir, exist_ok=True)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Frame History for Cleanup
        self.frame_history = []
        self.max_history = 5
        
        # Initialize prompts from config or default
        self.prompts = prompts or [
            "person", "bicycle", "car", "motorcycle", "bus", "truck",
            "traffic light", "stop sign", "fire hydrant", "bench",
            "pothole", "stairs", "traffic cone", "trash can", 
            "rough ground", "stone", "road", "footpath","sidewalk", "tree", "building", "rocks","concrete"
        ]
        
        # Set custom classes for YOLO-World
        try:
            if hasattr(self.model, 'set_classes'):
                self.model.set_classes(self.prompts)
        except Exception as e:
             print(f"Warning: Failed to set classes on model: {e}")

    def detect(self, frame, visualize=False):
        """
        Run inference on a single frame.
        
        Args:
            frame (numpy.ndarray): Input image (BGR).
            visualize (bool): If True, returns a plotted visualization frame.
            
        Returns:
            dict: {
                'detections': list of objects,
                'masks': segmentation masks (optional),
                'vis_frame': plotted image (if visualize=True)
            }
        """
        start_time = time.time()
        
        # --- Preprocessing Step ---
        frame = self._preprocess_frame(frame)
        h, w = frame.shape[:2]
        
        # Run inference
        # half=True significantly speeds up inference on modern GPUs (FP16 vs FP32)
        results = self.model.predict(frame, conf=self.conf_threshold, verbose=False, device=self.device, half=True)
        result = results[0] # Single frame
        
        detections = []
        masks_np = None
        
        if visualize:
            # Create a copy for visualization to avoid modifying the original frame if used elsewhere
            vis_frame = frame.copy()
        
        if result.masks is not None:
            # Masks as numpy array (N, H, W), resize to original image size if needed
            # result.masks.data is usually on GPU, result.masks.xy is segments
            # converting to numpy cpu
            masks_tensor = result.masks.data
            if masks_tensor is not None:
                masks_np = masks_tensor.cpu().numpy()
            
            if visualize:
                 # We disable default labels and boxes here to avoid duplication with our custom overlays.
                 vis_frame = result.plot(labels=False, boxes=False) 
        
        if result.boxes is not None:
            boxes = result.boxes
            for i, box in enumerate(boxes):
                # BBox
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Metadata
                cls_id = int(box.cls[0])
                if hasattr(self.model, 'names') and cls_id < len(self.model.names):
                    label = self.model.names[cls_id]
                else:
                    label = str(cls_id)
                
                conf = float(box.conf[0])

                # Split confidence filtering:
                # Path surfaces run at a lower threshold (0.3).
                # All other objects use a higher threshold (0.5) to avoid noise.
                lower_label = label.lower()
                
                # --- GLOBAL LABEL FILTER ---
                # Only allow detections that match our desired prompts. 
                # This stops "Cat", "Mirror", or "Sink" from triggering if set_classes fails.
                if self.prompts and not any(p.lower() in lower_label for p in self.prompts):
                    continue

                is_path = any(kw in lower_label for kw in self.path_label_keywords)
                
                if is_path:
                    min_conf = self.path_conf_threshold
                else:
                    min_conf = self.obj_conf_threshold
                    
                if conf < min_conf:
                    continue

                # Use mask for better precision, otherwise fallback to bbox
                obj_mask = masks_np[i] if masks_np is not None else None
                color_name = self._extract_color(frame, mask=obj_mask, bbox=[x1, y1, x2, y2])
                
                detections.append({
                    "label": label,
                    "confidence": round(conf, 2),
                    "bbox": [x1, y1, x2, y2],
                    "color": color_name,
                    "mask_idx": i if masks_np is not None else None,
                    "timestamp": time.time()
                })
                
                if visualize: 
                     # Only draw the box; text is handled by the main controller for consistency
                     cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if visualize:
            # Add Diagnostic Overlay if enhancement is active
            if self.preprocessing_cfg.get("auto_enhance") or self.preprocessing_cfg.get("gamma", 1.0) != 1.0:
                cv2.putText(vis_frame, "ENHANCED", (w - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            return {
                "detections": detections,
                "masks": masks_np,
                "vis_frame": vis_frame
            }
        
        return {
            "detections": detections,
            "masks": masks_np
        }

    def _extract_color(self, frame, mask, bbox):
        """
        Determine the dominant color of the object.
        
        Strategy:
        1.  Use the mask (better) or bbox (faster) to isolate object pixels.
        2.  Convert to HSV color space.
        3.  Bucket into human-readable names (Red, Green, Blue, etc.).
        """
        x1, y1, x2, y2 = bbox
        # Clamp coordinates
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return "unknown"
            
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return "unknown"

        if mask is not None:
            # ROI from mask
            mask_h, mask_w = mask.shape[:2]
            if mask_h != h or mask_w != w:
                mx1 = int((x1 / w) * mask_w)
                my1 = int((y1 / h) * mask_h)
                mx2 = int((x2 / w) * mask_w)
                my2 = int((y2 / h) * mask_h)
                
                roi_mask_small = mask[my1:my2, mx1:mx2]
                roi_h = y2 - y1
                roi_w = x2 - x1
                
                if roi_w > 0 and roi_h > 0:
                    roi_mask = cv2.resize(roi_mask_small, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
                else:
                    roi_mask = np.zeros((roi_h, roi_w), dtype=mask.dtype)
            else:
                roi_mask = mask[y1:y2, x1:x2]
            
            # Application of mask to ROI
            # Calculate average color where mask > 0.5
            mask_binary = (roi_mask > 0.5).astype(np.uint8)
            if np.any(mask_binary):
                avg_stats = cv2.mean(roi, mask=mask_binary)
                b, g, r = avg_stats[:3]
            else:
                b, g, r = np.mean(roi, axis=(0, 1))
        else:
            b, g, r = np.mean(roi, axis=(0, 1))
        
        # Convert BGR to HSV for easier classification
        # Need 3D array for cv2.cvtColor
        query_color = np.uint8([[[b, g, r]]])
        hsv_color = cv2.cvtColor(query_color, cv2.COLOR_BGR2HSV)[0][0]
        h_val, s_val, v_val = hsv_color
        
        # Simple Logic Table for Color Naming
        # H: 0-179, S: 0-255, V: 0-255
        if s_val < 30 and v_val > 200:
            return "white"
        if v_val < 30:
            return "black"
        if s_val < 30:
            return "gray"
            
        if h_val < 10 or h_val > 170:
            return "red"
        elif h_val < 25:
            return "orange"
        elif h_val < 35:
            return "yellow"
        elif h_val < 85:
            return "green"
        elif h_val < 130:
            return "blue"
        elif h_val < 160:
            return "purple"
            
        return "colored"

    def _preprocess_frame(self, frame):
        """
        Enhances the frame for better detection accuracy in various lighting.
        Applies CLAHE, Brightness/Contrast adjustment, and Gamma correction.
        
        Scientifically grounded by research in low-light object detection:
        - Bhandari et al. (2020), arXiv:2006.05787
        - Zhao et al. (2024), Remote Sens. 2024, 16(23)
        """
        if not self.preprocessing_cfg:
            return frame

        processed = frame.copy()

        # 1. CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # Excellent for balancing "too much sunlight" and "indoor low-light"
        if self.preprocessing_cfg.get("auto_enhance"):
            # Convert to LAB for luminance-based equalization
            lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to the L-channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            
            # Merge back and convert to BGR
            limg = cv2.merge((cl, a, b))
            processed = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        # 2. Linear Brightness & Contrast
        # brightness: [-255, 255], contrast: [0, 3]
        brightness = self.preprocessing_cfg.get("brightness", 0.0)
        contrast = self.preprocessing_cfg.get("contrast", 1.0)
        if brightness != 0.0 or contrast != 1.0:
            # Formula: new_image = alpha * image + beta
            processed = cv2.convertScaleAbs(processed, alpha=contrast, beta=brightness)

        # 3. Gamma Correction
        # gamma > 1 moves towards shadows (darkens), gamma < 1 moves towards highlights (brightens)
        gamma = self.preprocessing_cfg.get("gamma", 1.0)
        if gamma != 1.0:
            invGamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            processed = cv2.LUT(processed, table)

        return processed

#     def save_to_json(self, detections, timestamp=None):
#         """
#         Save the current detections to the state file and a frame-specific file.
#         """
#         if timestamp is None:
#             timestamp = time.time()
#             
#         data = {
#             "timestamp": timestamp,
#             "count": len(detections),
#             "objects": detections
#         }
#         
#         # 1. Update latest vision state (legacy/singleton support)
#         temp_path = self.output_path + ".tmp"
#         with open(temp_path, 'w') as f:
#             json.dump(data, f)
#         os.replace(temp_path, self.output_path)
#         
#         # 2. Save frame-specific JSON
#         frame_filename = f"detection_{int(timestamp * 1000)}.json"
#         frame_path = os.path.join(self.det_dir, frame_filename)
#         with open(frame_path, 'w') as f:
#             json.dump(data, f)
#             
#         # 3. Track history and cleanup
#         self.frame_history.append(timestamp)
#         if len(self.frame_history) > self.max_history:
#             old_ts = self.frame_history.pop(0)
#             self._cleanup_old_frames(old_ts)
#
#     def _cleanup_old_frames(self, timestamp):
#         """Delete detection and segmentation files for a specific timestamp."""
#         ts_ms = int(timestamp * 1000)
#         
#         det_file = os.path.join(self.det_dir, f"detection_{ts_ms}.json")
#         seg_file = os.path.join(self.seg_dir, f"segmentation_{ts_ms}.npy")
#         
#         for f in [det_file, seg_file]:
#             if os.path.exists(f):
#                 try:
#                     os.remove(f)
#                 except Exception as e:
#                     print(f"Error deleting old frame file {f}: {e}")

    def save_masks(self, masks, timestamp=None):
        """
        Save segmentation masks to disk.
        """
        if masks is None:
            return
            
        if timestamp is None:
            timestamp = time.time()
            
        # 1. Update latest masks (legacy/singleton support)
        mask_path = self.output_path.replace(".json", "_masks.npy")
        temp_path = mask_path.replace(".npy", "_temp.npy")
        np.save(temp_path, masks)
        os.replace(temp_path, mask_path)
        
        # 2. Save frame-specific mask
        frame_filename = f"segmentation_{int(timestamp * 1000)}.npy"
        frame_path = os.path.join(self.seg_dir, frame_filename)
        np.save(frame_path, masks)

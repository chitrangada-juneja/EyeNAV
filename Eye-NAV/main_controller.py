import os
import json
import time
import cv2
import threading
import queue
from collections import deque
import numpy as np
import gc
import torch
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import modules
from src.perception.detection import ObjectDetector
from src.io.camera import CameraSource
from src.perception.depth import DepthEstimator
from src.perception.path_analyzer import PathAnalyzer
from src.io.haptics import HapticInterface
from src.io.audio import AudioInterface

#import more
from src.nlp.nav_controller import NavigationLLMController
from src.nlp.default_responses import DefaultResponse
#ADDED
from src.nlp.preprocessing import Preprocessing

class MainController:
    """
    The Central Orchestrator for the Assistive Navigation System.
    
    Responsibilities:
    1.  Owns the Video Source (Camera).
    2.  Schedules asynchronous vision tasks:
        -   Continuous Object Detection (YOLO).
        -   1Hz Depth Estimation (Metric3D).
    3.  Aggregates data (Fusion).
    4.  Executes immediate safety logic (Haptics/Audio).
    5.  Delegates complex reasoning to the LLM.
    """

    def __init__(self, config_path="config.json", video_source=None):
        """
        Initialize the controller, logic, and IO.
        """
        print("Initializing Assistive Navigation System (Vision Mode)...")
        
        # 0. Load Config
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load config from {config_path}: {e}")
            self.config = {}

        # 1. Initialize Hardware/IO
        # Use realtime_mode=True so video files play at normal speed
        if video_source is not None:
            camera_id = video_source
        else:
            camera_id = self.config.get('io', {}).get('camera_id', 0)
            
        self.camera = CameraSource(source=camera_id, realtime_mode=True) 
        self.haptics = HapticInterface()

        # UI rendering preferences (Loaded from config)
        ui_cfg = self.config.get("ui")
        self.font_scale = ui_cfg.get("telemetry_font_scale")
        self.font_thick = ui_cfg.get("telemetry_font_thickness")
        
        # 2. Initialize Visual Perception
        perception_cfg = self.config.get('perception')
        self.detector = ObjectDetector(
            model_path=perception_cfg.get('yolo_model'),
            conf_threshold=perception_cfg.get('yolo_conf_threshold'),
            prompts=perception_cfg.get('yolo_prompts'),
            preprocessing_cfg=perception_cfg.get('preprocessing')
        )
        
        self.path_analyzer = PathAnalyzer(
            use_com=True, 
            use_weighted_bias=True, 
            use_bev=False,
            path_labels=perception_cfg.get('path_labels'),
            forbidden_labels=perception_cfg.get('forbidden_labels')
        )
        
        # Configure BEV if points provided in config
        bev_cfg = perception_cfg.get('bev_config')
        if bev_cfg:
            self.path_analyzer.bev_src = np.float32(bev_cfg.get('src'))
            self.path_analyzer.bev_dst = np.float32(bev_cfg.get('dst'))
            
        self.depth_estimator = DepthEstimator(
            model_id=perception_cfg.get('depth_model'),
            depth_scale=perception_cfg.get('depth_scale'),
            is_metric=perception_cfg.get('is_metric')
        )
        
        if hasattr(torch, 'compile'):
            try:
                print("Compiling Metric3D for speed...")
                self.depth_estimator.model = torch.compile(self.depth_estimator.model)
            except: pass
        
        # 3. Initialize NLP & Decision Engine
        llm_cfg = self.config.get('llm')
        api_key = os.getenv(llm_cfg.get('api_key_env'))
        
        if not api_key:
            print("ERROR: GOOGLE_API_KEY not found in environment. LLM features will be disabled.")
            self.nav_controller = None
        else:
            self.nav_controller = NavigationLLMController(
                api_key=api_key,
                config=self.config
            )

        self.preprocessor=Preprocessing()
        self.default_responses = DefaultResponse()

        # 4. LLM Preferences & Quota Management
        self.user_preferences = {
            'verbose_mode': llm_cfg.get('verbose_mode'),
            'metric_units': llm_cfg.get('metric_units'),
            'critical_distance_threshold': llm_cfg.get('critical_distance_threshold')
        }
        
        # 5. Performance & Rate Limiting
        self.llm_interval = llm_cfg.get('interval_seconds')
        self.depth_interval = perception_cfg.get('depth_interval_seconds')
        self.last_llm_time = 0
        self.last_depth_time = 0
        
        # Safety & Decision Factors
        safety_cfg = self.config.get('safety')
        self.safety_thresholds = {
            'critical': safety_cfg.get('critical_distance_meters'),
            'warning': safety_cfg.get('warning_distance_meters'),  
            'advisory': safety_cfg.get('advisory_distance_meters')
        }
        self.nav_ttl = safety_cfg.get('navigation_ttl_seconds')
        
        # Threading/Sync for Async Depth
        self.running = False
        self.latest_depth_map = None
        self.fps_history = deque(maxlen=30)
        self.depth_lock = threading.Lock()
        self.depth_cond = threading.Condition()
        self.next_depth_frame = None
        self.depth_thread = None
        
        # GPU Lock to prevent YOLO/Metric3D thrashing
        self.gpu_lock = threading.Lock()
        
        # Async I/O Saver
        self.io_queue = queue.Queue()
        self.io_thread = None
        
        # 4. Fused Data Output Dir
        self.fused_dir = "data/states/fused_states"
        os.makedirs(self.fused_dir, exist_ok=True)
        self.fused_history = []
        
        # Initialize TTS
        self.audio = AudioInterface(tts_engine=self.config.get('io').get('tts_engine'))
        
        # 5. Global Navigation Decision State (With Age Tracking)
        self.nav_state = {
            "immediate_hazard": None,    
            "immediate_hazard_timestamp": 0.0,
            "llm_instruction": None,     
            "llm_timestamp": 0.0,
            "path_instruction": "Searching for path...", 
            "path_instruction_timestamp": 0.0,
            "final_decision": "Thinking. Please wait",
            "msg_level": 0
        }

        # TTS Rate-limiting and prioritization (Loaded from config)
        self.last_spoken_msg = ""
        self.last_spoken_time = 0.0
        self.last_spoken_level = 0
        
        io_cfg = self.config.get("io")
        tts_cfg = io_cfg.get("tts_settings")
        self.min_speech_interval = tts_cfg.get("min_speech_interval")
        self.critical_repeat_interval = tts_cfg.get("critical_repeat_interval")
        self.warning_factor = tts_cfg.get("warning_interval_factor")
        
        # Stability: 3-frame rule (relocated from PathAnalyzer)
        self.path_stability_counter = 0
        self.last_stable_path = "Searching for path..."
        
        # Logging/Perf tracking (Parity with src/main.py)
        self._last_log_time = 0.0
        self._last_log_msg = ""
        self._frame_counter = 0
        
        # Path Stability State (Hysteresis)
        self.path_stability_counter = 0
        self.last_stable_path = "Search for walkable path."
        
        print("System Initialized.")


    def start(self):
        """
        Start the main execution loop and background workers.
        """
        self.running = True
        
        # Start Workers
        self.depth_thread = threading.Thread(target=self._depth_worker, daemon=True)
        self.depth_thread.start()
        
        self.io_thread = threading.Thread(target=self._io_worker, daemon=True)
        self.io_thread.start()
        
        print("--- Starting Main Loop (Press 'q' to stop) ---")
        try:
            self._run_vision_loop()
        finally:
            self.stop()

    def _run_vision_loop(self):
        """
        The main loop for processing video frames.
        """
        while self.running:
            start_tick = time.perf_counter()
            
            # 0. Acquire Depth Snapshot (Pinned for the duration of this frame)
            with self.depth_lock:
                local_depth = self.latest_depth_map
            
            # 1. Acquire Frame
            frame = self.camera.get_frame(block=True)
            if frame is None:
                break
            
            # --- OPTIMIZATION: Source-Level Downscaling ---
            h, w = frame.shape[:2]
            if max(h, w) > 640:
                scale = 640 / max(h, w)
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
                
            # 2. Object Detection
            t_all_start = time.perf_counter()
            current_ts = time.time()
            is_incremental = False
            tts_action = "SILENCE"
            t_yolo_start = time.perf_counter()
            with self.gpu_lock:
                result = self.detector.detect(frame, visualize=True)
                if torch.cuda.is_available(): torch.cuda.synchronize()
            t_yolo = (time.perf_counter() - t_yolo_start) * 1000

            detections = result['detections']
            masks = result['masks']
            vis_frame = result.get('vis_frame', frame.copy())
            current_ts = time.time()

            # 2a. Navigation Instructions from Path Segmentation
            t_path_start = time.perf_counter()
            upcoming_turn_detection = False
            mask_labels = []
            if masks is not None and len(masks) > 0:
                mask_labels = [None] * len(masks)
                for det in detections:
                    if det.get('mask_idx') is not None:
                        mask_labels[det['mask_idx']] = det['label']
            
            # 2b. Get raw instruction from Analyzer
            # We pass the latest depth map and the safety warning threshold 
            # to detect 'Path coming to an end' early.
            raw_instr, upcoming_turn_detection = self.path_analyzer.get_instruction(
                masks, 
                mask_labels, 
                depth_map=local_depth,
                end_threshold=self.safety_thresholds.get('warning', 2.0)
            )
            
            # --- Path Stability Logic ---
            if raw_instr == "No walkable path detected":
                self.path_stability_counter += 1
                if self.path_stability_counter >= 10:
                    nav_instruction = raw_instr
                else:
                    # Temporary loss - stay on the last valid message to avoid flicker
                    nav_instruction = self.last_stable_path
            else:
                # Path found - reset and update history
                self.path_stability_counter = 0
                self.last_stable_path = raw_instr
                nav_instruction = raw_instr

            # 5. Path Analysis
            self.nav_state['path_instruction'] = nav_instruction
            self.nav_state['path_instruction_timestamp'] = current_ts
            
            # Overlay Path Information (NEW)
            if masks is not None and len(masks) > 0:
                if getattr(self.path_analyzer, 'use_com', False):
                    # Overlay Ideal Path / CoM
                    midline = getattr(self.path_analyzer, 'last_midline', None)
                    if midline and len(midline) > 1:
                        pts = np.array(midline, dtype=np.int32)
                        cv2.polylines(vis_frame, [pts], False, (0, 255, 0), 2)
                    com = getattr(self.path_analyzer, 'last_com', None)
                    if com:
                        cv2.circle(vis_frame, com, 10, (255, 0, 0), -1)
                else:
                    # Overlay Grid lines
                    grid = getattr(self.path_analyzer, 'last_grid', None)
                    if grid:
                        h, w = vis_frame.shape[:2]
                        for gy in grid['rows']:
                            cv2.line(vis_frame, (0, gy), (w, gy), (0, 255, 0), 1)
                        for gx in grid['cols']:
                            cv2.line(vis_frame, (gx, 0), (gx, h), (0, 255, 0), 1)
            t_path = (time.perf_counter() - t_path_start) * 1000
            
            # 3. Queue Masks for Async Saving (Detections are now handled via fused_state)
            if masks is not None:
                self.io_queue.put(('masks', masks, current_ts))
            
            # 4. Async Depth Trigger
            if current_ts - self.last_depth_time > self.depth_interval:
                with self.depth_cond:
                    self.next_depth_frame = frame.copy()
                    self.depth_cond.notify()
                self.last_depth_time = current_ts # Reset timer on signal
            
            # 5. Real-time Fusion & Overlay
            with self.depth_lock:
                local_depth = self.latest_depth_map
            
            if local_depth is not None:
                fused_objects = []
                
                # Update distances and build fused objects in a single pass
                for det in detections:
                    # Pass the corresponding mask if available for better accuracy
                    m_idx = det.get('mask_idx')
                    obj_mask = masks[m_idx] if (masks is not None and m_idx is not None) else None
                    dist = self.depth_estimator.calculate_object_distance(det['bbox'], local_depth, mask=obj_mask)
                    
                    if dist is not None:
                        det['distance'] = dist
                        # Overlay on visualizer
                        # Combined Overlay for Distance, Label, and Color
                        x1, y1, x2, y2 = det['bbox']
                        label_text = f"[{dist:.1f}m] {det['label']} ({det.get('color', 'unknown')})"
                        cv2.putText(vis_frame, label_text, (x1, int(y1 - 10)), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    
                    # Get list of occupied zones in 3x3 grid
                    h_v, w_v = vis_frame.shape[:2]
                    zones = self._get_spatial_position(det['bbox'], w_v, h_v)
                    
                    # FINAL CREATION OF FUSED_OBJECT WITH ALL REQUIRED INFORMATION
                    fused_objects.append({
                        "label": det['label'],
                        "confidence": det['confidence'],
                        "position": zones,
                        "distance": dist if dist is not None else float('inf'),
                        "color": det.get('color', 'unknown'),
                        "timestamp": det.get('timestamp', current_ts)
                    })
                
                # Queue depth stats for saving
                self.io_queue.put(('depth', local_depth, current_ts))
                
                fused_state = {
                    "timestamp": current_ts,
                    "count": len(fused_objects),
                    "objects": fused_objects,
                    "navigation_instruction": self.nav_state['path_instruction']
                }
                
                # NOW THAT WE CREATED, WE SEND IT OVER TO PREPROCESSING 
                all_sorted_objects = self.preprocessor.execute(fused_state, upcoming_turn_detection)
                
                # Queue for saving
                self.io_queue.put(('fused', fused_state, current_ts))
                
                # --- Direct LLM Testing (Blocking) ---
                if current_ts - self.last_llm_time > self.llm_interval:
                    llm_data = {"timestamp": current_ts, "object_data": fused_objects, "path_suggestion": self.nav_state['path_instruction']}
                    print("\n--- Sending State to LLM (Blocking) ---")
                    try:
                        llm_resp = self.process_with_llm(llm_data)
                        if llm_resp:
                            if llm_resp.get('quota_exceeded'):
                                self.nav_state['llm_instruction'] = "QUOTA EXCEED"
                                # We don't update the timestamp here to show it's stale/paused
                            else:
                                if instructions := llm_resp.get('instructions'):
                                    self.nav_state['llm_instruction'] = instructions
                                    self.nav_state['llm_timestamp'] = current_ts
                                    print(f"[LLM Instructions]: {instructions}")
                            
                            hazards = llm_resp.get('hazards', [])
                            if hazards:
                                print(f"[Hazards Detected]: {hazards}")
                    except Exception as e:
                        print(f"LLM Processing Error: {e}")
                    finally:
                        # Update the timer AFTER the response is received
                        # This ensures we wait for 'llm_interval' seconds of IDLE time between calls.
                        self.last_llm_time = time.time()
                
                # Risk Assessment (Logic could be refined for async distances)
                t_decide_start = time.perf_counter()
                final_msg, msg_level, hazard_state = self._fuse_and_assess_risk(all_sorted_objects, upcoming_turn_detection)
                self.nav_state['final_decision'] = final_msg
                self.nav_state['msg_level'] = msg_level
                t_decide = (time.perf_counter() - t_decide_start) * 1000
            else:
                # Default timings if depth is stale
                t_decide = 0.0
                hazard_state = "NEW"

            # 6. TTS Prioritization Logic (Bombardment Mitigation)
            tts_action = "SILENCE"
            current_msg = self.nav_state['final_decision']
            current_level = self.nav_state['msg_level']
            # h_state was populated in _run_frame from _fuse_and_assess_risk. If not, default to "NEW"
            h_state = hazard_state if 'hazard_state' in locals() else "NEW"
            
            time_since_speech = current_ts - self.last_spoken_time
            is_new_msg = (current_msg != self.last_spoken_msg)
            
            if current_level == 2:
                # For critical hazards, we speak if it's a NEW hazard, 
                # or if a DELTA occurred and we haven't spoken too recently,
                # or if it's the SAME hazard but the repeat interval has passed.
                
                can_speak_delta = (h_state == "DELTA" and time_since_speech > (self.min_speech_interval * self.warning_factor))
                
                # For 'SAME' hazards, we only repeat if a significant amount of time has passed
                # (e.g., 1.5x the critical repeat interval if the message is identical)
                repeat_threshold = self.critical_repeat_interval
                if not is_new_msg:
                    repeat_threshold *= 1.5
                    
                can_repeat_same = (h_state == "SAME" and time_since_speech > repeat_threshold)
                
                if h_state == "NEW" or can_speak_delta or can_repeat_same:
                    if h_state == "NEW":
                        tts_action = "INTERRUPT"
                    else:
                        tts_action = "SPEAK"
            elif current_level == 1:
                if is_new_msg and time_since_speech > (self.min_speech_interval * self.warning_factor):
                    if h_state == "NEW" and self.last_spoken_level < 1:
                        tts_action = "INTERRUPT"
                    elif h_state == "DELTA":
                        tts_action = "SPEAK"
                    else:
                        tts_action = "SPEAK"
            else:
                if is_new_msg and h_state != "SAME" and time_since_speech > self.min_speech_interval:
                    tts_action = "SPEAK"

            if tts_action != "SILENCE":
                self.last_spoken_msg = current_msg
                self.last_spoken_time = current_ts
                self.last_spoken_level = current_level
                
                # HIGHLIGHT SPEECH IN TERMINAL
                action_color = "\033[91m" if tts_action == "INTERRUPT" else "\033[94m"
                state_color = "\033[93m" if h_state == "NEW" else "\033[90m"
                reset_color = "\033[0m"
                print(
                    f"[{time.strftime('%H:%M:%S')}] "
                    f"{action_color}[TTS:{tts_action}]{reset_color} "
                    f"{state_color}({h_state}/L{current_level}){reset_color} "
                    f"\"{current_msg}\""
                )
                
                self.audio.speak(current_msg)

            # 7. Final Navigation Decision & UI Panel
            self._draw_decision_panel(vis_frame)

            # 8. FPS Overlay (Moved to bottom right)
            self.fps_history.append(time.perf_counter())
            fps = 0.0
            if len(self.fps_history) > 1:
                fps = len(self.fps_history) / (self.fps_history[-1] - self.fps_history[0])
                h_f, w_f = vis_frame.shape[:2]
                cv2.putText(vis_frame, f"FPS: {fps:.1f}", (w_f - 110, h_f - 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # --- 9. Performance Logging (Prioritize Speech Events) ---
            t_all = (time.perf_counter() - start_tick) * 1000
            
            if tts_action != "SILENCE":
                action_color = "\033[91m" if tts_action == "INTERRUPT" else "\033[94m" # Red for Interrupt, Blue for Speak
                reset_color = "\033[0m"
                print(
                    f"[{time.strftime('%H:%M:%S')}] "
                    f"{action_color}[TTS:{tts_action}]{reset_color} \"{current_msg}\" "
                    f"({t_all:.0f}ms)"
                )
            elif self._frame_counter % 30 == 0:
                # Keep a very quiet heartbeat every 30 frames
                print(f"[{time.strftime('%H:%M:%S')}] [System IDLE] Monitoring... ({t_all:.0f}ms)", end="\r")

            # Final Show
            cv2.imshow("You-POV (Realtime Optimization)", vis_frame)
            
            # 10. Cleanup heavy buffers immediately to prevent RAM build-up
            if masks is not None: del masks
            if vis_frame is not None: del vis_frame
            
            # Periodic manual garbage collection
            self._frame_counter += 1
            if self._frame_counter % 100 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Exit Condition
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop()
                break

    def _depth_worker(self):
        """Background thread for Metric3D processing."""
        while self.running:
            with self.depth_cond:
                while self.running and self.next_depth_frame is None:
                    self.depth_cond.wait(timeout=0.1)
                if not self.running: break
                target_frame = self.next_depth_frame
                self.next_depth_frame = None
            try:
                with self.gpu_lock:
                    new_map = self.depth_estimator.estimate_depth(target_frame)
                with self.depth_lock:
                    self.latest_depth_map = new_map
            except Exception as e:
                print(f"Depth Worker Error: {e}")

    def _io_worker(self):
        """Background thread for non-blocking disk writes."""
        while self.running or not self.io_queue.empty():
            try:
                task = self.io_queue.get(timeout=0.1)
                dtype, data, ts = task
                if dtype == 'masks': self.detector.save_masks(data, timestamp=ts)
                elif dtype == 'depth': self.depth_estimator.save_state(data, timestamp=ts)
                elif dtype == 'fused': self._save_fused_state(data, timestamp=ts)
                self.io_queue.task_done()
            except queue.Empty: continue
            except Exception as e: print(f"IO Error: {e}")

    def _get_spatial_position(self, bbox, width, height):
        """
        Determine which of the 9 squares (3x3 grid) a BBox occupies using overlap detection.
        Returns a list of occupied zones (e.g., ["high-left", "middle-left"]).
        """
        bx1, by1, bx2, by2 = bbox
        zones = []
        
        # Grid boundaries
        v_steps = [0, height/3, 2*height/3, height]
        h_steps = [0, width/3, 2*width/3, width]
        v_labels = ["high", "middle", "low"]
        h_labels = ["left", "center", "right"]
        
        for v in range(3):
            for h in range(3):
                # Intersection math
                ix1, iy1 = max(bx1, h_steps[h]), max(by1, v_steps[v])
                ix2, iy2 = min(bx2, h_steps[h+1]), min(by2, v_steps[v+1])
                if ix1 < ix2 and iy1 < iy2:
                    zones.append(f"{v_labels[v]}-{h_labels[h]}")
        return zones if zones else ["middle-center"]

    def _save_fused_state(self, data, timestamp):
        """Save fused AI state and cleanup old files."""
        ts_ms = int(timestamp * 1000)
        path = os.path.join(self.fused_dir, f"fused_{ts_ms}.json")
        with open(path, 'w') as f: json.dump(data, f, indent=2)
        
        # Singleton/Latest update
        latest_path = os.path.join(os.path.dirname(self.fused_dir), "fused_state.json")
        with open(latest_path, 'w') as f: json.dump(data, f, indent=2)
        
        # Cleanup
        self.fused_history.append(timestamp)
        if len(self.fused_history) > 5:
            old_ts = self.fused_history.pop(0)
            old_f = os.path.join(self.fused_dir, f"fused_{int(old_ts * 1000)}.json")
            if os.path.exists(old_f): os.remove(old_f)

    def _fuse_and_assess_risk(self, all_sorted_objects, upcoming_turn_detection):
        """Analyze threats based on distance and label using parameter-driven responses.
        it uses safety_threshold's values to decide what form of response to return."""
        # 1. Check for immediate hazards (Critical/Warning)
        urgent_warning, is_critical, hazard_state = self.default_responses.get_sos_response(
            all_sorted_objects,
            critical_dist=self.safety_thresholds['critical'],
            warning_dist=self.safety_thresholds['warning'],
            path_instruction=self.nav_state.get('path_instruction'),
            upcoming_turn=upcoming_turn_detection
        )
        if urgent_warning:
            self.nav_state['immediate_hazard'] = urgent_warning
            self.nav_state['immediate_hazard_timestamp'] = time.time()
            return urgent_warning, (2 if is_critical else 1), hazard_state
        
        # 2. Check for active LLM Instruction within TTL
        llm_instruction = self.nav_state.get('llm_instruction')
        if llm_instruction and (time.time() - self.nav_state.get('llm_timestamp', 0)) < self.nav_ttl:
            return llm_instruction, 0, "NEW"
            
        # 3. Check for advisory defaults if no immediate hazards or LLM exist
        default_message, hazard_state = self.default_responses.get_default_message_response(
            all_sorted_objects,
            advisory_dist=self.safety_thresholds['advisory'],
            path_instruction=self.nav_state.get('path_instruction'),
            upcoming_turn=upcoming_turn_detection
        )
        return (default_message or "Path clear."), 1, hazard_state

    def _draw_text_wrapped(self, frame, text, x, y, font, scale, color, thickness, max_w):
        """Helper to draw text with simple word wrapping. Returns the next Y position."""
        if not text: return y
        words = text.split(' ')
        lines, current_line = [], ""
        for word in words:
            test_line = (current_line + " " + word) if current_line else word
            (w, _), _ = cv2.getTextSize(test_line, font, scale, thickness)
            if w <= max_w: current_line = test_line
            else:
                if current_line: lines.append(current_line)
                current_line = word
        if current_line: lines.append(current_line)
        for line in lines:
            (_, h), _ = cv2.getTextSize(line, font, scale, thickness)
            cv2.putText(frame, line, (x, y), font, scale, color, thickness)
            y += int(h * 1.5) + 5
        return y

    def _draw_decision_panel(self, frame):
        """Draw full-text instructions for all channels with word wrapping."""
        h, w = frame.shape[:2]
        now = time.time()
        def age_str(ts): return "None" if ts <= 0 else f"{now - ts:.1f}s"
        
        scale, thick = getattr(self, 'font_scale', 0.85), getattr(self, 'font_thick', 2)
        max_width = w - 24 
        final_msg = self.nav_state.get('final_decision', "Thinking. Please wait")
        
        # 1. Prepare texts
        imm_age = age_str(self.nav_state['immediate_hazard_timestamp'])
        is_imm = self.nav_state['immediate_hazard'] == final_msg and final_msg is not None
        is_system = is_imm or (final_msg == self.nav_state['path_instruction'])
        sys_msg = f"{'> ' if is_system else '  '}SYSTEM ({imm_age}): {self.nav_state['immediate_hazard'] or self.nav_state['path_instruction']}"
        
        llm_age = age_str(self.nav_state['llm_timestamp'])
        is_llm = self.nav_state['llm_instruction'] == final_msg and not is_imm
        llm_msg = f"{'> ' if is_llm else '  '}LLM ({llm_age}): {self.nav_state['llm_instruction'] or 'None'}"
        
        active_msg = f"ACTIVE: {final_msg}"

        # 2. Helper to accurately pre-measure the wrapped text height
        def get_wrapped_height(text, font, scale, thickness):
            if not text: return 0
            words = text.split(' ')
            lines, current_line = [], ""
            for word in words:
                test_line = (current_line + " " + word) if current_line else word
                (tw, _), _ = cv2.getTextSize(test_line, font, scale, thickness)
                if tw <= max_width: current_line = test_line
                else:
                    if current_line: lines.append(current_line)
                    current_line = word
            if current_line: lines.append(current_line)
            total_h = 0
            for line in lines:
                (_, lh), _ = cv2.getTextSize(line, font, scale, thickness)
                total_h += int(lh * 1.5) + 5
            return total_h
            
        sys_h = get_wrapped_height(sys_msg, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        llm_h = get_wrapped_height(llm_msg, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        act_h = get_wrapped_height(active_msg, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        
        current_y_tracker = 35
        # Add 10 padding to the bottom of the grey region
        # Plus 15 for spacing between sys and llm
        grey_bg_bottom = current_y_tracker + sys_h + 15 + llm_h + 10
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, int(grey_bg_bottom)), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        msg_level = self.nav_state.get('msg_level', 0)
        bg_color = (0, 0, 200) if msg_level == 2 else ((0, 140, 255) if msg_level == 1 else (50, 50, 50))
        text_color = (255, 255, 255) if msg_level > 0 else (200, 200, 200)

        rect_top = int(grey_bg_bottom)
        rect_bottom = int(rect_top + act_h + 16) # Add padding below text
        cv2.rectangle(frame, (0, rect_top), (w, rect_bottom), bg_color, -1)
        
        # Draw texts
        sys_color = (0, 0, 255) if is_imm else ((0, 255, 0) if is_system else (200, 200, 200))
        current_y = self._draw_text_wrapped(frame, sys_msg, 10, current_y_tracker, cv2.FONT_HERSHEY_SIMPLEX, scale, sys_color, thick if is_system else 1, max_width)
        
        llm_color = (255, 255, 0) if is_llm else (200, 200, 200)
        current_y = self._draw_text_wrapped(frame, llm_msg, 10, current_y + 15, cv2.FONT_HERSHEY_SIMPLEX, scale, llm_color, thick if is_llm else 1, max_width)
        
        # Calculate proper baseline for the active_msg
        # In main_controller.py, scale is often 0.85, so 26 dropping is appropriate.
        self._draw_text_wrapped(frame, active_msg, 12, rect_top + 26, cv2.FONT_HERSHEY_SIMPLEX, scale, text_color, 2, max_width)

    def process_with_llm(self, data):
        """
        Process detection data through LLM for navigation instructions.
        
        Args:
            data: Object detection results
            
        Returns:
            Navigation instructions dict or None
        """
        if not self.nav_controller: return None
        
        # the modes are still under development, but we can pass them for now
        # they represent the different UI modes we want to support 
        """
        This is the code in prompts.py, that uses this info:
        Verbose mode: ENABLED (provide more detail)\n"
        else:
            context += "  • Concise mode: Keep instructions brief\n"
        
        if preferences.get("metric_units"):
            context += "  • Units: METRIC (meters)\n"
        else:
            context += "  • Units: IMPERIAL (feet/yards)\n"
        
        if critical_distance := preferences.get("critical_distance_threshold"):
            context += f"  • Critical hazard distance: {critical_distance}m\n"
            
        """
        return self.nav_controller.generate_instructions(detection_data=data, user_preferences=self.user_preferences)

    def stop(self):
        """
        Gracefully shutdown.
        """
        if not self.running and not hasattr(self, '_stopping'):
            return 
        self._stopping = True
        self.running = False
        
        # Wake up workers
        with self.depth_cond: self.depth_cond.notify_all()
        
        # Release hardware
        if hasattr(self, 'camera'): self.camera.release()
        
        # Close UI
        cv2.destroyAllWindows()
        
        # Wait for threads
        if self.depth_thread: self.depth_thread.join(timeout=1.0)
        if self.io_thread: self.io_thread.join(timeout=1.0)
        print("System Stopped.")

if __name__ == "__main__":
    import sys
    # Default to 0 (webcam). If a file path is provided as an argument, use that instead.
    # e.g., python main_controller.py data/validation_videos/footage_4.mp4
    video_source = sys.argv[1] if len(sys.argv) > 1 else 0
    
    # If the user provided a path but it doesn't exist, log a warning
    if isinstance(video_source, str) and not os.path.exists(video_source): 
        print(f"Warning: Video file '{video_source}' not found. Falling back to webcam.")
        video_source = 0
        
    controller = MainController(video_source=video_source)
    controller.start()

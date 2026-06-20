"""
MainProcessor — src/main.py
============================
This is the server-side brain of Eye-NAV. It receives individual camera frames
from the Flask backend, runs the AI pipeline, and returns a structured result.

Threading model:
  - InferenceWorker:  Runs YOLO object detection on every incoming frame.
  - DepthWorker:      Runs depth estimation periodically in the background.
  - LLMWorker:        Calls the language model API asynchronously.
  - IOWorker:         Handles disk writes so they never stall the vision loop.

The public API (process_frame) is fully non-blocking: it queues the new frame
and immediately returns the most recently computed state. This ensures the UI
always gets a response in under a millisecond regardless of how long the AI takes.
"""

import os
import json
import time
import torch
import cv2

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

import threading
import queue
from collections import deque
import base64
import numpy as np
import gc
from dotenv import load_dotenv

load_dotenv()

from src.perception.detection import ObjectDetector
from src.perception.depth import DepthEstimator
from src.perception.path_analyzer import PathAnalyzer
from src.io.haptics import HapticInterface
from src.io.audio import AudioInterface
from src.nlp.nav_controller import NavigationLLMController
from src.nlp.default_responses import DefaultResponse
from src.nlp.preprocessing import Preprocessing


class MainProcessor:
    """
    Orchestrates the full AI pipeline for the Eye-NAV assistive system.
    Designed for real-time, low-latency operation via a multi-threaded architecture.
    """

    def __init__(self, config_path="config.json", show_local=True):
        print("[MainProcessor] Initializing...")
        self.show_local = show_local

        # Load configuration from disk
        try:
            with open(config_path, "r") as f:
                self.config = json.load(f)
        except Exception as e:
            print(f"[MainProcessor] Warning: Could not load config ({e}). Using defaults.")
            self.config = {}

        # Hardware interfaces
        self.haptics = HapticInterface()
        self.audio   = AudioInterface()

        # Perception models
        perception_cfg = self.config.get("perception")

        self.detector = ObjectDetector(
            model_path=perception_cfg.get("yolo_model"),
            conf_threshold=perception_cfg.get("yolo_conf_threshold"),
            prompts=perception_cfg.get("yolo_prompts"),
            preprocessing_cfg=perception_cfg.get("preprocessing"),
            path_conf_threshold=perception_cfg.get("path_conf_threshold"),
            obj_conf_threshold=perception_cfg.get("obj_conf_threshold"),
        )

        self.path_analyzer = PathAnalyzer(
            use_com=True,
            use_weighted_bias=True,
            use_bev=False,
            primary_labels=perception_cfg.get("primary_labels"),
            fallback_labels=perception_cfg.get("fallback_labels"),
            forbidden_labels=perception_cfg.get("forbidden_labels"),
        )

        # Optional: apply Bird's-Eye View warp points from config
        bev_cfg = perception_cfg.get("bev_config")
        if bev_cfg:
            self.path_analyzer.bev_src = np.float32(bev_cfg.get("src"))
            self.path_analyzer.bev_dst = np.float32(bev_cfg.get("dst"))

        self.depth_estimator = DepthEstimator(
            model_id=perception_cfg.get("depth_model"),
            depth_scale=perception_cfg.get("depth_scale"),
            is_metric=perception_cfg.get("is_metric"),
        )
        
        # --- PERFORMANCE: Torch Compile (Torch 2.4+) ---
        if hasattr(torch, 'compile'):
            try:
                print("[MainProcessor] Compiling Metric3D for faster inference...")
                self.depth_estimator.model = torch.compile(self.depth_estimator.model)
            except Exception as e:
                print(f"[MainProcessor] Could not compile model: {e}")

        # Language Model (LLM) for navigation instructions
        llm_cfg = self.config.get("llm")
        api_key = os.getenv(llm_cfg.get("api_key_env"))
        if not api_key:
            print("[MainProcessor] WARNING: GOOGLE_API_KEY not set. LLM guidance is disabled.")
            self.nav_controller = None
        else:
            self.nav_controller = NavigationLLMController(api_key=api_key, config=self.config)

        self.preprocessor = Preprocessing()
        self.default_responses = DefaultResponse()

        # User-facing preferences (updated via the settings UI)
        self.user_preferences = {
            "verbose_mode":                llm_cfg.get("verbose_mode"),
            "metric_units":                llm_cfg.get("metric_units"),
            "critical_distance_threshold": llm_cfg.get("critical_distance_threshold"),
            "companion_mode":              False,
        }
        self.show_distances = True

        # UI rendering preferences (NEW)
        ui_cfg = self.config.get("ui")
        self.font_scale = ui_cfg.get("telemetry_font_scale")
        self.font_thick = ui_cfg.get("telemetry_font_thickness")

        # Intervals for rate-limited background tasks
        self.llm_interval    = llm_cfg.get("interval_seconds")
        self.depth_interval  = perception_cfg.get("depth_interval_seconds")
        self.last_llm_time   = 0.0
        self.last_depth_time = 0.0

        # Safety distance thresholds (read from config)
        safety_cfg = self.config.get("safety")
        self.safety_thresholds = {
            "critical": safety_cfg.get("critical_distance_meters"),
            "warning":  safety_cfg.get("warning_distance_meters"),
            "advisory": safety_cfg.get("advisory_distance_meters"),
            "unknown":  safety_cfg.get("unknown_obstacle_distance", 1.0),
        }
        self.nav_ttl = safety_cfg.get("navigation_ttl_seconds")

        self.ui_font_size       = ui_cfg.get("ui_font_size")

        self.decision_state = {
            "hazard_latch_msg": None,
            "hazard_latch_level": 0,
            "hazard_latch_ts": 0.0,
            "hazard_latch_duration": 1.2,
            "last_final_msg": "Path clear.",
            "last_final_level": 1,
            "last_final_ts": 0.0,
            "last_final_incremental": False,
            "instruction_sticky_duration": 2.5
        }

        # Thread synchronization primitives
        self.running          = True
        self.gpu_lock         = threading.Lock()  # Used only by the depth worker
        self.depth_lock       = threading.Lock()  # Protects latest_depth_map
        self.depth_cond       = threading.Condition()
        self.latest_depth_map = None
        self.next_depth_frame = None

        # Disk-write queue (keeps save operations off the vision thread)
        self.io_queue      = queue.Queue()
        self.fused_dir     = "data/states/fused_states"
        os.makedirs(self.fused_dir, exist_ok=True)
        self.fused_history = []

        # Display and FPS tracking
        self.display_queue = queue.Queue(maxsize=2)
        self.fps_history   = deque(maxlen=30)

        # Live navigation state — mutated by all workers and read by process_frame
        self.nav_state = {
            "immediate_hazard":           None,
            "immediate_hazard_timestamp": 0.0,
            "llm_instruction":            None,
            "llm_timestamp":              0.0,
            "path_instruction":           "Searching for path...",
            "path_instruction_timestamp": 0.0,
            "final_decision":             "Thinking. Please wait",
            "msg_level":                  0,
        }
        self._frame_counter = 0
        
        # Stability: 3-frame rule (relocated from PathAnalyzer)
        self.path_state = {
            "stability_counter": 0,
            "last_stable_instruction": "Searching for path..."
        }

        # TTS Rate-limiting and prioritization (Loaded from config)
        self.tts_state = {
            "last_spoken_msg": "",
            "last_spoken_time": 0.0,
            "last_spoken_level": 0
        }
        
        io_cfg = self.config.get("io")
        tts_cfg = io_cfg.get("tts_settings")
        self.min_speech_interval = tts_cfg.get("min_speech_interval")
        self.critical_repeat_interval = tts_cfg.get("critical_repeat_interval")
        self.warning_factor = tts_cfg.get("warning_interval_factor")

        # The last fully-processed result, returned instantly to any API caller
        self._latest_state_lock = threading.Lock()
        self._latest_state = {
            "timestamp":         0,
            "count":             0,
            "objects":           [],
            "immediate_danger":  False,
            "risk_level":        "clear",
            "msg_level":         0,
            "tts_action":        "SILENCE",
            "llm_response":      "...",
            "path_instruction":  "...",
            "final_instruction": "...",
            "depth_stats":       {"min_distance": 0.0, "mean_distance": 0.0, "max_distance": 0.0},
            "server_fps":        0.0,
            "render":            None,
        }

        # Start background threads
        self.llm_queue = queue.Queue(maxsize=1)
        self.llm_thread = threading.Thread(target=self._llm_worker, daemon=True, name="LLMWorker")
        self.llm_thread.start()

        self.depth_thread = threading.Thread(target=self._depth_worker, daemon=True, name="DepthWorker")
        self.depth_thread.start()

        self.io_thread = threading.Thread(target=self._io_worker, daemon=True, name="IOWorker")
        self.io_thread.start()

        print("[MainProcessor] System Initialized. Waiting for frames...")

    # ==========================================================================
    # Public API
    # ==========================================================================

    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Public API: Entry point for the Flask backend.
        Optimized: Resizes to inference resolution (640px) IMMEDIATELY to save RAM.
        Runs synchronously to ensure no stale/duplicate frames are returned to the client.
        """
        if frame is None:
            with self._latest_state_lock:
                return self._latest_state.copy()

        # --- OPTIMIZATION: Source-Level Downscaling ---
        h, w = frame.shape[:2]
        if max(h, w) > 640:
            scale = 640 / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)

        result = self._run_frame(frame)
        
        with self._latest_state_lock:
            self._latest_state = result
            
        # OPTIMIZATION: Flush GPU cache periodically
        if torch.cuda.is_available() and getattr(self, '_inf_count', 0) % 50 == 0:
            torch.cuda.empty_cache()
        self._inf_count = getattr(self, '_inf_count', 0) + 1
            
        return result

    def update_preferences(self, settings: dict):
        """Apply settings changes from the UI."""
        self.user_preferences["verbose_mode"]   = settings.get("descriptive_mode")
        self.user_preferences["companion_mode"] = settings.get("companion_mode")
        self.show_distances                      = settings.get("distance_estimation")
        self.dev_mode                            = settings.get("developer_mode")

        # Live Update Preprocessing Settings (if available from UI/Request)
        if "preprocessing" in settings:
            self.detector.preprocessing_cfg.update(settings["preprocessing"])
        elif "auto_enhance" in settings:
            # Flattened update support for simpler API calls
            self.detector.preprocessing_cfg["auto_enhance"] = settings["auto_enhance"]

    def stop(self):
        """Gracefully shut down all background workers."""
        self.running = False
        with self.depth_cond:
            self.depth_cond.notify_all()
        self.llm_queue.put(None)
        print("[MainProcessor] Stopped.")

    # ==========================================================================
    # Background Workers
    # ==========================================================================



    def _llm_worker(self):
        """
        Runs the language model API call in a background thread so it never
        blocks or slows down the vision loop.

        If the API returns a quota/rate-limit error it backs off 60 seconds
        and resumes — YOLO and depth keep running the whole time.
        """
        backoff_until = 0.0  # epoch time after which we are allowed to call again

        while self.running:
            try:
                payload = self.llm_queue.get(timeout=1.0)
                if payload is None:
                    break

                # Drain any stale items that piled up during a back-off
                while not self.llm_queue.empty():
                    try:
                        payload = self.llm_queue.get_nowait()
                    except queue.Empty:
                        break

                # If still in a back-off window, drop this payload and sleep.
                # The vision loop will queue a fresh request once the quota resets.
                remaining = backoff_until - time.time()
                if remaining > 0:
                    print(f"[LLM] Quota back-off active — {remaining:.0f}s remaining. Request dropped.")
                    time.sleep(min(remaining, 1.0))  # check self.running every second
                    continue

                print("[LLM] Calling API...")
                resp = (
                    self.nav_controller.generate_instructions(
                        detection_data=payload,
                        user_preferences=self.user_preferences,
                    )
                    if self.nav_controller
                    else None
                )

                if resp:
                    instruction = resp.get("instructions")
                    if instruction:
                        self.nav_state["llm_instruction"] = instruction
                        self.nav_state["llm_timestamp"]   = time.time()
                        print(f"[LLM] Result: {instruction}")

            except queue.Empty:
                continue
            except Exception as e:
                err_str = str(e).lower()
                # Quota / rate-limit: back off 60s and let the system keep running
                if any(k in err_str for k in ("quota", "rate", "resource_exhausted", "429")):
                    backoff_until = time.time() + 60.0
                    print("[LLM] Quota exceeded — pausing LLM for 60s. YOLO continues normally.")
                else:
                    print(f"[LLM] Error: {e}")

    def _depth_worker(self):
        """
        Runs depth estimation on frames queued from the main vision loop.
        Operates independently so it doesn't stall YOLO processing.
        """
        while self.running:
            with self.depth_cond:
                while self.running and self.next_depth_frame is None:
                    self.depth_cond.wait(timeout=0.1)
                if not self.running:
                    break
                target = self.next_depth_frame
                self.next_depth_frame = None

            # Small pause to let any pending YOLO calls settle on the GPU first
            time.sleep(0.01)

            try:
                t_depth_start = time.perf_counter()
                with self.gpu_lock:
                    result = self.depth_estimator.estimate_depth(target)
                with self.depth_lock:
                    self.latest_depth_map = result
                self.t_depth = (time.perf_counter() - t_depth_start) * 1000
                
                # OPTIMIZATION: Flush GPU cache after depth processing
                if torch.cuda.is_available() and getattr(self, '_depth_count', 0) % 20 == 0:
                    torch.cuda.empty_cache()
                self._depth_count = getattr(self, '_depth_count', 0) + 1
            except Exception as e:
                print(f"[Depth] Error: {e}")

    def _io_worker(self):
        """Writes fused state data to disk without blocking the vision loop."""
        while self.running or not self.io_queue.empty():
            try:
                dtype, data, ts = self.io_queue.get(timeout=0.1)
                if dtype == "fused":
                    self._save_fused_state(data, ts)
            except (queue.Empty, Exception):
                continue

    # ==========================================================================
    # Per-Frame Processing
    # ==========================================================================

    def _run_frame(self, frame: np.ndarray) -> dict:
        """
        Runs the full AI pipeline on a single frame:
          1. YOLO object detection
          2. Walkable path analysis
          3. Depth fusion (using the most recent depth map)
          4. Hazard risk assessment
          5. LLM trigger (if the interval has elapsed)
          6. Overlay drawing and FPS calculation
        """
        t_all_start = time.perf_counter()
        current_ts = time.time()
        
        # Initialize TTS state for this frame
        tts_action = "SILENCE"
        is_incremental = False
        
        # --- 0. Get Latest Depth Map Snapshot ---
        with self.depth_lock:
            local_depth = self.latest_depth_map

        # --- 1. Object Detection (YOLO) ---
        t_yolo_start = time.perf_counter()
        with self.gpu_lock:
            det_result = self.detector.detect(frame, visualize=True)
            if torch.cuda.is_available():
                torch.cuda.synchronize() # Ensure GPU finished before stopping timer
        t_yolo = (time.perf_counter() - t_yolo_start) * 1000

        detections = det_result["detections"]
        masks      = det_result["masks"]
        vis_frame  = det_result.get("vis_frame", frame.copy())

        # --- 2. Walkable Path Analysis ---
        t_path_start = time.perf_counter()
        mask_labels = []
        if masks is not None and len(masks) > 0:
            mask_labels = [None] * len(masks)
            for det in detections:
                idx = det.get("mask_idx")
                if idx is not None:
                    mask_labels[idx] = det["label"]

        # 2a. Get raw instruction from Analyzer
        # We pass local_depth and the safety warning threshold to detect 
        # when the path is coming to an end.
        raw_instr, upcoming_turn = self.path_analyzer.get_instruction(
            masks, 
            mask_labels, 
            depth_map=local_depth,
            end_threshold=self.safety_thresholds.get('warning', 2.0)
        )
        # 2b. Temporal Stability (Hysteresis) Logic
        # We only output "No walkable path detected" if it is sustained for 3+ frames. 
        # This prevents flickering when the segmentation mask is briefly noisy.
        if raw_instr == "No walkable path detected":
            self.path_state["stability_counter"] += 1
            if self.path_state["stability_counter"] >= 25: # Increased from 10 to 25 to suppress noise/flicker
                # Path is stably lost
                path_instruction = raw_instr
            else:
                # Path lost but not yet stable - use last known good result
                path_instruction = self.path_state["last_stable_instruction"]
        else:
            # Path is found - reset counter and update stable history
            self.path_state["stability_counter"] = 0
            self.path_state["last_stable_instruction"] = raw_instr
            path_instruction = raw_instr

        self.nav_state["path_instruction"]           = path_instruction
        self.nav_state["path_instruction_timestamp"] = current_ts
        
        if masks is not None and len(masks) > 0:
            self._draw_path_overlays(vis_frame)
            
        t_path = (time.perf_counter() - t_path_start) * 1000

        # --- 3. Trigger Depth Estimation (runs on its own thread) ---
        if current_ts - self.last_depth_time > self.depth_interval:
            with self.depth_cond:
                self.next_depth_frame = frame.copy()
                self.depth_cond.notify()
            self.last_depth_time = current_ts

        # --- 4. Depth Fusion & Hazard Risk ---
        t_decide = time.perf_counter()

        fused_objects = []
        risk_level    = "clear"
        if local_depth is not None:
            # First, build basic list for LLM/Saving
            fused_objects_raw = self._build_fused_objects(detections, masks, local_depth, vis_frame)
            
            # Second, pass to Preprocessor for logic-ready sorting
            fused_state = {
                "timestamp": current_ts,
                "count": len(fused_objects_raw),
                "objects": fused_objects_raw,
                "navigation_instruction": self.nav_state["path_instruction"]
            }
            all_sorted_objects = self.preprocessor.execute(fused_state, upcoming_turn)
            
            # Use refined risk assessment
            final_msg, msg_level, hazard_state = self._fuse_and_assess_risk(all_sorted_objects, upcoming_turn)
            self.nav_state["final_decision"] = final_msg
            self.nav_state["msg_level"] = msg_level
            
            fused_objects = fused_objects_raw # Return original list for JSON API compatibility
            risk_level = "danger" if msg_level == 2 else ("warning" if msg_level == 1 else "clear")

        # --- 5. LLM Trigger ---
        # We mark the time BEFORE the call to prevent re-triggering while the LLM is still running
        if local_depth is not None and current_ts - self.last_llm_time > self.llm_interval:
            self.last_llm_time = current_ts
            llm_payload = {
                "timestamp":       current_ts,
                "object_data":     fused_objects,
                "path_suggestion": self.nav_state["path_instruction"],
            }
            try:
                self.llm_queue.put_nowait(llm_payload)
            except queue.Full:
                pass  # LLM is still working on the previous request; skip this one

        # --- 6. Compute FPS first, then build header+video composite ---
        self.fps_history.append(time.perf_counter())
        fps = 0.0
        if len(self.fps_history) > 1:
            elapsed = self.fps_history[-1] - self.fps_history[0]
            if elapsed > 0:
                fps = (len(self.fps_history) - 1) / elapsed

        final_decision = self.nav_state["final_decision"]
        vis_frame = self._draw_decision_panel(vis_frame, final_decision, fps=fps)
        t_decide = (time.perf_counter() - t_decide) * 1000

        if self.show_local:
            self._push_display(vis_frame)

        # --- Build depth stats for the response ---
        depth_stats = {"min_distance": 0.0, "mean_distance": 0.0, "max_distance": 0.0}
        if local_depth is not None:
            depth_stats = {
                "min_distance":  float(np.min(local_depth)),
                "mean_distance": float(np.mean(local_depth)),
                "max_distance":  float(np.max(local_depth)),
            }

        # Only encode the frame when Dev Mode is enabled.
        # Optimized: Resolution is already 640px from Source-Level Downscaling.
        img_base64 = None
        t_encode = 0.0
        if self.dev_mode:
            t_encode_start = time.perf_counter()
            
            # Optimization: Downscale the return image to 400x400
            # The AI still processed it at 640px, but the phone screen doesn't need that much detail.
            vis_small = cv2.resize(vis_frame, (400, 400), interpolation=cv2.INTER_AREA)
            
            # Lowered to 45% JPEG to drastically reduce download latency over tunnels
            _, buffer = cv2.imencode(".jpg", vis_small, [cv2.IMWRITE_JPEG_QUALITY, 45])
            img_base64 = base64.b64encode(buffer).decode("utf-8")
            t_encode = (time.perf_counter() - t_encode_start) * 1000

        t_all = (time.perf_counter() - t_all_start) * 1000
        t_depth = getattr(self, 't_depth', 0.0)
        
        timings = {
            "YOLO": t_yolo,
            "Path": t_path,
            "Depth": t_depth,
            "Fusion": t_decide,
            "Encode": t_encode
        }
        bottleneck_name = max(timings, key=timings.get)
        bottleneck_time = timings[bottleneck_name]
        bottleneck_str = f"{bottleneck_name} ({bottleneck_time:.0f}ms)"

        profiler_data = {
            "total": float(t_all),
            "yolo": float(t_yolo),
            "path": float(t_path),
            "depth": float(t_depth),
            "logic": float(t_decide),
            "encode": float(t_encode),
            "bottleneck": bottleneck_str
        }

        # Determine current LLM instruction (respects the TTL so stale advice isn't shown)
        llm_response = (
            self.nav_state["llm_instruction"]
            if self.nav_state["llm_instruction"]
               and current_ts - self.nav_state["llm_timestamp"] < self.nav_ttl
            else "..."
        )

        # Cleanup heavy buffers immediately
        if masks is not None: del masks
        if vis_frame is not None: del vis_frame
        
        # Periodic explicit GC to prevent memory fragmentation
        self._frame_counter = getattr(self, '_frame_counter', 0) + 1
        if self._frame_counter % 100 == 0:
            gc.collect()

        # --- 7. TTS Prioritization Logic (Bombardment Mitigation) ---
        current_msg = final_decision
        current_level = self.nav_state["msg_level"]
        # `hazard_state` was populated during risk assessment. If it wasn't, default to NEW.
        h_state = hazard_state if 'hazard_state' in locals() else "NEW"
        
        time_since_speech = current_ts - self.tts_state["last_spoken_time"]
        
        # Determine if we should trigger a new TTS event
        is_new_msg = (current_msg != self.tts_state["last_spoken_msg"])
        
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
                if h_state == "NEW" and self.tts_state["last_spoken_level"] < 1:
                    tts_action = "INTERRUPT"
                elif h_state == "DELTA":
                    tts_action = "SPEAK"
                elif h_state == "SAME":
                    pass # Don't repeat identical warnings too often
        else:
            # NORMAL/ADVISORY: Speak only if new and h_state != SAME and interval passed
            if is_new_msg and h_state != "SAME" and time_since_speech > self.min_speech_interval:
                tts_action = "SPEAK"


        if tts_action != "SILENCE":
            self.tts_state["last_spoken_msg"] = current_msg
            self.tts_state["last_spoken_time"] = current_ts
            self.tts_state["last_spoken_level"] = current_level
            
            # HIGHLIGHT SPEECH IN TERMINAL
            action_color = "\033[91m" if tts_action == "INTERRUPT" else "\033[94m"
            state_color = "\033[93m" if h_state == "NEW" else "\033[90m"
            reset_color = "\033[0m"
            print(
                f"[{time.strftime('%H:%M:%S')}] "
                f"{action_color}[TTS:{tts_action}]{reset_color} "
                f"{state_color}({h_state}/L{current_level}){reset_color} "
                f"\"{current_msg}\" "
                f"({t_all:.0f}ms)"
            )
        elif self._frame_counter % 30 == 0:
            print(f"[{time.strftime('%H:%M:%S')}] [Server IDLE] Monitoring... ({t_all:.0f}ms)", end="\r")

        return {
            "timestamp":         current_ts,
            "count":             len(fused_objects),
            "objects":           fused_objects,
            "immediate_danger":  risk_level == "danger",
            "risk_level":        risk_level,
            "msg_level":         current_level,
            "tts_action":        tts_action,
            "llm_response":      llm_response,
            "path_instruction":  self.nav_state["path_instruction"],
            "final_instruction": current_msg,
            "depth_stats":       depth_stats,
            "server_fps":        float(fps),
            "render":            img_base64,  # None when dev mode is off
            "profiler":          profiler_data,
            "config": {
                "ui_font_size": self.ui_font_size
            }
        }


    # ==========================================================================
    # Helper Methods
    # ==========================================================================

    def _build_fused_objects(self, detections, masks, depth_map, vis_frame) -> list:
        """
        Combines YOLO detection results with depth map data to produce
        a list of objects enriched with real-world distance and position.
        """
        h, w = vis_frame.shape[:2]
        fused = []
        
        # Accumulate YOLO masks to find unmapped "unknown" regions
        combined_mask = np.zeros((h, w), dtype=bool)

        for det in detections:
            mask_idx = det.get("mask_idx")
            obj_mask = masks[mask_idx] if (masks is not None and mask_idx is not None) else None

            # Add to combined YOLO mask so we don't flag known objects twice
            if obj_mask is not None:
                mask_h, mask_w = obj_mask.shape[:2]
                if mask_h != h or mask_w != w:
                    resized_mask = cv2.resize(obj_mask, (w, h), interpolation=cv2.INTER_NEAREST) > 0.5
                    combined_mask |= resized_mask
                else:
                    combined_mask |= (obj_mask > 0.5)
            else:
                # If no mask, block out the bounding box
                x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
                ix1, iy1 = max(0, x1), max(0, y1)
                ix2, iy2 = min(w, x2), min(h, y2)
                if ix2 > ix1 and iy2 > iy1:
                    combined_mask[iy1:iy2, ix1:ix2] = True

            dist = self.depth_estimator.calculate_object_distance(det["bbox"], depth_map, mask=obj_mask)

            dist_str = "unknown"
            if dist is not None:
                det["distance"] = dist
                dist_str = f"{dist:.1f}m" if self.show_distances else ""

                if self.show_distances:
                    x1, y1 = det["bbox"][0], det["bbox"][1]
                    label_y = max(y1 - 10, 12)
                    cv2.putText(
                        vis_frame, f"[{dist:.1f}m] {det['label']}",
                        (int(x1), label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
                    )

            fused.append({
                "label":      det["label"],
                "confidence": det["confidence"],
                "position":   self._get_spatial_position(det["bbox"], w, h),
                "distance":   dist if dist is not None else float('inf'),
                "color":      det.get("color"),
            })

        # --- Generic Depth Obstacle Detection (Safety Net) ---
        # Only inject the generic safety net if YOLO didn't already find a critical threat ahead
        critical_dist = self.safety_thresholds['critical']
        already_has_critical_threat = any(
            obj["distance"] <= critical_dist and 
            ("center" in "".join(obj["position"]) or "ahead" in "".join(obj["position"]))
            for obj in fused
        )
        
        if not already_has_critical_threat:
            unknown_dist = self._detect_unknown_obstacles(depth_map, combined_mask, h, w)
            if unknown_dist is not None:
                fused.append({
                    "label": "unknown obstacle",
                    "confidence": 1.0, # High confidence if from raw depth cluster
                    "position": ["middle-center"], # We specifically isolated the center path
                    "distance": unknown_dist,
                    "color": (0, 0, 255) # Red for critical
                })
                if self.show_distances:
                    cv2.putText(
                        vis_frame, f"[{unknown_dist:.1f}m] Unknown Obstacle",
                        (int(w/2 - 80), int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
                    )

        return fused

    def _detect_unknown_obstacles(self, depth_map, combined_mask, h, w):
        """
        Analyzes raw depth map to find solid surfaces not caught by YOLO.
        Returns the median distance of the obstacle if found, else None.
        """
        if depth_map is None:
            return None

        unknown_dist = self.safety_thresholds['unknown']

        # --- GLOBAL UNIFORM WALL DETECTION ---
        # If 80% of the image is at a similar distance, it's likely a wall blocking the view.
        # Downsample for speed
        h_small, w_small = 40, 40
        small_depth = cv2.resize(depth_map, (w_small, h_small), interpolation=cv2.INTER_NEAREST)
        
        if not getattr(self.depth_estimator, 'is_metric', False):
            small_meters = 1.0 / (small_depth + 1e-9) * self.depth_estimator.depth_scale
        else:
            small_meters = small_depth * self.depth_estimator.depth_scale
            
        global_median = np.median(small_meters)
        if global_median < unknown_dist:
            diff = np.abs(small_meters - global_median)
            uniform_count = np.count_nonzero(diff < (global_median * 0.15))
            if uniform_count > (h_small * w_small * 0.8):
                return float(global_median)
            
        # 1. Define Collision ROI — narrow central band to avoid corridor walls
        ix1, ix2 = int(w * 0.40), int(w * 0.60)
        iy1, iy2 = int(h * 0.30), int(h * 0.55)
        
        roi_depth = depth_map[iy1:iy2, ix1:ix2]
        roi_mask = combined_mask[iy1:iy2, ix1:ix2]
        
        # 2. Filter out pixels already classified by YOLO
        roi_valid = roi_depth[~roi_mask]
        
        if roi_valid.size == 0:
            return None
            
        # 3. Convert to Metric distance (safer division to avoid overflow)
        if not getattr(self.depth_estimator, 'is_metric', False):
            roi_meters = 1.0 / (roi_valid + 1e-9) * self.depth_estimator.depth_scale
        else:
            roi_meters = roi_valid * self.depth_estimator.depth_scale
            
        roi_filtered = roi_meters[np.isfinite(roi_meters)]
        if roi_filtered.size == 0:
            return None
            
        # 4. Thresholding: solid obstacle must occupy a larger portion of the now narrower ROI
        unknown_dist = self.safety_thresholds['unknown']
        roi_area = (ix2 - ix1) * (iy2 - iy1)
        cluster_threshold = int(roi_area * 0.15)
        
        close_pixels = roi_filtered[roi_filtered < unknown_dist]
        
        if close_pixels.size > cluster_threshold:
            return float(np.median(close_pixels))
            
        return None

    def _fuse_and_assess_risk(self, all_sorted_objects, upcoming_turn_detection):
        """
        Analyze threats based on distance and label using parameter-driven responses.
        Picks the highest-priority active instruction:
          1. Immediate hazard (Level 2 / 1)
          2. LLM instruction (Level 0)
          3. Path analysis fallback (Advisory/Clear)
        """
        current_ts = time.time()
        provisional_msg = None
        provisional_level = 0
        provisional_incremental = False
        
        # 1. Check for immediate hazards (Critical/Warning)
        urgent_warning, is_critical, is_incremental = self.default_responses.get_sos_response(
            all_sorted_objects,
            critical_dist=self.safety_thresholds['critical'],
            warning_dist=self.safety_thresholds['warning'],
            path_instruction=self.nav_state.get('path_instruction'),
            upcoming_turn=upcoming_turn_detection
        )
        warning_level = 2 if is_critical else 1
        
        # --- Temporal Hazard Latch ---
        if urgent_warning:
            self.decision_state["hazard_latch_msg"] = urgent_warning
            self.decision_state["hazard_latch_level"] = warning_level
            self.decision_state["hazard_latch_ts"] = current_ts
            provisional_msg = urgent_warning
            provisional_level = warning_level
            provisional_incremental = is_incremental
        else:
            if self.decision_state["hazard_latch_msg"] and (current_ts - self.decision_state["hazard_latch_ts"] < self.decision_state["hazard_latch_duration"]):
                provisional_msg = self.decision_state["hazard_latch_msg"]
                provisional_level = self.decision_state["hazard_latch_level"]
            else:
                self.decision_state["hazard_latch_msg"] = None
                self.decision_state["hazard_latch_level"] = 0
                provisional_incremental = False

        # (Stop! suffixing removed to prevent string jitters - handled by DefaultResponse or LLM)

        # 2. Check for LLM or Advisory if no hazard found yet
        if not provisional_msg:
            llm_instruction = self.nav_state.get('llm_instruction')
            llm_age = current_ts - self.nav_state.get('llm_timestamp', 0)
            if llm_instruction and llm_age < self.nav_ttl:
                provisional_msg = llm_instruction
                lower_msg = provisional_msg.lower()
                if "stop" in lower_msg or "danger" in lower_msg or "critical hazard" in lower_msg:
                    provisional_level = 2
                elif "warning" in lower_msg or "caution" in lower_msg or "no walkable path" in lower_msg:
                    provisional_level = 1
                else:
                    provisional_level = 0
            else:
                default_message, is_incremental = self.default_responses.get_default_message_response(
                    all_sorted_objects,
                    advisory_dist=self.safety_thresholds['advisory'],
                    path_instruction=self.nav_state.get('path_instruction'),
                    upcoming_turn=upcoming_turn_detection
                )
                provisional_msg = default_message if default_message else "Path clear."
                provisional_incremental = is_incremental
                if provisional_msg == "No walkable path detected":
                    provisional_level = 1
                else:
                    provisional_level = 0

        # --- NEW: Instruction Stickiness (Human-Speed Lock) ---
        # Logic: If we recently issued a non-clear instruction, lock it for 2.5s 
        # unless a HIGHER priority instruction (Level 2) arrives.
        time_since_last = current_ts - self.decision_state["last_final_ts"]
        is_sticky = time_since_last < self.decision_state["instruction_sticky_duration"]
        
        # If we are in the sticky window, and the new instruction is NOT an emergency override (critical level 2),
        # keep the old one to prevent flickering.
        if is_sticky and self.decision_state["last_final_level"] >= 1 and provisional_level < 2:
            # We return the last message. We force state to 'SAME' because we are sticking 
            # to an old message and shouldn't trigger 'NEW' speech for a message that hasn't changed.
            return self.decision_state["last_final_msg"], self.decision_state["last_final_level"], "SAME"

        # Update the sticky state if the instruction actually changed
        if provisional_msg != self.decision_state["last_final_msg"]:
            self.decision_state["last_final_msg"] = provisional_msg
            self.decision_state["last_final_level"] = provisional_level
            self.decision_state["last_final_incremental"] = provisional_incremental
            self.decision_state["last_final_ts"] = current_ts

        return provisional_msg, provisional_level, provisional_incremental


    def _draw_text_wrapped(self, frame, text, x, y, font, scale, color, thickness, max_w):
        """Helper to draw text with simple word wrapping. Returns the next Y position."""
        if not text:
            return y
        
        words = text.split(' ')
        lines = []
        current_line = ""
        
        for word in words:
            test_line = (current_line + " " + word) if current_line else word
            (w, _), _ = cv2.getTextSize(test_line, font, scale, thickness)
            if w <= max_w:
                current_line = test_line
            else:
                if current_line: lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)
        
        for line in lines:
            (_, h), baseline = cv2.getTextSize(line, font, scale, thickness)
            cv2.putText(frame, line, (x, y), font, scale, color, thickness)
            y += h + 12 # spacing
        return y

    def _draw_decision_panel(self, frame, final_msg, fps: float = 0.0) -> np.ndarray:
        """
        Builds a header strip ABOVE the video frame containing the decision panel.
        Layout (top → bottom):
          ┌──────────────────────────────┐  ← FPS bar  (28 px, black)
          │  FPS: 12.3                   │
          ├──────────────────────────────┤  ← Info section (dark grey)
          │  SYSTEM: …                   │
          │  LLM: …                      │
          ├──────────────────────────────┤  ← Active bar (colour by risk)
          │  ACTIVE: …                   │
          └──────────────────────────────┘
          [  clean camera image below   ]
        Returns a new stacked frame so the video image is never touched.
        """
        h, w = frame.shape[:2]
        now = time.time()
        def age_str(ts): return "None" if ts <= 0 else f"{now - ts:.1f}s"

        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = self.font_scale
        thick      = self.font_thick
        max_width  = w - 20

        # ── Prepare message strings ──────────────────────────────────────
        imm_age = age_str(self.nav_state['immediate_hazard_timestamp'])
        is_imm  = self.nav_state['immediate_hazard'] == final_msg and final_msg is not None
        is_system = is_imm or (final_msg == self.nav_state['path_instruction'])
        indicator = "> " if is_system else "  "
        sys_msg_content = self.nav_state['immediate_hazard'] or self.nav_state['path_instruction']
        sys_msg  = f"{indicator}SYSTEM ({imm_age}): {sys_msg_content}"

        llm      = self.nav_state["llm_instruction"]
        is_llm   = bool(llm) and llm == final_msg and not is_imm
        llm_text = f"{'> ' if is_llm else '  '}LLM ({age_str(self.nav_state['llm_timestamp'])}): {llm or 'None'}"

        active_msg = f"ACTIVE: {final_msg}"

        # ── Measure wrapped text heights ─────────────────────────────────
        def wrapped_lines(text):
            """Split text into display lines respecting max_width."""
            if not text:
                return []
            words = text.split(' ')
            lines, cur = [], ""
            for word in words:
                test = (cur + " " + word) if cur else word
                (tw, _), _ = cv2.getTextSize(test, font, font_scale, thick)
                if tw <= max_width:
                    cur = test
                else:
                    if cur: lines.append(cur)
                    cur = word
            if cur: lines.append(cur)
            return lines

        def block_height(lines, scale, padding=10):
            total = 0
            for line in lines:
                (_, lh), _ = cv2.getTextSize(line, font, scale, thick)
                total += lh + padding
            return max(total, 0)

        sys_lines = wrapped_lines(sys_msg)
        llm_lines = wrapped_lines(llm_text)
        act_lines = wrapped_lines(active_msg)

        PAD = 8   # vertical padding around text blocks
        sys_block = block_height(sys_lines, font_scale)
        llm_block = block_height(llm_lines, font_scale)
        act_block = block_height(act_lines, 0.7)

        # Total header height: sys + llm section + active section
        info_h   = PAD + sys_block + 4 + llm_block + PAD
        active_h = PAD + act_block + PAD
        header_h = int(info_h + active_h)

        # ── Info section (dark background) ──────────────────────────────
        header = np.zeros((header_h, w, 3), dtype=np.uint8)

        # Info section (dark background)
        cv2.rectangle(header, (0, 0), (w, int(info_h)), (20, 20, 20), -1)

        # Active section (colour by risk level)
        msg_level = self.nav_state.get('msg_level', 0)
        if msg_level == 2:
            bg_color   = (0, 0, 200)
            text_color = (255, 255, 255)
        elif msg_level == 1:
            bg_color   = (0, 140, 255)
            text_color = (0, 0, 0)
        else:
            bg_color   = (50, 50, 50)
            text_color = (255, 255, 255)
        cv2.rectangle(header, (0, int(info_h)), (w, header_h), bg_color, -1)

        # ── Draw text into header ────────────────────────────────────────
        sys_color = (0, 0, 255) if is_imm else ((0, 230, 0) if is_system else (160, 160, 160))
        y = PAD
        for line in sys_lines:
            (_, lh), _ = cv2.getTextSize(line, font, font_scale, thick)
            cv2.putText(header, line, (10, y + lh), font, font_scale,
                        sys_color, thick if is_system else 1)
            y += lh + 10

        y += 4
        llm_color = (255, 220, 0) if is_llm else (160, 160, 160)
        for line in llm_lines:
            (_, lh), _ = cv2.getTextSize(line, font, font_scale, thick)
            cv2.putText(header, line, (10, y + lh), font, font_scale,
                        llm_color, thick if is_llm else 1)
            y += lh + 10

        y = int(info_h) + PAD
        for line in act_lines:
            (_, lh), _ = cv2.getTextSize(line, font, 0.7, 2)
            cv2.putText(header, line, (10, y + lh), font, 0.7, text_color, 2)
            y += lh + 10

        # ── Stack: info header / clean video ───────────────────
        return np.vstack([header, frame])

    def _draw_path_overlays(self, frame):
        """Draws the walkable corridor edges, midline and CoM onto the live video frame."""
        if not getattr(self.path_analyzer, "use_com", False):
            return

        # --- Cyan corridor edge polylines (left & right boundaries) ---
        path_bounds = getattr(self.path_analyzer, "last_path_bounds", [])
        if path_bounds and len(path_bounds) > 1:
            left_edge, right_edge = [], []
            for left_x, right_x, row_y in path_bounds:
                cx = (left_x + right_x) / 2
                half_w = (right_x - left_x) / 2 * 0.6  # inner 60 % corridor
                cl = int(cx - half_w)
                cr = int(cx + half_w)
                left_edge.append([cl, int(row_y)])
                right_edge.append([cr, int(row_y)])
            if len(left_edge) > 1:
                pts_l = np.array(left_edge,  dtype=np.int32).reshape(-1, 1, 2)
                pts_r = np.array(right_edge, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(frame, [pts_l], False, (0, 200, 255), 2)  # cyan left
                cv2.polylines(frame, [pts_r], False, (0, 200, 255), 2)  # cyan right

        # --- Green midline ---
        midline = getattr(self.path_analyzer, "last_midline", None)
        if midline and len(midline) > 1:
            pts = np.array([[cx, cy] for cx, cy in midline], dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(frame, [pts], False, (0, 255, 0), 2)

        # --- Blue CoM dot ---
        com = getattr(self.path_analyzer, "last_com", None)
        if com:
            cv2.circle(frame, com, 10, (255, 0, 0), -1)



    def _get_spatial_position(self, bbox, width, height) -> list:
        """
        Returns the grid zones (e.g. ["high-left", "middle-center"]) that a
        bounding box overlaps in a 3x3 spatial grid of the frame.
        """
        x1, y1, x2, y2 = bbox
        v_labels = ["high", "middle", "low"]
        h_labels = ["left", "center", "right"]
        zones = []

        for vi in range(3):
            for hi in range(3):
                zone_x1, zone_x2 = hi * width / 3,  (hi + 1) * width / 3
                zone_y1, zone_y2 = vi * height / 3, (vi + 1) * height / 3
                if max(x1, zone_x1) < min(x2, zone_x2) and max(y1, zone_y1) < min(y2, zone_y2):
                    zones.append(f"{v_labels[vi]}-{h_labels[hi]}")

        return zones or ["middle-center"]

    def _push_display(self, frame):
        """
        Places the annotated frame into the display queue for the OpenCV window.
        Drops the oldest frame if the queue is full to prevent lag buildup.
        """
        try:
            if self.display_queue.full():
                self.display_queue.get_nowait()
            
            # Optimization: Image is already ~640-800px from Source-Level Downscaling.
            # No need for heavy resizing here.
            debug_frame = getattr(self.path_analyzer, "last_debug_frame", None)
            self.display_queue.put_nowait({
                "main":  frame,
                "debug": debug_frame if debug_frame is not None else None,
            })
        except Exception:
            pass

    def _save_fused_state(self, data, ts):
        """
        Saves a fused detection state to disk (called from the IO worker thread).
        Keeps only the 5 most recent timestamped files to avoid filling the disk.
        """
        timestamped_path = os.path.join(self.fused_dir, f"fused_{int(ts * 1000)}.json")
        with open(timestamped_path, "w") as f:
            json.dump(data, f, indent=2)

        # Overwrite the "latest" file so other components can always find it
        latest_path = os.path.join(os.path.dirname(self.fused_dir), "fused_state.json")
        with open(latest_path, "w") as f:
            json.dump(data, f, indent=2)

        self.fused_history.append(ts)
        if len(self.fused_history) > 5:
            oldest = os.path.join(self.fused_dir, f"fused_{int(self.fused_history.pop(0) * 1000)}.json")
            if os.path.exists(oldest):
                os.remove(oldest)

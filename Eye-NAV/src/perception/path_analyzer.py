import numpy as np
import cv2
from collections import deque
from typing import List, Optional, Tuple, Union

class PathAnalyzer:
    """
    Analyzes segmentation masks to provide navigation instructions.
    Focuses on centering the 'walkable path' in the middle of the view.
    """
    def __init__(
        self, 
        density_threshold: float = 0.1, 
        use_com: bool = False, 
        use_weighted_bias: bool = False, 
        use_bev: bool = False,
        primary_labels: List[str] = None,
        fallback_labels: List[str] = None,
        forbidden_labels: List[str] = None
    ):
        """
        Initializes the PathAnalyzer with configuration parameters.
        
        Args:
            density_threshold: Minimum average pixel value in a grid cell to consider it as 'path'.
            use_com: If True, use Center of Mass (CoM) tracking via contour analysis.
            use_weighted_bias: If True, weight pixels at the bottom of the mask more heavily.
            use_bev: If True, apply a Bird's-Eye View (BEV) warp before analysis.
            primary_labels:  Highest-priority walkable surfaces (concrete, tile, sidewalk, etc.).
                             These are always preferred and never yield to fallback surfaces.
            fallback_labels: Secondary walkable surfaces (asphalt, road, rough ground, grass, dirt).
                             Only used when NO primary surface is detected in the frame.
            forbidden_labels: Surfaces that are never walkable (walls, ceilings, fences, pillars).
        """
        self.density_threshold = density_threshold
        self.use_com = use_com
        self.use_weighted_bias = use_weighted_bias
        self.use_bev = use_bev
        self.primary_labels  = primary_labels  or ["sidewalk", "footpath", "walkway", "pavement", "concrete", "path", "tile", "floor", "marble", "carpet", "hallway", "corridor", "walkable"]
        self.fallback_labels = fallback_labels or ["asphalt", "road", "rough ground", "gravel", "pebbles", "grass", "dirt"]
        self.forbidden_labels = forbidden_labels or ["wall", "ceiling", "pillar", "building", "fence"]
        
        # BEV Configuration (Defaults to heuristic if not provided)
        self.bev_src = None
        self.bev_dst = None
        
        # Temporal Smoothing: more frames → stabler steering signal
        self.history_length = 10
        self.bias_history: deque = deque(maxlen=self.history_length)

        self.last_grid: Optional[dict] = None
        self.last_path_bounds: list = []   # [(left_x, right_x, row_y), …]


    def get_instruction(self, masks: Union[np.ndarray, list, None], class_names: List[str], depth_map: Optional[np.ndarray] = None, end_threshold: float = 2.0) -> Tuple[str, bool]:
        """
        Analyze AI segmentation masks and return a verbal navigation instruction.
        
        Args:
            masks: Array or list of segmentation masks from the AI model.
            class_names: List of class labels corresponding to each mask.
            depth_map: Optional metric depth map in meters.
            end_threshold: Distance threshold (meters) to warn about the path ending.
            
        Returns:
            A tuple of (instruction_string, is_turn_detected).
        """
        self._reset_tracking_states()

        if masks is None or len(masks) == 0:
            return "No walkable path detected", False

        # 1. Combine masks matching our target path labels
        path_mask, surface_type = self._combine_path_masks(masks, class_names)

        if path_mask is None or np.sum(path_mask) == 0:
            return "No walkable path detected", False
            
        # Path found
        self.current_surface = surface_type


        # 2. Apply Bird's-Eye View warp if enabled
        if self.use_bev:
            path_mask = self._apply_bev(path_mask)

        # 3. Analyze path shape and compute instructions
  
        if self.use_com:
            raw_instruction = self._analyze_mask_com(path_mask)
            prediction = self._detect_upcoming_turn()

            # --- Depth-based 'Path Ending' check ---
            if depth_map is not None:
                is_ending = self._check_path_ending(path_mask, depth_map, end_threshold)

                # Unstable commeneted out for now.
                # if is_ending:
                #     is_turn_detected = True
                #     raw_instruction = f"Path coming to an end. Be careful. {raw_instruction}"

            if prediction:
                is_turn_detected = True # added boolean
                final_instr = f"{raw_instruction} {prediction}"
            else:
                is_turn_detected = False   # added boolean
                final_instr = raw_instruction
            
            return final_instr, is_turn_detected
        else:
            # no turn is ever detected if COM is false
            is_turn_detected = False   # added boolean
            raw_instr = self._analyze_mask_geometry(path_mask)
            
            if depth_map is not None:
                is_ending = self._check_path_ending(path_mask, depth_map, end_threshold)
                if is_ending:
                    is_turn_detected = True
                    raw_instr = f"Path coming to an end. Be careful. {raw_instr}"
            
            final_instr = raw_instr
            return final_instr, is_turn_detected

    def _check_path_ending(self, path_mask: np.ndarray, depth_map: np.ndarray, threshold: float) -> bool:
        """
        Checks if the walkable path directly in front of the user is ending.
        Focuses only on the center 40% of the frame horizontally.
        """
        h, w = path_mask.shape
        
        # 1. Focus only on the 'Center' column (e.g., 30% to 70% width)
        center_x1 = int(w * 0.3)
        center_x2 = int(w * 0.7)
        
        # 2. Find path pixels in this central ROI
        center_coords = np.argwhere(path_mask[:, center_x1:center_x2] > 0)
        
        if len(center_coords) < 10: 
            # If the center is completely empty, the path is already 'ended' or blocked in front
            # We return True to trigger the warning if there's any path elsewhere (meaning it likely just stopped)
            return True
            
        # 3. Find the farthest point (top-most) specifically in the center
        # Map ROI coords back to full image x-coord
        top_y = np.min(center_coords[:, 0])
        top_pts = center_coords[center_coords[:, 0] <= top_y + 5]
        
        # Adjust X coordinates (add center_x1 back)
        top_pts_full = top_pts.copy()
        top_pts_full[:, 1] += center_x1
        
        # 4. Sample and validate depth
        depths = depth_map[top_pts_full[:, 0], top_pts_full[:, 1]]
        valid_depths = depths[np.isfinite(depths)]
        
        if len(valid_depths) == 0:
            return False
            
        farthest_path_dist = np.median(valid_depths)
        
        # If the path in the center is too short, warn the user
        return farthest_path_dist < threshold

    def _reset_tracking_states(self) -> None:
        """Resets the visualization and tracking states."""
        self.last_debug_frame = None
        self.last_com = None
        self.last_midline = None
        self.last_path_bounds = []
        self.last_grid = None
        self.current_surface = None

    def _combine_path_masks(self, masks: Union[np.ndarray, list], class_names: List[str]) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """
        Combines masks into a single binary walkable-path mask using a three-tier priority system:

          Tier 1 — Primary surfaces (highest priority, always preferred):
            Concrete, pavement, sidewalk, tile, floor, marble, carpet, hallway, corridor, etc.

          Tier 2 — Fallback surfaces (used ONLY if no Tier 1 surface is detected):
            Asphalt, road, rough ground, grass, dirt, gravel, pebbles.

          Tier 3 — Always forbidden (never walkable):
            Walls, ceilings, pillars, buildings, fences, AND any physical object detected
            on the path (person, dog, car, bench, etc.) — all subtracted from the result.
        """
        primary_mask  = None
        fallback_mask = None
        forbidden_mask = None   # Tier 3: structural surfaces + detected objects

        surface_counts = {}

        for i, label in enumerate(class_names):
            label_lower = str(label).lower()
            m = (masks[i] > 0).astype(np.uint8)

            # --- Tier 3a: Explicitly forbidden structural surfaces ---
            if any(f in label_lower for f in self.forbidden_labels):
                forbidden_mask = m if forbidden_mask is None else np.maximum(forbidden_mask, m)
                continue

            # --- Tier 1: Primary walkable surfaces ---
            if any(p in label_lower for p in self.primary_labels):
                area = int(np.sum(m))
                if area > 0:
                    clean_label = next((p for p in self.primary_labels if p in label_lower), label_lower)
                    surface_counts[clean_label] = surface_counts.get(clean_label, 0) + area
                primary_mask = m if primary_mask is None else np.maximum(primary_mask, m)
                continue

            # --- Tier 2: Fallback surfaces ---
            if any(f in label_lower for f in self.fallback_labels):
                fallback_mask = m if fallback_mask is None else np.maximum(fallback_mask, m)
                continue

            # --- Tier 3b: Physical objects (anything not a surface) ---
            # Persons, dogs, cars, benches, bikes, etc. block the walkable area.
            forbidden_mask = m if forbidden_mask is None else np.maximum(forbidden_mask, m)

        # --- Decide which surface tier to use ---
        primary_detected = primary_mask is not None and np.sum(primary_mask) > 0

        if primary_detected:
            combined_path = primary_mask
        elif fallback_mask is not None:
            combined_path = fallback_mask
            dominant_fallback = next(
                (f for f in self.fallback_labels
                 if any(f in str(n).lower() for n in class_names)),
                "alternate surface"
            )
            surface_counts[dominant_fallback] = int(np.sum(fallback_mask))
        else:
            return None, None

        # --- Subtract all Tier-3 regions (structures + objects) from final mask ---
        if forbidden_mask is not None:
            combined_path = cv2.bitwise_and(combined_path, cv2.bitwise_not(forbidden_mask))

        if np.sum(combined_path) == 0:
            return None, None

        dominant_surface = max(surface_counts, key=surface_counts.get) if surface_counts else None
        return combined_path, dominant_surface


    def _apply_bev(self, mask: np.ndarray) -> np.ndarray:
        """
        Applies a heuristic inverse-perspective warp to the mask to flatten the ground plane.
        
        Args:
            mask: The binary path mask.
            
        Returns:
            The warped mask.
        """
        h, w = mask.shape
        # Use provided points or fallback to defaults
        if self.bev_src is not None and self.bev_dst is not None:
            src = self.bev_src
            dst = self.bev_dst
        else:
            src = np.float32([[0, h], [w, h], [w * 0.25, h * 0.5], [w * 0.75, h * 0.5]])
            dst = np.float32([[w * 0.2, h], [w * 0.8, h], [w * 0.2, 0], [w * 0.8, 0]])
            
        matrix = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(mask, matrix, (w, h), flags=cv2.INTER_LINEAR)

    def _analyze_mask_com(self, mask: np.ndarray) -> str:
        """
        Corridor-aware Center of Mass steering.

        Instead of tracking a single pixel midline, this method:
          - Computes a smoothed polynomial midline through the path centre.
          - Measures the walkable corridor width at every row.
          - Only counts rows that are wide enough for a person to walk through
            (>= MIN_CORRIDOR_PX), avoiding needle-thin edges skewing the CoM.
          - Applies a bottom-weighted, 10-frame rolling average for stability.
        """
        h, w = mask.shape
        mask_uint8 = (mask > 0).astype(np.uint8) * 255

        # 1. Isolate the main path (largest contiguous region)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            self.bias_history.clear()
            return "No clear path ahead. Stop."

        main_contour = max(contours, key=cv2.contourArea)
        clean_mask = np.zeros_like(mask_uint8)
        cv2.drawContours(clean_mask, [main_contour], -1, 255, -1)

        # 2. Compute smoothed midline + per-row path bounds
        self._calculate_midline(clean_mask)

        if not self.last_midline:
            self.bias_history.clear()
            return "Path lost. Search."

        # 3. Corridor-aware Global CoM
        # Only count rows where the walkable corridor is wide enough (~60 px @ 640px width).
        # This stops slivers at the path edge from pulling the CoM off to one side.
        MIN_CORRIDOR_PX = max(40, int(w * 0.08))   # ~8 % of frame width

        bounds_dict = {int(b[2]): (int(b[0]), int(b[1])) for b in self.last_path_bounds}

        weighted_sum_cx = 0.0
        total_weight    = 0.0

        for cx, cy in self.last_midline:
            bounds = bounds_dict.get(cy)
            if bounds is None:
                continue
            corridor_width = bounds[1] - bounds[0]
            if corridor_width < MIN_CORRIDOR_PX:
                continue   # Too narrow — skip this row
            weight = (cy / h) * 0.9 + 0.1 if self.use_weighted_bias else 1.0
            weighted_sum_cx += cx * weight
            total_weight    += weight

        if total_weight == 0:
            self.bias_history.clear()
            return "Path lost. Search."

        global_cx  = weighted_sum_cx / total_weight
        raw_bias   = (global_cx - (w / 2)) / (w / 2)
        self.bias_history.append(raw_bias)
        center_bias = float(np.mean(self.bias_history))

        # Smoothed CoM dot position (blue dot in debug view)
        smoothed_cx  = (center_bias * (w / 2)) + (w / 2)
        self.last_com = (int(smoothed_cx), int(h * 0.75))

        # 4. Debug visualisation
        self._draw_com_debug_frame(clean_mask, w, h)

        # 5. Frame-centre corridor check
        # If the frame's vertical centre column is outside the walkable corridor
        # at the bottom third of the frame, force a Bank instruction regardless
        # of the CoM bias — the user is physically walking off the path.
        frame_cx = w / 2
        bottom_bounds = [
            (lx, rx) for lx, rx, ry in self.last_path_bounds
            if ry >= int(h * 0.60)  # bottom 40% of frame = near-field ground
        ]
        if bottom_bounds:
            # Average left/right edges at the bottom
            avg_left  = float(np.mean([b[0] for b in bottom_bounds]))
            avg_right = float(np.mean([b[1] for b in bottom_bounds]))
            if frame_cx < avg_left:
                # Centre of screen is to the left of the path → steer right
                return "Bank right."
            elif frame_cx > avg_right:
                # Centre of screen is to the right of the path → steer left
                return "Bank left."

        # 6. Map CoM bias to instruction
        return self._map_bias_to_instruction(center_bias)

    def _calculate_midline(self, clean_mask: np.ndarray) -> None:
        """
        Calculates a smoothed path midline and per-row corridor bounds.

        Steps:
          1. Compute row-wise centroid (mean x) and left/right edges vectorised.
          2. Fit a low-degree polynomial through the centroids so the green line
             follows the overall path curvature instead of pixel-level jitter.
          3. Store:
             - self.last_midline      : smoothed (x, y) pairs
             - self.last_path_bounds  : (left_x, right_x, y) per valid row
        """
        coords = np.argwhere(clean_mask > 0)
        if len(coords) == 0:
            self.last_midline = []
            self.last_path_bounds = []
            return

        h, w = clean_mask.shape
        y_coords = coords[:, 0].astype(np.int32)
        x_coords = coords[:, 1].astype(np.float32)

        counts = np.bincount(y_coords, minlength=h)
        valid_rows = counts > 0
        valid_y = np.where(valid_rows)[0]

        # Row-wise mean (centroid)
        x_sums = np.bincount(y_coords, weights=x_coords, minlength=h)
        x_means = x_sums[valid_rows] / counts[valid_rows]

        # Row-wise left/right edge via vectorised min/max scatter
        row_min = np.full(h, w, dtype=np.float32)
        row_max = np.zeros(h, dtype=np.float32)
        np.minimum.at(row_min, y_coords, x_coords)
        np.maximum.at(row_max, y_coords, x_coords)
        path_left  = row_min[valid_rows]
        path_right = row_max[valid_rows]

        # --- Advanced Polynomial smoothing of the centroid line ---
        # 1. Minimum width filtering: ignore rows that are just slivers of noise
        MIN_FOR_FIT_PX = max(20, int(w * 0.04))
        widths = path_right - path_left
        fit_mask = widths >= MIN_FOR_FIT_PX
        
        if np.any(fit_mask) and len(valid_y[fit_mask]) >= 5:
            fy = valid_y[fit_mask]
            fx = x_means[fit_mask]
            
            try:
                # 2. Degree 3 allows for S-curves and complex turns
                degree = 3
                
                # 3. Weighted fit: prioritize the near-field (bottom of image) 
                # for immediate steering stability. Weights increase quadratically with Y.
                weights = (fy / h) ** 2 + 0.1
                
                poly = np.polyfit(fy, fx, degree, w=weights)
                x_smooth_raw = np.polyval(poly, valid_y)
                
                # 4. Boundary Enforcement: midline MUST stay inside the physical path
                # We clip with a small buffer to avoid touching the absolute edges
                buffer = 5
                x_smooth = np.zeros_like(x_smooth_raw)
                for i in range(len(valid_y)):
                    l_limit = path_left[i] + buffer
                    r_limit = path_right[i] - buffer
                    # If corridor is too thin for buffer, use the raw mean
                    if r_limit <= l_limit:
                        x_smooth[i] = x_means[i]
                    else:
                        x_smooth[i] = np.clip(x_smooth_raw[i], l_limit, r_limit)
            except (np.linalg.LinAlgError, TypeError):
                x_smooth = x_means
        else:
            x_smooth = x_means

        self.last_midline = list(zip(
            x_smooth.astype(int).tolist(),
            valid_y.tolist()
        ))
        self.last_path_bounds = list(zip(
            path_left.astype(int).tolist(),
            path_right.astype(int).tolist(),
            valid_y.tolist()
        ))

    def _draw_com_debug_frame(self, mask: np.ndarray, w: int, h: int) -> None:
        """Draws the debug visualisations for CoM tracking.

        Shows:
          - Walkable corridor outlined by two cyan edge polylines (inner 60% of path)
          - Smoothed green midline
          - Blue CoM dot
          - White screen-centre reference line
        """
        debug_frame = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_GRAY2BGR)

        # Draw walkable corridor as two edge polylines (left & right of inner 60%)
        if self.last_path_bounds and len(self.last_path_bounds) > 1:
            left_edge  = []
            right_edge = []
            for left_x, right_x, row_y in self.last_path_bounds:
                inset = int((right_x - left_x) * 0.20)
                cl = left_x  + inset
                cr = right_x - inset
                if cr > cl:
                    left_edge.append([cl, row_y])
                    right_edge.append([cr, row_y])

            if len(left_edge) > 1:
                pts_l = np.array(left_edge,  dtype=np.int32).reshape(-1, 1, 2)
                pts_r = np.array(right_edge, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(debug_frame, [pts_l], False, (0, 200, 255), 2)  # cyan-left
                cv2.polylines(debug_frame, [pts_r], False, (0, 200, 255), 2)  # cyan-right

        # Draw smoothed midline (green)
        if self.last_midline and len(self.last_midline) > 1:
            pts = np.array([[cx, cy] for cx, cy in self.last_midline], dtype=np.int32)
            cv2.polylines(debug_frame, [pts.reshape(-1, 1, 2)], False, (0, 255, 0), 2)

        # Draw CoM dot (blue)
        if self.last_com:
            cv2.circle(debug_frame, self.last_com, 8, (255, 0, 0), -1)

        # Draw screen-centre reference
        cv2.line(debug_frame, (w // 2, 0), (w // 2, h), (255, 255, 255), 1)

        self.last_debug_frame = debug_frame

    def _map_bias_to_instruction(self, bias: float) -> str:
        """
        Maps a steering bias (-1.0 to 1.0) to a verbal navigation command.

        Tiers (symmetric):
          |bias| < 0.15  → Path centered. Go straight.
          |bias| < 0.35  → Bank left/right   (gentle correction needed)
          |bias| < 0.60  → Slide left/right  (moderate correction)
          |bias| >= 0.60 → Sharp turn        (edge of corridor)
        """
        abs_bias = abs(bias)
        if abs_bias < 0.10:
            return "Path centered. Go straight."
        elif bias < -0.50:
            return "Sharp turn left."
        elif bias < -0.25:
            return "Slide left to center."
        elif bias < -0.10:
            return "Bank left."
        elif bias > 0.50:
            return "Sharp turn right."
        elif bias > 0.25:
            return "Slide right to center."
        else:  # 0.10 <= bias <= 0.25
            return "Bank right."
    def _detect_upcoming_turn(self) -> Optional[str]:
        """
        Predictively detects turns by comparing near-field and far-field midline centers.
        
        Returns:
            A string indicating the upcoming turn, or None if straight.
        """
        if not self.last_midline or len(self.last_midline) < 20:
            return None
            
        # 1. Divide midline into Near-field and Far-field zones
        midline = self.last_midline
        num_pts = len(midline)
        
        # We look at the top 30% (far) and bottom 30% (near)
        far_pts = midline[:int(num_pts * 0.3)]
        near_pts = midline[int(num_pts * 0.7):]
        
        if not far_pts or not near_pts:
            return None
            
        # 2. Compute the horizontal shift
        far_avg_x = np.mean([p[0] for p in far_pts])
        near_avg_x = np.mean([p[0] for p in near_pts])
        
        # Shift normalized relative to image width
        w = self.last_debug_frame.shape[1] if self.last_debug_frame is not None else 640
        shift = (far_avg_x - near_avg_x) / (w / 2)
        
        # 3. Map shift to prediction
        if abs(shift) < 0.15:
            return None # Path is relatively consistent or straight
        elif shift < -0.5:
            return "Upcoming sharp left!"
        elif shift < -0.15:
            return "Upcoming slight left."
        elif shift > 0.5:
            return "Upcoming sharp right!"
        elif shift > 0.15:
            return "Upcoming slight right."

    def _analyze_mask_geometry(self, mask: np.ndarray) -> str:
        """
        Legacy 3x3 Grid Analysis.
        Divides the screen into a 3x3 grid and computes path density in each sector.
        
        Args:
            mask: The binary path mask.
            
        Returns:
            The computed navigation instruction.
        """
        h, w = mask.shape
        grid_h, grid_w = h // 3, w // 3
        
        # Store for visualization
        self.last_grid = {
            'rows': [grid_h, 2 * grid_h],
            'cols': [grid_w, 2 * grid_w]
        }
        self.bias_history.clear() # Grid mode doesn't use temporal smoothing

        # Calculate density for each of the 9 cells
        densities = np.zeros((3, 3))
        for r in range(3):
            for c in range(3):
                y1, y2 = r * grid_h, (r + 1) * grid_h
                x1, x2 = c * grid_w, (c + 1) * grid_w
                densities[r, c] = np.mean(mask[y1:y2, x1:x2])
        
        self._draw_grid_debug_frame(mask, h, w)

        col_sums = np.sum(densities, axis=0)
        T = self.density_threshold
        
        # 1. Check for sharp turns
        if col_sums[0] > col_sums[2] and col_sums[0] > col_sums[1] * 1.5: 
            return "Sharp turn left until centered."
        if col_sums[2] > col_sums[0] and col_sums[2] > col_sums[1] * 1.5: 
            return "Sharp turn right until centered."

        # 2. Check for slight turns
        if col_sums[0] > col_sums[2] and col_sums[0] > col_sums[1] * 0.8: 
            return "Slight turn left until centered."
        if col_sums[2] > col_sums[0] and col_sums[2] > col_sums[1] * 0.8: 
            return "Slight turn right until centered."

        # 3. Check if path is centered
        if densities[2, 1] > T and densities[1, 1] > T: 
            return "Path centered. Go straight."

        # 4. Fallback: Path is in center but not bottom/middle (far ahead)
        if col_sums[1] > T: 
            return "Path ahead in distance. Move forward."

        return "No clear path ahead. Stop or search."

    def _draw_grid_debug_frame(self, mask: np.ndarray, h: int, w: int) -> None:
        """Draws the debug visualizations for Legacy Grid tracking."""
        debug_frame = cv2.cvtColor((mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        if self.last_grid:
            for gy in self.last_grid['rows']:
                cv2.line(debug_frame, (0, gy), (w, gy), (0, 255, 0), 1)
            for gx in self.last_grid['cols']:
                cv2.line(debug_frame, (gx, 0), (gx, h), (0, 255, 0), 1)
        self.last_debug_frame = debug_frame

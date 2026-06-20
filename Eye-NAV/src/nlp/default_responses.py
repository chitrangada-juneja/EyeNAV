"""
This file holds the default hazard detection responses that can 
be used when encountering an object at a distance threshold decided
by our system internally. it bypasses LLM responses
the full list of hazards is here.
prompts.py only has these for now: 
self.prioritized_hazards = [
            "person", "car", "motorcycle", "skateboard", "bicycle", 
            "vehicle", "unknown obstacle","delivery robot"
        ]

it can also be used when LLM isnt responding within a certain time limit.

For the purposes of this program, we define 3 types of distances:
critical_distance, warning_distance, advisory_distance. We define their 
threshold values within the main_controller. 
 

"""

from src.nlp.preprocessing import Preprocessing
import time

class DefaultResponse:
    def __init__(self):
        self.PERSON_LABELS      = {"person", "moving person"}
        self.VEHICLE_LABELS     = {"car", "truck", "bus", "motorcycle", "bicycle", "vehicle"}
        self.ANIMAL_LABELS      = {"dog", "cat", "animal"}
        self.FURNITURE_LABELS   = {"chair", "bench", "couch", "dining table", "furniture"}
        self.INFRASTRUCTURE_LABELS    = {"stairs", "door", "elevator", "building", "tree", "bench"}
        self.TRAFFIC_LABELS     = {"traffic light", "stop sign", "traffic sign", "fire hydrant", "parking meter", "traffic cone", "cone"}
        self.PATH_OBSTACLE_LABELS = {"pothole", "puddle", "cone", "barrier", "debris", "trash can", "bin", "rocks", "stone", "pole", "lamppost", "street light", "signpost", "emergency post"}
        self.PATH_LABELS = {
            "walkable path", "sidewalk", "tile", "floor",
            "concrete", "path", "footpath", "walkway", "rough ground",
            "carpet", "rug", "mat", "passage", "grass", "dirt"
        }
        # Keyword-based path detection: any label containing these words is a surface
        self.PATH_KEYWORDS = {
            "floor", "corridor", "hallway", "sidewalk", "footpath", "walkway",
            "pavement", "concrete", "path", "tile", "marble", "carpet", "rug",
            "mat", "passage", "grass", "dirt", "asphalt", "road surface"
        }
        # Structural/overhead elements that CAN NEVER block the walking path.
        # These are silently ignored, never announced as hazards.
        self.IGNORE_KEYWORDS = {
            "ceiling", "roof", "sky",       # overhead — cannot walk into
            "wall", "pillar", "column",     # structural background (YOLO picks these up often)
            "baseboard", "moulding",         # decorative interior elements
            "window", "mirror", "picture",   # background surfaces
            "shadow", "reflection",          # optical artefacts
        }
        self.CUSTOM_LABELS = {"delivery robot"}      #low priority
        
        self.preprocessor = Preprocessing()
        
        self.plural_exceptions = {
            "bus": "buses",
            "bench": "benches",
            "couch": "couches",
            "person": "people",
            "debris": "debris"
        }
        
        
        self.final_sos_response: str | None = None
        self.final_default_response: str | None = None
        
        # Conversational State Memory
        # {label: {"count": X, "dist": Y, "pos": Z, "last_seen": timestamp}}
        self.state_memory = {} # { label: {dist, pos, last_seen, count} }
        self.forget_timeout = 5.0 # Seconds to remember a disappeared hazard
        self.last_top_label = None # Hysteresis for hazard selection

    def _is_ignorable_label(self, label: str) -> bool:
        """
        Returns True for structural/overhead elements that can never physically
        block the walking path: ceilings, roofs, walls, windows, etc.
        These are silently dropped — never announced.
        """
        lbl = label.lower()
        return any(kw in lbl for kw in self.IGNORE_KEYWORDS)

    def _is_path_label(self, label: str) -> bool:
        """
        Returns True if the label represents a walkable surface (not a hazard).
        Uses both exact-set match and keyword substring match to catch multi-word
        labels like 'indoor corridor floor' or 'hallway floor surface'.
        """
        lbl = label.lower()
        if lbl in self.PATH_LABELS:
            return True
        return any(kw in lbl for kw in self.PATH_KEYWORDS)

    def _pluralize(self, label: str, count: int) -> str:
        if count <= 1:
            return label
        return self.plural_exceptions.get(label, f"{label}s")

    def _coarsen_distance(self, dist: float) -> str:
        """
        Rounds distance for conversational stability. 
        - Under 1.0m: Keep 0.1m precision for safety.
        - Under 5.0m: Round to nearest 0.5m.
        - Over 5.0m: Round to nearest integer.
        """
        if dist < 1.0:
            return f"{dist:.1f}"
        if dist >= 5.0:
            return f"{round(dist)}"
        # Round to nearest 0.5
        rounded = round(dist * 2) / 2
        return f"{rounded:.1f}"


    def get_sos_response(self, all_sorted_objects: dict, critical_dist: float, warning_dist: float,
                         path_instruction: str = None, upcoming_turn: bool = False) -> tuple[str | None, bool, str]: 
        """
        Evaluates immediate SOS hazards using the preprocessed dictionary structure.
        all_sorted_objects: { 'person': [{'distance': '1.2m', ...}, ...], ... }
        """
        responses = [] # List format: (is_critical, pos_score, dist_val, message_string, label, count, pos_text)

        # 1. Process all objects and generate potential messages
        for label, objects in all_sorted_objects.items():
            if not objects or self._is_path_label(label):
                continue
            
            closest_obj = objects[0]
            dist_val = closest_obj.get('distance', float('inf'))
            pos_score = self.preprocessor.get_position_score(closest_obj)
            count = len(objects)
            pos_text = self.preprocessor.get_averaged_position_text(closest_obj)

            msg = None
            is_critical = False

            if dist_val <= critical_dist:
                if pos_text == "ahead":
                    is_critical = True
                    msg = self._generate_msg(label, count, dist_val, pos_text, True)
                else:
                    msg = self._generate_msg(label, count, dist_val, pos_text, False)
            elif dist_val <= warning_dist:
                msg = self._generate_msg(label, count, dist_val, pos_text, False)

            if msg:
                responses.append((is_critical, pos_score, dist_val, msg, label, count, pos_text))

        if not responses:
            self.last_top_label = None
            return None, False, "SAME"

        # --- INTELLIGENT OBSTACLE FILTERING ---
        # If we have named hazards (person, car, etc.), suppress 'unknown obstacle' 
        # UNLESS it is the closest thing in the frame or within the critical zone.
        named_hazards = [r for r in responses if r[4] != "unknown obstacle"]
        if named_hazards:
            min_named_dist = min(r[2] for r in named_hazards)
            # Filter: Keep unknown only if it's the closest thing OR in critical zone
            responses = [r for r in responses if r[4] != "unknown obstacle" or r[2] < min_named_dist or r[2] <= critical_dist]
            
        if not responses:
            self.last_top_label = None
            return None, False, "SAME"

        # 2. Sort by: 1. Criticality (True first), 2. Distance (closest first)
        responses.sort(key=lambda x: (not x[0], x[2]))

        # --- PRIORITY HYSTERESIS ---
        if responses and self.last_top_label:
            # If the previously reported hazard is still here and close to the new top, stick with it.
            top = responses[0]
            for i in range(1, len(responses)):
                candidate = responses[i]
                if candidate[4] == self.last_top_label:
                    # If same criticality and distance diff < 0.3m, swap candidate to the top
                    if candidate[0] == top[0] and abs(candidate[2] - top[2]) < 0.3:
                        responses.pop(i)
                        responses.insert(0, candidate)
                    break
        
        if responses:
            self.last_top_label = responses[0][4]
        else:
            self.last_top_label = None

        # 3. Determine the cumulative hazard state (NEW, DELTA, or SAME)
        # We track ALL objects in memory so they don't flip-flop when priority changes.
        # But we only return the state of the objects we are actually announcing.
        final_hazard_state = "SAME"
        
        # We announce up to 3 hazards
        announced_hazards = responses[:3]
        
        has_new = False
        has_delta = False
        
        # Update memory and check state for ALL detected hazards to maintain stability
        for r in responses:
            label, count, dist, pos = r[4], r[5], r[2], r[6]
            state = self._is_incremental(label, count, dist, pos)
            
            # If this hazard is in the announced set, it contributes to the final state
            if r in announced_hazards:
                if state == "NEW": has_new = True
                elif state == "DELTA": has_delta = True
        
        if has_new: final_hazard_state = "NEW"
        elif has_delta: final_hazard_state = "DELTA"

        # 4. Assemble the final message
        final_msgs = [r[3] for r in announced_hazards]
        highest_severity_is_critical = any(r[0] for r in announced_hazards)

        # Track path instruction changes
        path_changed = False
        if path_instruction:
            if not hasattr(self, 'last_path_msg'): self.last_path_msg = None
            if path_instruction != self.last_path_msg:
                # Significant if it's not just 'straight'
                if "straight" not in path_instruction.lower():
                    path_changed = True
            self.last_path_msg = path_instruction

        if path_instruction:
            is_straight = "straight" in path_instruction.lower() or "centered" in path_instruction.lower()
            if not is_straight or not highest_severity_is_critical:
                final_msgs.append(path_instruction)
            
        if path_changed:
            final_hazard_state = "NEW" # Force speech for new path info

        self.final_sos_response = ". ".join(final_msgs)
        return self.final_sos_response, highest_severity_is_critical, final_hazard_state

    def _is_incremental(self, label: str, count: int, dist: float, pos: str) -> str:
        """
        Logic to determine if we should generate a 'Delta' message or a 'Full' message.
        Returns 'NEW', 'DELTA', or 'SAME'.
        """
        now = time.time()
        prev = self.state_memory.get(label)
        
        # 1. Reset check: If seen long ago, it's a NEW hazard
        if not prev or (now - prev['last_seen']) > self.forget_timeout:
            self.state_memory[label] = {
                "count": count, 
                "dist": dist, 
                "reported_dist": dist,  # Reference for hysteresis
                "pos": pos, 
                "last_seen": now
            }
            return "NEW"

        # 2. Check for significant deltas
        # Use the reported_dist as the reference for hysteresis, not just the previous frame
        ref_dist = prev.get('reported_dist', prev['dist'])
        dist_change = abs(dist - ref_dist)
        pos_changed = (pos != prev['pos'])
        count_changed = (count != prev['count'])
        
        # Threshold for Delta (0.5m for distance stability)
        if dist_change >= 0.5 or pos_changed or count_changed:
            # Significant change: update both memory and reference
            prev['reported_dist'] = dist
            prev['dist'] = dist
            prev['count'] = count
            prev['pos'] = pos
            prev['last_seen'] = now
            return "DELTA" # It is a delta update to an existing hazard
            
        # Refresh last_seen even if no significant change occurred to prevent forget_timeout
        prev['last_seen'] = now
        return "SAME" # No significant change, or redundant

    def _generate_msg(self, label: str, count: int, dist: float, pos: str, is_critical: bool) -> str:
        """Generates a conversational message based on state deltas and object categories."""
        now = time.time()
        prev = self.state_memory.get(label)
        
        # Use reported_dist for the message string
        msg_dist = prev.get('reported_dist', dist) if prev else dist
        coarse_dist = self._coarsen_distance(msg_dist)
            
        plural_label = self._pluralize(label, count)
        count_str = f"{count} {plural_label}" if count > 1 else label
            
        # 1. Full descriptive message (Initial detection or after forget timeout)
        if not prev or (now - prev['last_seen']) > self.forget_timeout:
            # STRICT PREFIX CONTROL: 
            # 'Warning/Danger/Stop' ONLY for is_critical (Center + < critical_dist)
            if label in self.PERSON_LABELS:
                if is_critical:
                    return f"Warning, {count_str} at {pos}, closest at {coarse_dist} metres"
                return f"{count_str.capitalize()} at {pos}, closest at {coarse_dist} metres"
            
            elif label in self.VEHICLE_LABELS:
                if is_critical:
                    return f"Danger, {count_str} at {pos}, {coarse_dist} metres"
                return f"{count_str.capitalize()} at {pos} at {coarse_dist} metres"
            
            elif label in self.ANIMAL_LABELS:
                if is_critical:
                    return f"Warning, {count_str} at {pos}, {coarse_dist} metres"
                return f"{count_str.capitalize()} at {pos} at {coarse_dist} metres"
            
            elif label in self.INFRASTRUCTURE_LABELS:
                if label == "stairs":
                    if is_critical:
                        return f"Stairs at {pos}, {coarse_dist} metres. Locate handrail"
                    return f"Stairs at {pos} at {coarse_dist} metres"
                elif label == "door":
                    if is_critical:
                        return f"Door at {pos}, {coarse_dist} metres"
                    return f"Door at {pos} at {coarse_dist} metres"
                return f"{label.capitalize()} at {pos}, {coarse_dist} metres"
            
            elif label in self.TRAFFIC_LABELS:
                if is_critical:
                    return f"Alert, {count_str} at {pos}, {coarse_dist} metres"
                return f"{count_str.capitalize()} at {pos} at {coarse_dist} metres"
            
            elif label in self.PATH_OBSTACLE_LABELS:
                if is_critical:
                    return f"Hazard, {count_str} at {pos}, {coarse_dist} metres"
                return f"{count_str.capitalize()} at {pos} at {coarse_dist} metres"
            
            else:
                # Default fallback for unknown items
                display_label = label.replace('_', ' ')
                if is_critical:
                    return f"Hazard, {display_label} at {pos}, {coarse_dist} metres"
                return f"{display_label.capitalize()} at {pos} at {coarse_dist} metres"

        # 2. Delta Logic (Concise Conversational Updates)
        # Priorities: Position > Distance > Count
        if pos != prev['pos']:
            if pos == "ahead":
                return f"{label.capitalize()} moved ahead"
            return f"{label.capitalize()} moved to the {pos}"
            
        if abs(dist - prev['dist']) > 0.5:
            if dist < prev['dist']:
                return f"{label.capitalize()} is getting closer, {coarse_dist} metres"
            return f"{label.capitalize()} is further away, {coarse_dist} metres"
            
        if count != prev['count']:
            return f"Now there are {count_str}"

        # Redundant fallback (should ideally not be spoken if is_incremental logic remains sharp)
        return f"{count_str.capitalize()} at {pos}, {coarse_dist} metres"


    def get_default_message_response(self, all_sorted_objects: dict, advisory_dist: float,
                                     path_instruction: str = None, upcoming_turn: bool = False) -> tuple[str | None, str]: 
        """
        Returns a default message for something that is not an immediate hazard.
        Uses preprocessed data to aggregate multiples.
        """
        responses = []

        for label, objects in all_sorted_objects.items():
            if not objects or self._is_path_label(label):
                continue
            
            closest_obj = objects[0]
            dist_val = closest_obj.get('distance', float('inf'))

            pos_score = self.preprocessor.get_position_score(closest_obj)

            count = len(objects)
            plural_label = self._pluralize(label, count)
            count_str = f"{count} {plural_label}" if count > 1 else label
            person_count_str = f"{count} {plural_label}"

            msg = None

            pos_text = self.preprocessor.get_averaged_position_text(closest_obj)

            if dist_val <= advisory_dist:
                msg = self._generate_msg(label, count, dist_val, pos_text, False)

            if msg:
                responses.append((pos_score, dist_val, msg, label, count, pos_text))

        if not responses:
            return (path_instruction or "Path clear."), "SAME"

        # Sort: Highest threat pos_score first, then distance.
        responses.sort(key=lambda x: (x[0], x[1]))

        # Determine state of hazards
        final_hazard_state = "SAME"
        has_new = False
        has_delta = False
        
        # Use a more conservative limit for default messages to avoid bombardment
        # We only announce the top 2 default hazards
        announced_hazards = responses[:2]
        
        for r in responses:
            # Indices: (pos_score, dist_val, msg, label, count, pos_text)
            label, count, dist, pos = r[3], r[4], r[1], r[5]
            state = self._is_incremental(label, count, dist, pos)
            
            if r in announced_hazards:
                if state == "NEW": has_new = True
                elif state == "DELTA": has_delta = True
        
        if has_new: final_hazard_state = "NEW"
        elif has_delta: final_hazard_state = "DELTA"

        final_msgs = [r[2] for r in announced_hazards]
        
        if path_instruction:
            final_msgs.append(path_instruction)
            
        self.final_default_response = ". ".join(final_msgs)
        return self.final_default_response, final_hazard_state
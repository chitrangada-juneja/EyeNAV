"""
Enhanced prompt engineering for navigation assistance.
Provides context-aware, safety-focused prompts with improved clarity.
"""

import json
from typing import Dict, Any, Optional, List


class NavigationPromptManager:
    """
    Manages system and user prompts with advanced prompt engineering.
    
    Focus areas:
    - Safety prioritization (pedestrians, vehicles first)
    - Concise, actionable language
    - Accessibility considerations
    - Real-time responsiveness
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm_cfg = config.get('llm', {})
        self.percep_cfg = config.get('perception', {})
        
        self.max_objects_detail = self.llm_cfg.get('max_scene_objects', 5)
        self.prioritized_hazards = self.llm_cfg.get('prioritized_hazards', [
            "person", "car", "motorcycle", "skateboard", "bicycle", 
            "vehicle", "unknown obstacle", "delivery robot"
        ])
        
        self.conf_levels = self.percep_cfg.get('confidence_levels', {
            "high": 0.9, "medium": 0.7, "low": 0.5
        })
    
    def get_system_prompt(self) -> str:
        """
        Get the system instruction prompt for the LLM.
        
        Returns:
            System prompt string
        """
        return """You are an AI-powered real-time navigation assistant for visually impaired users.

Your primary responsibilities:
1. **Safety First**: Identify and prioritize hazards.
2. **Clear Communication**: Provide concise, immediate, actionable instructions.
3. **Accessibility**: Use directional language (left/right/ahead), distance references, and relative positioning.
4. **Confidence**: Only report detections with sufficient confidence; ignore low-confidence noise.

Communication Style:
- Be direct and clear: "Stop - person ahead at 5 meters"
- Use spatial references: front/left/right/behind, near/far
- Provide urgency when needed: "CAUTION", "CLEAR", "PROCEED"
- Keep instructions brief (under 2 sentences for real-time use)
- Include confidence indicators when uncertain

Response Format (ALWAYS return valid JSON):
{
  "instructions": "Clear, actionable navigation instruction (1-2 sentences max)",
  "hazards": ["hazard1", "hazard2"],
  "safe_direction": "left/right/ahead/behind/none",
  "urgency": "critical/warning/caution/clear",
  "reasoning": "Brief explanation of the decision",
  "confidence": 0.0-1.0
}"""
    
    def build_user_prompt(
        self,
        detection_data: Dict[str, Any],
       
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Build the user prompt with detection data and context.
        
        Args:
            detection_data: Object detection results
            
            user_preferences: Optional user accessibility preferences
            dependent on settings in config 
            (verbose mode, critical threshold and metric units)
            
        Returns:
            Formatted user prompt string
        """
        # Prepare scene summary
        scene_summary = self._create_scene_summary(detection_data)
        
        # New: Path suggestion context (from the Path Analyzer)
        path_context = f"Internal Path Analyzer Suggestion: {detection_data.get('path_suggestion', 'None.')}\n"
        
        preferences_context = self._create_preferences_context(user_preferences) if user_preferences else ""
        
        prompt = f"""Current Scene Analysis:
            {scene_summary}
            {path_context}
           
            {preferences_context}

            Generate navigation instructions for this scene. Focus on:
            1. Immediate hazards (vehicles, pedestrians, obstacles)
            2. Safe walking direction
            3. Distance and spatial awareness
            4. Accessibility considerations

            Respond with valid JSON as specified."""
        
        return prompt
    
    def _create_scene_summary(self, detection_data: Dict[str, Any]) -> str:
        """
        Create a concise scene summary from detection data.
        
        Args:
            detection_data: Raw detection results
            
        Returns:
            Formatted scene summary
        """
        objects = detection_data.get("objects", [])
        
        if not objects:
            return "Scene: No objects detected - environment appears clear."
        
        # Prioritize hazardous objects
        prioritized = self._prioritize_objects(objects)
        
        # Limit to top objects for clarity

        # creates a summary of the scene with the most important objects,
        #  their positions, distances, and confidence levels. 
        # It prioritizes hazardous objects and limits the detail to a manageable number for the LLM to process effectively.
        summary_items = []
        for obj in prioritized[:self.max_objects_detail]:
            # Handle cases where confidence is omitted for brevity
            conf = obj.get("confidence")
            confidence_indicator = f" {self._get_confidence_indicator(conf)}" if conf is not None else ""
            
            dist = obj.get('distance', float('inf'))
            dist_str = f"{dist:.1f}m" if dist != float('inf') else "unknown"
            
            summary_items.append(
                f"  • {obj['label'].upper()} at {obj['position']} "
                f"({dist_str}){confidence_indicator}"
            )
        
        timestamp = detection_data.get("timestamp")
        object_count = len(objects)
        
        return f"""Scene Summary (timestamp: {timestamp}):
                Total objects detected: {object_count}
                Key objects:
                {chr(10).join(summary_items)}"""
        
   
    def _create_preferences_context(self, preferences: Dict[str, Any]) -> str:
        """
        Create user preference context.
        
        Args:
            preferences: User accessibility preferences
            
        Returns:
            Formatted preferences context
        """
        context = "\nUser Preferences:\n"
        
        if preferences.get("verbose_mode"):
            context += "  • Verbose mode: ENABLED (provide more detail)\n"
        else:
            context += "  • Concise mode: Keep instructions brief\n"
        
        if preferences.get("metric_units"):
            context += "  • Units: METRIC (meters)\n"
        else:
            context += "  • Units: IMPERIAL (feet/yards)\n"
        
        if critical_distance := preferences.get("critical_distance_threshold"):
            context += f"  • Critical hazard distance: {critical_distance}m\n"
        
        return context
    
    def _prioritize_objects(self, objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Sort objects by hazard priority and confidence.
        
        Args:
            objects: List of detected objects
            
        Returns:
            Prioritized list of objects
        """
        def priority_score(obj):
            # Higher priority for hazardous objects
            hazard_priority = 1000 if any(
                hazard in obj["label"].lower() for hazard in self.prioritized_hazards
            ) else 0
            
            # Higher priority for closer objects
            distance_score = -float(obj.get("distance"))
            
            # Higher priority for confident detections (default to 0.5 if missing)
            confidence_score = obj.get("confidence") * 100
            
            return hazard_priority + distance_score + confidence_score
        
        return sorted(objects, key=priority_score, reverse=True)
    
    def _get_confidence_indicator(self, confidence: float) -> str:
        """
        Get a human-readable confidence indicator.
        
        Args:
            confidence: Confidence score (0-1)
            
        Returns:
            Confidence indicator string
        """
        if confidence is None:
             return "[UNCERTAIN]"
             
        if confidence >= self.conf_levels.get('high'):
            return "[HIGH CONFIDENCE]"
        elif confidence >= self.conf_levels.get('medium'):
            return "[MEDIUM CONFIDENCE]"
        elif confidence >= self.conf_levels.get('low'):
            return "[LOW CONFIDENCE]"
        else:
            return "[UNCERTAIN]"
"""
Navigation Controller using Google GenAI SDK (Gemini).
Handles LLM-based instruction generation from vision data.
"""

import json
import time
import logging
from typing import Dict, Any, Optional
from google import genai
from google.genai import types

from .prompts import NavigationPromptManager
from .response_parser import ResponseParser

logger = logging.getLogger(__name__)


class NavigationLLMController:
    """
    Manages LLM-based navigation instruction generation using the modern Google GenAI SDK.
    
    Handles:
    - Client initialization
    - Prompt assembly and engineering
    - Response parsing and validation
    - Error handling and fallbacks
    """
    
    def __init__(self, api_key: str, config: Dict[str, Any]):
        """
        Initialize the Navigation LLM Controller.
        
        Args:
            api_key: Google GenAI API key
            config: Full system configuration dictionary
        """
        self.config = config
        self.llm_cfg = config.get('llm', {})
        self.model_name = self.llm_cfg.get('model_name', "gemini-2.5-flash")
        
        # Initialize the new Google GenAI Client
        self.client = genai.Client(api_key=api_key)
        
        self.prompt_manager = NavigationPromptManager(config)
        self.response_parser = ResponseParser()
        
        logger.info(f"Initialized NavigationLLMController (google-genai SDK) with model: {self.model_name}")
        
    def generate_instructions(
        self, 
        detection_data: Dict[str, Any],
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate navigation instructions from detection data.
        
        Args:
            detection_data: Object detection results with labels, positions, distances
            user_preferences: Optional user preferences
        
        Returns:
            Dict containing instructions, reasoning, hazards, confidence, etc.
        """
        try:
            # Build the user prompt
            user_prompt = self.prompt_manager.build_user_prompt(
                detection_data=detection_data,
                user_preferences=user_preferences
            )
            
            # Time the LLM inference
            start_time = time.perf_counter()
            
            # Generate content using the new SDK syntax
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=self.prompt_manager.get_system_prompt(),
                    response_mime_type="application/json"
                )
            )
            
            end_time = time.perf_counter()
            processing_time = end_time - start_time
            
            # Parse the response
            # Note: response.text is still available in the new SDK
            parsed_response = self.response_parser.parse_response(response.text)
            parsed_response['processing_time'] = processing_time
            
            logger.info(f"Generated instructions in {processing_time:.2f}s")
            return parsed_response
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error generating instructions: {error_msg}")
            
            # Check for Quota Exceeded error
            is_quota = "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg
            
            fallback = self._get_fallback_response(detection_data)
            if is_quota:
                fallback['quota_exceeded'] = True
            return fallback
    
    def _get_fallback_response(self, detection_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Provide fallback navigation instructions when LLM fails.
        """
        logger.warning("Using fallback response generation")
        
        hazards = []
        prioritized = self.prompt_manager.prioritized_hazards
        # Use 'object_data' to match the payload from MainProcessor
        objects = detection_data.get("object_data") or detection_data.get("objects") or []
        for obj in objects:
            if any(h in obj["label"].lower() for h in prioritized):
                hazards.append(f"{obj['label']} at {obj.get('position', ['center'])}")
        
        if hazards:
            instructions = f"Warning: Detected {', '.join(hazards)}. Proceed with caution."
        else:
            instructions = "Clear path ahead. Proceed carefully."
        
        return {
            "instructions": instructions,
            "reasoning": "Fallback rule-based response",
            "hazards": hazards,
            "confidence": 0.5,
            "processing_time": 0.0
        }
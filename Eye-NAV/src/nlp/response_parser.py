"""
Parses and validates LLM responses.
Handles JSON extraction, error recovery, and response validation.
"""

import json
import re
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ResponseParser:
    """
    Parses and validates LLM navigation responses.
    
    Handles:
    - JSON extraction and cleaning
    - Field validation
    - Fallback handling for malformed responses
    """
    
    # Expected fields in response
    REQUIRED_FIELDS = ["instructions", "hazards", "urgency"]
    OPTIONAL_FIELDS = ["reasoning", "safe_direction", "confidence"]
    
    def parse_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse LLM response text into structured format.
        
        Args:
            response_text: Raw LLM response text
            
        Returns:
            Dict with parsed response fields
        """
        try:
            # Extract JSON from response
            json_data = self._extract_json(response_text)
            
            # Validate and clean
            validated = self._validate_and_clean(json_data)
            
            return validated
            
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return self._get_fallback_parse(response_text)
    
    def _extract_json(self, text: str) -> Dict[str, Any]:
        """
        Extract JSON from response text, handling markdown fences.
        
        Args:
            text: Raw response text
            
        Returns:
            Parsed JSON dict
            
        Raises:
            json.JSONDecodeError: If no valid JSON found
        """
        # Remove markdown code fences
        cleaned = re.sub(r"```(?:json)?", "", text).strip()
        
        # Try direct JSON parse first
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON object in text
        json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # Last resort: try to extract valid JSON
        raise json.JSONDecodeError("No valid JSON found in response", text, 0)
    
    def _validate_and_clean(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and clean parsed response data.
        
        Args:
            data: Parsed JSON data
            
        Returns:
            Validated and cleaned dict
        """
        validated = {}
        
        # Validate required fields
        for field in self.REQUIRED_FIELDS:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")
            validated[field] = data[field]
        
        # Add optional fields with defaults
        validated["reasoning"] = data.get("reasoning")
        validated["safe_direction"] = data.get("safe_direction")
        validated["confidence"] = self._validate_confidence(
            data.get("confidence")
        )
        
        # Validate specific fields
        self._validate_urgency(validated["urgency"])
        self._validate_hazards(validated["hazards"])
        
        return validated
    
    def _validate_confidence(self, confidence: Any) -> float:
        """
        Validate and normalize confidence score.
        
        Args:
            confidence: Confidence value from response
            
        Returns:
            Normalized confidence score (0-1)
        """
        try:
            conf_float = float(confidence)
            return max(0.0, min(1.0, conf_float))
        except (TypeError, ValueError):
            logger.warning(f"Invalid confidence value: {confidence}, using 0.5")
            return 0.5
    
    def _validate_urgency(self, urgency: str) -> None:
        """
        Validate urgency level.
        
        Args:
            urgency: Urgency string from response
            
        Raises:
            ValueError: If urgency is invalid
        """
        valid_urgencies = ["critical", "warning", "caution", "clear"]
        if urgency.lower() not in valid_urgencies:
            logger.warning(f"Invalid urgency: {urgency}, defaulting to 'caution'")
    
    def _validate_hazards(self, hazards: Any) -> None:
        """
        Validate hazards list.
        
        Args:
            hazards: Hazards from response
            
        Raises:
            ValueError: If hazards format is invalid
        """
        if not isinstance(hazards, list):
            raise ValueError(f"Hazards must be a list, got {type(hazards)}")
    
    def _get_fallback_parse(self, response_text: str) -> Dict[str, Any]:
        """
        Generate fallback parsed response when parsing fails.
        
        Args:
            response_text: Original response text
            
        Returns:
            Fallback parsed response
        """
        logger.warning("Using fallback response parse")
        
        # Try to extract any text that looks like instructions
        instructions = response_text[:200] if response_text else "Please proceed carefully."
        
        return {
            "instructions": instructions,
            "hazards": [],
            "urgency": "caution",
            "reasoning": "Fallback response due to parse error",
            "safe_direction": "unknown",
            "confidence": 0.3
        }
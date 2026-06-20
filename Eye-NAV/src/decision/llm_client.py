
class LLMClient:
    """
    Interface for the Large Language Model (The "Small" Controller).
    
    Responsibilities:
    1.  Construct prompts from Scene State (Objects + Navigation context).
    2.  Send requests to the LLM backend (API or Local).
    3.  Receive and parse natural language responses.
    """

    def __init__(self, api_key=None, endpoint="local"):
        """
        Initialize the LLM client.
        
        Args:
            api_key (str): Authentication key (if using cloud API).
            endpoint (str): URL or identifier for the model backend.
        """
        pass

    def describe_scene(self, scene_data):
        """
        Ask the LLM to describe the scene based on structured data.
        
        Args:
            scene_data (dict): Aggregated data from Vision and Navigation modules.
                               Example: {'objects': [{'label': 'car', 'dist': 5.0}], 'next_turn': 'left'}
                               
        Returns:
            str: A natural language description suitable for TTS.
        """
        pass

    def evaluate_complexity(self, scene_data):
        """
        Optional: Ask LLM if the current situation requires user attention.
        """
        pass

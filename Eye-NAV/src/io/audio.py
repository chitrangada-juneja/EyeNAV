
class AudioInterface:
    """
    Manages all audio output for the system.
    
    Responsibilities:
    1.  Text-to-Speech (TTS): Converting LLM responses to spoken words.
    2.  Alerts: Playing immediate warning sounds (Beeps/Chimes) for safety hazards.
    """

    def __init__(self, tts_engine="coqui"):
        """
        Initialize the audio system.
        
        Args:
            tts_engine (str): config to select backend (e.g. 'elevenlabs', 'coqui', 'system').
        """
        pass

    def speak(self, text):
        """
        Convert text to speech and play it to the user.
        
        This method should be non-blocking or managed to avoid stalling the main loop.
        
        Args:
            text (str): The natural language string to speak.
        """
        pass

    def play_alert(self, alert_type):
        """
        Play a pre-defined sound effect for immediate feedback.
        
        Args:
            alert_type (str): Type of alert, e.g., "HAZARD", "TURN_LEFT", "STOP".
                              Should map to specific wav/mp3 files.
        """
        pass

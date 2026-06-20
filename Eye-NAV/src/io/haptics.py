
class HapticInterface:
    """
    Controls haptic feedback hardware (e.g., vibration motors).
    
    Used for non-verbal communication of directional cues and immediate danger warnings.
    """

    def __init__(self):
        """
        Initialize connection to the haptic controller (e.g., via Serial/GPIO).
        """
        pass

    def vibrate(self, pattern):
        """
        Trigger a vibration pattern.
        
        Args:
            pattern (str): The specific pattern to execute.
                           Examples:
                           - "PULSE_FAST": Urgent warning (Obstacle close).
                           - "PULSE_SLOW": Navigation cue.
                           - "CONTINUOUS": Stop immediately.
        """
        pass

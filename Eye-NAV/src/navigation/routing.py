
class NavigationEngine:
    """
    Handles Route Planning and GPS integration.
    
    Responsibilities:
    1.  Interface with GPS hardware to get current location.
    2.  Query Routing Engine (OSMR) for path to destination.
    3.  Provide immediate directional cues (e.g., "Turn Left in 10 meters").
    """

    def __init__(self, endpoint="http://router.project-osrm.org"):
        """
        Initialize the navigation engine.
        
        Args:
            endpoint (str): URL for the OSRM backend.
        """
        pass

    def get_next_instruction(self, current_location):
        """
        Determine the next immediate step for the user.
        
        Args:
           current_location (tuple): (latitude, longitude)
           
        Returns:
            dict: {
                'instruction': str (e.g. "Continue straight"),
                'distance': float (meters to next turn),
                'azimuth': float (degrees)
            }
        """
        pass

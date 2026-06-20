import cv2
import time
import threading
import numpy as np # Import numpy for type hinting if needed

class CameraSource:
    """
    Handles video input acquisition for the system.
    """

    def __init__(self, source=0, realtime_mode=False):
        """
        Initialize the camera source.
        
        Args:
            source (int or str): Device ID (e.g. 0 for webcam) or path to video file (str).
            realtime_mode (bool): If True, pace video file playback to maintain real-time speed.
        """
        self.source = source
        self.realtime_mode = realtime_mode
        self.cap = cv2.VideoCapture(source)
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video source: {source}")
            
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        # Default to 30 FPS if cap.get returns 0 or negative
        if self.fps <= 0: 
            self.fps = 30.0 
        
        self.is_file = isinstance(source, str)
        self.target_width = 640
        self.target_height = 640
        
        # Threading state
        self.frame: np.ndarray = None # Stores the latest captured frame
        self.ret: bool = False # Stores the success status of the last frame read
        self.running: bool = True # Controls the background thread loop
        self.lock = threading.Lock() # For thread-safe access to shared variables
        self.frame_id = 0 # Track unique frames generated
        
        # For real-time pacing of video files
        self.start_time = None # Tracks the start time for pacing calculation

        # Initialize and start the background thread
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        
        # Wait for the first frame to avoid returning None immediately
        start_wait = time.time()
        while not self.ret and time.time() - start_wait < 5.0: # 5s timeout
            time.sleep(0.01)

    def _update(self):
        """
        Background thread that continuously reads frames from the video source.
        """
        while self.running:
            # Check if the capture object is still open
            if not self.cap.isOpened():
                print(f"Warning: Video source {self.source} unexpectedly closed in update thread.")
                with self.lock:
                    self.ret = False
                self.running = False
                break

            # Real-time pacing for local video files when realtime_mode is enabled
            if self.is_file and self.realtime_mode:
                if self.start_time is None:
                    self.start_time = time.perf_counter()
                
                elapsed = time.perf_counter() - self.start_time
                
                # Get current video position in milliseconds or frames and calculate target time
                current_idx = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                target_time = current_idx / self.fps if self.fps > 0 else 0
                
                sleep_time = target_time - elapsed
                
                # Sleep to pace the frame ingestion, matching the file's natural FPS
                if sleep_time > 0:
                    time.sleep(sleep_time)
                elif sleep_time < -0.05: # Lagging by >50ms
                    # Skip frame decoder logic completely
                    self.cap.grab()
                    continue

            ret, frame = self.cap.read()
            
            with self.lock: # Acquire lock before updating shared variables
                if ret:
                    # Always resize to 640x640 for performance and consistency
                    self.frame = cv2.resize(frame, (self.target_width, self.target_height))
                    self.ret = True
                    self.frame_id += 1
                else:
                    self.ret = False
                    if self.is_file:
                        print(f"End of video file {self.source} reached.")
                    else:
                        print(f"Failed to read frame from webcam {self.source}.")
                    self.running = False # Stop the thread if frame reading fails
                    break # Exit the loop immediately if frame reading failed (especially for files)
            
            # Introduce a tiny sleep to prevent 100% CPU usage in the capture thread
            # This is particularly important for webcam feeds or when processing is faster than capture.
            # It's skipped if real-time pacing for files is active, as pacing already introduces delays.
            if not (self.is_file and self.realtime_mode):
                 time.sleep(1 / 120) # Aim for ~120 FPS max read rate for non-paced sources

    def get_frame(self, block=False) -> np.ndarray | None:
        """
        Retrieve the latest frame from the video source.
        
        Args:
            block: If True, blocks until a new, unread frame_id is available.
            
        Returns:
            numpy.ndarray: The captured frame in BGR format (OpenCV standard).
            None: If the stream has ended, fails, or no frame is currently available.
        """
        # Wait for the first frame to initialize before returning
        wait_start = time.time()
        while self.frame is None and self.running and time.time() - wait_start < 5.0:
            time.sleep(0.01)

        if block:
            last_read_id = getattr(self, '_last_read_id', -1)
            while self.running and self.frame_id <= last_read_id and self.ret:
                time.sleep(0.005)

        with self.lock: # Acquire lock to safely read shared variables
            if not self.ret or self.frame is None:
                return None
            self._last_read_id = self.frame_id
            return self.frame.copy() # Return a copy to prevent external modification of the internal buffer

    def is_running(self) -> bool:
        """
        Check if the camera source is still running and able to provide frames.
        
        Returns:
            bool: True if the source is active, False otherwise.
        """
        return self.running

    def release(self):
        """
        Release the video capture resource, stop the background thread, and free memory.
        """
        self.running = False # Signal the background thread to stop
        if self.thread.is_alive():
            self.thread.join(timeout=2.0) # Wait for the thread to finish, with a timeout
            if self.thread.is_alive():
                print(f"Warning: CameraSource thread for {self.source} did not terminate gracefully.")
        if self.cap.isOpened():
            self.cap.release()
        print(f"Camera source {self.source} released successfully.")
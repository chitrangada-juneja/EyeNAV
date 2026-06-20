
import numpy as np
import cv2
import sys
import os

def inspect_npy(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return

    print(f"--- Inspecting {file_path} ---")
    try:
        data = np.load(file_path)
    except Exception as e:
        print(f"Failed to load .npy file: {e}")
        return

    print(f"Shape: {data.shape}")
    print(f"Dtype: {data.dtype}")
    print(f"Min: {np.min(data):.4f}")
    print(f"Max: {np.max(data):.4f}")
    print(f"Mean: {np.mean(data):.4f}")

    # Visualization
    # Normalize to 0-255 for display
    if data.ndim == 2:
        # Likely a depth map or single channel mask
        data_min = np.min(data)
        data_max = np.max(data)
        
        if data_max - data_min == 0:
            print("Data is constant (cannot visualize contrast).")
            norm_data = np.zeros_like(data, dtype=np.uint8)
        else:
            norm_data = ((data - data_min) / (data_max - data_min) * 255).astype(np.uint8)
        
        # Apply colormap for better visibility (Inferno is good for depth)
        colored_map = cv2.applyColorMap(norm_data, cv2.COLORMAP_INFERNO)
        
        cv2.imshow("NPY Inspection", colored_map)
        print("Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Data is not 2D, skipping visualization.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default to the known depth map if no argument provided
        default_path = "data/states/depth_map.npy"
        print(f"Usage: python inspect_npy.py <path_to_npy_file>")
        print(f"No file provided, trying default: {default_path}")
        inspect_npy(default_path)
    else:
        inspect_npy(sys.argv[1])

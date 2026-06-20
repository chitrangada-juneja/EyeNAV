import sys
import os
import numpy as np
import cv2
from unittest.mock import MagicMock

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from perception.depth import DepthEstimator

def test_calculate_object_distance():
    print("Running verification test for calculate_object_distance...")
    
    # Mock depth map (100x100)
    # Background at 10m, object region at 2m
    depth_map = np.ones((100, 100), dtype=np.float32) * 10.0
    depth_map[40:60, 40:60] = 2.0
    
    # Mock bbox [x1, y1, x2, y2]
    bbox = [35, 35, 65, 65]
    
    # Mock mask (100x100) - only covering part of the object region inside the bbox
    mask = np.zeros((100, 100), dtype=np.float32)
    mask[45:55, 45:55] = 1.0 # 10x10 region at 2m depth
    
    # Mock estimator to avoid loading large model
    estimator = DepthEstimator.__new__(DepthEstimator)
    estimator.depth_scale = 1.0
    
    # 1. Test Disparity Mode (is_metric=False)
    # Higher raw value (e.g. 10.0) should mean CLOSER distance (1/10 = 0.1)
    estimator.is_metric = False
    dist_disparity = estimator.calculate_object_distance(bbox, depth_map, mask=mask)
    # Expected: 1 / (2.0 + 1e-6) ~= 0.5
    print(f"Disparity Mode - Distance with mask (2.0 raw): {dist_disparity:.4f}m (Expected ~0.5m)")
    assert abs(dist_disparity - 0.5) < 0.01
    
    # 2. Test without mask
    dist_no_mask = estimator.calculate_object_distance(bbox, depth_map, mask=None)
    # BBox covers 10.0 and 2.0. Median is 10.0. distance = 1/10 = 0.1
    print(f"Disparity Mode - Distance without mask: {dist_no_mask:.4f}m (Expected 0.1m)")
    assert abs(dist_no_mask - 0.1) < 0.01
    
    # 3. Test with scaling
    estimator.depth_scale = 100.0
    dist_scaled = estimator.calculate_object_distance(bbox, depth_map, mask=mask)
    # Expected: 0.5 * 100 = 50.0
    print(f"Scaling Test - Distance (100x): {dist_scaled} (Expected 50.0)")
    assert abs(dist_scaled - 50.0) < 0.01
    
    print("\nVerification successful!")

if __name__ == "__main__":
    try:
        test_calculate_object_distance()
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

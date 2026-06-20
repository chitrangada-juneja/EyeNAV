import numpy as np
import cv2
import matplotlib.pyplot as plt

# --- FILE PATHS ---
image_path = "image_06761.png"
depth_path = "depth_06761.npy"

# --- LOAD RGB IMAGE ---
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for matplotlib

# --- LOAD DEPTH MAP ---
depth = np.load(depth_path)  # shape: (H, W), dtype: float32 or float64

# --- DEPTH STATS ---
print(f"Depth shape: {depth.shape}")
print(f"Min depth: {np.min(depth):.2f} m")
print(f"Max depth: {np.max(depth):.2f} m")
print(f"Mean depth: {np.mean(depth):.2f} m")
print(f"Median depth: {np.median(depth):.2f} m")

# --- NORMALIZE DEPTH FOR VISUALIZATION ---
depth_vis = np.copy(depth)
depth_vis[depth_vis == 0] = np.nan  # Optional: hide invalid zero-depth
vmin = np.nanpercentile(depth_vis, 1)
vmax = np.nanpercentile(depth_vis, 99)

# --- PLOT SIDE BY SIDE ---
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title("RGB Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(depth_vis, cmap='plasma', vmin=vmin, vmax=vmax)
plt.title("Depth Map (in meters)")
plt.colorbar(label="Distance (m)")
plt.axis("off")

plt.tight_layout()
plt.show()

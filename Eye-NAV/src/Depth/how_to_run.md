# DepthPro Quick README

This repo uses two scripts:
- `src/Depth/run_depthpro.py` → runs DepthPro inference and saves depth outputs
- `src/Depth/visualizer.py` → displays RGB image + depth map side-by-side

## Install
pip install -r requirements.txt

## Run DepthPro
python src/Depth/run_depthpro.py --image path/to/image.jpg --outdir out --half

Arguments:
- `--image` (required): input `.jpg` / `.png`
- `--outdir` (optional): output folder (default: `out`)
- `--half` (optional): float16 on GPU for faster inference
- `--no_fov` (optional): disable FOV head to reduce compute/memory

Outputs in `out/`:
- `depth_meters.npy` (raw metric depth in meters)
- `depth_vis.png` (normalized depth visualization)

## Run Visualizer
1) Open `src/Depth/visualizer.py` and set:
   - `image_path = "path/to/image.png"`
   - `depth_path = "out/depth_meters.npy"`

2) Run:
python src/Depth/visualizer.py

Visualizer prints depth stats and shows RGB + depth map.

## Minimal Workflow
1) Run `run_depthpro.py`  
2) Use generated `out/depth_meters.npy` in `visualizer.py`  
3) Check stats + plot

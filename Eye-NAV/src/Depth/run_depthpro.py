import argparse
import os
import time
import numpy as np
import torch
from PIL import Image

from accelerate import Accelerator
from transformers import DepthProImageProcessorFast, DepthProForDepthEstimation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to an input image (jpg/png)")
    parser.add_argument("--outdir", default="out", help="Output directory")
    parser.add_argument("--no_fov", action="store_true", help="Disable FOV head to save compute/memory")
    parser.add_argument("--half", action="store_true", help="Use float16 on GPU for speed (recommended)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Device selection (Accelerate picks CUDA if available)
    device = Accelerator().device

    # Load image
    image = Image.open(args.image).convert("RGB")

    # Load processor + model
    processor = DepthProImageProcessorFast.from_pretrained("apple/DepthPro-hf")

    model_kwargs = {}
    if args.no_fov:
        model_kwargs["use_fov_model"] = False

    model = DepthProForDepthEstimation.from_pretrained("apple/DepthPro-hf", **model_kwargs)

    # Optional half precision (GPU only)
    if args.half and device.type == "cuda":
        model = model.to(dtype=torch.float16)

    model = model.to(device)
    model.eval()

    # Preprocess
    inputs = processor(images=image, return_tensors="pt").to(device)
    if args.half and device.type == "cuda":
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.float16)

    # ---------------- TIMED INFERENCE ----------------
    if device.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    with torch.no_grad():
        outputs = model(**inputs)

    if device.type == "cuda":
        torch.cuda.synchronize()

    end_time = time.perf_counter()
    inference_ms = (end_time - start_time) * 1000.0
    # -------------------------------------------------

    # Post-process back to original resolution
    pp = processor.post_process_depth_estimation(
        outputs,
        target_sizes=[(image.height, image.width)],
    )[0]

    depth_m = pp["predicted_depth"].detach().float().cpu().numpy()

    # Save raw metric depth
    npy_path = os.path.join(args.outdir, "depth_meters.npy")
    np.save(npy_path, depth_m)

    # Save visualization
    d = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
    dmin, dmax = float(d.min()), float(d.max())
    vis = (255.0 * (d - dmin) / (dmax - dmin + 1e-8)).astype(np.uint8)
    Image.fromarray(vis).save(os.path.join(args.outdir, "depth_vis.png"))

    # Print results
    cy, cx = image.height // 2, image.width // 2
    center_m = float(depth_m[cy, cx])

    print("\n=== DepthPro results ===")
    print(f"Image: {args.image}")
    print(f"Inference time:         {inference_ms:.2f} ms")
    print(f"Saved raw metric depth: {npy_path}")
    print(f"Depth range (m):        min={depth_m.min():.3f}, max={depth_m.max():.3f}")
    print(f"Center pixel depth (m): {center_m:.3f}")

    if "field_of_view" in pp:
        print(f"Estimated FOV (deg):    {float(pp['field_of_view']):.2f}")
        print(f"Estimated focal (px):   {float(pp['focal_length']):.1f}")


if __name__ == "__main__":
    main()

import os
import cv2
import torch
import numpy as np
import glob
from tqdm import tqdm


def create_directories(base_out_path):
    subdirs = ["sony_rgb", "zed_rgb", "zed_depth"]
    for sub in subdirs:
        os.makedirs(os.path.join(base_out_path, sub), exist_ok=True)


def process_rgb_frame(img_path, out_path, target_size):
    """Loads BGR image, converts to RGB, resizes, normalizes to [1], and saves as PyTorch tensor."""
    img = cv2.imread(img_path)
    if img is None:
        return False

    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Spatial downsampling to optimize VRAM (e.g., 640x360)
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

    # Tensor Normalization (0 to 1 range)
    img_normalized = img.astype(np.float32) / 255.0

    # Convert to PyTorch Tensor and reorder to [Channels, Height, Width]
    tensor = torch.from_numpy(img_normalized).permute(2, 0, 1)

    # Save as highly optimized .pt file
    torch.save(tensor, out_path.replace(".png", ".pt").replace(".jpg", ".pt"))
    return True


def process_depth_frame(img_path, out_path, target_size):
    """Loads depth map, resizes, normalizes, and saves as PyTorch tensor."""
    # Use IMREAD_UNCHANGED to preserve 16-bit depth data if applicable
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return False

    # Resize using nearest neighbor to preserve hard depth edges
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_NEAREST)

    # Normalize depth map (scaling relative to maximum depth in frame to fit [1])
    img_normalized = img.astype(np.float32)
    if img_normalized.max() > 0:
        img_normalized = img_normalized / np.max(img_normalized)

    # Convert to PyTorch Tensor and add Channel dimension -> [1, Height, Width]
    tensor = torch.from_numpy(img_normalized).unsqueeze(0)

    torch.save(tensor, out_path.replace(".png", ".pt").replace(".jpg", ".pt"))
    return True


def main():
    # Assuming step 1.4 compiled your synced frames into this directory
    INPUT_DIR = "data/synced"
    OUTPUT_DIR = "data/preprocessed_tensors"

    # Target resolution optimized for VRAM (16:9 aspect ratio downsampled from 1080p)
    TARGET_SIZE = (640, 360)

    create_directories(OUTPUT_DIR)
    print(
        f"Starting Pre-Processing. Downsampling to {TARGET_SIZE} and generating PyTorch Tensors..."
    )

    # Process Sony RGB
    sony_files = sorted(glob.glob(os.path.join(INPUT_DIR, "sony_rgb", "*.*")))
    print(f"Processing Sony RGB Tensors...")
    for f in tqdm(sony_files):
        out_path = os.path.join(OUTPUT_DIR, "sony_rgb", os.path.basename(f))
        process_rgb_frame(f, out_path, TARGET_SIZE)

    # Process ZED RGB
    zed_rgb_files = sorted(glob.glob(os.path.join(INPUT_DIR, "zed_rgb", "*.*")))
    print(f"Processing ZED RGB Tensors...")
    for f in tqdm(zed_rgb_files):
        out_path = os.path.join(OUTPUT_DIR, "zed_rgb", os.path.basename(f))
        process_rgb_frame(f, out_path, TARGET_SIZE)

    # Process ZED Depth
    zed_depth_files = sorted(glob.glob(os.path.join(INPUT_DIR, "zed_depth", "*.*")))
    print(f"Processing ZED Depth Tensors...")
    for f in tqdm(zed_depth_files):
        out_path = os.path.join(OUTPUT_DIR, "zed_depth", os.path.basename(f))
        process_depth_frame(f, out_path, TARGET_SIZE)

    print(f"\nSuccess! All frames pre-processed, normalized, and saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

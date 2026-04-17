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
    """Loads BGR, converts to RGB, resizes, scales 0-1, applies ImageNet Norm, saves as .pt"""
    img = cv2.imread(img_path)
    if img is None:
        return False

    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

    # 1. Scale to 0.0 - 1.0
    img_normalized = img.astype(np.float32) / 255.0

    # 2. Apply DeepLabV3 ImageNet Normalization globally
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_normalized = (img_normalized - mean) / std

    # Convert to Tensor [C, H, W]
    tensor = torch.from_numpy(img_normalized).permute(2, 0, 1)
    torch.save(tensor, out_path.replace(".png", ".pt").replace(".jpg", ".pt"))
    return True


def process_depth_frame(img_path, out_path, target_size):
    """Loads 16-bit depth, resizes, and saves RAW physical values (no normalization)."""
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return False

    # Nearest neighbor preserves hard physical depth edges
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_NEAREST)

    # Save the absolute scale (millimeters) as float32 to preserve spatial consistency
    img_raw_float = img.astype(np.float32)
    tensor = torch.from_numpy(img_raw_float).unsqueeze(0)

    torch.save(tensor, out_path.replace(".png", ".pt").replace(".jpg", ".pt"))
    return True


def main():
    INPUT_DIR = "data/synced"  # Pointing to the step 1.4 output
    OUTPUT_DIR = "data/preprocessed_tensors"
    TARGET_SIZE = (640, 360)

    create_directories(OUTPUT_DIR)
    print(f"Generating optimized PyTorch Tensors at {TARGET_SIZE}...")

    # Process Sony RGB
    sony_files = sorted(glob.glob(os.path.join(INPUT_DIR, "sony_rgb", "*.*")))
    print("Processing Sony RGB Tensors (with ImageNet Norm)...")
    for f in tqdm(sony_files):
        process_rgb_frame(
            f, os.path.join(OUTPUT_DIR, "sony_rgb", os.path.basename(f)), TARGET_SIZE
        )

    # Process ZED Depth
    zed_depth_files = sorted(glob.glob(os.path.join(INPUT_DIR, "zed_depth", "*.*")))
    print("Processing ZED Depth Tensors (Preserving Raw Absolute Scale)...")
    for f in tqdm(zed_depth_files):
        process_depth_frame(
            f, os.path.join(OUTPUT_DIR, "zed_depth", os.path.basename(f)), TARGET_SIZE
        )

    print(f"\nSuccess! All frames pre-processed and saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

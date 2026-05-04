"""
sync_frames.py
==============
Temporal Alignment Phase 1: Deep Feature Similarity Sync.

Uses a pre-trained ResNet18 model to extract deep visual features from
every frame, comparing the actual visual content (pose of the board)
rather than relying on motion velocity. Perfect for tripod recordings.

Dependencies: torch, torchvision, fastdtw, scipy, numpy, opencv-python
"""

import os
import sys
import glob
import json
import time
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
from fastdtw import fastdtw
from scipy.spatial.distance import cosine

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def extract_features(img_dir, model, preprocess, device, batch_size=64):
    """Extracts ResNet feature vectors for all images in a directory."""
    files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    if not files:
        return np.array([]), []

    features = []
    batch_imgs = []

    print(
        f"  -> Extracting features for {len(files)} frames in {os.path.basename(img_dir)}..."
    )

    model.eval()
    with torch.no_grad():
        for i, f in enumerate(files):
            img = Image.open(f).convert("RGB")
            batch_imgs.append(preprocess(img))

            if len(batch_imgs) == batch_size or i == len(files) - 1:
                batch_tensor = torch.stack(batch_imgs).to(device)
                # Output shape is (Batch, 512)
                feats = model(batch_tensor).cpu().numpy()
                features.extend(feats)
                batch_imgs = []

                if (i + 1) % 500 == 0 or i == len(files) - 1:
                    print(f"     Processed {i + 1} / {len(files)} frames")

    return np.array(features), files


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Deep Similarity Sync")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    args = parser.parse_args()

    # Paths
    ext_sony = os.path.join(BASE_DIR, "data", "extracted", args.dataset, "sony_rgb")
    ext_zed = os.path.join(base_dir, "data", "extracted", args.dataset, "zed_rgb")
    out_dir = os.path.join(BASE_DIR, "data", "json_output", args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, "frame_mapping.json")

    if not os.path.isdir(ext_sony) or not os.path.isdir(ext_zed):
        print(f"[ERROR] Extracted directories not found for {args.dataset}.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  AI Similarity Temporal Alignment — {args.dataset}")
    print(f"{'='*60}\n")

    # Initialize PyTorch & ResNet
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using compute device: {device.type.upper()}")

    # Load ResNet18 (Fast and excellent at structural similarity)
    weights = ResNet18_Weights.DEFAULT
    preprocess = weights.transforms()
    model = resnet18(weights=weights)

    # Remove the final classification layer to get the raw 512-D feature vector
    model.fc = torch.nn.Identity()
    model = model.to(device)

    # 1. Extract Features
    start_time = time.time()
    print("\n[Step 1/2] Analyzing Image Content...")
    sony_feats, sony_files = extract_features(ext_sony, model, preprocess, device)
    zed_feats, zed_files = extract_features(ext_zed, model, preprocess, device)

    if len(sony_feats) == 0 or len(zed_feats) == 0:
        print("[ERROR] No images found to process.")
        sys.exit(1)

    # 2. Dynamic Time Warping (using Cosine Distance)
    print("\n[Step 2/2] Aligning timelines using FastDTW (Cosine Similarity)...")
    print("           (This compares the geometry of every frame. Please wait...)")

    # Cosine distance compares the "shape" of the feature vectors, ignoring brightness differences
    distance, path = fastdtw(sony_feats, zed_feats, dist=cosine)

    print(f"  -> DTW Alignment Complete. Total Distance Cost: {distance:.4f}")

    # 3. Create Mapping Dictionary
    # We must ensure the mapping flows correctly for smooth_frames_mapping.py
    # Keys are 1-based index (string), Values are 1-based index (int)
    mapping = {}
    for sony_idx, zed_idx in path:
        str_idx = str(sony_idx + 1)
        # If multiple ZED frames map to one Sony frame, DTW keeps the last one
        mapping[str_idx] = int(zed_idx + 1)

    # 4. Save
    with open(out_json, "w") as f:
        json.dump(mapping, f, indent=4)

    end_time = time.time()
    print(f"\n[SUCCESS] Frame mapping saved -> {out_json}")
    print(f"[INFO] Total execution time: {(end_time - start_time):.1f} seconds\n")


if __name__ == "__main__":
    # Ensure correct base_dir definition for ext_zed as well
    base_dir = BASE_DIR
    main()

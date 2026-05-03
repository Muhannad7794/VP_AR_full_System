"""
preprocess_zed_frames.py
========================
Phase 2 preprocessing step for the VP_AR_full_System spatial alignment pipeline.

ZED camera frames are darker, have lower local contrast, and the board appears
smaller than in Sony frames due to the wider field of view. This script applies
a multi-step enhancement pipeline to ZED frames before calibration, saving
improved copies to a separate directory so the originals are preserved.

Processing pipeline applied to each ZED frame:
    1. Gamma correction   – brightens the overall dark ZED exposure
    2. CLAHE              – enhances local contrast across the board pattern
    3. Bilateral filter   – reduces noise while preserving sharp edges
    4. Unsharp mask       – increases apparent sharpness of marker edges

Output goes to:
    data/preprocessed/<dataset>/zed_rgb/   ← enhanced ZED frames
    data/preprocessed/<dataset>/sony_rgb/  ← Sony frames copied unchanged

The calibration pipeline then reads from data/preprocessed/ instead of
data/picked_for_alignment/.

Usage:
    python3 spatial_alignment/preprocess_zed_frames.py --dataset <name>
    python3 spatial_alignment/preprocess_zed_frames.py --dataset <name> --gamma 1.8
    python3 spatial_alignment/preprocess_zed_frames.py --dataset <name> --show_comparison
"""

import argparse
import os
import shutil
import sys

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_paths(dataset: str) -> dict:
    return {
        "sony_in": os.path.join(
            BASE_DIR, "data", "picked_for_alignment", dataset, "sony_rgb"
        ),
        "zed_in": os.path.join(
            BASE_DIR, "data", "picked_for_alignment", dataset, "zed_rgb"
        ),
        "sony_out": os.path.join(BASE_DIR, "data", "preprocessed", dataset, "sony_rgb"),
        "zed_out": os.path.join(BASE_DIR, "data", "preprocessed", dataset, "zed_rgb"),
        "compare": os.path.join(
            BASE_DIR, "data", "plots", dataset, "spatial", "preprocessing"
        ),
    }


def gamma_correction(gray: np.ndarray, gamma: float) -> np.ndarray:
    """
    Apply gamma correction to brighten or darken a grayscale image.
    gamma > 1.0 brightens, gamma < 1.0 darkens.
    A lookup table is used for efficiency.
    """
    inv_gamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8
    )
    return cv2.LUT(gray, table)


def apply_clahe(
    gray: np.ndarray, clip_limit: float = 3.0, tile_size: int = 8
) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization.
    Enhances local contrast without amplifying noise globally.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(gray)


def bilateral_denoise(gray: np.ndarray) -> np.ndarray:
    """
    Apply bilateral filter to reduce noise while preserving edges.
    Stronger than Gaussian blur at keeping sharp marker edges.
    """
    return cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)


def unsharp_mask(gray: np.ndarray, strength: float = 1.5) -> np.ndarray:
    """
    Apply unsharp masking to increase apparent edge sharpness.
    Creates a blurred version and subtracts it from the original,
    amplifying high-frequency edge detail.
    """
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=2.0)
    sharpened = cv2.addWeighted(gray, 1.0 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def enhance_zed_frame(bgr: np.ndarray, gamma: float = 1.6) -> np.ndarray:
    """
    Full enhancement pipeline for a single ZED BGR frame.

    Steps:
        1. Convert to grayscale for processing
        2. Gamma correction to brighten dark exposure
        3. CLAHE for local contrast enhancement
        4. Bilateral filter for edge-preserving noise reduction
        5. Unsharp mask for sharpness boost
        6. Return as BGR so it can be saved as a standard image

    Parameters
    ----------
    bgr   : input BGR image (ZED frame)
    gamma : gamma value for brightening. 1.0 = no change.
            1.4-2.0 is typical for dark ZED frames.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Step 1: Gamma correction
    bright = gamma_correction(gray, gamma)

    # Step 2: CLAHE
    contrasted = apply_clahe(bright, clip_limit=3.0, tile_size=8)

    # Step 3: Bilateral denoise
    denoised = bilateral_denoise(contrasted)

    # Step 4: Unsharp mask
    sharp = unsharp_mask(denoised, strength=1.2)

    # Return as BGR for consistent saving
    return cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)


def save_comparison(
    original_bgr: np.ndarray, enhanced_bgr: np.ndarray, fname: str, out_dir: str
):
    """
    Save a side-by-side comparison of original and enhanced ZED frames.
    Useful for verifying the preprocessing visually.
    """
    h, w = original_bgr.shape[:2]
    scale = min(1.0, 640 / w)
    thumb_size = (int(w * scale), int(h * scale))

    orig_thumb = cv2.resize(original_bgr, thumb_size)
    enh_thumb = cv2.resize(enhanced_bgr, thumb_size)

    # Add labels
    cv2.rectangle(orig_thumb, (0, 0), (200, 35), (0, 0, 0), -1)
    cv2.putText(
        orig_thumb,
        "ORIGINAL",
        (5, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    cv2.rectangle(enh_thumb, (0, 0), (200, 35), (0, 0, 0), -1)
    cv2.putText(
        enh_thumb, "ENHANCED", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2
    )

    comparison = np.hstack([orig_thumb, enh_thumb])
    out_path = os.path.join(out_dir, f"compare_{fname}")
    cv2.imwrite(out_path, comparison)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess ZED frames for improved ChArUco detection."
    )
    parser.add_argument(
        "--dataset", required=True, help="Dataset name inside picked_for_alignment/"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.6,
        help="Gamma for brightening ZED frames. "
        "1.0=no change, 1.4-2.0 for dark frames. "
        "Default: 1.6",
    )
    parser.add_argument(
        "--show_comparison",
        action="store_true",
        help="Save side-by-side comparison images to "
        "data/plots/<dataset>/spatial/preprocessing/",
    )
    args = parser.parse_args()

    paths = build_paths(args.dataset)

    for key in ["sony_in", "zed_in"]:
        if not os.path.isdir(paths[key]):
            print(f"[ERROR] Directory not found: {paths[key]}")
            sys.exit(1)

    os.makedirs(paths["sony_out"], exist_ok=True)
    os.makedirs(paths["zed_out"], exist_ok=True)
    if args.show_comparison:
        os.makedirs(paths["compare"], exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  ZED Frame Preprocessing – dataset: {args.dataset}")
    print(f"  Gamma: {args.gamma}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------ #
    # Copy Sony frames unchanged
    # ------------------------------------------------------------------ #
    sony_files = sorted(
        f
        for f in os.listdir(paths["sony_in"])
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    print(f"[INFO] Copying {len(sony_files)} Sony frames (unchanged) …")
    for fname in sony_files:
        shutil.copy2(
            os.path.join(paths["sony_in"], fname),
            os.path.join(paths["sony_out"], fname),
        )
    print(f"       Done.\n")

    # ------------------------------------------------------------------ #
    # Enhance ZED frames
    # ------------------------------------------------------------------ #
    zed_files = sorted(
        f
        for f in os.listdir(paths["zed_in"])
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    print(f"[INFO] Enhancing {len(zed_files)} ZED frames …")

    for fname in zed_files:
        in_path = os.path.join(paths["zed_in"], fname)
        out_path = os.path.join(paths["zed_out"], fname)

        original = cv2.imread(in_path)
        if original is None:
            print(f"  [WARN] Could not read: {fname}")
            continue

        enhanced = enhance_zed_frame(original, gamma=args.gamma)
        cv2.imwrite(out_path, enhanced)

        if args.show_comparison:
            save_comparison(original, enhanced, fname, paths["compare"])

        print(f"  ✓ {fname}")

    print(f"\n[INFO] Preprocessed frames saved to:")
    print(f"       Sony: {paths['sony_out']}")
    print(f"       ZED:  {paths['zed_out']}")
    if args.show_comparison:
        print(f"       Comparisons: {paths['compare']}")

    print(f"\n{'='*60}")
    print(f"  PREPROCESSING COMPLETE")
    print(f"{'='*60}\n")
    print("  Calibration pipeline will now use:")
    print(f"  data/preprocessed/{args.dataset}/")
    print()


if __name__ == "__main__":
    main()

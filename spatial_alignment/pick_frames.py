"""
pick_frames.py
==============
Scans data/synced/<dataset>/sony_rgb/ to find the best 40-50 frame pairs
for spatial (ChArUco stereo) calibration.

Selection criteria per frame:
  1. ChArUco board must be detected with >= 6 corners in the Sony frame.
  2. Frame must pass a minimum sharpness threshold (Laplacian variance).
  3. Frames are selected using greedy farthest-point sampling across four
     geometric features so the final set covers all four categories required
     by the calibration pipeline:
       - board centre (cx, cy)   → position diversity  (Edges)
       - board apparent area     → distance diversity   (Depths)
       - board bounding-box tilt → angle/rotation diversity (Angles + Rotations)
  4. For every candidate chosen from Sony, the matching ZED frame is verified
     to also yield a successful ChArUco detection before it is accepted.

Output:
  data/picked_for_alignment/<dataset>/sony_rgb/
  data/picked_for_alignment/<dataset>/zed_rgb/

Usage:
    python3 spatial_alignment/pick_frames.py --dataset dataset_03
    python3 spatial_alignment/pick_frames.py --dataset dataset_03 --count 45 --stride 10
"""

import argparse
import glob
import os
import shutil
import sys

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Board config — must match inspect_detections.py / stereo_calibrate.py
# ---------------------------------------------------------------------------
BOARD_COLS = 11
BOARD_ROWS = 8
SQUARE_LENGTH = 0.028
MARKER_LENGTH = 0.021
ARUCO_DICT_ID = cv2.aruco.DICT_4X4_50

DEFAULT_TARGET = 45
DEFAULT_STRIDE = 10  # scan every Nth synced frame for candidates
MIN_CORNERS = 6  # minimum ChArUco corners to accept a frame
SHARPNESS_PERCENTILE = 25  # discard frames below this Laplacian-variance percentile

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------
def _make_board(legacy: bool):
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    board = cv2.aruco.CharucoBoard(
        size=(BOARD_COLS, BOARD_ROWS),
        squareLength=SQUARE_LENGTH,
        markerLength=MARKER_LENGTH,
        dictionary=dictionary,
    )
    board.setLegacyPattern(legacy)
    return board


def detect_sony(gray: np.ndarray, board) -> np.ndarray | None:
    detector = cv2.aruco.CharucoDetector(board)
    corners, ids, _, _ = detector.detectBoard(gray)
    if corners is None or len(corners) < MIN_CORNERS:
        return None
    return corners


def detect_zed(gray: np.ndarray) -> np.ndarray | None:
    """Try both legacy settings; return the result with the most corners."""
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    best_corners = None
    best_count = 0
    for legacy in [True, False]:
        board = cv2.aruco.CharucoBoard(
            size=(BOARD_COLS, BOARD_ROWS),
            squareLength=SQUARE_LENGTH,
            markerLength=MARKER_LENGTH,
            dictionary=dictionary,
        )
        board.setLegacyPattern(legacy)
        det = cv2.aruco.CharucoDetector(board)
        corners, _, _, _ = det.detectBoard(gray)
        n = len(corners) if corners is not None else 0
        if n > best_count:
            best_count = n
            best_corners = corners
    return best_corners if best_count >= MIN_CORNERS else None


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def frame_features(corners: np.ndarray, h: int, w: int) -> np.ndarray:
    """
    Return a 4-element feature vector:
      [cx, cy, area, aspect]

    cx, cy   : board centroid, normalised to [0, 1] frame dimensions
    area     : bounding-box area as fraction of frame area — proxy for distance
    aspect   : bounding-box width/height ratio — proxy for tilt / rotation angle
    """
    pts = corners.reshape(-1, 2)
    cx = float(pts[:, 0].mean() / w)
    cy = float(pts[:, 1].mean() / h)
    x_min, x_max = float(pts[:, 0].min()), float(pts[:, 0].max())
    y_min, y_max = float(pts[:, 1].min()), float(pts[:, 1].max())
    bbox_w = (x_max - x_min) / w
    bbox_h = (y_max - y_min) / h
    area = bbox_w * bbox_h
    aspect = bbox_w / max(bbox_h, 1e-6)
    return np.array([cx, cy, area, aspect], dtype=np.float32)


def laplacian_sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# ---------------------------------------------------------------------------
# Greedy farthest-point selection
# ---------------------------------------------------------------------------
def farthest_point_select(
    features: list[np.ndarray],
    scores: list[float],
    n: int,
) -> list[int]:
    """
    Greedy farthest-point sampling in normalised feature space.
    Starts with the highest-scoring (sharpest) frame, then repeatedly
    picks the candidate that is farthest from the already-chosen set.
    This guarantees maximum geometric coverage with n picks.
    """
    feat_arr = np.array(features, dtype=np.float64)
    f_min = feat_arr.min(axis=0)
    f_max = feat_arr.max(axis=0)
    f_range = np.where(f_max - f_min > 0, f_max - f_min, 1.0)
    feat_norm = (feat_arr - f_min) / f_range

    score_arr = np.array(scores, dtype=np.float64)
    selected: list[int] = []
    selected.append(int(np.argmax(score_arr)))

    min_dists = np.full(len(feat_norm), np.inf)

    for _ in range(n - 1):
        if not selected:
            break
        last_f = feat_norm[selected[-1]]
        dists = np.linalg.norm(feat_norm - last_f, axis=1)
        min_dists = np.minimum(min_dists, dists)
        min_dists[selected] = -1.0  # exclude already-selected
        next_idx = int(np.argmax(min_dists))
        if min_dists[next_idx] < 0:
            break  # all remaining are already selected
        selected.append(next_idx)

    return selected


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Pick the best calibration frame pairs from a synced dataset."
    )
    parser.add_argument(
        "--dataset", required=True, help="Dataset name (e.g. dataset_03)"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_TARGET,
        help=f"Target number of frame pairs (default {DEFAULT_TARGET})",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=DEFAULT_STRIDE,
        help=f"Scan every Nth frame (default {DEFAULT_STRIDE})",
    )
    args = parser.parse_args()

    src_sony = os.path.join(BASE_DIR, "data", "synced", args.dataset, "sony_rgb")
    src_zed = os.path.join(BASE_DIR, "data", "synced", args.dataset, "zed_rgb")
    dst_sony = os.path.join(
        BASE_DIR, "data", "picked_for_alignment", args.dataset, "sony_rgb"
    )
    dst_zed = os.path.join(
        BASE_DIR, "data", "picked_for_alignment", args.dataset, "zed_rgb"
    )

    for path, label in [(src_sony, "sony_rgb"), (src_zed, "zed_rgb")]:
        if not os.path.isdir(path):
            print(f"[ERROR] Source directory not found: {path}")
            sys.exit(1)

    os.makedirs(dst_sony, exist_ok=True)
    os.makedirs(dst_zed, exist_ok=True)

    all_sony = sorted(glob.glob(os.path.join(src_sony, "*.png")))
    all_zed = sorted(glob.glob(os.path.join(src_zed, "*.png")))

    print(f"\n{'='*60}")
    print(f"  Frame Picker — {args.dataset}")
    print(f"  Total synced frames : Sony={len(all_sony)}  ZED={len(all_zed)}")
    print(f"  Scanning every {args.stride}th Sony frame for ChArUco detections...")
    print(f"{'='*60}\n")

    sony_board = _make_board(legacy=True)

    # -----------------------------------------------------------------------
    # Phase 1 — Scan Sony frames
    # -----------------------------------------------------------------------
    candidates: list[tuple[str, np.ndarray, float]] = []  # (fname, features, sharpness)

    sampled = all_sony[:: args.stride]
    for i, sony_path in enumerate(sampled):
        fname = os.path.basename(sony_path)
        gray = cv2.imread(sony_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue

        corners = detect_sony(gray, sony_board)
        if corners is None:
            continue

        h, w = gray.shape
        feat = frame_features(corners, h, w)
        sharp = laplacian_sharpness(gray)
        candidates.append((fname, feat, sharp))

        if (i + 1) % 50 == 0 or i == len(sampled) - 1:
            print(
                f"  Scanned {i + 1}/{len(sampled)} sampled frames "
                f"— {len(candidates)} with board detected"
            )

    print(
        f"\n  {len(candidates)} frames passed ChArUco detection (>= {MIN_CORNERS} corners)."
    )

    if not candidates:
        print(
            "[ERROR] No frames with ChArUco detection found. "
            "Check that this is a calibration recording."
        )
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Phase 2 — Sharpness filter (discard blurriest quartile)
    # -----------------------------------------------------------------------
    sharpnesses = [c[2] for c in candidates]
    threshold = float(np.percentile(sharpnesses, SHARPNESS_PERCENTILE))
    sharp_cands = [c for c in candidates if c[2] >= threshold]
    print(
        f"  {len(sharp_cands)} frames after sharpness filter "
        f"(Laplacian variance >= {threshold:.1f}).\n"
    )

    if len(sharp_cands) < 2:
        print(
            "[ERROR] Too few sharp frames to select from. "
            "Try a smaller --stride or check the recording quality."
        )
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Phase 3 — Greedy farthest-point selection for geometric diversity
    # -----------------------------------------------------------------------
    features = [c[1] for c in sharp_cands]
    scores = [c[2] for c in sharp_cands]
    n_select = min(args.count, len(sharp_cands))

    selected_indices = farthest_point_select(features, scores, n_select)
    selected_fnames = sorted(sharp_cands[i][0] for i in selected_indices)

    print(
        f"  Greedy selection chose {len(selected_fnames)} geometrically diverse frames."
    )
    print(f"  Now verifying ZED detection for each...\n")

    # -----------------------------------------------------------------------
    # Phase 4 — ZED verification and copy
    # -----------------------------------------------------------------------
    copied = 0
    skipped = 0
    for fname in selected_fnames:
        zed_path = os.path.join(src_zed, fname)
        if not os.path.exists(zed_path):
            print(f"  [SKIP] {fname}  — ZED frame not found")
            skipped += 1
            continue

        zed_gray = cv2.imread(zed_path, cv2.IMREAD_GRAYSCALE)
        if zed_gray is None:
            print(f"  [SKIP] {fname}  — could not read ZED frame")
            skipped += 1
            continue

        if detect_zed(zed_gray) is None:
            print(f"  [SKIP] {fname}  — ZED ChArUco detection failed")
            skipped += 1
            continue

        shutil.copy2(os.path.join(src_sony, fname), os.path.join(dst_sony, fname))
        shutil.copy2(zed_path, os.path.join(dst_zed, fname))
        print(f"  [OK]   {fname}")
        copied += 1

    print(f"\n{'='*60}")
    print(f"  DONE  —  {copied} frame pairs copied")
    print(f"    Sony → {dst_sony}")
    print(f"    ZED  → {dst_zed}")
    if skipped:
        print(f"  {skipped} frames skipped (ZED detection failed or missing).")
    print(f"{'='*60}")
    print()
    if copied < 30:
        print("[WARN] Fewer than 30 pairs accepted. Consider:")
        print("       - Lowering --stride to scan more candidates")
        print("       - Checking that the ZED recording covers the ChArUco board")
    else:
        print("  Next steps:")
        print(
            f"    1. docker compose run --rm align python3 "
            f"spatial_alignment/inspect_detections.py --dataset {args.dataset}"
        )
        print(f"    2. Remove any flagged frames from picked_for_alignment/")
        print(f"    3. docker compose run --rm align ./run_spatial.sh {args.dataset}")
    print()


if __name__ == "__main__":
    main()

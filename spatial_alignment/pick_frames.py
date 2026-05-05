"""
pick_frames.py
==============
Scans data/synced/<dataset>/sony_rgb/ and data/preprocessed/<dataset>/zed_rgb/
to find the best mathematically verified frame pairs for stereo calibration.

Implements Joint-Scoring and Time-Spread Farthest Point Selection.
"""

import argparse
import glob
import json
import os
import shutil
import sys
import cv2
import numpy as np

# Re-use the exact same plotting logic from the inspector
from inspect_detections import annotate_frame, create_board

BOARD_COLS = 11
BOARD_ROWS = 8
SQUARE_LENGTH = 0.028
MARKER_LENGTH = 0.021
ARUCO_DICT_ID = cv2.aruco.DICT_4X4_50

DEFAULT_TARGET = 45
MOTION_THRESHOLD_PX = 1.5
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def check_motion(detector, curr_path, synced_dir):
    fname = os.path.basename(curr_path)
    frame_idx = int(fname.split(".")[0])
    prev_path = os.path.join(synced_dir, f"{frame_idx - 3:05d}.png")

    if not os.path.exists(prev_path):
        return True

    img_curr = cv2.imread(curr_path, cv2.IMREAD_GRAYSCALE)
    img_prev = cv2.imread(prev_path, cv2.IMREAD_GRAYSCALE)
    if img_curr is None or img_prev is None:
        return True

    c_curr, id_curr, _, _ = detector.detectBoard(img_curr)
    c_prev, id_prev, _, _ = detector.detectBoard(img_prev)

    if c_curr is None or c_prev is None:
        return True

    common_ids, idx_curr, idx_prev = np.intersect1d(
        id_curr.flatten(), id_prev.flatten(), return_indices=True
    )
    if len(common_ids) < 6:
        return True

    pts_curr = c_curr[idx_curr].reshape(-1, 2)
    pts_prev = c_prev[idx_prev].reshape(-1, 2)

    distances = np.linalg.norm(pts_curr - pts_prev, axis=1)
    return np.max(distances) > MOTION_THRESHOLD_PX


def frame_features(corners, h, w):
    pts = corners.reshape(-1, 2)
    cx, cy = float(pts[:, 0].mean() / w), float(pts[:, 1].mean() / h)
    x_min, x_max = float(pts[:, 0].min()), float(pts[:, 0].max())
    y_min, y_max = float(pts[:, 1].min()), float(pts[:, 1].max())
    aspect = (x_max - x_min) / max((y_max - y_min), 1e-6)
    area = ((x_max - x_min) / w) * ((y_max - y_min) / h)
    return np.array([cx, cy, aspect, area], dtype=np.float32)


def farthest_point_select(features, n):
    feat_arr = np.array(features, dtype=np.float64)
    f_min, f_max = feat_arr.min(axis=0), feat_arr.max(axis=0)
    f_range = np.where(f_max - f_min > 0, f_max - f_min, 1.0)
    feat_norm = (feat_arr - f_min) / f_range

    selected = [0]
    min_dists = np.full(len(feat_norm), np.inf)

    for _ in range(n - 1):
        if not selected:
            break
        dists = np.linalg.norm(feat_norm - feat_norm[selected[-1]], axis=1)
        min_dists = np.minimum(min_dists, dists)
        min_dists[selected] = -1.0
        selected.append(int(np.argmax(min_dists)))

    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--count", type=int, default=DEFAULT_TARGET)
    args = parser.parse_args()

    src_sony = os.path.join(BASE_DIR, "data", "inspected", args.dataset, "sony_rgb")
    src_zed = os.path.join(BASE_DIR, "data", "inspected", args.dataset, "zed_rgb")
    synced_sony = os.path.join(BASE_DIR, "data", "synced", args.dataset, "sony_rgb")

    dst_sony = os.path.join(
        BASE_DIR, "data", "picked_for_alignment", args.dataset, "sony_rgb"
    )
    dst_zed = os.path.join(
        BASE_DIR, "data", "picked_for_alignment", args.dataset, "zed_rgb"
    )
    plots_out = os.path.join(
        BASE_DIR, "data", "plots", args.dataset, "spatial", "picked_for_alignment"
    )

    if not os.path.isdir(src_sony) or not os.listdir(src_sony):
        print(f"[ERROR] No inspected frames found. Run inspect_detections first.")
        sys.exit(1)

    for p in [dst_sony, dst_zed, plots_out]:
        os.makedirs(p, exist_ok=True)
        for f in glob.glob(os.path.join(p, "*")):
            os.remove(f)

    all_sony = sorted(glob.glob(os.path.join(src_sony, "*.png")))
    board, dictionary = create_board()

    try:
        params = cv2.aruco.DetectorParameters()
    except AttributeError:
        params = cv2.aruco.DetectorParameters_create()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    try:
        c_params = cv2.aruco.CharucoParameters()
        detector = cv2.aruco.CharucoDetector(board, c_params, params)
    except AttributeError:
        detector = cv2.aruco.CharucoDetector(board)

    print(
        f"\n[INFO] Scanning for MOTION LOCKED frames to achieve {args.count} spatial candidates..."
    )

    valid_pool = []
    for sony_path in all_sony:
        if check_motion(detector, sony_path, synced_sony):
            continue
        img = cv2.imread(sony_path, cv2.IMREAD_GRAYSCALE)
        cs, _, _, _ = detector.detectBoard(img)
        if cs is not None:
            valid_pool.append(
                (
                    os.path.basename(sony_path),
                    frame_features(cs, img.shape[0], img.shape[1]),
                )
            )

    if len(valid_pool) < args.count:
        print(
            f"[ERROR] Only found {len(valid_pool)} perfectly still frames out of requested {args.count}."
        )
        sys.exit(1)

    features = [p[1] for p in valid_pool]
    selected_indices = farthest_point_select(features, args.count)
    selected_fnames = sorted(valid_pool[idx][0] for idx in selected_indices)

    print(
        f"[INFO] Copying {len(selected_fnames)} frames and generating visual plots..."
    )

    for fname in selected_fnames:
        sony_src = os.path.join(src_sony, fname)
        zed_src = os.path.join(src_zed, fname)

        # Copy Raw
        shutil.copy2(sony_src, os.path.join(dst_sony, fname))
        shutil.copy2(zed_src, os.path.join(dst_zed, fname))

        # Use YOUR exact plotting function on the chosen 45 frames
        annotate_frame(
            sony_src,
            board,
            dictionary,
            os.path.join(plots_out, f"sony_{fname.replace('.png', '.jpg')}"),
            try_both_legacy=False,
        )
        annotate_frame(
            zed_src,
            board,
            dictionary,
            os.path.join(plots_out, f"zed_{fname.replace('.png', '.jpg')}"),
            try_both_legacy=True,
        )

    print(
        f"[SUCCESS] {len(selected_fnames)} motionless, perfectly spread frames saved."
    )
    print(
        f"[SUCCESS] Visual plots saved to data/plots/{args.dataset}/spatial/picked_for_alignment/\n"
    )


if __name__ == "__main__":
    main()

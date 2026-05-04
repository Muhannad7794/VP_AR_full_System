"""
pick_frames.py
==============
Scans data/synced/<dataset>/sony_rgb/ and data/preprocessed/<dataset>/zed_rgb/
to find the best mathematically verified frame pairs for stereo calibration.

Implements Joint-Scoring and Time-Spread Farthest Point Selection.
"""

import argparse
import glob
import os
import shutil
import sys
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Board config
# ---------------------------------------------------------------------------
BOARD_COLS = 11
BOARD_ROWS = 8
SQUARE_LENGTH = 0.028
MARKER_LENGTH = 0.021
ARUCO_DICT_ID = cv2.aruco.DICT_4X4_50

DEFAULT_TARGET = 45
DEFAULT_STRIDE = 10
MIN_CORNERS = 10

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _make_detector():
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    board = cv2.aruco.CharucoBoard(
        size=(BOARD_COLS, BOARD_ROWS),
        squareLength=SQUARE_LENGTH,
        markerLength=MARKER_LENGTH,
        dictionary=dictionary,
    )
    board.setLegacyPattern(True)

    try:
        params = cv2.aruco.DetectorParameters()
    except AttributeError:
        params = cv2.aruco.DetectorParameters_create()

    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.adaptiveThreshWinSizeMax = 73

    try:
        charuco_params = cv2.aruco.CharucoParameters()
        return cv2.aruco.CharucoDetector(board, charuco_params, params)
    except AttributeError:
        return cv2.aruco.CharucoDetector(board)


def frame_features(corners: np.ndarray, h: int, w: int, frame_idx: int) -> np.ndarray:
    pts = corners.reshape(-1, 2)
    cx = float(pts[:, 0].mean() / w)
    cy = float(pts[:, 1].mean() / h)

    x_min, x_max = float(pts[:, 0].min()), float(pts[:, 0].max())
    y_min, y_max = float(pts[:, 1].min()), float(pts[:, 1].max())

    bbox_w = (x_max - x_min) / w
    bbox_h = (y_max - y_min) / h
    area = bbox_w * bbox_h
    aspect = bbox_w / max(bbox_h, 1e-6)

    return np.array([cx, cy, area, aspect, frame_idx], dtype=np.float32)


def laplacian_sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def farthest_point_select(
    features: list[np.ndarray], scores: list[float], n: int
) -> list[int]:
    feat_arr = np.array(features, dtype=np.float64)

    f_min = feat_arr.min(axis=0)
    f_max = feat_arr.max(axis=0)
    f_range = np.where(f_max - f_min > 0, f_max - f_min, 1.0)
    feat_norm = (feat_arr - f_min) / f_range

    score_arr = np.array(scores, dtype=np.float64)
    selected: list[int] = [int(np.argmax(score_arr))]

    min_dists = np.full(len(feat_norm), np.inf)

    for _ in range(n - 1):
        if not selected:
            break
        last_f = feat_norm[selected[-1]]
        dists = np.linalg.norm(feat_norm - last_f, axis=1)
        min_dists = np.minimum(min_dists, dists)
        min_dists[selected] = -1.0
        next_idx = int(np.argmax(min_dists))
        if min_dists[next_idx] < 0:
            break
        selected.append(next_idx)

    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--count", type=int, default=DEFAULT_TARGET)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    args = parser.parse_args()

    src_sony = os.path.join(BASE_DIR, "data", "synced", args.dataset, "sony_rgb")
    src_zed = os.path.join(BASE_DIR, "data", "synced", args.dataset, "zed_rgb")
    dst_sony = os.path.join(
        BASE_DIR, "data", "picked_for_alignment", args.dataset, "sony_rgb"
    )
    dst_zed = os.path.join(
        BASE_DIR, "data", "picked_for_alignment", args.dataset, "zed_rgb"
    )

    if not os.path.isdir(src_sony) or not os.path.isdir(src_zed):
        print(f"[ERROR] Source directories not found. Did you run the preprocessor?")
        sys.exit(1)

    if os.path.exists(dst_sony):
        shutil.rmtree(dst_sony)
    if os.path.exists(dst_zed):
        shutil.rmtree(dst_zed)
    os.makedirs(dst_sony, exist_ok=True)
    os.makedirs(dst_zed, exist_ok=True)

    all_sony = sorted(glob.glob(os.path.join(src_sony, "*.png")))
    sampled = all_sony[:: args.stride]

    print(f"\n{'='*60}")
    print(f"  Frame Picker — {args.dataset}")
    print(f"  Scanning every {args.stride}th frame from Synced & Preprocessed pools...")
    print(f"{'='*60}\n")

    detector = _make_detector()
    valid_pool = []

    for i, sony_path in enumerate(sampled):
        fname = os.path.basename(sony_path)
        frame_idx = int(fname.split(".")[0])
        zed_path = os.path.join(src_zed, fname)

        gray_sony = cv2.imread(sony_path, cv2.IMREAD_GRAYSCALE)
        gray_zed = (
            cv2.imread(zed_path, cv2.IMREAD_GRAYSCALE)
            if os.path.exists(zed_path)
            else None
        )

        if gray_sony is None or gray_zed is None:
            continue

        s_corners, _, _, _ = detector.detectBoard(gray_sony)
        z_corners, _, _, _ = detector.detectBoard(gray_zed)

        s_count = len(s_corners) if s_corners is not None else 0
        z_count = len(z_corners) if z_corners is not None else 0

        if s_count >= MIN_CORNERS and z_count >= MIN_CORNERS:
            sharp_score = min(
                laplacian_sharpness(gray_sony), laplacian_sharpness(gray_zed)
            )
            corner_score = s_count + z_count
            joint_score = sharp_score * corner_score

            h, w = gray_sony.shape
            feats = frame_features(s_corners, h, w, frame_idx)
            valid_pool.append((fname, feats, joint_score))

        if (i + 1) % 50 == 0 or i == len(sampled) - 1:
            print(
                f"  Scanned {i + 1:03d}/{len(sampled)} — Found {len(valid_pool)} mutually valid pairs."
            )

    if not valid_pool:
        print(
            "\n[ERROR] No synchronous frames passed detection. Ensure images are not blurred."
        )
        sys.exit(1)

    print(f"\n  Scan Complete. Pooling {len(valid_pool)} highly confident frames.")

    n_select = min(args.count, len(valid_pool))
    features = [p[1] for p in valid_pool]
    scores = [p[2] for p in valid_pool]

    selected_indices = farthest_point_select(features, scores, n_select)
    selected_fnames = sorted(valid_pool[idx][0] for idx in selected_indices)

    print(
        f"  Greedy algorithm locked {len(selected_fnames)} frames across the timeline & geometry.\n"
    )

    for fname in selected_fnames:
        # Copying the enhanced ZED files directly into the calibration folder!
        shutil.copy2(os.path.join(src_sony, fname), os.path.join(dst_sony, fname))
        shutil.copy2(os.path.join(src_zed, fname), os.path.join(dst_zed, fname))

    print(f"{'='*60}")
    print(f"  DONE  —  Files saved to picked_for_alignment/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

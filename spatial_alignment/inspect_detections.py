"""
inspect_detections.py
=====================
Diagnostic tool for the spatial alignment pipeline.

Runs ChArUco corner detection on every frame in picked_for_alignment/
and saves annotated images showing which corners were detected.

Run this BEFORE stereo_calibrate.py to verify the board is detectable
in both cameras and to identify any blurry or problematic frames
that should be removed before calibration.

Output:
    data/plots/<dataset>/spatial/detection_check/
        sony_<frame>.jpg   – Sony frames with detected corners drawn
        zed_<frame>.jpg    – ZED frames with detected corners drawn
        detection_summary.json – per-frame detection counts

Usage:
    docker compose run --rm align python3 spatial_alignment/inspect_detections.py --dataset <name>
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

BOARD_COLS = 11  # squares horizontally (landscape orientation)
BOARD_ROWS = 8  # squares vertically   (landscape orientation)
SQUARE_LENGTH = 0.028
MARKER_LENGTH = 0.021
ARUCO_DICT_ID = cv2.aruco.DICT_4X4_50
LEGACY_PATTERN = True  # required for calib.io generated boards

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_paths(dataset: str) -> dict:
    return {
        "sony_rgb": os.path.join(
            BASE_DIR, "data", "picked_for_alignment", dataset, "sony_rgb"
        ),
        "zed_rgb": os.path.join(
            BASE_DIR, "data", "picked_for_alignment", dataset, "zed_rgb"
        ),
        "plots_out": os.path.join(
            BASE_DIR, "data", "plots", dataset, "spatial", "detection_check"
        ),
        "json_out": os.path.join(BASE_DIR, "data", "json_output", dataset),
    }


def create_board():
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    board = cv2.aruco.CharucoBoard(
        size=(BOARD_COLS, BOARD_ROWS),
        squareLength=SQUARE_LENGTH,
        markerLength=MARKER_LENGTH,
        dictionary=dictionary,
    )
    board.setLegacyPattern(LEGACY_PATTERN)
    return board, dictionary


def annotate_frame(
    image_path: str, board, dictionary, output_path: str, try_both_legacy: bool = False
) -> dict:
    """
    Detect and draw ChArUco corners on the image.
    If try_both_legacy=True (used for ZED frames), both legacy=True and
    legacy=False are tried and the result with more corners is used.
    Green circles = detected corners. Red text = rejected frame.
    """
    img = cv2.imread(image_path)
    if img is None:
        return {
            "file": os.path.basename(image_path),
            "status": "unreadable",
            "n_corners": 0,
        }

    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if try_both_legacy:
        # Try both legacy settings and use the one giving more corners
        best_corners = None
        best_ids = None
        best_markers_c = None
        best_markers_i = None
        best_count = 0
        for legacy in [True, False]:
            b = cv2.aruco.CharucoBoard(
                size=(BOARD_COLS, BOARD_ROWS),
                squareLength=SQUARE_LENGTH,
                markerLength=MARKER_LENGTH,
                dictionary=dictionary,
            )
            b.setLegacyPattern(legacy)
            det = cv2.aruco.CharucoDetector(b)
            cc, ci, mc, mi = det.detectBoard(gray)
            n = len(cc) if cc is not None else 0
            if n > best_count:
                best_count = n
                best_corners = cc
                best_ids = ci
                best_markers_c = mc
                best_markers_i = mi
        charuco_corners = best_corners
        charuco_ids = best_ids
        marker_corners = best_markers_c
        marker_ids = best_markers_i
    else:
        detector = cv2.aruco.CharucoDetector(board)
        charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(
            gray
        )

    if marker_ids is None or len(marker_ids) < 4:
        cv2.putText(
            img,
            "REJECTED – no ArUco markers",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 220),
            3,
        )
        cv2.imwrite(output_path, img)
        return {
            "file": os.path.basename(image_path),
            "status": "rejected_no_markers",
            "n_markers": 0,
            "n_corners": 0,
        }

    cv2.aruco.drawDetectedMarkers(img, marker_corners, marker_ids)

    n_corners = 0
    status = "rejected_few_corners"

    if charuco_corners is not None and len(charuco_corners) >= 6:
        n_corners = len(charuco_corners)
        status = "ok"
        for pt in charuco_corners.reshape(-1, 2):
            cv2.circle(img, (int(pt[0]), int(pt[1])), 6, (0, 200, 0), -1)

    color = (0, 180, 0) if status == "ok" else (0, 0, 220)
    label = f"Corners: {n_corners}  Status: {status}"
    cv2.rectangle(img, (0, 0), (700, 60), (0, 0, 0), -1)
    cv2.putText(img, label, (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)

    h, w = img.shape[:2]
    if w > 1280:
        scale = 1280 / w
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    cv2.imwrite(output_path, img)

    return {
        "file": os.path.basename(image_path),
        "status": status,
        "n_markers": len(marker_ids) if marker_ids is not None else 0,
        "n_corners": n_corners,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Inspect ChArUco detections in picked_for_alignment frames."
    )
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    paths = build_paths(args.dataset)
    os.makedirs(paths["plots_out"], exist_ok=True)
    os.makedirs(paths["json_out"], exist_ok=True)

    board, dictionary = create_board()

    print(f"\n{'='*60}")
    print(f"  Detection Inspection – dataset: {args.dataset}")
    print(f"{'='*60}\n")

    summary = {"sony": [], "zed": []}

    for camera_key, label in [("sony_rgb", "Sony"), ("zed_rgb", "ZED")]:
        src_dir = paths[camera_key]
        is_zed = camera_key == "zed_rgb"

        if not os.path.isdir(src_dir):
            print(f"[WARN] Directory not found: {src_dir}")
            continue

        files = sorted(
            f
            for f in os.listdir(src_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )

        # Sony: always legacy=True
        # ZED:  annotate_frame will try both legacy settings
        board, dictionary = create_board()

        print(f"[INFO] {label}: processing {len(files)} frames …")
        for fname in files:
            in_path = os.path.join(src_dir, fname)
            out_name = f"{camera_key.split('_')[0]}_{fname.replace('.png', '.jpg')}"
            out_path = os.path.join(paths["plots_out"], out_name)
            result = annotate_frame(
                in_path, board, dictionary, out_path, try_both_legacy=is_zed
            )
            summary[camera_key.split("_")[0]].append(result)

            status_sym = "✓" if result["status"] == "ok" else "✗"
            print(f"  {status_sym} {fname:30s}  corners={result['n_corners']:3d}")

    # Print summary statistics
    for cam in ["sony", "zed"]:
        items = summary[cam]
        n_ok = sum(1 for r in items if r["status"] == "ok")
        n_total = len(items)
        print(f"\n  {cam.upper()} detection rate: {n_ok}/{n_total} frames")
        if n_ok < n_total:
            bad = [r["file"] for r in items if r["status"] != "ok"]
            print(f"  Problematic frames to review or remove:")
            for f in bad:
                print(f"    – {f}")

    # Save summary JSON
    report_path = os.path.join(paths["json_out"], "detection_summary.json")
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[INFO] Detection summary saved → {report_path}")
    print(f"[INFO] Annotated images saved  → {paths['plots_out']}/\n")

    print(f"{'='*60}")
    print("  INSPECTION COMPLETE")
    print(f"{'='*60}\n")
    print("  Next step: remove any problematic frames from " "picked_for_alignment/")
    print("  then run: docker compose run --rm align --dataset " f"{args.dataset}")


if __name__ == "__main__":
    main()

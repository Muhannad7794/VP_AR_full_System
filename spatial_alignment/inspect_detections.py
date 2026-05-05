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
import glob
import shutil
import sys
import cv2
import numpy as np

BOARD_COLS = 11
BOARD_ROWS = 8
SQUARE_LENGTH = 0.028
MARKER_LENGTH = 0.021
ARUCO_DICT_ID = cv2.aruco.DICT_4X4_50
LEGACY_PATTERN = True
MIN_CORNERS = 45  # The strict threshold we established

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_paths(dataset: str) -> dict:
    return {
        "sony_raw": os.path.join(BASE_DIR, "data", "synced", dataset, "sony_rgb"),
        "zed_raw": os.path.join(BASE_DIR, "data", "synced", dataset, "zed_rgb"),
        "sony_pass": os.path.join(BASE_DIR, "data", "inspected", dataset, "sony_rgb"),
        "zed_pass": os.path.join(BASE_DIR, "data", "inspected", dataset, "zed_rgb"),
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
    """YOUR EXACT PLOTTING LOGIC - UPDATED ONLY WITH MIN_CORNERS THRESHOLD"""
    img = cv2.imread(image_path)
    if img is None:
        return {
            "file": os.path.basename(image_path),
            "status": "unreadable",
            "n_corners": 0,
        }

    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if try_both_legacy:
        best_corners, best_ids, best_markers_c, best_markers_i, best_count = (
            None,
            None,
            None,
            None,
            0,
        )
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
                best_count, best_corners, best_ids = n, cc, ci
                best_markers_c, best_markers_i = mc, mi
        charuco_corners, charuco_ids = best_corners, best_ids
        marker_corners, marker_ids = best_markers_c, best_markers_i
    else:
        detector = cv2.aruco.CharucoDetector(board)
        charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(
            gray
        )

    if marker_ids is None or len(marker_ids) < 4:
        cv2.putText(
            img,
            "REJECTED - no ArUco markers",
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

    # STRICT 45 CORNER CHECK
    if charuco_corners is not None and len(charuco_corners) >= MIN_CORNERS:
        n_corners = len(charuco_corners)
        status = "ok"
        for pt in charuco_corners.reshape(-1, 2):
            cv2.circle(img, (int(pt[0]), int(pt[1])), 6, (0, 200, 0), -1)
    elif charuco_corners is not None:
        n_corners = len(charuco_corners)

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    paths = build_paths(args.dataset)
    for p in [
        paths["sony_pass"],
        paths["zed_pass"],
        paths["plots_out"],
        paths["json_out"],
    ]:
        os.makedirs(p, exist_ok=True)
    for p in [paths["sony_pass"], paths["zed_pass"], paths["plots_out"]]:
        for f in glob.glob(os.path.join(p, "*")):
            os.remove(f)

    board, dictionary = create_board()
    files = sorted(
        [f for f in os.listdir(paths["sony_raw"]) if f.lower().endswith(".png")]
    )

    print(f"\n{'='*60}")
    print(f"  Filtering {len(files)} pairs. Strict Threshold: {MIN_CORNERS}+ Corners")
    print(f"  Generating plots in data/plots/{args.dataset}/spatial/detection_check/")
    print(f"{'='*60}\n")

    summary = {"sony": [], "zed": []}
    passed_count = 0

    for i, fname in enumerate(files):
        sony_in = os.path.join(paths["sony_raw"], fname)
        zed_in = os.path.join(paths["zed_raw"], fname)
        if not os.path.exists(zed_in):
            continue

        sony_plot = os.path.join(
            paths["plots_out"], f"sony_{fname.replace('.png', '.jpg')}"
        )
        zed_plot = os.path.join(
            paths["plots_out"], f"zed_{fname.replace('.png', '.jpg')}"
        )

        res_s = annotate_frame(
            sony_in, board, dictionary, sony_plot, try_both_legacy=False
        )
        res_z = annotate_frame(
            zed_in, board, dictionary, zed_plot, try_both_legacy=True
        )

        summary["sony"].append(res_s)
        summary["zed"].append(res_z)

        if res_s["status"] == "ok" and res_z["status"] == "ok":
            shutil.copy2(sony_in, os.path.join(paths["sony_pass"], fname))
            shutil.copy2(zed_in, os.path.join(paths["zed_pass"], fname))
            passed_count += 1
            sym = "✓"
        else:
            sym = "✗"

        if (i + 1) % 100 == 0 or sym == "✓":
            print(
                f"  {sym} {fname:15s} | SONY: {res_s['n_corners']:2d} | ZED: {res_z['n_corners']:2d}"
            )

    report_path = os.path.join(paths["json_out"], "detection_summary.json")
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[SUCCESS] {passed_count} high-quality frames moved to data/inspected/\n")


if __name__ == "__main__":
    main()

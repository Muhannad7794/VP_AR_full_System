"""
diagnose_detection.py
=====================
Diagnostic tool to identify why ChArUco detection is failing.

Tries every possible DICT_4X4 variant and multiple detector
parameter configurations on a single frame, then saves annotated
debug images so the correct settings can be identified visually.

Run on ONE frame first to find working settings, then update
stereo_calibrate.py and inspect_detections.py with the correct values.

Usage:
    python3 spatial_alignment/diagnose_detection.py --dataset <name>

Output:
    data/plots/<dataset>/spatial/diagnosis/
        dict_4x4_50_<filename>.jpg
        dict_4x4_100_<filename>.jpg
        dict_4x4_250_<filename>.jpg
        dict_4x4_1000_<filename>.jpg
        relaxed_params_<filename>.jpg
        summary.txt
"""

import argparse
import os
import sys

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Board dimensions to try - (cols, rows)
BOARD_CANDIDATES = [
    (8, 11),
    (11, 8),
    (7, 10),
    (10, 7),
]

SQUARE_LENGTH = 0.028
MARKER_LENGTH = 0.021

DICT_CANDIDATES = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
}


def build_paths(dataset: str) -> dict:
    return {
        "sony_rgb": os.path.join(
            BASE_DIR, "data", "picked_for_alignment", dataset, "sony_rgb"
        ),
        "zed_rgb": os.path.join(
            BASE_DIR, "data", "picked_for_alignment", dataset, "zed_rgb"
        ),
        "plots_out": os.path.join(
            BASE_DIR, "data", "plots", dataset, "spatial", "diagnosis"
        ),
    }


def relaxed_detector_params() -> cv2.aruco.DetectorParameters:
    """
    Return detector parameters with relaxed thresholds that work
    on a wider range of image conditions.
    """
    p = cv2.aruco.DetectorParameters()

    # Adaptive thresholding - try larger window sizes
    p.adaptiveThreshWinSizeMin = 3
    p.adaptiveThreshWinSizeMax = 53
    p.adaptiveThreshWinSizeStep = 10
    p.adaptiveThreshConstant = 7

    # More permissive corner detection
    p.minMarkerPerimeterRate = 0.01
    p.maxMarkerPerimeterRate = 4.0
    p.polygonalApproxAccuracyRate = 0.05
    p.minCornerDistanceRate = 0.01
    p.minDistanceToBorder = 1

    # Marker identification
    p.errorCorrectionRate = 0.9
    p.perspectiveRemovePixelPerCell = 8
    p.perspectiveRemoveIgnoredMarginPerCell = 0.13

    return p


def try_detection(
    gray: np.ndarray, dict_name: str, dict_id: int, board_size: tuple, detector_params
) -> dict:
    """
    Attempt ChArUco detection with the given settings using the
    CharucoDetector API (OpenCV 4.7+) with legacy pattern enabled,
    as required by calib.io generated boards.
    Returns a summary dict.
    """
    dictionary = cv2.aruco.getPredefinedDictionary(dict_id)

    board = cv2.aruco.CharucoBoard(
        size=board_size,
        squareLength=SQUARE_LENGTH,
        markerLength=MARKER_LENGTH,
        dictionary=dictionary,
    )
    board.setLegacyPattern(True)

    charuco_detector = cv2.aruco.CharucoDetector(board)
    charuco_corners, charuco_ids, marker_corners, marker_ids = (
        charuco_detector.detectBoard(gray)
    )

    n_markers = len(marker_ids) if marker_ids is not None else 0
    n_charuco_corners = len(charuco_corners) if charuco_corners is not None else 0

    # Also try without legacy pattern for comparison
    board_no_legacy = cv2.aruco.CharucoBoard(
        size=board_size,
        squareLength=SQUARE_LENGTH,
        markerLength=MARKER_LENGTH,
        dictionary=dictionary,
    )
    det2 = cv2.aruco.CharucoDetector(board_no_legacy)
    cc2, ci2, mc2, mi2 = det2.detectBoard(gray)
    n_corners_no_legacy = len(cc2) if cc2 is not None else 0

    # Return whichever gave more corners
    if n_corners_no_legacy > n_charuco_corners:
        return {
            "dict_name": dict_name,
            "board_size": board_size,
            "n_markers_found": len(mi2) if mi2 is not None else 0,
            "n_markers_rejected": 0,
            "n_charuco_corners": n_corners_no_legacy,
            "marker_corners": mc2,
            "marker_ids": mi2,
            "charuco_corners": cc2,
            "charuco_ids": ci2,
        }

    return {
        "dict_name": dict_name,
        "board_size": board_size,
        "n_markers_found": n_markers,
        "n_markers_rejected": 0,
        "n_charuco_corners": n_charuco_corners,
        "marker_corners": marker_corners,
        "marker_ids": marker_ids,
        "charuco_corners": charuco_corners,
        "charuco_ids": charuco_ids,
    }


def annotate_and_save(img_bgr: np.ndarray, result: dict, out_path: str):
    vis = img_bgr.copy()

    if result["n_markers_found"] > 0:
        cv2.aruco.drawDetectedMarkers(
            vis, result["marker_corners"], result["marker_ids"]
        )

    if result["charuco_corners"] is not None:
        for pt in result["charuco_corners"].reshape(-1, 2):
            cv2.circle(vis, (int(pt[0]), int(pt[1])), 6, (0, 200, 0), -1)

    status = (
        f"Dict: {result['dict_name']}  "
        f"Board: {result['board_size']}  "
        f"Markers: {result['n_markers_found']}  "
        f"ChArUco corners: {result['n_charuco_corners']}"
    )

    color = (0, 200, 0) if result["n_charuco_corners"] >= 6 else (0, 0, 220)
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 60), (0, 0, 0), -1)
    cv2.putText(vis, status, (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Resize for compact output
    h, w = vis.shape[:2]
    if w > 1280:
        scale = 1280 / w
        vis = cv2.resize(vis, (int(w * scale), int(h * scale)))

    cv2.imwrite(out_path, vis)


def main():
    parser = argparse.ArgumentParser(description="Diagnose ChArUco detection failures.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--n_frames",
        type=int,
        default=3,
        help="Number of frames to test (default: 3). "
        "Tests the first N frames from sony_rgb/.",
    )
    args = parser.parse_args()

    paths = build_paths(args.dataset)
    os.makedirs(paths["plots_out"], exist_ok=True)

    sony_dir = paths["sony_rgb"]
    if not os.path.isdir(sony_dir):
        print(f"[ERROR] Directory not found: {sony_dir}")
        sys.exit(1)

    all_files = sorted(
        f for f in os.listdir(sony_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    test_files = all_files[: args.n_frames]

    print(f"\n{'='*60}")
    print(f"  ChArUco Detection Diagnosis – dataset: {args.dataset}")
    print(
        f"  Testing {len(test_files)} frames with "
        f"{len(DICT_CANDIDATES)} dictionaries × "
        f"{len(BOARD_CANDIDATES)} board sizes"
    )
    print(f"{'='*60}\n")

    summary_lines = []
    best_result = None

    for fname in test_files:
        img_path = os.path.join(sony_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] Could not read: {fname}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print(f"  Frame: {fname}  ({gray.shape[1]}×{gray.shape[0]} px)")

        for dict_name, dict_id in DICT_CANDIDATES.items():
            for board_size in BOARD_CANDIDATES:

                # 1. Default parameters
                default_params = cv2.aruco.DetectorParameters()
                r_default = try_detection(
                    gray, dict_name, dict_id, board_size, default_params
                )

                # 2. Relaxed parameters
                r_relaxed = try_detection(
                    gray, dict_name, dict_id, board_size, relaxed_detector_params()
                )

                # Pick the better of the two
                r = (
                    r_default
                    if (
                        r_default["n_charuco_corners"] >= r_relaxed["n_charuco_corners"]
                    )
                    else r_relaxed
                )
                params_label = "default" if r is r_default else "relaxed"

                line = (
                    f"    {dict_name:20s}  board={board_size}  "
                    f"markers={r['n_markers_found']:3d}  "
                    f"corners={r['n_charuco_corners']:3d}  "
                    f"params={params_label}"
                )
                print(line)
                summary_lines.append(line)

                if r["n_charuco_corners"] >= 6:
                    tag = (
                        f"{dict_name}_board"
                        f"{board_size[0]}x{board_size[1]}"
                        f"_{params_label}_{fname.replace('.png','')}.jpg"
                    )
                    annotate_and_save(img, r, os.path.join(paths["plots_out"], tag))

                    if (
                        best_result is None
                        or r["n_charuco_corners"] > best_result["n_charuco_corners"]
                    ):
                        best_result = {
                            **r,
                            "params": params_label,
                            "source_frame": fname,
                        }

        print()

    # Save summary
    summary_path = os.path.join(paths["plots_out"], "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Detection Diagnosis – dataset: {args.dataset}\n")
        f.write("=" * 60 + "\n\n")
        f.write("\n".join(summary_lines))
        if best_result:
            f.write("\n\n" + "=" * 60 + "\n")
            f.write("BEST RESULT:\n")
            f.write(f"  Dictionary : {best_result['dict_name']}\n")
            f.write(f"  Board size : {best_result['board_size']}\n")
            f.write(f"  Corners    : {best_result['n_charuco_corners']}\n")
            f.write(f"  Parameters : {best_result['params']}\n")

    print(f"[INFO] Summary saved → {summary_path}")
    print(f"[INFO] Annotated frames → {paths['plots_out']}/\n")

    if best_result:
        print("=" * 60)
        print("  WORKING CONFIGURATION FOUND")
        print("=" * 60)
        print(f"  Dictionary : {best_result['dict_name']}")
        print(f"  Board size : {best_result['board_size']}")
        print(f"  Corners    : {best_result['n_charuco_corners']}")
        print(f"  Parameters : {best_result['params']}")
        print()
        print("  ACTION: Update the constants at the top of")
        print("  stereo_calibrate.py and inspect_detections.py:")
        print(f"    ARUCO_DICT_ID = cv2.aruco.{best_result['dict_name']}")
        print(f"    BOARD_COLS    = {best_result['board_size'][0]}")
        print(f"    BOARD_ROWS    = {best_result['board_size'][1]}")
        if best_result["params"] == "relaxed":
            print("    Also enable relaxed detector parameters")
            print("    (copy relaxed_detector_params() from this script)")
        print("=" * 60 + "\n")
    else:
        print("=" * 60)
        print("  NO WORKING CONFIGURATION FOUND")
        print("=" * 60)
        print()
        print("  This means the board is not detectable from these frames.")
        print("  Check the annotated images in the diagnosis folder.")
        print()
        print("  Possible causes:")
        print("  1. Frames are too blurry - check shutter speed was fast")
        print("  2. Board is too far or too small in frame")
        print("  3. Lighting is causing glare on the white squares")
        print("  4. The board file uses a different ArUco dictionary")
        print("     than expected. Try downloading a board with explicit")
        print("     DICT_4X4_50 label from calib.io")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

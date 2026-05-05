"""
stereo_calibrate.py
===================
Phase 2: Spatial Alignment.
"""

import argparse
import json
import os
import sys
import time
import cv2
import numpy as np

BOARD_COLS = 11
BOARD_ROWS = 8
SQUARE_LENGTH = 0.028
MARKER_LENGTH = 0.021
ARUCO_DICT_ID = cv2.aruco.DICT_4X4_50

SONY_LEGACY_PATTERN = True
ZED_LEGACY_PATTERN = True

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_paths(dataset: str) -> dict:
    return {
        "sony_rgb": os.path.join(
            BASE_DIR, "data", "picked_for_alignment", dataset, "sony_rgb"
        ),
        "zed_rgb": os.path.join(
            BASE_DIR, "data", "picked_for_alignment", dataset, "zed_rgb"
        ),
        "json_out": os.path.join(BASE_DIR, "data", "json_output", dataset),
    }


def create_board(legacy: bool):
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    board = cv2.aruco.CharucoBoard(
        size=(BOARD_COLS, BOARD_ROWS),
        squareLength=SQUARE_LENGTH,
        markerLength=MARKER_LENGTH,
        dictionary=dictionary,
    )
    board.setLegacyPattern(legacy)
    return board, dictionary


def detect_charuco(image_path: str, board) -> tuple:
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None, None, None

    image_size = (gray.shape[1], gray.shape[0])

    try:
        params = cv2.aruco.DetectorParameters()
    except AttributeError:
        params = cv2.aruco.DetectorParameters_create()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    try:
        charuco_params = cv2.aruco.CharucoParameters()
        detector = cv2.aruco.CharucoDetector(board, charuco_params, params)
    except AttributeError:
        detector = cv2.aruco.CharucoDetector(board)

    charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)

    if charuco_corners is None or len(charuco_corners) < 6:
        return None, None, image_size

    return charuco_corners, charuco_ids, image_size


def collect_valid_pairs(sony_dir: str, zed_dir: str) -> dict:
    sony_board, _ = create_board(legacy=SONY_LEGACY_PATTERN)
    zed_board, _ = create_board(legacy=ZED_LEGACY_PATTERN)

    sony_files = sorted(
        [
            f
            for f in os.listdir(sony_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    )
    zed_files = sorted(
        [
            f
            for f in os.listdir(zed_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    )

    pairs = list(zip(sony_files, zed_files))
    print(f"[INFO] Processing {len(pairs)} perfectly synced frame pairs …")

    obj_points_list = []
    sony_corners_list = []
    zed_corners_list = []
    sony_image_size = None
    zed_image_size = None
    rejected = 0

    for i, (sf, zf) in enumerate(pairs):
        s_corners, s_ids, s_size = detect_charuco(
            os.path.join(sony_dir, sf), sony_board
        )
        z_corners, z_ids, z_size = detect_charuco(os.path.join(zed_dir, zf), zed_board)

        if s_size is not None and sony_image_size is None:
            sony_image_size = s_size
        if z_size is not None and zed_image_size is None:
            zed_image_size = z_size

        if s_corners is None or z_corners is None:
            rejected += 1
            continue

        common_ids, idx_s, idx_z = np.intersect1d(
            s_ids.flatten(), z_ids.flatten(), return_indices=True
        )

        if len(common_ids) < 45:  # STRICT MATCHING THRESHOLD
            rejected += 1
            print(
                f"  [{i+1:03d}] REJECTED – Only {len(common_ids)} matched corners (Needs 45)"
            )
            continue

        all_board_corners = sony_board.getChessboardCorners()
        obj_pts = all_board_corners[common_ids].astype(np.float32)

        obj_points_list.append(obj_pts)
        sony_corners_list.append(s_corners[idx_s].astype(np.float32))
        zed_corners_list.append(z_corners[idx_z].astype(np.float32))
        print(f"  [{i+1:03d}] OK – {len(common_ids)} perfectly matched corners")

    return {
        "obj_points": obj_points_list,
        "sony_corners": sony_corners_list,
        "zed_corners": zed_corners_list,
        "sony_image_size": sony_image_size,
        "zed_image_size": zed_image_size,
    }


def calibrate_single_camera(
    obj_points: list, img_points: list, image_size: tuple, name: str
) -> tuple:
    print(f"\n[INFO] Calibrating {name} camera …")
    rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None, flags=0
    )
    print(f"  RMS reprojection error ({name}): {rms:.4f} px")
    return K, D, rms, rvecs, tvecs


def run_stereo_calibration(
    obj_points,
    sony_corners,
    zed_corners,
    K_sony,
    D_sony,
    K_zed,
    D_zed,
    sony_size,
    zed_size,
) -> dict:
    print("\n[INFO] Running stereo calibration …")

    # RESTORED: Lock the perfect individual intrinsics so rotation doesn't warp
    flags = cv2.CALIB_FIX_INTRINSIC

    rms, K_sony_out, D_sony_out, K_zed_out, D_zed_out, R, T, E, F = cv2.stereoCalibrate(
        obj_points,
        zed_corners,  # Cam 1 = ZED
        sony_corners,  # Cam 2 = Sony
        K_zed,
        D_zed,
        K_sony,
        D_sony,
        zed_size,
        flags=flags,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-7),
    )

    print(f"  Stereo RMS reprojection error: {rms:.4f} px")
    print(
        f"  Translation ZED→Sony (metres): X={T[0,0]:.4f}  Y={T[1,0]:.4f}  Z={T[2,0]:.4f}"
    )

    rod, _ = cv2.Rodrigues(R)
    deg = np.degrees(rod).flatten()
    print(
        f"  Rotation ZED→Sony (degrees):  Rx={deg[0]:.3f}  Ry={deg[1]:.3f}  Rz={deg[2]:.3f}"
    )

    return {"R": R, "T": T, "E": E, "F": F, "stereo_rms": float(rms)}


def save_calibration(
    output_dir,
    dataset,
    K_sony,
    D_sony,
    rms_sony,
    K_zed,
    D_zed,
    rms_zed,
    stereo,
    board_cfg,
    n_pairs,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "spatial_calibration.json")
    rod, _ = cv2.Rodrigues(stereo["R"])

    # RESTORED: Full JSON payload so validate_calibration.py and apply_calibration.py don't crash
    payload = {
        "meta": {
            "dataset": dataset,
            "valid_pairs_used": n_pairs,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "sony_camera": {
            "camera_matrix": K_sony.tolist(),
            "distortion_coeffs": D_sony.tolist(),
        },
        "zed_camera": {
            "camera_matrix": K_zed.tolist(),
            "distortion_coeffs": D_zed.tolist(),
        },
        "extrinsics": {
            "rotation_matrix_3x3": stereo["R"].tolist(),
            "rotation_rodrigues": rod.flatten().tolist(),
            "rotation_degrees": np.degrees(rod).flatten().tolist(),
            "translation_metres": stereo["T"].flatten().tolist(),
            "translation_cm": (stereo["T"].flatten() * 100).tolist(),
            "essential_matrix": stereo["E"].tolist(),
            "fundamental_matrix": stereo["F"].tolist(),
            "stereo_rms_px": stereo["stereo_rms"],
        },
        "ue5_offset": {
            "X_cm": float((stereo["T"].flatten() * 100)[0]),
            "Y_cm": float((stereo["T"].flatten() * 100)[1]),
            "Z_cm": float((stereo["T"].flatten() * 100)[2]),
        },
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    paths = build_paths(args.dataset)

    print(f"\n{'='*60}\n  Spatial Calibration – dataset: {args.dataset}\n{'='*60}\n")
    data = collect_valid_pairs(paths["sony_rgb"], paths["zed_rgb"])

    # Abort logic if too few valid pairs found
    if len(data["obj_points"]) < 5:
        print(
            f"\n[ERROR] Not enough valid pairs for calibration. Found {len(data['obj_points'])} pairs with at least 45 matched corners."
        )
        sys.exit(1)

    K_sony, D_sony, rms_sony, _, _ = calibrate_single_camera(
        data["obj_points"], data["sony_corners"], data["sony_image_size"], "Sony"
    )
    K_zed, D_zed, rms_zed, _, _ = calibrate_single_camera(
        data["obj_points"], data["zed_corners"], data["zed_image_size"], "ZED"
    )

    stereo = run_stereo_calibration(
        data["obj_points"],
        data["sony_corners"],
        data["zed_corners"],
        K_sony,
        D_sony,
        K_zed,
        D_zed,
        data["sony_image_size"],
        data["zed_image_size"],
    )

    save_calibration(
        paths["json_out"],
        args.dataset,
        K_sony,
        D_sony,
        rms_sony,
        K_zed,
        D_zed,
        rms_zed,
        stereo,
        {},
        len(data["obj_points"]),
    )

    print(f"\n{'='*60}\n  CALIBRATION COMPLETE")
    print(f"  Stereo RMS : {stereo['stereo_rms']:.4f} px\n{'='*60}\n")


if __name__ == "__main__":
    main()

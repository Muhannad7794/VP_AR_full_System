"""
stereo_calibrate.py
===================
Phase 2 of the VP_AR_full_System pipeline: Spatial Alignment.

Reads manually curated frame pairs from:
    data/picked_for_alignment/<dataset>/sony_rgb/
    data/picked_for_alignment/<dataset>/zed_rgb/

Runs ChArUco-based stereo camera calibration and writes the resulting
intrinsic and extrinsic matrices to:
    data/json_output/<dataset>/spatial_calibration.json

Usage (via Docker):
    docker compose run --rm align --dataset <dataset_name>

Usage (direct):
    python3 spatial_alignment/stereo_calibrate.py --dataset <dataset_name>

Board specification (calib_io_charuco_200x150_8x11_15_11_DICT_4X4, printed A3):
    Columns (squares):  8
    Rows    (squares): 11
    Square side length: 0.028 m  (measured on printed A3 sheet)
    Marker side length: 0.021 m  (measured on printed A3 sheet)
    ArUco dictionary:   DICT_4X4_50
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Board configuration – update these values if a different board is used
# ---------------------------------------------------------------------------
BOARD_COLS = 11  # number of squares horizontally (landscape orientation)
BOARD_ROWS = 8  # number of squares vertically  (landscape orientation)
SQUARE_LENGTH = 0.028  # metres, measured on the actual A3 print
MARKER_LENGTH = 0.021  # metres, measured on the actual A3 print
ARUCO_DICT_ID = cv2.aruco.DICT_4X4_50
SONY_LEGACY_PATTERN = True  # Sony consistently needs legacy=True
ZED_LEGACY_PATTERN = "auto"  # ZED frames are tested with both True and False
# per frame; detect_charuco_zed picks the best

# Minimum detections required before calibration is attempted
MIN_VALID_PAIRS = 4


# ---------------------------------------------------------------------------
# Helper: build paths relative to the Docker working directory
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Board factory
# ---------------------------------------------------------------------------
def create_board(legacy: bool):
    """
    Create a CharucoBoard with the specified legacy pattern setting.
    Sony frames consistently use legacy=True.
    ZED frames are tested with both settings via detect_charuco_zed.
    """
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    board = cv2.aruco.CharucoBoard(
        size=(BOARD_COLS, BOARD_ROWS),
        squareLength=SQUARE_LENGTH,
        markerLength=MARKER_LENGTH,
        dictionary=dictionary,
    )
    board.setLegacyPattern(legacy)
    return board, dictionary


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
def detect_charuco(image_path: str, board) -> tuple:
    """
    Detect ChArUco corners using the CharucoDetector API.
    The board must have the correct legacyPattern already set.
    Use detect_charuco_zed for ZED frames to auto-select legacy setting.
    """
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        print(f"  [WARN] Could not read image: {image_path}")
        return None, None, None

    image_size = (gray.shape[1], gray.shape[0])
    detector = cv2.aruco.CharucoDetector(board)
    charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)

    if charuco_corners is None or len(charuco_corners) < 6:
        return None, None, image_size

    return charuco_corners, charuco_ids, image_size


def detect_charuco_zed(image_path: str) -> tuple:
    """
    Detect ChArUco corners in a ZED frame by trying both legacy pattern
    settings and returning the result with the most corners.

    ZED frames vary in which legacy setting works best depending on the
    recording conditions and ZED SDK image processing pipeline. Testing
    both settings guarantees the maximum number of usable frames.
    """
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None, None, None

    image_size = (gray.shape[1], gray.shape[0])
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    best_corners = None
    best_ids = None
    best_count = 0

    for legacy in [True, False]:
        board = cv2.aruco.CharucoBoard(
            size=(BOARD_COLS, BOARD_ROWS),
            squareLength=SQUARE_LENGTH,
            markerLength=MARKER_LENGTH,
            dictionary=dictionary,
        )
        board.setLegacyPattern(legacy)
        detector = cv2.aruco.CharucoDetector(board)
        corners, ids, _, _ = detector.detectBoard(gray)

        n = len(corners) if corners is not None else 0
        if n > best_count:
            best_count = n
            best_corners = corners
            best_ids = ids

    if best_count < 6:
        return None, None, image_size

    return best_corners, best_ids, image_size


# ---------------------------------------------------------------------------
# Frame pair collection
# ---------------------------------------------------------------------------
def collect_valid_pairs(sony_dir: str, zed_dir: str) -> dict:
    """
    Iterate over matched frame pairs, run detection on both cameras,
    and collect valid pairs. Sony uses legacyPattern=True. ZED tries
    both legacy settings per frame and uses whichever gives more corners.
    """
    sony_board, _ = create_board(legacy=True)

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

    if len(sony_files) != len(zed_files):
        print(
            f"[WARN] Frame count mismatch: "
            f"{len(sony_files)} Sony vs {len(zed_files)} ZED."
        )

    pairs = list(zip(sony_files, zed_files))
    print(f"[INFO] Processing {len(pairs)} frame pairs …")
    print(f"       Sony board: legacyPattern=True")
    print(f"       ZED  board: auto (tries True and False, picks best)")

    obj_points_list = []
    sony_corners_list = []
    zed_corners_list = []
    sony_image_size = None
    zed_image_size = None
    rejected = 0

    for i, (sf, zf) in enumerate(pairs):
        sony_path = os.path.join(sony_dir, sf)
        zed_path = os.path.join(zed_dir, zf)

        s_corners, s_ids, s_size = detect_charuco(sony_path, sony_board)
        z_corners, z_ids, z_size = detect_charuco_zed(zed_path)

        if s_size is not None and sony_image_size is None:
            sony_image_size = s_size
        if z_size is not None and zed_image_size is None:
            zed_image_size = z_size

        if s_corners is None or z_corners is None:
            rejected += 1
            print(
                f"  [{i+1:03d}] REJECTED – insufficient detections "
                f"(sony={'ok' if s_corners is not None else 'fail'}, "
                f"zed={'ok' if z_corners is not None else 'fail'})"
            )
            continue

        s_id_set = set(s_ids.flatten())
        z_id_set = set(z_ids.flatten())
        common_ids = sorted(s_id_set & z_id_set)

        if len(common_ids) < 6:
            rejected += 1
            print(f"  [{i+1:03d}] REJECTED – only {len(common_ids)} common corners")
            continue

        s_mask = np.isin(s_ids.flatten(), common_ids)
        z_mask = np.isin(z_ids.flatten(), common_ids)

        all_board_corners = sony_board.getChessboardCorners()
        obj_pts = all_board_corners[np.array(common_ids)].astype(np.float32)

        obj_points_list.append(obj_pts)
        sony_corners_list.append(s_corners[s_mask].astype(np.float32))
        zed_corners_list.append(z_corners[z_mask].astype(np.float32))
        print(f"  [{i+1:03d}] OK – {len(common_ids)} common corners")

    print(
        f"\n[INFO] Valid pairs: {len(obj_points_list)} / {len(pairs)} "
        f"({rejected} rejected)"
    )

    return {
        "obj_points": obj_points_list,
        "sony_corners": sony_corners_list,
        "zed_corners": zed_corners_list,
        "sony_image_size": sony_image_size,
        "zed_image_size": zed_image_size,
    }


# ---------------------------------------------------------------------------
# Individual camera calibration
# ---------------------------------------------------------------------------
def calibrate_single_camera(
    obj_points: list, img_points: list, image_size: tuple, name: str
) -> tuple:
    """
    Calibrate one camera independently to obtain its intrinsic matrix
    and distortion coefficients.

    Returns
    -------
    K : 3×3 camera matrix
    D : distortion vector
    rms : reprojection RMS error (pixels)
    """
    print(f"\n[INFO] Calibrating {name} camera …")
    rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
        obj_points,
        img_points,
        image_size,
        None,
        None,
        flags=cv2.CALIB_RATIONAL_MODEL,
    )
    print(f"  RMS reprojection error ({name}): {rms:.4f} px")
    return K, D, rms, rvecs, tvecs


# ---------------------------------------------------------------------------
# Stereo calibration
# ---------------------------------------------------------------------------
def run_stereo_calibration(
    obj_points: list,
    sony_corners: list,
    zed_corners: list,
    K_sony: np.ndarray,
    D_sony: np.ndarray,
    K_zed: np.ndarray,
    D_zed: np.ndarray,
    sony_size: tuple,
    zed_size: tuple,
) -> dict:
    """
    Compute the extrinsic transform from ZED camera space to Sony camera space.

    Returns a dict containing R, T, E, F and the stereo RMS error.
    """
    print("\n[INFO] Running stereo calibration …")

    flags = cv2.CALIB_FIX_INTRINSIC  # keep the individually calibrated K and D

    rms, K_sony_out, D_sony_out, K_zed_out, D_zed_out, R, T, E, F = cv2.stereoCalibrate(
        obj_points,
        zed_corners,  # camera 1 = ZED  (source of skeleton tracking)
        sony_corners,  # camera 2 = Sony (destination / RGB feed)
        K_zed,
        D_zed,
        K_sony,
        D_sony,
        zed_size,  # image size of camera 1
        flags=flags,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-7),
    )

    print(f"  Stereo RMS reprojection error: {rms:.4f} px")
    print(
        f"  Translation ZED→Sony (metres): "
        f"X={T[0,0]:.4f}  Y={T[1,0]:.4f}  Z={T[2,0]:.4f}"
    )

    rod, _ = cv2.Rodrigues(R)
    deg = np.degrees(rod).flatten()
    print(
        f"  Rotation ZED→Sony (degrees):  "
        f"Rx={deg[0]:.3f}  Ry={deg[1]:.3f}  Rz={deg[2]:.3f}"
    )

    return {
        "R": R,
        "T": T,
        "E": E,
        "F": F,
        "stereo_rms": float(rms),
    }


# ---------------------------------------------------------------------------
# Output serialisation
# ---------------------------------------------------------------------------
def save_calibration(
    output_dir: str,
    dataset: str,
    K_sony: np.ndarray,
    D_sony: np.ndarray,
    rms_sony: float,
    K_zed: np.ndarray,
    D_zed: np.ndarray,
    rms_zed: float,
    stereo: dict,
    board_cfg: dict,
    n_pairs: int,
) -> str:
    """
    Serialise all calibration data to spatial_calibration.json.
    The JSON is structured so that the UE5 Blueprint can read R and T directly.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "spatial_calibration.json")

    rod, _ = cv2.Rodrigues(stereo["R"])

    payload = {
        "meta": {
            "dataset": dataset,
            "calibration_type": "charuco_stereo",
            "valid_pairs_used": n_pairs,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "board": board_cfg,
        "sony_camera": {
            "individual_rms_px": rms_sony,
            "camera_matrix": K_sony.tolist(),
            "distortion_coeffs": D_sony.tolist(),
        },
        "zed_camera": {
            "individual_rms_px": rms_zed,
            "camera_matrix": K_zed.tolist(),
            "distortion_coeffs": D_zed.tolist(),
        },
        "extrinsics": {
            "description": (
                "Transform from ZED camera space to Sony camera space. "
                "Apply R and T to a ZED world-space 3-D point to get the "
                "corresponding point in Sony camera space."
            ),
            "stereo_rms_px": stereo["stereo_rms"],
            "rotation_matrix_3x3": stereo["R"].tolist(),
            "rotation_rodrigues": rod.flatten().tolist(),
            "rotation_degrees": np.degrees(rod).flatten().tolist(),
            "translation_metres": stereo["T"].flatten().tolist(),
            "translation_cm": (stereo["T"].flatten() * 100).tolist(),
            "essential_matrix": stereo["E"].tolist(),
            "fundamental_matrix": stereo["F"].tolist(),
        },
        "ue5_offset": {
            "description": (
                "Ready-to-use offset values for the BP_TrackerAnchor LensOffset "
                "variable in Unreal Engine 5. Units are centimetres, matching "
                "Unreal's default unit system."
            ),
            "X_cm": float((stereo["T"].flatten() * 100)[0]),
            "Y_cm": float((stereo["T"].flatten() * 100)[1]),
            "Z_cm": float((stereo["T"].flatten() * 100)[2]),
        },
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\n[INFO] Calibration saved → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="ChArUco stereo calibration for ZED ↔ Sony camera pair."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Name of the dataset folder inside data/picked_for_alignment/",
    )
    args = parser.parse_args()

    paths = build_paths(args.dataset)

    for key, path in [("sony_rgb", paths["sony_rgb"]), ("zed_rgb", paths["zed_rgb"])]:
        if not os.path.isdir(path):
            print(f"[ERROR] Directory not found: {path}")
            print(
                f"        Run Phase 1 first and pick frames into "
                f"data/picked_for_alignment/{args.dataset}/"
            )
            sys.exit(1)

    board_cfg = {
        "cols": BOARD_COLS,
        "rows": BOARD_ROWS,
        "square_length": SQUARE_LENGTH,
        "marker_length": MARKER_LENGTH,
        "aruco_dict": "DICT_4X4_50",
        "sony_legacy": SONY_LEGACY_PATTERN,
        "zed_legacy": ZED_LEGACY_PATTERN,
    }

    print(f"\n{'='*60}")
    print(f"  Spatial Calibration – dataset: {args.dataset}")
    print(f"{'='*60}\n")

    data = collect_valid_pairs(paths["sony_rgb"], paths["zed_rgb"])

    n_valid = len(data["obj_points"])
    if n_valid < MIN_VALID_PAIRS:
        print(
            f"\n[ERROR] Only {n_valid} valid pairs found. "
            f"Minimum required: {MIN_VALID_PAIRS}."
        )
        print("        Add more frame pairs to picked_for_alignment/ " "and re-run.")
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
        output_dir=paths["json_out"],
        dataset=args.dataset,
        K_sony=K_sony,
        D_sony=D_sony,
        rms_sony=rms_sony,
        K_zed=K_zed,
        D_zed=D_zed,
        rms_zed=rms_zed,
        stereo=stereo,
        board_cfg=board_cfg,
        n_pairs=n_valid,
    )

    print(f"\n{'='*60}")
    print("  CALIBRATION COMPLETE")
    print(
        f"  Stereo RMS : {stereo['stereo_rms']:.4f} px  "
        f"(< 1.0 is good, < 0.5 is excellent)"
    )
    print(f"  Valid pairs: {n_valid}")
    print(f"{'='*60}\n")

    if stereo["stereo_rms"] > 2.0:
        print("[WARN] RMS error is high (> 2.0 px). Consider:")
        print("       - Removing blurry or partially visible frames")
        print("       - Adding more frames at varied depths and angles")
        print("       - Re-running validate_calibration.py to inspect per-frame errors")


if __name__ == "__main__":
    main()

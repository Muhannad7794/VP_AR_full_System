"""
calib_common.py
===============
Shared definitions for the spatial alignment pipeline.

Every step of the pipeline imports its board definition, detector, quality
rules, paths, intrinsics loaders and coordinate conversions from this module,
so the steps cannot drift apart.

Coordinate conventions
----------------------
OpenCV camera frame (used for all calibration maths and by the HLSL
reprojection shader):
    x right, y down, z forward. Right-handed. Units: metres.

Extrinsics (R, T) map a point from the ZED left camera frame to the Sony
camera frame:
    X_sony = R @ X_zed + T

Unreal Engine frame (used only for the BP_TrackerAnchor offset):
    X forward, Y right, Z up. Left-handed. Units: centimetres.
    Mapping from OpenCV: ue = (z, x, -y).

ZED intrinsics
--------------
Frames saved with sl.VIEW.LEFT and depth retrieved with sl.MEASURE.DEPTH are
rectified by the ZED SDK. Rectification resamples the image onto a virtual
pinhole camera whose intrinsics are reported by the SDK
(camera_configuration.calibration_parameters.left_cam) with zero distortion.
The depth map is defined on exactly that grid, so these intrinsics are used
as given and are never estimated from board images.
"""

import gzip
import json
import math
import os

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Board definition (calib.io ChArUco board)
# ---------------------------------------------------------------------------
BOARD_COLS = 11  # squares horizontally
BOARD_ROWS = 8  # squares vertically
SQUARE_LENGTH = 0.028  # metres
MARKER_LENGTH = 0.021  # metres
ARUCO_DICT_ID = cv2.aruco.DICT_4X4_50
LEGACY_PATTERN = True  # property of the printed board, identical for both cameras

# Area spanned by all inner corners on the board plane (square metres).
FULL_INNER_AREA = (BOARD_COLS - 2) * (BOARD_ROWS - 2) * SQUARE_LENGTH**2

# ---------------------------------------------------------------------------
# View quality rules
# ---------------------------------------------------------------------------
# A view is usable when it has enough corners AND the corners span enough of
# the board. Corner spread matters more than corner count: about 15 well
# spread sub-pixel corners already fix a board pose, while clustered or
# collinear corners do not.
MIN_CORNERS_VIEW = 15
MIN_BOARD_COVERAGE = 0.25  # fraction of FULL_INNER_AREA covered by the convex hull

# ---------------------------------------------------------------------------
# Stillness and hold rules
# ---------------------------------------------------------------------------
# The motion of a frame is the MEDIAN displacement of the board corners
# against the frames STILL_OFFSET indices before and after it (the larger of
# the two values). The median is used instead of the maximum because the
# position of one corner among ~70 is easily disturbed by video compression
# or a marker mis-identification; the maximum would then label a stationary
# board as moving, while the median follows the motion of the board as a
# whole. A frame is still when its motion is at most the threshold.
#
# The Sony camera decides which frames are still: its focal length is about
# twice that of the ZED, so the same board motion produces about twice the
# pixel displacement and a clearer signal. Both cameras sit on one rigid rig
# and observe the same board, so a board that is still in the Sony image is
# physically still. The ZED threshold is therefore intentionally lenient: it
# only rejects frames in which the ZED image shows clear motion that the Sony
# image does not explain, such as a frame taken at a different moment. Pairs
# that are inconsistent for any other reason are removed later by the outlier
# rejection of stereo_calibrate.py.
#
# A "hold" is a run of consecutive still frames showing one physical board
# pose. Each hold contributes at most one view, so no single pose is counted
# several times.
STILL_OFFSET = 3
STILL_THRESHOLD_PX = {"sony": 1.5, "zed": 3.0}
HOLD_SPLIT_PX = {"sony": 4.0, "zed": 2.5}

# ---------------------------------------------------------------------------
# Frame selection
# ---------------------------------------------------------------------------
# Feature scales for diversity-based selection. A difference of one scale unit
# counts as "one unit of novelty" in every feature, so no feature dominates
# because of its numeric range.
FEATURE_SCALES = {
    "u": 0.5,  # board centre, fraction of image width
    "v": 0.5,  # board centre, fraction of image height
    "inv_z": 0.4,  # inverse board distance, 1/m
    "nx": 0.35,  # board normal x component (sine of the tilt about the vertical axis)
    "ny": 0.35,  # board normal y component (sine of the tilt about the horizontal axis)
}
# Every TEST_EVERY-th hold in diversity order is reserved for validation.
TEST_EVERY = 4

# Calibration-set warnings: below these spans, translation and rotation (and
# focal length and distance) are poorly separated by the data.
MIN_INV_Z_SPAN = 0.25  # 1/m, e.g. 1.0 m to 1.33 m
MIN_TILT_DEG = 20.0

# ---------------------------------------------------------------------------
# Validation grading (corner transfer error on held-out pairs, pixels)
# ---------------------------------------------------------------------------
GRADE_EXCELLENT_PX = 0.5
GRADE_GOOD_PX = 1.0
GRADE_ACCEPTABLE_PX = 2.0

# Distance bins (metres) used for every error-versus-distance report.
DISTANCE_BINS_M = [0.0, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 100.0]

CAMERAS = ("sony", "zed")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def paths(dataset):
    """Returns every path used by the pipeline for one dataset."""
    data = os.path.join(BASE_DIR, "data")
    json_out = os.path.join(data, "json_output", dataset)
    plots = os.path.join(data, "plots", dataset, "spatial")
    return {
        "raw": os.path.join(data, "raw", dataset),
        "synced": os.path.join(data, "synced", dataset),
        "synced_sony": os.path.join(data, "synced", dataset, "sony_rgb"),
        "synced_zed": os.path.join(data, "synced", dataset, "zed_rgb"),
        "synced_depth": os.path.join(data, "synced", dataset, "zed_depth"),
        "picked": os.path.join(data, "picked_for_alignment", dataset),
        "picked_test": os.path.join(data, "picked_for_validation", dataset),
        "json_out": json_out,
        "plots": plots,
        "zed_intrinsics": os.path.join(json_out, "zed_intrinsics.json"),
        "detections": os.path.join(json_out, "detections.json.gz"),
        "detection_summary": os.path.join(json_out, "detection_summary.json"),
        "sony_intrinsics": os.path.join(json_out, "sony_intrinsics.json"),
        "pair_selection": os.path.join(json_out, "pair_selection.json"),
        "calibration": os.path.join(json_out, "spatial_calibration.json"),
        "validation": os.path.join(json_out, "validation_report.json"),
    }


def frame_names(directory):
    """Sorted frame stems (e.g. '00084') of the PNG files in a directory."""
    if not os.path.isdir(directory):
        return []
    return sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(directory)
        if f.lower().endswith(".png")
    )


def frame_index(name):
    return int(name)


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------
def save_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "wt") as f:
        json.dump(payload, f, indent=None if path.endswith(".gz") else 2)


def load_json(path, step_hint=None):
    if not os.path.isfile(path):
        hint = f" Run {step_hint} first." if step_hint else ""
        raise SystemExit(f"[ERROR] Missing file: {path}.{hint}")
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Board and detection
# ---------------------------------------------------------------------------
def create_board():
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    board = cv2.aruco.CharucoBoard(
        (BOARD_COLS, BOARD_ROWS), SQUARE_LENGTH, MARKER_LENGTH, dictionary
    )
    board.setLegacyPattern(LEGACY_PATTERN)
    return board


def board_corners_3d():
    """(N, 3) inner-corner coordinates on the board plane, indexed by ChArUco id."""
    return create_board().getChessboardCorners().astype(np.float64)


def create_detector():
    """ChArUco detector with sub-pixel marker refinement. The detector refines
    the interpolated chessboard corners to sub-pixel accuracy internally."""
    board = create_board()
    detector_params = cv2.aruco.DetectorParameters()
    detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    charuco_params = cv2.aruco.CharucoParameters()
    charuco_params.tryRefineMarkers = True
    return cv2.aruco.CharucoDetector(board, charuco_params, detector_params)


def detect_image(detector, image_path):
    """Detects the board in one image.

    Returns (result, image_size). result is None when no ChArUco corner was
    found, otherwise a dict with 'ids' (list of int), 'corners' (list of
    [x, y]) and 'n_markers'. image_size is (width, height) or None when the
    file cannot be read.
    """
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None, None
    size = (gray.shape[1], gray.shape[0])
    corners, ids, _, marker_ids = detector.detectBoard(gray)
    if corners is None or ids is None or len(ids) == 0:
        return None, size
    return {
        "ids": [int(i) for i in ids.flatten()],
        "corners": [
            [round(float(x), 4), round(float(y), 4)] for x, y in corners.reshape(-1, 2)
        ],
        "n_markers": 0 if marker_ids is None else int(len(marker_ids)),
    }, size


def board_coverage(ids, corners_3d):
    """Fraction of the full inner-corner area spanned by the given corner ids."""
    if len(ids) < 3:
        return 0.0
    pts = corners_3d[np.asarray(ids)][:, :2].astype(np.float32)
    hull = cv2.convexHull(pts)
    return float(cv2.contourArea(hull)) / FULL_INNER_AREA


def view_is_usable(
    ids, corners_3d, min_corners=MIN_CORNERS_VIEW, min_coverage=MIN_BOARD_COVERAGE
):
    return len(ids) >= min_corners and board_coverage(ids, corners_3d) >= min_coverage


def common_points(det_a, det_b):
    """Corners of the ids detected in both views, in matching order.

    Returns (ids, points_a (N, 2), points_b (N, 2)).
    """
    ids_a = np.asarray(det_a["ids"])
    ids_b = np.asarray(det_b["ids"])
    common, ia, ib = np.intersect1d(ids_a, ids_b, return_indices=True)
    pa = np.asarray(det_a["corners"], dtype=np.float64)[ia]
    pb = np.asarray(det_b["corners"], dtype=np.float64)[ib]
    return common.astype(int), pa, pb


def view_points(det):
    return np.asarray(det["ids"], dtype=int), np.asarray(
        det["corners"], dtype=np.float64
    )


# ---------------------------------------------------------------------------
# Stillness and holds
# ---------------------------------------------------------------------------
def median_displacement(det_a, det_b):
    """Median corner displacement between two detections of the same camera
    (pixels), or None when they share fewer than 6 corners."""
    if det_a is None or det_b is None:
        return None
    ids, pa, pb = common_points(det_a, det_b)
    if len(ids) < 6:
        return None
    return float(np.median(np.linalg.norm(pa - pb, axis=1)))


def frame_motion_px(detections_by_index, index, camera):
    """Motion of the board at one frame in pixels: the larger of the median
    corner displacements against the neighbours STILL_OFFSET indices before and
    after. Returns None when the frame or both neighbours cannot be compared."""
    current = detections_by_index.get(index, {}).get(camera)
    if current is None:
        return None
    shifts = []
    for neighbour in (index - STILL_OFFSET, index + STILL_OFFSET):
        other = detections_by_index.get(neighbour, {}).get(camera)
        shift = median_displacement(current, other)
        if shift is not None:
            shifts.append(shift)
    return max(shifts) if shifts else None


def is_still(motion_px, camera, threshold=None):
    """True when the measured motion does not exceed the stillness threshold.
    A frame without a motion measurement is treated as moving."""
    if motion_px is None:
        return False
    limit = STILL_THRESHOLD_PX[camera] if threshold is None else threshold
    return motion_px <= limit


def build_holds(indices, detections_by_index, reference_camera):
    """Groups qualifying frame indices into holds (one physical pose each).

    indices: sorted frame indices that already passed the quality and
    stillness tests. A new hold starts after a gap in the index sequence, or
    when the board in the reference camera has drifted more than
    HOLD_SPLIT_PX from the first frame of the current hold.
    """
    holds = []
    current = []
    for idx in indices:
        if current:
            gap = idx - current[-1] > STILL_OFFSET
            drift = median_displacement(
                detections_by_index[current[0]][reference_camera],
                detections_by_index[idx][reference_camera],
            )
            if gap or drift is None or drift > HOLD_SPLIT_PX[reference_camera]:
                holds.append(current)
                current = []
        current.append(idx)
    if current:
        holds.append(current)
    return holds


def hold_representative(hold, corner_count):
    """Frame of a hold used as its view: the frame nearest the middle of the
    hold among frames with at least 90 % of the hold's best corner count. The
    middle of a hold is the frame least affected by a small temporal offset
    between the two recordings."""
    best = max(corner_count(i) for i in hold)
    eligible = [i for i in hold if corner_count(i) >= 0.9 * best]
    middle = hold[len(hold) // 2]
    return min(eligible, key=lambda i: abs(i - middle))


# ---------------------------------------------------------------------------
# Pose features and diversity selection
# ---------------------------------------------------------------------------
def solve_board_pose(obj_pts, img_pts, K, D):
    """Board pose (rvec, tvec) in the camera frame, or None on failure."""
    ok, rvec, tvec = cv2.solvePnP(
        obj_pts.astype(np.float64),
        img_pts.astype(np.float64),
        K,
        D,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None
    rvec, tvec = cv2.solvePnPRefineLM(
        obj_pts.astype(np.float64), img_pts.astype(np.float64), K, D, rvec, tvec
    )
    return rvec, tvec


def pose_features(rvec, tvec, img_pts, image_size):
    """Diversity features of one board view.

    u, v   : centre of the detected corners as a fraction of the image size
    inv_z  : inverse distance of the board centre (1/m)
    nx, ny : x and y components of the board normal in the camera frame
    tilt   : angle between the board normal and the optical axis (degrees)
    z      : distance of the board centre (m)
    """
    R, _ = cv2.Rodrigues(rvec)
    # Centre of the inner-corner grid on the board plane.
    centre_board = np.array(
        [BOARD_COLS * SQUARE_LENGTH / 2, BOARD_ROWS * SQUARE_LENGTH / 2, 0.0]
    )
    centre_cam = R @ centre_board + tvec.reshape(3)
    normal = R[:, 2]
    if normal[2] > 0:
        normal = -normal  # orient the normal towards the camera
    tilt = math.degrees(math.acos(min(1.0, abs(float(normal[2])))))
    pts = np.asarray(img_pts).reshape(-1, 2)
    return {
        "u": float(pts[:, 0].mean() / image_size[0]),
        "v": float(pts[:, 1].mean() / image_size[1]),
        "inv_z": float(1.0 / max(centre_cam[2], 1e-6)),
        "nx": float(normal[0]),
        "ny": float(normal[1]),
        "tilt_deg": float(tilt),
        "z_m": float(centre_cam[2]),
    }


def feature_vector(features):
    return np.array(
        [features[k] / s for k, s in FEATURE_SCALES.items()], dtype=np.float64
    )


def diversity_order(vectors, start_index=0):
    """Farthest-point ordering: each next element is the one farthest from
    everything already chosen. The prefix of length n is the most diverse
    subset of size n that this greedy rule can find."""
    vectors = np.asarray(vectors, dtype=np.float64)
    n = len(vectors)
    if n == 0:
        return []
    order = [start_index]
    min_dist = np.linalg.norm(vectors - vectors[start_index], axis=1)
    min_dist[start_index] = -1.0
    while len(order) < n:
        nxt = int(np.argmax(min_dist))
        order.append(nxt)
        min_dist = np.minimum(min_dist, np.linalg.norm(vectors - vectors[nxt], axis=1))
        min_dist[order] = -1.0
    return order


def split_calibration_test(order, max_calibration):
    """Interleaves the diversity order into a calibration set and a held-out
    test set, so both sets span the full variety of the data. Every
    TEST_EVERY-th element goes to the test set; calibration elements beyond
    max_calibration are also moved to the test set."""
    calibration, test = [], []
    for rank, item in enumerate(order):
        if rank % TEST_EVERY == TEST_EVERY - 1:
            test.append(item)
        elif len(calibration) < max_calibration:
            calibration.append(item)
        else:
            test.append(item)
    return calibration, test


def span_warnings(features_list):
    """Warnings when the calibration views do not vary enough in distance or tilt."""
    warnings = []
    if not features_list:
        return ["No calibration views."]
    inv_z = [f["inv_z"] for f in features_list]
    tilts = [f["tilt_deg"] for f in features_list]
    span = max(inv_z) - min(inv_z)
    if span < MIN_INV_Z_SPAN:
        warnings.append(
            f"Distance variety is low: board distance spans {1 / max(inv_z):.2f}-{1 / min(inv_z):.2f} m "
            f"(1/z span {span:.2f} 1/m, target >= {MIN_INV_Z_SPAN}). Translation and rotation are "
            f"poorly separated and errors will grow away from this range."
        )
    if max(tilts) < MIN_TILT_DEG:
        warnings.append(
            f"Tilt variety is low: largest board tilt {max(tilts):.1f} deg (target >= {MIN_TILT_DEG}). "
            f"Focal length and distance are poorly separated."
        )
    return warnings


# ---------------------------------------------------------------------------
# Intrinsics
# ---------------------------------------------------------------------------
def camera_matrix(fx, fy, cx, cy):
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def load_zed_intrinsics(dataset):
    """Rectified ZED left-camera intrinsics written by print_zed_intrinsics.py.

    Returns (K, D, image_size). D is all zeros: rectified images have no
    distortion.
    """
    info = load_json(paths(dataset)["zed_intrinsics"], "print_zed_intrinsics.py")
    rect = info["rectified_left"]
    K = camera_matrix(rect["fx"], rect["fy"], rect["cx"], rect["cy"])
    size = (int(rect["width"]), int(rect["height"]))
    return K, np.zeros(5), size


def load_sony_intrinsics(dataset, which):
    """Sony intrinsics written by calibrate_sony_intrinsics.py.

    which: 'physical' for the model selected on held-out error, 'runtime'
    for the pinhole model used by the Unreal Engine shader and camera.
    Returns (K, D, image_size, model_name).
    """
    info = load_json(paths(dataset)["sony_intrinsics"], "calibrate_sony_intrinsics.py")
    entry = info[which]
    K = np.asarray(entry["camera_matrix"], dtype=np.float64)
    D = np.asarray(entry["distortion_coeffs"], dtype=np.float64).reshape(-1)
    return K, D, tuple(info["image_size"]), entry["model"]


# ---------------------------------------------------------------------------
# Error statistics
# ---------------------------------------------------------------------------
def rms(values):
    values = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(values**2))) if values.size else float("nan")


def distance_bin_label(z):
    for lo, hi in zip(DISTANCE_BINS_M[:-1], DISTANCE_BINS_M[1:]):
        if lo <= z < hi:
            return f"{lo:.2f}-{hi:.2f} m" if hi < 100 else f">= {lo:.2f} m"
    return "out of range"


def grade(test_rms_px):
    if not np.isfinite(test_rms_px):
        return "Not available"
    if test_rms_px < GRADE_EXCELLENT_PX:
        return "Excellent"
    if test_rms_px < GRADE_GOOD_PX:
        return "Good"
    if test_rms_px < GRADE_ACCEPTABLE_PX:
        return "Acceptable"
    return "Poor"


def robust_outliers(values, k=3.0, floor=0.5):
    """Indices of values above median + k * 1.4826 * MAD (with an absolute
    floor, so a very consistent set does not reject normal scatter)."""
    values = np.asarray(values, dtype=np.float64)
    median = np.median(values)
    mad = 1.4826 * np.median(np.abs(values - median))
    limit = median + max(k * mad, floor)
    return [int(i) for i in np.where(values > limit)[0]], float(limit)


# ---------------------------------------------------------------------------
# OpenCV <-> Unreal Engine conversion
# ---------------------------------------------------------------------------
# Rows map OpenCV camera axes to Unreal axes: ue = C @ cv = (z, x, -y).
CV_TO_UE = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def ue_rotation_matrix(pitch_deg, yaw_deg, roll_deg):
    """Unreal Engine FRotationMatrix for an FRotator. Rows are the rotated
    X (forward), Y (right) and Z (up) axes in parent coordinates."""
    p, y, r = (math.radians(a) for a in (pitch_deg, yaw_deg, roll_deg))
    sp, cp, sy, cy, sr, cr = (
        math.sin(p),
        math.cos(p),
        math.sin(y),
        math.cos(y),
        math.sin(r),
        math.cos(r),
    )
    return np.array(
        [
            [cp * cy, cp * sy, sp],
            [sr * sp * cy - cr * sy, sr * sp * sy + cr * cy, -sr * cp],
            [-(cr * sp * cy + sr * sy), cy * sr - cr * sp * sy, cr * cp],
        ]
    )


def ue_rotator_from_axes(x_axis, y_axis, z_axis):
    """FRotator (pitch, yaw, roll in degrees) of a frame given its X, Y and Z
    axes in parent Unreal coordinates, following FMatrix::Rotator()."""
    x_axis, y_axis, z_axis = (
        np.asarray(a, dtype=np.float64) for a in (x_axis, y_axis, z_axis)
    )
    pitch = math.degrees(math.atan2(x_axis[2], math.hypot(x_axis[0], x_axis[1])))
    yaw = math.degrees(math.atan2(x_axis[1], x_axis[0]))
    sy_axis = ue_rotation_matrix(pitch, yaw, 0.0)[1]
    roll = math.degrees(math.atan2(float(z_axis @ sy_axis), float(y_axis @ sy_axis)))
    return pitch, yaw, roll


def sony_pose_in_zed_ue(R, T):
    """Pose of the Sony camera relative to the ZED left camera in the ZED's
    local Unreal axes.

    Returns (location_cm, (pitch, yaw, roll) in degrees). The location is the
    Sony optical centre; the rotation turns the ZED's view direction into the
    Sony's view direction.
    """
    R = np.asarray(R, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64).reshape(3)
    centre_cv = -R.T @ T  # Sony optical centre in the ZED OpenCV frame
    location_cm = CV_TO_UE @ centre_cv * 100.0
    # Sony camera axes expressed in the ZED OpenCV frame (columns of R^T).
    forward_cv = R.T @ np.array([0.0, 0.0, 1.0])
    right_cv = R.T @ np.array([1.0, 0.0, 0.0])
    up_cv = R.T @ np.array([0.0, -1.0, 0.0])
    rotator = ue_rotator_from_axes(
        CV_TO_UE @ forward_cv, CV_TO_UE @ right_cv, CV_TO_UE @ up_cv
    )
    return location_cm, rotator

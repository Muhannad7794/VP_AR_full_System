"""
stereo_calibrate.py
===================
Step 4 of the spatial alignment pipeline.

Estimates the extrinsics (R, T) from the ZED left camera to the Sony camera:
    X_sony = R @ X_zed + T      (OpenCV camera frames, metres)

Both cameras' intrinsics are held fixed:
    ZED   rectified SDK intrinsics, zero distortion (print_zed_intrinsics.py)
    Sony  intrinsics from Sony-only views (calibrate_sony_intrinsics.py)

Fixing the intrinsics leaves six unknowns, so errors in focal length or
principal point can no longer be absorbed into R and T.

The solve uses the Sony physical model (camera matrix and lens distortion),
so R and T describe the true geometry of the rig. The runtime values for
Unreal Engine use the same R and T and the same camera matrix without the
distortion; the distortion they omit is a known radial displacement reported
by calibrate_sony_intrinsics.py and measured by validate_calibration.py.

Robustness and uncertainty:
    - Pairs whose error is far above the rest (for example a mis-synced pair)
      are removed once with a median/MAD rule, and the solve is repeated.
    - A bootstrap over the calibration pairs gives the standard deviation of
      every translation component and of the rotation. It measures noise in
      the pairs only: the intrinsics are held fixed, so a bias in them (for
      example a Sony focal length estimated from views at one distance) is
      not included. The held-out, distance-resolved results of
      validate_calibration.py are the check for that.
    - The relative pose is also computed independently per pair from two
      separate board poses; the spread of these estimates shows how
      consistent the pairs are.

Output:
    data/json_output/<dataset>/spatial_calibration.json

Usage:
    python3 spatial_alignment/stereo_calibrate.py --dataset <name> [--bootstrap 200]
"""

import argparse
import math
import sys
import time

import cv2
import numpy as np

import calib_common as cc

CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 300, 1e-10)
MAX_REJECTED_FRACTION = 0.2


def pair_arrays(frame, corners_3d):
    ids, pts_s, pts_z = cc.common_points(frame["sony"], frame["zed"])
    return (
        corners_3d[ids].astype(np.float32),
        pts_z.astype(np.float32),
        pts_s.astype(np.float32),
    )


def solve_extrinsics(obj, img_z, img_s, K_z, D_z, K_s, D_s, zed_size):
    """Stereo solve with fixed intrinsics. Camera 1 is the ZED, camera 2 the Sony."""
    out = cv2.stereoCalibrateExtended(
        obj,
        img_z,
        img_s,
        K_z,
        D_z,
        K_s,
        D_s,
        zed_size,
        None,
        None,
        flags=cv2.CALIB_FIX_INTRINSIC,
        criteria=CRITERIA,
    )
    rms, R, T, E, F, per_view = out[0], out[5], out[6], out[7], out[8], out[11]
    per_view = np.asarray(per_view).reshape(-1, 2)
    combined = np.sqrt(np.mean(per_view**2, axis=1))
    return float(rms), R, T.reshape(3), E, F, combined


def rotation_angle_deg(Ra, Rb):
    """Angle of the rotation that takes Ra to Rb."""
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def per_pair_relative_poses(obj, img_z, img_s, K_z, D_z, K_s, D_s):
    """Relative pose from each pair on its own: board pose in each camera,
    then T_sony_board * inverse(T_zed_board)."""
    poses = []
    for o, iz, is_ in zip(obj, img_z, img_s):
        pz = cc.solve_board_pose(o, iz, K_z, D_z)
        ps = cc.solve_board_pose(o, is_, K_s, D_s)
        if pz is None or ps is None:
            continue
        Rz, _ = cv2.Rodrigues(pz[0])
        Rs, _ = cv2.Rodrigues(ps[0])
        R = Rs @ Rz.T
        T = ps[1].reshape(3) - R @ pz[1].reshape(3)
        poses.append((R, T))
    return poses


def consistency(poses, R_ref, T_ref):
    if not poses:
        return None
    T = np.array([t for _, t in poses])
    angles = [rotation_angle_deg(R_ref, r) for r, _ in poses]
    return {
        "n_pairs": len(poses),
        "translation_std_mm": (T.std(axis=0) * 1000).tolist(),
        "translation_median_minus_joint_mm": (
            (np.median(T, axis=0) - T_ref) * 1000
        ).tolist(),
        "rotation_deviation_median_deg": float(np.median(angles)),
        "rotation_deviation_max_deg": float(np.max(angles)),
    }


def bootstrap(obj, img_z, img_s, K_z, D_z, K_s, D_s, zed_size, R_ref, T_ref, n, seed=0):
    """Standard deviation of R and T over resamples of the calibration pairs."""
    rng = np.random.default_rng(seed)
    Ts, angles = [], []
    m = len(obj)
    for _ in range(n):
        idx = rng.integers(0, m, m)
        if len(set(idx.tolist())) < 3:
            continue
        try:
            _, R, T, _, _, _ = solve_extrinsics(
                [obj[i] for i in idx],
                [img_z[i] for i in idx],
                [img_s[i] for i in idx],
                K_z,
                D_z,
                K_s,
                D_s,
                zed_size,
            )
        except cv2.error:
            continue
        Ts.append(T)
        angles.append(rotation_angle_deg(R_ref, R))
    if not Ts:
        return None
    Ts = np.array(Ts)
    return {
        "n_resamples": len(Ts),
        "translation_std_mm": (Ts.std(axis=0) * 1000).tolist(),
        "rotation_std_deg": float(np.sqrt(np.mean(np.square(angles)))),
    }


def solution_record(R, T, E, F, rms, per_view, names, rejected, K_s, D_s, model):
    rod, _ = cv2.Rodrigues(R)
    return {
        "sony_model": model,
        "sony_camera_matrix": K_s.tolist(),
        "sony_distortion_coeffs": D_s.tolist(),
        "rotation_matrix_3x3": R.tolist(),
        "rotation_rodrigues": rod.flatten().tolist(),
        "rotation_degrees": np.degrees(rod).flatten().tolist(),
        "translation_metres": T.tolist(),
        "translation_cm": (T * 100).tolist(),
        "essential_matrix": E.tolist(),
        "fundamental_matrix": F.tolist(),
        "stereo_rms_px": rms,
        "per_pair_rms_px": {n: float(e) for n, e in zip(names, per_view)},
        "rejected_pairs": rejected,
    }


def run_solution(label, names, frames, corners_3d, K_z, D_z, K_s, D_s, zed_size):
    obj, img_z, img_s = zip(*(pair_arrays(frames[n], corners_3d) for n in names))
    obj, img_z, img_s = list(obj), list(img_z), list(img_s)
    rms, R, T, E, F, per_view = solve_extrinsics(
        obj, img_z, img_s, K_z, D_z, K_s, D_s, zed_size
    )

    bad, limit = cc.robust_outliers(per_view)
    rejected = {}
    if bad and len(bad) <= MAX_REJECTED_FRACTION * len(names):
        rejected = {names[i]: float(per_view[i]) for i in bad}
        keep = [i for i in range(len(names)) if i not in bad]
        names = [names[i] for i in keep]
        obj, img_z, img_s = (
            [obj[i] for i in keep],
            [img_z[i] for i in keep],
            [img_s[i] for i in keep],
        )
        rms, R, T, E, F, per_view = solve_extrinsics(
            obj, img_z, img_s, K_z, D_z, K_s, D_s, zed_size
        )
    elif bad:
        print(
            f"  [WARN] {label}: {len(bad)} pairs above {limit:.2f} px; too many to reject automatically."
        )

    print(
        f"  {label:8s} stereo RMS {rms:.3f} px over {len(names)} pairs"
        + (f", rejected {list(rejected)}" if rejected else "")
    )
    return {
        "names": names,
        "obj": obj,
        "img_z": img_z,
        "img_s": img_s,
        "rms": rms,
        "R": R,
        "T": T,
        "E": E,
        "F": F,
        "per_view": per_view,
        "rejected": rejected,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Estimate the ZED-to-Sony extrinsics with fixed intrinsics."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--bootstrap", type=int, default=200, help="Bootstrap resamples (0 disables)."
    )
    args = parser.parse_args()

    p = cc.paths(args.dataset)
    frames = cc.load_json(p["detections"], "inspect_detections.py")["frames"]
    selection = cc.load_json(p["pair_selection"], "pick_frames.py")
    zed_info = cc.load_json(p["zed_intrinsics"], "print_zed_intrinsics.py")
    K_z, D_z, zed_size = cc.load_zed_intrinsics(args.dataset)
    K_s, D_s, sony_size, model_p = cc.load_sony_intrinsics(args.dataset, "physical")
    K_sr, D_sr, _, model_r = cc.load_sony_intrinsics(args.dataset, "runtime")
    corners_3d = cc.board_corners_3d()
    names = selection["calibration"]

    print(f"\n{'=' * 60}\n  Stereo calibration - dataset: {args.dataset}\n{'=' * 60}\n")
    print(
        f"  ZED  K (SDK rectified): fx {K_z[0, 0]:.3f} fy {K_z[1, 1]:.3f} cx {K_z[0, 2]:.3f} cy {K_z[1, 2]:.3f}"
    )
    print(
        f"  Sony K ({model_p}): fx {K_s[0, 0]:.3f} fy {K_s[1, 1]:.3f} cx {K_s[0, 2]:.3f} cy {K_s[1, 2]:.3f}"
    )
    print(f"  Sony distortion       : {np.round(D_s, 5).tolist()}")
    print(
        f"  Board square length   : {cc.SQUARE_LENGTH * 1000:.3f} mm (sets the metric scale of T)"
    )
    print(f"  Calibration pairs     : {len(names)}\n")

    sol = run_solution(
        "stereo", names, frames, corners_3d, K_z, D_z, K_s, D_s, zed_size
    )
    R, T = sol["R"], sol["T"]

    poses = per_pair_relative_poses(
        sol["obj"], sol["img_z"], sol["img_s"], K_z, D_z, K_s, D_s
    )
    cons = consistency(poses, R, T)
    boot = None
    if args.bootstrap > 0:
        boot = bootstrap(
            sol["obj"],
            sol["img_z"],
            sol["img_s"],
            K_z,
            D_z,
            K_s,
            D_s,
            zed_size,
            R,
            T,
            args.bootstrap,
        )

    location_cm, rotator = cc.sony_pose_in_zed_ue(R, T)
    runtime_values = {
        "description": (
            "Values for M_ZED_DepthReproject (OpenCV frames, metres, pixels) and for "
            "BP_TrackerAnchor (Unreal axes, centimetres, degrees)."
        ),
        "Sony_FX": float(K_sr[0, 0]),
        "Sony_FY": float(K_sr[1, 1]),
        "Sony_CX": float(K_sr[0, 2]),
        "Sony_CY": float(K_sr[1, 2]),
        "Sony_Distortion_k1_k2_p1_p2_k3": [float(v) for v in D_s[:5]],
        "ZED_FX": float(K_z[0, 0]),
        "ZED_FY": float(K_z[1, 1]),
        "ZED_CX": float(K_z[0, 2]),
        "ZED_CY": float(K_z[1, 2]),
        "R_Row0": R[0].tolist(),
        "R_Row1": R[1].tolist(),
        "R_Row2": R[2].tolist(),
        "T_Metres": T.tolist(),
        "ImageWidth": int(sony_size[0]),
        "ImageHeight": int(sony_size[1]),
        "lens_offset_location_cm": {
            "X": float(location_cm[0]),
            "Y": float(location_cm[1]),
            "Z": float(location_cm[2]),
        },
        "lens_offset_rotation_deg": {
            "Pitch": rotator[0],
            "Yaw": rotator[1],
            "Roll": rotator[2],
        },
        "lens_offset_convention": (
            "Sony optical centre and orientation relative to the ZED left optical centre, "
            "in the ZED's local Unreal axes (X forward, Y right, Z up)."
        ),
    }

    payload = {
        "meta": {
            "dataset": args.dataset,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "opencv_version": cv2.__version__,
            "zed_intrinsics_source": zed_info.get("source"),
            "zed_sdk_version": zed_info.get("sdk_version"),
            "valid_pairs_used": len(sol["names"]),
            "convention": "X_sony = R @ X_zed + T (OpenCV camera frames, metres)",
        },
        "board": cc.board_definition(),
        "zed_camera": {
            "camera_matrix": K_z.tolist(),
            "distortion_coeffs": D_z.tolist(),
            "image_size": list(zed_size),
            "model": "SDK rectified pinhole",
        },
        "sony_camera": {
            "camera_matrix": K_s.tolist(),
            "distortion_coeffs": D_s.tolist(),
            "image_size": list(sony_size),
            "model": model_p,
        },
        "sony_camera_runtime": {
            "camera_matrix": K_sr.tolist(),
            "distortion_coeffs": D_sr.tolist(),
            "image_size": list(sony_size),
            "model": model_r,
        },
        "extrinsics": solution_record(
            R,
            T,
            sol["E"],
            sol["F"],
            sol["rms"],
            sol["per_view"],
            sol["names"],
            sol["rejected"],
            K_s,
            D_s,
            model_p,
        ),
        "uncertainty": {"bootstrap": boot, "per_pair_consistency": cons},
        "runtime": runtime_values,
        "ue5_offset": {
            "X_cm": float(location_cm[0]),
            "Y_cm": float(location_cm[1]),
            "Z_cm": float(location_cm[2]),
        },
    }
    cc.save_json(p["calibration"], payload)

    rod_deg = np.degrees(cv2.Rodrigues(R)[0]).flatten()
    print(
        f"\n  Translation ZED->Sony (m) : X={T[0]:+.4f}  Y={T[1]:+.4f}  Z={T[2]:+.4f}"
    )
    print(
        f"  Rotation (Rodrigues, deg) : {rod_deg[0]:+.3f}  {rod_deg[1]:+.3f}  {rod_deg[2]:+.3f}"
    )
    if boot:
        s = boot["translation_std_mm"]
        print(
            f"  Bootstrap std             : T {s[0]:.1f} / {s[1]:.1f} / {s[2]:.1f} mm, "
            f"R {boot['rotation_std_deg']:.3f} deg ({boot['n_resamples']} resamples; "
            f"pair noise only, excludes intrinsic bias)"
        )
    if cons:
        s = cons["translation_std_mm"]
        print(
            f"  Per-pair spread           : T {s[0]:.1f} / {s[1]:.1f} / {s[2]:.1f} mm, "
            f"R median {cons['rotation_deviation_median_deg']:.3f} deg"
        )
    print(f"\n  Saved: {p['calibration']}\n")


if __name__ == "__main__":
    sys.exit(main())

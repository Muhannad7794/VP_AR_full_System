"""
calibrate_sony_intrinsics.py
============================
Step 2 of the spatial alignment pipeline.

Estimates the Sony intrinsics from Sony frames alone. The ZED does not need
to see the board in these frames and no temporal pairing is involved, so
close-range views and views at the edges of the Sony frame (which the ZED
cannot share) are all usable.

Method
------
1. Usable and still Sony frames are grouped into holds (one board pose each)
   and one representative frame is taken per hold.
2. A diversity ordering over board position, distance and tilt splits the
   holds into a calibration set and a held-out test set that both span the
   full variety of the recording.
3. Several distortion models, from pure pinhole to the full five-coefficient
   model, are fitted on the calibration set. Each is scored on the test set:
   the board pose of every test view is re-estimated with the model and the
   reprojection error is measured. The simplest model within 2 % of the best
   test error is selected as the physical model.
4. The runtime model is the physical model's camera matrix with the
   distortion removed, because the Unreal Engine camera and the reprojection
   shader are pinhole. Its error is the lens distortion alone: zero at the
   principal point and largest at the frame corners. It is reported as a
   displacement map.

   A pinhole model FITTED to the board views is deliberately not used at
   runtime. Fitting a pinhole to a lens with distortion lengthens the focal
   length to spread the distortion over the frame, and the extrinsics solved
   with that focal length absorb the error into the translation (mostly along
   the optical axis), which also corrupts the lens offset used to place the
   CG camera.

   Note on the held-out test: re-estimating the board pose per test view
   partly absorbs focal-length errors (focal length and board distance trade
   off), so this single-camera test separates distortion models but is weak
   at verifying the focal length itself. The cross-camera validation in
   validate_calibration.py is the decisive check.

Outputs:
    data/json_output/<dataset>/sony_intrinsics.json
    data/plots/<dataset>/spatial/sony_intrinsics_models.png
    data/plots/<dataset>/spatial/sony_intrinsics_coverage.png
    data/plots/<dataset>/spatial/sony_runtime_displacement.png

Usage:
    python3 spatial_alignment/calibrate_sony_intrinsics.py --dataset <name>
        [--max-views 40] [--exclude frames.txt] [--include frames.txt]
"""

import argparse
import os
import sys
import time

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import calib_common as cc

BASE_FLAGS = cv2.CALIB_USE_INTRINSIC_GUESS
# Ordered from simplest to most complex. Each entry: (name, flags).
MODELS = [
    (
        "pinhole",
        BASE_FLAGS
        | cv2.CALIB_FIX_K1
        | cv2.CALIB_FIX_K2
        | cv2.CALIB_FIX_K3
        | cv2.CALIB_ZERO_TANGENT_DIST,
    ),
    (
        "k1",
        BASE_FLAGS | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST,
    ),
    ("k1_k2", BASE_FLAGS | cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST),
    ("k1_k2_tangential", BASE_FLAGS | cv2.CALIB_FIX_K3),
    ("k1_k2_k3_tangential", BASE_FLAGS),
]
# A more complex model must beat the best simpler one by more than this to be chosen.
SELECTION_TOLERANCE_REL = 0.02
SELECTION_TOLERANCE_ABS_PX = 0.01
MIN_CALIBRATION_VIEWS = 8
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-9)
UNDISTORT_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 50, 1e-10)


def read_name_list(path):
    if not path:
        return set()
    with open(path) as f:
        return {
            line.strip().replace(".png", "")
            for line in f
            if line.strip() and not line.startswith("#")
        }


def view_arrays(det, corners_3d):
    ids, pts = cc.view_points(det)
    return corners_3d[ids].astype(np.float32), pts.astype(np.float32)


def fit_model(flags, obj, img, image_size, K0):
    rms, K, D, _, _, std_int, _, per_view = cv2.calibrateCameraExtended(
        obj, img, image_size, K0.copy(), np.zeros(5), flags=flags, criteria=CRITERIA
    )
    return float(rms), K, D.reshape(-1), std_int.reshape(-1), per_view.reshape(-1)


def heldout_errors(K, D, obj, img):
    """Per-point reprojection errors on views not used for fitting. The board
    pose of each view is re-estimated with the given intrinsics."""
    errors = []
    for o, i in zip(obj, img):
        pose = cc.solve_board_pose(o, i, K, D)
        if pose is None:
            continue
        proj, _ = cv2.projectPoints(o.astype(np.float64), pose[0], pose[1], K, D)
        errors.append(np.linalg.norm(proj.reshape(-1, 2) - i.reshape(-1, 2), axis=1))
    return np.concatenate(errors) if errors else np.array([])


def runtime_displacement(K_phys, D_phys, K_run, image_size, step=20):
    """Pixel displacement between the physical model and the pinhole runtime
    model for the same viewing ray, over a grid covering the image. With
    K_run equal to K_phys this is the displacement caused by the lens
    distortion alone."""
    w, h = image_size
    xs, ys = np.meshgrid(
        np.arange(0, w, step, dtype=np.float64), np.arange(0, h, step, dtype=np.float64)
    )
    grid = np.stack([xs.ravel(), ys.ravel()], axis=1)
    rays = cv2.undistortPointsIter(
        grid.reshape(-1, 1, 2), K_phys, D_phys, None, None, UNDISTORT_CRITERIA
    ).reshape(-1, 2)
    proj = np.stack(
        [
            K_run[0, 0] * rays[:, 0] + K_run[0, 2],
            K_run[1, 1] * rays[:, 1] + K_run[1, 2],
        ],
        axis=1,
    )
    disp = np.linalg.norm(proj - grid, axis=1)
    # Central region: inside 80 % of the half-diagonal from the image centre.
    r = np.hypot(grid[:, 0] - w / 2, grid[:, 1] - h / 2) / np.hypot(w / 2, h / 2)
    return disp.reshape(xs.shape), {
        "max_px": float(disp.max()),
        "p95_px": float(np.percentile(disp, 95)),
        "max_central_80pct_px": float(disp[r <= 0.8].max()),
        "mean_px": float(disp.mean()),
    }


def plot_models(results, chosen, out):
    names = [r["model"] for r in results]
    train = [r["train_rms_px"] for r in results]
    test = [r["test_rms_px"] for r in results]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 0.18, train, 0.36, label="Calibration views (fitted)", color="#90CAF9")
    ax.bar(x + 0.18, test, 0.36, label="Held-out views", color="#1565C0")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [n + (" (selected)" if n == chosen else "") for n in names], rotation=15
    )
    ax.set_ylabel("RMS reprojection error (px)")
    ax.set_title("Sony distortion model comparison")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    plt.close()


def plot_coverage(calib_img, test_img, image_size, out):
    fig, ax = plt.subplots(figsize=(10, 6))
    for pts, color, label in (
        (calib_img, "#2E7D32", "Calibration views"),
        (test_img, "#EF6C00", "Held-out views"),
    ):
        if pts:
            allp = np.concatenate([p.reshape(-1, 2) for p in pts])
            ax.scatter(allp[:, 0], allp[:, 1], s=2, color=color, label=label)
    ax.set_xlim(0, image_size[0])
    ax.set_ylim(image_size[1], 0)
    ax.set_aspect("equal")
    ax.set_title("Sony corner coverage (intrinsics step)")
    ax.legend(loc="upper right", markerscale=6)
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    plt.close()


def plot_displacement(disp, stats, image_size, out):
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(disp, extent=[0, image_size[0], image_size[1], 0], cmap="viridis")
    plt.colorbar(im, ax=ax, label="Displacement (px)")
    ax.set_title(
        "Lens distortion omitted by the pinhole runtime model\n"
        f"max {stats['max_px']:.2f} px, central 80 % max {stats['max_central_80pct_px']:.2f} px"
    )
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Estimate the Sony intrinsics from Sony-only views."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--max-views", type=int, default=40, help="Maximum calibration views."
    )
    parser.add_argument(
        "--exclude", help="Text file with frame names to exclude, one per line."
    )
    parser.add_argument(
        "--include", help="Text file with frame names forced into the calibration set."
    )
    args = parser.parse_args()

    p = cc.paths(args.dataset)
    det = cc.load_json(p["detections"], "inspect_detections.py")
    frames = det["frames"]
    image_size = tuple(det["meta"]["image_size"]["sony"])
    corners_3d = cc.board_corners_3d()
    excluded = read_name_list(args.exclude)
    forced = read_name_list(args.include)

    print(f"\n{'=' * 60}\n  Sony intrinsics - dataset: {args.dataset}\n{'=' * 60}\n")

    by_index = {cc.frame_index(n): frames[n] for n in frames}
    candidates = sorted(
        cc.frame_index(n)
        for n, f in frames.items()
        if f["still"]["sony"] and n not in excluded
    )
    holds = cc.build_holds(candidates, by_index, "sony")
    reps = [
        cc.hold_representative(h, lambda i: len(by_index[i]["sony"]["ids"]))
        for h in holds
    ]
    names = [f"{i:05d}" for i in reps]
    for name in forced:
        if name in frames and frames[name]["usable"]["sony"] and name not in names:
            names.append(name)
    print(
        f"  Still Sony frames: {len(candidates)}  ->  holds (distinct poses): {len(holds)}"
    )
    if len(names) < MIN_CALIBRATION_VIEWS + 2:
        raise SystemExit(
            f"[ERROR] Only {len(names)} distinct still poses. At least {MIN_CALIBRATION_VIEWS + 2} are needed."
        )

    obj_all, img_all = zip(*(view_arrays(frames[n]["sony"], corners_3d) for n in names))

    # Initial pinhole estimate from up to 30 views spread over the image, used
    # to compute board poses for the diversity features.
    centres = np.array(
        [
            [im[:, 0].mean() / image_size[0], im[:, 1].mean() / image_size[1]]
            for im in img_all
        ]
    )
    seed = cc.diversity_order(centres)[:30]
    w, h = image_size
    K_guess = cc.camera_matrix(w, w, w / 2, h / 2)
    _, K0, _, _, _ = fit_model(
        MODELS[0][1],
        [obj_all[i] for i in seed],
        [img_all[i] for i in seed],
        image_size,
        K_guess,
    )

    features = []
    keep = []
    for k, (o, im) in enumerate(zip(obj_all, img_all)):
        pose = cc.solve_board_pose(o, im, K0, np.zeros(5))
        if pose is None:
            continue
        keep.append(k)
        features.append(cc.pose_features(pose[0], pose[1], im, image_size))
    names = [names[k] for k in keep]
    obj_all = [obj_all[k] for k in keep]
    img_all = [img_all[k] for k in keep]
    vectors = [cc.feature_vector(f) for f in features]
    start = int(
        np.argmin(
            [
                abs(f["u"] - 0.5) + abs(f["v"] - 0.5) + f["tilt_deg"] / 90
                for f in features
            ]
        )
    )
    order = cc.diversity_order(vectors, start)
    forced_idx = [names.index(n) for n in forced if n in names]
    order = forced_idx + [i for i in order if i not in forced_idx]
    calib_idx, test_idx = cc.split_calibration_test(order, args.max_views)
    for i in forced_idx:  # forced views stay in the calibration set
        if i in test_idx:
            test_idx.remove(i)
            calib_idx.append(i)
    if len(calib_idx) < MIN_CALIBRATION_VIEWS:
        raise SystemExit("[ERROR] Too few calibration views after the split.")

    obj_c = [obj_all[i] for i in calib_idx]
    img_c = [img_all[i] for i in calib_idx]
    obj_t = [obj_all[i] for i in test_idx]
    img_t = [img_all[i] for i in test_idx]
    print(f"  Calibration views: {len(calib_idx)}   Held-out views: {len(test_idx)}")

    calib_features = [features[i] for i in calib_idx]
    warnings = cc.span_warnings(calib_features)
    zs = [f["z_m"] for f in calib_features]
    tilts = [f["tilt_deg"] for f in calib_features]
    print(
        f"  Board distance {min(zs):.2f}-{max(zs):.2f} m, tilt up to {max(tilts):.1f} deg"
    )

    results = []
    for name, flags in MODELS:
        train_rms, K, D, std_int, _ = fit_model(flags, obj_c, img_c, image_size, K0)
        test_err = heldout_errors(K, D, obj_t, img_t)
        test_rms = cc.rms(test_err) if test_err.size else float("nan")
        results.append(
            {
                "model": name,
                "train_rms_px": train_rms,
                "test_rms_px": test_rms,
                "camera_matrix": K.tolist(),
                "distortion_coeffs": D.tolist(),
                "std_fx_fy_cx_cy": [float(v) for v in std_int[:4]],
            }
        )
        print(
            f"    {name:20s} train {train_rms:.3f} px | held-out {test_rms:.3f} px | "
            f"fx {K[0, 0]:.1f} (+-{std_int[0]:.1f}) fy {K[1, 1]:.1f} cx {K[0, 2]:.1f} (+-{std_int[2]:.1f}) "
            f"cy {K[1, 2]:.1f} (+-{std_int[3]:.1f})"
        )

    if test_idx:
        best = min(r["test_rms_px"] for r in results)
        limit = best * (1 + SELECTION_TOLERANCE_REL) + SELECTION_TOLERANCE_ABS_PX
        chosen = next(r for r in results if r["test_rms_px"] <= limit)
    else:
        warnings.append(
            "No held-out views: the model was chosen on calibration error and is not validated."
        )
        chosen = next(r for r in results if r["model"] == "k1_k2")

    K_phys = np.asarray(chosen["camera_matrix"])
    D_phys = np.asarray(chosen["distortion_coeffs"])
    runtime = {
        "model": f"{chosen['model']} camera matrix, distortion removed",
        "camera_matrix": chosen["camera_matrix"],
        "distortion_coeffs": [0.0] * len(chosen["distortion_coeffs"]),
    }
    disp, disp_stats = runtime_displacement(K_phys, D_phys, K_phys, image_size)

    if abs(K_phys[0, 0] / K_phys[1, 1] - 1) > 0.01:
        warnings.append(
            f"fx/fy = {K_phys[0, 0] / K_phys[1, 1]:.4f}; square pixels give 1.0."
        )
    if disp_stats["max_central_80pct_px"] > 2.0:
        warnings.append(
            f"Lens distortion moves pixels by up to {disp_stats['max_central_80pct_px']:.1f} px inside the central "
            f"80 % of the frame ({disp_stats['max_px']:.1f} px at the corners). The pinhole shader and camera are "
            f"exact at the image centre only; check the camera's distortion compensation setting, or add the "
            f"distortion model to the reprojection shader."
        )

    os.makedirs(p["plots"], exist_ok=True)
    plot_models(
        results, chosen["model"], os.path.join(p["plots"], "sony_intrinsics_models.png")
    )
    plot_coverage(
        img_c,
        img_t,
        image_size,
        os.path.join(p["plots"], "sony_intrinsics_coverage.png"),
    )
    plot_displacement(
        disp,
        disp_stats,
        image_size,
        os.path.join(p["plots"], "sony_runtime_displacement.png"),
    )

    cc.save_json(
        p["sony_intrinsics"],
        {
            "dataset": args.dataset,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "image_size": list(image_size),
            "n_holds": len(holds),
            "calibration_frames": [names[i] for i in calib_idx],
            "test_frames": [names[i] for i in test_idx],
            "physical": {
                **chosen,
                "role": "Selected on held-out error. Describes the lens.",
            },
            "runtime": {
                **runtime,
                "role": "Physical camera matrix without distortion, for the pinhole Unreal Engine camera and shader.",
            },
            "runtime_vs_physical_displacement": disp_stats,
            "models": results,
            "view_features": {names[i]: features[i] for i in calib_idx + test_idx},
            "warnings": warnings,
        },
    )

    print(
        f"\n  Physical model : {chosen['model']}  (held-out {chosen['test_rms_px']:.3f} px)"
    )
    print(f"  Runtime model  : {runtime['model']}")
    print(
        f"  Distortion omitted at runtime: central 80 % max {disp_stats['max_central_80pct_px']:.2f} px, "
        f"full frame max {disp_stats['max_px']:.2f} px"
    )
    for wmsg in warnings:
        print(f"  [WARN] {wmsg}")
    print(f"\n  Saved: {p['sony_intrinsics']}\n")


if __name__ == "__main__":
    sys.exit(main())

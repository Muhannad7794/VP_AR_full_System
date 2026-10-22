"""
validate_calibration.py
=======================
Step 5 of the spatial alignment pipeline.

Measures the calibration on the calibration pairs and, separately, on the
held-out pairs that were never used for fitting. Every metric is reported
against the board distance, because a calibration error from a mismatch
between translation and rotation is invisible at the distance it was fitted
at and grows away from it.

Metrics
-------
A. Corner transfer (calibration geometry only)
   Board pose from the ZED corners (PnP with the SDK intrinsics), corners
   moved into the Sony frame with R and T, projected with the Sony
   intrinsics, compared with the Sony detections.

B. End-to-end through the ZED depth (what the live system does)
   For every shared corner, the ZED SDK depth at that pixel is back-projected
   with the SDK intrinsics, moved with R and T and projected into the Sony
   image, the same chain as M_ZED_DepthReproject. Compared with the Sony
   detections. This includes depth noise, so its scatter is larger than in
   A; its mean offset per distance bin is the calibration bias that the live
   occlusion sees.

C. Depth scale
   Ratio of the ZED SDK depth to the depth predicted by the board pose at the
   same corners. A ratio far from 1.0 means the depth and the intrinsics
   disagree.

D. Bias against inverse distance
   The mean signed offset of every pair is fitted as a + b * (1 / z). The
   slope b (pixel metres) is the signature of a translation error, which
   produces an image shift proportional to 1 / z; b / f gives the implied
   translation error in metres. The intercept a is a distance-independent
   shift (rotation or principal point).

E. Sony floor
   For the same corners, the board pose is fitted to the Sony detections
   alone and the corners are reprojected. With six free pose parameters per
   pair this is the smallest error any extrinsic calibration could reach on
   these corners: it contains the Sony corner noise, the residual of the
   Sony lens model and any non-flatness of the board.

Grading and checks
------------------
Calibration grade   on the held-out excess over the floor,
                    sqrt(transfer^2 - floor^2) (see calib_common.py).
Scale check         median SDK depth / board-pose depth over all pairs. A
                    value away from 1.0 means the board square length or the
                    ZED depth scale is wrong. The square length that would
                    make the two agree is reported.
Runtime bias check  largest mean end-to-end offset of any distance bin: the
                    systematic shift that occlusion edges show in the
                    composite.

Step 4 versus step 5
--------------------
stereo_calibrate.py reports the stereo RMS: the residual of the fit itself,
in which a common board pose per pair is optimised together with R and T on
the calibration pairs. It measures how well the model fits its own data.
The corner transfer here keeps the board pose fixed to what the ZED sees and
predicts the Sony corners; it measures how well the calibration predicts.
The prediction error is always the larger of the two.

When the Sony model has lens distortion, every metric is computed twice with
the same R and T (with a pinhole Sony model the two are identical and only
one is printed):
    full_model  Sony camera matrix and lens distortion. Measures the
                calibration itself; the assessment is based on it.
    runtime     Sony camera matrix without distortion. Measures what the
                pinhole shader and Unreal camera reproduce; the difference to
                full_model is the lens distortion, largest at the frame edges
                (see error_across_sony_frame.png).

Outputs:
    data/json_output/<dataset>/validation_report.json
    data/plots/<dataset>/spatial/reprojection_error_analysis.png
    data/plots/<dataset>/spatial/error_vs_distance.png
    data/plots/<dataset>/spatial/bias_vs_inverse_distance.png
    data/plots/<dataset>/spatial/depth_scale_check.png
    data/plots/<dataset>/spatial/error_across_sony_frame.png
    data/plots/<dataset>/spatial/corner_coverage_sony_rgb.png
    data/plots/<dataset>/spatial/corner_coverage_zed_rgb.png

Usage:
    python3 spatial_alignment/validate_calibration.py --dataset <name>
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

DEPTH_WINDOW = 5  # pixels; median window for sampling the ZED depth at a corner
MIN_VALID_DEPTH_SAMPLES = 9
# Below this number of held-out pairs the grade is dominated by chance and is
# reported with a low-confidence note.
MIN_TEST_PAIRS_FOR_GRADE = 5
SETS = ("calibration", "test")
SET_COLORS = {"calibration": "#2E7D32", "test": "#EF6C00"}


def sample_depth_m(depth_mm, u, v):
    """Median of the valid depth values (metres) in a small window around a pixel."""
    h, w = depth_mm.shape
    r = DEPTH_WINDOW // 2
    x, y = int(round(u)), int(round(v))
    if x < r or y < r or x >= w - r or y >= h - r:
        return None
    window = depth_mm[y - r : y + r + 1, x - r : x + r + 1].astype(np.float64)
    valid = window[window > 0]
    if valid.size < MIN_VALID_DEPTH_SAMPLES:
        return None
    return float(np.median(valid)) / 1000.0


def evaluate_pair(frame, depth_mm, corners_3d, K_z, K_s, D_s, R, T):
    """Metrics A, B and C for one pair. Returns None when the ZED pose fails."""
    ids_z, pts_z_all = cc.view_points(frame["zed"])
    pose = cc.solve_board_pose(corners_3d[ids_z], pts_z_all, K_z, np.zeros(5))
    if pose is None:
        return None
    Rb, _ = cv2.Rodrigues(pose[0])
    tb = pose[1].reshape(3)

    ids, pts_s, pts_z = cc.common_points(frame["sony"], frame["zed"])
    board_zed = (Rb @ corners_3d[ids].T).T + tb  # corners in the ZED frame (metres)
    z_board = float(np.median(board_zed[:, 2]))

    # A. Corner transfer.
    in_sony = (R @ board_zed.T).T + T
    proj, _ = cv2.projectPoints(in_sony, np.zeros(3), np.zeros(3), K_s, D_s)
    diff_a = proj.reshape(-1, 2) - pts_s
    centre = np.array([K_s[0, 2], K_s[1, 2]])
    result = {
        "n_corners": int(len(ids)),
        "board_distance_m": z_board,
        "transfer_errors_px": np.linalg.norm(diff_a, axis=1).tolist(),
        "transfer_mean_offset_px": diff_a.mean(axis=0).tolist(),
        # Distance of each Sony corner from the principal point, for the radial analysis.
        "sony_radius_px": np.linalg.norm(pts_s - centre, axis=1).tolist(),
    }

    # E. Sony floor: the board pose fitted to the Sony corners alone.
    pose_s = cc.solve_board_pose(corners_3d[ids], pts_s, K_s, D_s)
    if pose_s is not None:
        proj_s, _ = cv2.projectPoints(corners_3d[ids], pose_s[0], pose_s[1], K_s, D_s)
        result["floor_errors_px"] = np.linalg.norm(
            proj_s.reshape(-1, 2) - pts_s, axis=1
        ).tolist()

    # B and C. End-to-end through the ZED depth.
    if depth_mm is not None:
        points, observed, ratios = [], [], []
        fx, fy, cx, cy = K_z[0, 0], K_z[1, 1], K_z[0, 2], K_z[1, 2]
        for (u, v), pz_pred, ps in zip(pts_z, board_zed, pts_s):
            z = sample_depth_m(depth_mm, u, v)
            if z is None:
                continue
            X = np.array([(u - cx) * z / fx, (v - cy) * z / fy, z])
            Xs = R @ X + T
            if Xs[2] <= 0:
                continue
            points.append(Xs)
            observed.append(ps)
            ratios.append(z / pz_pred[2])
        if points:
            proj, _ = cv2.projectPoints(
                np.array(points), np.zeros(3), np.zeros(3), K_s, D_s
            )
            diffs = proj.reshape(-1, 2) - np.array(observed)
            result["e2e_errors_px"] = np.linalg.norm(diffs, axis=1).tolist()
            result["e2e_mean_offset_px"] = diffs.mean(axis=0).tolist()
            result["e2e_sony_radius_px"] = np.linalg.norm(
                np.array(observed) - centre, axis=1
            ).tolist()
            result["depth_ratio_median"] = float(np.median(ratios))
            result["e2e_n_corners"] = int(len(diffs))
    return result


def aggregate(pairs, key):
    errors = [e for r in pairs.values() if key in r for e in r[key]]
    return {
        "rms_px": cc.rms(errors),
        "p95_px": float(np.percentile(errors, 95)) if errors else float("nan"),
        "n_points": len(errors),
    }


def by_distance(pairs, err_key, offset_key):
    bins = {}
    for r in pairs.values():
        if err_key not in r:
            continue
        label = cc.distance_bin_label(r["board_distance_m"])
        b = bins.setdefault(label, {"z": [], "errors": [], "offsets": [], "ratios": []})
        b["z"].append(r["board_distance_m"])
        b["errors"].extend(r[err_key])
        b["offsets"].append(r[offset_key])
        if "depth_ratio_median" in r:
            b["ratios"].append(r["depth_ratio_median"])
    out = {}
    for label, b in sorted(bins.items(), key=lambda kv: np.mean(kv[1]["z"])):
        mean_offset = np.mean(b["offsets"], axis=0)
        out[label] = {
            "n_pairs": len(b["offsets"]),
            "rms_px": cc.rms(b["errors"]),
            "mean_offset_px": mean_offset.tolist(),
            "mean_offset_magnitude_px": float(np.linalg.norm(mean_offset)),
            "depth_ratio_median": (
                float(np.median(b["ratios"])) if b["ratios"] else None
            ),
        }
    return out


def inverse_distance_fit(pairs, offset_key, f_sony):
    """Fits the mean signed offset of each pair as a + b / z for x and y."""
    rows = [
        (1.0 / r["board_distance_m"], r[offset_key])
        for r in pairs.values()
        if offset_key in r
    ]
    if len(rows) < 4:
        return {"available": False, "reason": "fewer than 4 pairs"}
    inv_z = np.array([r[0] for r in rows])
    off = np.array([r[1] for r in rows])
    if inv_z.max() - inv_z.min() < 0.1:
        return {
            "available": False,
            "reason": "distance range too small to separate slope from offset",
        }
    A = np.stack([np.ones_like(inv_z), inv_z], axis=1)
    fit = {"available": True}
    for axis, name, f in ((0, "x", f_sony[0]), (1, "y", f_sony[1])):
        coef, *_ = np.linalg.lstsq(A, off[:, axis], rcond=None)
        fit[name] = {
            "offset_px": float(coef[0]),
            "slope_px_m": float(coef[1]),
            "implied_translation_error_mm": float(coef[1] / f * 1000),
        }
    return fit


def scale_check(pairs):
    """Median ratio of the ZED SDK depth to the board-pose depth over all
    pairs, with the board square length that would make the two agree."""
    ratios = np.array(
        [r["depth_ratio_median"] for r in pairs.values() if "depth_ratio_median" in r]
    )
    if ratios.size == 0:
        return {"available": False}
    median = float(np.median(ratios))
    q25, q75 = (float(v) for v in np.percentile(ratios, [25, 75]))
    ok = abs(median - 1.0) <= cc.SCALE_TOLERANCE
    implied_mm = cc.SQUARE_LENGTH * 1000.0 * median
    if ok:
        verdict = "Consistent"
        note = f"ZED SDK depth and board geometry agree within {cc.SCALE_TOLERANCE * 100:.1f} %."
    else:
        verdict = "Inconsistent"
        note = (
            f"ZED SDK depth is {(median - 1.0) * 100:+.2f} % against the board geometry. Either the "
            f"board square is {implied_mm:.2f} mm instead of {cc.SQUARE_LENGTH * 1000:.2f} mm (measure "
            f"the printed board across all {cc.BOARD_COLS} squares of a row with a steel ruler and set "
            f"--square-mm), or the ZED depth scale is off. The end-to-end path shows this as a bias "
            f"that grows towards the camera."
        )
    return {
        "available": True,
        "n_pairs": int(ratios.size),
        "depth_ratio_median": median,
        "depth_ratio_iqr": [q25, q75],
        "board_square_mm": cc.SQUARE_LENGTH * 1000.0,
        "square_mm_implied_by_zed_depth": implied_mm,
        "verdict": verdict,
        "note": note,
    }


def runtime_bias_check(bins):
    """Largest systematic end-to-end offset over the distance bins."""
    rows = [(label, b) for label, b in bins.items() if b["n_pairs"] >= 3]
    if not rows:
        return {"available": False}
    label, worst = max(rows, key=lambda kv: kv[1]["mean_offset_magnitude_px"])
    magnitude = worst["mean_offset_magnitude_px"]
    ok = magnitude <= cc.RUNTIME_BIAS_WARN_PX
    return {
        "available": True,
        "worst_bin": label,
        "worst_bias_px": magnitude,
        "limit_px": cc.RUNTIME_BIAS_WARN_PX,
        "verdict": "No visible systematic shift" if ok else "Systematic shift",
    }


def solutions_identical(solutions):
    (K_a, D_a), (K_b, D_b) = solutions["full_model"], solutions["runtime"]
    return np.allclose(K_a, K_b) and np.allclose(D_a, D_b)


def plot_error_analysis(results, out):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for s in SETS:
        errs = [e for r in results[s].values() for e in r["transfer_errors_px"]]
        if errs:
            axes[0].hist(
                errs,
                bins=40,
                alpha=0.6,
                color=SET_COLORS[s],
                label=f"{s} (RMS {cc.rms(errs):.2f} px)",
            )
    axes[0].set(
        xlabel="Corner transfer error (px)",
        ylabel="Corners",
        title="Error distribution",
    )
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    x = 0
    floor_label = "Sony floor (pose fitted to Sony corners)"
    for s in SETS:
        names = sorted(results[s])
        vals = [cc.rms(results[s][n]["transfer_errors_px"]) for n in names]
        axes[1].bar(range(x, x + len(vals)), vals, color=SET_COLORS[s], label=s)
        floors = [
            (x + i, cc.rms(results[s][n]["floor_errors_px"]))
            for i, n in enumerate(names)
            if "floor_errors_px" in results[s][n]
        ]
        if floors:
            axes[1].scatter(
                *zip(*floors), color="black", s=6, zorder=3, label=floor_label
            )
            floor_label = None
        x += len(vals) + 2
    axes[1].set(
        xlabel="Pair",
        ylabel="RMS (px)",
        title="Per-pair corner transfer error and Sony floor",
    )
    axes[1].legend()
    axes[1].grid(alpha=0.3, axis="y")
    fig.suptitle(
        "Reprojection error analysis - ZED to Sony (full lens model)", fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    plt.close()


def plot_error_vs_distance(results, out):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=False)
    for ax, key, title in (
        (axes[0], "transfer_errors_px", "A. Corner transfer"),
        (axes[1], "e2e_errors_px", "B. End-to-end through ZED depth"),
    ):
        for s in SETS:
            pts = [
                (r["board_distance_m"], cc.rms(r[key]))
                for r in results[s].values()
                if key in r
            ]
            if pts:
                ax.scatter(*zip(*pts), color=SET_COLORS[s], label=s)
        ax.set(
            xlabel="Board distance from ZED (m)",
            ylabel="Per-pair RMS (px)",
            title=title,
        )
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Error against distance", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    plt.close()


def plot_bias(results, fits, out):
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    for row, (key, label) in enumerate(
        (
            ("transfer_mean_offset_px", "A. Corner transfer"),
            ("e2e_mean_offset_px", "B. End-to-end"),
        )
    ):
        for col, axis in enumerate(("x", "y")):
            ax = axes[row, col]
            allz = []
            for s in SETS:
                pts = [
                    (1 / r["board_distance_m"], r[key][col])
                    for r in results[s].values()
                    if key in r
                ]
                if pts:
                    ax.scatter(*zip(*pts), color=SET_COLORS[s], label=s)
                    allz += [q[0] for q in pts]
            fit = fits.get(key, {})
            if fit.get("available") and allz:
                xs = np.linspace(min(allz), max(allz), 20)
                a, b = fit[axis]["offset_px"], fit[axis]["slope_px_m"]
                ax.plot(
                    xs,
                    a + b * xs,
                    color="black",
                    linewidth=1,
                    label=f"fit: {a:+.2f} px + {b:+.2f} px*m / z  (dT {fit[axis]['implied_translation_error_mm']:+.1f} mm)",
                )
            ax.axhline(0, color="#9E9E9E", linewidth=0.8)
            ax.set(
                xlabel="1 / distance (1/m)",
                ylabel=f"Mean offset {axis} (px)",
                title=f"{label}, {axis}",
            )
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
    fig.suptitle(
        "Signed bias against inverse distance (a slope means a translation error)",
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    plt.close()


def plot_depth_scale(results, out):
    fig, ax = plt.subplots(figsize=(9, 5))
    for s in SETS:
        pts = [
            (r["board_distance_m"], r["depth_ratio_median"])
            for r in results[s].values()
            if "depth_ratio_median" in r
        ]
        if pts:
            ax.scatter(*zip(*pts), color=SET_COLORS[s], label=s)
    ax.axhline(1.0, color="black", linewidth=1)
    ax.set(
        xlabel="Board distance from ZED (m)",
        ylabel="SDK depth / pose depth",
        title="Depth scale check (1.0 = SDK depth agrees with the SDK intrinsics)",
    )
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    plt.close()


def plot_coverage(frames, selection, cam, image_size, out):
    fig, ax = plt.subplots(figsize=(10, 6))
    for s in SETS:
        pts = [np.asarray(frames[n][cam]["corners"]) for n in selection[s]]
        if pts:
            allp = np.concatenate(pts)
            ax.scatter(allp[:, 0], allp[:, 1], s=2, color=SET_COLORS[s], label=s)
    ax.set_xlim(0, image_size[0])
    ax.set_ylim(image_size[1], 0)
    ax.set_aspect("equal")
    ax.set_title(f"Corner coverage - {cam.upper()} (stereo pairs)")
    ax.legend(markerscale=6, loc="upper right")
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    plt.close()


def plot_radial(all_results, out):
    """Per-corner error against distance from the Sony principal point, for
    the full lens model and the pinhole runtime model. The gap between the two
    curves is the lens distortion the runtime omits."""
    fig, ax = plt.subplots(figsize=(10, 5))
    styles = {
        "full_model": ("#1565C0", "Full lens model"),
        "runtime": ("#C62828", "Pinhole runtime"),
    }
    for sol, (color, label) in styles.items():
        if sol == "runtime" and all_results["runtime"] is all_results["full_model"]:
            continue  # pinhole Sony model: the runtime curve equals the full curve
        r, e = [], []
        for s in SETS:
            for res in all_results[sol][s].values():
                r += res["sony_radius_px"]
                e += res["transfer_errors_px"]
        if not r:
            continue
        r, e = np.array(r), np.array(e)
        edges = np.linspace(0, r.max() + 1, 12)
        centres, vals = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (r >= lo) & (r < hi)
            if m.sum() >= 5:
                centres.append((lo + hi) / 2)
                vals.append(cc.rms(e[m]))
        ax.scatter(r, e, s=2, alpha=0.25, color=color)
        ax.plot(centres, vals, color=color, linewidth=2, label=f"{label} (binned RMS)")
    ax.set(
        xlabel="Distance from the Sony principal point (px)",
        ylabel="Corner transfer error (px)",
        title="Error across the Sony frame",
    )
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Validate the spatial calibration.")
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    p = cc.paths(args.dataset)
    det = cc.load_json(p["detections"], "inspect_detections.py")
    frames = det["frames"]
    selection = cc.load_json(p["pair_selection"], "pick_frames.py")
    cal = cc.load_json(p["calibration"], "stereo_calibrate.py")
    cc.check_board_matches(cal.get("board"), "stereo_calibrate.py")
    K_z, _, zed_size = cc.load_zed_intrinsics(args.dataset)
    corners_3d = cc.board_corners_3d()
    has_depth = os.path.isdir(p["synced_depth"])

    sony_single = None
    if os.path.isfile(p["sony_intrinsics"]):
        sony_single = cc.load_json(p["sony_intrinsics"])["physical"].get("test_rms_px")

    R = np.asarray(cal["extrinsics"]["rotation_matrix_3x3"])
    T = np.asarray(cal["extrinsics"]["translation_metres"])
    # Same geometry evaluated two ways: with the Sony lens distortion (calibration
    # quality) and without it (what the pinhole shader and camera reproduce).
    solutions = {
        "full_model": (
            np.asarray(cal["sony_camera"]["camera_matrix"]),
            np.asarray(cal["sony_camera"]["distortion_coeffs"]),
        ),
        "runtime": (
            np.asarray(cal["sony_camera_runtime"]["camera_matrix"]),
            np.asarray(cal["sony_camera_runtime"]["distortion_coeffs"]),
        ),
    }
    identical = solutions_identical(solutions)

    print(f"\n{'=' * 64}\n  Validation - dataset: {args.dataset}\n{'=' * 64}\n")
    print(f"  Board square length: {cc.SQUARE_LENGTH * 1000:.3f} mm")
    if not has_depth:
        print(
            f"  [WARN] No depth folder at {p['synced_depth']}; end-to-end metrics are skipped."
        )

    depth_cache = {}

    def depth_for(name):
        if not has_depth:
            return None
        if name not in depth_cache:
            depth_cache[name] = cv2.imread(
                os.path.join(p["synced_depth"], name + ".png"), cv2.IMREAD_UNCHANGED
            )
        return depth_cache[name]

    report = {
        "dataset": args.dataset,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "board": cc.board_definition(),
        "solutions": {},
    }
    all_results = {}
    for sol_name, (K_s, D_s) in solutions.items():
        if sol_name == "runtime" and identical:
            # A pinhole Sony model: the runtime evaluation equals the full one.
            all_results[sol_name] = all_results["full_model"]
            report["solutions"][sol_name] = {
                "identical_to": "full_model",
                "reason": "The Sony model has no lens distortion.",
            }
            continue
        results = {s: {} for s in SETS}
        for s in SETS:
            for name in selection[s]:
                r = evaluate_pair(
                    frames[name], depth_for(name), corners_3d, K_z, K_s, D_s, R, T
                )
                if r is not None:
                    results[s][name] = r
        all_results[sol_name] = results
        combined = {**results["calibration"], **results["test"]}
        f_sony = (K_s[0, 0], K_s[1, 1])
        entry = {
            "transfer": {s: aggregate(results[s], "transfer_errors_px") for s in SETS},
            "sony_floor": {s: aggregate(results[s], "floor_errors_px") for s in SETS},
            "transfer_by_distance_test": by_distance(
                results["test"], "transfer_errors_px", "transfer_mean_offset_px"
            ),
            "transfer_inverse_distance_fit": inverse_distance_fit(
                combined, "transfer_mean_offset_px", f_sony
            ),
        }
        entry["excess_over_floor_px"] = {
            s: cc.excess_over_floor(
                entry["transfer"][s]["rms_px"], entry["sony_floor"][s]["rms_px"]
            )
            for s in SETS
        }
        if has_depth:
            entry["end_to_end"] = {
                s: aggregate(results[s], "e2e_errors_px") for s in SETS
            }
            entry["end_to_end_by_distance_all"] = by_distance(
                combined, "e2e_errors_px", "e2e_mean_offset_px"
            )
            entry["end_to_end_inverse_distance_fit"] = inverse_distance_fit(
                combined, "e2e_mean_offset_px", f_sony
            )
            entry["scale_check"] = scale_check(combined)
            entry["runtime_bias_check"] = runtime_bias_check(
                entry["end_to_end_by_distance_all"]
            )
        entry["per_pair"] = results
        report["solutions"][sol_name] = entry

    full = report["solutions"]["full_model"]
    excess_test = full["excess_over_floor_px"]["test"]
    report["assessment"] = cc.grade(excess_test)
    report["grading_rule"] = (
        f"Held-out excess over the Sony floor, sqrt(transfer^2 - floor^2), full Sony lens model: "
        f"< {cc.GRADE_EXCELLENT_PX} px Excellent, < {cc.GRADE_GOOD_PX} px Good, "
        f"< {cc.GRADE_ACCEPTABLE_PX} px Acceptable, otherwise Poor."
    )
    report["n_pairs"] = {s: len(all_results["full_model"][s]) for s in SETS}
    n_test = report["n_pairs"]["test"]
    if n_test < MIN_TEST_PAIRS_FOR_GRADE:
        report["assessment"] += f" (low confidence: only {n_test} held-out pairs)"
    report["summary"] = {
        "sony_single_camera_test_rms_px": sony_single,
        "stereo_fit_rms_px": cal["extrinsics"]["stereo_rms_px"],
        "sony_floor_test_rms_px": full["sony_floor"]["test"]["rms_px"],
        "transfer_test_rms_px": full["transfer"]["test"]["rms_px"],
        "excess_test_px": excess_test,
        "end_to_end_test_rms_px": (
            full["end_to_end"]["test"]["rms_px"] if "end_to_end" in full else None
        ),
        "scale_check": full.get("scale_check", {}).get("verdict"),
        "runtime_bias_check": full.get("runtime_bias_check", {}).get("verdict"),
    }
    report["scale_check"] = full.get("scale_check")
    cc.save_json(p["validation"], report)

    os.makedirs(p["plots"], exist_ok=True)
    res_full = all_results["full_model"]
    plot_error_analysis(
        res_full, os.path.join(p["plots"], "reprojection_error_analysis.png")
    )
    plot_error_vs_distance(res_full, os.path.join(p["plots"], "error_vs_distance.png"))
    fits = {"transfer_mean_offset_px": full["transfer_inverse_distance_fit"]}
    if "end_to_end_inverse_distance_fit" in full:
        fits["e2e_mean_offset_px"] = full["end_to_end_inverse_distance_fit"]
    plot_bias(res_full, fits, os.path.join(p["plots"], "bias_vs_inverse_distance.png"))
    plot_radial(all_results, os.path.join(p["plots"], "error_across_sony_frame.png"))
    if has_depth:
        plot_depth_scale(res_full, os.path.join(p["plots"], "depth_scale_check.png"))
    sizes = det["meta"]["image_size"]
    plot_coverage(
        frames,
        selection,
        "sony",
        sizes["sony"],
        os.path.join(p["plots"], "corner_coverage_sony_rgb.png"),
    )
    plot_coverage(
        frames,
        selection,
        "zed",
        sizes["zed"],
        os.path.join(p["plots"], "corner_coverage_zed_rgb.png"),
    )

    # Console summary: every error in Sony pixels, from the easiest test to the
    # one the live system faces.
    def fmt(v):
        return f"{v:6.3f} px" if v is not None and np.isfinite(v) else "     -   "

    tr, fl = full["transfer"], full["sony_floor"]
    print(
        "\n  Errors in Sony pixels (calibration pairs | held-out pairs):\n"
        f"    Sony alone, step 2 (Sony intrinsics, held-out views) : {'':10s}   {fmt(sony_single)}\n"
        f"    Stereo fit, step 4 (fit residual, poses optimised)   : {fmt(cal['extrinsics']['stereo_rms_px'])}\n"
        f"    E. Sony floor (best any calibration could reach)     : {fmt(fl['calibration']['rms_px'])} | {fmt(fl['test']['rms_px'])}\n"
        f"    A. Corner transfer (prediction, ZED pose -> Sony)    : {fmt(tr['calibration']['rms_px'])} | {fmt(tr['test']['rms_px'])}\n"
        f"       excess over the floor                             : {fmt(full['excess_over_floor_px']['calibration'])} | {fmt(excess_test)}"
    )
    if "end_to_end" in full:
        e2e = full["end_to_end"]
        print(
            f"    B. End-to-end (ZED SDK depth -> Sony, live path)     : {fmt(e2e['calibration']['rms_px'])} | {fmt(e2e['test']['rms_px'])}"
        )
    if identical:
        print(
            "    (Sony model is pinhole: the runtime shader reproduces these values exactly.)"
        )
    else:
        rt = report["solutions"]["runtime"]
        print(
            f"    Pinhole runtime, corner transfer                     : {fmt(rt['transfer']['calibration']['rms_px'])} | "
            f"{fmt(rt['transfer']['test']['rms_px'])}"
        )
        if "end_to_end" in rt:
            print(
                f"    Pinhole runtime, end-to-end                          : {fmt(rt['end_to_end']['calibration']['rms_px'])} | "
                f"{fmt(rt['end_to_end']['test']['rms_px'])}"
            )

    if "end_to_end_by_distance_all" in full:
        print("\n  End-to-end bias by distance (all pairs):")
        print(
            f"    {'distance':14s} {'pairs':>5s} {'RMS px':>8s} {'bias x':>8s} {'bias y':>8s} {'depth ratio':>12s}"
        )
        for label, b in full["end_to_end_by_distance_all"].items():
            ratio = f"{b['depth_ratio_median']:.4f}" if b["depth_ratio_median"] else "-"
            print(
                f"    {label:14s} {b['n_pairs']:5d} {b['rms_px']:8.2f} {b['mean_offset_px'][0]:+8.2f} "
                f"{b['mean_offset_px'][1]:+8.2f} {ratio:>12s}"
            )
    for label, key in (
        ("Corner transfer", "transfer_inverse_distance_fit"),
        ("End-to-end", "end_to_end_inverse_distance_fit"),
    ):
        fit = full.get(key)
        if not fit:
            continue
        if fit["available"]:
            print(
                f"\n  {label} bias vs 1/z: x {fit['x']['offset_px']:+.2f} px + {fit['x']['slope_px_m']:+.2f} px*m/z "
                f"(dT {fit['x']['implied_translation_error_mm']:+.1f} mm) | y {fit['y']['offset_px']:+.2f} px + "
                f"{fit['y']['slope_px_m']:+.2f} px*m/z (dT {fit['y']['implied_translation_error_mm']:+.1f} mm)"
            )
        else:
            print(f"\n  {label} bias vs 1/z: not available ({fit['reason']})")

    sc = full.get("scale_check")
    if sc and sc["available"]:
        print(
            f"\n  Scale check       : {sc['verdict']} - depth ratio {sc['depth_ratio_median']:.4f} "
            f"(IQR {sc['depth_ratio_iqr'][0]:.4f}-{sc['depth_ratio_iqr'][1]:.4f}, {sc['n_pairs']} pairs); "
            f"square implied by the ZED depth {sc['square_mm_implied_by_zed_depth']:.2f} mm"
        )
        if sc["verdict"] != "Consistent":
            print(f"                      {sc['note']}")
    rb = full.get("runtime_bias_check")
    if rb and rb["available"]:
        print(
            f"  Runtime bias check: {rb['verdict']} - largest bin bias {rb['worst_bias_px']:.2f} px "
            f"({rb['worst_bin']}, limit {rb['limit_px']:.1f} px)"
        )

    print(
        f"\n{'=' * 64}\n  VALIDATION COMPLETE - Calibration assessment: {report['assessment']}"
    )
    print(f"  ({report['grading_rule']})")
    checks = [
        f"Scale check: {sc['verdict']}" if sc and sc["available"] else None,
        f"Runtime bias check: {rb['verdict']}" if rb and rb["available"] else None,
    ]
    checks = [c for c in checks if c]
    if checks:
        print("  " + " | ".join(checks))
    if (sc and sc.get("verdict") == "Inconsistent") or (
        rb and rb.get("verdict") == "Systematic shift"
    ):
        print(
            "  The calibration geometry is graded above; the live composite will still show the\n"
            "  systematic shift reported by the checks until its cause is fixed."
        )
    print("=" * 64)
    print(f"\n  Saved: {p['validation']}\n")


if __name__ == "__main__":
    sys.exit(main())

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

Every metric is computed twice with the same R and T:
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
    for s in SETS:
        names = sorted(results[s])
        vals = [cc.rms(results[s][n]["transfer_errors_px"]) for n in names]
        axes[1].bar(range(x, x + len(vals)), vals, color=SET_COLORS[s], label=s)
        x += len(vals) + 2
    for t, c in ((cc.GRADE_GOOD_PX, "#2E7D32"), (cc.GRADE_ACCEPTABLE_PX, "#C62828")):
        axes[1].axhline(t, color=c, linestyle="--", linewidth=1)
    axes[1].set(
        xlabel="Pair", ylabel="RMS (px)", title="Per-pair corner transfer error"
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
    K_z, _, zed_size = cc.load_zed_intrinsics(args.dataset)
    corners_3d = cc.board_corners_3d()
    has_depth = os.path.isdir(p["synced_depth"])

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

    print(f"\n{'=' * 60}\n  Validation - dataset: {args.dataset}\n{'=' * 60}\n")
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
        "solutions": {},
    }
    all_results = {}
    for sol_name, (K_s, D_s) in solutions.items():
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
            "transfer_by_distance_test": by_distance(
                results["test"], "transfer_errors_px", "transfer_mean_offset_px"
            ),
            "transfer_inverse_distance_fit": inverse_distance_fit(
                combined, "transfer_mean_offset_px", f_sony
            ),
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
        entry["per_pair"] = results
        report["solutions"][sol_name] = entry

    full = report["solutions"]["full_model"]
    report["assessment"] = cc.grade(full["transfer"]["test"]["rms_px"])
    report["grading_rule"] = (
        f"Held-out corner transfer RMS with the full Sony lens model: < {cc.GRADE_EXCELLENT_PX} Excellent, "
        f"< {cc.GRADE_GOOD_PX} Good, < {cc.GRADE_ACCEPTABLE_PX} Acceptable, otherwise Poor."
    )
    report["n_pairs"] = {s: len(all_results["full_model"][s]) for s in SETS}
    n_test = report["n_pairs"]["test"]
    if n_test < MIN_TEST_PAIRS_FOR_GRADE:
        report["assessment"] += f" (low confidence: only {n_test} held-out pairs)"
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

    # Console summary.
    labels = {"full_model": "full lens model", "runtime": "pinhole runtime"}
    for sol_name in ("full_model", "runtime"):
        e = report["solutions"][sol_name]
        line = (
            f"  {labels[sol_name]:16s} corner transfer RMS: calibration {e['transfer']['calibration']['rms_px']:.3f} px | "
            f"held-out {e['transfer']['test']['rms_px']:.3f} px"
        )
        if "end_to_end" in e:
            line += (
                f"\n  {'':16s} end-to-end RMS     : calibration {e['end_to_end']['calibration']['rms_px']:.3f} px | "
                f"held-out {e['end_to_end']['test']['rms_px']:.3f} px"
            )
        print(line)
    if "end_to_end_by_distance_all" in full:
        print("\n  End-to-end bias by distance, full lens model (all pairs):")
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

    print(f"\n{'=' * 60}\n  VALIDATION COMPLETE - Assessment: {report['assessment']}")
    print(f"  ({report['grading_rule']})\n{'=' * 60}")
    print(f"\n  Saved: {p['validation']}\n")


if __name__ == "__main__":
    sys.exit(main())

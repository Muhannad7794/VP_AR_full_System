"""
pick_frames.py
==============
Step 3 of the spatial alignment pipeline.

Selects the synced frame pairs used to estimate the extrinsics (R, T) and the
held-out pairs used to validate them.

Method
------
1. Candidate pairs: the board is usable and still in BOTH cameras, and the
   corners seen by both cameras are numerous and spread over the board.
2. Candidates are grouped into holds (one physical board pose each). One
   representative per hold is used, taken from the middle of the hold, where
   a small temporal offset between the recordings has no effect.
3. Each representative is described by the board position in the Sony image,
   the board distance and the board tilt (from the ZED pose, computed with
   the SDK intrinsics). A diversity ordering over these features is split
   into a calibration set and a held-out test set.

Distance and tilt variety are what separate translation from rotation in the
extrinsics, so the step warns when the calibration set lacks them.

Outputs:
    data/json_output/<dataset>/pair_selection.json
    data/picked_for_alignment/<dataset>/{sony_rgb,zed_rgb}/   calibration pairs
    data/picked_for_validation/<dataset>/{sony_rgb,zed_rgb}/  held-out pairs
    data/plots/<dataset>/spatial/picked_for_alignment/        annotated calibration pairs
    data/plots/<dataset>/spatial/picked_for_validation/       annotated held-out pairs
    data/plots/<dataset>/spatial/pair_selection_features.png

Usage:
    python3 spatial_alignment/pick_frames.py --dataset <name>
        [--max-pairs 40] [--exclude frames.txt] [--include frames.txt]
"""

import argparse
import glob
import os
import shutil
import sys
import time

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import calib_common as cc


def read_name_list(path):
    if not path:
        return set()
    with open(path) as f:
        return {
            line.strip().replace(".png", "")
            for line in f
            if line.strip() and not line.startswith("#")
        }


def annotate(src, dst, det, common_ids, label):
    img = cv2.imread(src)
    if img is None:
        return
    ids, pts = cc.view_points(det)
    shared = np.isin(ids, common_ids)
    for (x, y), s in zip(pts, shared):
        cv2.circle(
            img,
            (int(round(x)), int(round(y))),
            6,
            (0, 200, 0) if s else (0, 140, 255),
            -1,
        )
    cv2.rectangle(img, (0, 0), (img.shape[1], 56), (0, 0, 0), -1)
    cv2.putText(img, label, (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    h, w = img.shape[:2]
    if w > 1280:
        img = cv2.resize(img, (1280, int(h * 1280 / w)))
    cv2.imwrite(dst, img)


def reset_dir(path):
    os.makedirs(path, exist_ok=True)
    for f in glob.glob(os.path.join(path, "*")):
        if os.path.isfile(f):
            os.remove(f)


def plot_features(features, calib, test, out):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for names, color, label in (
        (calib, "#2E7D32", "Calibration"),
        (test, "#EF6C00", "Held-out"),
    ):
        f = [features[n] for n in names]
        if not f:
            continue
        axes[0].scatter(
            [x["u"] for x in f], [x["v"] for x in f], color=color, label=label
        )
        axes[1].scatter(
            [x["z_m"] for x in f], [x["tilt_deg"] for x in f], color=color, label=label
        )
        axes[2].scatter(
            [x["nx"] for x in f], [x["ny"] for x in f], color=color, label=label
        )
    axes[0].set(
        xlim=(0, 1),
        ylim=(1, 0),
        xlabel="Board centre u (Sony, fraction)",
        ylabel="v",
        title="Image position",
    )
    axes[1].set(
        xlabel="Board distance from ZED (m)",
        ylabel="Tilt (deg)",
        title="Distance and tilt",
    )
    axes[2].set(
        xlim=(-1, 1),
        ylim=(1, -1),
        xlabel="Normal x",
        ylabel="Normal y",
        title="Tilt direction",
    )
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Selected stereo pairs", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Select calibration and held-out stereo pairs."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--max-pairs", type=int, default=40, help="Maximum calibration pairs."
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
    sizes = det["meta"]["image_size"]
    K_zed, D_zed, zed_size = cc.load_zed_intrinsics(args.dataset)
    if tuple(sizes["zed"]) != tuple(zed_size):
        raise SystemExit(
            f"[ERROR] ZED frames are {sizes['zed']} but the ZED intrinsics are for {zed_size}."
        )
    corners_3d = cc.board_corners_3d()
    excluded = read_name_list(args.exclude)
    forced = read_name_list(args.include)

    print(f"\n{'=' * 60}\n  Pair selection - dataset: {args.dataset}\n{'=' * 60}\n")

    def pair_ok(name):
        f = frames[name]
        if name in excluded or not (f["still"]["sony"] and f["still"]["zed"]):
            return False
        ids, _, _ = cc.common_points(f["sony"], f["zed"])
        return cc.view_is_usable(ids, corners_3d)

    by_index = {cc.frame_index(n): frames[n] for n in frames}
    candidates = sorted(cc.frame_index(n) for n in frames if pair_ok(n))
    holds = cc.build_holds(candidates, by_index, "sony")

    def common_count(i):
        return len(cc.common_points(by_index[i]["sony"], by_index[i]["zed"])[0])

    reps = [f"{cc.hold_representative(h, common_count):05d}" for h in holds]
    for name in forced:
        if name in frames and name not in reps:
            f = frames[name]
            if f["usable"]["sony"] and f["usable"]["zed"]:
                reps.append(name)
            else:
                print(
                    f"  [WARN] Forced frame {name} is not usable in both cameras and is ignored."
                )
    print(f"  Still pairs: {len(candidates)}  ->  holds (distinct poses): {len(holds)}")
    if len(reps) < 6:
        n_sony = sum(1 for f in frames.values() if f["still"]["sony"])
        n_zed = sum(1 for f in frames.values() if f["still"]["zed"])
        raise SystemExit(
            f"[ERROR] Only {len(reps)} distinct still poses seen by both cameras; at least 6 are needed.\n"
            f"        Still frames: Sony {n_sony}, ZED {n_zed}, both {len(candidates)}.\n"
            "        If one camera has far fewer still frames than the other, its stillness threshold is too strict\n"
            "        for its corner noise: compare the motion percentiles printed by inspect_detections.py and re-run\n"
            "        that step with --still-sony / --still-zed. If both counts are low, the recording contains too few\n"
            "        steady holds; record more poses and keep the board motionless for about one second each."
        )

    features = {}
    for name in reps:
        f = frames[name]
        ids_z, pts_z = cc.view_points(f["zed"])
        pose = cc.solve_board_pose(corners_3d[ids_z], pts_z, K_zed, D_zed)
        if pose is None:
            continue
        feat = cc.pose_features(pose[0], pose[1], pts_z, zed_size)
        _, pts_s, _ = cc.common_points(f["sony"], f["zed"])
        feat["u"] = float(pts_s[:, 0].mean() / sizes["sony"][0])
        feat["v"] = float(pts_s[:, 1].mean() / sizes["sony"][1])
        feat["n_common"] = int(len(pts_s))
        features[name] = feat
    names = [n for n in reps if n in features]

    vectors = [cc.feature_vector(features[n]) for n in names]
    start = int(
        np.argmin(
            [abs(features[n]["u"] - 0.5) + abs(features[n]["v"] - 0.5) for n in names]
        )
    )
    order = [names[i] for i in cc.diversity_order(vectors, start)]
    order = [n for n in forced if n in names] + [n for n in order if n not in forced]
    calib, test = cc.split_calibration_test(order, args.max_pairs)
    for n in forced:
        if n in test:
            test.remove(n)
            calib.append(n)
    calib.sort()
    test.sort()

    warnings = cc.span_warnings([features[n] for n in calib])
    zs = [features[n]["z_m"] for n in calib]
    tilts = [features[n]["tilt_deg"] for n in calib]
    print(f"  Calibration pairs: {len(calib)}   Held-out pairs: {len(test)}")
    print(
        f"  Board distance {min(zs):.2f}-{max(zs):.2f} m, tilt up to {max(tilts):.1f} deg"
    )
    for w in warnings:
        print(f"  [WARN] {w}")

    # Copy and annotate the selected pairs for visual inspection.
    for subset, root, plot_dir, tag in (
        (calib, p["picked"], os.path.join(p["plots"], "picked_for_alignment"), "CALIB"),
        (
            test,
            p["picked_test"],
            os.path.join(p["plots"], "picked_for_validation"),
            "TEST",
        ),
    ):
        for cam in cc.CAMERAS:
            reset_dir(os.path.join(root, f"{cam}_rgb"))
        reset_dir(plot_dir)
        for name in subset:
            f = frames[name]
            common_ids = cc.common_points(f["sony"], f["zed"])[0]
            ft = features[name]
            label = (
                f"{tag} {name} | {len(common_ids)} shared corners | "
                f"{ft['z_m']:.2f} m | tilt {ft['tilt_deg']:.0f} deg"
            )
            for cam, src_dir in (("sony", p["synced_sony"]), ("zed", p["synced_zed"])):
                src = os.path.join(src_dir, name + ".png")
                shutil.copy2(src, os.path.join(root, f"{cam}_rgb", name + ".png"))
                annotate(
                    src,
                    os.path.join(plot_dir, f"{cam}_{name}.jpg"),
                    f[cam],
                    common_ids,
                    label,
                )

    plot_features(
        features, calib, test, os.path.join(p["plots"], "pair_selection_features.png")
    )
    cc.save_json(
        p["pair_selection"],
        {
            "dataset": args.dataset,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "n_still_pairs": len(candidates),
            "n_holds": len(holds),
            "calibration": calib,
            "test": test,
            "features": features,
            "excluded": sorted(excluded),
            "forced": sorted(forced),
            "warnings": warnings,
        },
    )
    print(f"\n  Saved: {p['pair_selection']}")
    print(f"  Frames copied to {p['picked']} and {p['picked_test']}\n")


if __name__ == "__main__":
    sys.exit(main())

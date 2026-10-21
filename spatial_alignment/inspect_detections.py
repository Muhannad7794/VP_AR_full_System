"""
inspect_detections.py
=====================
Step 1 of the spatial alignment pipeline.

Detects the ChArUco board once in every synced frame of both cameras and
stores the results, so every later step works on identical detections.

For every frame and camera it records:
    - the detected corner ids and sub-pixel positions
    - whether the view is usable (enough corners, spread over the board)
    - the board motion in pixels (median corner displacement against the
      frames STILL_OFFSET indices before and after) and whether the board is
      still (motion at most the stillness threshold of that camera)

Outputs:
    data/json_output/<dataset>/detections.json.gz        full detections (input to later steps)
    data/json_output/<dataset>/detection_summary.json    counts per category
    data/plots/<dataset>/spatial/detection_timeline.png  corners and still frames over time

Usage:
    python3 spatial_alignment/inspect_detections.py --dataset <name> [--workers 8]
        [--still-sony <px>] [--still-zed <px>]
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import calib_common as cc

_DETECTOR = None


def _init_worker():
    global _DETECTOR
    cv2.setNumThreads(1)
    _DETECTOR = cc.create_detector()


def _detect_pair(job):
    """Detects the board in the Sony and ZED images of one synced frame."""
    name, sony_path, zed_path = job
    sony, sony_size = cc.detect_image(_DETECTOR, sony_path)
    zed, zed_size = cc.detect_image(_DETECTOR, zed_path)
    return name, sony, zed, sony_size, zed_size


def plot_timeline(frames, plots_dir):
    names = sorted(frames)
    idx = np.array([cc.frame_index(n) for n in names])
    fig, axes = plt.subplots(2, 1, figsize=(16, 7), sharex=True)
    for ax, cam, label in zip(axes, cc.CAMERAS, ("Sony", "ZED")):
        counts = np.array(
            [len(frames[n][cam]["ids"]) if frames[n][cam] else 0 for n in names]
        )
        usable = np.array([frames[n]["usable"][cam] for n in names])
        still = np.array([frames[n]["still"][cam] for n in names])
        ax.plot(idx, counts, color="#9E9E9E", linewidth=0.6, label="Detected corners")
        ax.scatter(
            idx[usable & ~still],
            counts[usable & ~still],
            s=3,
            color="#FF9800",
            label="Usable, moving",
        )
        ax.scatter(
            idx[usable & still],
            counts[usable & still],
            s=3,
            color="#2E7D32",
            label="Usable and still",
        )
        ax.axhline(
            cc.MIN_CORNERS_VIEW,
            color="#C62828",
            linestyle="--",
            linewidth=1,
            label="Minimum corners",
        )
        ax.set_ylabel(f"{label} corners")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    axes[1].set_xlabel("Synced frame index")
    fig.suptitle("ChArUco detection over the recording", fontweight="bold")
    plt.tight_layout()
    out = os.path.join(plots_dir, "detection_timeline.png")
    plt.savefig(out, dpi=130)
    plt.close()
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Detect the ChArUco board in all synced frames."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1)
    )
    parser.add_argument(
        "--still-sony",
        type=float,
        default=None,
        help=f"Sony stillness threshold in pixels (default {cc.STILL_THRESHOLD_PX['sony']}).",
    )
    parser.add_argument(
        "--still-zed",
        type=float,
        default=None,
        help=f"ZED stillness threshold in pixels (default {cc.STILL_THRESHOLD_PX['zed']}).",
    )
    args = parser.parse_args()

    p = cc.paths(args.dataset)
    sony_names = set(cc.frame_names(p["synced_sony"]))
    zed_names = set(cc.frame_names(p["synced_zed"]))
    names = sorted(sony_names & zed_names)
    if not names:
        raise SystemExit(
            f"[ERROR] No synced frame pairs in {p['synced']}. Run the temporal pipeline first."
        )
    unpaired = len(sony_names ^ zed_names)
    if unpaired:
        print(
            f"[WARN] {unpaired} frames exist in only one camera folder and are ignored."
        )

    print(f"\n{'=' * 60}\n  Detection - dataset: {args.dataset}")
    print(f"  {len(names)} synced pairs, {args.workers} workers\n{'=' * 60}\n")

    jobs = [
        (
            n,
            os.path.join(p["synced_sony"], n + ".png"),
            os.path.join(p["synced_zed"], n + ".png"),
        )
        for n in names
    ]
    frames = {}
    sizes = {"sony": set(), "zed": set()}
    t0 = time.time()
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=_init_worker
    ) as pool:
        for k, (name, sony, zed, sony_size, zed_size) in enumerate(
            pool.map(_detect_pair, jobs, chunksize=8)
        ):
            frames[name] = {"sony": sony, "zed": zed}
            if sony_size:
                sizes["sony"].add(sony_size)
            if zed_size:
                sizes["zed"].add(zed_size)
            if (k + 1) % 500 == 0:
                print(f"  {k + 1}/{len(jobs)} frames ({time.time() - t0:.0f} s)")

    for cam in cc.CAMERAS:
        if len(sizes[cam]) != 1:
            raise SystemExit(
                f"[ERROR] {cam} frames have inconsistent or unreadable sizes: {sizes[cam]}"
            )
    image_size = {cam: list(next(iter(sizes[cam]))) for cam in cc.CAMERAS}

    # Quality and stillness flags.
    corners_3d = cc.board_corners_3d()
    by_index = {cc.frame_index(n): frames[n] for n in names}
    for name in names:
        entry = frames[name]
        entry["usable"] = {
            cam: bool(
                entry[cam] is not None
                and cc.view_is_usable(entry[cam]["ids"], corners_3d)
            )
            for cam in cc.CAMERAS
        }
    thresholds = dict(cc.STILL_THRESHOLD_PX)
    if args.still_sony is not None:
        thresholds["sony"] = args.still_sony
    if args.still_zed is not None:
        thresholds["zed"] = args.still_zed
    for name in names:
        entry = frames[name]
        idx = cc.frame_index(name)
        entry["motion_px"] = {
            cam: (
                cc.frame_motion_px(by_index, idx, cam) if entry["usable"][cam] else None
            )
            for cam in cc.CAMERAS
        }
        entry["still"] = {
            cam: bool(
                entry["usable"][cam]
                and cc.is_still(entry["motion_px"][cam], cam, thresholds[cam])
            )
            for cam in cc.CAMERAS
        }

    summary = {
        "dataset": args.dataset,
        "n_frames": len(names),
        "image_size": image_size,
        "per_camera": {},
    }
    for cam in cc.CAMERAS:
        motion = np.array(
            [
                frames[n]["motion_px"][cam]
                for n in names
                if frames[n]["motion_px"][cam] is not None
            ]
        )
        summary["per_camera"][cam] = {
            "detected": sum(1 for n in names if frames[n][cam] is not None),
            "usable": sum(1 for n in names if frames[n]["usable"][cam]),
            "usable_and_still": sum(1 for n in names if frames[n]["still"][cam]),
            "motion_px_percentiles": (
                {
                    str(q): float(np.percentile(motion, q))
                    for q in (5, 10, 25, 50, 75, 90)
                }
                if len(motion)
                else {}
            ),
        }
    summary["pairs_usable_and_still"] = sum(
        1 for n in names if all(frames[n]["still"][c] for c in cc.CAMERAS)
    )
    summary["rules"] = {
        "min_corners": cc.MIN_CORNERS_VIEW,
        "min_board_coverage": cc.MIN_BOARD_COVERAGE,
        "still_offset_frames": cc.STILL_OFFSET,
        "still_threshold_px": thresholds,
        "still_statistic": "median corner displacement, larger of the two neighbours",
    }

    cc.save_json(
        p["detections"],
        {
            "meta": {
                "dataset": args.dataset,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "opencv_version": cv2.__version__,
                "board": {
                    "cols": cc.BOARD_COLS,
                    "rows": cc.BOARD_ROWS,
                    "square_m": cc.SQUARE_LENGTH,
                    "marker_m": cc.MARKER_LENGTH,
                    "legacy_pattern": cc.LEGACY_PATTERN,
                },
                "image_size": image_size,
            },
            "frames": frames,
        },
    )
    cc.save_json(p["detection_summary"], summary)
    os.makedirs(p["plots"], exist_ok=True)
    timeline = plot_timeline(frames, p["plots"])

    for cam in cc.CAMERAS:
        s = summary["per_camera"][cam]
        print(
            f"  {cam.upper():5s} detected {s['detected']:6d} | usable {s['usable']:6d} | "
            f"usable and still {s['usable_and_still']:6d}   (image {image_size[cam][0]} x {image_size[cam][1]})"
        )
    print(
        f"  Pairs usable and still in both cameras: {summary['pairs_usable_and_still']}"
    )

    print(
        "\n  Board motion between frames 3 apart (median corner displacement, pixels):"
    )
    print(
        f"  {'':6s}{'threshold':>10s}{'p5':>8s}{'p10':>8s}{'p25':>8s}{'p50':>8s}{'p75':>8s}{'p90':>8s}"
    )
    for cam in cc.CAMERAS:
        pc = summary["per_camera"][cam]["motion_px_percentiles"]
        if pc:
            print(
                f"  {cam.upper():6s}{thresholds[cam]:10.2f}"
                + "".join(f"{pc[str(q)]:8.2f}" for q in (5, 10, 25, 50, 75, 90))
            )
    n_sony = summary["per_camera"]["sony"]["usable_and_still"]
    n_zed = summary["per_camera"]["zed"]["usable_and_still"]
    if n_sony >= 20 and n_zed < 0.5 * n_sony:
        print(
            f"\n  [WARN] The ZED has far fewer still frames ({n_zed}) than the Sony ({n_sony}) although both see the\n"
            "         same board on a rigid rig. The ZED corner positions are probably noisier than the threshold\n"
            "         allows. Compare the ZED percentiles above with the Sony ones and raise --still-zed if the ZED\n"
            "         noise floor (p5-p10) is close to its threshold."
        )
    print(
        f"\n  Saved: {p['detections']}\n  Saved: {p['detection_summary']}\n  Saved: {timeline}\n"
    )


if __name__ == "__main__":
    sys.exit(main())

# occlusion_refinement/validate_occlusion.py
#
# Applies the occlusion refinement pipeline (occlusion_pipeline.py) to the
# analysed frame range of a synced dataset, computes diagnostic metrics, and
# saves visualisations. Mirrors the role of validate_kinematics.py in
# skeleton_filtering/: the pipeline module contains no CLI or file I/O of
# its own, and this script is the only place dataset paths and per-dataset
# aggregation logic are handled.
#
# Analysed range
# --------------
# Recordings of a dataset start and stop with the performer entering or
# leaving the frame to operate the cameras. Those frames carry no
# information about the pipeline, so the first --trim-start and last
# --trim-end frames are excluded from every computation, plot and sample
# selection (they are not even read from disk). Statistics are therefore
# computed over the analysed range only.
#
# Metric
# ------
# Only stage 1 (confidence-based pixel gating) is implemented, so one
# metric is computed: the share of valid depth pixels that the gate removes.
# This is a proxy for the show-through incidence rate defined in the project
# proposal's evaluation framework (Chapter 8). It is a proxy, not the literal
# metric: measuring actual incorrect virtual-over-physical compositing
# requires the live compositor. The occlusion edge error metric from the
# same framework is not computed -- it depends on stages 2 and 3, which are
# not yet implemented.
#
# A pixel counts as "valid" when its depth is non-zero in the extracted
# dataset. Pixels without depth are excluded from the denominator and from
# the removed count, since the gate cannot remove what is already missing.
#
# Scene motion
# ------------
# Each frame also receives a motion energy value: the mean absolute depth
# change, in millimetres, between the frame and the frame
# MOTION_LAG_FRAMES earlier, over pixels valid in both (computed on a
# spatially downsampled copy). It is a relative indicator of how much of the
# scene is moving, used to (a) select sample frames spanning slow to fast
# movement and (b) show whether rejection tracks movement. It also contains
# a small baseline from depth sensor noise, so only its relative size is
# meaningful. A lag of several frames is used instead of the previous frame
# because the synced dataset can repeat the same source frame in
# consecutive positions, which would produce spurious zero values.

import argparse
import datetime
import glob
import json
import os
import sys
from collections import deque

import cv2
import matplotlib

matplotlib.use("Agg")  # headless rendering inside the container
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, PowerNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from occlusion_pipeline import OcclusionRefiner

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MOTION_LAG_FRAMES = 5  # frame distance used for the motion energy comparison
MOTION_DOWNSAMPLE = 4  # spatial subsampling factor for the motion energy
ZOOM_FRACTION = 0.25  # zoom window size as a fraction of frame width/height
PERSISTENT_REJECTION_FRACTION = (
    0.5  # share of frames above which a pixel counts as a static problem pixel
)
MIN_TRANSIENT_PIXELS = 200  # minimum non-persistent removed pixels needed to centre the zoom window on them
SAMPLE_MIN_GAP_DIVISOR = (
    3  # minimum spacing between samples = range length / (sample_count * this)
)
DEPTH_RANGE_PROBE_STEP = (
    50  # frames between probes used to fix the shared depth colour range
)

REMOVED_COLOR = (0.91, 0.07, 0.18)  # red used for pixels removed by the gate
KEPT_COLOR_HEX = "#2b6cb0"
SERIES_COLOR_REJECTION = "#c8312f"
SERIES_COLOR_MOTION = "#3b6ea5"


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Validate the occlusion refinement pipeline against a synced dataset."
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Name of the dataset folder"
    )
    parser.add_argument(
        "--confidence-threshold",
        type=int,
        default=50,
        help=(
            "Pixels with a confidence value above this threshold are removed by "
            "stage 1. Range [1, 100]. Not yet empirically tuned -- sweep this "
            "against the rejection-rate plot. Default: 50."
        ),
    )
    parser.add_argument(
        "--trim-start",
        type=int,
        default=400,
        help="Number of frames excluded from the start of the dataset. Default: 400.",
    )
    parser.add_argument(
        "--trim-end",
        type=int,
        default=942,
        help="Number of frames excluded from the end of the dataset. Default: 942.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=30,
        help=(
            "Number of frames exported as detailed before/after figures, selected "
            "to span the range of scene motion within the analysed frames. Default: 30."
        ),
    )
    parser.add_argument(
        "--frames",
        type=str,
        default="",
        help=(
            "Optional comma-separated synced frame indices (e.g. 2100,3300) that are "
            "rendered in addition to the automatically selected samples."
        ),
    )
    return parser.parse_args()


def parse_frame_list(text, n_frames):
    if not text.strip():
        return []
    frames = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = int(token)
        except ValueError:
            print(f"ERROR: --frames contains a non-integer value: '{token}'")
            sys.exit(1)
        if not (0 <= value < n_frames):
            print(
                f"ERROR: --frames value {value} is outside the dataset range [0, {n_frames - 1}]"
            )
            sys.exit(1)
        frames.append(value)
    return frames


# ---------------------------------------------------------------------------
# Data access and per-frame metrics
# ---------------------------------------------------------------------------


def load_frame_pair(depth_path, confidence_path):
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    confidence = cv2.imread(confidence_path, cv2.IMREAD_UNCHANGED)

    if depth is None:
        raise IOError(f"Failed to read depth frame: {depth_path}")
    if confidence is None:
        raise IOError(f"Failed to read confidence frame: {confidence_path}")

    return depth, confidence


def compute_motion_energy(current_small, reference_small):
    """
    Mean absolute depth change (mm) between two downsampled depth maps,
    restricted to pixels that carry a depth value in both. Differences are
    computed in int32 because uint16 subtraction would wrap around.
    """
    valid = (current_small > 0) & (reference_small > 0)
    if not np.any(valid):
        return 0.0
    difference = np.abs(
        current_small.astype(np.int32) - reference_small.astype(np.int32)
    )
    return float(difference[valid].mean())


# ---------------------------------------------------------------------------
# Sample frame selection
# ---------------------------------------------------------------------------


def select_sample_frames(records, sample_count, min_gap):
    """
    Select frames that span the range of scene motion in the analysed range.

    Target motion values are taken at evenly spaced quantiles from the
    highest to the lowest motion energy; for each target, the frame whose
    motion energy is closest is chosen, skipping frames closer than `min_gap`
    to one already chosen so that the samples come from different moments of
    the recording rather than from one movement. The frame with the highest
    rejection rate is always added. Motion energy measures how much of the
    scene is moving, not which pose is held, so this guarantees variety in
    movement speed and in position within the recording, not coverage of
    every distinct pose; --frames covers specific moments by hand.

    Returns a dict mapping frame index to a human-readable selection reason.
    """
    selected = {}
    candidates = [r for r in records if r["motion_energy_mm"] is not None]

    if candidates:
        indices = np.array([r["frame_idx"] for r in candidates])
        motion = np.array([r["motion_energy_mm"] for r in candidates])
        ranks = motion.argsort().argsort() / max(len(motion) - 1, 1)
        n_quantiles = max(sample_count - 1, 1)

        for quantile in np.linspace(1.0, 0.0, n_quantiles):
            target = np.quantile(motion, quantile)
            for j in np.argsort(np.abs(motion - target)):
                if all(abs(int(indices[j]) - chosen) >= min_gap for chosen in selected):
                    selected[int(indices[j])] = (
                        f"scene motion at percentile {int(round(100 * ranks[j]))} "
                        f"of the analysed frames"
                    )
                    break
    else:
        picks = np.linspace(0, len(records) - 1, max(sample_count - 1, 1))
        for p in picks:
            selected[records[int(round(p))]["frame_idx"]] = (
                "evenly spaced (motion unavailable)"
            )

    worst = max(records, key=lambda r: r["pct_rejected_of_previously_valid_pixels"])
    worst_idx = worst["frame_idx"]
    if worst_idx in selected:
        selected[worst_idx] += "; also the highest rejection rate"
    else:
        selected[worst_idx] = "highest rejection rate in the analysed range"

    return selected


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def build_confidence_colormap(threshold):
    """
    One colour per integer confidence value 1..100. Values the gate keeps
    (<= threshold) use a blue ramp, darkest for the most reliable; values it
    removes (> threshold) use a yellow-to-red ramp. The colour break sits at
    the threshold, so warm pixels in the confidence map are exactly the
    pixels the gate removes.
    """
    n_kept = max(threshold, 1)
    n_removed = max(100 - threshold, 0)
    kept = plt.get_cmap("Blues")(np.linspace(0.92, 0.38, n_kept))
    removed = (
        plt.get_cmap("YlOrRd")(np.linspace(0.30, 1.0, n_removed))
        if n_removed
        else np.empty((0, 4))
    )
    return ListedColormap(np.vstack([kept, removed]), name="confidence_split")


def find_zoom_window(removed_mask, persistent_mask, win_h, win_w):
    """
    Locate the window containing the most removed pixels. Pixels removed in
    most analysed frames (static scene edges) are ignored so that the window
    follows what changes between frames, such as the performer; if too few
    non-persistent pixels exist, all removed pixels are used, and if none
    exist the window is centred. A summed-area table gives every window sum
    in one vectorised step.
    """
    height, width = removed_mask.shape
    transient = removed_mask & ~persistent_mask

    if int(transient.sum()) >= MIN_TRANSIENT_PIXELS:
        source, mode = transient, "transient"
    elif removed_mask.any():
        source, mode = removed_mask, "any"
    else:
        return (height - win_h) // 2, (width - win_w) // 2, "centre"

    integral = cv2.integral(source.astype(np.uint8))
    window_sums = (
        integral[win_h:, win_w:]
        - integral[:-win_h, win_w:]
        - integral[win_h:, :-win_w]
        + integral[:-win_h, :-win_w]
    )
    y0, x0 = np.unravel_index(np.argmax(window_sums), window_sums.shape)
    return int(y0), int(x0), mode


def compose_overlay(depth, removed_mask, depth_range):
    """Greyscale depth (dark = near, bright = far) with removed pixels in red and missing depth in white."""
    vmin, vmax = depth_range
    normalised = np.clip(
        (depth.astype(np.float32) - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0
    )
    grey = 0.18 + 0.64 * normalised
    image = np.dstack([grey, grey, grey]).astype(np.float32)
    image[depth == 0] = 1.0
    image[removed_mask] = REMOVED_COLOR
    return image


def style_image_axis(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def save_sample_figure(
    depth,
    confidence,
    rejected_mask,
    persistent_mask,
    threshold,
    depth_range,
    header,
    out_path,
):
    height, width = depth.shape
    win_h, win_w = int(height * ZOOM_FRACTION), int(width * ZOOM_FRACTION)
    valid = depth > 0
    removed = rejected_mask & valid

    y0, x0, _ = find_zoom_window(removed, persistent_mask & valid, win_h, win_w)
    ys, xs = slice(y0, y0 + win_h), slice(x0, x0 + win_w)

    n_valid = int(valid.sum())
    pct_removed = 100.0 * int(removed.sum()) / n_valid if n_valid else 0.0

    vmin, vmax = depth_range
    depth_cmap = plt.get_cmap("viridis").copy()
    depth_cmap.set_bad(color="white")
    depth_display = np.ma.masked_where(~valid, depth)
    conf_cmap = build_confidence_colormap(threshold)

    # The full-frame overlay is thickened so removed pixels stay visible after
    # the image is shrunk to panel size; the zoom row uses the exact mask.
    # The kernel scales with the frame width and is always odd.
    thickness = max(3, int(round(width / 384)) | 1)
    thick_removed = (
        cv2.dilate(
            removed.astype(np.uint8), np.ones((thickness, thickness), np.uint8)
        ).astype(bool)
        & valid
    )
    overlay_full = compose_overlay(depth, thick_removed, depth_range)
    overlay_exact = compose_overlay(depth, removed, depth_range)

    # Three image columns, each followed by a narrow colour-bar column and a spacer
    # that leaves room for tick labels, so all image panels have identical size.
    fig = plt.figure(figsize=(22, 10.8))
    grid = fig.add_gridspec(
        2,
        7,
        width_ratios=[1, 0.035, 0.24, 1, 0.035, 0.40, 1],
        left=0.05,
        right=0.985,
        top=0.78,
        bottom=0.15,
        wspace=0.0,
        hspace=0.07,
    )
    axes = [[fig.add_subplot(grid[r, c]) for c in (0, 3, 6)] for r in range(2)]
    cax_depth = fig.add_subplot(grid[:, 1])
    cax_conf = fig.add_subplot(grid[:, 4])
    for row in axes:
        for ax in row:
            style_image_axis(ax)

    # Column 1: depth
    im_depth = axes[0][0].imshow(depth_display, cmap=depth_cmap, vmin=vmin, vmax=vmax)
    axes[1][0].imshow(depth_display[ys, xs], cmap=depth_cmap, vmin=vmin, vmax=vmax)
    cbar_depth = fig.colorbar(im_depth, cax=cax_depth)
    cbar_depth.set_label(
        "Distance from the camera (mm)\ndark = near, bright = far", fontsize=12
    )

    # Column 2: confidence
    im_conf = axes[0][1].imshow(
        confidence, cmap=conf_cmap, vmin=0.5, vmax=100.5, interpolation="nearest"
    )
    axes[1][1].imshow(
        confidence[ys, xs],
        cmap=conf_cmap,
        vmin=0.5,
        vmax=100.5,
        interpolation="nearest",
    )
    cbar_conf = fig.colorbar(im_conf, cax=cax_conf)
    cbar_conf.set_ticks([1, threshold, 100])
    cbar_conf.set_ticklabels(
        ["1\nmost reliable", f"{threshold}\nthreshold", "100\nleast reliable"]
    )
    cbar_conf.set_label(
        "Confidence score\nblue = kept, orange/red = removed", fontsize=12
    )

    # Column 3: result
    axes[0][2].imshow(overlay_full)
    axes[1][2].imshow(overlay_exact[ys, xs])

    # Zoom window outline on the full-frame images
    for col in range(3):
        axes[0][col].add_patch(
            Rectangle(
                (x0, y0), win_w, win_h, fill=False, edgecolor="white", linewidth=2.2
            )
        )
        axes[0][col].add_patch(
            Rectangle(
                (x0, y0),
                win_w,
                win_h,
                fill=False,
                edgecolor="black",
                linewidth=0.8,
                linestyle=(0, (3, 3)),
            )
        )

    axes[0][0].set_title(
        "1. Depth map\ndistance of every pixel from the camera", fontsize=14, pad=10
    )
    axes[0][1].set_title(
        "2. Confidence map\nthe sensor's own reliability score per pixel",
        fontsize=14,
        pad=10,
    )
    axes[0][2].set_title(
        f"3. Stage 1 result\nred = pixels removed (confidence score above {threshold})",
        fontsize=14,
        pad=10,
    )
    axes[0][0].set_ylabel("Full frame", fontsize=14, labelpad=10)
    axes[1][0].set_ylabel(
        "Zoom on the busiest region\n(4x, native pixels)", fontsize=14, labelpad=10
    )

    axes[1][0].legend(
        handles=[
            Patch(
                facecolor="white", edgecolor="#555555", label="white = no depth value"
            )
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
        fontsize=12,
    )
    axes[1][2].legend(
        handles=[
            Patch(facecolor=REMOVED_COLOR, label="removed by the gate"),
            Patch(facecolor="#8c8c8c", label="kept (grey = depth)"),
            Patch(facecolor="white", edgecolor="#555555", label="no depth value"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        frameon=False,
        fontsize=11.5,
        columnspacing=1.2,
        handletextpad=0.5,
    )

    fig.suptitle(
        f"{header}\nRemoved by the gate: {pct_removed:.2f}% of the pixels that have a depth value "
        f"(confidence threshold {threshold})",
        fontsize=16,
        y=0.975,
    )
    fig.text(
        0.5,
        0.012,
        "Dashed box on the full-frame row = area shown in the zoom row. In the full-frame result image, removed pixels are "
        f"thickened to {thickness} px so they stay visible when the image is shrunk; the zoom row is exact.\n"
        f"The zoom box ignores pixels that are removed in more than {int(PERSISTENT_REJECTION_FRACTION * 100)}% of the analysed "
        "frames (static scene edges), so it follows what changes from frame to frame.",
        ha="center",
        fontsize=10.5,
        color="#444444",
    )

    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def save_timeseries_plot(
    records, markers, threshold, analysed_range, trims, mean_rejection, out_path
):
    indices = np.array([r["frame_idx"] for r in records])
    rejection = np.array(
        [r["pct_rejected_of_previously_valid_pixels"] for r in records]
    )
    motion = np.array(
        [
            np.nan if r["motion_energy_mm"] is None else r["motion_energy_mm"]
            for r in records
        ]
    )

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(18, 9.8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 2], "hspace": 0.08},
    )
    fig.subplots_adjust(top=0.84)
    fig.suptitle(
        f"Stage 1 confidence gating (threshold {threshold}) over frames {analysed_range[0]}-{analysed_range[1]}\n"
        f"first {trims[0]} and last {trims[1]} frames excluded (performer entering or leaving the frame)",
        fontsize=14,
        y=0.985,
    )

    ax_top.plot(
        indices,
        rejection,
        color=SERIES_COLOR_REJECTION,
        linewidth=1.4,
        label="Pixels removed by the gate",
    )
    ax_top.axhline(
        mean_rejection,
        color="#555555",
        linestyle="--",
        linewidth=1.2,
        label=f"Mean over the analysed range: {mean_rejection:.2f}%",
    )
    peak = float(rejection.max()) if len(rejection) else 1.0
    ax_top.set_ylim(0, (peak if peak > 0 else 1.0) * 1.18)
    ax_top.set_ylabel(
        "Removed pixels\n(% of pixels that have a depth value)", fontsize=12
    )
    ax_top.grid(True, alpha=0.3)

    ax_bottom.plot(indices, motion, color=SERIES_COLOR_MOTION, linewidth=1.2)
    ax_bottom.set_ylabel(
        f"Scene motion\n(mean depth change over {MOTION_LAG_FRAMES} frames, mm)",
        fontsize=12,
    )
    ax_bottom.set_xlabel("Synced frame index", fontsize=12)
    ax_bottom.set_ylim(bottom=0)
    ax_bottom.grid(True, alpha=0.3)

    top_limit = ax_top.get_ylim()[1]
    for number, frame_idx in markers:
        for ax in (ax_top, ax_bottom):
            ax.axvline(frame_idx, color="#888888", linestyle=":", linewidth=1.0)
        ax_top.text(
            frame_idx,
            top_limit * 0.985,
            str(number),
            ha="center",
            va="top",
            fontsize=10,
            color="#222222",
            bbox=dict(
                boxstyle="round,pad=0.15",
                facecolor="white",
                edgecolor="#888888",
                linewidth=0.6,
            ),
        )

    handles, labels = ax_top.get_legend_handles_labels()
    if markers:
        handles.append(Line2D([0], [0], color="#888888", linestyle=":", linewidth=1.0))
        labels.append("numbered line = sample figure in the samples/ folder")
    ax_top.legend(
        handles,
        labels,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.0),
        ncol=3,
        frameon=False,
        fontsize=11,
    )

    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def save_rejection_frequency_map(counts, n_frames_analysed, out_path):
    frequency = 100.0 * counts.astype(np.float32) / max(n_frames_analysed, 1)

    fig, ax = plt.subplots(figsize=(13, 8))
    image = ax.imshow(
        frequency, cmap="YlOrRd", norm=PowerNorm(gamma=0.5, vmin=0, vmax=100)
    )
    style_image_axis(ax)
    cbar = fig.colorbar(
        image, ax=ax, fraction=0.035, pad=0.02, ticks=[0, 5, 10, 25, 50, 75, 100]
    )
    cbar.set_label(
        "Share of analysed frames in which the pixel was removed (%)\n"
        "square-root colour scale: low values are stretched so they stay visible",
        fontsize=11,
    )
    ax.set_title(
        "Where the gate removes pixels, and how persistently\n"
        "100% = removed in every analysed frame (static scene edge)\n"
        "low values = removed only some of the time (for example around the performer)",
        fontsize=13,
    )
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_arguments()
    dataset = args.dataset

    if not (1 <= args.confidence_threshold <= 100):
        print("ERROR: --confidence-threshold must be in [1, 100].")
        sys.exit(1)
    if args.trim_start < 0 or args.trim_end < 0:
        print("ERROR: --trim-start and --trim-end must not be negative.")
        sys.exit(1)

    depth_dir = os.path.join("data", "synced", dataset, "zed_depth")
    confidence_dir = os.path.join("data", "synced", dataset, "zed_confidence")

    if not os.path.exists(depth_dir):
        print(f"ERROR: Synced depth folder not found at {depth_dir}")
        sys.exit(1)
    if not os.path.exists(confidence_dir):
        print(f"ERROR: Synced confidence folder not found at {confidence_dir}")
        sys.exit(1)

    depth_files = sorted(glob.glob(os.path.join(depth_dir, "*.png")))
    confidence_files = sorted(glob.glob(os.path.join(confidence_dir, "*.png")))

    if not depth_files or not confidence_files:
        print(f"ERROR: No frames found in {depth_dir} or {confidence_dir}")
        sys.exit(1)
    if len(depth_files) != len(confidence_files):
        print(
            f"ERROR: Frame count mismatch -- depth: {len(depth_files)}, "
            f"confidence: {len(confidence_files)}. Refusing to proceed on "
            f"mismatched data rather than silently pairing frames incorrectly."
        )
        sys.exit(1)

    n_frames = len(depth_files)
    first_frame = args.trim_start
    last_frame = n_frames - 1 - args.trim_end
    if last_frame - first_frame < MOTION_LAG_FRAMES + 1:
        print(
            f"ERROR: Trimming {args.trim_start} frames from the start and {args.trim_end} from the end "
            f"leaves no usable range in a dataset of {n_frames} frames."
        )
        sys.exit(1)

    extra_frames = parse_frame_list(args.frames, n_frames)

    plots_dir = os.path.join("data", "plots", dataset, "occlusion")
    samples_dir = os.path.join(plots_dir, "samples")
    json_dir = os.path.join("data", "json_output", dataset)
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)

    refiner = OcclusionRefiner(confidence_threshold=args.confidence_threshold)

    n_analysed = last_frame - first_frame + 1
    print("====================================================")
    print(f"Validating Occlusion Refinement (Stage 1) for dataset: {dataset}")
    print(f"Confidence threshold: {args.confidence_threshold}")
    print(f"Dataset frames: {n_frames}")
    print(
        f"Analysed range: {first_frame}-{last_frame} ({n_analysed} frames; "
        f"first {args.trim_start} and last {args.trim_end} excluded)"
    )
    print("====================================================")

    records = []
    motion_buffer = deque(maxlen=MOTION_LAG_FRAMES)
    rejection_counts = None
    depth_range_probes = []

    for position, idx in enumerate(range(first_frame, last_frame + 1)):
        depth, confidence = load_frame_pair(depth_files[idx], confidence_files[idx])
        result = refiner.refine_frame(depth, confidence)

        valid = depth > 0
        removed = result["rejected_mask"] & valid

        if rejection_counts is None:
            rejection_counts = np.zeros(depth.shape, dtype=np.uint16)
        rejection_counts += removed

        n_valid = int(valid.sum())
        n_removed = int(removed.sum())
        pct_removed = 100.0 * n_removed / n_valid if n_valid > 0 else 0.0

        small = depth[::MOTION_DOWNSAMPLE, ::MOTION_DOWNSAMPLE].copy()
        motion = (
            compute_motion_energy(small, motion_buffer[0])
            if len(motion_buffer) == MOTION_LAG_FRAMES
            else None
        )
        motion_buffer.append(small)

        if position % DEPTH_RANGE_PROBE_STEP == 0 and n_valid > 0:
            depth_range_probes.append(np.percentile(depth[valid], [1, 99]))

        records.append(
            {
                "frame_idx": idx,
                "n_valid_depth_pixels": n_valid,
                "n_removed_pixels": n_removed,
                "pct_rejected_of_previously_valid_pixels": pct_removed,
                "motion_energy_mm": None if motion is None else round(motion, 4),
            }
        )

        if position % 500 == 0:
            print(f"\rProcessed: {position} / {n_analysed}", end="")

    print(f"\rProcessed: {n_analysed} / {n_analysed}")

    rejection_series = [r["pct_rejected_of_previously_valid_pixels"] for r in records]
    worst_position = int(np.argmax(rejection_series))
    summary = {
        "mean_pct_rejected_of_previously_valid_pixels": float(
            np.mean(rejection_series)
        ),
        "median_pct_rejected_of_previously_valid_pixels": float(
            np.median(rejection_series)
        ),
        "max_pct_rejected_of_previously_valid_pixels": float(
            rejection_series[worst_position]
        ),
        "frame_with_max_rejection": records[worst_position]["frame_idx"],
    }

    # Shared depth colour range, so every sample figure uses the same colour scale.
    probes = (
        np.array(depth_range_probes) if depth_range_probes else np.array([[0.0, 1.0]])
    )
    depth_range = (float(probes[:, 0].min()), float(probes[:, 1].max()))
    persistent_mask = rejection_counts >= (PERSISTENT_REJECTION_FRACTION * n_analysed)

    # Sample selection
    min_gap = max(1, n_analysed // (max(args.sample_count, 1) * SAMPLE_MIN_GAP_DIVISOR))
    selected = select_sample_frames(records, args.sample_count, min_gap)
    for frame_idx in extra_frames:
        if frame_idx in selected:
            selected[frame_idx] += "; also requested with --frames"
        else:
            selected[frame_idx] = "requested with --frames"
    ordered_samples = sorted(selected.items())

    # Remove stale sample figures from earlier runs, which may have used different selections.
    stale = glob.glob(os.path.join(samples_dir, "sample_*.png")) + glob.glob(
        os.path.join(samples_dir, "frame_*.png")
    )
    for path in stale:
        os.remove(path)

    sample_refiner = OcclusionRefiner(confidence_threshold=args.confidence_threshold)
    sample_files = []
    for number, (frame_idx, reason) in enumerate(ordered_samples, start=1):
        depth, confidence = load_frame_pair(
            depth_files[frame_idx], confidence_files[frame_idx]
        )
        result = sample_refiner.refine_frame(depth, confidence)
        header = f"Sample {number} of {len(ordered_samples)}  |  synced frame {frame_idx}  |  {reason}"
        out_name = f"sample_{number:02d}_frame_{frame_idx:05d}.png"
        save_sample_figure(
            depth,
            confidence,
            result["rejected_mask"],
            persistent_mask,
            args.confidence_threshold,
            depth_range,
            header,
            os.path.join(samples_dir, out_name),
        )
        sample_files.append(
            {
                "sample": number,
                "frame_idx": frame_idx,
                "reason": reason,
                "file": out_name,
            }
        )

    markers = [
        (entry["sample"], entry["frame_idx"])
        for entry in sample_files
        if first_frame <= entry["frame_idx"] <= last_frame
    ]

    timeseries_path = os.path.join(plots_dir, "confidence_gating_rejection_rate.png")
    save_timeseries_plot(
        records,
        markers,
        args.confidence_threshold,
        (first_frame, last_frame),
        (args.trim_start, args.trim_end),
        summary["mean_pct_rejected_of_previously_valid_pixels"],
        timeseries_path,
    )
    frequency_path = os.path.join(plots_dir, "rejection_frequency_map.png")
    save_rejection_frequency_map(rejection_counts, n_analysed, frequency_path)

    report = {
        "meta": {
            "dataset": dataset,
            "stage": "confidence_gating",
            "confidence_threshold": args.confidence_threshold,
            "n_frames_in_dataset": n_frames,
            "analysed_range": {
                "first_frame": first_frame,
                "last_frame": last_frame,
                "trim_start": args.trim_start,
                "trim_end": args.trim_end,
            },
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "definitions": {
                "pct_rejected_of_previously_valid_pixels": (
                    "Pixels removed by the gate, as a percentage of pixels that have a "
                    "non-zero depth value in the extracted dataset."
                ),
                "motion_energy_mm": (
                    f"Mean absolute depth change in millimetres against the frame "
                    f"{MOTION_LAG_FRAMES} frames earlier, over pixels valid in both, "
                    f"on a {MOTION_DOWNSAMPLE}x subsampled depth map. Null for the first "
                    f"{MOTION_LAG_FRAMES} analysed frames. Relative indicator only."
                ),
            },
            "note": (
                "Occlusion edge error is not computed -- it requires stages 2 and 3 "
                "(guided filtering, segmentation-guided alpha), which are not yet "
                "implemented. The rejection figures are a proxy for the show-through "
                "incidence rate defined in the project proposal, not a direct "
                "measurement of incorrect compositing, which requires the live compositor."
            ),
        },
        "summary": summary,
        "samples": sample_files,
        "per_frame": records,
    }

    report_path = os.path.join(json_dir, "occlusion_stage1_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("====================================================")
    print("Stage 1 validation complete.")
    print(f"Analysed frames {first_frame}-{last_frame} ({n_analysed} frames).")
    print(
        f"Mean removed (of pixels with depth):   {summary['mean_pct_rejected_of_previously_valid_pixels']:.3f}%"
    )
    print(
        f"Median removed (of pixels with depth): {summary['median_pct_rejected_of_previously_valid_pixels']:.3f}%"
    )
    print(
        f"Max removed (of pixels with depth):    {summary['max_pct_rejected_of_previously_valid_pixels']:.3f}% "
        f"(frame {summary['frame_with_max_rejection']})"
    )
    print("Sample figures:")
    for entry in sample_files:
        print(
            f"  {entry['sample']:>2}. frame {entry['frame_idx']:>5}  {entry['reason']}"
        )
    print(f"Report:            {report_path}")
    print(f"Rate plot:         {timeseries_path}")
    print(f"Frequency map:     {frequency_path}")
    print(f"Sample figures in: {samples_dir}/")
    print("Occlusion edge error metric: not computed -- pending stages 2/3.")
    print("====================================================")


if __name__ == "__main__":
    main()

# temporal_alignment/smooth_frames_mapping.py
import json
import numpy as np
import os
import argparse
from scipy.signal import medfilt
import matplotlib.pyplot as plt


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, required=True, help="Name of the dataset folder"
    )
    parser.add_argument(
        "--window",
        type=int,
        default=31,
        help=(
            "Median filter kernel size, in frames. Must be odd. This should "
            "be wide enough to absorb local DTW jitter within a stable "
            "plateau, but narrow enough to stay well clear of genuine "
            "offset transitions elsewhere in the sequence -- unlike a "
            "polynomial smoother, a median filter does not blend a real "
            "step across its full width, but a window that is too wide "
            "can still start to erode a genuinely short transition. "
            "Default of 31 is a starting point, not a validated value; "
            "tune it against the drift plot for each dataset."
        ),
    )
    parser.add_argument(
        "--transition-span",
        type=int,
        default=25,
        help=(
            "Half-width, in frames, of the rolling window used to detect "
            "and annotate genuine transition regions on the drift plot."
        ),
    )
    parser.add_argument(
        "--transition-threshold",
        type=int,
        default=5,
        help=(
            "Minimum change, in frames, within a transition-span window "
            "for a region to be flagged as a genuine transition on the "
            "drift plot, rather than local noise."
        ),
    )
    return parser.parse_args()


def find_transition_regions(values, half_span, threshold, merge_gap):
    """Flags contiguous regions where `values` moves by at least `threshold`
    within a window of width 2*half_span+1, independent of how the plot
    axes end up scaled. Adjacent flagged regions separated by a gap
    smaller than `merge_gap` are merged into one, so a single real
    transition does not get reported as several fragmented ones."""
    n = len(values)
    rolling_range = np.zeros(n)
    for i in range(n):
        lo = max(0, i - half_span)
        hi = min(n, i + half_span + 1)
        window_vals = values[lo:hi]
        rolling_range[i] = window_vals.max() - window_vals.min()

    flagged = rolling_range >= threshold

    regions = []
    start = None
    for i, is_flagged in enumerate(flagged):
        if is_flagged and start is None:
            start = i
        elif not is_flagged and start is not None:
            regions.append((start, i - 1))
            start = None
    if start is not None:
        regions.append((start, n - 1))

    merged = []
    for region in regions:
        if merged and region[0] - merged[-1][1] <= merge_gap:
            merged[-1] = (merged[-1][0], region[1])
        else:
            merged.append(region)

    return merged


def smooth_dtw_mapping():
    args = parse_arguments()

    window = args.window
    if window % 2 == 0:
        window += 1
        print(
            f"[WARN] --window must be odd for a median filter; "
            f"using {window} instead of {args.window}."
        )

    input_json = f"data/json_output/{args.dataset}/frame_mapping.json"
    output_json = f"data/json_output/{args.dataset}/smoothed_frame_mapping.json"
    plot_dir = f"data/plots/{args.dataset}/"
    os.makedirs(plot_dir, exist_ok=True)

    with open(input_json, "r") as f:
        mapping = json.load(f)

    sony_frames = np.array([int(k) for k in mapping.keys()])
    zed_frames = np.array([int(v) for v in mapping.values()])

    # A polynomial smoother (the previous Savitzky-Golay approach) fits a
    # continuous curve over its whole window, so it cannot represent a
    # genuine step change in the underlying offset -- it blends that step
    # across the full window width instead, however sharp the real
    # transition actually was. This signal is not smoothly varying: it is
    # piecewise-constant, with long stable plateaus interrupted by
    # occasional real jumps (hardware clock drift resolving, or a dropped
    # frame). A median filter matches that structure instead: within a
    # stable plateau it removes local DTW jitter same as any smoother
    # would, but once a majority of the window's samples belong to a new,
    # sustained plateau, the median value switches to it directly rather
    # than ramping gradually toward it, so a real transition stays sharp
    # instead of being smeared across hundreds of frames on either side.
    #
    # scipy's medfilt zero-pads at the array boundaries by default, which
    # would drag the first and last half-window of frames toward zero --
    # not appropriate here, since frame indices near the start and end of
    # a recording are not actually close to zero. Edge-padding with the
    # boundary value avoids that artifact.
    half_window = window // 2
    padded = np.pad(zed_frames.astype(float), pad_width=half_window, mode="edge")
    smoothed_padded = medfilt(padded, kernel_size=window)
    smoothed_zed = smoothed_padded[half_window : len(smoothed_padded) - half_window]

    original_drift = sony_frames - zed_frames
    smoothed_drift = sony_frames - smoothed_zed

    # Independently auto-scaled X and Y axes make visual slope an
    # unreliable guide to whether a change is significant: a Y-axis
    # spanning ~70 frames drawn across the same canvas width as an X-axis
    # spanning ~7000 frames will always look dramatic, regardless of the
    # real relative magnitude, and a genuinely large change can just as
    # easily be visually compressed into looking flat. Rather than try to
    # fix this by rescaling, the plot instead computes and marks the
    # actual transition regions directly from the numbers, so the
    # conclusion never depends on reading a slope by eye.
    transition_regions = find_transition_regions(
        smoothed_drift,
        half_span=args.transition_span,
        threshold=args.transition_threshold,
        merge_gap=args.transition_span,
    )

    fig, ax = plt.subplots(figsize=(20, 6))
    ax.plot(
        sony_frames,
        original_drift,
        label="Original DTW Jitter",
        color="blue",
        alpha=0.4,
    )
    ax.plot(
        sony_frames, smoothed_drift, label="Smoothed Drift", color="red", linewidth=2
    )

    for start_i, end_i in transition_regions:
        x0, x1 = sony_frames[start_i], sony_frames[end_i]
        delta = int(smoothed_drift[end_i] - smoothed_drift[start_i])
        ax.axvspan(x0, x1, color="orange", alpha=0.15)
        y_mid = (smoothed_drift[start_i] + smoothed_drift[end_i]) / 2
        ax.annotate(
            f"Sony {x0}-{x1}\n\u0394{delta:+d} frames",
            xy=((x0 + x1) / 2, y_mid),
            ha="center",
            fontsize=8,
            color="darkorange",
        )

    ax.set_xlabel("Sony Frame Number")
    ax.set_ylabel("Frame Offset (Sony - ZED)")
    ax.set_title(
        f"Hardware Drift Analysis - {args.dataset} (median filter, window={window})"
    )
    ax.legend(loc="upper left")
    ax.grid()
    fig.text(
        0.5,
        -0.02,
        "Note: X and Y axes are independently auto-scaled and are not "
        "proportional to each other -- shaded regions and annotations "
        "mark objectively-detected transitions; visual slope alone does "
        "not indicate significance.",
        ha="center",
        fontsize=8,
        style="italic",
    )

    fig.savefig(os.path.join(plot_dir, "drift_comparison.jpg"), bbox_inches="tight")
    plt.close(fig)

    if transition_regions:
        print(f"Detected {len(transition_regions)} transition region(s):")
        for start_i, end_i in transition_regions:
            delta = int(smoothed_drift[end_i] - smoothed_drift[start_i])
            print(
                f"  Sony frames {sony_frames[start_i]}-{sony_frames[end_i]}: "
                f"offset changed by {delta:+d} frames"
            )
    else:
        print("No transition regions detected above threshold.")

    smooth_mapping = {}
    prev_val = 1
    for sf, zf in zip(sony_frames, smoothed_zed):
        val = int(round(zf))
        if val < prev_val:
            val = prev_val
        smooth_mapping[str(sf)] = val
        prev_val = val

    with open(output_json, "w") as f:
        json.dump(smooth_mapping, f, indent=4)
    print(f"Saved final smoothed mapping successfully to {output_json}")


if __name__ == "__main__":
    smooth_dtw_mapping()

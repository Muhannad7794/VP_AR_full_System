# temporal_alignment/smooth_frames_mapping.py
import json
import numpy as np
import os
import argparse
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, required=True, help="Name of the dataset folder"
    )
    return parser.parse_args()


def smooth_dtw_mapping():
    args = parse_arguments()

    input_json = f"data/json_output/{args.dataset}/frame_mapping.json"
    output_json = f"data/json_output/{args.dataset}/smoothed_frame_mapping.json"
    plot_dir = f"data/plots/{args.dataset}/"
    os.makedirs(plot_dir, exist_ok=True)

    with open(input_json, "r") as f:
        mapping = json.load(f)

    sony_frames = np.array([int(k) for k in mapping.keys()])
    zed_frames = np.array([int(v) for v in mapping.values()])

    smoothed_zed = savgol_filter(zed_frames, window_length=1501, polyorder=2)

    original_drift = sony_frames - zed_frames
    smoothed_drift = sony_frames - smoothed_zed

    plt.figure(figsize=(12, 6))
    plt.plot(
        sony_frames,
        original_drift,
        label="Original DTW Jitter",
        color="blue",
        alpha=0.4,
    )
    plt.plot(
        sony_frames, smoothed_drift, label="Smoothed Drift", color="red", linewidth=2
    )
    plt.xlabel("Sony Frame Number")
    plt.ylabel("Frame Offset (Sony - ZED)")
    plt.title(f"Hardware Drift Analysis - {args.dataset}")
    plt.legend()
    plt.grid()

    plt.savefig(os.path.join(plot_dir, "drift_comparison.jpg"))
    plt.close()

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

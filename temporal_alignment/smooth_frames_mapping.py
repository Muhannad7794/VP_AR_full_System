import json
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt


def smooth_dtw_mapping():
    # 1. Load the erratic DTW mapping
    with open("data/frame_mapping.json", "r") as f:
        mapping = json.load(f)

    sony_frames = np.array([int(k) for k in mapping.keys()])
    zed_frames = np.array([int(v) for v in mapping.values()])

    # 2. Apply Savitzky-Golay Filter
    # widow length is 50 second at 30 fps. Polynomial order is 2 for smooth curve fitting.
    smoothed_zed = savgol_filter(zed_frames, window_length=1501, polyorder=2)

    # 3. plot the original and smoothed mappings for visual comparison
    # Calculate the exact drift (difference) between the cameras
    original_drift = sony_frames - zed_frames
    smoothed_drift = sony_frames - smoothed_zed

    plt.figure(figsize=(12, 6))

    # Plot the raw DTW jitter
    plt.plot(
        sony_frames,
        original_drift,
        label="Original DTW Jitter & Jumps",
        color="blue",
        alpha=0.4,
    )

    # Plot the smooth polynomial hardware drift
    plt.plot(
        sony_frames,
        smoothed_drift,
        label="Smoothed Hardware Drift (Savitzky-Golay)",
        color="red",
        linewidth=2,
    )

    plt.xlabel("Sony Frame Number")
    plt.ylabel("Frame Offset (Sony - ZED)")
    plt.title("Hardware Clock Drift & Smoothing Analysis")
    plt.legend()
    plt.grid()

    # Save the updated plot
    plt.savefig("data/plots/drift_comparison.jpg")
    plt.close()

    # 4. Enforce Forward Time (Monotonicity)
    smooth_mapping = {}
    prev_val = 1

    for sf, zf in zip(sony_frames, smoothed_zed):
        val = int(round(zf))
        if val < prev_val:
            val = prev_val

        # FIX: Cast the numpy.int64 back to a standard Python string for JSON
        smooth_mapping[str(sf)] = val
        prev_val = val

    # 5. Save the corrected, non-linear smoothed mapping
    with open("data/smoothed_frame_mapping.json", "w") as f:
        json.dump(smooth_mapping, f, indent=4)

    print("Saved final smoothed mapping successfully!")


if __name__ == "__main__":
    smooth_dtw_mapping()

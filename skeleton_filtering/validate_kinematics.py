"""
Validates the kinematic noise reduction and LMA metric extraction.
Applies the 1 Euro Filter to raw 3D coordinates and computes the LMA
Expansiveness proxy (maximum distal distance to spine).
Features modular arguments to specify the target joint and axis for
geometric noise reduction analysis.
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from one_euro_filter import OneEuroFilter


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Validate kinematics and generate LMA plots."
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Target dataset directory name."
    )
    parser.add_argument(
        "--joint",
        type=str,
        default="r_wrist",
        choices=["l_wrist", "r_wrist", "spine"],
        help="Target joint for noise analysis.",
    )
    parser.add_argument(
        "--axis",
        type=str,
        default="x",
        choices=["x", "y", "z"],
        help="Target spatial axis for noise analysis.",
    )
    return parser.parse_args()


def calculate_distance(p1, p2):
    """Computes Euclidean distance between two 3D coordinates."""
    return np.sqrt(np.sum((p1 - p2) ** 2))


def main():
    args = parse_arguments()
    dataset = args.dataset
    target_joint = args.joint
    target_axis = args.axis
    target_col = f"{target_joint}_{target_axis}"
    filtered_col = f"{target_col}_filtered"

    # Define paths
    input_csv = os.path.join(
        "/VP_AR_full_System_dockerized/data",
        "extracted",
        dataset,
        "kinematics",
        "raw_skeleton_data.csv",
    )
    plot_dir = os.path.join(
        "/VP_AR_full_System_dockerized/data", "plots", dataset, "kinematics"
    )
    os.makedirs(plot_dir, exist_ok=True)

    if not os.path.exists(input_csv):
        print(f"ERROR: Kinematic data not found at {input_csv}")
        return

    # Load data
    df = pd.read_csv(input_csv)

    if target_col not in df.columns:
        print(f"ERROR: Column {target_col} not found in the dataset.")
        return

    # Simulation Parameters (30 Hz ZED recording)
    fps = 30.0
    dt = 1.0 / fps
    timestamps = df["frame_idx"] * dt

    # 1 Euro Filter Hardware Tuning Parameters (Stereolab ZED 2i baseline)
    min_cutoff = 1.0  # Aggressive low-pass for 10-20mm idle positional noise
    beta = 0.05  # Speed coefficient to preserve ballistic movement

    # Initialize independent filter for the dynamically selected target
    t0 = timestamps.iloc[0]
    filter_target = OneEuroFilter(
        t0, df[target_col].iloc[0], min_cutoff=min_cutoff, beta=beta
    )

    filtered_target_data = []
    expansiveness_raw = []

    print(f"Applying adaptive filtering to {target_col} and computing LMA metrics...")

    for i in range(len(df)):
        t = timestamps.iloc[i]

        # Hardware Noise Filtering validation (Target Axis)
        raw_val = df[target_col].iloc[i]
        f_val = filter_target(t, raw_val)
        filtered_target_data.append(f_val)

        # LMA Expansiveness Calculation (Spatial Proxy - requires full 3D data)
        spine = np.array(
            [df["spine_x"].iloc[i], df["spine_y"].iloc[i], df["spine_z"].iloc[i]]
        )
        l_wrist = np.array(
            [df["l_wrist_x"].iloc[i], df["l_wrist_y"].iloc[i], df["l_wrist_z"].iloc[i]]
        )
        r_wrist = np.array(
            [df["r_wrist_x"].iloc[i], df["r_wrist_y"].iloc[i], df["r_wrist_z"].iloc[i]]
        )

        dist_l = calculate_distance(spine, l_wrist)
        dist_r = calculate_distance(spine, r_wrist)

        # Expansiveness defined as the maximum boundary of the kinesphere
        expansiveness_raw.append(max(dist_l, dist_r))

    df[filtered_col] = filtered_target_data
    df["expansiveness_raw"] = expansiveness_raw

    # ==========================================
    # Plot 1: Geometric Noise Reduction
    # ==========================================
    joint_display_name = target_joint.replace("_", " ").title()
    axis_display_name = target_axis.upper()

    plt.figure(figsize=(12, 6))
    plt.plot(
        timestamps,
        df[target_col],
        label="Raw ZED SDK Output",
        color="red",
        alpha=0.4,
        linewidth=1,
    )
    plt.plot(
        timestamps,
        df[filtered_col],
        label="1 Euro Filter Output",
        color="blue",
        linewidth=1.5,
    )
    plt.title(
        f"Geometric Sensor Noise Reduction ({joint_display_name} {axis_display_name}-Axis)"
    )
    plt.xlabel("Time (Seconds)")
    plt.ylabel("World Coordinate (mm)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    noise_plot_path = os.path.join(
        plot_dir, f"1Euro_noise_reduction_{target_joint}_{target_axis}.png"
    )
    plt.savefig(noise_plot_path, dpi=300)
    plt.close()

    # ==========================================
    # Plot 2: LMA Expansiveness Metric
    # ==========================================
    plt.figure(figsize=(12, 6))
    plt.plot(
        timestamps,
        df["expansiveness_raw"],
        label="Kinesphere Expansiveness (mm)",
        color="green",
        linewidth=1.5,
    )
    plt.title("LMA Space Proxy: Kinesphere Bounding Volume")
    plt.xlabel("Time (Seconds)")
    plt.ylabel("Distance (Spine to Distal Joint in mm)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    expansion_plot_path = os.path.join(plot_dir, "lma_expansiveness.png")
    plt.savefig(expansion_plot_path, dpi=300)
    plt.close()

    print(f"Validation plots successfully generated in: {plot_dir}")


if __name__ == "__main__":
    main()

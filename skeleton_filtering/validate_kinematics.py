"""
Validates kinematic noise reduction and LMA metric extraction across all 38 joints.
Applies the 1 Euro Filter to raw 3D coordinates based on dynamic CLI parameters.
Generates stacked subplot comparisons for clear Raw vs. Filtered visual validation.
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from one_euro_filter import OneEuroFilter

JOINTS_38 = [
    "pelvis",
    "spine_1",
    "spine_2",
    "spine_3",
    "neck",
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_clavicle",
    "right_clavicle",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_big_toe",
    "right_big_toe",
    "left_small_toe",
    "right_small_toe",
    "left_heel",
    "right_heel",
    "left_hand_thumb_4",
    "right_hand_thumb_4",
    "left_hand_index_1",
    "right_hand_index_1",
    "left_hand_middle_4",
    "right_hand_middle_4",
    "left_hand_pinky_1",
    "right_hand_pinky_1",
]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Validate comprehensive 38-joint kinematics."
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Target dataset directory name."
    )
    parser.add_argument(
        "--joint",
        type=str,
        default="right_wrist",
        choices=JOINTS_38,
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
        print(f"ERROR: Kinematic data not found at {input_csv}.")
        return

    df = pd.read_csv(input_csv)

    fps = 30.0
    dt = 1.0 / fps
    timestamps = df["frame_idx"] * dt

    # 1 Euro Filter Hardware Tuning Parameters
    min_cutoff = 1.0
    beta = 0.05

    t0 = timestamps.iloc[0]
    filter_target = OneEuroFilter(
        t0, df[target_col].iloc[0], min_cutoff=min_cutoff, beta=beta
    )

    filtered_target_data = []
    expansiveness_raw = []

    print(f"Processing 1 Euro filter for {target_col} and generating stacked plots...")

    for i in range(len(df)):
        t = timestamps.iloc[i]

        # 1. Target Joint Filtering
        raw_val = df[target_col].iloc[i]
        f_val = filter_target(t, raw_val)
        filtered_target_data.append(f_val)

        # 2. LMA Expansiveness Calculation (Strictly uses spine_2, left_wrist, right_wrist)
        spine = np.array(
            [df["spine_2_x"].iloc[i], df["spine_2_y"].iloc[i], df["spine_2_z"].iloc[i]]
        )
        l_wrist = np.array(
            [
                df["left_wrist_x"].iloc[i],
                df["left_wrist_y"].iloc[i],
                df["left_wrist_z"].iloc[i],
            ]
        )
        r_wrist = np.array(
            [
                df["right_wrist_x"].iloc[i],
                df["right_wrist_y"].iloc[i],
                df["right_wrist_z"].iloc[i],
            ]
        )

        dist_l = calculate_distance(spine, l_wrist)
        dist_r = calculate_distance(spine, r_wrist)

        expansiveness_raw.append(max(dist_l, dist_r))

    df[filtered_col] = filtered_target_data
    df["expansiveness_raw"] = expansiveness_raw

    # ==========================================
    # Plot 1: Stacked Subplot (Raw vs Filtered)
    # ==========================================
    joint_display_name = target_joint.replace("_", " ").title()
    axis_display_name = target_axis.upper()

    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True, sharey=True)

    # Top Plot: Raw Data
    axs[0].plot(
        timestamps,
        df[target_col],
        label=f"Raw Output ({axis_display_name})",
        color="red",
        alpha=0.8,
        linewidth=1.2,
    )
    axs[0].set_title(
        f"Raw ZED Sensor Data ({joint_display_name} | Axis: {axis_display_name})"
    )
    axs[0].set_ylabel("World Coordinate (mm)")
    axs[0].legend(loc="upper right")
    axs[0].grid(True, linestyle="--", alpha=0.6)

    # Bottom Plot: Filtered Data
    axs[1].plot(
        timestamps,
        df[filtered_col],
        label=f"1 Euro Filtered ({axis_display_name})",
        color="blue",
        linewidth=1.5,
    )
    axs[1].set_title(f"1 Euro Filtered Output")
    axs[1].set_xlabel("Time (Seconds)")
    axs[1].set_ylabel("World Coordinate (mm)")
    axs[1].legend(loc="upper right")
    axs[1].grid(True, linestyle="--", alpha=0.6)

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

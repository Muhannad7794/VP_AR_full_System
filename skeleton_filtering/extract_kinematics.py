"""
Extracts 3D kinematic joint coordinates from ZED 2i SVO recordings.
Utilizes the ZED SDK HUMAN_BODY_ACCURATE tracking model to isolate
and export all 38 skeletal joints defined in the BODY_38 format.
Outputs comprehensive spatial data as a structured CSV for subsequent
temporal filtering and dynamic LMA descriptor analysis.
"""

import sys
import pyzed.sl as sl
import os
import csv
import argparse
import glob
import numpy as np

# ZED BODY_38 Joint Definitions
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
        description="Extract 38-joint 3D skeletal kinematics from ZED SVO."
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Target dataset directory name."
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    dataset = args.dataset

    raw_dir = os.path.join("/VP_AR_full_System_dockerized/data", "raw", dataset)
    output_dir = os.path.join(
        "/VP_AR_full_System_dockerized/data", "extracted", dataset, "kinematics"
    )
    os.makedirs(output_dir, exist_ok=True)
    csv_output_path = os.path.join(output_dir, "raw_skeleton_data.csv")

    svo_files = glob.glob(os.path.join(raw_dir, "*.svo*"))
    if not svo_files:
        print(f"ERROR: SVO file not found in {raw_dir}")
        sys.exit(1)

    svo_file_path = svo_files[0]

    init_parameters = sl.InitParameters()
    init_parameters.set_from_svo_file(svo_file_path)
    init_parameters.depth_mode = sl.DEPTH_MODE.ULTRA
    init_parameters.coordinate_units = sl.UNIT.MILLIMETER

    zed = sl.Camera()
    err = zed.open(init_parameters)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"ERROR: Failed to open SVO file: {err}")
        sys.exit(1)

    # Initialize Positional Tracking (Mandatory for Body Tracking)
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    if (
        zed.enable_positional_tracking(positional_tracking_parameters)
        != sl.ERROR_CODE.SUCCESS
    ):
        print("ERROR: Positional tracking initialization failed.")
        sys.exit(1)

    # Configure Body Tracking
    body_params = sl.BodyTrackingParameters()
    body_params.enable_tracking = True
    body_params.enable_body_fitting = True
    body_params.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE
    body_params.body_format = sl.BODY_FORMAT.BODY_38

    if zed.enable_body_tracking(body_params) != sl.ERROR_CODE.SUCCESS:
        print("ERROR: Body tracking module initialization failed.")
        sys.exit(1)

    bodies = sl.Bodies()

    # Construct dynamic CSV Header
    csv_header = ["frame_idx"]
    for joint in JOINTS_38:
        csv_header.extend([f"{joint}_x", f"{joint}_y", f"{joint}_z"])

    with open(csv_output_path, mode="w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(csv_header)

        print(
            f"Extracting 38-Joint Kinematics from {os.path.basename(svo_file_path)}..."
        )

        frame_idx = 0
        while True:
            err = zed.grab()
            if err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                break
            elif err != sl.ERROR_CODE.SUCCESS:
                continue

            zed.retrieve_bodies(bodies)

            if bodies.is_new and len(bodies.body_list) > 0:
                body = bodies.body_list[0]
                keypoints = body.keypoint

                # Validation: Ensure core root joint (PELVIS) is actively tracked
                if not np.isnan(keypoints[0][0]):
                    row_data = [frame_idx]
                    for i in range(38):
                        row_data.extend(
                            [keypoints[i][0], keypoints[i][1], keypoints[i][2]]
                        )

                    csv_writer.writerow(row_data)

            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"\rProcessed Frames: {frame_idx}", end="")

    zed.disable_body_tracking()
    zed.disable_positional_tracking()
    zed.close()
    print(f"\nKinematic extraction complete: {csv_output_path}")


if __name__ == "__main__":
    main()

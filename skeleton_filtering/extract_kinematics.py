"""
Extracts 3D kinematic joint coordinates from ZED 2i SVO recordings.
Utilizes the ZED SDK HUMAN_BODY_ACCURATE tracking model to isolate
the spine and distal joints (wrists) required for Laban Movement Analysis (LMA).
Outputs data as a structured CSV for subsequent temporal filtering.
"""

import sys
import pyzed.sl as sl
import os
import csv
import argparse
import glob
import numpy as np


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Extract 3D skeletal kinematics from ZED SVO."
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Target dataset directory name."
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    dataset = args.dataset

    # Define input/output paths
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

    # Initialize ZED Camera parameters
    init_parameters = sl.InitParameters()
    init_parameters.set_from_svo_file(svo_file_path)
    init_parameters.depth_mode = sl.DEPTH_MODE.ULTRA
    init_parameters.coordinate_units = sl.UNIT.MILLIMETER

    zed = sl.Camera()
    err = zed.open(init_parameters)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"ERROR: Failed to open SVO file: {err}")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # FIX: Initialize Positional Tracking (Mandatory prerequisite for Body Tracking)
    # -------------------------------------------------------------------------
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    # Set to standard modes suitable for SVO playback
    err = zed.enable_positional_tracking(positional_tracking_parameters)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"ERROR: Positional tracking initialization failed: {err}")
        sys.exit(1)

    # Configure Body Tracking Neural Network
    body_params = sl.BodyTrackingParameters()
    body_params.enable_tracking = True
    body_params.enable_body_fitting = True
    body_params.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE
    body_params.body_format = sl.BODY_FORMAT.BODY_38

    if zed.enable_body_tracking(body_params) != sl.ERROR_CODE.SUCCESS:
        print("ERROR: Body tracking module initialization failed.")
        sys.exit(1)

    bodies = sl.Bodies()

    with open(csv_output_path, mode="w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        # Header definition for spatial LMA proxy joints
        csv_writer.writerow(
            [
                "frame_idx",
                "spine_x",
                "spine_y",
                "spine_z",
                "l_wrist_x",
                "l_wrist_y",
                "l_wrist_z",
                "r_wrist_x",
                "r_wrist_y",
                "r_wrist_z",
            ]
        )

        print(f"Extracting Kinematics from {os.path.basename(svo_file_path)}...")

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

                # BODY_38 Index Mapping: Spine=3, Left Wrist=16, Right Wrist=17
                spine = keypoints[3]
                l_wrist = keypoints[16]
                r_wrist = keypoints[17]

                # Validation: Discard frames with failed neural tracking (NaN values)
                if not (
                    np.isnan(spine[0]) or np.isnan(l_wrist[0]) or np.isnan(r_wrist[0])
                ):
                    csv_writer.writerow(
                        [
                            frame_idx,
                            spine[0],
                            spine[1],
                            spine[2],
                            l_wrist[0],
                            l_wrist[1],
                            l_wrist[2],
                            r_wrist[0],
                            r_wrist[1],
                            r_wrist[2],
                        ]
                    )

            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"\rProcessed Frames: {frame_idx}", end="")

    zed.disable_body_tracking()
    zed.disable_positional_tracking()
    zed.close()
    print(f"\nKinematic extraction complete: {csv_output_path}")


if __name__ == "__main__":
    main()

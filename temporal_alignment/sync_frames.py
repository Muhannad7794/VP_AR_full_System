# temporal_alignment/sync_frames.py
import cv2
import numpy as np
import os
import glob
import json
import argparse
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, required=True, help="Name of the dataset folder"
    )
    return parser.parse_args()


def get_motion_energy(frame_folder):
    frame_paths = sorted(glob.glob(os.path.join(frame_folder, "*.png")))
    energy_list = [0.0]  # First frame has 0 motion relative to itself

    print(
        f"Calculating motion energy for {len(frame_paths)} frames in {frame_folder}..."
    )

    # Read first frame and blur heavily to remove sensor noise
    prev_frame = cv2.imread(frame_paths[0], cv2.IMREAD_GRAYSCALE)
    prev_frame = cv2.GaussianBlur(prev_frame, (21, 21), 0)

    for path in frame_paths[1:]:
        curr_frame = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        curr_frame = cv2.GaussianBlur(curr_frame, (21, 21), 0)

        # Calculate absolute difference between current and previous frame
        diff = cv2.absdiff(prev_frame, curr_frame)

        # The mean of the difference represents the total "Motion Energy" of the frame
        energy = np.mean(diff)
        energy_list.append(energy)

        prev_frame = curr_frame

    # FastDTW expects a 2D array of coordinates, so we reshape the 1D list to (N, 1)
    return np.array(energy_list).reshape(-1, 1), frame_paths


def synchronize_datasets():
    args = parse_arguments()

    sony_folder = f"data/extracted/{args.dataset}/sony_rgb/"
    zed_folder = f"data/extracted/{args.dataset}/zed_rgb/"

    json_dir = f"data/json_output/{args.dataset}/"
    os.makedirs(json_dir, exist_ok=True)
    json_output_path = os.path.join(json_dir, "frame_mapping.json")

    # 1. Extract 1D Motion Energy
    sony_energy, _ = get_motion_energy(sony_folder)
    zed_energy, _ = get_motion_energy(zed_folder)

    if len(sony_energy) == 1 or len(zed_energy) == 1:
        print("Error: Could not extract frames. Check your folder paths.")
        return

    print("\nAligning motion velocity peaks using DTW (This will be very fast)...")

    # 2. Use Euclidean distance to match the amplitude of the motion peaks
    distance, path = fastdtw(sony_energy, zed_energy, dist=euclidean)
    print(f"DTW Alignment Complete. Total Distance: {distance:.4f}")

    # 3. Create Mapping
    frame_mapping = {}
    for sony_idx, zed_idx in path:
        actual_sony_frame = int(sony_idx + 1)
        actual_zed_frame = int(zed_idx + 1)

        # Keep the first matched ZED frame for each Sony frame
        if actual_sony_frame not in frame_mapping:
            frame_mapping[actual_sony_frame] = actual_zed_frame

    # Save to JSON
    try:
        with open(json_output_path, "w") as json_file:
            json.dump(frame_mapping, json_file, indent=4)
        print(f"\nSUCCESS: Frame mapping saved -> {json_output_path}")
    except Exception as e:
        print(f"\nFAILED to save JSON: {e}")


if __name__ == "__main__":
    synchronize_datasets()

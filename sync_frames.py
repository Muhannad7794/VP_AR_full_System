import cv2
import numpy as np
import os
import glob
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


def calculate_motion_signature(frame_folder):
    """
    Calculates the structural motion magnitude across a sequence of frames.
    Since the camera is on a tripod, pixel differences represent the actor's movement.
    """
    # Load all frame paths and sort them numerically
    frame_paths = sorted(glob.glob(os.path.join(frame_folder, "*.png")))

    motion_signature = []
    prev_gray = None

    print(f"Processing {len(frame_paths)} frames in {frame_folder}...")

    for path in frame_paths:
        frame = cv2.imread(path)
        # Convert to grayscale to simplify calculations
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            # Calculate the absolute difference between the current and previous frame
            frame_diff = cv2.absdiff(gray, prev_gray)
            # Sum the pixel differences to get a single 'motion magnitude' value
            motion_magnitude = np.sum(frame_diff)
            motion_signature.append(motion_magnitude)

        prev_gray = gray

    # Normalize the signature so both cameras are on the same scale (0 to 1)
    motion_signature = np.array(motion_signature)
    motion_signature = motion_signature / np.max(motion_signature)

    return motion_signature, frame_paths


def synchronize_datasets():
    # Define paths to your extracted frame datasets
    sony_folder = "data/extracted/sony_rgb/"
    zed_folder = "data/extracted/zed_rgb/"

    # Step 1: Extract 1D motion signatures from both camera datasets
    sony_motion, sony_paths = calculate_motion_signature(sony_folder)
    zed_motion, zed_paths = calculate_motion_signature(zed_folder)

    print("Calculating Dynamic Time Warping (DTW) path to fix clock drift...")

    # Reshape the arrays to (N, 1) so SciPy treats each frame's value as a 1D vector
    sony_motion = sony_motion.reshape(-1, 1)
    zed_motion = zed_motion.reshape(-1, 1)

    # Step 2: Use DTW to dynamically align the two motion signatures
    # DTW will return a 'path' which is a list of index pairs: [(sony_idx, zed_idx), ...]
    distance, path = fastdtw(sony_motion, zed_motion, dist=euclidean)

    print(f"DTW Alignment Distance: {distance}")

    # Step 3: Create a mapping dictionary to rename/sync the files
    frame_mapping = {}
    for sony_idx, zed_idx in path:
        # Since we compared frame differences, index 0 in motion corresponds to frame 1
        actual_sony_frame = sony_idx + 1
        actual_zed_frame = zed_idx + 1

        # Keep the first matching ZED frame for each Sony frame to handle drift
        if actual_sony_frame not in frame_mapping:
            frame_mapping[actual_sony_frame] = actual_zed_frame

    # Print a sample of the drift correction
    print("\nSample of the dynamic synchronization mapping:")
    for i in range(1, len(frame_mapping), len(frame_mapping) // 10):
        print(f"Sony Frame {i} -> Maps to ZED Frame {frame_mapping[i]}")

    # (Optional) Step 4: You can now use this frame_mapping dictionary to rename
    # or copy your ZED frames so they perfectly match the Sony frame numbers!


if __name__ == "__main__":
    synchronize_datasets()
 
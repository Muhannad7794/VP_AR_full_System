import sys
import pyzed.sl as sl
import cv2
import os
import numpy as np
from PIL import Image
from tqdm import tqdm


# --- CONFIGURATION ---
# Set the raw data paths
SVO_FILE_PATH = "data/raw/svo/HD1080_SN30014451_13-29-13.svo2"
MP4_FILE_PATH = "data/raw/mp4/training_video.mp4"

# Set the output directories for extracted frames
OUTPUT_ZED_DEPTH = "data/extracted/zed_depth/"
OUTPUT_ZED_RGB = "data/extracted/zed_rgb/"
OUTPUT_SONY_RGB = "data/extracted/sony_rgb/"

# Extract every Nth frame (set skipping to 1 to not skip any frames)
FRAME_SKIP = 1
# ---------------------


def create_directories():
    os.makedirs(OUTPUT_ZED_DEPTH, exist_ok=True)
    os.makedirs(OUTPUT_ZED_RGB, exist_ok=True)
    os.makedirs(OUTPUT_SONY_RGB, exist_ok=True)


def main():
    create_directories()

    # 1. Initialize ZED SDK for SVO Reading
    init_parameters = sl.InitParameters()
    init_parameters.set_from_svo_file(SVO_FILE_PATH)
    init_parameters.svo_real_time_mode = False
    init_parameters.depth_mode = sl.DEPTH_MODE.ULTRA  # Force highest quality
    init_parameters.coordinate_units = sl.UNIT.MILLIMETERP

    zed = sl.Camera()
    err = zed.open(init_parameters)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Failed to open SVO file: {err}")
        sys.exit(1)

    total_zed_frames = zed.get_svo_number_of_frames()

    # 2. Initialize OpenCV for MP4 Reading
    cap = cv2.VideoCapture(MP4_FILE_PATH)
    if not cap.isOpened():
        print(f"Failed to open MP4 file: {MP4_FILE_PATH}")
        sys.exit(1)

    total_sony_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Found {total_zed_frames} ZED frames and {total_sony_frames} Sony frames.")

    # Prepare ZED Image/Depth matrices
    zed_image = sl.Mat()
    zed_depth = sl.Mat()

    # 3. Extraction Loop
    frame_index = 0
    saved_count = 0

    print("Extracting frames...")

    while True:
        # Grab ZED frames
        err = zed.grab()
        # Grab Sony frames
        ret, sony_frame = cap.read()

        # 1. Check if the end of the video is reached
        if err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED or not ret:
            print(f"\nReached the end of the video files. Stopping extraction.")
            break

        # 2. Check if ZED just had a temporary glitch/dropped frame
        if err != sl.ERROR_CODE.SUCCESS:
            print(
                f"\n[Warning] ZED dropped a frame at index {frame_index}. Skipping..."
            )
            frame_index += 1
            continue  # Skip the rest of the loop and try the next frame

        # 3. Process the frame if it's perfectly healthy
        if frame_index % FRAME_SKIP == 0:
            # Extract ZED Data
            zed.retrieve_image(zed_image, sl.VIEW.LEFT)
            zed.retrieve_measure(zed_depth, sl.MEASURE.DEPTH)

            # 1. Force a copy of the data to ensure it's contiguous in memory for NumPy
            raw_bgra = np.array(zed_image.get_data(), copy=True)
            raw_depth = np.array(zed_depth.get_data(), copy=True)

            # 2. Pure NumPy conversion
            # Slice off Alpha [:, :, :3], then reverse the BGR colors to standard RGB [:, :, ::-1]
            rgb_image = raw_bgra[:, :, :3][:, :, ::-1]

            # 3. Clean depth data to 16-bit
            depth_16bit = np.nan_to_num(raw_depth).astype(np.uint16)

            # 4. Save ZED images using Pillow
            Image.fromarray(rgb_image).save(
                os.path.join(OUTPUT_ZED_RGB, f"zed_rgb_{saved_count:05d}.png")
            )
            Image.fromarray(depth_16bit).save(
                os.path.join(OUTPUT_ZED_DEPTH, f"zed_depth_{saved_count:05d}.png")
            )

            # --- Extract Sony Data ---
            cv2.imwrite(
                os.path.join(OUTPUT_SONY_RGB, f"sony_rgb_{saved_count:05d}.png"),
                sony_frame,
            )

            saved_count += 1

        # ALWAYS increment the frame counter at the end of the loop!
        frame_index += 1

        # Print progress every 100 frames
        if frame_index % 100 == 0:
            print(f"\rProcessed {frame_index} / {total_sony_frames} frames...", end="")

    # Cleanup
    zed.close()
    cap.release()
    print(f"\nExtraction complete! Saved {saved_count} synchronized frame sets.")


if __name__ == "__main__":
    main()

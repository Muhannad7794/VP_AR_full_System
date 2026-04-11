import sys
import pyzed.sl as sl
import cv2
import os
import numpy as np
from tqdm import tqdm


# --- CONFIGURATION ---
SVO_FILE_PATH = "data/raw/svo/HD1080_SN30014451_19-53-32.svo2"
MP4_FILE_PATH = "data/raw/mp4/training_video.mp4"

OUTPUT_ZED_DEPTH = "data/extracted/zed_depth/"
OUTPUT_ZED_RGB = "data/extracted/zed_rgb/"
OUTPUT_SONY_RGB = "data/extracted/sony_rgb/"

# Extract every Nth frame (1 = every frame, 2 = every other frame, 5 = every 5th frame)
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
    init_parameters.depth_mode = sl.DEPTH_MODE.ULTRA  # Force highest quality
    init_parameters.coordinate_units = sl.UNIT.MILLIMETER

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
    # We loop until either video runs out
    while True:
        # Grab ZED Frame
        err = zed.grab()

        # Grab Sony Frame
        ret, sony_frame = cap.read()

        if err != sl.ERROR_CODE.SUCCESS or not ret:
            break  # End of one or both videos

        if frame_index % FRAME_SKIP == 0:
            # --- Extract ZED Data ---
            zed.retrieve_image(zed_image, sl.VIEW.LEFT)
            zed.retrieve_measure(zed_depth, sl.MEASURE.DEPTH)

            # Get the raw data from ZED
            raw_bgra = zed_image.get_data()
            raw_depth = zed_depth.get_data()

            # THE BLANK CANVAS TRICK (RGB)
            # 1. Create a pure Python/NumPy blank canvas (Height, Width, 3 Colors)
            bgr_image = np.zeros(
                (raw_bgra.shape[0], raw_bgra.shape[1], 3), dtype=np.uint8
            )
            # 2. Copy the pixels from the ZED array into our pure Python array
            bgr_image[:, :, :] = raw_bgra[:, :, :3]

            # THE BLANK CANVAS TRICK (DEPTH)
            # 1. Clean the NaNs and convert to 16-bit
            clean_depth = np.nan_to_num(raw_depth).astype(np.uint16)
            # 2. Create a pure Python blank canvas (Height, Width)
            depth_16bit = np.zeros(
                (raw_depth.shape[0], raw_depth.shape[1]), dtype=np.uint16
            )
            # 3. Copy the pixels over
            depth_16bit[:, :] = clean_depth[:, :]

            # Save ZED RGB
            cv2.imwrite(
                os.path.join(OUTPUT_ZED_RGB, f"zed_rgb_{saved_count:05d}.png"),
                bgr_image,
            )

            # Save ZED Depth
            cv2.imwrite(
                os.path.join(OUTPUT_ZED_DEPTH, f"zed_depth_{saved_count:05d}.png"),
                depth_16bit,
            )

            # --- Extract Sony Data ---
            cv2.imwrite(
                os.path.join(OUTPUT_SONY_RGB, f"sony_rgb_{saved_count:05d}.png"),
                sony_frame,
            )

            saved_count += 1

        if frame_index % 100 == 0:
            print(f"Processed {frame_index} frames...")

    # Cleanup
    zed.close()
    cap.release()
    print(f"Extraction complete! Saved {saved_count} synchronized frame sets.")


if __name__ == "__main__":
    main()

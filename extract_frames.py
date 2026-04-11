import sys
import pyzed.sl as sl
import cv2
import os
from tqdm import tqdm

# --- CONFIGURATION ---
SVO_FILE_PATH = "data/raw/svo/HD1080_SN30014451_19-53-32.svo2"
MP4_FILE_PATH = (
    "data/raw/mp4/your_sony_video.mp4"  # Update this to your actual file name
)

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

            # Convert ZED data to numpy arrays
            bgra_image = zed_image.get_data()
            bgr_image = cv2.cvtColor(bgra_image, cv2.COLOR_BGRA2BGR)
            depth_data = (
                zed_depth.get_data()
            )  # Returns a 32-bit float array in millimeters

            # Save ZED RGB
            cv2.imwrite(
                os.path.join(OUTPUT_ZED_RGB, f"zed_rgb_{saved_count:05d}.png"),
                bgr_image,
            )

            # Save ZED Depth (Convert 32-bit float to 16-bit unsigned integer for PNG saving)
            # 1 unit = 1 millimeter. Max representable depth is 65.5 meters.
            depth_16bit = depth_data.astype("uint16")
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

        frame_index += 1

        if frame_index % 100 == 0:
            print(f"Processed {frame_index} frames...")

    # Cleanup
    zed.close()
    cap.release()
    print(f"Extraction complete! Saved {saved_count} synchronized frame sets.")


if __name__ == "__main__":
    main()

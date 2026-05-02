# temporal_alignment/extract_frames.py
import sys
import pyzed.sl as sl
import cv2
import os
import numpy as np
from PIL import Image
import argparse
import glob


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Extract frames from ZED SVO and Sony MP4."
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Name of the dataset folder"
    )
    parser.add_argument("--skip", type=int, default=1, help="Extract every Nth frame")
    return parser.parse_args()


def main():
    args = parse_arguments()
    dataset = args.dataset

    # 1. Auto-discover the raw files
    raw_dir = os.path.join("data", "raw", dataset)
    if not os.path.exists(raw_dir):
        print(f"ERROR: Raw folder not found at {raw_dir}")
        sys.exit(1)

    svo_files = glob.glob(os.path.join(raw_dir, "*.svo*"))
    mp4_files = glob.glob(os.path.join(raw_dir, "*.mp4"))

    if not svo_files or not mp4_files:
        print(f"ERROR: Missing SVO or MP4 file in {raw_dir}")
        sys.exit(1)

    SVO_FILE_PATH = svo_files[0]
    MP4_FILE_PATH = mp4_files[0]

    OUTPUT_ZED_DEPTH = os.path.join("data", "extracted", dataset, "zed_depth")
    OUTPUT_ZED_RGB = os.path.join("data", "extracted", dataset, "zed_rgb")
    OUTPUT_SONY_RGB = os.path.join("data", "extracted", dataset, "sony_rgb")
    FRAME_SKIP = args.skip

    os.makedirs(OUTPUT_ZED_DEPTH, exist_ok=True)
    os.makedirs(OUTPUT_ZED_RGB, exist_ok=True)
    os.makedirs(OUTPUT_SONY_RGB, exist_ok=True)

    # ==========================================
    #             EXTRACT ZED
    # ==========================================
    print(f"\n--- Starting ZED Extraction ({os.path.basename(SVO_FILE_PATH)}) ---")
    init_parameters = sl.InitParameters()
    init_parameters.set_from_svo_file(SVO_FILE_PATH)
    init_parameters.svo_real_time_mode = False
    init_parameters.depth_mode = sl.DEPTH_MODE.ULTRA
    init_parameters.coordinate_units = sl.UNIT.MILLIMETER

    zed = sl.Camera()
    err = zed.open(init_parameters)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Failed to open SVO file: {err}")
        sys.exit(1)

    total_zed_frames = zed.get_svo_number_of_frames()
    zed_image = sl.Mat()
    zed_depth = sl.Mat()

    zed_idx = 0
    saved_zed = 0

    while True:
        err = zed.grab()
        if err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            break
        elif err != sl.ERROR_CODE.SUCCESS:
            zed_idx += 1
            continue

        if zed_idx % FRAME_SKIP == 0:
            zed.retrieve_image(zed_image, sl.VIEW.LEFT)
            zed.retrieve_measure(zed_depth, sl.MEASURE.DEPTH)

            raw_bgra = np.array(zed_image.get_data(), copy=True)
            raw_depth = np.array(zed_depth.get_data(), copy=True)

            rgb_image = raw_bgra[:, :, :3][:, :, ::-1]
            depth_16bit = np.nan_to_num(raw_depth).astype(np.uint16)

            Image.fromarray(rgb_image).save(
                os.path.join(OUTPUT_ZED_RGB, f"zed_rgb_{saved_zed:05d}.png")
            )
            Image.fromarray(depth_16bit).save(
                os.path.join(OUTPUT_ZED_DEPTH, f"zed_depth_{saved_zed:05d}.png")
            )
            saved_zed += 1

        zed_idx += 1
        if zed_idx % 100 == 0:
            print(f"\rProcessed ZED: {zed_idx} / {total_zed_frames}", end="")

    zed.close()
    print(f"\nFinished ZED. Saved {saved_zed} frames.")

    # ==========================================
    #             EXTRACT SONY
    # ==========================================
    print(f"\n--- Starting Sony Extraction ({os.path.basename(MP4_FILE_PATH)}) ---")
    cap = cv2.VideoCapture(MP4_FILE_PATH)
    total_sony_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    sony_idx = 0
    saved_sony = 0

    while True:
        ret, sony_frame = cap.read()
        if not ret:
            break

        if sony_idx % FRAME_SKIP == 0:
            cv2.imwrite(
                os.path.join(OUTPUT_SONY_RGB, f"sony_rgb_{saved_sony:05d}.png"),
                sony_frame,
            )
            saved_sony += 1

        sony_idx += 1
        if sony_idx % 100 == 0:
            print(f"\rProcessed Sony: {sony_idx} / {total_sony_frames}", end="")

    cap.release()
    print(f"\nFinished Sony. Saved {saved_sony} frames.")

    print("\n==========================================")
    print(f"Extraction Complete for {dataset}!")
    print(f"Total files saved -> ZED: {saved_zed} | Sony: {saved_sony}")
    print("Ready for DTW Temporal Alignment.")
    print("==========================================\n")


if __name__ == "__main__":
    main()

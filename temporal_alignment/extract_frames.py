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
    OUTPUT_ZED_CONFIDENCE = os.path.join("data", "extracted", dataset, "zed_confidence")
    OUTPUT_SONY_RGB = os.path.join("data", "extracted", dataset, "sony_rgb")
    FRAME_SKIP = args.skip

    os.makedirs(OUTPUT_ZED_DEPTH, exist_ok=True)
    os.makedirs(OUTPUT_ZED_RGB, exist_ok=True)
    os.makedirs(OUTPUT_ZED_CONFIDENCE, exist_ok=True)
    os.makedirs(OUTPUT_SONY_RGB, exist_ok=True)

    # ==========================================
    #             EXTRACT ZED
    # ==========================================
    print(f"\n--- Starting ZED Extraction ({os.path.basename(SVO_FILE_PATH)}) ---")
    init_parameters = sl.InitParameters()
    init_parameters.set_from_svo_file(SVO_FILE_PATH)
    init_parameters.svo_real_time_mode = False
    init_parameters.depth_mode = sl.DEPTH_MODE.NEURAL
    init_parameters.coordinate_units = sl.UNIT.MILLIMETER

    zed = sl.Camera()
    err = zed.open(init_parameters)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Failed to open SVO file: {err}")
        sys.exit(1)

    total_zed_frames = zed.get_svo_number_of_frames()
    zed_image = sl.Mat()
    zed_confidence = sl.Mat()
    zed_depth = sl.Mat()

    # Depth is stored in millimetres as uint16, so any finite value beyond
    # this ceiling would overflow the cast. 65535 mm (~65.5 m) is already
    # far past the camera's usable range, so clamping here only ever
    # affects genuinely invalid values, never real scene depth.
    DEPTH_MAX_MM = 65535

    # The ZED SDK's confidence measure is documented as an integer-like
    # scale from 1 to 100, where LOW values mean the pixel is reliable and
    # HIGH values mean it should be rejected.
    CONFIDENCE_RELIABLE_MIN = 1
    CONFIDENCE_REJECT_MAX = 100
    CONFIDENCE_INVALID_SENTINEL = 100  # maximally "do not trust"

    # Running totals for the end-of-run validation summary. These exist so
    # that a mismatch between how depth and confidence report invalid
    # pixels is caught in this same pass, not discovered after the fact
    # on a second run.
    total_pixels_seen = 0
    depth_invalid_count = 0
    confidence_invalid_on_its_own_count = 0
    depth_invalid_but_confidence_finite_count = 0
    confidence_finite_but_out_of_spec_count = 0

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
            zed.retrieve_measure(zed_confidence, sl.MEASURE.CONFIDENCE)

            raw_bgra = np.array(zed_image.get_data(), copy=True)
            raw_depth = np.array(zed_depth.get_data(), copy=True)
            raw_confidence = np.array(zed_confidence.get_data(), copy=True)
            rgb_image = raw_bgra[:, :, :3][:, :, ::-1]

            # --- Validity masks, computed independently per channel ---
            # np.isfinite catches NaN and +/-inf in one check, covering all
            # three documented causes of an invalid depth pixel (stereo
            # occlusion, out-of-range distance, confidence-threshold
            # rejection) without needing to know which one produced it.
            depth_valid_mask = np.isfinite(raw_depth)
            confidence_valid_mask = np.isfinite(raw_confidence)

            # --- Depth: clamp invalid pixels to 0, guard against overflow ---
            depth_clamped = np.where(depth_valid_mask, raw_depth, 0.0)
            depth_clamped = np.clip(depth_clamped, 0, DEPTH_MAX_MM)
            depth_16bit = depth_clamped.astype(np.uint16)

            # --- Confidence: force the reject end wherever EITHER channel
            # says the pixel has no reliable data, rather than trusting
            # confidence's own raw value in isolation. This keeps the two
            # saved channels consistent with each other even if the SDK's
            # NaN behaviour differs between them for a given rejection
            # cause, so a Stage 1 gate reading confidence never mistakes a
            # depth-invalid pixel for a trustworthy one. ---
            no_reliable_data_mask = (~depth_valid_mask) | (~confidence_valid_mask)
            confidence_adjusted = np.where(
                no_reliable_data_mask, CONFIDENCE_INVALID_SENTINEL, raw_confidence
            )
            confidence_adjusted = np.clip(
                confidence_adjusted, CONFIDENCE_RELIABLE_MIN, CONFIDENCE_REJECT_MAX
            )
            confidence_8bit = confidence_adjusted.astype(np.uint8)

            # --- Validation bookkeeping (evidence, not just a print) ---
            total_pixels_seen += raw_depth.size
            depth_invalid_count += int((~depth_valid_mask).sum())
            confidence_invalid_on_its_own_count += int((~confidence_valid_mask).sum())
            depth_invalid_but_confidence_finite_count += int(
                (~depth_valid_mask & confidence_valid_mask).sum()
            )
            out_of_spec_mask = confidence_valid_mask & (
                (raw_confidence < CONFIDENCE_RELIABLE_MIN)
                | (raw_confidence > CONFIDENCE_REJECT_MAX)
            )
            confidence_finite_but_out_of_spec_count += int(out_of_spec_mask.sum())

            Image.fromarray(rgb_image).save(
                os.path.join(OUTPUT_ZED_RGB, f"zed_rgb_{saved_zed:05d}.png")
            )
            Image.fromarray(depth_16bit).save(
                os.path.join(OUTPUT_ZED_DEPTH, f"zed_depth_{saved_zed:05d}.png")
            )
            Image.fromarray(confidence_8bit).save(
                os.path.join(
                    OUTPUT_ZED_CONFIDENCE, f"zed_confidence_{saved_zed:05d}.png"
                )
            )
            saved_zed += 1

        zed_idx += 1
        if zed_idx % 100 == 0:
            print(f"\rProcessed ZED: {zed_idx} / {total_zed_frames}", end="")

    zed.close()
    print(f"\nFinished ZED. Saved {saved_zed} frames.")

    # --- End-of-run validation summary ---
    print("\n--- ZED Confidence/Depth Validation Summary ---")
    if total_pixels_seen > 0:
        print(
            f"Depth invalid (NaN/Inf):                 "
            f"{depth_invalid_count} / {total_pixels_seen} "
            f"({100.0 * depth_invalid_count / total_pixels_seen:.3f}%)"
        )
        print(
            f"Confidence invalid on its own (NaN/Inf):  "
            f"{confidence_invalid_on_its_own_count} / {total_pixels_seen} "
            f"({100.0 * confidence_invalid_on_its_own_count / total_pixels_seen:.3f}%)"
        )
        print(
            f"Depth invalid but confidence WAS finite:  "
            f"{depth_invalid_but_confidence_finite_count} / {depth_invalid_count if depth_invalid_count else 1} "
            f"of invalid-depth pixels"
        )
        print(
            f"Confidence finite but outside documented "
            f"[1,100] range: {confidence_finite_but_out_of_spec_count}"
        )
        if confidence_invalid_on_its_own_count == 0:
            print(
                "-> Confidence measure never returned NaN/Inf on its own in this "
                "run: it appears to be populated everywhere, including where "
                "depth is invalid. The cross-channel masking is still applied "
                "for safety, but was not strictly necessary for this dataset."
            )
        else:
            print(
                "-> Confidence measure DID return NaN/Inf independently of depth "
                "in this run. The cross-channel masking above was necessary and "
                "is correctly forcing those pixels to the reject sentinel."
            )
    print("------------------------------------------------\n")

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

#!/usr/bin/env python3
"""
extract_frames.py
=================
Frame extraction for one dataset: the ZED SVO and the Sony MP4 in
data/raw/<dataset>/.

Image outputs (data/extracted/<dataset>/), unchanged from earlier versions
-------------------------------------------------------------------------
    zed_rgb/zed_rgb_NNNNN.png                 rectified left image (RGB)
    zed_depth/zed_depth_NNNNN.png             depth, uint16 millimetres, 0 = invalid
    zed_confidence/zed_confidence_NNNNN.png   uint8 confidence, 1 (reliable) .. 100 (reject)
    sony_rgb/sony_rgb_NNNNN.png               Sony frame (BGR as decoded)

Timing and signal outputs (used by temporal_alignment/sync_frames.py)
---------------------------------------------------------------------
    zed_frames.csv    one row per decoded SVO frame
                      grab_index, saved_index (-1 if not saved), camera_timestamp_ns,
                      luma, motion
    sony_frames.csv   one row per decoded MP4 frame
                      frame_index, saved_index, container_ms, luma, motion
    extraction_info.json

    data/json_output/<dataset>/zed_intrinsics_extraction.json
        Rectified intrinsics of the session that produced the saved ZED
        images, in the format of spatial_alignment/print_zed_intrinsics.py.

The signals are computed for EVERY decoded frame, also when --skip > 1:
    luma    mean grey level of a 160 x 90 version of the frame (0-255)
    motion  mean absolute grey-level difference to the previous frame on the
            same 160 x 90 grid (NaN for the first frame)
They are cheap to compute and allow the temporal alignment to work from
timestamps and scene changes instead of comparing every image pair.

ZED timestamps are read from the SVO itself (TIME_REFERENCE.IMAGE). When the
recorder log <svo name>_timestamps.csv sits next to the SVO, the SVO frame
count and timestamps are compared with it and the result is reported.

Usage
-----
    python3 extract_frames.py --dataset dataset_05
    python3 extract_frames.py --dataset dataset_05 --skip 2
    python3 extract_frames.py --dataset dataset_05 --disable-self-calib
"""

import argparse
import csv
import glob
import json
import os
import sys
import time

import cv2
import numpy as np
import pyzed.sl as sl
from PIL import Image

# Size of the reduced grey image used for the luma and motion signals.
SIGNAL_SIZE = (160, 90)

# Clamping depth to the 16-bit unsigned integer range: the ZED SDK returns
# depth in millimetres and the maximum representable value is 65535 mm.
DEPTH_MAX_MM = 65535

# The ZED SDK confidence measure is an integer-like scale from 1 to 100,
# 1 being the most reliable and 100 the least reliable.
CONFIDENCE_RELIABLE_MIN = 1
CONFIDENCE_REJECT_MAX = 100
CONFIDENCE_INVALID_SENTINEL = 100  # maximally "do not trust"

# Consecutive grab errors after which the SVO is treated as unreadable.
MAX_CONSECUTIVE_GRAB_ERRORS = 50


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Extract frames from ZED SVO and Sony MP4."
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Name of the dataset folder"
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=1,
        help="Save every Nth frame (signals use every frame)",
    )
    parser.add_argument(
        "--disable-self-calib",
        action="store_true",
        help="Disable the ZED self-calibration at opening (factory calibration only). "
        "Off by default, as in earlier versions of this script.",
    )
    return parser.parse_args()


class FrameSignals:
    """Luma and frame-to-frame motion of a reduced grey image."""

    def __init__(self):
        self.prev = None

    def update(self, gray_small):
        img = gray_small.astype(np.float32)
        luma = float(img.mean())
        motion = (
            float(np.abs(img - self.prev).mean())
            if self.prev is not None
            else float("nan")
        )
        self.prev = img
        return luma, motion


def fmt(value, digits):
    return "" if value != value else f"{value:.{digits}f}"  # NaN -> empty field


def single_file(raw_dir, patterns, label):
    """Exactly one file matching the patterns, or exit with a clear message."""
    found = sorted({f for p in patterns for f in glob.glob(os.path.join(raw_dir, p))})
    if not found:
        sys.exit(f"ERROR: No {label} file in {raw_dir}")
    if len(found) > 1:
        names = "\n  ".join(found)
        sys.exit(
            f"ERROR: Several {label} files in {raw_dir}; keep exactly one:\n  {names}"
        )
    return found[0]


def intrinsics_record(zed, svo_path):
    """Rectified and raw left intrinsics in the format of print_zed_intrinsics.py."""
    info = zed.get_camera_information()
    config = info.camera_configuration
    rect = config.calibration_parameters.left_cam
    raw = config.calibration_parameters_raw.left_cam
    try:
        # Units follow InitParameters.coordinate_units (millimetres here).
        baseline_m = float(config.calibration_parameters.get_camera_baseline()) / 1000.0
    except Exception:
        baseline_m = None
    return {
        "source": f"svo:{os.path.basename(svo_path)} (extraction session)",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "sdk_version": str(sl.Camera.get_sdk_version()),
        "serial_number": int(info.serial_number),
        "camera_model": str(info.camera_model),
        "resolution": [int(config.resolution.width), int(config.resolution.height)],
        "rectified_left": {
            "fx": float(rect.fx),
            "fy": float(rect.fy),
            "cx": float(rect.cx),
            "cy": float(rect.cy),
            "width": int(config.resolution.width),
            "height": int(config.resolution.height),
            "distortion": [float(v) for v in rect.disto],
        },
        "raw_left": {
            "fx": float(raw.fx),
            "fy": float(raw.fy),
            "cx": float(raw.cx),
            "cy": float(raw.cy),
            "distortion": [float(v) for v in raw.disto],
        },
        "baseline_m": baseline_m,
    }


def compare_intrinsics(record, existing_path):
    """Report the difference to an existing zed_intrinsics.json, if there is one."""
    if not os.path.isfile(existing_path):
        return None
    with open(existing_path) as f:
        previous = json.load(f)["rectified_left"]
    current = record["rectified_left"]
    return {k: current[k] - previous[k] for k in ("fx", "fy", "cx", "cy")}


def compare_with_recorder_log(svo_path, zed_timestamps):
    """Compare SVO frame count and timestamps with the recorder log, if present."""
    log_path = os.path.splitext(svo_path)[0] + "_timestamps.csv"
    if not os.path.isfile(log_path):
        found = glob.glob(os.path.join(os.path.dirname(svo_path), "*_timestamps.csv"))
        if len(found) != 1:
            return {"log": None}
        log_path = found[0]
    with open(log_path, newline="") as f:
        log_ts = np.array(
            [int(r["camera_timestamp_ns"]) for r in csv.DictReader(f)], dtype=np.int64
        )
    svo_ts = np.asarray(zed_timestamps, dtype=np.int64)
    n = min(len(log_ts), len(svo_ts))
    same = int(np.sum(log_ts[:n] == svo_ts[:n]))
    first_mismatch = None
    if same < n:
        first_mismatch = int(np.flatnonzero(log_ts[:n] != svo_ts[:n])[0])
    return {
        "log": os.path.basename(log_path),
        "log_rows": int(len(log_ts)),
        "svo_frames_decoded": int(len(svo_ts)),
        "identical_timestamps": same,
        "first_mismatch_index": first_mismatch,
    }


def extract_zed(svo_path, dataset, out, frame_skip, disable_self_calib):
    print(f"\n--- Starting ZED Extraction ({os.path.basename(svo_path)}) ---")
    init_parameters = sl.InitParameters()
    init_parameters.set_from_svo_file(svo_path)
    init_parameters.svo_real_time_mode = False
    init_parameters.depth_mode = sl.DEPTH_MODE.NEURAL
    init_parameters.coordinate_units = sl.UNIT.MILLIMETER
    init_parameters.camera_disable_self_calib = bool(disable_self_calib)

    zed = sl.Camera()
    err = zed.open(init_parameters)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Failed to open SVO file: {err}")
        print(
            f"ZED SDK in use: {sl.Camera.get_sdk_version()}. An SVO recorded with a newer SDK "
            "may not be readable by an older one."
        )
        sys.exit(1)

    record = intrinsics_record(zed, svo_path)
    total_zed_frames = zed.get_svo_number_of_frames()
    zed_image = sl.Mat()
    zed_confidence = sl.Mat()
    zed_depth = sl.Mat()
    small_mat = sl.Mat()
    small_res = sl.Resolution(SIGNAL_SIZE[0], SIGNAL_SIZE[1])

    # Running totals for the end-of-run validation summary. These exist so
    # that a mismatch between how depth and confidence report invalid
    # pixels is caught in this same pass, not discovered after the fact
    # on a second run.
    total_pixels_seen = 0
    depth_invalid_count = 0
    confidence_invalid_on_its_own_count = 0
    depth_invalid_but_confidence_finite_count = 0
    confidence_finite_but_out_of_spec_count = 0

    signals = FrameSignals()
    timestamps = []
    grab_index = 0
    saved_zed = 0
    grab_errors = 0
    consecutive_errors = 0

    csv_file = open(os.path.join(out["root"], "zed_frames.csv"), "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(
        ["grab_index", "saved_index", "camera_timestamp_ns", "luma", "motion"]
    )

    while True:
        err = zed.grab()
        if err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            break
        if err != sl.ERROR_CODE.SUCCESS:
            grab_errors += 1
            consecutive_errors += 1
            if consecutive_errors > MAX_CONSECUTIVE_GRAB_ERRORS:
                print(
                    f"\n[WARN] {consecutive_errors} consecutive grab errors ({err}); stopping the ZED pass."
                )
                break
            continue
        consecutive_errors = 0

        ts_ns = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()
        timestamps.append(ts_ns)
        zed.retrieve_image(small_mat, sl.VIEW.LEFT, sl.MEM.CPU, small_res)
        luma, motion = signals.update(
            cv2.cvtColor(small_mat.get_data(), cv2.COLOR_BGRA2GRAY)
        )

        saved_index = -1
        if grab_index % frame_skip == 0:
            zed.retrieve_image(zed_image, sl.VIEW.LEFT)
            zed.retrieve_measure(zed_depth, sl.MEASURE.DEPTH)
            zed.retrieve_measure(zed_confidence, sl.MEASURE.CONFIDENCE)

            raw_bgra = np.array(zed_image.get_data(), copy=True)
            raw_depth = np.array(zed_depth.get_data(), copy=True)
            raw_confidence = np.array(zed_confidence.get_data(), copy=True)
            rgb_image = np.ascontiguousarray(raw_bgra[:, :, :3][:, :, ::-1])

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
            # says the pixel has no reliable data, so a gate reading
            # confidence never mistakes a depth-invalid pixel for a
            # trustworthy one. ---
            no_reliable_data_mask = (~depth_valid_mask) | (~confidence_valid_mask)
            confidence_adjusted = np.where(
                no_reliable_data_mask, CONFIDENCE_INVALID_SENTINEL, raw_confidence
            )
            confidence_adjusted = np.clip(
                confidence_adjusted, CONFIDENCE_RELIABLE_MIN, CONFIDENCE_REJECT_MAX
            )
            confidence_8bit = confidence_adjusted.astype(np.uint8)

            # --- Validation bookkeeping ---
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
                os.path.join(out["zed_rgb"], f"zed_rgb_{saved_zed:05d}.png")
            )
            Image.fromarray(depth_16bit).save(
                os.path.join(out["zed_depth"], f"zed_depth_{saved_zed:05d}.png")
            )
            Image.fromarray(confidence_8bit).save(
                os.path.join(
                    out["zed_confidence"], f"zed_confidence_{saved_zed:05d}.png"
                )
            )
            saved_index = saved_zed
            saved_zed += 1

        writer.writerow([grab_index, saved_index, ts_ns, fmt(luma, 4), fmt(motion, 5)])
        grab_index += 1
        if grab_index % 100 == 0:
            print(f"\rProcessed ZED: {grab_index} / {total_zed_frames}", end="")

    csv_file.close()
    zed.close()
    print(
        f"\nFinished ZED. Decoded {grab_index} frames, saved {saved_zed}, grab errors {grab_errors}."
    )

    # --- End-of-run validation summary ---
    print("\n--- ZED Confidence/Depth Validation Summary ---")
    if total_pixels_seen > 0:
        print(
            f"Depth invalid (NaN/Inf):                 {depth_invalid_count} / {total_pixels_seen} "
            f"({100.0 * depth_invalid_count / total_pixels_seen:.3f}%)"
        )
        print(
            f"Confidence invalid on its own (NaN/Inf):  {confidence_invalid_on_its_own_count} / "
            f"{total_pixels_seen} ({100.0 * confidence_invalid_on_its_own_count / total_pixels_seen:.3f}%)"
        )
        print(
            f"Depth invalid but confidence WAS finite:  {depth_invalid_but_confidence_finite_count} / "
            f"{depth_invalid_count if depth_invalid_count else 1} of invalid-depth pixels"
        )
        print(
            f"Confidence finite but outside documented [1,100] range: {confidence_finite_but_out_of_spec_count}"
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

    return (
        {
            "svo": os.path.basename(svo_path),
            "svo_frames_reported": int(total_zed_frames),
            "frames_decoded": int(grab_index),
            "frames_saved": int(saved_zed),
            "grab_errors": int(grab_errors),
            "first_timestamp_ns": int(timestamps[0]) if timestamps else None,
            "last_timestamp_ns": int(timestamps[-1]) if timestamps else None,
            "self_calibration_disabled": bool(disable_self_calib),
            "depth_mode": "NEURAL",
        },
        record,
        timestamps,
    )


def extract_sony(mp4_path, out, frame_skip):
    print(f"\n--- Starting Sony Extraction ({os.path.basename(mp4_path)}) ---")
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        sys.exit(f"ERROR: Could not open {mp4_path}")
    total_sony_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))

    signals = FrameSignals()
    csv_file = open(os.path.join(out["root"], "sony_frames.csv"), "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["frame_index", "saved_index", "container_ms", "luma", "motion"])

    sony_idx = 0
    saved_sony = 0
    while True:
        ret, sony_frame = cap.read()
        if not ret:
            break
        container_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC))
        gray_small = cv2.resize(
            cv2.cvtColor(sony_frame, cv2.COLOR_BGR2GRAY),
            SIGNAL_SIZE,
            interpolation=cv2.INTER_AREA,
        )
        luma, motion = signals.update(gray_small)

        saved_index = -1
        if sony_idx % frame_skip == 0:
            cv2.imwrite(
                os.path.join(out["sony_rgb"], f"sony_rgb_{saved_sony:05d}.png"),
                sony_frame,
            )
            saved_index = saved_sony
            saved_sony += 1

        writer.writerow(
            [sony_idx, saved_index, fmt(container_ms, 3), fmt(luma, 4), fmt(motion, 5)]
        )
        sony_idx += 1
        if sony_idx % 100 == 0:
            print(f"\rProcessed Sony: {sony_idx} / {total_sony_frames}", end="")

    cap.release()
    csv_file.close()
    print(f"\nFinished Sony. Decoded {sony_idx} frames, saved {saved_sony}.")
    return {
        "mp4": os.path.basename(mp4_path),
        "frames_reported": total_sony_frames,
        "frames_decoded": sony_idx,
        "frames_saved": saved_sony,
        "fps": fps,
    }


def main():
    args = parse_arguments()
    dataset = args.dataset
    if args.skip < 1:
        sys.exit("ERROR: --skip must be 1 or larger")

    raw_dir = os.path.join("data", "raw", dataset)
    if not os.path.exists(raw_dir):
        sys.exit(f"ERROR: Raw folder not found at {raw_dir}")
    svo_path = single_file(raw_dir, ["*.svo", "*.svo2"], "SVO")
    mp4_path = single_file(raw_dir, ["*.mp4", "*.MP4"], "MP4")

    root = os.path.join("data", "extracted", dataset)
    out = {
        "root": root,
        "zed_rgb": os.path.join(root, "zed_rgb"),
        "zed_depth": os.path.join(root, "zed_depth"),
        "zed_confidence": os.path.join(root, "zed_confidence"),
        "sony_rgb": os.path.join(root, "sony_rgb"),
    }
    for path in out.values():
        os.makedirs(path, exist_ok=True)
    stale = [
        k for k in ("zed_rgb", "sony_rgb") if glob.glob(os.path.join(out[k], "*.png"))
    ]
    if stale:
        print(
            f"[WARN] {', '.join(stale)} already contain frames; files with the same names are "
            "overwritten, older extra files are not removed."
        )

    zed_info, record, zed_timestamps = extract_zed(
        svo_path, dataset, out, args.skip, args.disable_self_calib
    )

    # Intrinsics of this extraction session, next to the pipeline's own file.
    json_dir = os.path.join("data", "json_output", dataset)
    os.makedirs(json_dir, exist_ok=True)
    intr_path = os.path.join(json_dir, "zed_intrinsics_extraction.json")
    with open(intr_path, "w") as f:
        json.dump(record, f, indent=2)
    diff = compare_intrinsics(record, os.path.join(json_dir, "zed_intrinsics.json"))

    log_check = compare_with_recorder_log(svo_path, zed_timestamps)
    sony_info = extract_sony(mp4_path, out, args.skip)

    info = {
        "dataset": dataset,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "skip": args.skip,
        "signal_size": list(SIGNAL_SIZE),
        "zed": zed_info,
        "sony": sony_info,
        "recorder_log_check": log_check,
        "zed_intrinsics_file": intr_path,
        "intrinsics_difference_to_zed_intrinsics_json": diff,
    }
    with open(os.path.join(root, "extraction_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    r = record["rectified_left"]
    print("\n==========================================")
    print(f"Extraction Complete for {dataset}!")
    print(
        f"Total files saved -> ZED: {zed_info['frames_saved']} | Sony: {sony_info['frames_saved']}"
    )
    print(
        f"ZED rectified intrinsics of this session: fx {r['fx']:.4f} fy {r['fy']:.4f} "
        f"cx {r['cx']:.4f} cy {r['cy']:.4f}  ->  {intr_path}"
    )
    if diff is not None:
        print(
            "  Difference to zed_intrinsics.json (extraction - existing): "
            + "  ".join(f"{k} {v:+.4f}" for k, v in diff.items())
        )
        if max(abs(v) for v in diff.values()) > 0.5:
            print(
                "  [WARN] Above 0.5 px. The saved ZED images follow the extraction values; "
                "pass this file to run_spatial.sh as the second argument."
            )
    if log_check.get("log"):
        ok = (
            log_check["log_rows"] == log_check["svo_frames_decoded"]
            and log_check["first_mismatch_index"] is None
        )
        print(
            f"Recorder log {log_check['log']}: {log_check['log_rows']} rows, SVO decoded "
            f"{log_check['svo_frames_decoded']} frames, identical timestamps "
            f"{log_check['identical_timestamps']}  ->  {'OK' if ok else 'MISMATCH'}"
        )
    print(f"Sony: {sony_info['frames_decoded']} frames at {sony_info['fps']:.3f} fps")
    print("Ready for temporal alignment (run_temporal.sh).")
    print("==========================================\n")


if __name__ == "__main__":
    main()

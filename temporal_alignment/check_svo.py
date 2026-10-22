#!/usr/bin/env python3
"""
check_svo.py
============
Integrity check of a recorded SVO2 file against the logs written by the Jetson
recorder (record_zed_svo.py).

Checks
------
1. The SVO opens with the installed ZED SDK. A file written by a newer SDK may
   not be readable by an older one; an open failure is reported with the
   SDK version in use.
2. Number of frames stored in the SVO versus the number of rows in
   <name>_timestamps.csv.
3. Whether SVO frame k carries the timestamp of log row k. It is tested on the
   first frame, the last frame and a number of evenly spread positions. A
   constant shift between the two would mean that the frame index and the log
   are offset. When the SVO and the log share the same clock (first timestamps
   within one second of each other) absolute timestamps are compared, so a shift
   of whole frames is visible. If the clocks differ, time relative to the first
   frame is used, which cannot reveal a shift of the very first frame.
4. Optional (--full): the whole file is decoded sequentially. This confirms
   that every frame decodes, and every timestamp is compared with the log.

Usage (repository root, inside the Docker container)
----------------------------------------------------
    python3 temporal_alignment/check_svo.py --dataset dataset_05
    python3 temporal_alignment/check_svo.py --dataset dataset_05 --full
    python3 temporal_alignment/check_svo.py --svo path/to/file.svo2

The dataset folder is data/raw/<dataset>. Depth is not computed, so the check
is fast; --full decodes every frame and takes longer.

Exit code: 0 when all checks pass, 1 otherwise.
"""

import argparse
import csv
import glob
import os
import sys

import numpy as np
import pyzed.sl as sl

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Largest accepted difference between an SVO timestamp (relative to the first
# frame) and the matching log row.
TOLERANCE_MS = 0.5


def attr_value(obj, name):
    """Read an SDK attribute that is a property in some versions and a method in others."""
    value = getattr(obj, name)
    return value() if callable(value) else value


def find_inputs(args):
    """Resolve the SVO path and the matching timestamp log."""
    if args.svo:
        svo_path = args.svo
    else:
        raw_dir = os.path.join(BASE_DIR, "data", "raw", args.dataset)
        if not os.path.isdir(raw_dir):
            sys.exit(f"[ERROR] Raw folder not found: {raw_dir}")
        candidates = sorted(glob.glob(os.path.join(raw_dir, "*.svo*")))
        if not candidates:
            sys.exit(f"[ERROR] No .svo/.svo2 file in {raw_dir}")
        if len(candidates) > 1:
            names = "\n  ".join(candidates)
            sys.exit(
                f"[ERROR] Several SVO files found, pass one with --svo:\n  {names}"
            )
        svo_path = candidates[0]
    if not os.path.isfile(svo_path):
        sys.exit(f"[ERROR] SVO file not found: {svo_path}")

    stem = os.path.splitext(svo_path)[0]
    log_path = args.log or f"{stem}_timestamps.csv"
    if not os.path.isfile(log_path):
        found = sorted(
            glob.glob(os.path.join(os.path.dirname(svo_path), "*_timestamps.csv"))
        )
        log_path = found[0] if len(found) == 1 else None
    return svo_path, log_path


def read_log(log_path):
    """Camera timestamps (ns) from the recorder log, as an int64 array."""
    stamps = []
    with open(log_path, newline="") as handle:
        for row in csv.DictReader(handle):
            stamps.append(int(row["camera_timestamp_ns"]))
    return np.asarray(stamps, dtype=np.int64)


def nearest_rows(relative_ns, log_relative_ns):
    """Index of the log row closest in time to each relative timestamp."""
    idx = np.searchsorted(log_relative_ns, relative_ns)
    idx = np.clip(idx, 1, len(log_relative_ns) - 1)
    left, right = log_relative_ns[idx - 1], log_relative_ns[idx]
    return np.where((relative_ns - left) <= (right - relative_ns), idx - 1, idx)


def open_svo(svo_path):
    init = sl.InitParameters()
    init.set_from_svo_file(svo_path)
    init.svo_real_time_mode = False
    init.depth_mode = sl.DEPTH_MODE.NONE
    cam = sl.Camera()
    err = cam.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"[ERROR] The SVO could not be opened: {err}")
        print(f"        ZED SDK in use: {sl.Camera.get_sdk_version()}")
        print(
            "        If the file was recorded with a newer SDK than this one, an older"
        )
        print(
            "        reader may refuse it. Rebuild the image with the matching SDK version."
        )
        sys.exit(1)
    return cam


def grab_timestamp(cam, runtime):
    """Grab one frame; return (timestamp_ns, error)."""
    err = cam.grab(runtime)
    if err != sl.ERROR_CODE.SUCCESS:
        return None, err
    return cam.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds(), err


def sample_check(cam, runtime, n_svo, log_ns, samples):
    """Compare frames at evenly spread positions with the log. Returns (ok, ts0)."""
    cam.set_svo_position(0)
    ts0, err = grab_timestamp(cam, runtime)
    if ts0 is None:
        print(f"[FAIL] The first frame could not be read: {err}")
        return False, None
    same_clock = abs(ts0 - int(log_ns[0])) < 1_000_000_000
    # Time origins: shared clock -> absolute comparison; otherwise each series is
    # measured from its own first frame.
    svo_origin = int(log_ns[0]) if same_clock else int(ts0)
    log_origin = int(log_ns[0])
    log_rel = log_ns - log_origin
    print(
        f"\nFirst frame timestamp: SVO {ts0} ns | log {log_ns[0]} ns "
        f"(difference {(ts0 - int(log_ns[0])) / 1e6:+.3f} ms; "
        f"{'same clock, absolute comparison' if same_clock else 'different clock origin, relative comparison'})"
    )

    positions = np.unique(np.linspace(0, n_svo - 1, samples).astype(int))
    print(
        f"\n{'requested':>9} {'pos after grab':>14} {'matched log row':>16} "
        f"{'row - requested':>16} {'residual ms':>12}"
    )
    offsets, residuals = [], []
    for k in positions:
        cam.set_svo_position(int(k))
        ts, err = grab_timestamp(cam, runtime)
        if ts is None:
            print(f"{k:>9}  grab failed: {err}")
            return False, ts0
        rel = np.asarray([ts - svo_origin], dtype=np.int64)
        row = int(nearest_rows(rel, log_rel)[0])
        residual_ms = (rel[0] - log_rel[row]) / 1e6
        offsets.append(row - int(k))
        residuals.append(residual_ms)
        print(
            f"{k:>9} {cam.get_svo_position():>14} {row:>16} {row - int(k):>16} "
            f"{residual_ms:>12.3f}"
        )

    constant = len(set(offsets)) == 1
    offset = offsets[0]
    worst = max(abs(r) for r in residuals)
    print()
    if constant and offset == 0 and worst <= TOLERANCE_MS:
        print(
            "[PASS] SVO frame k carries the timestamp of log row k at every sampled position."
        )
        return True, ts0
    if constant and worst <= TOLERANCE_MS:
        print(
            f"[WARN] SVO frame k matches log row k{offset:+d} at every sampled position: "
            "the frame index and the log are shifted by a constant."
        )
        return False, ts0
    print(
        "[FAIL] The offset between SVO frames and log rows is not constant "
        f"(offsets seen: {sorted(set(offsets))}, worst residual {worst:.3f} ms). "
        "Frames were probably dropped or reordered between the log and the file."
    )
    return False, ts0


def full_check(cam, runtime, log_ns):
    """Decode the whole file sequentially and compare every timestamp with the log."""
    print("\nDecoding the whole file ...")
    cam.set_svo_position(0)
    stamps = []
    while True:
        ts, err = grab_timestamp(cam, runtime)
        if ts is None:
            if err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                break
            print(f"[FAIL] Decoding stopped after {len(stamps)} frames: {err}")
            return False
        stamps.append(ts)
        if len(stamps) % 2000 == 0:
            print(f"  {len(stamps)} frames decoded")
    stamps = np.asarray(stamps, dtype=np.int64)
    print(f"Decoded {len(stamps)} frames; the log has {len(log_ns)} rows.")
    if len(stamps) != len(log_ns):
        print("[FAIL] Frame count differs from the log.")
        return False
    diff_ms = ((stamps - stamps[0]) - (log_ns - log_ns[0])) / 1e6
    bad = np.flatnonzero(np.abs(diff_ms) > TOLERANCE_MS)
    if len(bad):
        print(
            f"[FAIL] {len(bad)} timestamps differ from the log by more than "
            f"{TOLERANCE_MS} ms; first at frame {bad[0]} ({diff_ms[bad[0]]:.3f} ms)."
        )
        return False
    print("[PASS] Every frame decodes and every timestamp matches the log.")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Check a recorded SVO2 against the recorder timestamp log."
    )
    parser.add_argument("--dataset", help="dataset folder name under data/raw")
    parser.add_argument("--svo", help="path to a specific SVO file")
    parser.add_argument("--log", help="path to the matching *_timestamps.csv")
    parser.add_argument(
        "--samples",
        type=int,
        default=25,
        help="number of positions checked besides the first and last",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="decode the whole file and compare every timestamp",
    )
    args = parser.parse_args()
    if not args.dataset and not args.svo:
        parser.error("give --dataset or --svo")

    svo_path, log_path = find_inputs(args)
    print(f"SVO : {svo_path}")
    print(f"Log : {log_path if log_path else 'not found'}")
    print(f"ZED SDK in use: {sl.Camera.get_sdk_version()}")

    cam = open_svo(svo_path)
    runtime = sl.RuntimeParameters()
    ok = True
    try:
        info = cam.get_camera_information()
        cfg = attr_value(info, "camera_configuration")
        res = attr_value(cfg, "resolution")
        print(
            f"Camera serial {attr_value(info, 'serial_number')}, "
            f"{attr_value(res, 'width')}x{attr_value(res, 'height')} "
            f"@ {attr_value(cfg, 'fps')} fps"
        )
    except Exception as exc:  # attribute layout differs between SDK versions
        print(f"(camera information not readable: {exc})")

    n_svo = cam.get_svo_number_of_frames()
    print(f"\nFrames stored in the SVO: {n_svo}")

    if log_path is None:
        print("[WARN] No timestamp log found; only the frame count can be reported.")
        cam.close()
        return 1
    log_ns = read_log(log_path)
    print(f"Rows in the timestamp log: {len(log_ns)}")
    if n_svo == len(log_ns):
        print("[PASS] Frame count equals the number of log rows.")
    else:
        ok = False
        print(
            f"[FAIL] Frame count differs from the log by {n_svo - len(log_ns):+d}. "
            "The recorder logs a frame when it is grabbed; a smaller SVO means "
            "frames were not written."
        )

    if n_svo >= 2 and len(log_ns) >= 2:
        passed, _ = sample_check(
            cam, runtime, min(n_svo, len(log_ns)), log_ns, max(args.samples, 2)
        )
        ok = ok and passed
        if args.full:
            ok = full_check(cam, runtime, log_ns) and ok
    cam.close()
    print("\nOVERALL:", "PASS" if ok else "ATTENTION NEEDED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

"""
print_zed_intrinsics.py
=======================
Step 0 of the spatial alignment pipeline.

Reads the rectified left-camera intrinsics of the ZED from the ZED SDK and
writes them to data/json_output/<dataset>/zed_intrinsics.json. Every later
step reads the ZED intrinsics from that file.

The ZED SDK rectifies the left image onto a virtual pinhole camera with zero
distortion and computes depth on the same grid. These intrinsics therefore
describe the saved ZED frames and the depth maps exactly, and are not
estimated from board images.

Sources
-------
--dataset NAME          Read the SVO in data/raw/NAME/ (default source) and
                        write data/json_output/NAME/zed_intrinsics.json.
--svo PATH              Read a specific SVO file.
--live                  Open the connected camera (e.g. on the Jetson; stop the
                        streamer first, since the camera can only be opened once).
--from-json PATH        Use a JSON file written by this script elsewhere (for
                        example on the Jetson with --live --out) as the source.

Output resolution must match the frames used for calibration and the live
stream (HD1080, 1920 x 1080).

Usage:
    python3 spatial_alignment/print_zed_intrinsics.py --dataset dataset_04
    python3 spatial_alignment/print_zed_intrinsics.py --live --out zed_live.json
    python3 spatial_alignment/print_zed_intrinsics.py --dataset dataset_04 --from-json zed_live.json
"""

import argparse
import glob
import json
import os
import sys
import time

import calib_common as cc


def read_with_sdk(svo_path=None, live=False):
    """Opens an SVO file or the live camera and returns the intrinsics record."""
    try:
        import pyzed.sl as sl
    except ImportError:
        raise SystemExit(
            "[ERROR] pyzed is not available. Run inside the 'align' container or on the Jetson."
        )

    init = sl.InitParameters()
    init.depth_mode = sl.DEPTH_MODE.NONE  # intrinsics do not require depth computation
    init.coordinate_units = sl.UNIT.METER  # baseline reported in metres
    if svo_path:
        init.set_from_svo_file(svo_path)
        init.svo_real_time_mode = False
    else:
        init.camera_resolution = sl.RESOLUTION.HD1080

    cam = sl.Camera()
    err = cam.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        raise SystemExit(
            f"[ERROR] Could not open the {'SVO file' if svo_path else 'camera'}: {err}"
        )

    try:
        info = cam.get_camera_information()
        config = info.camera_configuration
        rect = config.calibration_parameters.left_cam
        raw = config.calibration_parameters_raw.left_cam
        try:
            baseline_m = float(config.calibration_parameters.get_camera_baseline())
        except Exception:
            baseline_m = None
        record = {
            "source": (
                f"svo:{os.path.basename(svo_path)}" if svo_path else "live camera"
            ),
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
    finally:
        cam.close()
    return record


def print_record(record):
    r = record["rectified_left"]
    print(f"\n  Source         : {record['source']}")
    print(f"  ZED SDK        : {record.get('sdk_version', 'unknown')}")
    print(f"  Serial number  : {record.get('serial_number', 'unknown')}")
    print(f"  Resolution     : {r['width']} x {r['height']}")
    print("  Rectified left camera (used by the pipeline and the shader):")
    print(
        f"    fx = {r['fx']:.4f}   fy = {r['fy']:.4f}   cx = {r['cx']:.4f}   cy = {r['cy']:.4f}"
    )
    if any(abs(v) > 1e-9 for v in r.get("distortion", [])):
        print("    [WARN] Non-zero distortion reported for the rectified camera.")
    raw = record.get("raw_left")
    if raw:
        print("  Raw left camera (reference only, not used):")
        print(
            f"    fx = {raw['fx']:.4f}   fy = {raw['fy']:.4f}   cx = {raw['cx']:.4f}   cy = {raw['cy']:.4f}"
        )
    if record.get("baseline_m"):
        print(f"  Stereo baseline: {record['baseline_m'] * 1000:.2f} mm")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Write the rectified ZED intrinsics used by the pipeline."
    )
    parser.add_argument(
        "--dataset",
        help="Dataset name; reads data/raw/<dataset>/*.svo* and writes the pipeline file.",
    )
    parser.add_argument("--svo", help="Explicit SVO path.")
    parser.add_argument(
        "--live", action="store_true", help="Read from the connected camera at HD1080."
    )
    parser.add_argument(
        "--from-json", help="Use a previously written intrinsics JSON as the source."
    )
    parser.add_argument(
        "--out", help="Output path (default: the pipeline file of --dataset)."
    )
    args = parser.parse_args()

    if args.from_json:
        with open(args.from_json) as f:
            record = json.load(f)
        record["source"] = (
            f"{record.get('source', 'unknown')} (via {os.path.basename(args.from_json)})"
        )
    elif args.live:
        record = read_with_sdk(live=True)
    else:
        svo = args.svo
        if not svo:
            if not args.dataset:
                parser.error("Provide --dataset, --svo, --live or --from-json.")
            found = sorted(
                glob.glob(os.path.join(cc.paths(args.dataset)["raw"], "*.svo*"))
            )
            if not found:
                raise SystemExit(
                    f"[ERROR] No SVO file in {cc.paths(args.dataset)['raw']}"
                )
            svo = found[0]
        record = read_with_sdk(svo_path=svo)

    print_record(record)

    out = args.out or (
        cc.paths(args.dataset)["zed_intrinsics"] if args.dataset else None
    )
    if not out:
        print("[INFO] No output path given (use --out or --dataset); nothing written.")
        return

    # When an SVO-derived file already exists, report any difference to the new source.
    if os.path.isfile(out) and args.from_json:
        with open(out) as f:
            previous = json.load(f)["rectified_left"]
        current = record["rectified_left"]
        diffs = {k: current[k] - previous[k] for k in ("fx", "fy", "cx", "cy")}
        print("  Difference to the previous file (new - previous):")
        print("    " + "   ".join(f"{k} {v:+.4f}" for k, v in diffs.items()))
        if max(abs(v) for v in diffs.values()) > 0.5:
            print(
                "    [WARN] Difference above 0.5 px. Calibrate with the values the live system uses."
            )
        print()

    cc.save_json(out, record)
    print(f"[SUCCESS] ZED intrinsics written to {out}")


if __name__ == "__main__":
    sys.exit(main())

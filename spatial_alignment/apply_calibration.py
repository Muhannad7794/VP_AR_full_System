"""
apply_calibration.py
====================
Runtime utility for the VP_AR_full_System spatial alignment pipeline.

Reads spatial_calibration.json and transforms a ZED camera-space 3-D point
(e.g., a skeleton joint position) into Sony camera space.

This script can be run in two modes:

1. SINGLE POINT MODE – supply X Y Z in metres to transform one point:
       python3 spatial_alignment/apply_calibration.py \\
           --dataset <name> --point 0.1 -0.05 1.5

2. UE5 OFFSET MODE – print the ready-to-use LensOffset values
   for the BP_TrackerAnchor blueprint:
       python3 spatial_alignment/apply_calibration.py \\
           --dataset <name> --ue5

The values printed by --ue5 are the X/Y/Z cm values to enter into the
LensOffset → Location field in the BP_TrackerAnchor Details panel in UE5.
"""

import argparse
import json
import os
import sys

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_calibration(dataset: str) -> dict:
    path = os.path.join(
        BASE_DIR, "data", "json_output", dataset, "spatial_calibration.json"
    )
    if not os.path.isfile(path):
        print(f"[ERROR] Calibration not found: {path}")
        print("        Run stereo_calibrate.py first.")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def transform_point(point_zed: np.ndarray, R: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Transform a single 3-D point from ZED camera space to Sony camera space.

    Parameters
    ----------
    point_zed : (3,) array, coordinates in metres
    R         : (3,3) rotation matrix  (extrinsics ZED→Sony)
    T         : (3,1) translation vector (extrinsics ZED→Sony)

    Returns
    -------
    point_sony : (3,) array, coordinates in metres in Sony space
    """
    return (R @ point_zed.reshape(3, 1) + T).flatten()


def print_ue5_offset(cal: dict):
    """
    Print the three LensOffset cm values suitable for direct entry into
    the BP_TrackerAnchor Details panel in UE5.
    """
    ue5 = cal["ue5_offset"]
    print("\n" + "=" * 60)
    print("  UE5 BP_TrackerAnchor – LensOffset values")
    print("=" * 60)
    print(f"  Location X : {ue5['X_cm']:+.3f} cm")
    print(f"  Location Y : {ue5['Y_cm']:+.3f} cm")
    print(f"  Location Z : {ue5['Z_cm']:+.3f} cm")
    print("=" * 60)
    print("\n  Enter these values in:")
    print("  Details panel → Default → Lens Offset → Location")
    print("  on the BP_TrackerAnchor instance in your level.\n")

    rod_deg = cal["extrinsics"]["rotation_degrees"]
    print("  Rotation offset (for reference only):")
    print(f"  Roll  (X) : {rod_deg[0]:+.3f}°")
    print(f"  Pitch (Y) : {rod_deg[1]:+.3f}°")
    print(f"  Yaw   (Z) : {rod_deg[2]:+.3f}°")
    print()
    print("  Note: For small physical offsets between ZED and Sony")
    print("  (shoe-mount separation of a few cm), rotation is usually")
    print("  negligible and only Location needs to be applied.")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Apply spatial calibration to ZED 3-D points or "
        "print UE5 offset values."
    )
    parser.add_argument(
        "--dataset", required=True, help="Dataset name matching the calibration JSON."
    )
    parser.add_argument(
        "--point",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="Transform a single point given in metres " "in ZED camera space.",
    )
    parser.add_argument(
        "--ue5",
        action="store_true",
        help="Print the ready-to-use UE5 LensOffset values.",
    )

    args = parser.parse_args()

    cal = load_calibration(args.dataset)
    R = np.array(cal["extrinsics"]["rotation_matrix_3x3"])
    T = np.array(cal["extrinsics"]["translation_metres"]).reshape(3, 1)

    print(f"\n[INFO] Calibration loaded – dataset: {args.dataset}")
    print(
        f"       Stereo RMS: {cal['extrinsics']['stereo_rms_px']:.4f} px  "
        f"({cal['meta']['valid_pairs_used']} pairs used)"
    )

    if args.ue5:
        print_ue5_offset(cal)

    if args.point:
        pt_zed = np.array(args.point, dtype=float)
        pt_sony = transform_point(pt_zed, R, T)
        print(
            f"\n  Input  (ZED  space, metres): "
            f"X={pt_zed[0]:+.4f}  Y={pt_zed[1]:+.4f}  Z={pt_zed[2]:+.4f}"
        )
        print(
            f"  Output (Sony space, metres): "
            f"X={pt_sony[0]:+.4f}  Y={pt_sony[1]:+.4f}  "
            f"Z={pt_sony[2]:+.4f}\n"
        )

    if not args.ue5 and not args.point:
        parser.print_help()
        print("\n[HINT] Use --ue5 to print the UE5 LensOffset values, or")
        print("       --point X Y Z to transform a specific 3-D point.\n")


if __name__ == "__main__":
    main()

"""
apply_calibration.py
====================
Step 6 of the spatial alignment pipeline.

Prints the deployable values from spatial_calibration.json (runtime solution):

--ue5     Values for the Custom node of M_ZED_DepthReproject and the lens
          offset of BP_TrackerAnchor.
--point   Transforms one ZED camera-space point (metres, OpenCV axes) into
          Sony camera space.

Lens offset convention
----------------------
The location is the Sony optical centre relative to the ZED left optical
centre, and the rotation turns the ZED's viewing direction into the Sony's,
both in the ZED's local Unreal axes (X forward, Y right, Z up; centimetres,
degrees). It is computed as -R^T @ T mapped from OpenCV axes (x right,
y down, z forward) to Unreal axes. T itself is the ZED origin seen from the
Sony and is not the lens offset.

Usage:
    python3 spatial_alignment/apply_calibration.py --dataset <name> --ue5
    python3 spatial_alignment/apply_calibration.py --dataset <name> --point 0.1 -0.05 1.5
"""

import argparse
import sys

import numpy as np

import calib_common as cc


def fmt_vec(v, digits=6):
    return "(" + ", ".join(f"{x:.{digits}f}" for x in v) + ")"


def print_ue5(cal):
    rt = cal["runtime"]
    report = None
    try:
        report = cc.load_json(cc.paths(cal["meta"]["dataset"])["validation"])
    except SystemExit:
        pass

    print("\n" + "=" * 64)
    print("  M_ZED_DepthReproject - Custom node inputs")
    print("=" * 64)
    print(f"  Sony_FX  {rt['Sony_FX']:.4f}")
    print(f"  Sony_FY  {rt['Sony_FY']:.4f}")
    print(f"  Sony_CX  {rt['Sony_CX']:.4f}")
    print(f"  Sony_CY  {rt['Sony_CY']:.4f}")
    print(f"  ZED_FX   {rt['ZED_FX']:.4f}")
    print(f"  ZED_FY   {rt['ZED_FY']:.4f}")
    print(f"  ZED_CX   {rt['ZED_CX']:.4f}")
    print(f"  ZED_CY   {rt['ZED_CY']:.4f}")
    print(f"  R_Row0   {fmt_vec(rt['R_Row0'])}")
    print(f"  R_Row1   {fmt_vec(rt['R_Row1'])}")
    print(f"  R_Row2   {fmt_vec(rt['R_Row2'])}")
    print(f"  T_Metres {fmt_vec(rt['T_Metres'])}")
    print(f"  ImageWidth {rt['ImageWidth']}   ImageHeight {rt['ImageHeight']}")
    d = rt.get("Sony_Distortion_k1_k2_p1_p2_k3")
    if d and any(abs(v) > 0 for v in d):
        print(
            f"  Sony lens distortion (k1, k2, p1, p2, k3), not used by the pinhole shader: {fmt_vec(d, 5)}"
        )

    loc = rt["lens_offset_location_cm"]
    rot = rt["lens_offset_rotation_deg"]
    print("\n" + "=" * 64)
    print("  BP_TrackerAnchor - Lens Offset")
    print("=" * 64)
    print(
        f"  Location  X {loc['X']:+.3f} cm   Y {loc['Y']:+.3f} cm   Z {loc['Z']:+.3f} cm"
    )
    print(
        f"  Rotation  Pitch {rot['Pitch']:+.3f} deg   Yaw {rot['Yaw']:+.3f} deg   Roll {rot['Roll']:+.3f} deg"
    )
    print(f"  Convention: {rt['lens_offset_convention']}")

    unc = cal.get("uncertainty", {}).get("bootstrap")
    if unc:
        s = unc["translation_std_mm"]
        print(
            f"\n  Uncertainty (bootstrap): T {s[0]:.1f} / {s[1]:.1f} / {s[2]:.1f} mm, R {unc['rotation_std_deg']:.3f} deg"
            f"\n  (pair noise only; intrinsic bias is assessed by the held-out validation)"
        )
    if report:
        print(f"  Validation assessment  : {report['assessment']}")
        scale = report.get("scale_check") or {}
        if scale.get("available"):
            print(
                f"  Scale check            : {scale['verdict']} (ZED depth / board depth "
                f"{scale['depth_ratio_median']:.4f}, board square {scale['board_square_mm']:.2f} mm)"
            )
            if scale["verdict"] != "Consistent":
                print(f"  [WARN] {scale['note']}")
    print(
        "\n  The ZED values are the SDK rectified intrinsics. The Sony values are the\n"
        "  calibrated camera matrix without lens distortion; the displacement this\n"
        "  omits is reported in sony_intrinsics.json (runtime_vs_physical_displacement)\n"
        "  and in error_across_sony_frame.png.\n"
    )


def main():
    parser = argparse.ArgumentParser(description="Print deployable calibration values.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--ue5", action="store_true", help="Print the Unreal Engine values."
    )
    parser.add_argument(
        "--point",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="ZED-space point in metres.",
    )
    args = parser.parse_args()

    cal = cc.load_json(cc.paths(args.dataset)["calibration"], "stereo_calibrate.py")
    R = np.asarray(cal["extrinsics"]["rotation_matrix_3x3"])
    T = np.asarray(cal["extrinsics"]["translation_metres"]).reshape(3)

    print(f"\n[INFO] Calibration loaded - dataset: {args.dataset}")
    print(
        f"       Stereo RMS {cal['extrinsics']['stereo_rms_px']:.4f} px "
        f"({cal['meta']['valid_pairs_used']} pairs), ZED intrinsics from {cal['meta']['zed_intrinsics_source']}"
    )

    if args.ue5:
        print_ue5(cal)
    if args.point:
        pt = np.array(args.point, dtype=float)
        out = R @ pt + T
        print(f"\n  ZED  (m): X={pt[0]:+.4f}  Y={pt[1]:+.4f}  Z={pt[2]:+.4f}")
        print(f"  Sony (m): X={out[0]:+.4f}  Y={out[1]:+.4f}  Z={out[2]:+.4f}\n")
    if not args.ue5 and not args.point:
        parser.print_help()


if __name__ == "__main__":
    sys.exit(main())

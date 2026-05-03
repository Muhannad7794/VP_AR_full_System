"""
validate_calibration.py
=======================
Phase 2 validation step for the VP_AR_full_System spatial alignment pipeline.

Reads the spatial_calibration.json produced by stereo_calibrate.py and
performs a rigorous per-frame analysis of the reprojection error, producing:

    data/plots/<dataset>/spatial/reprojection_error_distribution.png
    data/plots/<dataset>/spatial/per_frame_error.png
    data/plots/<dataset>/spatial/corner_map_sony.png
    data/plots/<dataset>/spatial/corner_map_zed.png
    data/plots/<dataset>/spatial/validation_report.json

Usage (via Docker – add a 'validate' service to docker-compose.yml, or reuse align):
    docker compose run --rm align python3 spatial_alignment/validate_calibration.py --dataset <name>

Usage (direct):
    python3 spatial_alignment/validate_calibration.py --dataset <dataset_name>
"""

import argparse
import json
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Board configuration – must match stereo_calibrate.py
# ---------------------------------------------------------------------------
BOARD_COLS = 11  # squares horizontally (landscape orientation)
BOARD_ROWS = 8  # squares vertically   (landscape orientation)
SQUARE_LENGTH = 0.028
MARKER_LENGTH = 0.021
ARUCO_DICT_ID = cv2.aruco.DICT_4X4_50
LEGACY_PATTERN = True  # required for calib.io generated boards


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_paths(dataset: str) -> dict:
    return {
        "sony_rgb": os.path.join(
            BASE_DIR, "data", "picked_for_alignment", dataset, "sony_rgb"
        ),
        "zed_rgb": os.path.join(
            BASE_DIR, "data", "picked_for_alignment", dataset, "zed_rgb"
        ),
        "calib_json": os.path.join(
            BASE_DIR, "data", "json_output", dataset, "spatial_calibration.json"
        ),
        "plots_out": os.path.join(BASE_DIR, "data", "plots", dataset, "spatial"),
        "json_out": os.path.join(BASE_DIR, "data", "json_output", dataset),
    }


# ---------------------------------------------------------------------------
# Detection (mirrors stereo_calibrate.py)
# ---------------------------------------------------------------------------
def create_board():
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    board = cv2.aruco.CharucoBoard(
        size=(BOARD_COLS, BOARD_ROWS),
        squareLength=SQUARE_LENGTH,
        markerLength=MARKER_LENGTH,
        dictionary=dictionary,
    )
    board.setLegacyPattern(LEGACY_PATTERN)
    return board, dictionary


def detect_charuco(image_path, board, dictionary):
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None, None, None
    size = (gray.shape[1], gray.shape[0])
    detector = cv2.aruco.CharucoDetector(board)
    c_corners, c_ids, m_corners, m_ids = detector.detectBoard(gray)
    if c_corners is None or len(c_corners) < 6:
        return None, None, size
    return c_corners, c_ids, size


# ---------------------------------------------------------------------------
# Per-frame reprojection error
# ---------------------------------------------------------------------------
def compute_reprojection_errors(
    paths: dict, board, dictionary, K_sony, D_sony, K_zed, D_zed, R, T
) -> list:
    """
    For each valid frame pair, project ZED-detected ChArUco corners into
    Sony image space using the stereo extrinsics and measure the pixel
    distance to the corners actually detected in the Sony frame.

    Returns a list of dicts, one per valid pair.
    """
    sony_files = sorted(
        f
        for f in os.listdir(paths["sony_rgb"])
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    zed_files = sorted(
        f
        for f in os.listdir(paths["zed_rgb"])
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )

    results = []
    for sf, zf in zip(sony_files, zed_files):
        sony_path = os.path.join(paths["sony_rgb"], sf)
        zed_path = os.path.join(paths["zed_rgb"], zf)

        s_corners, s_ids, _ = detect_charuco(sony_path, board, dictionary)
        z_corners, z_ids, _ = detect_charuco(zed_path, board, dictionary)

        if s_corners is None or z_corners is None:
            continue

        s_id_set = set(s_ids.flatten())
        z_id_set = set(z_ids.flatten())
        common = sorted(s_id_set & z_id_set)
        if len(common) < 4:
            continue

        s_mask = np.isin(s_ids.flatten(), common)
        z_mask = np.isin(z_ids.flatten(), common)
        s_pts = s_corners[s_mask].reshape(-1, 2)
        z_pts = z_corners[z_mask].reshape(-1, 2)
        obj_pts = board.getChessboardCorners()[np.array(common)].astype(np.float32)

        # Estimate pose of the board from ZED detections
        ret, rvec, tvec = cv2.solvePnP(
            obj_pts, z_pts, K_zed, D_zed, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ret:
            continue

        # Project board corners from ZED camera space into Sony image space
        # by chaining ZED intrinsics with stereo extrinsics
        # Transform: board_3d → ZED_cam → Sony_cam → Sony_image
        R_board, _ = cv2.Rodrigues(rvec)
        pts_zed_cam = (R_board @ obj_pts.T + tvec).T  # (N, 3)
        pts_sony_cam = (R @ pts_zed_cam.T + T).T  # (N, 3)

        projected, _ = cv2.projectPoints(
            pts_sony_cam, np.zeros(3), np.zeros(3), K_sony, D_sony
        )
        projected = projected.reshape(-1, 2)

        errors = np.linalg.norm(projected - s_pts, axis=1)

        results.append(
            {
                "frame_pair": (sf, zf),
                "n_corners": len(common),
                "mean_error_px": float(np.mean(errors)),
                "max_error_px": float(np.max(errors)),
                "std_error_px": float(np.std(errors)),
                "errors_px": errors.tolist(),
            }
        )

    return results


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------
def plot_error_distribution(results: list, plots_dir: str):
    all_errors = [e for r in results for e in r["errors_px"]]
    mean_errors = [r["mean_error_px"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Reprojection Error Analysis – ZED → Sony", fontsize=13, fontweight="bold"
    )

    # Histogram of all individual corner errors
    axes[0].hist(all_errors, bins=40, color="#2196F3", edgecolor="white", alpha=0.85)
    axes[0].axvline(
        np.mean(all_errors),
        color="#F44336",
        linewidth=2,
        label=f"Mean = {np.mean(all_errors):.3f} px",
    )
    axes[0].axvline(
        np.percentile(all_errors, 95),
        color="#FF9800",
        linewidth=2,
        linestyle="--",
        label=f"95th pct = {np.percentile(all_errors, 95):.3f} px",
    )
    axes[0].set_xlabel("Reprojection error (pixels)")
    axes[0].set_ylabel("Corner count")
    axes[0].set_title("Distribution of all corner errors")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Per-frame mean error
    frame_indices = range(len(mean_errors))
    colors = [
        "#4CAF50" if e < 1.0 else "#FF9800" if e < 2.0 else "#F44336"
        for e in mean_errors
    ]
    axes[1].bar(frame_indices, mean_errors, color=colors, edgecolor="white")
    axes[1].axhline(
        1.0,
        color="#4CAF50",
        linewidth=1.5,
        linestyle="--",
        label="Good threshold (1.0 px)",
    )
    axes[1].axhline(
        2.0,
        color="#F44336",
        linewidth=1.5,
        linestyle="--",
        label="Reject threshold (2.0 px)",
    )
    axes[1].set_xlabel("Frame pair index")
    axes[1].set_ylabel("Mean reprojection error (pixels)")
    axes[1].set_title("Per-frame mean reprojection error")
    axes[1].legend()
    axes[1].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    out = os.path.join(plots_dir, "reprojection_error_analysis.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_corner_coverage(
    paths: dict,
    board,
    dictionary,
    camera_key: str,
    label: str,
    plots_dir: str,
    image_size: tuple,
):
    """
    Draw all detected ChArUco corner positions on a blank canvas to show
    how well the corners cover the image plane.
    Coverage of the full image plane is essential for accurate distortion modelling.
    """
    files = sorted(
        f
        for f in os.listdir(paths[camera_key])
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )

    canvas = np.zeros((image_size[1], image_size[0]), dtype=np.float32)

    for fname in files:
        corners, _, _ = detect_charuco(
            os.path.join(paths[camera_key], fname), board, dictionary
        )
        if corners is None:
            continue
        for pt in corners.reshape(-1, 2):
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < image_size[0] and 0 <= y < image_size[1]:
                cv2.circle(canvas, (x, y), 8, 1.0, -1)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(
        canvas, cmap="hot", origin="upper", extent=[0, image_size[0], image_size[1], 0]
    )
    ax.set_title(
        f"Corner coverage – {label}\n"
        f"Bright = more detections. Full coverage improves accuracy.",
        fontsize=11,
    )
    ax.set_xlabel("Pixel X")
    ax.set_ylabel("Pixel Y")
    plt.colorbar(ax.images[0], ax=ax, label="Detection density")
    plt.tight_layout()

    out = os.path.join(plots_dir, f"corner_coverage_{camera_key}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Validate spatial calibration via reprojection error analysis."
    )
    parser.add_argument(
        "--dataset", required=True, help="Dataset name used in stereo_calibrate.py"
    )
    args = parser.parse_args()

    paths = build_paths(args.dataset)

    if not os.path.isfile(paths["calib_json"]):
        print(f"[ERROR] Calibration JSON not found: {paths['calib_json']}")
        print("        Run stereo_calibrate.py first.")
        sys.exit(1)

    with open(paths["calib_json"]) as f:
        cal = json.load(f)

    K_sony = np.array(cal["sony_camera"]["camera_matrix"])
    D_sony = np.array(cal["sony_camera"]["distortion_coeffs"])
    K_zed = np.array(cal["zed_camera"]["camera_matrix"])
    D_zed = np.array(cal["zed_camera"]["distortion_coeffs"])
    R = np.array(cal["extrinsics"]["rotation_matrix_3x3"])
    T = np.array(cal["extrinsics"]["translation_metres"]).reshape(3, 1)

    board, dictionary = create_board()
    os.makedirs(paths["plots_out"], exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Validation – dataset: {args.dataset}")
    print(f"{'='*60}\n")

    print("[INFO] Computing per-frame reprojection errors …")
    results = compute_reprojection_errors(
        paths, board, dictionary, K_sony, D_sony, K_zed, D_zed, R, T
    )

    if not results:
        print("[ERROR] No valid frame pairs found for validation.")
        sys.exit(1)

    all_errors = [e for r in results for e in r["errors_px"]]
    overall_mean = float(np.mean(all_errors))
    overall_std = float(np.std(all_errors))
    pct_95 = float(np.percentile(all_errors, 95))
    n_good = sum(1 for r in results if r["mean_error_px"] < 1.0)
    n_warn = sum(1 for r in results if 1.0 <= r["mean_error_px"] < 2.0)
    n_bad = sum(1 for r in results if r["mean_error_px"] >= 2.0)

    print(f"\n  Overall mean reprojection error : {overall_mean:.4f} px")
    print(f"  Standard deviation              : {overall_std:.4f} px")
    print(f"  95th percentile                 : {pct_95:.4f} px")
    print(f"  Frames < 1.0 px (good)  : {n_good}")
    print(f"  Frames 1–2 px  (warn)   : {n_warn}")
    print(f"  Frames ≥ 2.0 px (bad)   : {n_bad}")

    # Frames that should be removed
    bad_frames = [r["frame_pair"][0] for r in results if r["mean_error_px"] >= 2.0]
    if bad_frames:
        print(
            f"\n  [WARN] Consider removing these Sony frames from "
            f"picked_for_alignment/ and re-calibrating:"
        )
        for fname in bad_frames:
            print(f"         {fname}")

    print("\n[INFO] Generating plots …")
    plot_error_distribution(results, paths["plots_out"])

    # Determine image sizes from first readable file
    sony_size = zed_size = None
    for fname in sorted(os.listdir(paths["sony_rgb"])):
        img = cv2.imread(os.path.join(paths["sony_rgb"], fname))
        if img is not None:
            sony_size = (img.shape[1], img.shape[0])
            break
    for fname in sorted(os.listdir(paths["zed_rgb"])):
        img = cv2.imread(os.path.join(paths["zed_rgb"], fname))
        if img is not None:
            zed_size = (img.shape[1], img.shape[0])
            break

    if sony_size:
        plot_corner_coverage(
            paths,
            board,
            dictionary,
            "sony_rgb",
            "Sony ZV-E10 II",
            paths["plots_out"],
            sony_size,
        )
    if zed_size:
        plot_corner_coverage(
            paths,
            board,
            dictionary,
            "zed_rgb",
            "ZED 2i (Left)",
            paths["plots_out"],
            zed_size,
        )

    # Save validation report
    report = {
        "dataset": args.dataset,
        "n_frames_validated": len(results),
        "overall_mean_px": overall_mean,
        "overall_std_px": overall_std,
        "percentile_95_px": pct_95,
        "frames_good": n_good,
        "frames_warn": n_warn,
        "frames_bad": n_bad,
        "bad_frame_names": bad_frames,
        "assessment": (
            "EXCELLENT"
            if overall_mean < 0.5
            else (
                "GOOD"
                if overall_mean < 1.0
                else (
                    "ACCEPTABLE"
                    if overall_mean < 2.0
                    else "POOR – recalibrate with better frames"
                )
            )
        ),
        "per_frame": results,
    }

    report_path = os.path.join(paths["json_out"], "validation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Validation report saved → {report_path}")

    print(f"\n{'='*60}")
    print(f"  VALIDATION COMPLETE – Assessment: {report['assessment']}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

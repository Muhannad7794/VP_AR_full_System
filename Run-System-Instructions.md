# How to Control the System using CMD

## 🚀 Getting Started

**Step 1: Build the Environment**

Clone this repository to your local machine, open your terminal in the root folder, and build the GPU-accelerated Docker container. This only needs to be done once.

```
docker compose build --no-cache
```
**Step 2: Prepare Your Raw Data**
1. Create a new folder inside data/raw/ with a descriptive name for your recording session (e.g., dataset_01).

2. Drop your raw video files into that folder:
- One mp4 file.
- One svo file.

## ⏱️ Phase 1: Temporal Alignment
Once your raw files are in place, run the entire extraction and syncing pipeline with a single command.

Replace dataset_01 with the exact name of your folder:

```
docker compose run --rm sync ./run_temporal.sh dataset_01
```
**What This Does:**
- Extracts the ZED video completely, then extracts the RGB video completely.

- Analyzes every frame using a ResNet50 neural network.

- Aligns the frames in time using Dynamic Time Warping (DTW).

- Smooths any hardware clock drift via Savitzky-Golay filtering.

- Copies the perfectly synced frames into data/synced/dataset_01/.

## 📐 Phase 2: Spatial Alignment
Once the cameras are matched in time, the system calculates the optical and geometric relationship between the two cameras. This is a two-step process: curating frames, then running the calibration pipeline.

###  2.1 Record a Calibration Dataset
Before picking frames, record a dedicated calibration clip using the ChArUco board:

- Use the board printed from calib_io_charuco_200x150_8x11_15_11_DICT_4X4.pdf (A3 print, mounted flat on rigid cardboard).
- Hold the board 0.5 m to 1.2 m from both cameras simultaneously — this is critical for ZED detection.
- Move the board to cover all four corners of the frame, all four edges, and the center.
- At each position, tilt the board left, right, and toward the camera (~30° angles).
- Record for 3–4 minutes. Run Phase 1 on this recording to get synchronized frames.


###  2.2 Pick the Best Frames
Do not feed thousands of frames into the spatial calibrator. The algorithm requires a curated set of 30 to 50 perfect frame pairs with geometric diversity.
The Golden Rule: Every frame must show the ChArUco board razor-sharp with zero motion blur.
Open data/synced/<dataset>/sony_rgb/ and select frames covering these four categories:

- Edges (10 frames): Board flat and facing both cameras, positioned at the top, bottom, left, right, and four corners of the frame.
- Angles (10 frames): Board near center but heavily tilted backward, forward, left, and right (~30°).
- Depths (10 frames): Board at 0.5 m (close) and 1.0–1.2 m (medium), 5 frames each.
- Rotations (10 frames): Board flat but rotated clockwise and counter-clockwise like a steering wheel.

**The Copying Workflow:**
- Identify a good Sony frame (e.g., 00450.png) and copy it to data/picked_for_alignment/<dataset>/sony_rgb/.
- Find the exact same frame number in data/synced/<dataset>/zed_rgb/ and copy it to data/picked_for_alignment/<dataset>/zed_rgb/.
- Repeat until you have 30–50 matched pairs in both folders.

###  2.3 Run the Spatial Calibrator
Once you have your curated set of frames, run the spatial calibration pipeline:
```
docker compose run --rm sync ./run_spatial.sh dataset_01
```
**What This Does:**
- [1/4] inspect_detections.py — Runs ChArUco detection on every picked frame and saves annotated images to data/plots/<dataset>/spatial/detection_check/. Prints a per-camera detection rate. Review the annotated images and remove any frames where detection failed before proceeding.
- [2/4] stereo_calibrate.py — Runs individual camera calibration on each camera followed by full stereo calibration. Outputs data/json_output/<dataset>/spatial_calibration.json containing:

    - Intrinsic camera matrices (K) and distortion coefficients (D) for both cameras.
    - Extrinsic rotation matrix (R) and translation vector (T) from ZED to Sony coordinate space.
    - Ready-to-use UE5 LensOffset values in centimetres.

- [3/4] validate_calibration.py — Computes per-frame reprojection errors and saves analysis plots to data/plots/<dataset>/spatial/. The overall RMS error should be below 1.0 px for good calibration. Frames above 2.0 px are flagged for removal.
- [4/4] apply_calibration.py — Reads the calibration JSON and prints the final UE5 offset values directly to the terminal.

### 2.4 Diagnostic Tools
If detection is failing, run the diagnostic script to identify the correct board settings:
```
docker compose run --rm align python3 spatial_alignment/diagnose_detection.py --dataset dataset_01
```
This tests all supported ArUco dictionary variants and board size combinations and identifies the working configuration.

## Phase 3: Runtime Integration in Unreal Engine 5
- After spatial calibration is complete, apply the results in UE5:
- Open the spatial_calibration.json output file.
- In the UE5 level, select BP_TrackerAnchor.
- In the Details panel, find Lens Offset → Location.
- Enter the ue5_offset X, Y, Z values from the JSON (in centimetres).
- The virtual camera will now track from the Sony optical centre rather than the ZED optical centre, eliminating the spatial offset between the skeleton holdout and the live-action performer.

## Output Reference

| Path | Contents |
| --- | --- |
| `data/extracted/<dataset>/sony_rgb/` | All raw Sony frames extracted from the MP4 |
| `data/extracted/<dataset>/zed_rgb/` | All raw ZED left-eye frames extracted from the SVO2 |
| `data/extracted/<dataset>/zed_depth/` | All raw 16-bit ZED depth maps |
| `data/json_output/<dataset>/frame_mapping.json` | Raw DTW temporal alignment mapping |
| `data/json_output/<dataset>/smoothed_frame_mapping.json` | Savitzky-Golay filtered mapping |
| `data/plots/<dataset>/drift_comparison.jpg` | Raw vs. smoothed temporal drift graph |
| `data/synced/<dataset>/sony_rgb/` | Temporally aligned Sony frames |
| `data/synced/<dataset>/zed_rgb/` | Temporally aligned ZED frames |
| `data/synced/<dataset>/zed_depth/` | Temporally aligned depth maps |
| `data/picked_for_alignment/<dataset>/sony_rgb/` | Manually curated Sony calibration frames |
| `data/picked_for_alignment/<dataset>/zed_rgb/` | Matching manually curated ZED calibration frames |
| `data/json_output/<dataset>/spatial_calibration.json` | Intrinsics, extrinsics, and UE5 offsets |
| `data/json_output/<dataset>/detection_summary.json` | Per-frame ChArUco detection counts |
| `data/json_output/<dataset>/validation_report.json` | Per-frame reprojection error analysis |
| `data/plots/<dataset>/spatial/reprojection_error_analysis.png` | Error distribution and per-frame bar chart |
| `data/plots/<dataset>/spatial/corner_coverage_sony_rgb.png` | Sony corner detection density map |
| `data/plots/<dataset>/spatial/corner_coverage_zed_rgb.png` | ZED corner detection density map |
| `data/plots/<dataset>/spatial/detection_check/` | Annotated frames from inspect_detections.py |
| `data/plots/<dataset>/spatial/diagnosis/` | Diagnostic output from diagnose_detection.py |
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
Now that the cameras are matched in time, we calculate the optical math. Do not feed thousands of frames into the spatial calibrator—you only need a handful of perfect images.

### 2.1 Pick The Best Frames
Do **not** feed thousands of frames into the spatial calibrator. Feeding the algorithm too much data (especially redundant or blurry frames) will cause it to overfit, slow down, and output incorrect matrices. You only need a curated set of **30 to 50 perfect frame pairs**.

**The Golden Rule:** Every frame you pick must have **zero motion blur**. The black-and-white corners of the ChArUco board must be razor-sharp.

Open `data/synced/<dataset>/<camera_name>_rgb/` and pick frames that satisfy these four specific categories to ensure the math covers every physical extreme of the lens:
**NOTE:** This repo uses "sony_" as a prefix for for the RGB camera framessince a Sony camera was used during development. But the same process applies regardless of your RGB camera model. You can just change

* **The Edges (10 frames):** Keep the board flat and facing the camera, but pick frames where the board is pushed right up against the top, bottom, left, right edges, and the four corners of the frame. *(This maps the radial/barrel distortion of the lens).*
* **The Angles (10 frames):** Pick frames where the board is near the center, but heavily tilted backward, tilted forward, tilted left, and tilted right. *(This calculates focal length and perspective skew).*
* **The Depths (10 frames):** Pick 5 frames where the board is as close to the camera as possible (while still in focus and visible to both cameras), and 5 frames where the board is several meters away. *(This is critical for triangulating the Stereo X/Y/Z offset).*
* **The Rotations (10 frames):** Pick frames where the board is flat but spun like a steering wheel clockwise and counter-clockwise. *(This eliminates tangential distortion).*

**The Copying Workflow:**
1. Once you identify a perfect RGB frame (e.g., `00450.png`), copy it into `data/picked_for_alignment/<dataset>/<camera_name>_rgb/`.
2. Go to the ZED folder (`data/synced/<dataset>/zed_rgb/`), find that **exact same frame number** (`00450.png`), and copy it into `data/picked_for_alignment/<dataset>/zed_rgb/`.
3. Repeat until you have 30 to 50 perfectly matched pairs in your `picked_for_alignment` folders.

### 2.2 Run the Spatial Calibrator
Execute the spatial alignment container, pointing it to your curated dataset
```
docker compose run --rm align --dataset <dataset_name>
```
**What This Does:**
The script will analyze those specific frames and print four crucial matrices to your terminal:
- ***Camera Intrinsics*** The true focal length and optical center.
- ***Lens Distortion (D):*** The radial/tangential curvature of the glass.
- ***Extrinsic Rotation (R):*** The pitch/yaw/roll offset of the ZED compared to the RGB camera.
- ***Extrinsic Translation (T):*** The physical X/Y/Z offset in meters.
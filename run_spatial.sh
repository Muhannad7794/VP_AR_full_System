#!/bin/bash
# Spatial alignment pipeline: ZED 2i (left, rectified) -> Sony ZV-E10 II.
#
# Usage:
#   ./run_spatial.sh <dataset_name> [zed_intrinsics.json]
#
# The optional second argument is an intrinsics file written on the Jetson with
#   python3 spatial_alignment/print_zed_intrinsics.py --live --out zed_live.json
# When given, it replaces the intrinsics read from the dataset's SVO, so the
# calibration uses exactly the values of the live system.

if [ -z "$1" ]; then
  echo "Usage: ./run_spatial.sh <dataset_name> [zed_intrinsics.json]"
  echo "Example: ./run_spatial.sh dataset_04"
  exit 1
fi

DATASET=$1
ZED_JSON=$2

run_step() {
  local label=$1
  shift
  echo ""
  echo "$label"
  "$@"
  if [ $? -ne 0 ]; then
    echo ""
    echo "[ERROR] Step failed: $*"
    exit 1
  fi
}

echo "===================================================="
echo " Spatial Alignment Pipeline"
echo " Dataset: $DATASET"
echo "===================================================="

# Step 0: rectified ZED intrinsics from the ZED SDK (never estimated from images).
if [ -n "$ZED_JSON" ]; then
  run_step "[0/6] Reading ZED intrinsics from $ZED_JSON..." \
    python3 spatial_alignment/print_zed_intrinsics.py --dataset "$DATASET" --from-json "$ZED_JSON"
else
  run_step "[0/6] Reading ZED intrinsics from the SVO..." \
    python3 spatial_alignment/print_zed_intrinsics.py --dataset "$DATASET"
fi

# Step 1: board detection in every synced frame, usability and stillness flags.
run_step "[1/6] Detecting the board in all synced frames..." \
  python3 spatial_alignment/inspect_detections.py --dataset "$DATASET"

# Step 2: Sony intrinsics from Sony-only views, distortion model chosen on held-out views.
run_step "[2/6] Calibrating the Sony intrinsics..." \
  python3 spatial_alignment/calibrate_sony_intrinsics.py --dataset "$DATASET"

# Step 3: diverse stereo pairs, split into calibration and held-out sets.
run_step "[3/6] Selecting stereo pairs..." \
  python3 spatial_alignment/pick_frames.py --dataset "$DATASET"

# Step 4: extrinsics with both cameras' intrinsics fixed.
run_step "[4/6] Estimating the extrinsics..." \
  python3 spatial_alignment/stereo_calibrate.py --dataset "$DATASET"

# Step 5: held-out, distance-resolved and end-to-end (through ZED depth) validation.
run_step "[5/6] Validating..." \
  python3 spatial_alignment/validate_calibration.py --dataset "$DATASET"

# Step 6: values for Unreal Engine.
run_step "[6/6] Unreal Engine values..." \
  python3 spatial_alignment/apply_calibration.py --dataset "$DATASET" --ue5

echo ""
echo "===================================================="
echo " Spatial Alignment Pipeline Complete"
echo "===================================================="
echo ""
echo " Outputs (data/json_output/$DATASET/):"
echo "   zed_intrinsics.json       ZED SDK rectified intrinsics"
echo "   detections.json.gz        board detections, usability and stillness"
echo "   detection_summary.json    detection counts"
echo "   sony_intrinsics.json      Sony models, held-out errors, runtime displacement"
echo "   pair_selection.json       calibration and held-out pairs with pose features"
echo "   spatial_calibration.json  extrinsics, uncertainty and runtime values"
echo "   validation_report.json    held-out, distance-resolved and end-to-end errors"
echo " Plots: data/plots/$DATASET/spatial/"
echo ""
echo " Enter the values printed in step 6 into M_ZED_DepthReproject and"
echo " BP_TrackerAnchor (Lens Offset location and rotation)."
echo "===================================================="
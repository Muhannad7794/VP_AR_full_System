#!/bin/bash
# Spatial alignment pipeline: ZED 2i (left, rectified) -> Sony ZV-E10 II.
#
# Usage:
#   ./run_spatial.sh <dataset_name> [zed_intrinsics.json] [--sony-model NAME] [--from-step N] [--square-mm MM]
#
# --sony-model NAME  Fix the Sony lens model in step 2 instead of selecting it
#                    on held-out error (pinhole, k1, k1_k2, k1_k2_tangential,
#                    k1_k2_k3_tangential). Use pinhole for a camera that corrects
#                    its lens distortion internally.
# --from-step N      Start at step N (0-6). Earlier outputs are reused, e.g.
#                    --from-step 2 skips the ZED intrinsics and board detection.
# --square-mm MM     Measured square length of the printed board in mm. Overrides
#                    MEASURED_SQUARE_MM in calib_common.py for every step. After
#                    a change of the square length, start at step 2 or earlier:
#                    board detection (step 1) does not depend on it, every later
#                    step does.
#
# The optional second argument is an intrinsics file written on the Jetson with
#   python3 spatial_alignment/print_zed_intrinsics.py --live --out zed_live.json
# When given, it replaces the intrinsics read from the dataset's SVO, so the
# calibration uses exactly the values of the live system.

usage() {
  echo "Usage: ./run_spatial.sh <dataset_name> [zed_intrinsics.json] [--sony-model NAME] [--from-step N] [--square-mm MM]"
  echo "Example: ./run_spatial.sh dataset_05 --sony-model pinhole --from-step 2"
  exit 1
}

DATASET=""
ZED_JSON=""
SONY_MODEL=""
FROM_STEP=0
while [ $# -gt 0 ]; do
  case "$1" in
    --sony-model) SONY_MODEL=$2; shift 2 ;;
    --from-step) FROM_STEP=$2; shift 2 ;;
    --square-mm) export VP_BOARD_SQUARE_MM=$2; shift 2 ;;
    -*) echo "Unknown option: $1"; usage ;;
    *)
      if [ -z "$DATASET" ]; then DATASET=$1
      elif [ -z "$ZED_JSON" ]; then ZED_JSON=$1
      else usage
      fi
      shift ;;
  esac
done
[ -z "$DATASET" ] && usage

SONY_MODEL_ARGS=()
if [ -n "$SONY_MODEL" ]; then
  SONY_MODEL_ARGS=(--model "$SONY_MODEL")
fi

run_step() {
  local step=$1
  local label=$2
  shift 2
  if [ "$step" -lt "$FROM_STEP" ]; then
    echo ""
    echo "$label (skipped, --from-step $FROM_STEP)"
    return
  fi
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
if [ -n "$VP_BOARD_SQUARE_MM" ]; then
  echo " Board square: $VP_BOARD_SQUARE_MM mm (--square-mm)"
else
  echo " Board square: MEASURED_SQUARE_MM in spatial_alignment/calib_common.py"
fi
echo "===================================================="

# Step 0: rectified ZED intrinsics from the ZED SDK (never estimated from images).
if [ -n "$ZED_JSON" ]; then
  run_step 0 "[0/6] Reading ZED intrinsics from $ZED_JSON..." \
    python3 spatial_alignment/print_zed_intrinsics.py --dataset "$DATASET" --from-json "$ZED_JSON"
else
  run_step 0 "[0/6] Reading ZED intrinsics from the SVO..." \
    python3 spatial_alignment/print_zed_intrinsics.py --dataset "$DATASET"
fi

# Step 1: board detection in every synced frame, usability and stillness flags.
run_step 1 "[1/6] Detecting the board in all synced frames..." \
  python3 spatial_alignment/inspect_detections.py --dataset "$DATASET"

# Step 2: Sony intrinsics from Sony-only views, distortion model chosen on held-out views.
run_step 2 "[2/6] Calibrating the Sony intrinsics..." \
  python3 spatial_alignment/calibrate_sony_intrinsics.py --dataset "$DATASET" "${SONY_MODEL_ARGS[@]}"

# Step 3: diverse stereo pairs, split into calibration and held-out sets.
run_step 3 "[3/6] Selecting stereo pairs..." \
  python3 spatial_alignment/pick_frames.py --dataset "$DATASET"

# Step 4: extrinsics with both cameras' intrinsics fixed.
run_step 4 "[4/6] Estimating the extrinsics..." \
  python3 spatial_alignment/stereo_calibrate.py --dataset "$DATASET"

# Step 5: held-out, distance-resolved and end-to-end (through ZED depth) validation.
run_step 5 "[5/6] Validating..." \
  python3 spatial_alignment/validate_calibration.py --dataset "$DATASET"

# Step 6: values for Unreal Engine.
run_step 6 "[6/6] Unreal Engine values..." \
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
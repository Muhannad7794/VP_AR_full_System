#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: ./run_spatial.sh <dataset_name>"
  echo "Example: ./run_spatial.sh dataset_01"
  exit 1
fi

DATASET=$1

echo "===================================================="
echo " Starting Full Spatial Alignment Pipeline"
echo " Dataset: $DATASET"
echo "===================================================="

# step 1: filter raw frames for high detection quality
echo ""
echo "[1/5] Filtering raw frames for high detection quality..."
python3 spatial_alignment/inspect_detections.py --dataset "$DATASET"
if [ $? -ne 0 ]; then exit 1; fi

# step 2: pick a diverse set of spatial candidates for calibration
echo ""
echo "[2/5] Picking diverse spatial candidates..."
python3 spatial_alignment/pick_frames.py --dataset "$DATASET" --count 45
if [ $? -ne 0 ]; then exit 1; fi

# step 3: run stereo calibration to compute spatial alignment
echo ""
echo "[3/5] Running stereo calibration..."
python3 spatial_alignment/stereo_calibrate.py --dataset "$DATASET"

if [ $? -ne 0 ]; then
  echo ""
  echo "[ERROR] Stereo calibration failed."
  echo "        Review the detection output above."
  exit 1
fi

# step 4: validate calibration results with reprojection error analysis
echo ""
echo "[4/5] Validating calibration with reprojection error analysis..."
python3 spatial_alignment/validate_calibration.py --dataset "$DATASET"

if [ $? -ne 0 ]; then
  echo "[WARN] Validation step failed. Check validation_report.json"
fi

# step 5: extract UE5 offset values from calibration result
echo ""
echo "[5/5] Extracting UE5 offset values from calibration result..."
python3 spatial_alignment/apply_calibration.py --dataset "$DATASET" --ue5

echo ""
echo "===================================================="
echo " Spatial Alignment Pipeline Complete"
echo "===================================================="
echo ""
echo " Outputs:"
echo "   Calibration matrices : data/json_output/$DATASET/spatial_calibration.json"
echo "   Validation report    : data/json_output/$DATASET/validation_report.json"
echo "   Detection check      : data/json_output/$DATASET/detection_summary.json"
echo "   Plots                : data/plots/$DATASET/spatial/"
echo ""
echo " Next step:"
echo "   Copy the LensOffset X/Y/Z values printed above into"
echo "   BP_TrackerAnchor → Details → Lens Offset → Location in UE5."
echo "===================================================="
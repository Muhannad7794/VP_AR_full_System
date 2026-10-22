#!/bin/bash
# Temporal alignment for one dataset.
#
# Usage:
#   ./run_temporal.sh <dataset_name> [sync_frames.py options]
# Examples:
#   ./run_temporal.sh dataset_05
#   ./run_temporal.sh dataset_05 --window 30.0 262.0
#   ./run_temporal.sh dataset_05 --method dtw        (legacy ResNet18 + FastDTW)
#
# Every step must succeed before the next one runs, so a failed
# synchronisation never leaves an older mapping to be compiled.

if [ -z "$1" ]; then
  echo "Usage: ./run_temporal.sh <dataset_name> [sync_frames.py options]"
  echo "Example: ./run_temporal.sh dataset_05"
  exit 1
fi

DATASET=$1
shift

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
echo "Starting Full Temporal Pipeline for dataset: $DATASET"
echo "===================================================="

run_step "[1/4] Mapping Sony frames to ZED frames..." \
  python3 temporal_alignment/sync_frames.py --dataset "$DATASET" "$@"

run_step "[2/4] Smoothing the mapping..." \
  python3 temporal_alignment/smooth_frames_mapping.py --dataset "$DATASET"

run_step "[3/4] Comparing mappings..." \
  python3 temporal_alignment/compare_mappings.py --dataset "$DATASET"

run_step "[4/4] Compiling the synced dataset..." \
  python3 temporal_alignment/compile_dataset.py --dataset "$DATASET"

echo ""
echo "===================================================="
echo "Pipeline Complete! Fully synced data saved to data/synced/$DATASET"
echo " Check the plots in data/plots/$DATASET/sync/ before the spatial alignment."
echo "===================================================="

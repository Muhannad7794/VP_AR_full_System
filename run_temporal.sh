#!/bin/bash

# Ensure the user provided a dataset argument
if [ -z "$1" ]; then
  echo "Usage: ./run_temporal.sh <dataset_name>"
  echo "Example: ./run_temporal.sh dataset_01"
  exit 1
fi

DATASET=$1
echo "===================================================="
echo "Starting Full Temporal Pipeline for dataset: $DATASET"
echo "===================================================="

echo "[1/5] Extracting Frames from Raw Video..."
python3 temporal_alignment/extract_frames.py --dataset "$DATASET"

echo "[2/5] Syncing Frames via AI & DTW..."
python3 temporal_alignment/sync_frames.py --dataset "$DATASET"

echo "[3/5] Smoothing DTW Mapping..."
python3 temporal_alignment/smooth_frames_mapping.py --dataset "$DATASET"

echo "[4/5] Comparing Mappings..."
python3 temporal_alignment/compare_mappings.py --dataset "$DATASET"

echo "[5/5] Compiling Clean Dataset..."
python3 temporal_alignment/compile_dataset.py --dataset "$DATASET"

echo "===================================================="
echo "Pipeline Complete! Fully synced data saved to data/synced/$DATASET"
echo "====================================================" 

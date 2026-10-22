#!/bin/bash
# Frame extraction for one dataset (ZED SVO + Sony MP4 in data/raw/<dataset>/).
#
# Usage:
#   ./run_extract.sh <dataset_name> [extract_frames.py options]
# Examples:
#   ./run_extract.sh dataset_05
#   ./run_extract.sh dataset_05 --skip 2

if [ -z "$1" ]; then
  echo "Usage: ./run_extract.sh <dataset_name> [extract_frames.py options]"
  echo "Example: ./run_extract.sh dataset_05"
  exit 1
fi

DATASET=$1
shift
echo "===================================================="
echo "Starting Full extract Pipeline for dataset: $DATASET"
echo "===================================================="

echo "Extracting Frames from Raw Video..."
python3 ./extract_frames.py --dataset "$DATASET" "$@"
if [ $? -ne 0 ]; then
  echo ""
  echo "[ERROR] Extraction failed."
  exit 1
fi

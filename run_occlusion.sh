#!/bin/bash

# Ensure the user provided a dataset argument
if [ -z "$1" ]; then
  echo "Usage: ./run_occlusion.sh <dataset_name> [--confidence-threshold N] [--sample-count N]"
  echo "Example: ./run_occlusion.sh dataset_01"
  exit 1
fi

DATASET=$1
shift
echo "===================================================="
echo "Starting Occlusion Refinement Validation for dataset: $DATASET"
echo "===================================================="

python3 occlusion_refinement/validate_occlusion.py --dataset "$DATASET" "$@"

echo "===================================================="
echo "Occlusion Refinement Validation Complete for $DATASET"
echo "Outputs saved to data/plots/$DATASET/occlusion/ and data/json_output/$DATASET/"
echo "===================================================="
#!/bin/bash
# ==============================================================================
# Script: run_filtering.sh
# Purpose: Executes the Phase 3 Kinematic Extraction and Filtering pipeline.
#          Designed to be run inside the dockerized 'filter' service environment.
# Usage: docker compose run --rm filter ./run_filtering.sh <dataset_name>
# ==============================================================================

if [ -z "$1" ]; then
    echo "ERROR: Dataset name required."
    echo "Usage: ./run_filtering.sh <dataset_name>"
    exit 1
fi

DATASET=$1
echo "================================================================="
echo "Initializing Kinematic Filtering Pipeline for dataset: $DATASET"
echo "================================================================="

# Execute the Python pipeline directly (environment is already containerized)
# python3 /VP_AR_full_System_dockerized/skeleton_filtering/extract_kinematics.py --dataset "$DATASET"

# Check if extraction was successful before proceeding to validation
if [ $? -eq 0 ]; then
    # Inside run_filtering.sh
    python3 /VP_AR_full_System_dockerized/skeleton_filtering/validate_kinematics.py --dataset "$DATASET" "$2" "$3" "$4" "$5"
    
    if [ $? -eq 0 ]; then
        echo "================================================================="
        echo "Phase 3 Pipeline Complete."
        echo "Validation plots saved to: data/plots/$DATASET/kinematics/"
        echo "================================================================="
    else
        echo "================================================================="
        echo "ERROR: Kinematic validation and plotting failed."
        echo "================================================================="
        exit 1
    fi
else
    echo "================================================================="
    echo "ERROR: Kinematic extraction failed."
    echo "================================================================="
    exit 1
fi
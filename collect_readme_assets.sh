#!/bin/bash
# Copies the plots shown in README.md from data/plots/ (not tracked by Git)
# into docs/assets/ (tracked), so that GitHub can display them.
#
# Usage:
#   ./collect_readme_assets.sh [spatial_dataset] [kinematics_dataset] [occlusion_dataset]
# Defaults:
#   ./collect_readme_assets.sh dataset_05 dataset_01 dataset_01
#
# The spatial dataset also provides the temporal alignment plots, because the
# README shows both phases for the same recording. Missing plots are reported
# and skipped; the script never deletes anything.

SPATIAL=${1:-dataset_05}
KINEMATICS=${2:-dataset_01}
OCCLUSION=${3:-dataset_01}

MISSING=0

copy() {
  local src=$1
  local dst=$2
  if [ -f "$src" ]; then
    mkdir -p "$(dirname "$dst")"
    cp "$src" "$dst"
    echo "  copied  $src"
  else
    echo "  MISSING $src"
    MISSING=$((MISSING + 1))
  fi
}

echo "Temporal alignment ($SPATIAL)"
for f in sync_luma sync_correlation sync_drift sync_residuals; do
  copy "data/plots/$SPATIAL/sync/$f.png" "docs/assets/temporal/$f.png"
done
copy "data/plots/$SPATIAL/drift_comparison.jpg" "docs/assets/temporal/drift_comparison.jpg"

echo "Spatial alignment ($SPATIAL)"
for f in detection_timeline sony_intrinsics_models sony_intrinsics_coverage \
         sony_runtime_displacement pair_selection_features reprojection_error_analysis \
         error_vs_distance bias_vs_inverse_distance depth_scale_check \
         error_across_sony_frame corner_coverage_sony_rgb corner_coverage_zed_rgb; do
  copy "data/plots/$SPATIAL/spatial/$f.png" "docs/assets/spatial/$f.png"
done

echo "Kinematics ($KINEMATICS)"
for f in 1Euro_noise_reduction_right_wrist_x 1Euro_noise_reduction_spine_2_y lma_expansiveness; do
  copy "data/plots/$KINEMATICS/kinematics/$f.png" "docs/assets/kinematics/$f.png"
done

echo "Occlusion refinement ($OCCLUSION)"
for f in confidence_gating_rejection_rate rejection_frequency_map; do
  copy "data/plots/$OCCLUSION/occlusion/$f.png" "docs/assets/occlusion/$f.png"
done
# First sample figure, saved under a fixed name for the README.
SAMPLE=$(ls data/plots/$OCCLUSION/occlusion/samples/sample_*.png 2>/dev/null | head -n 1)
copy "${SAMPLE:-data/plots/$OCCLUSION/occlusion/samples/sample_*.png}" "docs/assets/occlusion/sample_example.png"

echo ""
if [ "$MISSING" -gt 0 ]; then
  echo "$MISSING plot(s) missing. Run the phase that produces them, or pass another dataset name."
  exit 1
fi
echo "All README figures are in docs/assets/. Add them with: git add docs/assets"
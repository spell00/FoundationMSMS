#!/bin/bash
# Batch download and process datasets with cleanup

set -e  # Exit on error

# Configuration
BASE_DIR="data"
MAX_PRIORITY=5  # Download only high-priority datasets
PARALLEL=8      # Number of parallel downloads
WORKERS=4       # Number of parallel workers for file conversion
CLEANUP_RAW=false
CLEANUP_MZML=false

echo "=========================================="
echo "Foundation LCMS Batch Processing"
echo "=========================================="
echo ""

# Step 1: Batch download with parallel downloads
echo "📥 Step 1: Downloading datasets (parallel=${PARALLEL})"
python -u -m foundationmsms.preprocessing batch-download \
    --max-priority ${MAX_PRIORITY} \
    --parallel ${PARALLEL} \
    --base-dir ${BASE_DIR} \
    --skip-existing

echo ""
echo "✅ Download complete!"
echo ""

# Step 2: Process each dataset through the pipeline
echo "🔄 Step 2: Processing datasets through pipeline"
echo ""

for dataset_path in ${BASE_DIR}/raw/pride/*/ ${BASE_DIR}/raw/massive/*/; do
    if [ -d "$dataset_path" ]; then
        dataset_name=$(basename "$dataset_path")
        echo "Processing: $dataset_name"
        
        # Check if there are RAW files
        raw_count=$(find "$dataset_path" -iname "*.raw" | wc -l)
        if [ "$raw_count" -gt 0 ]; then
            output_dir="${BASE_DIR}"
            
            cleanup_flags=""
            if [ "$CLEANUP_RAW" = true ]; then
                cleanup_flags="$cleanup_flags --cleanup-raw"
            fi
            if [ "$CLEANUP_MZML" = true ]; then
                cleanup_flags="$cleanup_flags --cleanup-mzml"
            fi
            
            python -u -m foundationmsms.preprocessing pipeline \
                --dataset-dir "$dataset_path" \
                --output-base "$output_dir" \
                $cleanup_flags \
                --mz-bin 1.0 \
                --mz-parent-bin 1.0 \
                --rt-bin-sec 1.0 \
                --window-sec 30 \
                --stride-sec 15 \
                --workers ${WORKERS}
            
            echo "✅ Completed: $dataset_name"
        else
            echo "⏭️  Skipping $dataset_name - no RAW files found"
        fi
        echo ""
    fi
done

echo ""
echo "=========================================="
echo "✅ All datasets processed!"
echo "=========================================="
echo ""
echo "Results:"
echo "  - Raw data: ${BASE_DIR}/raw/"
echo "  - mzML data: ${BASE_DIR}/mzml/"
echo "  - Voxel data: ${BASE_DIR}/voxel/"
echo "  - Windowed data: ${BASE_DIR}/windows/"
echo ""

#!/bin/bash

# ==============================================================================
# Script to parse complex layout directories and copy mean_info.npy files
# ==============================================================================

# Base destination directory
DEST_BASE_DIR="results/prior_samples"
mkdir -p "$DEST_BASE_DIR"

echo "Starting file extraction and copy process..."
echo "----------------------------------------"

# Loop through all matching directories
for dir in ../WH_ABMs/results/prior_samples/ABC_*/; do
    
    # Check if any directories actually match the wildcard
    [ -d "$dir" ] || { echo "No matching directories found."; exit 1; }
    
    # Extract just the folder name from the full path
    folder=$(basename "$dir")
    
    # Fixed Regex pattern to break down the highly structured layout
    PATTERN="^ABC_(rm_rp|rm_pint_rp|rm_rp_a0_a1|rm_pint_rp_a0_a1)_uniform_sampling_(ABM|PDE)_(WH|artificial)_(dens_[0-9]+|simulation_[0-9]+)(_pred_final)?(_individual_testing)?$"
    
    if [[ $folder =~ $PATTERN ]]; then
        # Map the regex capture groups to descriptive variables
        model_name="${BASH_REMATCH[1]}"
        model_type="${BASH_REMATCH[2]}"
        data_type="${BASH_REMATCH[3]}"
        data_id="${BASH_REMATCH[4]}"
        fit_type="${BASH_REMATCH[5]}"
        f_type="${BASH_REMATCH[6]}"
        
        #src_file="${dir}mean_info.npy"
        src_file="${dir}"
        
        # Verify the target file exists before trying to copy it
        if [ -f "$src_file" ]; then
            # 1. Define the specific subfolder path for this dataset
            sub_dir="${DEST_BASE_DIR}/ABC_${model_name}_uniform_sampling_${model_type}_${data_type}_${data_id}${fit_type}${f_type}"
            
            # 2. CREATE THE NEW DIRECTORY YET TO EXIST
            mkdir -p "$sub_dir"
            
#             # 3. Define the final file destination path
#             dest_file="${sub_dir}/mean_info.npy"
            
#             # Perform the copy
#             cp "$src_file" "$dest_file"
#             echo "SUCCESS: Created folder and copied mean_info.npy for $folder"
        else
            echo "WARNING: mean_info.npy missing in $folder"
        fi
    else
        echo "SKIPPED: $folder (Did not match expected naming convention)"
    fi
done

echo "----------------------------------------"
echo "Done! Check your files in $DEST_BASE_DIR"
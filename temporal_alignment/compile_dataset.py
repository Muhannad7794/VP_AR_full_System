import json
import os
import glob
import shutil
from tqdm import tqdm


def compile_dataset():
    # 1. Define paths
    json_path = "data/smoothed_frame_mapping.json"

    src_sony = "data/extracted/sony_rgb"
    src_zed_rgb = "data/extracted/zed_rgb"
    src_zed_depth = "data/extracted/zed_depth"

    # create a new parent directory for the clean dataset
    dst_sony = "data/synced/sony_rgb"
    dst_zed_rgb = "data/synced/zed_rgb"
    dst_zed_depth = "data/synced/zed_depth"

    os.makedirs(dst_sony, exist_ok=True)
    os.makedirs(dst_zed_rgb, exist_ok=True)
    os.makedirs(dst_zed_depth, exist_ok=True)

    # 2. Load the smoothed mapping
    print(f"Loading mapping from {json_path}...")
    with open(json_path, "r") as f:
        mapping = json.load(f)

    # 3. Get sorted lists of the original files
    # Using glob ensures grabbing the data in the exact order of extraction
    sony_files = sorted(glob.glob(os.path.join(src_sony, "*.png")))
    zed_rgb_files = sorted(glob.glob(os.path.join(src_zed_rgb, "*.png")))
    zed_depth_files = sorted(glob.glob(os.path.join(src_zed_depth, "*.png")))

    print("Copying and renaming files to form the unified dataset...")

    # 4. Iterate through the JSON and physically copy the files
    # wrap with mapping.items() in tqdm to get a progress bar in the console
    for sony_frame_str, zed_frame_int in tqdm(mapping.items()):
        # Convert 1-based JSON keys back to 0-based list indices
        sony_idx = int(sony_frame_str) - 1
        zed_idx = int(zed_frame_int) - 1

        # Safety check: ignore if index goes out of bounds
        if (
            sony_idx >= len(sony_files)
            or zed_idx >= len(zed_rgb_files)
            or zed_idx >= len(zed_depth_files)
        ):
            continue

        # Get the actual source file paths
        s_file = sony_files[sony_idx]
        z_rgb_file = zed_rgb_files[zed_idx]
        z_depth_file = zed_depth_files[zed_idx]

        # Create a clean, unified filename padded to 5 digits (e.g., "00001.png")
        new_filename = f"{int(sony_frame_str):05d}.png"

        # Copy the files into the new synced folders
        # The script uses shutil.copy2 to preserve the original file metadata
        shutil.copy2(s_file, os.path.join(dst_sony, new_filename))
        shutil.copy2(z_rgb_file, os.path.join(dst_zed_rgb, new_filename))
        shutil.copy2(z_depth_file, os.path.join(dst_zed_depth, new_filename))

    print("\nDataset successfully compiled! Ready for ML training.")


if __name__ == "__main__":
    compile_dataset()

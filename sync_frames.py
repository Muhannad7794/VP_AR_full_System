"""
This file removes the first 51 frames of the Sony footage then indexes the \n
remaining file starting from 0 so the frames sync from both the ZED and the\n
Sony cameras.\n"""

import os
import glob

# Paths as they appear INSIDE the Docker container
SONY_FOLDER = "/VP_AR_full_System_dockerized/data/extracted/sony_rgb/"
ZED_RGB_FOLDER = "/VP_AR_full_System_dockerized/data/extracted/zed_rgb/"
ZED_DEPTH_FOLDER = "/VP_AR_full_System_dockerized/data/extracted/zed_depth/"

OFFSET = 51


def sync_folders():
    # 1. Get sorted lists of all files
    sony_files = sorted(glob.glob(os.path.join(SONY_FOLDER, "*.png")))
    zed_rgb_files = sorted(glob.glob(os.path.join(ZED_RGB_FOLDER, "*.png")))
    zed_depth_files = sorted(glob.glob(os.path.join(ZED_DEPTH_FOLDER, "*.png")))

    print(f"Found {len(sony_files)} Sony frames and {len(zed_rgb_files)} ZED frames.")

    # 2. Delete the leading Sony frames that have no ZED match
    print(f"Deleting first {OFFSET} unmatched Sony frames...")
    for i in range(OFFSET):
        os.remove(sony_files[i])

    # 3. Rename the remaining Sony frames to start from 00000
    print("Re-indexing Sony frames...")
    remaining_sony = sony_files[OFFSET:]
    for idx, old_path in enumerate(remaining_sony):
        new_name = f"sony_rgb_{idx:05d}.png"
        os.rename(old_path, os.path.join(SONY_FOLDER, new_name))

    # 4. Trim the END of the ZED folders
    # Because we shifted Sony, we now have 51 'orphaned' ZED frames at the very end
    print(f"Trimming last {OFFSET} unmatched ZED frames...")
    num_to_keep = len(remaining_sony)

    for i in range(num_to_keep, len(zed_rgb_files)):
        os.remove(zed_rgb_files[i])
        os.remove(zed_depth_files[i])

    print(f"Done! You now have {num_to_keep} perfectly synchronized frame sets.")


if __name__ == "__main__":
    sync_folders()

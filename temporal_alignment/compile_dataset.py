# temporal_alignment/compile_dataset.py
import json
import os
import glob
import shutil
import argparse
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, required=True, help="Name of the dataset folder"
    )
    return parser.parse_args()


def compile_dataset():
    args = parse_arguments()
    json_path = f"data/json_output/{args.dataset}/smoothed_frame_mapping.json"

    src_sony = f"data/extracted/{args.dataset}/sony_rgb"
    src_zed_rgb = f"data/extracted/{args.dataset}/zed_rgb"
    src_zed_depth = f"data/extracted/{args.dataset}/zed_depth"
    src_zed_confidence = f"data/extracted/{args.dataset}/zed_confidence"

    dst_sony = f"data/synced/{args.dataset}/sony_rgb"
    dst_zed_rgb = f"data/synced/{args.dataset}/zed_rgb"
    dst_zed_depth = f"data/synced/{args.dataset}/zed_depth"
    dst_zed_confidence = f"data/synced/{args.dataset}/zed_confidence"

    os.makedirs(dst_sony, exist_ok=True)
    os.makedirs(dst_zed_rgb, exist_ok=True)
    os.makedirs(dst_zed_depth, exist_ok=True)

    print(f"Loading mapping from {json_path}...")
    with open(json_path, "r") as f:
        mapping = json.load(f)

    sony_files = sorted(glob.glob(os.path.join(src_sony, "*.png")))
    zed_rgb_files = sorted(glob.glob(os.path.join(src_zed_rgb, "*.png")))
    zed_depth_files = sorted(glob.glob(os.path.join(src_zed_depth, "*.png")))

    # Confidence extraction was added after some earlier datasets were
    # already captured, so an extracted dataset may or may not have it.
    # Rather than assume it exists and fail confusingly (an empty file
    # list would otherwise make every frame fail the bounds check below
    # and silently compile an empty dataset), check explicitly and compile
    # without a confidence channel if it is absent, so older datasets
    # still work exactly as before.
    zed_confidence_files = sorted(glob.glob(os.path.join(src_zed_confidence, "*.png")))
    has_confidence = len(zed_confidence_files) > 0
    if has_confidence:
        os.makedirs(dst_zed_confidence, exist_ok=True)
    else:
        print(
            f"[WARN] No zed_confidence frames found at {src_zed_confidence} -- "
            f"this dataset was likely extracted before confidence retrieval "
            f"was added. Compiling without a confidence channel."
        )

    print(f"Copying files to form unified dataset at data/synced/{args.dataset}/...")

    for sony_frame_str, zed_frame_int in tqdm(mapping.items()):
        sony_idx = int(sony_frame_str) - 1
        zed_idx = int(zed_frame_int) - 1

        if (
            sony_idx >= len(sony_files)
            or zed_idx >= len(zed_rgb_files)
            or zed_idx >= len(zed_depth_files)
            or (has_confidence and zed_idx >= len(zed_confidence_files))
        ):
            continue

        new_filename = f"{int(sony_frame_str):05d}.png"

        shutil.copy2(sony_files[sony_idx], os.path.join(dst_sony, new_filename))
        shutil.copy2(zed_rgb_files[zed_idx], os.path.join(dst_zed_rgb, new_filename))
        shutil.copy2(
            zed_depth_files[zed_idx], os.path.join(dst_zed_depth, new_filename)
        )
        if has_confidence:
            shutil.copy2(
                zed_confidence_files[zed_idx],
                os.path.join(dst_zed_confidence, new_filename),
            )

    print(f"\nDataset {args.dataset} successfully compiled!")


if __name__ == "__main__":
    compile_dataset()

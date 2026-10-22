# temporal_alignment/compile_dataset.py
"""
Temporal alignment, last step: copies the mapped frame pairs into
data/synced/<dataset>/, named by the 1-based Sony frame number.

Changes from the earlier version
--------------------------------
* The destination folders of the dataset are emptied first (--keep-existing
  disables this). Otherwise frames from an earlier run with a different
  mapping or window would stay in data/synced and be picked up by the
  spatial alignment.
* When sync_frames.py wrote frame_mapping_detail.csv, a synced_index.csv is
  written next to the frames: synced file name, extracted Sony and ZED
  indices, the times of both frames and their residual.
"""

import argparse
import csv
import glob
import json
import os
import shutil

from tqdm import tqdm

SUBDIRS = ("sony_rgb", "zed_rgb", "zed_depth", "zed_confidence")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, required=True, help="Name of the dataset folder"
    )
    parser.add_argument(
        "--mapping",
        type=str,
        default=None,
        help="Mapping JSON to compile (default: data/json_output/<dataset>/smoothed_frame_mapping.json)",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Do not empty data/synced/<dataset>/ before copying",
    )
    return parser.parse_args()


def compile_dataset():
    args = parse_arguments()
    json_path = (
        args.mapping or f"data/json_output/{args.dataset}/smoothed_frame_mapping.json"
    )
    detail_path = f"data/json_output/{args.dataset}/frame_mapping_detail.csv"

    src = {k: f"data/extracted/{args.dataset}/{k}" for k in SUBDIRS}
    dst = {k: f"data/synced/{args.dataset}/{k}" for k in SUBDIRS}

    if not args.keep_existing:
        removed = 0
        for k in SUBDIRS:
            for f in glob.glob(os.path.join(dst[k], "*.png")):
                os.remove(f)
                removed += 1
        if removed:
            print(
                f"Removed {removed} frames of an earlier compilation from data/synced/{args.dataset}/"
            )

    print(f"Loading mapping from {json_path}...")
    with open(json_path, "r") as f:
        mapping = json.load(f)

    files = {k: sorted(glob.glob(os.path.join(src[k], "*.png"))) for k in SUBDIRS}

    # Confidence extraction was added after some earlier datasets were
    # already captured, so an extracted dataset may or may not have it.
    # Rather than assume it exists and fail confusingly (an empty file
    # list would otherwise make every frame fail the bounds check below
    # and silently compile an empty dataset), check explicitly and compile
    # without a confidence channel if it is absent, so older datasets
    # still work exactly as before.
    has_confidence = len(files["zed_confidence"]) > 0
    if not has_confidence:
        print(
            f"[WARN] No zed_confidence frames found at {src['zed_confidence']} -- "
            f"this dataset was likely extracted before confidence retrieval "
            f"was added. Compiling without a confidence channel."
        )
    for k in SUBDIRS:
        if k != "zed_confidence" or has_confidence:
            os.makedirs(dst[k], exist_ok=True)

    detail = {}
    if os.path.isfile(detail_path):
        with open(detail_path, newline="") as f:
            for row in csv.DictReader(f):
                detail[int(row["sony_saved_index"]) + 1] = row

    print(f"Copying files to form unified dataset at data/synced/{args.dataset}/...")
    copied, skipped = 0, 0
    index_rows = []
    for sony_frame_str, zed_frame_int in tqdm(mapping.items()):
        sony_idx = int(sony_frame_str) - 1
        zed_idx = int(zed_frame_int) - 1

        if (
            sony_idx >= len(files["sony_rgb"])
            or zed_idx >= len(files["zed_rgb"])
            or zed_idx >= len(files["zed_depth"])
            or (has_confidence and zed_idx >= len(files["zed_confidence"]))
        ):
            skipped += 1
            continue

        new_filename = f"{int(sony_frame_str):05d}.png"
        shutil.copy2(
            files["sony_rgb"][sony_idx], os.path.join(dst["sony_rgb"], new_filename)
        )
        shutil.copy2(
            files["zed_rgb"][zed_idx], os.path.join(dst["zed_rgb"], new_filename)
        )
        shutil.copy2(
            files["zed_depth"][zed_idx], os.path.join(dst["zed_depth"], new_filename)
        )
        if has_confidence:
            shutil.copy2(
                files["zed_confidence"][zed_idx],
                os.path.join(dst["zed_confidence"], new_filename),
            )
        copied += 1

        row = detail.get(int(sony_frame_str))
        index_rows.append(
            [
                new_filename,
                sony_idx,
                zed_idx,
                row["sony_time_s"] if row else "",
                row["zed_time_s"] if row else "",
                row["residual_ms"] if row else "",
            ]
        )

    with open(f"data/synced/{args.dataset}/synced_index.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "synced_file",
                "sony_extracted_index",
                "zed_extracted_index",
                "sony_time_s",
                "zed_time_s",
                "residual_ms",
            ]
        )
        w.writerows(index_rows)

    print(
        f"\nDataset {args.dataset} successfully compiled: {copied} pairs"
        + (
            f", {skipped} mapping entries skipped (index out of range)"
            if skipped
            else ""
        )
    )


if __name__ == "__main__":
    compile_dataset()

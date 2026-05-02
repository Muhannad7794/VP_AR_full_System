# temporal_alignment/compare_mappings.py
import json
import os
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, required=True, help="Name of the dataset folder"
    )
    return parser.parse_args()


def compare_mappings():
    args = parse_arguments()
    print(f"Comparing JSON maps for {args.dataset}...")

    raw_json = f"data/json_output/{args.dataset}/frame_mapping.json"
    smooth_json = f"data/json_output/{args.dataset}/smoothed_frame_mapping.json"

    txt_dir = f"data/txt_output/{args.dataset}"
    os.makedirs(txt_dir, exist_ok=True)
    output_file = os.path.join(txt_dir, "mapping_comparison.txt")

    with open(raw_json, "r") as f:
        raw_map = json.load(f)

    with open(smooth_json, "r") as f:
        smooth_map = json.load(f)

    mismatches = 0
    with open(output_file, "w") as out:
        for sony_frame in sorted(raw_map.keys(), key=int):
            raw_zed = raw_map[sony_frame]
            smooth_zed = smooth_map[str(sony_frame)]

            sony_str = f"{int(sony_frame):05d}"
            raw_str = f"{raw_zed:05d}"

            if raw_zed == smooth_zed:
                out.write(f"Sony Frame {sony_str} -> ZED Frame {raw_str}\n\n")
            else:
                smooth_str = f"{smooth_zed:05d}"
                out.write(
                    f"Sony Frame {sony_str} -> ZED Frame {raw_str}-> Smoothed {smooth_str}\n\n"
                )
                mismatches += 1

    print(f"Scan complete! {mismatches} frames out of {len(raw_map)} were smoothed.")
    print(f"Full log saved to: {output_file}")


if __name__ == "__main__":
    compare_mappings()

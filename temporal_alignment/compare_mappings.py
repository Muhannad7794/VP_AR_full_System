import json


def compare_mappings():
    print("Loading JSON maps for comparison...")

    # Load the raw DTW mapping
    with open("data/frame_mapping.json", "r") as f:
        raw_map = json.load(f)

    # Load the Savitzky-Golay smoothed mapping
    with open("data/smoothed_frame_mapping.json", "r") as f:
        smooth_map = json.load(f)

    output_file = "data/mapping_comparison.txt"
    mismatches = 0

    # Write the formatted output to a text file
    with open(output_file, "w") as out:
        for sony_frame in sorted(raw_map.keys(), key=int):
            raw_zed = raw_map[sony_frame]
            smooth_zed = smooth_map[str(sony_frame)]

            # Zero-pad to 5 digits
            sony_str = f"{int(sony_frame):05d}"
            raw_str = f"{raw_zed:05d}"

            if raw_zed == smooth_zed:
                # If they match, print the standard mapping
                out.write(f"Sony Frame {sony_str} -> ZED Frame {raw_str}\n\n")
            else:
                # If they differ, append the -> Smoothed XXXXX text
                smooth_str = f"{smooth_zed:05d}"
                out.write(
                    f"Sony Frame {sony_str} -> ZED Frame {raw_str}-> Smoothed {smooth_str}\n\n"
                )
                mismatches += 1

    print(f"Scan complete! {mismatches} frames out of {len(raw_map)} were smoothed.")
    print(f"Full log saved successfully to: {output_file}")


if __name__ == "__main__":
    compare_mappings()

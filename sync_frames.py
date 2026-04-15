import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import glob
import json
from fastdtw import fastdtw
from scipy.spatial.distance import cosine


def get_resnet_features(frame_folder, model, transform, device):
    """
    Extracts deep semantic features from all images in a folder using ResNet50.
    """
    frame_paths = sorted(glob.glob(os.path.join(frame_folder, "*.png")))
    features_list = []

    print(f"Extracting AI features from {len(frame_paths)} frames in {frame_folder}...")

    # Set model to evaluation mode
    model.eval()

    # We don't need to calculate gradients for feature extraction
    with torch.no_grad():
        for path in frame_paths:
            try:
                # Load image, convert to RGB
                img = Image.open(path).convert("RGB")

                # Apply standard ResNet preprocessing
                img_t = transform(img).unsqueeze(0).to(device)

                # Extract features
                features = model(img_t)

                # Flatten to 1D array and move back to CPU
                features_np = features.cpu().numpy().flatten()
                features_list.append(features_np)

            except Exception as e:
                print(f"Error processing {path}: {e}")
                # Append a zero vector if an image fails to load
                features_list.append(np.zeros(2048))

    return np.array(features_list), frame_paths


def synchronize_datasets():
    sony_folder = "data/extracted/sony_rgb/"
    zed_folder = "data/extracted/zed_rgb/"
    json_output_path = "data/frame_mapping.json"

    # --- Setup PyTorch and ResNet50 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    # Load pre-trained ResNet50
    resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # Remove the final classification layer (fc) to get the raw 2048-d feature vector
    resnet_feature_extractor = torch.nn.Sequential(*list(resnet50.children())[:-1]).to(
        device
    )

    # Standard ImageNet preprocessing transforms
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # --- Step 1: Extract Features ---
    sony_features, _ = get_resnet_features(
        sony_folder, resnet_feature_extractor, preprocess, device
    )
    zed_features, _ = get_resnet_features(
        zed_folder, resnet_feature_extractor, preprocess, device
    )

    if len(sony_features) == 0 or len(zed_features) == 0:
        print("Error: Could not extract features. Check your folder paths.")
        return

    print("\nCalculating sequence alignment using DTW and Cosine Similarity...")
    print("This may take a few minutes depending on dataset size...")

    # --- Step 2: Dynamic Time Warping (DTW) ---
    # Using 'cosine' distance. Cosine distance is 1 - Cosine Similarity.
    # Lower distance = higher similarity.
    distance, path = fastdtw(sony_features, zed_features, dist=cosine)

    print(f"DTW Alignment Complete. Total Distance: {distance:.4f}")

    # --- Step 3: Create Mapping and Save to JSON ---
    frame_mapping = {}

    # DTW returns a path of index pairs. This keeps the first matched ZED frame for each Sony frame
    for sony_idx, zed_idx in path:
        actual_sony_frame = int(sony_idx + 1)
        actual_zed_frame = int(zed_idx + 1)

        if actual_sony_frame not in frame_mapping:
            frame_mapping[actual_sony_frame] = actual_zed_frame

    # Print a sample of the mapping to the console
    print("\nSample of the synchronized mapping:")
    keys = list(frame_mapping.keys())
    for i in range(0, len(keys), max(1, len(keys) // 10)):
        sony_f = keys[i]
        print(f"Sony Frame {sony_f:05d} -> ZED Frame {frame_mapping[sony_f]:05d}")

    # Save to JSON
    try:
        with open(json_output_path, "w") as json_file:
            json.dump(frame_mapping, json_file, indent=4)
        print(f"\nSUCCESS: Frame mapping successfully saved to -> {json_output_path}")
    except Exception as e:
        print(f"\nFAILED to save JSON: {e}")


if __name__ == "__main__":
    synchronize_datasets()

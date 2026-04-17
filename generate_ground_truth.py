import os
import glob
import torch
from torchvision.models.segmentation import (
    deeplabv3_resnet101,
    DeepLabV3_ResNet101_Weights,
)
from tqdm import tqdm


def generate_masks():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading heavy Teacher model (ResNet101) on {DEVICE}...")

    # Load the most accurate, pre-trained weights available
    weights = DeepLabV3_ResNet101_Weights.DEFAULT
    model = deeplabv3_resnet101(weights=weights).to(DEVICE)
    model.eval()  # Set to evaluation/inference mode

    rgb_files = sorted(glob.glob("data/preprocessed_tensors/sony_rgb/*.pt"))
    out_dir = "data/preprocessed_tensors/ground_truth_masks"
    os.makedirs(out_dir, exist_ok=True)

    print(f"Generating perfect human masks for {len(rgb_files)} frames...")

    with torch.no_grad():  # Disable gradients to save VRAM and speed up processing
        for f in tqdm(rgb_files):
            # Load the pre-normalized RGB tensor and add a batch dimension: [1, 3, 360, 640]
            rgb_tensor = torch.load(f).to(DEVICE).unsqueeze(0)

            # Run the heavy AI
            output = model(rgb_tensor)["out"]
            predictions = output.argmax(1).squeeze(
                0
            )  # Get the most likely class per pixel

            # In the COCO dataset that this model was trained on, 'Person' is class 15.
            # We create a mask where 1 = Person, 0 = Everything else (Desks, walls, etc.)
            binary_mask = (predictions == 15).long()

            # Save the perfect mask
            out_path = os.path.join(out_dir, os.path.basename(f))
            torch.save(binary_mask.cpu(), out_path)

    print(f"\nSuccess! All Teacher masks saved to {out_dir}")


if __name__ == "__main__":
    generate_masks()

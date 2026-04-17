import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from tqdm import tqdm

BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FastTensorDataset(Dataset):
    def __init__(self, tensor_dir):
        self.rgb_files = sorted(glob.glob(os.path.join(tensor_dir, "sony_rgb", "*.pt")))
        self.mask_files = sorted(
            glob.glob(os.path.join(tensor_dir, "ground_truth_masks", "*.pt"))
        )

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        rgb_tensor = torch.load(self.rgb_files[idx])
        binary_mask = torch.load(self.mask_files[idx])
        return rgb_tensor, binary_mask


def get_model():
    print("Loading lightweight Student model (MobileNetV3)...")
    model = models.segmentation.deeplabv3_mobilenet_v3_large(weights="DEFAULT")
    model.classifier = DeepLabHead(960, 2)
    return model.to(DEVICE)


def train():
    os.makedirs("data/models", exist_ok=True)
    dataset = FastTensorDataset("data/preprocessed_tensors")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    model = get_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Starting Domain Adaptation on {DEVICE}...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for rgb, mask in loop:
            rgb, mask = rgb.to(DEVICE), mask.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(rgb)["out"]
            loss = criterion(outputs, mask)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

    print("\nTraining Complete! Saving PyTorch Weights...")
    torch.save(model.state_dict(), "data/models/studio_occlusion_model.pth")
    print("Saved as data/models/studio_occlusion_model.pth")


if __name__ == "__main__":
    train()

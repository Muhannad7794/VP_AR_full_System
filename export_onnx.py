import torch
import onnx
from torchvision import models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def export_saved_model():
    print("1. Loading the MobileNetV3 Student Architecture...")
    model = models.segmentation.deeplabv3_mobilenet_v3_large(weights=None)
    model.classifier = DeepLabHead(960, 2)

    print("2. Injecting your custom trained weights from the .pth file...")
    model.load_state_dict(
        torch.load("data/models/studio_occlusion_model.pth", map_location=DEVICE),
        strict=False,
    )
    model.to(DEVICE)
    model.eval()

    print("3. Translating to ONNX format (Opset 18)...")
    dummy_input = torch.randn(1, 3, 360, 640, device=DEVICE)
    onnx_path = "data/models/studio_occlusion_model.onnx"

    # Removed dynamic_axes (Fixed shapes prevent DirectX 12 crashes in UE5)
    # Updated to opset 18 to satisfy the PyTorch Dynamo exporter
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )

    print("4. Forcing monolithic file merge for Unreal Engine...")
    # Load the split model using the native ONNX library
    onnx_model = onnx.load(onnx_path)

    # Save it back out, explicitly forcing it to merge the .data weights back inside
    onnx.save_model(
        onnx_model, onnx_path, save_as_external_data=False, all_tensors_to_one_file=True
    )

    # Clean up the leftover .data file so it doesn't cause confusion
    data_file = onnx_path + ".data"
    if os.path.exists(data_file):
        os.remove(data_file)

    print(f"Success! Monolithic, Engine-ready model exported to {onnx_path}")


if __name__ == "__main__":
    export_saved_model()

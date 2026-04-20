import torch
import onnx
import os
from torchvision import models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class UEInferenceWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)["out"]


def export_saved_model():
    print("1. Loading the MobileNetV3 Student Architecture...")
    base_model = models.segmentation.deeplabv3_mobilenet_v3_large(weights=None)
    base_model.classifier = DeepLabHead(960, 2)

    print("2. Injecting your custom trained weights from the .pth file...")
    base_model.load_state_dict(
        torch.load("data/models/studio_occlusion_model.pth", map_location=DEVICE),
        strict=False,
    )

    print("3. Wrapping model to strip dictionaries for Unreal Engine...")
    model = UEInferenceWrapper(base_model)
    model.to(DEVICE)
    model.eval()

    print("4. Translating to ONNX format (Opset 18)...")
    dummy_input = torch.randn(1, 3, 360, 640, device=DEVICE)
    onnx_path = "data/models/studio_occlusion_model.onnx"

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

    print("5. Forcing monolithic file merge and fixing IR Version for UE 5.4...")
    onnx_model = onnx.load(onnx_path)

    # CRITICAL FIX: Downgrade the ID card so UE 5.4's security checker lets it through
    onnx_model.ir_version = 8

    onnx.save_model(
        onnx_model, onnx_path, save_as_external_data=False, all_tensors_to_one_file=True
    )

    data_file = onnx_path + ".data"
    if os.path.exists(data_file):
        os.remove(data_file)

    print("6. Running strict C++ integrity check...")
    onnx.checker.check_model(onnx_model)

    print(f"Success! Clean, UE-compatible model exported to {onnx_path}")


if __name__ == "__main__":
    export_saved_model()

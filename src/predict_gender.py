from pathlib import Path
import sys

from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v3_small


sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import MODELS_DIR


MODEL_PATH = MODELS_DIR / "gender_classifier" / "best_gender_mobilenet_v3_small.pth"
IMAGE_SIZE = 224


def build_gender_model():
    model = mobilenet_v3_small(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 1)
    return model


def get_gender_transform():
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def load_gender_model(model_path=MODEL_PATH, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(Path(model_path), map_location=device)
    model = build_gender_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, device, {
        "model_path": str(model_path),
        "model_name": checkpoint.get("model_name"),
        "best_val_acc": checkpoint.get("best_val_acc"),
        "epoch": checkpoint.get("epoch"),
        "device": str(device),
    }


@torch.no_grad()
def predict_gender_from_pil(image: Image.Image, model, device):
    tensor = get_gender_transform()(image.convert("RGB")).unsqueeze(0).to(device)
    probability_female = float(torch.sigmoid(model(tensor).squeeze()).cpu().item())
    label = "female" if probability_female >= 0.5 else "male"
    confidence = probability_female if label == "female" else 1.0 - probability_female
    return {
        "label": label,
        "confidence": confidence,
        "probability_female": probability_female,
    }


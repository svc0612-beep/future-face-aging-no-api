from pathlib import Path
import sys

from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import convnext_tiny


sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import MODELS_DIR


# ============================================================
# 설정
# ============================================================

MODEL_PATH = MODELS_DIR / "age_estimator" / "best_convnext_tiny_finetune_lr1e5.pth"
IMAGE_SIZE = 224


# ============================================================
# Transform
# ============================================================

def get_age_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# ============================================================
# Model
# ============================================================

def build_age_model():
    model = convnext_tiny(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 1)
    return model


def load_age_model(model_path=MODEL_PATH, device=None):
    """
    학습된 나이 예측 모델을 불러온다.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    model = build_age_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    info = {
        "model_path": str(model_path),
        "model_name": checkpoint.get("model_name"),
        "best_val_mae": checkpoint.get("best_val_mae"),
        "epoch": checkpoint.get("epoch"),
        "image_size": checkpoint.get("image_size"),
        "device": str(device),
    }

    return model, device, info


# ============================================================
# Predict
# ============================================================

@torch.no_grad()
def predict_age_from_pil(image: Image.Image, model, device):
    """
    PIL 이미지 1장을 입력받아 나이를 예측한다.
    """
    transform = get_age_transform()

    image = image.convert("RGB")
    image_tensor = transform(image).unsqueeze(0)
    image_tensor = image_tensor.to(device)

    output = model(image_tensor).squeeze()
    pred_age = float(output.detach().cpu().item())

    # 비정상 값 방지
    pred_age = max(0.0, min(pred_age, 120.0))

    return pred_age


def predict_age_from_path(image_path, model, device):
    """
    이미지 경로를 입력받아 나이를 예측한다.
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")

    image = Image.open(image_path).convert("RGB")
    return predict_age_from_pil(image, model, device)


def make_future_ages(current_age, years_list=None):
    """
    현재 예측 나이를 기준으로 10년 후~50년 후 나이를 계산한다.
    """
    if years_list is None:
        years_list = [10, 20, 30, 40, 50]

    result = []

    for years in years_list:
        future_age = current_age + years

        result.append({
            "years_later": years,
            "future_age": future_age,
        })

    return result


# ============================================================
# Test
# ============================================================

def main():
    print("=" * 80)
    print("나이 예측 모델 로드 테스트")
    print("=" * 80)

    print("MODEL_PATH:", MODEL_PATH)

    print("\nPyTorch:", torch.__version__)
    print("CUDA 사용 가능:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    model, device, info = load_age_model()

    print("\n모델 로드 정보:")
    for k, v in info.items():
        print(f"{k}: {v}")

    print("\n모델 로드 성공")

    # 샘플 이미지 1장으로 테스트하고 싶으면 아래 경로를 실제 이미지 경로로 바꿔서 주석 해제
    # sample_image_path = r"C:\Users\svc06\OneDrive\Desktop\future_face_project\data\raw\utkface\part1\25_0_0_20170117153303310.jpg"
    # pred_age = predict_age_from_path(sample_image_path, model, device)
    # print(f"\n예측 나이: {pred_age:.2f}세")
    # print("미래 나이 계산:")
    # for item in make_future_ages(pred_age):
    #     print(f"{item['years_later']}년 후: {item['future_age']:.1f}세")


if __name__ == "__main__":
    main()

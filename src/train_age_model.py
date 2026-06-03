from pathlib import Path
import sys
import time
import random

import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights


sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR


# ============================================================
# 설정값
# ============================================================

TRAIN_CSV = PROCESSED_DIR / "train_metadata.csv"
VAL_CSV = PROCESSED_DIR / "val_metadata.csv"

MODEL_SAVE_DIR = MODELS_DIR / "age_estimator"
MODEL_SAVE_PATH = MODEL_SAVE_DIR / "best_age_resnet18.pth"

LOG_DIR = OUTPUTS_DIR / "logs"
LOG_CSV = LOG_DIR / "train_age_resnet18_log.csv"

# CPU 테스트용 설정
# 처음에는 작게 돌려서 코드 정상 작동 확인
MAX_TRAIN_SAMPLES = 3000
MAX_VAL_SAMPLES = 800

# 본학습 때는 아래처럼 바꾸면 됨
# MAX_TRAIN_SAMPLES = None
# MAX_VAL_SAMPLES = None

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 1

LEARNING_RATE = 1e-4
RANDOM_SEED = 42

# 현재 CPU 환경이므로 pretrained weight 다운로드를 피하기 위해 False 권장
# 인터넷 가능하고 GPU 환경이면 True로 바꿔도 됨
USE_PRETRAINED = False

NUM_WORKERS = 0


# ============================================================
# 재현성 설정
# ============================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Dataset
# ============================================================

class AgeDataset(Dataset):
    def __init__(self, csv_path, transform=None, max_samples=None):
        self.df = pd.read_csv(csv_path)

        if max_samples is not None:
            self.df = self.df.sample(
                n=min(max_samples, len(self.df)),
                random_state=RANDOM_SEED
            ).reset_index(drop=True)

        self.transform = transform

        required_cols = ["image_path", "age", "dataset"]
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"필수 컬럼이 없습니다: {col}")

        self.df["age"] = self.df["age"].astype(float)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = row["image_path"]
        age = float(row["age"])
        dataset_name = row["dataset"]

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"이미지 열기 실패: {image_path}\n에러: {e}")

        if self.transform is not None:
            image = self.transform(image)

        age_tensor = torch.tensor(age, dtype=torch.float32)

        return image, age_tensor, dataset_name


# ============================================================
# Transform
# ============================================================

def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=8),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return train_transform, val_transform


# ============================================================
# Model
# ============================================================

def build_model():
    if USE_PRETRAINED:
        weights = ResNet18_Weights.DEFAULT
    else:
        weights = None

    model = resnet18(weights=weights)

    in_features = model.fc.in_features

    # 나이 예측은 회귀 문제라 출력 1개
    model.fc = nn.Linear(in_features, 1)

    return model


# ============================================================
# Train / Validate
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    total_abs_error = 0.0
    total_count = 0

    start_time = time.time()

    for batch_idx, (images, ages, dataset_names) in enumerate(loader, start=1):
        images = images.to(device)
        ages = ages.to(device)

        optimizer.zero_grad()

        outputs = model(images).squeeze(1)

        loss = criterion(outputs, ages)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        total_abs_error += torch.abs(outputs.detach() - ages).sum().item()
        total_count += batch_size

        if batch_idx % 20 == 0 or batch_idx == len(loader):
            current_mae = total_abs_error / total_count
            elapsed = time.time() - start_time

            print(
                f"Train Batch [{batch_idx}/{len(loader)}] "
                f"Loss: {loss.item():.4f} "
                f"MAE: {current_mae:.4f} "
                f"Elapsed: {elapsed:.1f}s"
            )

    avg_loss = total_loss / total_count
    avg_mae = total_abs_error / total_count

    return avg_loss, avg_mae


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_abs_error = 0.0
    total_count = 0

    for batch_idx, (images, ages, dataset_names) in enumerate(loader, start=1):
        images = images.to(device)
        ages = ages.to(device)

        outputs = model(images).squeeze(1)

        loss = criterion(outputs, ages)

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        total_abs_error += torch.abs(outputs - ages).sum().item()
        total_count += batch_size

    avg_loss = total_loss / total_count
    avg_mae = total_abs_error / total_count

    return avg_loss, avg_mae


# ============================================================
# Main
# ============================================================

def main():
    set_seed(RANDOM_SEED)

    print("=" * 80)
    print("얼굴 나이 예측 ResNet18 학습 테스트 시작")
    print("=" * 80)

    print(f"TRAIN_CSV: {TRAIN_CSV}")
    print(f"VAL_CSV  : {VAL_CSV}")

    if not TRAIN_CSV.exists():
        raise FileNotFoundError(f"train csv가 없습니다: {TRAIN_CSV}")

    if not VAL_CSV.exists():
        raise FileNotFoundError(f"val csv가 없습니다: {VAL_CSV}")

    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print("\nPyTorch 버전:", torch.__version__)
    print("CUDA 사용 가능:", torch.cuda.is_available())

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU 이름:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("현재 CPU로 실행됩니다. 테스트 학습은 가능하지만 본학습은 느릴 수 있습니다.")

    print("\n설정")
    print(f"IMAGE_SIZE       : {IMAGE_SIZE}")
    print(f"BATCH_SIZE       : {BATCH_SIZE}")
    print(f"EPOCHS           : {EPOCHS}")
    print(f"LEARNING_RATE    : {LEARNING_RATE}")
    print(f"USE_PRETRAINED   : {USE_PRETRAINED}")
    print(f"MAX_TRAIN_SAMPLES: {MAX_TRAIN_SAMPLES}")
    print(f"MAX_VAL_SAMPLES  : {MAX_VAL_SAMPLES}")

    train_transform, val_transform = get_transforms()

    train_dataset = AgeDataset(
        TRAIN_CSV,
        transform=train_transform,
        max_samples=MAX_TRAIN_SAMPLES,
    )

    val_dataset = AgeDataset(
        VAL_CSV,
        transform=val_transform,
        max_samples=MAX_VAL_SAMPLES,
    )

    print("\nDataset 크기")
    print(f"train_dataset: {len(train_dataset):,}")
    print(f"val_dataset  : {len(val_dataset):,}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    print("\nDataLoader 배치 수")
    print(f"train_loader: {len(train_loader):,}")
    print(f"val_loader  : {len(val_loader):,}")

    model = build_model()
    model = model.to(device)

    # 나이 예측은 MAE를 직접 최적화하기 위해 L1Loss 사용
    criterion = nn.L1Loss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_val_mae = float("inf")
    logs = []

    for epoch in range(1, EPOCHS + 1):
        print("\n" + "=" * 80)
        print(f"Epoch [{epoch}/{EPOCHS}]")
        print("=" * 80)

        epoch_start = time.time()

        train_loss, train_mae = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        val_loss, val_mae = validate_one_epoch(
            model,
            val_loader,
            criterion,
            device,
        )

        epoch_time = time.time() - epoch_start

        print("\nEpoch 결과")
        print(f"Train Loss/MAE: {train_loss:.4f}")
        print(f"Val Loss/MAE  : {val_loss:.4f}")
        print(f"Epoch Time    : {epoch_time:.1f}s")

        log_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_mae": train_mae,
            "val_loss": val_loss,
            "val_mae": val_mae,
            "epoch_time_sec": epoch_time,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "batch_size": BATCH_SIZE,
            "device": str(device),
            "use_pretrained": USE_PRETRAINED,
        }

        logs.append(log_row)

        log_df = pd.DataFrame(logs)
        log_df.to_csv(LOG_CSV, index=False, encoding="utf-8-sig")

        if val_mae < best_val_mae:
            best_val_mae = val_mae

            save_data = {
                "model_state_dict": model.state_dict(),
                "model_name": "resnet18_age_regression",
                "image_size": IMAGE_SIZE,
                "best_val_mae": best_val_mae,
                "epoch": epoch,
                "use_pretrained": USE_PRETRAINED,
            }

            torch.save(save_data, MODEL_SAVE_PATH)

            print(f"Best model 저장 완료: {MODEL_SAVE_PATH}")
            print(f"Best Val MAE: {best_val_mae:.4f}")

    print("\n" + "=" * 80)
    print("학습 테스트 완료")
    print("=" * 80)
    print(f"최고 Val MAE: {best_val_mae:.4f}")
    print(f"모델 저장 경로: {MODEL_SAVE_PATH}")
    print(f"로그 저장 경로: {LOG_CSV}")


if __name__ == "__main__":
    main()
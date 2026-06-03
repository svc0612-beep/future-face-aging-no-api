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

from tqdm import tqdm


sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR


# ============================================================
# 설정값
# ============================================================

TRAIN_CSV = PROCESSED_DIR / "train_metadata.csv"
VAL_CSV = PROCESSED_DIR / "val_metadata.csv"
TEST_CSV = PROCESSED_DIR / "test_metadata.csv"

MODEL_SAVE_DIR = MODELS_DIR / "age_estimator"
BEST_MODEL_SAVE_PATH = MODEL_SAVE_DIR / "best_age_resnet18_full.pth"
LAST_MODEL_SAVE_PATH = MODEL_SAVE_DIR / "last_age_resnet18_full.pth"

LOG_DIR = OUTPUTS_DIR / "logs"
LOG_CSV = LOG_DIR / "train_age_resnet18_full_log.csv"

IMAGE_SIZE = 224

# MX450 2GB 기준 안전 설정
# CUDA out of memory가 나면 4로 낮추면 됨
BATCH_SIZE = 8

EPOCHS = 10
PATIENCE = 3

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

RANDOM_SEED = 42

# 본학습이므로 사전학습 가중치 사용
USE_PRETRAINED = True

# 전체 데이터 사용
MAX_TRAIN_SAMPLES = None
MAX_VAL_SAMPLES = None

# Windows 환경에서는 num_workers=0이 가장 안정적
NUM_WORKERS = 0

# CUDA일 때만 AMP 사용
USE_AMP = True


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
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.10,
            hue=0.03,
        ),
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
        print("ResNet18 사전학습 가중치 사용")
        weights = ResNet18_Weights.DEFAULT
    else:
        print("사전학습 가중치 사용 안 함")
        weights = None

    model = resnet18(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)

    return model


# ============================================================
# Train / Validate
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, device, epoch, scaler=None):
    model.train()

    total_loss = 0.0
    total_abs_error = 0.0
    total_count = 0

    start_time = time.time()

    progress_bar = tqdm(
        loader,
        desc=f"Train Epoch {epoch}",
        dynamic_ncols=True,
        leave=True
    )

    for batch_idx, (images, ages, dataset_names) in enumerate(progress_bar, start=1):
        images = images.to(device, non_blocking=True)
        ages = ages.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(images).squeeze(1)
                loss = criterion(outputs, ages)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, ages)
            loss.backward()
            optimizer.step()

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        total_abs_error += torch.abs(outputs.detach() - ages).sum().item()
        total_count += batch_size

        current_loss = total_loss / total_count
        current_mae = total_abs_error / total_count
        elapsed = time.time() - start_time

        progress_bar.set_postfix({
            "loss": f"{current_loss:.4f}",
            "mae": f"{current_mae:.4f}",
            "elapsed": f"{elapsed:.0f}s"
        })

    avg_loss = total_loss / total_count
    avg_mae = total_abs_error / total_count

    return avg_loss, avg_mae


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device, epoch):
    model.eval()

    total_loss = 0.0
    total_abs_error = 0.0
    total_count = 0

    progress_bar = tqdm(
        loader,
        desc=f"Valid Epoch {epoch}",
        dynamic_ncols=True,
        leave=True
    )

    for batch_idx, (images, ages, dataset_names) in enumerate(progress_bar, start=1):
        images = images.to(device, non_blocking=True)
        ages = ages.to(device, non_blocking=True)

        if device.type == "cuda" and USE_AMP:
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(images).squeeze(1)
                loss = criterion(outputs, ages)
        else:
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, ages)

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        total_abs_error += torch.abs(outputs - ages).sum().item()
        total_count += batch_size

        current_loss = total_loss / total_count
        current_mae = total_abs_error / total_count

        progress_bar.set_postfix({
            "loss": f"{current_loss:.4f}",
            "mae": f"{current_mae:.4f}"
        })

    avg_loss = total_loss / total_count
    avg_mae = total_abs_error / total_count

    return avg_loss, avg_mae


# ============================================================
# Main
# ============================================================

def main():
    set_seed(RANDOM_SEED)

    print("=" * 80)
    print("얼굴 나이 예측 ResNet18 본학습 시작")
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
        print("현재 CPU로 실행됩니다. 본학습은 오래 걸릴 수 있습니다.")

    print("\n설정")
    print(f"IMAGE_SIZE       : {IMAGE_SIZE}")
    print(f"BATCH_SIZE       : {BATCH_SIZE}")
    print(f"EPOCHS           : {EPOCHS}")
    print(f"PATIENCE         : {PATIENCE}")
    print(f"LEARNING_RATE    : {LEARNING_RATE}")
    print(f"WEIGHT_DECAY     : {WEIGHT_DECAY}")
    print(f"USE_PRETRAINED   : {USE_PRETRAINED}")
    print(f"USE_AMP          : {USE_AMP}")
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

    criterion = nn.L1Loss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=1,
    )

    scaler = None
    if device.type == "cuda" and USE_AMP:
        scaler = torch.amp.GradScaler("cuda")
        print("AMP 사용: True")
    else:
        print("AMP 사용: False")

    best_val_mae = float("inf")
    best_epoch = 0
    no_improve_count = 0
    logs = []

    total_start_time = time.time()

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
            epoch,
            scaler=scaler,
        )

        val_loss, val_mae = validate_one_epoch(
            model,
            val_loader,
            criterion,
            device,
            epoch,
        )

        scheduler.step(val_mae)

        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start
        total_elapsed = time.time() - total_start_time

        print("\nEpoch 결과")
        print(f"Train Loss/MAE: {train_loss:.4f}")
        print(f"Val Loss/MAE  : {val_loss:.4f}")
        print(f"Current LR    : {current_lr:.8f}")
        print(f"Epoch Time    : {epoch_time:.1f}s")
        print(f"Total Time    : {total_elapsed / 60:.1f}분")

        log_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_mae": train_mae,
            "val_loss": val_loss,
            "val_mae": val_mae,
            "best_val_mae": min(best_val_mae, val_mae),
            "epoch_time_sec": epoch_time,
            "total_time_sec": total_elapsed,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "batch_size": BATCH_SIZE,
            "device": str(device),
            "use_pretrained": USE_PRETRAINED,
            "use_amp": device.type == "cuda" and USE_AMP,
            "lr": current_lr,
        }

        logs.append(log_row)

        log_df = pd.DataFrame(logs)
        log_df.to_csv(LOG_CSV, index=False, encoding="utf-8-sig")

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_name": "resnet18_age_regression",
                "image_size": IMAGE_SIZE,
                "epoch": epoch,
                "val_mae": val_mae,
                "use_pretrained": USE_PRETRAINED,
            },
            LAST_MODEL_SAVE_PATH,
        )

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            no_improve_count = 0

            save_data = {
                "model_state_dict": model.state_dict(),
                "model_name": "resnet18_age_regression",
                "image_size": IMAGE_SIZE,
                "best_val_mae": best_val_mae,
                "epoch": epoch,
                "use_pretrained": USE_PRETRAINED,
            }

            torch.save(save_data, BEST_MODEL_SAVE_PATH)

            print(f"Best model 저장 완료: {BEST_MODEL_SAVE_PATH}")
            print(f"Best Val MAE: {best_val_mae:.4f}")
        else:
            no_improve_count += 1
            print(f"개선 없음: {no_improve_count}/{PATIENCE}")

        if no_improve_count >= PATIENCE:
            print("\nEarly Stopping 실행")
            print(f"Best Epoch: {best_epoch}")
            print(f"Best Val MAE: {best_val_mae:.4f}")
            break

    total_time = time.time() - total_start_time

    print("\n" + "=" * 80)
    print("본학습 완료")
    print("=" * 80)
    print(f"Best Epoch   : {best_epoch}")
    print(f"Best Val MAE : {best_val_mae:.4f}")
    print(f"Total Time   : {total_time / 60:.1f}분")
    print(f"Best 모델 경로: {BEST_MODEL_SAVE_PATH}")
    print(f"Last 모델 경로: {LAST_MODEL_SAVE_PATH}")
    print(f"로그 저장 경로: {LOG_CSV}")


if __name__ == "__main__":
    main()
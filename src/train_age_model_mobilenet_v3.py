from pathlib import Path
import argparse
import random
import sys
import time

import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V3_Large_Weights,
    efficientnet_b0,
    mobilenet_v3_large,
)
from tqdm import tqdm


sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import MODELS_DIR, OUTPUTS_DIR, PROCESSED_DIR


TRAIN_CSV = PROCESSED_DIR / "train_metadata.csv"
VAL_CSV = PROCESSED_DIR / "val_metadata.csv"
MODEL_DIR = MODELS_DIR / "age_estimator"
LOG_DIR = OUTPUTS_DIR / "logs"


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AgeDataset(Dataset):
    def __init__(self, csv_path, transform, max_samples=None, seed=42):
        self.df = pd.read_csv(csv_path)
        if max_samples is not None:
            self.df = self.df.sample(
                n=min(max_samples, len(self.df)),
                random_state=seed,
            ).reset_index(drop=True)
        self.df["age"] = self.df["age"].astype(float)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        return self.transform(image), torch.tensor(float(row["age"]), dtype=torch.float32)


def build_transforms(image_size):
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=6),
            transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return train_transform, val_transform


def build_model(model_name):
    if model_name == "mobilenet_v3_large":
        model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, 1)
        return model

    if model_name == "efficientnet_b0":
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, 1)
        return model

    raise ValueError(f"Unsupported model: {model_name}")


def run_epoch(model, loader, criterion, optimizer, device, scaler, train, epoch):
    model.train(train)
    total_loss = 0.0
    total_abs = 0.0
    total_count = 0
    mode = "Train" if train else "Valid"

    progress = tqdm(loader, desc=f"{mode} Epoch {epoch}", dynamic_ncols=True)
    for images, ages in progress:
        images = images.to(device, non_blocking=True)
        ages = ages.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            if scaler is not None:
                with torch.amp.autocast(device_type="cuda"):
                    outputs = model(images).squeeze(1)
                    loss = criterion(outputs, ages)
            else:
                outputs = model(images).squeeze(1)
                loss = criterion(outputs, ages)

            if train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_abs += torch.abs(outputs.detach() - ages).sum().item()
        total_count += batch_size

        progress.set_postfix(
            {
                "loss": f"{total_loss / total_count:.4f}",
                "mae": f"{total_abs / total_count:.4f}",
            }
        )

    return total_loss / total_count, total_abs / total_count


def save_checkpoint(path, model, optimizer, args, epoch, best_val_mae):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_name": args.model,
            "best_val_mae": best_val_mae,
            "epoch": epoch,
            "image_size": args.image_size,
            "args": vars(args),
        },
        path,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["mobilenet_v3_large", "efficientnet_b0"], default="mobilenet_v3_large")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    run_name = args.run_name or f"age_{args.model}"
    best_path = MODEL_DIR / f"best_{run_name}.pth"
    last_path = MODEL_DIR / f"last_{run_name}.pth"
    log_path = LOG_DIR / f"train_{run_name}_log.csv"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print("device:", device)
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))

    train_transform, val_transform = build_transforms(args.image_size)
    train_dataset = AgeDataset(TRAIN_CSV, train_transform, args.max_train_samples, args.seed)
    val_dataset = AgeDataset(VAL_CSV, val_transform, args.max_val_samples, args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(args.model).to(device)
    criterion = nn.SmoothL1Loss(beta=3.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    best_val_mae = float("inf")
    stale_epochs = 0
    logs = []

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_mae = run_epoch(model, train_loader, criterion, optimizer, device, scaler, True, epoch)
        val_loss, val_mae = run_epoch(model, val_loader, criterion, optimizer, device, None, False, epoch)
        scheduler.step(val_mae)

        elapsed = time.time() - start
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_mae": train_mae,
            "val_loss": val_loss,
            "val_mae": val_mae,
            "elapsed_sec": elapsed,
            "lr": optimizer.param_groups[0]["lr"],
        }
        logs.append(row)
        pd.DataFrame(logs).to_csv(log_path, index=False, encoding="utf-8-sig")

        print(
            f"Epoch {epoch}: train_mae={train_mae:.4f}, val_mae={val_mae:.4f}, "
            f"elapsed={elapsed:.1f}s"
        )

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            stale_epochs = 0
            save_checkpoint(best_path, model, optimizer, args, epoch, best_val_mae)
            print("saved best:", best_path)
        else:
            stale_epochs += 1

        save_checkpoint(last_path, model, optimizer, args, epoch, best_val_mae)

        if stale_epochs >= args.patience:
            print("early stopping")
            break

    print("best_val_mae:", best_val_mae)
    print("best_path:", best_path)


if __name__ == "__main__":
    main()

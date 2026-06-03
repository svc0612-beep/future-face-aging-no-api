from __future__ import annotations

from pathlib import Path
import argparse
import json
import random
import sys
import time

import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import convnext_tiny
from tqdm import tqdm


sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import MODELS_DIR, OUTPUTS_DIR, PROCESSED_DIR


TRAIN_CSV = PROCESSED_DIR / "train_metadata.csv"
VAL_CSV = PROCESSED_DIR / "val_metadata.csv"
MODEL_DIR = MODELS_DIR / "age_estimator"
LOG_DIR = OUTPUTS_DIR / "logs"
STATUS_PATH = LOG_DIR / "age_improvement_status.json"


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AgeDataset(Dataset):
    def __init__(self, csv_path, transform):
        self.df = pd.read_csv(csv_path)
        self.df["age"] = self.df["age"].astype(float)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        return self.transform(image), torch.tensor(float(row["age"]), dtype=torch.float32)


def build_transforms(image_size):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=4),
            transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.05),
            transforms.ToTensor(),
            normalize,
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, val_transform


def build_model():
    model = convnext_tiny(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 1)
    return model


def write_status(**kwargs):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    previous = {}
    if STATUS_PATH.exists():
        try:
            previous = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous = {}
    previous.update(kwargs)
    previous["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    STATUS_PATH.write_text(json.dumps(previous, ensure_ascii=False, indent=2), encoding="utf-8")


def save_checkpoint(path, model, optimizer, args, epoch, best_val_mae):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_name": "convnext_tiny_finetuned",
            "best_val_mae": best_val_mae,
            "epoch": epoch,
            "image_size": args.image_size,
            "base_checkpoint": args.base_checkpoint,
            "args": vars(args),
        },
        path,
    )


def run_epoch(model, loader, criterion, optimizer, device, scaler, train, epoch, total_epochs, epoch_start, run_start):
    model.train(train)
    total_loss = 0.0
    total_abs = 0.0
    total_count = 0
    phase = "train" if train else "valid"
    progress = tqdm(loader, desc=f"{phase} epoch {epoch}", dynamic_ncols=True)
    total_steps = len(loader)

    for step, (images, ages) in enumerate(progress, start=1):
        images = images.to(device, non_blocking=True)
        ages = ages.to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(images).squeeze(1)
                loss = criterion(outputs, ages)
            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_abs += torch.abs(outputs.detach() - ages).sum().item()
        total_count += batch_size
        current_loss = total_loss / total_count
        current_mae = total_abs / total_count
        progress.set_postfix({"loss": f"{current_loss:.4f}", "mae": f"{current_mae:.4f}"})

        phase_elapsed = time.time() - epoch_start
        step_time = phase_elapsed / max(step, 1)
        phase_eta = step_time * (total_steps - step)
        epoch_progress = (epoch - 1 + (0.5 if not train else 0.0) + step / total_steps * 0.5) / total_epochs
        write_status(
            status="running",
            task="convnext_finetune",
            current_model="ConvNeXt-Tiny fine-tuning",
            phase=phase,
            epoch=epoch,
            total_epochs=total_epochs,
            step=step,
            total_steps=total_steps,
            progress_percent=round(epoch_progress * 100, 2),
            current_loss=round(current_loss, 4),
            current_mae=round(current_mae, 4),
            phase_eta_seconds=round(phase_eta, 1),
            elapsed_seconds=round(time.time() - run_start, 1),
        )

    return total_loss / total_count, total_abs / total_count


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", default=str(MODEL_DIR / "best_convnext_tiny_full.pth"))
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default="convnext_tiny_finetune_lr1e5")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for fine-tuning.")

    base_checkpoint = Path(args.base_checkpoint)
    if not base_checkpoint.exists():
        raise FileNotFoundError(base_checkpoint)

    device = torch.device("cuda")
    train_transform, val_transform = build_transforms(args.image_size)
    train_loader = DataLoader(
        AgeDataset(TRAIN_CSV, train_transform),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        AgeDataset(VAL_CSV, val_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    checkpoint = torch.load(base_checkpoint, map_location=device)
    model = build_model().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = nn.SmoothL1Loss(beta=3.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
    scaler = torch.amp.GradScaler("cuda")

    best_path = MODEL_DIR / f"best_{args.run_name}.pth"
    last_path = MODEL_DIR / f"last_{args.run_name}.pth"
    log_path = LOG_DIR / f"train_{args.run_name}_log.csv"

    base_best = float(checkpoint.get("best_val_mae", float("inf")))
    best_val_mae = base_best
    stale_epochs = 0
    logs = []
    run_start = time.time()

    write_status(
        status="running",
        task="convnext_finetune",
        current_model="ConvNeXt-Tiny fine-tuning",
        phase="start",
        epoch=0,
        total_epochs=args.epochs,
        base_best_val_mae=round(base_best, 4),
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        patience=args.patience,
        deadline_note="내일 오전 8시 전 완료 목표",
    )

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss, train_mae = run_epoch(
            model, train_loader, criterion, optimizer, device, scaler, True, epoch, args.epochs, epoch_start, run_start
        )
        val_loss, val_mae = run_epoch(
            model, val_loader, criterion, optimizer, device, scaler, False, epoch, args.epochs, epoch_start, run_start
        )
        scheduler.step(val_mae)
        elapsed = time.time() - epoch_start

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_mae": train_mae,
            "val_loss": val_loss,
            "val_mae": val_mae,
            "elapsed_sec": elapsed,
            "lr": optimizer.param_groups[0]["lr"],
            "base_best_val_mae": base_best,
        }
        logs.append(row)
        pd.DataFrame(logs).to_csv(log_path, index=False, encoding="utf-8-sig")

        improved = val_mae < best_val_mae
        if improved:
            best_val_mae = val_mae
            stale_epochs = 0
            save_checkpoint(best_path, model, optimizer, args, epoch, best_val_mae)
        else:
            stale_epochs += 1

        save_checkpoint(last_path, model, optimizer, args, epoch, best_val_mae)
        write_status(
            status="running",
            task="convnext_finetune",
            phase="epoch_done",
            epoch=epoch,
            total_epochs=args.epochs,
            train_mae=round(train_mae, 4),
            val_mae=round(val_mae, 4),
            best_val_mae=round(best_val_mae, 4),
            improved=improved,
            stale_epochs=stale_epochs,
            elapsed_seconds=round(time.time() - run_start, 1),
            log_path=str(log_path),
            best_path=str(best_path if best_path.exists() else base_checkpoint),
        )

        if stale_epochs >= args.patience:
            break

    final_best_path = best_path if best_path.exists() else base_checkpoint
    write_status(
        status="completed",
        task="convnext_finetune",
        current_model="ConvNeXt-Tiny fine-tuning",
        phase="completed",
        best_val_mae=round(best_val_mae, 4),
        base_best_val_mae=round(base_best, 4),
        best_path=str(final_best_path),
        last_path=str(last_path),
        log_path=str(log_path),
        progress_percent=100,
        eta_seconds=0,
        elapsed_seconds=round(time.time() - run_start, 1),
    )
    print(f"best_val_mae={best_val_mae:.4f}")
    print(f"best_path={final_best_path}")


if __name__ == "__main__":
    main()

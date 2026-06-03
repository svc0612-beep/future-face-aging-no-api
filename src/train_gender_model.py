from pathlib import Path
import argparse
from datetime import datetime
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
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
from tqdm import tqdm


sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import MODELS_DIR, OUTPUTS_DIR, PROCESSED_DIR


TRAIN_CSV = PROCESSED_DIR / "train_metadata.csv"
VAL_CSV = PROCESSED_DIR / "val_metadata.csv"
MODEL_DIR = MODELS_DIR / "gender_classifier"
LOG_DIR = OUTPUTS_DIR / "logs"
STATUS_PATH = LOG_DIR / "gender_mobilenet_v3_small_status.json"
RECOVERY_PATH = MODEL_DIR / "recovery_gender_mobilenet_v3_small.pth"


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class GenderDataset(Dataset):
    def __init__(self, csv_path, transform, max_samples=None, seed=42):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df["gender"].notna()].reset_index(drop=True)
        if max_samples is not None:
            self.df = self.df.sample(
                n=min(max_samples, len(self.df)),
                random_state=seed,
            ).reset_index(drop=True)
        self.df["gender"] = self.df["gender"].astype(float)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        return self.transform(image), torch.tensor(float(row["gender"]), dtype=torch.float32)


def build_transforms(image_size):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.06),
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
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 1)
    return model


def write_status(epoch, max_epochs, phase, step, total_steps, loss, acc, started_at):
    STATUS_PATH.write_text(
        json.dumps(
            {
                "epoch": epoch,
                "max_epochs": max_epochs,
                "phase": phase,
                "step": step,
                "total_steps": total_steps,
                "loss": loss,
                "accuracy": acc,
                "started_at": started_at,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def save_recovery(model, optimizer, args, epoch, step, best_val_acc):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_name": "mobilenet_v3_small",
            "best_val_acc": best_val_acc,
            "epoch": epoch,
            "step": step,
            "image_size": args.image_size,
            "args": vars(args),
        },
        RECOVERY_PATH,
    )


def run_epoch(model, loader, criterion, optimizer, device, scaler, train, epoch, args, best_val_acc, started_at):
    model.train(train)
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    mode = "Train" if train else "Valid"

    progress = tqdm(loader, desc=f"{mode} Epoch {epoch}", dynamic_ncols=True)
    for step, (images, labels) in enumerate(progress, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            if scaler is not None:
                with torch.amp.autocast(device_type="cuda"):
                    logits = model(images).squeeze(1)
                    loss = criterion(logits, labels)
            else:
                logits = model(images).squeeze(1)
                loss = criterion(logits, labels)

            if train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        predictions = (torch.sigmoid(logits.detach()) >= 0.5).float()
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (predictions == labels).sum().item()
        total_count += batch_size
        progress.set_postfix(
            {
                "loss": f"{total_loss / total_count:.4f}",
                "acc": f"{total_correct / total_count:.4f}",
            }
        )
        write_status(
            epoch=epoch,
            max_epochs=args.epochs,
            phase=mode.lower(),
            step=step,
            total_steps=len(loader),
            loss=total_loss / total_count,
            acc=total_correct / total_count,
            started_at=started_at,
        )
        if train and step % 250 == 0:
            save_recovery(model, optimizer, args, epoch, step, best_val_acc)

    return total_loss / total_count, total_correct / total_count


def save_checkpoint(path, model, optimizer, args, epoch, best_val_acc):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_name": "mobilenet_v3_small",
            "best_val_acc": best_val_acc,
            "epoch": epoch,
            "image_size": args.image_size,
            "args": vars(args),
        },
        path,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--run-name", default="gender_mobilenet_v3_small")
    parser.add_argument("--resume-recovery", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    best_path = MODEL_DIR / f"best_{args.run_name}.pth"
    last_path = MODEL_DIR / f"last_{args.run_name}.pth"
    log_path = LOG_DIR / f"train_{args.run_name}_log.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, flush=True)
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0), flush=True)

    train_transform, val_transform = build_transforms(args.image_size)
    train_dataset = GenderDataset(TRAIN_CSV, train_transform, args.max_train_samples, args.seed)
    val_dataset = GenderDataset(VAL_CSV, val_transform, args.max_val_samples, args.seed)
    print(f"train_samples={len(train_dataset)}, val_samples={len(val_dataset)}", flush=True)

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

    model = build_model().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    best_val_acc = 0.0
    start_epoch = 1
    if args.resume_recovery and RECOVERY_PATH.exists():
        recovery = torch.load(RECOVERY_PATH, map_location=device)
        model.load_state_dict(recovery["model_state_dict"])
        optimizer.load_state_dict(recovery["optimizer_state_dict"])
        best_val_acc = float(recovery.get("best_val_acc", 0.0))
        start_epoch = int(recovery.get("epoch", 1))
        print(
            f"resumed recovery: epoch={start_epoch}, step={recovery.get('step')}, "
            f"best_val_acc={best_val_acc:.4f}",
            flush=True,
        )
    stale_epochs = 0
    logs = []
    started_at = datetime.now().isoformat(timespec="seconds")
    for epoch in range(start_epoch, args.epochs + 1):
        start = time.time()
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, scaler, True, epoch, args, best_val_acc, started_at
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, optimizer, device, None, False, epoch, args, best_val_acc, started_at
        )
        scheduler.step(val_acc)
        elapsed = time.time() - start

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "elapsed_sec": elapsed,
            "lr": optimizer.param_groups[0]["lr"],
        }
        logs.append(row)
        pd.DataFrame(logs).to_csv(log_path, index=False, encoding="utf-8-sig")
        print(
            f"Epoch {epoch}: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, "
            f"val_loss={val_loss:.4f}, elapsed={elapsed:.1f}s",
            flush=True,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            stale_epochs = 0
            save_checkpoint(best_path, model, optimizer, args, epoch, best_val_acc)
            print("saved best:", best_path, flush=True)
        else:
            stale_epochs += 1

        save_checkpoint(last_path, model, optimizer, args, epoch, best_val_acc)
        if stale_epochs >= args.patience:
            print("early stopping", flush=True)
            break

    print("best_val_acc:", best_val_acc, flush=True)
    print("best_path:", best_path, flush=True)
    write_status(
        epoch=epoch,
        max_epochs=args.epochs,
        phase="complete",
        step=1,
        total_steps=1,
        loss=val_loss,
        acc=best_val_acc,
        started_at=started_at,
    )


if __name__ == "__main__":
    main()

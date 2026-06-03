from __future__ import annotations

from pathlib import Path
import argparse
import json
import math
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


TEST_CSV = PROCESSED_DIR / "test_metadata.csv"
OUTPUT_DIR = OUTPUTS_DIR / "evaluation"
STATUS_PATH = OUTPUTS_DIR / "logs" / "age_improvement_status.json"
IMAGE_SIZE = 224
BATCH_SIZE = 8
NUM_WORKERS = 0


class AgeDataset(Dataset):
    def __init__(self, csv_path: Path, transform):
        self.df = pd.read_csv(csv_path)
        self.df["age"] = self.df["age"].astype(float)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        return {
            "image": self.transform(image),
            "age": torch.tensor(float(row["age"]), dtype=torch.float32),
            "dataset": row.get("dataset", "unknown"),
            "image_path": row["image_path"],
        }


def build_model():
    model = convnext_tiny(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 1)
    return model


def transform():
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


def write_status(**kwargs):
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    previous = {}
    if STATUS_PATH.exists():
        try:
            previous = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous = {}
    previous.update(kwargs)
    previous["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    STATUS_PATH.write_text(json.dumps(previous, ensure_ascii=False, indent=2), encoding="utf-8")


def metrics(pred_df: pd.DataFrame):
    true_age = pred_df["true_age"].to_numpy()
    pred_age = pred_df["pred_age"].to_numpy()
    abs_error = pred_df["abs_error"].to_numpy()
    return {
        "mae": float(abs_error.mean()),
        "rmse": float(math.sqrt(((pred_age - true_age) ** 2).mean())),
        "within_3": float((abs_error <= 3).mean() * 100),
        "within_5": float((abs_error <= 5).mean() * 100),
        "within_10": float((abs_error <= 10).mean() * 100),
    }


@torch.no_grad()
def evaluate(model, loader, device, run_name: str):
    model.eval()
    rows = []
    total_abs_error = 0.0
    total_count = 0
    total_steps = len(loader)
    start = time.time()

    progress = tqdm(loader, desc=f"{run_name} test", dynamic_ncols=True)
    for step, batch in enumerate(progress, start=1):
        images = batch["image"].to(device, non_blocking=True)
        true_ages = batch["age"].to(device, non_blocking=True)
        outputs = model(images).squeeze(1)
        abs_errors = torch.abs(outputs - true_ages)

        total_abs_error += abs_errors.sum().item()
        total_count += images.size(0)
        current_mae = total_abs_error / total_count
        progress.set_postfix({"mae": f"{current_mae:.4f}"})

        elapsed = time.time() - start
        step_time = elapsed / max(step, 1)
        eta = step_time * (total_steps - step)
        write_status(
            status="running",
            task="test_evaluation",
            current_model=run_name,
            phase="test",
            step=step,
            total_steps=total_steps,
            progress_percent=round(step / total_steps * 100, 2),
            current_mae=round(current_mae, 4),
            eta_seconds=round(eta, 1),
        )

        preds_cpu = outputs.detach().cpu().numpy()
        true_cpu = true_ages.detach().cpu().numpy()
        errors_cpu = abs_errors.detach().cpu().numpy()

        for image_path, dataset_name, true_age, pred_age, error in zip(
            batch["image_path"],
            batch["dataset"],
            true_cpu,
            preds_cpu,
            errors_cpu,
        ):
            rows.append(
                {
                    "image_path": image_path,
                    "dataset": dataset_name,
                    "true_age": float(true_age),
                    "pred_age": float(pred_age),
                    "pred_age_rounded": int(round(float(pred_age))),
                    "abs_error": float(error),
                }
            )

    return pd.DataFrame(rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=str(MODELS_DIR / "age_estimator" / "best_convnext_tiny_full.pth"))
    parser.add_argument("--run-name", default="convnext_tiny_full")
    return parser.parse_args()


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    model = build_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    loader = DataLoader(
        AgeDataset(TEST_CSV, transform()),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    write_status(
        status="running",
        task="test_evaluation",
        current_model=args.run_name,
        phase="start",
        model_path=str(model_path),
    )
    pred_df = evaluate(model, loader, device, args.run_name)
    pred_path = OUTPUT_DIR / f"test_predictions_{args.run_name}.csv"
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
    overall = metrics(pred_df)

    dataset_summary = (
        pred_df.groupby("dataset")
        .agg(
            count=("abs_error", "count"),
            mae=("abs_error", "mean"),
            rmse=("abs_error", lambda x: math.sqrt((x ** 2).mean())),
            mean_true_age=("true_age", "mean"),
            mean_pred_age=("pred_age", "mean"),
        )
        .reset_index()
    )

    bins = [0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 120]
    labels = ["00s", "10s", "20s", "30s", "40s", "50s", "60s", "70s", "80s", "90plus"]
    pred_df["age_group"] = pd.cut(pred_df["true_age"], bins=bins, labels=labels, include_lowest=True)
    age_group_summary = (
        pred_df.groupby("age_group", observed=False)
        .agg(
            count=("abs_error", "count"),
            mae=("abs_error", "mean"),
            mean_true_age=("true_age", "mean"),
            mean_pred_age=("pred_age", "mean"),
        )
        .reset_index()
    )

    dataset_summary_path = OUTPUT_DIR / f"test_dataset_summary_{args.run_name}.csv"
    age_group_summary_path = OUTPUT_DIR / f"test_age_group_summary_{args.run_name}.csv"
    report_path = OUTPUT_DIR / f"test_report_{args.run_name}.txt"
    dataset_summary.to_csv(dataset_summary_path, index=False, encoding="utf-8-sig")
    age_group_summary.to_csv(age_group_summary_path, index=False, encoding="utf-8-sig")

    report = [
        f"{args.run_name} Test 평가 결과",
        "=" * 60,
        f"Model path: {model_path}",
        f"Test count: {len(pred_df):,}",
        "",
        "[Overall]",
        f"MAE: {overall['mae']:.4f}",
        f"RMSE: {overall['rmse']:.4f}",
        f"Within 3 years: {overall['within_3']:.2f}%",
        f"Within 5 years: {overall['within_5']:.2f}%",
        f"Within 10 years: {overall['within_10']:.2f}%",
        "",
        "[By dataset]",
        dataset_summary.to_string(index=False),
        "",
        "[By age group]",
        age_group_summary.to_string(index=False),
    ]
    report_path.write_text("\n".join(report), encoding="utf-8")

    write_status(
        status="completed",
        task="test_evaluation",
        current_model=args.run_name,
        test_mae=round(overall["mae"], 4),
        test_rmse=round(overall["rmse"], 4),
        within_5=round(overall["within_5"], 2),
        within_10=round(overall["within_10"], 2),
        prediction_csv=str(pred_path),
        report_path=str(report_path),
        progress_percent=100,
        eta_seconds=0,
    )
    print(report_path)


if __name__ == "__main__":
    main()

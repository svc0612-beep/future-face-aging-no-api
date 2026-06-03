from pathlib import Path
import sys
import math

import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.models import resnet18

from tqdm import tqdm


sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR


# ============================================================
# 설정
# ============================================================

TEST_CSV = PROCESSED_DIR / "test_metadata.csv"

MODEL_PATH = MODELS_DIR / "age_estimator" / "best_age_resnet18_full.pth"

OUTPUT_DIR = OUTPUTS_DIR / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PRED_CSV = OUTPUT_DIR / "test_predictions_resnet18.csv"
REPORT_TXT = OUTPUT_DIR / "test_report_resnet18.txt"

IMAGE_SIZE = 224
BATCH_SIZE = 8
NUM_WORKERS = 0


# ============================================================
# Dataset
# ============================================================

class AgeDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
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
        true_age = float(row["age"])
        dataset_name = row["dataset"]

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"이미지 열기 실패: {image_path}\n에러: {e}")

        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "age": torch.tensor(true_age, dtype=torch.float32),
            "dataset": dataset_name,
            "image_path": image_path,
        }


# ============================================================
# Model
# ============================================================

def build_model():
    model = resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)
    return model


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(pred_df: pd.DataFrame):
    true_age = pred_df["true_age"].values
    pred_age = pred_df["pred_age"].values
    abs_error = pred_df["abs_error"].values

    mae = abs_error.mean()
    rmse = math.sqrt(((pred_age - true_age) ** 2).mean())

    within_3 = (abs_error <= 3).mean() * 100
    within_5 = (abs_error <= 5).mean() * 100
    within_10 = (abs_error <= 10).mean() * 100

    return {
        "mae": mae,
        "rmse": rmse,
        "within_3": within_3,
        "within_5": within_5,
        "within_10": within_10,
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    rows = []
    total_abs_error = 0.0
    total_count = 0

    progress_bar = tqdm(loader, desc="Test 평가", dynamic_ncols=True)

    for batch in progress_bar:
        images = batch["image"].to(device, non_blocking=True)
        true_ages = batch["age"].to(device, non_blocking=True)

        outputs = model(images).squeeze(1)

        abs_errors = torch.abs(outputs - true_ages)

        total_abs_error += abs_errors.sum().item()
        total_count += images.size(0)

        current_mae = total_abs_error / total_count
        progress_bar.set_postfix({"mae": f"{current_mae:.4f}"})

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
            rows.append({
                "image_path": image_path,
                "dataset": dataset_name,
                "true_age": float(true_age),
                "pred_age": float(pred_age),
                "pred_age_rounded": int(round(float(pred_age))),
                "abs_error": float(error),
            })

    pred_df = pd.DataFrame(rows)
    return pred_df


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 80)
    print("ResNet18 나이 예측 모델 Test 평가 시작")
    print("=" * 80)

    print(f"TEST_CSV  : {TEST_CSV}")
    print(f"MODEL_PATH: {MODEL_PATH}")

    if not TEST_CSV.exists():
        raise FileNotFoundError(f"test_metadata.csv가 없습니다: {TEST_CSV}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"모델 파일이 없습니다: {MODEL_PATH}")

    print("\nPyTorch 버전:", torch.__version__)
    print("CUDA 사용 가능:", torch.cuda.is_available())

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU 이름:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("CPU로 평가합니다.")

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    test_dataset = AgeDataset(TEST_CSV, transform=transform)

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    print("\nTest dataset 크기:", len(test_dataset))
    print("Test loader 배치 수:", len(test_loader))

    checkpoint = torch.load(MODEL_PATH, map_location=device)

    print("\n체크포인트 정보:")
    print("model_name:", checkpoint.get("model_name"))
    print("best_val_mae:", checkpoint.get("best_val_mae"))
    print("epoch:", checkpoint.get("epoch"))
    print("image_size:", checkpoint.get("image_size"))

    model = build_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    pred_df = evaluate(model, test_loader, device)

    pred_df.to_csv(PRED_CSV, index=False, encoding="utf-8-sig")

    overall_metrics = calculate_metrics(pred_df)

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

    age_group_bins = [0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 120]
    age_group_labels = ["00s", "10s", "20s", "30s", "40s", "50s", "60s", "70s", "80s", "90plus"]

    pred_df["age_group"] = pd.cut(
        pred_df["true_age"],
        bins=age_group_bins,
        labels=age_group_labels,
        include_lowest=True,
    )

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

    dataset_summary_csv = OUTPUT_DIR / "test_dataset_summary_resnet18.csv"
    age_group_summary_csv = OUTPUT_DIR / "test_age_group_summary_resnet18.csv"

    dataset_summary.to_csv(dataset_summary_csv, index=False, encoding="utf-8-sig")
    age_group_summary.to_csv(age_group_summary_csv, index=False, encoding="utf-8-sig")

    report_lines = []
    report_lines.append("ResNet18 나이 예측 모델 Test 평가 결과")
    report_lines.append("=" * 60)
    report_lines.append(f"Test 전체 개수: {len(pred_df):,}")
    report_lines.append("")
    report_lines.append("[전체 성능]")
    report_lines.append(f"MAE : {overall_metrics['mae']:.4f}")
    report_lines.append(f"RMSE: {overall_metrics['rmse']:.4f}")
    report_lines.append(f"오차 3세 이내 비율 : {overall_metrics['within_3']:.2f}%")
    report_lines.append(f"오차 5세 이내 비율 : {overall_metrics['within_5']:.2f}%")
    report_lines.append(f"오차 10세 이내 비율: {overall_metrics['within_10']:.2f}%")
    report_lines.append("")
    report_lines.append("[데이터셋별 성능]")
    report_lines.append(dataset_summary.to_string(index=False))
    report_lines.append("")
    report_lines.append("[나이대별 성능]")
    report_lines.append(age_group_summary.to_string(index=False))
    report_lines.append("")
    report_lines.append("[저장 파일]")
    report_lines.append(f"예측 결과 CSV: {PRED_CSV}")
    report_lines.append(f"데이터셋별 요약 CSV: {dataset_summary_csv}")
    report_lines.append(f"나이대별 요약 CSV: {age_group_summary_csv}")

    report_text = "\n".join(report_lines)
    REPORT_TXT.write_text(report_text, encoding="utf-8")

    print("\n" + report_text)
    print("\n평가 완료")


if __name__ == "__main__":
    main()
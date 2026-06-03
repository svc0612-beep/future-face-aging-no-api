from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "outputs" / "logs"
OUT_DIR = PROJECT_ROOT / "outputs" / "reports" / "model_summary"


AGE_LOGS = {
    "MobileNetV3": LOG_DIR / "train_mobilenet_v3_full_log.csv",
    "ConvNeXt-Tiny": LOG_DIR / "train_convnext_tiny_full_log.csv",
    "ConvNeXt-Tiny Fine-tuned": LOG_DIR / "train_convnext_tiny_finetune_lr1e5_log.csv",
    "ConvNeXt-Tiny Fine-tuned 2nd": LOG_DIR / "train_convnext_tiny_finetune_lr1e5_epoch2_log.csv",
    "EfficientNet-B0": LOG_DIR / "train_efficientnet_b0_full_log.csv",
}

GENDER_LOG = LOG_DIR / "train_gender_mobilenet_v3_small_log.csv"
TEST_REPORTS = {
    "ConvNeXt-Tiny before fine-tuning": PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "test_report_convnext_tiny_full.txt",
    "ConvNeXt-Tiny after fine-tuning": PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "test_report_convnext_tiny_finetune_lr1e5_epoch2.txt",
}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _available_age_logs():
    return {name: path for name, path in AGE_LOGS.items() if path.exists()}


def _read_test_report(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="ignore")
    metrics = {}
    for key, pattern in {
        "test_mae": r"MAE:\s*([0-9.]+)",
        "test_rmse": r"RMSE:\s*([0-9.]+)",
        "within_3": r"Within 3 years:\s*([0-9.]+)%",
        "within_5": r"Within 5 years:\s*([0-9.]+)%",
        "within_10": r"Within 10 years:\s*([0-9.]+)%",
    }.items():
        match = re.search(pattern, text)
        if match:
            metrics[key] = float(match.group(1))
    return metrics


def plot_age_mae():
    plt.figure(figsize=(10, 6))
    for model_name, path in _available_age_logs().items():
        df = _read_csv(path)
        x = range(1, len(df) + 1) if model_name.startswith("ConvNeXt-Tiny Fine-tuned") else df["epoch"]
        plt.plot(x, df["val_mae"], marker="o", linewidth=2, label=model_name)
    plt.title("Age Estimation Validation MAE including Fine-tuning")
    plt.xlabel("Epoch")
    plt.ylabel("Validation MAE (years, lower is better)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "age_val_mae_comparison.png", dpi=180)
    plt.close()


def plot_age_loss():
    plt.figure(figsize=(10, 6))
    for model_name, path in _available_age_logs().items():
        df = _read_csv(path)
        x = range(1, len(df) + 1) if model_name.startswith("ConvNeXt-Tiny Fine-tuned") else df["epoch"]
        plt.plot(x, df["val_loss"], marker="o", linewidth=2, label=model_name)
    plt.title("Age Estimation Validation Loss including Fine-tuning")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss (lower is better)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "age_val_loss_comparison.png", dpi=180)
    plt.close()


def plot_age_best_bar():
    rows = []
    for model_name, path in _available_age_logs().items():
        df = _read_csv(path)
        best_row = df.loc[df["val_mae"].idxmin()]
        rows.append(
            {
                "model": model_name,
                "best_val_mae": float(best_row["val_mae"]),
                "best_epoch": int(best_row["epoch"]),
            }
        )
    summary = pd.DataFrame(rows).sort_values("best_val_mae")
    summary.to_csv(OUT_DIR / "age_model_best_summary.csv", index=False)

    plt.figure(figsize=(9, 5))
    bars = plt.bar(summary["model"], summary["best_val_mae"])
    plt.title("Best Validation MAE by Age Model")
    plt.xlabel("Model")
    plt.ylabel("Best Validation MAE (years, lower is better)")
    plt.grid(axis="y", alpha=0.3)
    for bar, mae, epoch in zip(bars, summary["best_val_mae"], summary["best_epoch"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.03,
            f"{mae:.2f}\nEp {epoch}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.tight_layout()
    plt.savefig(OUT_DIR / "age_model_best_mae_bar.png", dpi=180)
    plt.close()


def plot_age_test_improvement():
    rows = []
    for model_name, path in TEST_REPORTS.items():
        if not path.exists():
            continue
        metrics = _read_test_report(path)
        if "test_mae" in metrics:
            rows.append({"model": model_name, **metrics})
    if not rows:
        return
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "age_test_improvement_summary.csv", index=False)

    plt.figure(figsize=(9, 5))
    bars = plt.bar(summary["model"], summary["test_mae"], color=["#64748b", "#1f6feb"][: len(summary)])
    plt.title("ConvNeXt-Tiny Test MAE Before and After Fine-tuning")
    plt.xlabel("Model")
    plt.ylabel("Test MAE (years, lower is better)")
    plt.grid(axis="y", alpha=0.3)
    for bar, mae in zip(bars, summary["test_mae"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.03,
            f"{mae:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    if len(summary) >= 2:
        before = float(summary.iloc[0]["test_mae"])
        after = float(summary.iloc[-1]["test_mae"])
        diff = before - after
        plt.figtext(
            0.5,
            0.01,
            f"Improvement: {diff:.2f} years lower Test MAE",
            ha="center",
            fontsize=10,
            color="#166534" if diff > 0 else "#991b1b",
        )
    plt.tight_layout(rect=(0, 0.04, 1, 1))
    plt.savefig(OUT_DIR / "age_test_mae_improvement.png", dpi=180)
    plt.close()


def plot_gender_metrics():
    df = _read_csv(GENDER_LOG)
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(df["epoch"], df["train_acc"] * 100, marker="o", label="Train Accuracy")
    ax1.plot(df["epoch"], df["val_acc"] * 100, marker="o", label="Validation Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy (%)")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(df["epoch"], df["train_loss"], linestyle="--", color="tab:red", label="Train Loss")
    ax2.plot(df["epoch"], df["val_loss"], linestyle="--", color="tab:purple", label="Validation Loss")
    ax2.set_ylabel("Loss")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="center right")
    plt.title("Gender Classification Metrics")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "gender_training_metrics.png", dpi=180)
    plt.close()

    best_row = df.loc[df["val_acc"].idxmax()]
    pd.DataFrame(
        [
            {
                "model": "MobileNetV3-Small",
                "best_val_acc": float(best_row["val_acc"]),
                "best_epoch": int(best_row["epoch"]),
                "best_val_loss": float(best_row["val_loss"]),
            }
        ]
    ).to_csv(OUT_DIR / "gender_model_best_summary.csv", index=False)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_age_mae()
    plot_age_loss()
    plot_age_best_bar()
    plot_age_test_improvement()
    plot_gender_metrics()
    print(f"Saved visualizations to: {OUT_DIR}")


if __name__ == "__main__":
    main()

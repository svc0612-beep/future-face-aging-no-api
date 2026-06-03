from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
LOG_DIR = PROJECT_ROOT / "outputs" / "logs"
STATUS_PATH = LOG_DIR / "age_improvement_status.json"
CONSOLE_LOG = LOG_DIR / "age_improvement_pipeline_console.log"


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


def run_step(name: str, command: list[str]):
    write_status(status="running", pipeline_step=name, last_command=" ".join(command))
    with CONSOLE_LOG.open("a", encoding="utf-8", errors="replace") as log:
        log.write(f"\n\n===== {name} =====\n")
        log.flush()
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    if completed.returncode != 0:
        write_status(status="error", pipeline_step=name, error=f"returncode={completed.returncode}")
        raise RuntimeError(f"{name} failed with returncode={completed.returncode}")


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    write_status(
        status="running",
        task="age_improvement_pipeline",
        pipeline_step="start",
        started_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        deadline_note="내일 오전 8시 전 완료 목표 - 마감 우선 1 epoch 모드",
    )
    baseline_report = PROJECT_ROOT / "outputs" / "evaluation" / "test_report_convnext_tiny_full.txt"
    if baseline_report.exists():
        write_status(
            status="running",
            pipeline_step="baseline_test_eval",
            phase="completed",
            current_model="ConvNeXt-Tiny baseline",
            progress_percent=100,
            report_path=str(baseline_report),
            note="기존 Test 평가는 이미 완료되어 재사용합니다.",
        )
    else:
        run_step(
            "baseline_test_eval",
            [
                str(PYTHON),
                str(PROJECT_ROOT / "src" / "evaluate_age_model_convnext_tiny.py"),
                "--model-path",
                str(PROJECT_ROOT / "models" / "age_estimator" / "best_convnext_tiny_full.pth"),
                "--run-name",
                "convnext_tiny_full",
            ],
        )
    run_step(
        "low_lr_finetune",
        [
            str(PYTHON),
            str(PROJECT_ROOT / "src" / "train_age_model_convnext_tiny_finetune.py"),
            "--epochs",
            "1",
            "--patience",
            "1",
            "--learning-rate",
            "1e-5",
            "--batch-size",
            "8",
        ],
    )

    status = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    best_path = status.get("best_path", str(PROJECT_ROOT / "models" / "age_estimator" / "best_convnext_tiny_full.pth"))
    run_step(
        "finetuned_test_eval",
        [
            str(PYTHON),
            str(PROJECT_ROOT / "src" / "evaluate_age_model_convnext_tiny.py"),
            "--model-path",
            best_path,
            "--run-name",
            "convnext_tiny_finetune_lr1e5",
        ],
    )
    run_step(
        "refresh_visuals",
        [
            str(PYTHON),
            str(PROJECT_ROOT / "src" / "generate_model_summary_visuals.py"),
        ],
    )
    write_status(
        status="completed",
        task="age_improvement_pipeline",
        pipeline_step="completed",
        completed_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        progress_percent=100,
        eta_seconds=0,
        console_log=str(CONSOLE_LOG),
    )


if __name__ == "__main__":
    main()

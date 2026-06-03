from __future__ import annotations

from pathlib import Path
import json
import subprocess
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
LOG_DIR = PROJECT_ROOT / "outputs" / "logs"
STATUS_PATH = LOG_DIR / "age_improvement_status.json"
CONSOLE_LOG = LOG_DIR / "age_improvement_pipeline_console.log"


def read_status() -> dict:
    if not STATUS_PATH.exists():
        return {}
    try:
        return json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def write_status(**kwargs):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    previous = read_status()
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


def wait_for_first_epoch():
    write_status(
        status="running",
        pipeline_step="wait_first_epoch",
        current_model="ConvNeXt-Tiny fine-tuning",
        phase="첫 번째 추가 epoch 완료 대기",
        deadline_note="내일 오전 8시 전 완료 목표 - 총 2 epoch 마감 모드",
    )
    while True:
        status = read_status()
        if status.get("task") == "convnext_finetune" and status.get("phase") == "completed":
            return status
        time.sleep(20)


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    first_status = wait_for_first_epoch()
    first_best_path = first_status.get(
        "best_path",
        str(PROJECT_ROOT / "models" / "age_estimator" / "best_convnext_tiny_full.pth"),
    )

    run_step(
        "second_low_lr_finetune",
        [
            str(PYTHON),
            str(PROJECT_ROOT / "src" / "train_age_model_convnext_tiny_finetune.py"),
            "--base-checkpoint",
            first_best_path,
            "--epochs",
            "1",
            "--patience",
            "1",
            "--learning-rate",
            "1e-5",
            "--batch-size",
            "8",
            "--run-name",
            "convnext_tiny_finetune_lr1e5_epoch2",
        ],
    )

    second_status = read_status()
    best_path = second_status.get("best_path", first_best_path)
    run_step(
        "finetuned_test_eval",
        [
            str(PYTHON),
            str(PROJECT_ROOT / "src" / "evaluate_age_model_convnext_tiny.py"),
            "--model-path",
            best_path,
            "--run-name",
            "convnext_tiny_finetune_lr1e5_epoch2",
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
        current_model="ConvNeXt-Tiny fine-tuned total 2 epochs",
        phase="completed",
        progress_percent=100,
        eta_seconds=0,
        console_log=str(CONSOLE_LOG),
        deadline_note="내일 오전 8시 전 완료 목표 - 총 2 epoch 완료",
    )


if __name__ == "__main__":
    main()

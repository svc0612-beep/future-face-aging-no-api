from datetime import datetime
from pathlib import Path
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "outputs" / "logs"
CONVNEXT_LOG = LOG_DIR / "convnext_tiny_full_console.log"
EFFICIENTNET_LOG = LOG_DIR / "efficientnet_b0_full_console.log"
RUNNER_LOG = LOG_DIR / "efficientnet_b0_queue_runner.log"
TRAIN_SCRIPT = PROJECT_ROOT / "src" / "train_age_model_efficientnet_b0.py"


def write_status(message):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    with RUNNER_LOG.open("a", encoding="utf-8") as file:
        file.write(line + "\n")
    print(line, flush=True)


def convnext_finished():
    if not CONVNEXT_LOG.exists():
        return False
    return "training finished" in CONVNEXT_LOG.read_text(encoding="utf-16", errors="ignore")


def main():
    if EFFICIENTNET_LOG.exists() and "training finished" in EFFICIENTNET_LOG.read_text(encoding="utf-8", errors="ignore"):
        write_status("EfficientNet-B0 was already completed. Nothing to do.")
        return

    write_status("Waiting for ConvNeXt-Tiny training to finish.")
    while not convnext_finished():
        time.sleep(30)

    write_status("ConvNeXt-Tiny finished. Starting EfficientNet-B0 GPU training.")
    with EFFICIENTNET_LOG.open("w", encoding="utf-8") as console:
        result = subprocess.run(
            [
                sys.executable,
                "-B",
                str(TRAIN_SCRIPT),
                "--epochs",
                "15",
                "--batch-size",
                "8",
                "--patience",
                "2",
                "--run-name",
                "efficientnet_b0_full",
            ],
            cwd=PROJECT_ROOT,
            stdout=console,
            stderr=subprocess.STDOUT,
            check=False,
        )
    write_status(f"EfficientNet-B0 training process exited with code {result.returncode}.")


if __name__ == "__main__":
    main()

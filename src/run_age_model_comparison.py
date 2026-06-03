from datetime import datetime
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_EXE = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
LOG_DIR = PROJECT_ROOT / "outputs" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


RUNS = [
    {
        "name": "mobilenet_v3_full",
        "model": "mobilenet_v3_large",
    },
    {
        "name": "efficientnet_b0_full",
        "model": "efficientnet_b0",
    },
]


def run_training(run):
    command = [
        str(PYTHON_EXE),
        "-B",
        "src/train_age_model_mobilenet_v3.py",
        "--model",
        run["model"],
        "--epochs",
        "15",
        "--batch-size",
        "8",
        "--patience",
        "4",
        "--run-name",
        run["name"],
    ]

    log_path = LOG_DIR / f"{run['name']}_console.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"Started {run['name']} at {datetime.now().isoformat()}\n")
        log_file.write("Command: " + " ".join(command) + "\n\n")
        log_file.flush()

        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return_code = process.wait()

        log_file.write(f"\nFinished {run['name']} at {datetime.now().isoformat()}\n")
        log_file.write(f"Return code: {return_code}\n")
        log_file.flush()

    if return_code != 0:
        raise RuntimeError(f"{run['name']} failed with return code {return_code}. See {log_path}")


def main():
    overall_log = LOG_DIR / "age_model_comparison_runner.log"
    with overall_log.open("a", encoding="utf-8") as log_file:
        log_file.write("=" * 80 + "\n")
        log_file.write(f"Comparison run started at {datetime.now().isoformat()}\n")
        log_file.flush()

        for run in RUNS:
            log_file.write(f"Starting {run['name']}\n")
            log_file.flush()
            run_training(run)
            log_file.write(f"Completed {run['name']}\n")
            log_file.flush()

        log_file.write(f"Comparison run finished at {datetime.now().isoformat()}\n")
        log_file.flush()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(exc, file=sys.stderr)
        raise

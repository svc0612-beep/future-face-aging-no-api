from __future__ import annotations

from pathlib import Path
import os
import subprocess
import sys
import uuid

from PIL import Image

from src.aging_simulator import generate_future_faces


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAM_ROOT = PROJECT_ROOT / "third_party" / "SAM"
SAM_CHECKPOINT = SAM_ROOT / "pretrained_models" / "sam_ffhq_aging.pt"
JOB_ROOT = PROJECT_ROOT / "outputs" / "sam_jobs"
DEFAULT_YEARS = [10, 20, 30, 40, 50]


def _clamp_target_age(age: float) -> int:
    return int(max(0, min(round(age), 100)))


def _validate_sam_files() -> None:
    if not SAM_ROOT.exists():
        raise FileNotFoundError(f"SAM source directory was not found: {SAM_ROOT}")
    if not SAM_CHECKPOINT.exists():
        raise FileNotFoundError(f"SAM checkpoint was not found: {SAM_CHECKPOINT}")


def _run_sam(image: Image.Image, target_ages: list[int], resize_outputs: bool = True) -> dict[int, Image.Image]:
    _validate_sam_files()

    job_id = uuid.uuid4().hex
    job_dir = JOB_ROOT / job_id
    input_dir = job_dir / "input"
    output_dir = job_dir / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = input_dir / f"face_{job_id}.png"
    image.convert("RGB").save(input_path)

    command = [
        sys.executable,
        str(SAM_ROOT / "scripts" / "inference.py"),
        "--exp_dir",
        str(output_dir),
        "--checkpoint_path",
        str(SAM_CHECKPOINT),
        "--data_path",
        str(input_dir),
        "--test_batch_size",
        "1",
        "--test_workers",
        "0",
        "--n_images",
        "1",
        "--target_age",
        ",".join(str(age) for age in target_ages),
        "--couple_outputs",
    ]
    if resize_outputs:
        command.append("--resize_outputs")

    env_path = str(PROJECT_ROOT / ".venv" / "Scripts")
    completed = subprocess.run(
        command,
        cwd=SAM_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=420,
        env={**dict(os.environ), "Path": env_path + ";" + os.environ.get("Path", "")},
    )

    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout)[-4000:]
        raise RuntimeError(f"SAM generation failed.\n{detail}")

    results = {}
    for age in target_ages:
        result_path = output_dir / "inference_results" / str(age) / input_path.name
        if not result_path.exists():
            raise FileNotFoundError(f"SAM output was not found: {result_path}")
        results[age] = Image.open(result_path).convert("RGB")

    return results


def generate_future_faces_sam(image: Image.Image, current_age: float, years_list=None):
    if years_list is None:
        years_list = DEFAULT_YEARS

    target_ages = [_clamp_target_age(current_age + years) for years in years_list]

    try:
        generated = _run_sam(image, target_ages=target_ages)
    except Exception as exc:
        fallback_results = generate_future_faces(image, years_list=years_list)
        for item in fallback_results:
            item["engine"] = "rule_based_fallback"
            item["engine_note"] = str(exc)
            item["target_age"] = _clamp_target_age(current_age + item["years_later"])
        return fallback_results

    results = []
    for years, target_age in zip(years_list, target_ages):
        results.append(
            {
                "years_later": years,
                "target_age": target_age,
                "image": generated[target_age],
                "engine": "sam",
            }
        )

    return results

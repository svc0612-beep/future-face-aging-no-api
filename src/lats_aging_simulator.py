from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import uuid

from PIL import Image

from src.aging_simulator import generate_future_faces


DEFAULT_YEARS = [10, 20, 30, 40, 50]
AGE_ANCHORS = [1.0, 4.5, 8.0, 17.0, 34.5, 59.5]
INTERP_STEP = 0.25
TILE_SIZE = 256
ROW_SPACER = 10

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LATS_ROOT = PROJECT_ROOT / "third_party" / "Lifespan_Age_Transformation_Synthesis"
JOB_ROOT = PROJECT_ROOT / "outputs" / "lats_jobs"


def _checkpoint_name(style: str) -> str:
    return "females_model" if style == "female" else "males_model"


def _progression_ages(interp_step: float = INTERP_STEP) -> list[float]:
    ages = []
    steps_per_interval = round(1.0 / interp_step)
    for start_age, end_age in zip(AGE_ANCHORS, AGE_ANCHORS[1:]):
        for step_index in range(steps_per_interval):
            progress = step_index * interp_step
            ages.append(start_age * (1.0 - progress) + end_age * progress)
    ages.append(AGE_ANCHORS[-1])
    return ages


def _split_progression_row(row_path: Path, expected_tiles: int) -> list[Image.Image]:
    row = Image.open(row_path).convert("RGB")
    start_x = TILE_SIZE + ROW_SPACER
    required_width = start_x + expected_tiles * TILE_SIZE
    if row.height < TILE_SIZE or row.width < required_width:
        raise RuntimeError(
            f"LATS progression row has unexpected size: {row.size}, "
            f"expected at least ({required_width}, {TILE_SIZE})"
        )

    return [
        row.crop((start_x + index * TILE_SIZE, 0, start_x + (index + 1) * TILE_SIZE, TILE_SIZE))
        for index in range(expected_tiles)
    ]


def _run_lats(image: Image.Image, style: str) -> tuple[list[Image.Image], list[float]]:
    if not LATS_ROOT.exists():
        raise FileNotFoundError(f"LATS source directory was not found: {LATS_ROOT}")

    checkpoint_name = _checkpoint_name(style)
    checkpoint_dir = LATS_ROOT / "checkpoints" / checkpoint_name
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"LATS checkpoint was not found: {checkpoint_dir}")

    job_id = uuid.uuid4().hex
    job_dir = JOB_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    input_path = job_dir / f"face_{job_id}.png"
    list_path = job_dir / "input_images.txt"
    image.convert("RGB").save(input_path)
    list_path.write_text(str(input_path), encoding="utf-8")

    command = [
        sys.executable,
        "test.py",
        "--name",
        checkpoint_name,
        "--which_epoch",
        "latest",
        "--display_id",
        "0",
        "--traverse",
        "--interp_step",
        str(INTERP_STEP),
        "--image_path_file",
        str(list_path),
        "--in_the_wild",
        "--gpu_ids",
        "0",
    ]
    completed = subprocess.run(
        command,
        cwd=LATS_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=360,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout)[-4000:]
        raise RuntimeError(f"LATS generation failed.\n{detail}")

    row_path = (
        LATS_ROOT
        / "results"
        / checkpoint_name
        / "test_latest"
        / "traversal"
        / input_path.name
    )
    if not row_path.exists():
        raise FileNotFoundError(f"LATS output row was not found: {row_path}")

    ages = _progression_ages()
    return _split_progression_row(row_path, expected_tiles=len(ages)), ages


def generate_future_faces_lats(
    image: Image.Image,
    current_age: float,
    years_list=None,
    style: str = "male",
):
    if years_list is None:
        years_list = DEFAULT_YEARS

    try:
        progression_images, progression_ages = _run_lats(image, style=style)
    except Exception as exc:
        fallback_results = generate_future_faces(image, years_list=years_list)
        for item in fallback_results:
            item["engine"] = "rule_based_fallback"
            item["engine_note"] = str(exc)
        return fallback_results

    results = []
    for years in years_list:
        target_age = current_age + years
        closest_index = min(
            range(len(progression_ages)),
            key=lambda index: abs(progression_ages[index] - target_age),
        )
        selected_image = progression_images[closest_index]
        selected_age = progression_ages[closest_index]
        extrapolated = target_age > AGE_ANCHORS[-1]

        results.append(
            {
                "years_later": years,
                "image": selected_image,
                "engine": "lats",
                "selected_lats_age": selected_age,
                "extrapolated": extrapolated,
            }
        )

    return results

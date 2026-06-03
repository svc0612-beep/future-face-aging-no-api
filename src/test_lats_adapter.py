from pathlib import Path
import sys

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.lats_aging_simulator import generate_future_faces_lats


def main():
    input_path = PROJECT_ROOT / "outputs" / "aging_samples" / "original.png"
    output_dir = PROJECT_ROOT / "outputs" / "lats_adapter_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(input_path).convert("RGB")
    results = generate_future_faces_lats(image, current_age=35.0, style="male")

    summary = []
    for item in results:
        years = item["years_later"]
        item["image"].save(output_dir / f"future_{years}years.png")
        summary.append(
            (
                years,
                item["engine"],
                round(item.get("selected_lats_age", -1), 2),
                item.get("extrapolated"),
            )
        )

    print(summary)


if __name__ == "__main__":
    main()


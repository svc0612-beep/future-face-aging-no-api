from pathlib import Path
import pandas as pd
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import AIHUB_ROOT, PROCESSED_DIR, IMAGE_EXTS


def check_split(split_name):
    img_dir = AIHUB_ROOT / split_name / "01.원천데이터"
    label_dir = AIHUB_ROOT / split_name / "02.라벨링데이터"

    image_paths = [
        p for p in img_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]

    json_paths = [
        p for p in label_dir.rglob("*.json")
        if p.is_file()
    ]

    image_map = {p.stem: p for p in image_paths}
    json_map = {p.stem: p for p in json_paths}

    image_stems = set(image_map.keys())
    json_stems = set(json_map.keys())

    image_without_json = sorted(image_stems - json_stems)
    json_without_image = sorted(json_stems - image_stems)

    print(f"\n[{split_name}]")
    print(f"이미지 수: {len(image_paths):,}")
    print(f"JSON 수: {len(json_paths):,}")
    print(f"JSON 없는 이미지 수: {len(image_without_json):,}")
    print(f"이미지 없는 JSON 수: {len(json_without_image):,}")

    rows_img = []
    for stem in image_without_json:
        rows_img.append({
            "split": split_name,
            "stem": stem,
            "image_path": str(image_map[stem]),
        })

    rows_json = []
    for stem in json_without_image:
        rows_json.append({
            "split": split_name,
            "stem": stem,
            "json_path": str(json_map[stem]),
        })

    return rows_img, rows_json


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    all_img_rows = []
    all_json_rows = []

    for split in ["Training", "Validation"]:
        img_rows, json_rows = check_split(split)
        all_img_rows.extend(img_rows)
        all_json_rows.extend(json_rows)

    img_df = pd.DataFrame(all_img_rows)
    json_df = pd.DataFrame(all_json_rows)

    img_out = PROCESSED_DIR / "aihub_images_without_json.csv"
    json_out = PROCESSED_DIR / "aihub_json_without_images.csv"

    img_df.to_csv(img_out, index=False, encoding="utf-8-sig")
    json_df.to_csv(json_out, index=False, encoding="utf-8-sig")

    print("\n저장 완료:")
    print(img_out)
    print(json_out)

    print("\n이미지 쪽 샘플:")
    print(img_df.head(20))

    print("\nJSON 쪽 샘플:")
    print(json_df.head(20))


if __name__ == "__main__":
    main()
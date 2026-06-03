from pathlib import Path
import sys

import pandas as pd
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import PROCESSED_DIR, OUTPUTS_DIR
from aging_simulator import generate_future_faces


TEST_CSV = PROCESSED_DIR / "test_metadata.csv"
OUTPUT_DIR = OUTPUTS_DIR / "aging_samples"


def main():
    print("=" * 80)
    print("미래 얼굴 시뮬레이션 샘플 생성 테스트")
    print("=" * 80)

    if not TEST_CSV.exists():
        raise FileNotFoundError(f"test_metadata.csv가 없습니다: {TEST_CSV}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(TEST_CSV)

    print(f"test 데이터 수: {len(df):,}")
    print("컬럼:", list(df.columns))

    # 너무 어린 아이나 너무 고령 이미지는 피하고, 20~40대 샘플 중 하나 선택
    sample_df = df[
        (df["age"] >= 20) &
        (df["age"] <= 40)
    ].copy()

    if len(sample_df) == 0:
        sample_df = df.copy()

    # AIHub 이미지가 크고 얼굴이 비교적 선명한 경우가 많아서 우선 선택
    aihub_df = sample_df[sample_df["dataset"] == "AIHub"].copy()

    if len(aihub_df) > 0:
        row = aihub_df.sample(1, random_state=42).iloc[0]
    else:
        row = sample_df.sample(1, random_state=42).iloc[0]

    image_path = Path(row["image_path"])
    true_age = row["age"]
    dataset = row["dataset"]

    print("\n선택된 샘플 이미지:")
    print("dataset:", dataset)
    print("age:", true_age)
    print("image_path:", image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"이미지 파일이 없습니다: {image_path}")

    image = Image.open(image_path).convert("RGB")

    # 원본 저장
    original_out = OUTPUT_DIR / "original.png"
    image.save(original_out)

    print("\n원본 저장:", original_out)

    # 10년 후 ~ 50년 후 생성
    results = generate_future_faces(
        image,
        years_list=[10, 20, 30, 40, 50]
    )

    saved_paths = []

    for item in results:
        years = item["years_later"]
        aged_image = item["image"]

        out_path = OUTPUT_DIR / f"future_{years}years.png"
        aged_image.save(out_path)

        saved_paths.append(out_path)
        print(f"{years}년 후 이미지 저장:", out_path)

    print("\n전체 저장 완료")
    print(f"결과 폴더: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
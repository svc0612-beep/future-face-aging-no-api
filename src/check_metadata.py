from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMBINED_CSV = PROJECT_ROOT / "data" / "processed" / "combined_metadata.csv"
UTK_CSV = PROJECT_ROOT / "data" / "processed" / "utkface_metadata.csv"
AIHUB_CSV = PROJECT_ROOT / "data" / "processed" / "aihub_metadata.csv"

def check_csv(path):
    print(f"\n확인 파일: {path}")
    df = pd.read_csv(path)

    print("행 수:", len(df))
    print("컬럼:", list(df.columns))

    print("\n데이터셋별 개수:")
    if "dataset" in df.columns:
        print(df["dataset"].value_counts())

    print("\nsplit별 개수:")
    if "split" in df.columns:
        print(df["split"].value_counts(dropna=False))

    print("\nage 통계:")
    print(df["age"].describe())

    print("\n결측치 개수:")
    print(df.isna().sum())

    print("\n실제 이미지 파일 존재 여부 샘플 확인:")
    sample = df.sample(min(20, len(df)), random_state=42)
    missing = 0

    for p in sample["image_path"]:
        if not Path(p).exists():
            missing += 1
            print("없는 파일:", p)

    print(f"샘플 20개 중 없는 이미지: {missing}")

    return df

def main():
    utk = check_csv(UTK_CSV)
    aihub = check_csv(AIHUB_CSV)
    combined = check_csv(COMBINED_CSV)

    print("\n나이대별 개수:")
    bins = [0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 120]
    labels = ["0대", "10대", "20대", "30대", "40대", "50대", "60대", "70대", "80대", "90대+"]
    combined["age_group"] = pd.cut(combined["age"], bins=bins, labels=labels, include_lowest=True)
    print(combined["age_group"].value_counts().sort_index())

if __name__ == "__main__":
    main()
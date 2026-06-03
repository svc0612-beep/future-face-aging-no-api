from pathlib import Path
import sys
import re

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import PROCESSED_DIR, COMBINED_METADATA_CSV


FINAL_METADATA_CSV = PROCESSED_DIR / "final_metadata.csv"
TRAIN_METADATA_CSV = PROCESSED_DIR / "train_metadata.csv"
VAL_METADATA_CSV = PROCESSED_DIR / "val_metadata.csv"
TEST_METADATA_CSV = PROCESSED_DIR / "test_metadata.csv"


RANDOM_SEED = 42

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10


def make_age_group(age: int) -> str:
    """
    나이대 그룹 생성.
    너무 세밀하게 나누면 split이 불안정해질 수 있어서 10년 단위로 묶는다.
    """
    age = int(age)

    if age <= 9:
        return "00s"
    elif age <= 19:
        return "10s"
    elif age <= 29:
        return "20s"
    elif age <= 39:
        return "30s"
    elif age <= 49:
        return "40s"
    elif age <= 59:
        return "50s"
    elif age <= 69:
        return "60s"
    elif age <= 79:
        return "70s"
    elif age <= 89:
        return "80s"
    else:
        return "90plus"


def extract_file_stem(path_text: str) -> str:
    return Path(str(path_text)).stem


def extract_aihub_person_id(file_stem: str) -> str:
    """
    AI Hub 파일명 예:
    0039_1988_34_00000048_d

    여기서 첫 번째 토큰 0039를 사람 ID로 사용한다.
    이 가정은 파일명 샘플을 근거로 한 프로젝트용 처리 방식이다.
    """
    parts = str(file_stem).split("_")

    if len(parts) >= 1 and parts[0]:
        return parts[0]

    return file_stem


def make_group_id(row) -> str:
    """
    데이터 누수를 줄이기 위한 group_id 생성.

    AI Hub:
    - 같은 사람으로 보이는 ID를 기준으로 묶는다.
    - 파일명 첫 번째 토큰을 person_id로 사용.

    UTKFace:
    - 동일 인물 ID 정보가 명확하지 않으므로 이미지 1장 = 1그룹으로 둔다.
    """
    dataset = row["dataset"]
    stem = extract_file_stem(row["image_path"])

    if dataset == "AIHub":
        person_id = extract_aihub_person_id(stem)
        return f"AIHub_person_{person_id}"

    if dataset == "UTKFace":
        return f"UTKFace_image_{stem}"

    return f"{dataset}_{stem}"


def assign_group_splits(group_df: pd.DataFrame) -> pd.DataFrame:
    """
    group 단위로 train/val/test를 나눈다.

    stratify_key = dataset + age_group
    즉, 데이터셋별/나이대별 분포가 최대한 유지되도록 그룹 단위로 나눈다.
    """
    assigned_parts = []

    for key, sub in group_df.groupby("stratify_key"):
        sub = sub.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

        n = len(sub)

        if n == 1:
            sub["final_split"] = "train"

        elif n == 2:
            sub.loc[0, "final_split"] = "train"
            sub.loc[1, "final_split"] = "val"

        else:
            n_train = int(round(n * TRAIN_RATIO))
            n_val = int(round(n * VAL_RATIO))

            # 최소 조건 보정
            if n_train < 1:
                n_train = 1

            if n_val < 1:
                n_val = 1

            if n_train + n_val >= n:
                n_train = max(1, n - 2)
                n_val = 1

            train_end = n_train
            val_end = n_train + n_val

            sub["final_split"] = "test"
            sub.loc[:train_end - 1, "final_split"] = "train"
            sub.loc[train_end:val_end - 1, "final_split"] = "val"

        assigned_parts.append(sub)

    result = pd.concat(assigned_parts, ignore_index=True)
    return result


def main():
    print("[1] combined_metadata.csv 읽기")
    print(f"입력 파일: {COMBINED_METADATA_CSV}")

    if not COMBINED_METADATA_CSV.exists():
        raise FileNotFoundError(f"combined_metadata.csv가 없습니다: {COMBINED_METADATA_CSV}")

    df = pd.read_csv(COMBINED_METADATA_CSV)

    print(f"전체 데이터 수: {len(df):,}")
    print("컬럼:", list(df.columns))

    required_cols = ["image_path", "age", "dataset"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"필수 컬럼이 없습니다: {col}")

    print("\n[2] 기본 정리")
    df = df.dropna(subset=["image_path", "age", "dataset"]).copy()
    df["age"] = df["age"].astype(int)

    # 실제 파일 존재 여부 확인
    print("\n[3] 실제 이미지 경로 존재 여부 확인")
    df["image_exists"] = df["image_path"].apply(lambda p: Path(str(p)).exists())

    missing_count = (~df["image_exists"]).sum()
    print(f"존재하지 않는 이미지 수: {missing_count:,}")

    if missing_count > 0:
        missing_csv = PROCESSED_DIR / "missing_images.csv"
        df.loc[~df["image_exists"]].to_csv(missing_csv, index=False, encoding="utf-8-sig")
        print(f"없는 이미지 목록 저장: {missing_csv}")

        # 학습에는 존재하는 이미지만 사용
        df = df[df["image_exists"]].copy()

    print(f"이미지 존재 확인 후 데이터 수: {len(df):,}")

    print("\n[4] age_group / group_id 생성")
    df["age_group"] = df["age"].apply(make_age_group)
    df["group_id"] = df.apply(make_group_id, axis=1)

    # split용 stratify key
    df["stratify_key"] = df["dataset"].astype(str) + "_" + df["age_group"].astype(str)

    print("\n데이터셋별 개수:")
    print(df["dataset"].value_counts())

    print("\n나이대별 개수:")
    print(df["age_group"].value_counts().sort_index())

    print("\n데이터셋 x 나이대:")
    print(pd.crosstab(df["dataset"], df["age_group"]))

    print("\n[5] group 단위 테이블 생성")
    group_df = (
        df.groupby("group_id")
        .agg(
            dataset=("dataset", "first"),
            age_group=("age_group", "first"),
            stratify_key=("stratify_key", "first"),
            row_count=("image_path", "count"),
            min_age=("age", "min"),
            max_age=("age", "max"),
        )
        .reset_index()
    )

    print(f"전체 group 수: {len(group_df):,}")
    print("\n데이터셋별 group 수:")
    print(group_df["dataset"].value_counts())

    print("\n[6] train / val / test split 생성")
    assigned_group_df = assign_group_splits(group_df)

    split_map = dict(zip(assigned_group_df["group_id"], assigned_group_df["final_split"]))
    df["final_split"] = df["group_id"].map(split_map)

    if df["final_split"].isna().any():
        raise RuntimeError("일부 데이터에 final_split이 배정되지 않았습니다.")

    print("\n최종 split별 이미지 수:")
    print(df["final_split"].value_counts())

    print("\n최종 split 비율:")
    print((df["final_split"].value_counts(normalize=True) * 100).round(2))

    print("\n데이터셋별 split:")
    print(pd.crosstab(df["dataset"], df["final_split"]))

    print("\n나이대별 split:")
    print(pd.crosstab(df["age_group"], df["final_split"]))

    print("\n[7] group 누수 검사")
    group_split_counts = df.groupby("group_id")["final_split"].nunique()
    leakage_count = (group_split_counts > 1).sum()

    print(f"여러 split에 섞인 group 수: {leakage_count}")

    if leakage_count > 0:
        leakage_csv = PROCESSED_DIR / "group_leakage_check.csv"
        leakage_ids = group_split_counts[group_split_counts > 1].index
        df[df["group_id"].isin(leakage_ids)].to_csv(leakage_csv, index=False, encoding="utf-8-sig")
        print(f"누수 의심 목록 저장: {leakage_csv}")
        raise RuntimeError("group_id가 여러 split에 섞였습니다. split 로직 확인 필요.")

    print("\n[8] CSV 저장")

    # 기존 split 컬럼과 구분하기 위해 final_split을 split으로 최종 사용
    df["original_split"] = df.get("split", "unknown")
    df["split"] = df["final_split"]

    # 저장할 컬럼 순서 정리
    preferred_cols = [
        "image_path",
        "json_path",
        "age",
        "gender",
        "race",
        "dataset",
        "split",
        "original_split",
        "age_group",
        "group_id",
        "source_folder",
        "image_exists",
    ]

    existing_cols = [c for c in preferred_cols if c in df.columns]
    other_cols = [c for c in df.columns if c not in existing_cols]
    df = df[existing_cols + other_cols].copy()

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    df.to_csv(FINAL_METADATA_CSV, index=False, encoding="utf-8-sig")
    train_df.to_csv(TRAIN_METADATA_CSV, index=False, encoding="utf-8-sig")
    val_df.to_csv(VAL_METADATA_CSV, index=False, encoding="utf-8-sig")
    test_df.to_csv(TEST_METADATA_CSV, index=False, encoding="utf-8-sig")

    print(f"저장 완료: {FINAL_METADATA_CSV}")
    print(f"저장 완료: {TRAIN_METADATA_CSV}")
    print(f"저장 완료: {VAL_METADATA_CSV}")
    print(f"저장 완료: {TEST_METADATA_CSV}")

    print("\n최종 저장 개수:")
    print(f"train: {len(train_df):,}")
    print(f"val  : {len(val_df):,}")
    print(f"test : {len(test_df):,}")

    print("\n전체 작업 완료")


if __name__ == "__main__":
    main()
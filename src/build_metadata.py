import json
import re
from pathlib import Path

import pandas as pd

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import (
    UTKFACE_ROOT,
    AIHUB_ROOT,
    PROCESSED_DIR,
    UTKFACE_METADATA_CSV,
    AIHUB_METADATA_CSV,
    COMBINED_METADATA_CSV,
    IMAGE_EXTS,
)


def safe_read_json(json_path: Path):
    """
    JSON 파일을 안전하게 읽는다.
    인코딩이 다를 수 있으므로 여러 인코딩을 순서대로 시도한다.
    """
    for enc in ["utf-8-sig", "utf-8", "cp949"]:
        try:
            with open(json_path, "r", encoding=enc) as f:
                return json.load(f)
        except Exception:
            continue
    return None


def recursive_find_value(obj, target_keywords):
    """
    JSON 전체를 재귀 탐색하면서 특정 키워드가 들어간 값을 찾는다.
    예:
    target_keywords = ["age", "나이", "연령"]
    """
    found = []

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                k_lower = str(k).lower()

                if any(keyword.lower() in k_lower for keyword in target_keywords):
                    found.append(v)

                walk(v)

        elif isinstance(x, list):
            for item in x:
                walk(item)

    walk(obj)
    return found


def normalize_age(value):
    """
    나이 값을 정수로 변환한다.
    예:
    34
    "34"
    "34세"
    """
    if value is None:
        return None

    if isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        age = int(value)
        if 0 <= age <= 120:
            return age
        return None

    if isinstance(value, str):
        m = re.search(r"\d+", value)
        if m:
            age = int(m.group())
            if 0 <= age <= 120:
                return age

    return None


def extract_age_from_json(data):
    """
    AI Hub JSON에서 나이 값을 추출한다.
    JSON 구조가 조금 달라도 최대한 찾을 수 있도록 전체를 탐색한다.
    """
    if data is None:
        return None

    age_candidates = recursive_find_value(
        data,
        target_keywords=[
            "age",
            "나이",
            "연령",
            "aged",
        ],
    )

    for value in age_candidates:
        age = normalize_age(value)
        if age is not None:
            return age

    return None


def normalize_gender(value):
    """
    성별 값을 최대한 표준화한다.
    UTKFace 기준:
    0 = male
    1 = female
    """
    if value is None:
        return None

    if isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        if int(value) in [0, 1]:
            return int(value)
        return str(value)

    if isinstance(value, str):
        v = value.strip().lower()

        male_words = ["m", "male", "man", "남", "남자", "남성"]
        female_words = ["f", "female", "woman", "여", "여자", "여성"]

        if v in male_words:
            return 0
        if v in female_words:
            return 1

        return value.strip()

    return str(value)


def extract_gender_from_json(data):
    """
    AI Hub JSON에서 성별 값을 추출한다.
    없으면 None으로 둔다.
    """
    if data is None:
        return None

    gender_candidates = recursive_find_value(
        data,
        target_keywords=[
            "gender",
            "sex",
            "성별",
        ],
    )

    for value in gender_candidates:
        gender = normalize_gender(value)
        if gender is not None:
            return gender

    return None


def build_utkface_metadata():
    """
    UTKFace 파일명 형식:
    [age]_[gender]_[race]_[date&time].jpg

    예:
    32_1_2_20170116192028254.jpg

    의미:
    age = 32
    gender = 1
    race = 2
    """
    print("\n[1] UTKFace metadata 생성 시작")
    print(f"UTKFACE_ROOT: {UTKFACE_ROOT}")

    image_paths = [
        p for p in UTKFACE_ROOT.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]

    rows = []
    bad_files = []

    for img_path in image_paths:
        name = img_path.name

        # age_gender_race_ 로 시작하는 파일만 사용
        m = re.match(r"^(\d+)_(\d+)_(\d+)_", name)

        if not m:
            bad_files.append(str(img_path))
            continue

        age = int(m.group(1))
        gender = int(m.group(2))
        race = int(m.group(3))

        if not (0 <= age <= 120):
            bad_files.append(str(img_path))
            continue

        rows.append({
            "image_path": str(img_path),
            "json_path": None,
            "age": age,
            "gender": gender,
            "race": race,
            "dataset": "UTKFace",
            "split": "unknown",
            "source_folder": img_path.parent.name,
        })

    df = pd.DataFrame(rows)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(UTKFACE_METADATA_CSV, index=False, encoding="utf-8-sig")

    print(f"UTKFace 전체 이미지 파일 수: {len(image_paths):,}")
    print(f"UTKFace 정상 파싱 수: {len(df):,}")
    print(f"UTKFace 제외 파일 수: {len(bad_files):,}")
    print(f"저장 완료: {UTKFACE_METADATA_CSV}")

    if len(df) > 0:
        print("\nUTKFace 샘플:")
        print(df.head())

    return df


def collect_json_map(label_dir: Path):
    """
    JSON 파일을 stem 기준으로 매핑한다.

    중요:
    AI Hub 일부 파일은 이미지명 끝이 소문자 d/f이고,
    JSON명 끝이 대문자 D/F인 경우가 있다.

    예:
    이미지: 0039_1988_34_00000048_d.png
    JSON : 0039_1988_34_00000048_D.json

    그래서 stem을 lower() 처리해서 대소문자 차이를 무시하고 매칭한다.
    """
    json_paths = [
        p for p in label_dir.rglob("*.json")
        if p.is_file()
    ]

    json_map = {}
    duplicate_count = 0

    for jp in json_paths:
        stem = jp.stem.lower()

        if stem in json_map:
            duplicate_count += 1
            continue

        json_map[stem] = jp

    return json_map, len(json_paths), duplicate_count


def build_aihub_split(split_name: str):
    """
    AI Hub split 처리.
    split_name: Training 또는 Validation
    """
    img_dir = AIHUB_ROOT / split_name / "01.원천데이터"
    label_dir = AIHUB_ROOT / split_name / "02.라벨링데이터"

    print(f"\nAI Hub {split_name} 처리")
    print(f"이미지 폴더: {img_dir}")
    print(f"라벨 폴더: {label_dir}")

    if not img_dir.exists():
        print(f"[경고] 이미지 폴더 없음: {img_dir}")
        return []

    if not label_dir.exists():
        print(f"[경고] 라벨 폴더 없음: {label_dir}")
        return []

    image_paths = [
        p for p in img_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]

    json_map, total_json_count, duplicate_json_count = collect_json_map(label_dir)

    print(f"{split_name} 이미지 수: {len(image_paths):,}")
    print(f"{split_name} JSON 수: {total_json_count:,}")
    print(f"{split_name} JSON stem 중복 수: {duplicate_json_count:,}")

    rows = []
    no_json_count = 0
    no_age_count = 0

    for img_path in image_paths:
        # 중요:
        # 이미지 stem도 lower() 처리해서 JSON stem과 대소문자 무시 매칭
        stem = img_path.stem.lower()
        json_path = json_map.get(stem)

        if json_path is None:
            no_json_count += 1
            continue

        data = safe_read_json(json_path)
        age = extract_age_from_json(data)
        gender = extract_gender_from_json(data)

        if age is None:
            no_age_count += 1
            continue

        rows.append({
            "image_path": str(img_path),
            "json_path": str(json_path),
            "age": age,
            "gender": gender,
            "race": None,
            "dataset": "AIHub",
            "split": "train" if split_name == "Training" else "val",
            "source_folder": img_path.parent.name,
        })

    print(f"{split_name} 매핑 성공 수: {len(rows):,}")
    print(f"{split_name} JSON 매칭 실패 수: {no_json_count:,}")
    print(f"{split_name} age 추출 실패 수: {no_age_count:,}")

    return rows


def build_aihub_metadata():
    print("\n[2] AI Hub metadata 생성 시작")
    print(f"AIHUB_ROOT: {AIHUB_ROOT}")

    train_rows = build_aihub_split("Training")
    val_rows = build_aihub_split("Validation")

    rows = train_rows + val_rows
    df = pd.DataFrame(rows)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(AIHUB_METADATA_CSV, index=False, encoding="utf-8-sig")

    print(f"\nAI Hub 최종 매핑 수: {len(df):,}")
    print(f"저장 완료: {AIHUB_METADATA_CSV}")

    if len(df) > 0:
        print("\nAI Hub 샘플:")
        print(df.head())

    return df


def build_combined_metadata(utk_df, aihub_df):
    print("\n[3] combined metadata 생성 시작")

    combined = pd.concat([utk_df, aihub_df], ignore_index=True)

    # age가 없는 행 제거
    combined = combined.dropna(subset=["age"]).copy()
    combined["age"] = combined["age"].astype(int)

    combined.to_csv(COMBINED_METADATA_CSV, index=False, encoding="utf-8-sig")

    print(f"통합 데이터 수: {len(combined):,}")
    print(f"저장 완료: {COMBINED_METADATA_CSV}")

    print("\n데이터셋별 개수:")
    print(combined["dataset"].value_counts())

    print("\n나이 통계:")
    print(combined["age"].describe())

    return combined


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    utk_df = build_utkface_metadata()
    aihub_df = build_aihub_metadata()
    combined_df = build_combined_metadata(utk_df, aihub_df)

    print("\n전체 작업 완료")
    print(f"UTKFace CSV: {UTKFACE_METADATA_CSV}")
    print(f"AI Hub CSV: {AIHUB_METADATA_CSV}")
    print(f"Combined CSV: {COMBINED_METADATA_CSV}")


if __name__ == "__main__":
    main()
from pathlib import Path
import sys

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import PROCESSED_DIR


TRAIN_CSV = PROCESSED_DIR / "train_metadata.csv"
VAL_CSV = PROCESSED_DIR / "val_metadata.csv"
TEST_CSV = PROCESSED_DIR / "test_metadata.csv"


class AgeDataset(Dataset):
    def __init__(self, csv_path, transform=None, max_samples=None):
        self.df = pd.read_csv(csv_path)

        if max_samples is not None:
            self.df = self.df.head(max_samples).copy()

        self.transform = transform

        required_cols = ["image_path", "age", "dataset", "split"]
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"필수 컬럼이 없습니다: {col}")

        self.df["age"] = self.df["age"].astype(float)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = row["image_path"]
        age = float(row["age"])
        dataset_name = row["dataset"]

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"이미지 열기 실패: {image_path}\n에러: {e}")

        if self.transform is not None:
            image = self.transform(image)

        age_tensor = torch.tensor(age, dtype=torch.float32)

        return {
            "image": image,
            "age": age_tensor,
            "dataset": dataset_name,
            "image_path": image_path,
        }


def check_csv_basic(csv_path):
    print("\n" + "=" * 80)
    print(f"CSV 확인: {csv_path}")
    print("=" * 80)

    df = pd.read_csv(csv_path)

    print(f"행 수: {len(df):,}")
    print("컬럼:", list(df.columns))

    print("\n데이터셋별 개수:")
    print(df["dataset"].value_counts())

    print("\nsplit별 개수:")
    print(df["split"].value_counts())

    print("\nage 통계:")
    print(df["age"].describe())

    print("\n이미지 경로 샘플 5개:")
    for p in df["image_path"].head(5):
        print(p)

    missing_count = 0
    for p in df["image_path"].head(100):
        if not Path(str(p)).exists():
            missing_count += 1

    print(f"\n앞 100개 중 존재하지 않는 이미지 수: {missing_count}")

    return df


def check_image_open(csv_path, sample_count=20):
    print("\n" + "=" * 80)
    print(f"이미지 열기 테스트: {csv_path}")
    print("=" * 80)

    df = pd.read_csv(csv_path)
    sample_df = df.sample(min(sample_count, len(df)), random_state=42)

    success = 0
    fail = 0

    for _, row in sample_df.iterrows():
        image_path = row["image_path"]

        try:
            img = Image.open(image_path).convert("RGB")
            print(f"OK | size={img.size} | age={row['age']} | dataset={row['dataset']} | {Path(image_path).name}")
            success += 1
        except Exception as e:
            print(f"FAIL | {image_path} | {e}")
            fail += 1

    print(f"\n이미지 열기 성공: {success}")
    print(f"이미지 열기 실패: {fail}")


def check_dataloader(csv_path, batch_size=16):
    print("\n" + "=" * 80)
    print(f"DataLoader 테스트: {csv_path}")
    print("=" * 80)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    dataset = AgeDataset(
        csv_path=csv_path,
        transform=transform,
        max_samples=128,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    batch = next(iter(loader))

    images = batch["image"]
    ages = batch["age"]
    datasets = batch["dataset"]
    image_paths = batch["image_path"]

    print("배치 이미지 shape:", images.shape)
    print("배치 age shape:", ages.shape)
    print("age dtype:", ages.dtype)
    print("image dtype:", images.dtype)

    print("\n배치 age 샘플:")
    print(ages[:10])

    print("\n배치 dataset 샘플:")
    print(datasets[:10])

    print("\n배치 image_path 샘플:")
    for p in image_paths[:5]:
        print(p)

    print("\nDataLoader 테스트 성공")


def main():
    print("PyTorch 버전:", torch.__version__)
    print("CUDA 사용 가능:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU 이름:", torch.cuda.get_device_name(0))

    for csv_path in [TRAIN_CSV, VAL_CSV, TEST_CSV]:
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV 파일이 없습니다: {csv_path}")

    check_csv_basic(TRAIN_CSV)
    check_csv_basic(VAL_CSV)
    check_csv_basic(TEST_CSV)

    check_image_open(TRAIN_CSV, sample_count=20)
    check_image_open(VAL_CSV, sample_count=20)
    check_image_open(TEST_CSV, sample_count=20)

    check_dataloader(TRAIN_CSV, batch_size=16)
    check_dataloader(VAL_CSV, batch_size=16)
    check_dataloader(TEST_CSV, batch_size=16)

    print("\n전체 이미지 로딩 / DataLoader 점검 완료")


if __name__ == "__main__":
    main()
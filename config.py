from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# 원본 데이터 경로
UTKFACE_ROOT = PROJECT_ROOT / "data" / "raw" / "utkface"
AIHUB_ROOT = PROJECT_ROOT / "data" / "raw" / "aihub_aging"

# 결과 저장 경로
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# CSV 저장 경로
UTKFACE_METADATA_CSV = PROCESSED_DIR / "utkface_metadata.csv"
AIHUB_METADATA_CSV = PROCESSED_DIR / "aihub_metadata.csv"
COMBINED_METADATA_CSV = PROCESSED_DIR / "combined_metadata.csv"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
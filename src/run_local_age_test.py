from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

from PIL import Image
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.face_preprocessing import FaceNotDetectedError, crop_largest_face
from src.predict_age import load_age_model, predict_age_from_pil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="정면 얼굴 이미지 경로")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / "outputs" / "local_age_tests" / f"test_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.txt"

    print("[1/3] 입력 이미지 확인")
    image = Image.open(args.image).convert("RGB")
    image.save(output_dir / "original.png")

    print("[2/3] 정면 얼굴 검출")
    try:
        crop_result = crop_largest_face(image)
    except FaceNotDetectedError as exc:
        report_path.write_text(
            f"status=REJECTED\ninput={args.image}\nreason={exc}\n",
            encoding="utf-8",
        )
        print(f"입력 거부: {exc}")
        print(f"보고서: {report_path}")
        return 2

    crop_result.image.save(output_dir / "face_crop.png")

    print("[3/3] ConvNeXt-Tiny 현재 나이 예측")
    model, device, model_info = load_age_model(device=torch.device("cpu"))
    predicted_age = predict_age_from_pil(crop_result.image, model, device)
    report_path.write_text(
        "\n".join(
            [
                "status=SUCCESS",
                f"input={args.image}",
                f"face_detector={crop_result.detector}",
                f"crop_box={crop_result.box}",
                f"age_model={model_info.get('model_name')}",
                f"best_validation_mae={model_info.get('best_val_mae')}",
                f"best_epoch={model_info.get('epoch')}",
                f"predicted_age={predicted_age:.2f}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"현재 예측 나이: {predicted_age:.1f}세")
    print(f"얼굴 crop: {output_dir / 'face_crop.png'}")
    print(f"보고서: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


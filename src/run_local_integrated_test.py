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
from src.predict_gender import MODEL_PATH as GENDER_MODEL_PATH
from src.predict_gender import load_gender_model, predict_gender_from_pil


def _write_report(path: Path, lines: list[str]):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="정면 얼굴 이미지 경로")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / "outputs" / "local_integrated_tests" / f"test_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.txt"
    report_lines = [f"input={args.image}"]

    print("[1/5] 입력 이미지 확인")
    image = Image.open(args.image).convert("RGB")
    image.save(output_dir / "original.png")

    print("[2/5] 정면 얼굴 검출")
    try:
        crop_result = crop_largest_face(image)
    except FaceNotDetectedError as exc:
        report_lines.extend(["status=REJECTED", f"reason={exc}"])
        _write_report(report_path, report_lines)
        print(f"입력 거부: {exc}")
        print(f"보고서: {report_path}")
        return 2

    face_crop = crop_result.image
    face_crop.save(output_dir / "face_crop.png")
    report_lines.extend([f"face_detector={crop_result.detector}", f"crop_box={crop_result.box}"])

    print("[3/5] ConvNeXt-Tiny 현재 나이 예측")
    age_model, age_device, age_info = load_age_model(device=torch.device("cpu"))
    predicted_age = predict_age_from_pil(face_crop, age_model, age_device)
    report_lines.extend(
        [
            f"age_model={age_info.get('model_name')}",
            f"best_validation_mae={age_info.get('best_val_mae')}",
            f"predicted_age={predicted_age:.2f}",
        ]
    )
    print(f"현재 예측 나이: {predicted_age:.1f}세")

    print("[4/5] MobileNetV3-Small 성별 자동 판별")
    if not GENDER_MODEL_PATH.exists():
        report_lines.extend(
            [
                "status=WAITING_FOR_GENDER_MODEL",
                f"missing_model={GENDER_MODEL_PATH}",
            ]
        )
        _write_report(report_path, report_lines)
        print("성별 판별 모델이 현재 학습 중입니다.")
        print(f"보고서: {report_path}")
        return 3

    gender_model, gender_device, _ = load_gender_model(device=torch.device("cpu"))
    gender_result = predict_gender_from_pil(face_crop, gender_model, gender_device)
    report_lines.extend(
        [
            f"gender={gender_result['label']}",
            f"gender_confidence={gender_result['confidence']:.4f}",
            f"probability_female={gender_result['probability_female']:.4f}",
        ]
    )
    print(f"성별 판별: {gender_result['label']} / 신뢰도 {gender_result['confidence']:.1%}")

    print("[5/5] 미래 얼굴 생성기 상태 확인")
    report_lines.extend(
        [
            "status=WAITING_FOR_APPROVED_AGING_GENERATOR",
            "aging_generator=LATS_REJECTED",
            "reason=LATS failed to preserve Korean identity and produced misleading results.",
        ]
    )
    _write_report(report_path, report_lines)
    print("현재 LATS 생성기는 기준 미달로 제외했습니다.")
    print("한국인 얼굴 특징을 유지하는 생성기 검증이 끝난 뒤 이 단계가 자동으로 이어집니다.")
    print(f"얼굴 crop: {output_dir / 'face_crop.png'}")
    print(f"보고서: {report_path}")
    return 4


if __name__ == "__main__":
    raise SystemExit(main())


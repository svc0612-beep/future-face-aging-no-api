from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

from PIL import Image, ImageDraw
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.face_preprocessing import FaceNotDetectedError, crop_largest_face
from src.lats_aging_simulator import generate_future_faces_lats
from src.predict_age import load_age_model, predict_age_from_pil
from src.predict_gender import MODEL_PATH as GENDER_MODEL_PATH
from src.predict_gender import load_gender_model, predict_gender_from_pil


def _make_contact_sheet(
    original: Image.Image,
    face_crop: Image.Image,
    predicted_age: float,
    future_results,
) -> Image.Image:
    tile_size = 256
    padding = 18
    header_height = 92
    caption_height = 48
    items = [("입력 원본", original), ("검출 얼굴", face_crop)]
    items.extend(
        (f"{item['years_later']}년 뒤 / {predicted_age + item['years_later']:.1f}세", item["image"])
        for item in future_results
    )

    width = padding + len(items) * (tile_size + padding)
    height = header_height + tile_size + caption_height + padding
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((padding, 18), f"로컬 미래 얼굴 검사 결과 / 현재 예측 나이: {predicted_age:.1f}세", fill="black")
    draw.text((padding, 48), "LATS 기반 AI 시뮬레이션이며 실제 미래 모습과 다를 수 있습니다.", fill=(100, 70, 35))

    for index, (caption, image) in enumerate(items):
        x = padding + index * (tile_size + padding)
        prepared = image.convert("RGB").resize((tile_size, tile_size), Image.Resampling.LANCZOS)
        sheet.paste(prepared, (x, header_height))
        draw.text((x, header_height + tile_size + 12), caption, fill="black")

    return sheet


def _write_report(path: Path, lines: list[str]):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="정면 얼굴 이미지 경로")
    parser.add_argument("--style", choices=["auto", "male", "female"], default="auto")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / "outputs" / "local_future_face_tests" / f"test_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.txt"
    report_lines = [f"input={args.image}", f"style={args.style}"]

    print("[1/5] 입력 이미지 확인")
    image = Image.open(args.image).convert("RGB")
    image.save(output_dir / "original.png")

    print("[2/5] 정면 얼굴 검출 및 crop")
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
    report_lines.extend([f"detector={crop_result.detector}", f"crop_box={crop_result.box}"])

    print("[3/5] ConvNeXt-Tiny 현재 나이 예측")
    model, device, model_info = load_age_model(device=torch.device("cpu"))
    predicted_age = predict_age_from_pil(face_crop, model, device)
    report_lines.extend([f"age_model={model_info.get('model_name')}", f"predicted_age={predicted_age:.2f}"])
    print(f"현재 예측 나이: {predicted_age:.1f}세")

    print("[4/6] MobileNetV3-Small 성별 판별")
    if args.style == "auto":
        if not GENDER_MODEL_PATH.exists():
            report_lines.extend(["status=WAITING_FOR_GENDER_MODEL", f"missing_model={GENDER_MODEL_PATH}"])
            _write_report(report_path, report_lines)
            print(f"성별 판별 모델 학습이 필요합니다: {GENDER_MODEL_PATH}")
            print(f"보고서: {report_path}")
            return 3
        gender_model, gender_device, _ = load_gender_model(device=torch.device("cpu"))
        gender_result = predict_gender_from_pil(face_crop, gender_model, gender_device)
        style = gender_result["label"]
        report_lines.extend(
            [
                f"gender={style}",
                f"gender_confidence={gender_result['confidence']:.4f}",
                f"probability_female={gender_result['probability_female']:.4f}",
            ]
        )
        print(f"성별 판별: {style} / 신뢰도 {gender_result['confidence']:.1%}")
    else:
        style = args.style
        report_lines.append(f"gender_override={style}")

    print("[5/6] LATS 비교용 미래 얼굴 5장 생성")
    future_results = generate_future_faces_lats(
        face_crop,
        current_age=predicted_age,
        years_list=[10, 20, 30, 40, 50],
        style=style,
    )
    for item in future_results:
        years = item["years_later"]
        item["image"].save(output_dir / f"future_{years}years.png")
        report_lines.append(
            f"future_{years}years=engine:{item.get('engine')},"
            f"target_age:{predicted_age + years:.2f},"
            f"selected_lats_age:{item.get('selected_lats_age')},"
            f"extrapolated:{item.get('extrapolated')}"
        )

    print("[6/6] 한눈에 보는 결과판 저장")
    contact_sheet = _make_contact_sheet(image, face_crop, predicted_age, future_results)
    contact_sheet.save(output_dir / "contact_sheet.png")
    report_lines.append("status=SUCCESS")
    _write_report(report_path, report_lines)

    print(f"완료 폴더: {output_dir}")
    print(f"결과판: {output_dir / 'contact_sheet.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.predict_age import load_age_model, predict_age_from_pil


def detect_largest_face_crop(image: Image.Image):
    image = image.convert("RGB")
    width, height = image.size
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=5,
        minSize=(max(48, width // 12), max(48, height // 12)),
    )
    if not len(faces):
        return None, []

    x, y, face_width, face_height = max(faces, key=lambda face: face[2] * face[3])
    side = int(max(face_width, face_height) * 1.75)
    cx = x + face_width // 2
    cy = y + face_height // 2 - int(face_height * 0.04)
    left = max(0, cx - side // 2)
    top = max(0, cy - side // 2)
    right = min(width, left + side)
    bottom = min(height, top + side)
    left = max(0, right - side)
    top = max(0, bottom - side)
    return image.crop((left, top, right, bottom)), faces


def main():
    job_dirs = sorted((PROJECT_ROOT / "outputs" / "lats_jobs").iterdir(), key=lambda path: path.stat().st_mtime)
    image_path = next(job_dirs[-1].glob("face_*.png"))
    image = Image.open(image_path).convert("RGB")
    face_crop, faces = detect_largest_face_crop(image)

    print("source:", image_path)
    print("source_size:", image.size)
    print("detected_faces:", len(faces))
    if face_crop is None:
        return

    diagnostic_dir = PROJECT_ROOT / "outputs" / "diagnostics"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    crop_path = diagnostic_dir / "latest_detected_face_crop.png"
    face_crop.save(crop_path)

    model, device, _ = load_age_model(device=torch.device("cpu"))
    print("detected_crop_size:", face_crop.size)
    print("source_prediction:", round(predict_age_from_pil(image, model, device), 2))
    print("detected_crop_prediction:", round(predict_age_from_pil(face_crop, model, device), 2))
    print("saved_crop:", crop_path)


if __name__ == "__main__":
    main()


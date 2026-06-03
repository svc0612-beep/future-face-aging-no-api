from __future__ import annotations

from dataclasses import dataclass

import cv2
import dlib
import numpy as np
from PIL import Image


class FaceNotDetectedError(ValueError):
    pass


@dataclass(frozen=True)
class FaceCropResult:
    image: Image.Image
    detector: str
    box: tuple[int, int, int, int]


def _opencv_faces(rgb_array: np.ndarray) -> list[tuple[int, int, int, int]]:
    height, width = rgb_array.shape[:2]
    gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
    min_face_size = (max(48, width // 12), max(48, height // 12))
    faces = []

    for cascade_file in [
        "haarcascade_frontalface_default.xml",
        "haarcascade_frontalface_alt2.xml",
    ]:
        detector = cv2.CascadeClassifier(cv2.data.haarcascades + cascade_file)
        detected = detector.detectMultiScale(
            gray,
            scaleFactor=1.06,
            minNeighbors=4,
            minSize=min_face_size,
        )
        faces.extend(tuple(map(int, face)) for face in detected)

    return faces


def _dlib_faces(rgb_array: np.ndarray) -> list[tuple[int, int, int, int]]:
    detector = dlib.get_frontal_face_detector()
    detected = detector(rgb_array, 1)
    return [
        (face.left(), face.top(), face.width(), face.height())
        for face in detected
    ]


def _expanded_square_box(
    box: tuple[int, int, int, int],
    image_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    x, y, face_width, face_height = box
    width, height = image_size
    side = int(max(face_width, face_height) * 1.75)
    cx = x + face_width // 2
    cy = y + face_height // 2 - int(face_height * 0.04)

    left = max(0, cx - side // 2)
    top = max(0, cy - side // 2)
    right = min(width, left + side)
    bottom = min(height, top + side)
    left = max(0, right - side)
    top = max(0, bottom - side)
    return left, top, right, bottom


def crop_largest_face(image: Image.Image) -> FaceCropResult:
    image = image.convert("RGB")
    rgb_array = np.array(image)

    candidates = _opencv_faces(rgb_array)
    detector_name = "opencv"
    if not candidates:
        candidates = _dlib_faces(rgb_array)
        detector_name = "dlib"

    if not candidates:
        raise FaceNotDetectedError(
            "정면 얼굴을 찾지 못했습니다. 카메라를 바로 보고 얼굴을 크게 촬영해 주세요. "
            "선글라스, 모자, 강한 역광은 피하는 것이 좋습니다."
        )

    largest = max(candidates, key=lambda face: face[2] * face[3])
    crop_box = _expanded_square_box(largest, image.size)
    return FaceCropResult(
        image=image.crop(crop_box),
        detector=detector_name,
        box=crop_box,
    )


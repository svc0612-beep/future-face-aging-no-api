from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


DEFAULT_YEARS = [10, 20, 30, 40, 50]
MAX_PROCESS_SIZE = 1400


def pil_to_array(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("RGB")).astype(np.float32)


def array_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def smoothstep(edge0: float, edge1: float, x):
    x = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _face_box(width: int, height: int):
    return {
        "left": int(width * 0.22),
        "right": int(width * 0.78),
        "top": int(height * 0.18),
        "bottom": int(height * 0.82),
        "cx": width // 2,
    }


def bilinear_sample(arr: np.ndarray, sample_x: np.ndarray, sample_y: np.ndarray) -> np.ndarray:
    h, w, _ = arr.shape
    sample_x = np.clip(sample_x, 0, w - 1)
    sample_y = np.clip(sample_y, 0, h - 1)

    x0 = np.floor(sample_x).astype(np.int32)
    y0 = np.floor(sample_y).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)

    wx = (sample_x - x0)[..., None]
    wy = (sample_y - y0)[..., None]

    top = arr[y0, x0] * (1 - wx) + arr[y0, x1] * wx
    bottom = arr[y1, x0] * (1 - wx) + arr[y1, x1] * wx
    return top * (1 - wy) + bottom * wy


def apply_age_shape_warp(image: Image.Image, level: float) -> Image.Image:
    """Approximate age-related face shape changes without landmark dependencies."""
    arr = pil_to_array(image)
    h, w, _ = arr.shape
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    cx = w * 0.50

    face_ellipse = (
        ((xx - cx) / (w * 0.34)) ** 2
        + ((yy - h * 0.52) / (h * 0.38)) ** 2
        <= 1.0
    ).astype(np.float32)

    lower_weight = smoothstep(h * 0.38, h * 0.84, yy) * face_ellipse
    cheek_weight = smoothstep(h * 0.48, h * 0.76, yy) * (1 - smoothstep(h * 0.78, h * 0.90, yy)) * face_ellipse

    left_eye = np.exp(-(((xx - w * 0.38) / (w * 0.10)) ** 2 + ((yy - h * 0.38) / (h * 0.055)) ** 2))
    right_eye = np.exp(-(((xx - w * 0.62) / (w * 0.10)) ** 2 + ((yy - h * 0.38) / (h * 0.055)) ** 2))
    eye_weight = np.clip(left_eye + right_eye, 0, 1)

    mouth_weight = np.exp(-(((xx - cx) / (w * 0.16)) ** 2 + ((yy - h * 0.64) / (h * 0.06)) ** 2))

    sag_y = (
        lower_weight * (h * (0.028 + 0.082 * level))
        + eye_weight * (h * (0.010 + 0.035 * level))
        + mouth_weight * (h * (0.010 + 0.036 * level))
    ) * level

    jowl_expand = cheek_weight * (0.052 + 0.125 * level) * level
    lower_narrow = smoothstep(h * 0.74, h * 0.92, yy) * face_ellipse * (0.025 + 0.050 * level) * level
    scale_x = 1.0 + jowl_expand - lower_narrow

    sample_x = cx + (xx - cx) / np.maximum(scale_x, 0.85)
    sample_y = yy - sag_y

    warped = bilinear_sample(arr, sample_x, sample_y)
    blend = np.clip((face_ellipse * (0.65 + 0.35 * level))[..., None], 0, 1)
    arr = arr * (1 - blend) + warped * blend
    return array_to_pil(arr)


def adjust_tone(image: Image.Image, level: float) -> Image.Image:
    image = ImageEnhance.Brightness(image).enhance(1.02 - 0.22 * level)
    image = ImageEnhance.Contrast(image).enhance(1.08 + 0.72 * level)
    image = ImageEnhance.Color(image).enhance(0.92 - 0.58 * level)
    image = ImageEnhance.Sharpness(image).enhance(1.20 + 1.80 * level)
    return image


def add_skin_texture(image: Image.Image, level: float, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = pil_to_array(image)
    h, w, _ = arr.shape

    gray = arr.mean(axis=2, keepdims=True)
    noise = rng.normal(0, 8 + 30 * level, size=arr.shape)
    pore = rng.normal(0, 1, size=(h, w, 1))
    pore = np.repeat(pore, 3, axis=2) * (10 + 32 * level)

    yy, xx = np.mgrid[:h, :w]
    box = _face_box(w, h)
    face_mask = (
        ((xx - box["cx"]) / (w * 0.32)) ** 2
        + ((yy - h * 0.50) / (h * 0.36)) ** 2
        <= 1.0
    )[..., None]

    arr = np.where(face_mask, arr + noise + pore, arr + noise * 0.25)
    arr = arr * (1 - 0.10 * level) + gray * (0.10 * level)
    return array_to_pil(arr)


def add_age_spots(image: Image.Image, level: float, seed: int) -> Image.Image:
    random.seed(seed)
    image = image.convert("RGB")
    w, h = image.size
    box = _face_box(w, h)

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    spot_count = int(25 + 115 * level)
    for _ in range(spot_count):
        x = random.randint(box["left"], box["right"])
        y = random.randint(int(h * 0.30), int(h * 0.72))
        r = random.randint(1, max(2, int(2 + 6 * level)))
        alpha = random.randint(int(24 + 55 * level), int(45 + 95 * level))
        color = (
            random.randint(75, 135),
            random.randint(45, 90),
            random.randint(30, 70),
            alpha,
        )
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color)

    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=0.35 + 0.8 * level))
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def add_wrinkles(image: Image.Image, level: float, seed: int) -> Image.Image:
    random.seed(seed)
    image = image.convert("RGB")
    w, h = image.size
    box = _face_box(w, h)
    cx = box["cx"]

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    dark = (42, 34, 30, int(70 + 95 * level))
    light = (230, 220, 205, int(22 + 36 * level))
    width = max(1, int(1 + 2 * level))

    # Forehead wrinkles.
    for i in range(int(4 + 9 * level)):
        y = int(h * 0.24) + i * int(5 + 3 * level) + random.randint(-2, 2)
        x1 = int(w * 0.28) + random.randint(-10, 6)
        x2 = int(w * 0.72) + random.randint(-6, 10)
        mid = cx + random.randint(-12, 12)
        draw.line((x1, y, mid, y + random.randint(-4, 4), x2, y + random.randint(-3, 3)), fill=dark, width=width)
        draw.line((x1, y - 1, mid, y - 1, x2, y - 1), fill=light, width=1)

    # Crow's feet and under-eye lines.
    for side in (-1, 1):
        eye_x = int(cx + side * w * 0.18)
        eye_y = int(h * 0.38)
        for _ in range(int(8 + 14 * level)):
            y = eye_y + random.randint(-26, 28)
            length = random.randint(int(16 + 14 * level), int(32 + 46 * level))
            draw.line((eye_x, y, eye_x + side * length, y + random.randint(-12, 12)), fill=dark, width=width)

        for _ in range(int(4 + 8 * level)):
            x1 = int(cx + side * w * 0.06)
            x2 = int(cx + side * w * 0.24)
            y = int(h * 0.43) + random.randint(-7, 11)
            draw.arc((min(x1, x2), y, max(x1, x2), y + int(10 + 18 * level)), 0, 180, fill=dark, width=width)

    # Nasolabial folds and mouth lines.
    for side in (-1, 1):
        draw.line(
            (
                int(cx + side * w * 0.07),
                int(h * 0.49),
                int(cx + side * w * 0.17),
                int(h * 0.69),
            ),
            fill=(35, 28, 25, int(88 + 110 * level)),
            width=max(2, int(2 + 3 * level)),
        )
        for _ in range(int(5 + 10 * level)):
            sx = int(cx + side * w * 0.13) + random.randint(-3, 3)
            sy = int(h * 0.63) + random.randint(-17, 24)
            ex = sx + side * random.randint(int(14 + 10 * level), int(30 + 40 * level))
            ey = sy + random.randint(5, int(14 + 24 * level))
            draw.line((sx, sy, ex, ey), fill=dark, width=width)

    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=0.25))
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def add_sagging_and_shadows(image: Image.Image, level: float) -> Image.Image:
    image = image.convert("RGB")
    w, h = image.size
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    alpha = int(48 + 100 * level)
    shadow = (45, 34, 30, alpha)
    highlight = (235, 226, 210, int(18 + 28 * level))

    draw.ellipse((int(w * 0.29), int(h * 0.55), int(w * 0.52), int(h * 0.78)), fill=shadow)
    draw.ellipse((int(w * 0.48), int(h * 0.55), int(w * 0.71), int(h * 0.78)), fill=shadow)
    draw.ellipse((int(w * 0.36), int(h * 0.68), int(w * 0.64), int(h * 0.84)), fill=(38, 31, 29, int(alpha * 0.65)))
    draw.rectangle((int(w * 0.31), int(h * 0.76), int(w * 0.69), int(h * 0.89)), fill=(38, 31, 29, int(alpha * 0.35)))
    draw.ellipse((int(w * 0.34), int(h * 0.36), int(w * 0.66), int(h * 0.58)), fill=highlight)

    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=int(9 + 14 * level)))
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def add_drooping_eye_and_jowl_marks(image: Image.Image, level: float) -> Image.Image:
    image = image.convert("RGB")
    w, h = image.size
    cx = w // 2

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    dark_alpha = int(42 + 98 * level)
    soft_shadow = (35, 28, 25, dark_alpha)
    highlight = (235, 225, 210, int(20 + 26 * level))

    for side in (-1, 1):
        eye_x = int(cx + side * w * 0.17)
        eye_y = int(h * 0.37)
        draw.arc(
            (
                eye_x - int(w * 0.095),
                eye_y - int(h * 0.030),
                eye_x + int(w * 0.095),
                eye_y + int(h * 0.050),
            ),
            185,
            350,
            fill=soft_shadow,
            width=max(2, int(2 + 4 * level)),
        )
        draw.arc(
            (
                eye_x - int(w * 0.080),
                eye_y + int(h * 0.035),
                eye_x + int(w * 0.080),
                eye_y + int(h * 0.090),
            ),
            0,
            180,
            fill=soft_shadow,
            width=max(1, int(1 + 3 * level)),
        )

        jaw_x = int(cx + side * w * 0.19)
        draw.line(
            (
                jaw_x,
                int(h * 0.61),
                jaw_x + side * int(w * 0.035),
                int(h * 0.78),
            ),
            fill=soft_shadow,
            width=max(2, int(2 + 5 * level)),
        )
        draw.line(
            (
                jaw_x - side * int(w * 0.010),
                int(h * 0.61),
                jaw_x - side * int(w * 0.035),
                int(h * 0.76),
            ),
            fill=highlight,
            width=1,
        )

    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=0.35 + 0.6 * level))
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def add_cheek_hollow_and_volume_change(image: Image.Image, level: float) -> Image.Image:
    image = image.convert("RGB")
    w, h = image.size
    cx = w // 2

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    hollow = (42, 31, 27, int(38 + 92 * level))
    lower_fullness = (72, 55, 46, int(24 + 58 * level))
    skin_lift = (235, 218, 198, int(10 + 22 * level))

    for side in (-1, 1):
        cheek_x = int(cx + side * w * 0.145)
        draw.ellipse(
            (
                cheek_x - int(w * 0.105),
                int(h * 0.49),
                cheek_x + int(w * 0.105),
                int(h * 0.66),
            ),
            fill=hollow,
        )
        draw.ellipse(
            (
                int(cx + side * w * 0.155) - int(w * 0.10),
                int(h * 0.64),
                int(cx + side * w * 0.155) + int(w * 0.13),
                int(h * 0.80),
            ),
            fill=lower_fullness,
        )
        draw.ellipse(
            (
                cheek_x - int(w * 0.08),
                int(h * 0.42),
                cheek_x + int(w * 0.08),
                int(h * 0.53),
            ),
            fill=skin_lift,
        )

    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=int(13 + 18 * level)))
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def apply_gray_hair_effect(image: Image.Image, level: float) -> Image.Image:
    arr = pil_to_array(image)
    gray = arr.mean(axis=2)
    h, w = gray.shape
    yy, xx = np.mgrid[:h, :w]

    dark_mask = gray < (105 + 30 * level)
    top_hair = yy < h * 0.25
    side_hair = (yy < h * 0.56) & ((xx < w * 0.22) | (xx > w * 0.78))
    hair_mask = dark_mask & (top_hair | side_hair)

    gray_target = np.stack([gray + 72 + 55 * level] * 3, axis=2)
    blend = min(0.95, 0.22 + 0.78 * level)
    arr[hair_mask] = arr[hair_mask] * (1 - blend) + gray_target[hair_mask] * blend
    return array_to_pil(arr)


def add_edge_vignette(image: Image.Image, level: float) -> Image.Image:
    arr = pil_to_array(image)
    h, w, _ = arr.shape
    yy, xx = np.mgrid[:h, :w]
    dist = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    vignette = 1 - np.clip((dist - 0.35) / 0.85, 0, 1) * (0.20 + 0.18 * level)
    arr = arr * vignette[..., None]
    return array_to_pil(arr)


def simulate_aging(image: Image.Image, years_later: int) -> Image.Image:
    """Create a visibly stronger rule-based aging simulation for MVP use."""
    level = max(0.0, min(years_later / 50.0, 1.0))
    image = image.convert("RGB")
    original_size = image.size

    scale = MAX_PROCESS_SIZE / max(original_size)
    if scale < 1.0:
        resized_size = (
            max(1, int(original_size[0] * scale)),
            max(1, int(original_size[1] * scale)),
        )
        image = image.resize(resized_size, Image.Resampling.LANCZOS)

    aged = apply_age_shape_warp(image, level)
    aged = adjust_tone(aged, level)
    aged = apply_gray_hair_effect(aged, level)
    aged = add_sagging_and_shadows(aged, level)
    aged = add_cheek_hollow_and_volume_change(aged, level)
    aged = add_drooping_eye_and_jowl_marks(aged, level)
    aged = add_wrinkles(aged, level, seed=years_later)
    aged = add_age_spots(aged, level, seed=years_later + 100)
    aged = add_skin_texture(aged, level, seed=years_later + 200)
    aged = add_edge_vignette(aged, level)
    aged = ImageEnhance.Sharpness(aged).enhance(1.15 + 1.10 * level)

    if aged.size != original_size:
        aged = aged.resize(original_size, Image.Resampling.LANCZOS)

    return aged


def generate_future_faces(image: Image.Image, years_list=None):
    if years_list is None:
        years_list = DEFAULT_YEARS

    return [
        {
            "years_later": years,
            "image": simulate_aging(image, years),
        }
        for years in years_list
    ]


def save_future_faces(image_path, output_dir, years_list=None):
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path).convert("RGB")
    results = generate_future_faces(image, years_list=years_list)

    saved_paths = []
    for item in results:
        out_path = output_dir / f"future_{item['years_later']}years.png"
        item["image"].save(out_path)
        saved_paths.append(out_path)

    return saved_paths


if __name__ == "__main__":
    print("aging_simulator.py provides simulate_aging() and generate_future_faces().")

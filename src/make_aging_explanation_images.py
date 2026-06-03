from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = PROJECT_ROOT / "outputs" / "aging_samples"
FONT_PATH = Path("C:/Windows/Fonts/malgun.ttf")


def get_font(size):
    if FONT_PATH.exists():
        return ImageFont.truetype(str(FONT_PATH), size)
    return ImageFont.load_default()


def fit_to_panel(image, width, height):
    image = image.copy()
    image.thumbnail((width, height - 70))
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(image, ((width - image.width) // 2, 55))
    return canvas


def draw_label(draw, anchor, target, text, font):
    draw.line([anchor, target], fill=(215, 30, 30), width=4)
    x, y = target
    text_box = draw.textbbox((x, y), text, font=font)
    box_w = text_box[2] - text_box[0] + 22
    box_h = text_box[3] - text_box[1] + 16
    draw.rounded_rectangle(
        (x - 10, y - box_h - 8, x - 10 + box_w, y - 8),
        radius=7,
        fill=(255, 246, 214),
        outline=(215, 30, 30),
        width=2,
    )
    draw.text((x + 1, y - box_h - 2), text, fill=(0, 0, 0), font=font)


def make_annotated_image():
    original = Image.open(SAMPLE_DIR / "original.png").convert("RGB")
    future = Image.open(SAMPLE_DIR / "future_50years.png").convert("RGB")

    panel_w, panel_h = 560, 652
    gap = 40
    title_font = get_font(25)
    label_font = get_font(18)

    left = fit_to_panel(original, panel_w, panel_h)
    right = fit_to_panel(future, panel_w, panel_h)

    sheet = Image.new("RGB", (panel_w * 2 + gap, panel_h), "white")
    sheet.paste(left, (0, 0))
    sheet.paste(right, (panel_w + gap, 0))

    draw = ImageDraw.Draw(sheet)
    draw.text((20, 14), "\uc6d0\ubcf8", fill=(0, 0, 0), font=title_font)
    draw.text(
        (panel_w + gap + 20, 14),
        "50\ub144 \ub4a4 \uc2dc\ubbac\ub808\uc774\uc158",
        fill=(0, 0, 0),
        font=title_font,
    )

    ox = panel_w + gap
    labels = [
        ((ox + 310, 172), (ox + 404, 104), "\uba38\ub9ac/\uc804\uccb4 \ucc44\ub3c4 \uac10\uc18c"),
        ((ox + 282, 246), (ox + 390, 196), "\ub208 \uc8fc\ubcc0 \ucc98\uc9d0"),
        ((ox + 282, 346), (ox + 394, 308), "\ubcfc \ud328\uc784/\uc7a1\ud2f0"),
        ((ox + 287, 414), (ox + 398, 388), "\ud314\uc790\uc8fc\ub984 \uac15\ud654"),
        ((ox + 292, 514), (ox + 385, 474), "\ud558\uad00 \ucc98\uc9d0/\ud131\uc120 \ubcc0\ud654"),
    ]

    for anchor, target, text in labels:
        draw_label(draw, anchor, target, text, label_font)

    out_path = SAMPLE_DIR / "annotated_aging_changes.png"
    sheet.save(out_path)
    return out_path


def make_difference_heatmap():
    original = Image.open(SAMPLE_DIR / "original.png").convert("RGB").resize((420, 490))
    future = Image.open(SAMPLE_DIR / "future_50years.png").convert("RGB").resize((420, 490))

    original_arr = np.asarray(original).astype(np.float32)
    future_arr = np.asarray(future).astype(np.float32)
    diff = np.abs(future_arr - original_arr).mean(axis=2)
    diff = diff / max(float(diff.max()), 1.0)

    heat = np.zeros((*diff.shape, 3), dtype=np.uint8)
    heat[..., 0] = np.clip(diff * 255, 0, 255).astype(np.uint8)
    heat[..., 1] = np.clip((1 - np.abs(diff - 0.5) * 2) * 160, 0, 160).astype(np.uint8)
    heat[..., 2] = np.clip((1 - diff) * 80, 0, 80).astype(np.uint8)

    heat_img = Image.fromarray(heat).filter(ImageFilter.SMOOTH)
    blended = Image.blend(future, heat_img, 0.45)

    panel_w, panel_h = 420, 540
    title_font = get_font(22)
    label_font = get_font(16)
    sheet = Image.new("RGB", (panel_w * 3 + 40, panel_h), "white")

    panels = [
        (original, "\uc6d0\ubcf8"),
        (future, "50\ub144 \ub4a4"),
        (blended, "\ubcc0\ud654 \ud788\ud2b8\ub9f5"),
    ]

    draw = ImageDraw.Draw(sheet)
    for i, (img, title) in enumerate(panels):
        x = i * (panel_w + 20)
        sheet.paste(img, (x, 45))
        draw.text((x + 12, 12), title, fill=(0, 0, 0), font=title_font)

    draw.text(
        (panel_w * 2 + 52, 505),
        "\ube68\uac15\uc77c\uc218\ub85d \uc6d0\ubcf8\uacfc \ucc28\uc774\uac00 \ud070 \uc601\uc5ed",
        fill=(90, 20, 20),
        font=label_font,
    )

    out_path = SAMPLE_DIR / "aging_difference_heatmap.png"
    sheet.save(out_path)
    return out_path


def main():
    print(make_annotated_image())
    print(make_difference_heatmap())


if __name__ == "__main__":
    main()

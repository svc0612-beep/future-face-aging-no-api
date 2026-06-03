from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CUSP_ROOT = PROJECT_ROOT / "third_party" / "CUSP"
WEIGHTS_DIR = PROJECT_ROOT / "models" / "future_face" / "cusp" / "weights"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "cusp_tests"

CUSP_MODEL_FILES = {
    "ffhq_ls": {
        "filename": "cusp_ffhq_ls_pretrained.pkl",
        "drive_url": "https://drive.google.com/uc?id=1sWSH3tHgm9DkHrc19hoEMrR-KQgnaFuw",
        "side": 256,
        "labels": [1, 3, 5, 6, 7, 8],
        "note": "FFHQ-LS/LATS age-bin labels. Labels are bins, not exact ages.",
    },
    "ffhq_rr": {
        "filename": "cusp_ffhq_rr_pretrained.pkl",
        "drive_url": "https://drive.google.com/uc?id=17BOTEa6z3r6JFVs1KDutDxWEkTWbzaeD",
        "side": 224,
        "labels": [30, 40, 50, 60, 65],
        "note": "FFHQ-RR numeric age labels.",
    },
}

DEX_FILE = {
    "filename": "dex_imdb_wiki.caffemodel.pt",
    "drive_url": "https://drive.google.com/uc?id=1a0jghZx44uC_kIIvD-UEcNp4Xz8Bpso1",
}


def require_cusp_imports() -> None:
    if not CUSP_ROOT.exists():
        raise FileNotFoundError(
            f"CUSP repository was not found: {CUSP_ROOT}\n"
            "Clone https://github.com/guillermogotre/CUSP into third_party/CUSP first."
        )
    sys.path.insert(0, str(CUSP_ROOT))


def missing_files(model_key: str) -> list[tuple[str, str, Path]]:
    cfg = CUSP_MODEL_FILES[model_key]
    required = [
        ("CUSP pretrained model", cfg["drive_url"], WEIGHTS_DIR / cfg["filename"]),
        ("DEX VGG age classifier", DEX_FILE["drive_url"], WEIGHTS_DIR / DEX_FILE["filename"]),
    ]
    return [(name, url, path) for name, url, path in required if not path.exists()]


def print_missing_and_exit(model_key: str) -> None:
    missing = missing_files(model_key)
    if not missing:
        return

    print("\n[CUSP] Required weight files are missing.\n")
    print(f"Put the files in:\n  {WEIGHTS_DIR}\n")
    for name, url, path in missing:
        print(f"- {name}")
        print(f"  expected path: {path}")
        print(f"  official URL : {url}")
    print("\nGoogle Drive may require browser login/authentication, so CLI download can fail.")
    raise SystemExit(2)


def load_cusp_model(model_key: str, device: torch.device):
    require_cusp_imports()
    import legacy  # type: ignore
    from training.networks import VGG, module_no_grad  # type: ignore

    cfg = CUSP_MODEL_FILES[model_key]
    model_path = WEIGHTS_DIR / cfg["filename"]
    vgg_path = WEIGHTS_DIR / DEX_FILE["filename"]

    with model_path.open("rb") as f:
        contents = legacy.load_network_pkl(f)

    generator = contents["G_ema"]

    vgg = VGG()
    state_dict = torch.load(vgg_path, map_location="cpu")
    state_dict = {k.replace("-", "_"): v for k, v in state_dict.items()}
    vgg.load_state_dict(state_dict)
    module_no_grad(vgg)

    generator.skip_grad_blur.model.classifier = vgg
    generator = generator.to(device).eval().requires_grad_(False)
    return generator


def preprocess_image(image_path: Path, side: int, device: torch.device) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB").resize((side, side), Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.float32).transpose((2, 0, 1))
    tensor = torch.tensor(arr, device=device).unsqueeze(0)
    return tensor / 256.0 * 2.0 - 1.0


def run_model(generator, img: torch.Tensor, labels: torch.Tensor, global_blur_val=0.2, mask_blur_val=0.8):
    cls = torch.nn.functional.one_hot(
        labels,
        num_classes=generator.attr_map.fc0.init_args[0],
    ).to(img.device)

    repeated_img = img.expand(labels.shape[0], *img.shape[1:]).contiguous()

    _, c_out_skip = generator.content_enc(repeated_img)
    s_out = generator.style_enc(repeated_img)[0].mean((2, 3))
    s_out = generator.style_map(s_out, None, 1, None)
    a_out = generator.attr_map(cls.to(s_out.device), None, 1, None)
    w = generator.__interleave_attr_style__(a_out, s_out)

    for i, (f, _) in enumerate(zip(generator.skip_transf, c_out_skip)):
        if f is not None:
            c_out_skip[i] = generator._batch_blur(c_out_skip[i], blur_val=global_blur_val)

    cam = generator.skip_grad_blur(repeated_img.float())
    msk = cam
    for i, (f, c) in enumerate(zip(generator.skip_transf, c_out_skip)):
        if f is not None:
            im_size = c.size(-1)
            blur_c = generator._batch_blur(c, blur_val=mask_blur_val)
            if msk.size(2) != im_size:
                msk = F.interpolate(msk, size=(im_size, im_size), mode="area")
            c_out_skip[i] = c * msk + blur_c * (1 - msk)

    return generator.image_dec(c_out_skip, w)


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    arr = (tensor.detach().cpu().numpy().transpose((1, 2, 0)) + 1) * 128
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def make_grid(original: Image.Image, outputs: list[tuple[str, Image.Image]], output_path: Path) -> None:
    tile = original.size[0]
    label_h = 34
    cols = len(outputs) + 1
    grid = Image.new("RGB", (cols * tile, tile + label_h), "white")
    draw = ImageDraw.Draw(grid)

    tiles = [("input", original), *outputs]
    for idx, (label, image) in enumerate(tiles):
        x = idx * tile
        grid.paste(image.resize((tile, tile), Image.Resampling.LANCZOS), (x, 0))
        draw.text((x + 8, tile + 8), label, fill=(20, 20, 20))

    grid.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--model", choices=sorted(CUSP_MODEL_FILES), default="ffhq_ls")
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--labels", nargs="*", type=int, default=None)
    args = parser.parse_args()

    print_missing_and_exit(args.model)

    if not torch.cuda.is_available():
        raise RuntimeError("CUSP local test requires CUDA.")

    cfg = CUSP_MODEL_FILES[args.model]
    labels = args.labels or cfg["labels"]
    device = torch.device("cuda", 0)

    outdir = args.outdir or OUTPUT_ROOT / f"{args.model}_{args.image.stem}"
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[CUSP] model={args.model}")
    print(f"[CUSP] labels={labels}")
    print(f"[CUSP] note={cfg['note']}")
    print(f"[CUSP] output={outdir}")

    generator = load_cusp_model(args.model, device)
    input_tensor = preprocess_image(args.image, cfg["side"], device)
    labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)

    with torch.no_grad():
        output_tensor = run_model(generator, input_tensor, labels_tensor)

    original = tensor_to_image(input_tensor[0])
    original.save(outdir / "input.png")

    results: list[tuple[str, Image.Image]] = []
    for label, tensor in zip(labels, output_tensor):
        image = tensor_to_image(tensor)
        filename = f"label_{label}.png"
        image.save(outdir / filename)
        results.append((f"label {label}", image))

    make_grid(original, results, outdir / "comparison_grid.png")

    report = {
        "model": args.model,
        "labels": labels,
        "input": str(args.image),
        "output_dir": str(outdir),
        "caution": "CUSP labels are model-specific. Inspect identity, ethnicity, gender, and aging steps visually.",
    }
    (outdir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[CUSP] done")
    print(f"[CUSP] grid={outdir / 'comparison_grid.png'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
segment.py
----------
Generate foreground masks for real photos with SAM or SAM 2, so query.py can
crop to the object before embedding (see query.py and common.masked_crop).

Output matches common.py's mask scheme:
    0 = background
    1 = foreground (the printed part, including any support material)
Masks are saved as single-channel uint8 PNGs named <stem>.png to match the
image stems, ready to feed straight into query.py:

    python segment.py real_photos/ --out-dir real_masks/
    python query.py  real_photos/ --mask-dir real_masks/

Backends
    --backend sam2 (default)   facebook/sam2-hiera-large via HuggingFace, auto-
                               downloaded on first run.
                               pip install "git+https://github.com/facebookresearch/sam2.git"
    --backend sam              classic Segment Anything; needs a local checkpoint.
                               pip install segment-anything   (+ a .pth, e.g.
                               sam_vit_h_4b8939.pth) and pass --checkpoint.

LIMITATION -- no support class
    SAM separates foreground from background only; it cannot distinguish a part
    from its print supports. Everything it segments is labelled 1, so masks
    here only ever contain 0 and 1. This is compatible with the pipeline:
    masked_crop folds labels 1 and 2 together unless query.py gets --no-support,
    and there are no label-2 pixels to drop. If you genuinely need supports as a
    separate class you'd have to train a small segmentation head -- SAM can't.

Prompting
    A single centre point is the default (matches the query.py docstring hint).
    If your object sits off-centre, or the centre lands on a hole, override with
    --points "x,y;x,y" or add a near-full-frame --box.

Heavy deps (torch, sam2 / segment_anything, scipy) are imported lazily so this
file parses and --help works without them installed.
"""
import argparse
import contextlib
import os

import numpy as np
from PIL import Image


IMAGE_EXTS = ("png", "jpg", "jpeg", "webp", "bmp")


def collect_images(path):
    """Sorted list of image paths. Case-insensitive on extension, unlike a
    plain glob, so phone-camera .JPG files aren't silently skipped."""
    if os.path.isdir(path):
        files = [
            os.path.join(path, e)
            for e in os.listdir(path)
            if e.lower().rsplit(".", 1)[-1] in IMAGE_EXTS
        ]
        return sorted(files)
    return [path]


def parse_points(s):
    """'120,200;300,150' -> [[120.0, 200.0], [300.0, 150.0]] (x, y order)."""
    pts = []
    for chunk in s.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        x, y = chunk.split(",")
        pts.append([float(x), float(y)])
    return pts


def pick_device(requested):
    import torch
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def build_predictor(backend, device, sam2_model, sam_checkpoint, sam_model_type):
    """Return a predictor exposing .set_image(rgb) and .predict(...). The two
    backends already share that interface, so callers don't branch again."""
    if backend == "sam2":
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        try:
            return SAM2ImagePredictor.from_pretrained(sam2_model, device=device)
        except TypeError:
            # older sam2 builds don't take a device kwarg here
            return SAM2ImagePredictor.from_pretrained(sam2_model)
    if backend == "sam":
        if not sam_checkpoint:
            raise SystemExit("--backend sam requires --checkpoint path/to/sam.pth")
        from segment_anything import sam_model_registry, SamPredictor
        sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
        sam.to(device)
        return SamPredictor(sam)
    raise SystemExit(f"unknown backend {backend!r}")


def segment_one(predictor, rgb, points, labels, box, device):
    """Run the predictor once and return the highest-scoring mask as bool."""
    import torch
    predictor.set_image(rgb)
    pc = np.array(points, dtype=np.float32) if points else None
    pl = np.array(labels, dtype=np.int32) if labels else None
    bx = np.array(box, dtype=np.float32) if box is not None else None

    if device == "cuda":
        amp = torch.autocast("cuda", dtype=torch.bfloat16)
    else:
        amp = contextlib.nullcontext()

    with torch.inference_mode(), amp:
        masks, scores, _ = predictor.predict(
            point_coords=pc,
            point_labels=pl,
            box=bx,
            multimask_output=True,
        )
    best = int(np.argmax(scores))
    return np.asarray(masks[best]).astype(bool)


def clean_mask(mask):
    """Keep only the largest connected blob and fill interior holes. Removes
    stray specks and segmentation pinholes that would otherwise corrupt the
    bounding box masked_crop computes. No-op if scipy isn't installed."""
    try:
        from scipy import ndimage
    except Exception:
        return mask
    labelled, n = ndimage.label(mask)
    if n > 1:
        counts = ndimage.sum(mask, labelled, index=range(1, n + 1))
        keep = int(np.argmax(counts)) + 1
        mask = labelled == keep
    return ndimage.binary_fill_holes(mask)


def save_mask(mask, out_path):
    """Single-channel PNG with literal values 0 and 1 (NOT 0/255), as
    common.load_mask expects. It will look almost black in an image viewer --
    that's why --preview-dir exists."""
    Image.fromarray(mask.astype(np.uint8), mode="L").save(out_path)


def save_preview(rgb, mask, out_path):
    """A human-viewable overlay: foreground tinted over the original photo."""
    overlay = rgb.copy()
    sel = mask.astype(bool)
    tint = np.array([255, 0, 80], dtype=np.float32)
    overlay[sel] = (0.45 * tint + 0.55 * overlay[sel]).astype(np.uint8)
    Image.fromarray(overlay).save(out_path)


def main():
    ap = argparse.ArgumentParser(
        description="Foreground masks for real photos (SAM / SAM 2)."
    )
    ap.add_argument("images", help="image file or directory of images")
    ap.add_argument("--out-dir", default="real_masks",
                    help="where 0/1 masks are written (default: real_masks)")
    ap.add_argument("--backend", default="sam2", choices=["sam2", "sam"])
    ap.add_argument("--sam2-model", default="facebook/sam2-hiera-large",
                    help="HuggingFace id for --backend sam2")
    ap.add_argument("--checkpoint", default=None,
                    help="SAM .pth path (required for --backend sam)")
    ap.add_argument("--sam-model-type", default="vit_h",
                    choices=["vit_h", "vit_l", "vit_b"])
    ap.add_argument("--points", default=None,
                    help='foreground points "x,y;x,y"; default = image centre')
    ap.add_argument("--box", action="store_true",
                    help="also prompt with a near-full-frame box")
    ap.add_argument("--box-inset", type=float, default=0.04,
                    help="box inset as a fraction of W/H (default 0.04)")
    ap.add_argument("--no-clean", action="store_true",
                    help="skip largest-component + hole-fill post-processing")
    ap.add_argument("--preview-dir", default=None,
                    help="also write visible mask overlays here for eyeballing")
    ap.add_argument("--device", default=None, help="cuda / mps / cpu (auto if unset)")
    args = ap.parse_args()

    paths = collect_images(args.images)
    if not paths:
        raise SystemExit(f"No images found at {args.images}")

    os.makedirs(args.out_dir, exist_ok=True)
    if args.preview_dir:
        os.makedirs(args.preview_dir, exist_ok=True)

    device = pick_device(args.device)
    print(f"[info] backend={args.backend} device={device} images={len(paths)}")
    predictor = build_predictor(args.backend, device, args.sam2_model,
                                args.checkpoint, args.sam_model_type)

    manual_pts = parse_points(args.points) if args.points else None

    for path in paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        rgb = np.array(Image.open(path).convert("RGB"))
        h, w = rgb.shape[:2]

        points = manual_pts if manual_pts else [[w / 2.0, h / 2.0]]
        labels = [1] * len(points)

        box = None
        if args.box:
            ix, iy = args.box_inset * w, args.box_inset * h
            box = [ix, iy, w - ix, h - iy]

        mask = segment_one(predictor, rgb, points, labels, box, device)
        if not args.no_clean:
            mask = clean_mask(mask)

        cov = float(mask.mean())
        if cov < 0.005:
            flag = "  <-- almost empty; check the prompt point"
        elif cov > 0.95:
            flag = "  <-- nearly full frame; segmentation likely failed"
        else:
            flag = ""

        save_mask(mask, os.path.join(args.out_dir, stem + ".png"))
        if args.preview_dir:
            save_preview(rgb, mask, os.path.join(args.preview_dir, stem + ".png"))
        print(f"  {stem:24s} coverage={cov:6.1%}{flag}")

    print(f"\n[done] masks -> {args.out_dir}")
    if args.preview_dir:
        print(f"[done] previews -> {args.preview_dir}  "
              f"(eyeball these before trusting the masks)")
    print(f"\nNext: python query.py {args.images} --mask-dir {args.out_dir}")


if __name__ == "__main__":
    main()
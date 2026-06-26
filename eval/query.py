#!/usr/bin/env python3
"""
query.py
--------
Predict the catalog part for arbitrary query image(s) -- e.g. real photos.

Real photos have no segmentation mask. For best results, segment the object
first and pass a matching 0/1/2 mask via --mask-dir (mask filenames must match
the image stems). Good options: SAM / SAM 2 with a centre-point prompt, rembg,
or background subtraction on a known print bed.

Without a mask the whole frame is embedded and your randomized-background
renders carry the sim-to-real load -- still works, but expect lower accuracy.
Use the SAME --model / --pooling / --no-support flags you used in build_gallery.py.

    python query.py photo.jpg
    python query.py real_photos/ --mask-dir real_masks/
    python query.py real_photos/ --model facebook/dinov3-vitb16-pretrain-lvd1689m
"""
import argparse
import os
import glob

import faiss
from PIL import Image

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.common import (load_embeddings, aggregate_votes, load_rgb, load_mask,
                    masked_crop, DinoEmbedder)


def collect(query_path):
    if os.path.isdir(query_path):
        files = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp"):
            files += glob.glob(os.path.join(query_path, ext))
        return sorted(files)
    return [query_path]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", help="image file or directory of images")
    ap.add_argument("--index", default="index.faiss")
    ap.add_argument("--gallery-prefix", default="gallery")
    ap.add_argument("--mask-dir", default=None,
                    help="optional dir of 0/1/2 masks named like the query images")
    ap.add_argument("--model", default="facebook/dinov2-base")
    ap.add_argument("--pooling", default="cls", choices=["cls", "cls+patch"])
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--vote", default="score", choices=["score", "count"])
    ap.add_argument("--topn", type=int, default=5)
    ap.add_argument("--no-support", action="store_true")
    args = ap.parse_args()

    index = faiss.read_index(args.index)
    _, g_meta = load_embeddings(args.gallery_prefix)
    embedder = DinoEmbedder(args.model, pooling=args.pooling)
    include_support = not args.no_support
    k = min(args.k, index.ntotal)

    paths = collect(args.query)
    if not paths:
        raise SystemExit(f"No images found at {args.query}")

    for path in paths:
        rgb = load_rgb(path)
        if args.mask_dir:
            stem = os.path.splitext(os.path.basename(path))[0]
            mpath = os.path.join(args.mask_dir, stem + ".png")
            if os.path.exists(mpath):
                img = masked_crop(rgb, load_mask(mpath), include_support=include_support)
            else:
                print(f"[warn] no mask for {stem}; embedding full frame")
                img = Image.fromarray(rgb)
        else:
            img = Image.fromarray(rgb)

        q = embedder.embed([img])
        sims, idxs = index.search(q.astype("float32"), k)
        ranked = aggregate_votes(sims[0], idxs[0], g_meta, mode=args.vote)
        print(f"\n{os.path.basename(path)}")
        for pid, score in ranked[:args.topn]:
            print(f"  {pid:18s} {score:.3f}")


if __name__ == "__main__":
    main()

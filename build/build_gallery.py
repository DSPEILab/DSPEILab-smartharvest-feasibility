#!/usr/bin/env python3
"""
build_gallery.py
----------------
Embed every render with a frozen DINO encoder and write two splits:

    gallery_emb.npy / gallery_meta.json   <- searchable gallery
    heldout_emb.npy / heldout_meta.json   <- held-out views for honest eval

Run this once per render batch (it's the slow step). Querying and evaluation
reuse these files without re-embedding.

Examples:
    python build_gallery.py --dataset dataset_pyrender
    python build_gallery.py --dataset dataset_pyrender --model facebook/dinov3-vitb16-pretrain-lvd1689m
    python build_gallery.py --dataset dataset_pyrender --no-support --pooling cls+patch
"""
import argparse

import numpy as np
import sys
from pathlib import Path

# Ensure project root is on sys.path so imports work when running from build/
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.common import (list_frames, split_frames, load_rgb, load_mask,
                          masked_crop, save_embeddings, DinoEmbedder)


def embed_frames(frames, embedder, batch_size, include_support):
    embs, meta = [], []
    batch_imgs, batch_meta = [], []

    def flush():
        if not batch_imgs:
            return
        embs.append(embedder.embed(batch_imgs))
        meta.extend(batch_meta)
        batch_imgs.clear()
        batch_meta.clear()

    for fr in frames:
        rgb = load_rgb(fr["rgb"])
        mask = load_mask(fr["mask"])
        img = masked_crop(rgb, mask, include_support=include_support)
        batch_imgs.append(img)
        batch_meta.append({
            "stem": fr["stem"], "part": fr["part"],
            "class_id": fr["class_id"], "view": fr["view"], "rgb": fr["rgb"],
        })
        if len(batch_imgs) >= batch_size:
            flush()
    flush()

    if not embs:
        return np.zeros((0, 1), dtype="float32"), meta
    return np.vstack(embs).astype("float32"), meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset_pyrender")
    ap.add_argument("--gallery-prefix", default="gallery")
    ap.add_argument("--heldout-prefix", default="heldout")
    ap.add_argument("--holdout-frac", type=float, default=0.2,
                    help="fraction of each part's views held out for eval (0 to disable)")
    ap.add_argument("--model", default="facebook/dinov2-base")
    ap.add_argument("--pooling", default="cls", choices=["cls", "cls+patch"])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--no-support", action="store_true",
                    help="exclude print supports from the foreground crop")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    frames = list_frames(args.dataset)
    if not frames:
        raise SystemExit(f"No frames found under {args.dataset}/meta/*.json")

    gallery_fr, heldout_fr = split_frames(frames, args.holdout_frac, args.seed)
    n_parts = len({f["part"] for f in frames})
    print(f"{len(frames)} frames across {n_parts} parts "
          f"-> {len(gallery_fr)} gallery / {len(heldout_fr)} held-out")

    embedder = DinoEmbedder(args.model, pooling=args.pooling)
    include_support = not args.no_support

    g_emb, g_meta = embed_frames(gallery_fr, embedder, args.batch_size, include_support)
    save_embeddings(args.gallery_prefix, g_emb, g_meta)
    print(f"gallery:  {g_emb.shape} -> {args.gallery_prefix}_emb.npy")

    if heldout_fr:
        h_emb, h_meta = embed_frames(heldout_fr, embedder, args.batch_size, include_support)
        save_embeddings(args.heldout_prefix, h_emb, h_meta)
        print(f"held-out: {h_emb.shape} -> {args.heldout_prefix}_emb.npy")


if __name__ == "__main__":
    main()

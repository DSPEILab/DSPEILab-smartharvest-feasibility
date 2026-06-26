#!/usr/bin/env python3
"""
build_index.py
--------------
Build an exact FAISS cosine-similarity index from the gallery embeddings.

Embeddings are already L2-normalized, so inner product == cosine similarity.
A flat (brute-force) index is exact and plenty fast for galleries up to ~1M
vectors; only switch to IVF/HNSW if flat search becomes a bottleneck.

    python build_index.py                         # gallery_*  -> index.faiss
    python build_index.py --gallery-prefix gallery --index-out index.faiss
"""
import argparse

import faiss

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.common import load_embeddings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gallery-prefix", default="gallery")
    ap.add_argument("--index-out", default="index.faiss")
    args = ap.parse_args()

    emb, _ = load_embeddings(args.gallery_prefix)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb.astype("float32"))
    faiss.write_index(index, args.index_out)
    print(f"Indexed {index.ntotal} vectors (dim {emb.shape[1]}) -> {args.index_out}")


if __name__ == "__main__":
    main()

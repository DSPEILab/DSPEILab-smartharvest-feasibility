#!/usr/bin/env python3
"""
evaluate.py
-----------
Evaluate the baseline on the held-out renders: top-1 / top-5 accuracy,
per-part accuracy, and the most-confused part pairs.

IMPORTANT: held-out renders are SYNTHETIC. High accuracy here proves the
encoder can separate your parts in principle, but says nothing about the
sim-to-real gap. Treat this as a sanity check; the number that decides whether
you ship is real-photo accuracy (use query.py with labelled real photos).

    python evaluate.py                         # uses index.faiss + gallery_* + heldout_*
    python evaluate.py --k 30 --vote count
"""
import argparse
from collections import defaultdict, Counter

import faiss

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.common import load_embeddings, aggregate_votes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="index.faiss")
    ap.add_argument("--gallery-prefix", default="gallery")
    ap.add_argument("--heldout-prefix", default="heldout")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--vote", default="score", choices=["score", "count"])
    args = ap.parse_args()

    index = faiss.read_index(args.index)
    _, g_meta = load_embeddings(args.gallery_prefix)
    h_emb, h_meta = load_embeddings(args.heldout_prefix)
    if len(h_meta) == 0:
        raise SystemExit("Held-out set is empty; rebuild gallery with --holdout-frac > 0")

    k = min(args.k, index.ntotal)
    sims, idxs = index.search(h_emb.astype("float32"), k)

    top1 = top5 = 0
    confusion = defaultdict(Counter)
    per_total = Counter()
    per_correct = Counter()

    for row in range(len(h_meta)):
        true_pid = h_meta[row]["part"]
        ranked = aggregate_votes(sims[row], idxs[row], g_meta, mode=args.vote)
        preds = [p for p, _ in ranked]
        is_top1 = bool(preds[:1] == [true_pid])
        top1 += is_top1
        top5 += (true_pid in preds[:5])
        per_total[true_pid] += 1
        per_correct[true_pid] += is_top1
        if not is_top1 and preds:
            confusion[true_pid][preds[0]] += 1

    n = len(h_meta)
    print(f"\nHeld-out renders: {n} queries | k={k} | vote={args.vote}")
    print(f"  top-1 accuracy: {top1 / n:.3f}")
    print(f"  top-5 accuracy: {top5 / n:.3f}\n")

    print("Per-part top-1:")
    for p in sorted(per_total):
        print(f"  {p:18s} {per_correct[p]:3d}/{per_total[p]:<3d} "
              f"= {per_correct[p] / per_total[p]:.2f}")

    pairs = sorted(
        ((t, w, c) for t, ws in confusion.items() for w, c in ws.items()),
        key=lambda x: -x[2],
    )
    if pairs:
        print("\nMost-confused (true -> predicted : count):")
        for t, w, c in pairs[:10]:
            print(f"  {t:18s} -> {w:18s} : {c}")
    else:
        print("\nNo misclassifications.")


if __name__ == "__main__":
    main()

"""
common.py
---------
Shared utilities for the DINO + FAISS zero-training part-ID baseline (step 2).

Matches the on-disk layout produced by render_pyrender.py:

    dataset_pyrender/
        rgb/<part>_<view:03d>.png     8-bit RGB
        depth/<part>_<view:03d>.png   16-bit, depth in 0.1 mm units (bg = 0)
        mask/<part>_<view:03d>.png    8-bit, 0=background 1=part 2=support
        meta/<part>_<view:03d>.json   {part, class_id, view, intrinsics, ...}

Note: part names contain underscores, so the label is read from meta JSON's
"part" field, never parsed from the filename.

torch / transformers are imported lazily inside DinoEmbedder, and faiss inside
the index helpers, so the pure-data utilities here can be used (and tested)
without those heavy dependencies installed.
"""
import os
import json
import glob
from collections import defaultdict

import numpy as np
from PIL import Image

# 16-bit depth PNGs are stored in units of 0.1 mm (see render_pyrender.DEPTH_SCALE).
DEPTH_SCALE_MM = 0.1

# Mask label scheme from the renderer.
LABEL_BACKGROUND, LABEL_PART, LABEL_SUPPORT = 0, 1, 2


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------
def list_frames(dataset_dir):
    """Read meta/*.json and return one dict per frame.

    Each dict: {stem, part, class_id, view, rgb, depth, mask, meta}.
    Paths are absolute-or-relative to cwd, built from the meta filename stem so
    they stay correct regardless of underscores in the part name.
    """
    frames = []
    meta_glob = os.path.join(dataset_dir, "meta", "*.json")
    for meta_path in sorted(glob.glob(meta_glob)):
        with open(meta_path) as f:
            m = json.load(f)
        stem = os.path.splitext(os.path.basename(meta_path))[0]
        frames.append({
            "stem": stem,
            "part": m["part"],
            "class_id": int(m["class_id"]),
            "view": int(m["view"]),
            "rgb":   os.path.join(dataset_dir, "rgb",   stem + ".png"),
            "depth": os.path.join(dataset_dir, "depth", stem + ".png"),
            "mask":  os.path.join(dataset_dir, "mask",  stem + ".png"),
            "meta":  meta_path,
        })
    return frames


def split_frames(frames, holdout_frac=0.2, seed=0):
    """Deterministic per-part split into (gallery, heldout).

    Holds out a random `holdout_frac` of each part's views so the held-out set
    contains poses/lighting NOT present in the gallery. Every part keeps at
    least one held-out frame when holdout_frac > 0.
    """
    rng = np.random.default_rng(seed)
    by_part = defaultdict(list)
    for fr in frames:
        by_part[fr["part"]].append(fr)

    gallery, heldout = [], []
    for part, items in by_part.items():
        items = sorted(items, key=lambda x: x["view"])
        order = rng.permutation(len(items))
        n_hold = max(1, round(len(items) * holdout_frac)) if holdout_frac > 0 else 0
        hold_positions = set(order[:n_hold].tolist())
        for pos, fr in enumerate(items):
            (heldout if pos in hold_positions else gallery).append(fr)
    return gallery, heldout


# ---------------------------------------------------------------------------
# Image loading + mask-based foreground crop
# ---------------------------------------------------------------------------
def load_rgb(path):
    return np.array(Image.open(path).convert("RGB"))


def load_mask(path):
    return np.array(Image.open(path))  # single-channel uint8: 0 / 1 / 2


def load_depth_mm(path):
    """Return depth in millimetres (float32); background reads as 0."""
    d16 = np.array(Image.open(path)).astype(np.float32)
    return d16 * DEPTH_SCALE_MM


def masked_crop(rgb, mask, include_support=True, pad_frac=0.08,
                bg_value=128, square=True):
    """Crop to the foreground bounding box and neutralize the background.

    include_support=True treats print supports as part of the object (matches a
    real photo taken before support removal). Set False to isolate the part.
    square=True pads the crop to a square with the neutral color so the model's
    resize doesn't distort the aspect ratio.
    """
    fg = (mask == LABEL_PART)
    if include_support:
        fg = fg | (mask == LABEL_SUPPORT)

    h, w = rgb.shape[:2]
    if not fg.any():
        return Image.fromarray(rgb)  # nothing labelled foreground; embed whole frame

    out = rgb.copy()
    out[~fg] = bg_value

    ys, xs = np.where(fg)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    px = int((x1 - x0) * pad_frac)
    py = int((y1 - y0) * pad_frac)
    x0, x1 = max(0, x0 - px), min(w, x1 + px)
    y0, y1 = max(0, y0 - py), min(h, y1 + py)
    crop = out[y0:y1, x0:x1]

    if square:
        ch, cw = crop.shape[:2]
        side = max(ch, cw)
        canvas = np.full((side, side, 3), bg_value, dtype=np.uint8)
        oy, ox = (side - ch) // 2, (side - cw) // 2
        canvas[oy:oy + ch, ox:ox + cw] = crop
        crop = canvas

    return Image.fromarray(crop)


# ---------------------------------------------------------------------------
# DINO embedder (frozen, no training)
# ---------------------------------------------------------------------------
class DinoEmbedder:
    """Frozen DINOv2 / DINOv3 feature extractor producing L2-normalized vectors.

    pooling:
        "cls"        -> image-level CLS token (default, standard for retrieval)
        "cls+patch"  -> concat CLS with the mean of patch tokens (more local
                        geometry; doubles the embedding dimension)
    """
    def __init__(self, model_name="facebook/dinov2-base", device=None, pooling="cls"):
        import torch
        from transformers import AutoImageProcessor, AutoModel
        self.torch = torch
        self.pooling = pooling
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.is_v3 = "dinov3" in model_name.lower()

    def embed(self, pil_images):
        """pil_images: list[PIL.Image] -> np.ndarray (N, D) float32, unit-norm."""
        torch = self.torch
        with torch.inference_mode():
            inputs = self.processor(images=pil_images, return_tensors="pt").to(self.device)
            out = self.model(**inputs)
            if self.is_v3:
                cls = out.pooler_output
                if self.pooling == "cls+patch":
                    n_reg = int(getattr(self.model.config, "num_register_tokens", 0))
                    patch = out.last_hidden_state[:, 1 + n_reg:].mean(dim=1)
                    feat = torch.cat([cls, patch], dim=-1)
                else:
                    feat = cls
            else:
                cls = out.last_hidden_state[:, 0]
                if self.pooling == "cls+patch":
                    patch = out.last_hidden_state[:, 1:].mean(dim=1)
                    feat = torch.cat([cls, patch], dim=-1)
                else:
                    feat = cls
            feat = torch.nn.functional.normalize(feat, dim=-1)
            return feat.float().cpu().numpy()


# ---------------------------------------------------------------------------
# Embedding + metadata persistence
# ---------------------------------------------------------------------------
def save_embeddings(prefix, emb, meta):
    np.save(prefix + "_emb.npy", emb.astype("float32"))
    with open(prefix + "_meta.json", "w") as f:
        json.dump(meta, f)


def load_embeddings(prefix):
    emb = np.load(prefix + "_emb.npy")
    with open(prefix + "_meta.json") as f:
        meta = json.load(f)
    return emb, meta


# ---------------------------------------------------------------------------
# FAISS helpers (cosine similarity via inner product on unit vectors)
# ---------------------------------------------------------------------------
def build_flat_ip_index(emb):
    import faiss
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb.astype("float32"))
    return index


def aggregate_votes(sims, idxs, gallery_meta, mode="score"):
    """Turn per-render neighbours into a ranked list of parts.

    mode="score": sum cosine similarity per part (confident matches weigh more)
    mode="count": plain vote count among the k neighbours
    Returns [(part_id, value), ...] sorted descending.
    """
    score = defaultdict(float)
    count = defaultdict(int)
    for s, i in zip(sims, idxs):
        if i < 0:          # faiss pads with -1 when k > ntotal
            continue
        pid = gallery_meta[i]["part"]
        score[pid] += float(s)
        count[pid] += 1
    table = score if mode == "score" else {k: float(v) for k, v in count.items()}
    return sorted(table.items(), key=lambda kv: kv[1], reverse=True)

# SmartHarvest — Part Identification Pipeline

A vision pipeline for an automated sorting system on a 3D-printing farm. An overhead RGB-D
camera identifies which catalog part is in front of it; a robotic arm (planned) then grasps,
sorts, and packages it.

This repository is the **perception side**: synthetic data generation, a zero-training
embedding-based part-retrieval baseline, evaluation, and real-photo inference. The manipulation
side is described under [Project status](#project-status) and is not implemented here.

---

## The problem

This is **closed-set retrieval over a known parts catalog**, not open-world object recognition.
Every part the system will ever see is a CAD model we already have, which makes matching far more
tractable than general detection.

The one real complication is **print supports**. Support structures are extruded from the same
filament as the part, so at fused contact points there is no color or depth edge separating
support from part. The pipeline handles this by rendering parts *with* their slicer-generated
supports and learning invariance to them, rather than trying to segment supports away at runtime.
The synthetic renderer can label part vs. support separately (mask values `1` vs. `2`), so you
can build a gallery with or without supports in the crop and measure how much they cost you.

A second choice runs throughout: **RGB does the matching; depth is only a scale gate.** On a 1 MP
RealSense and small, glossy printed parts, depth is too noisy to match on directly. The renderer
stores true metric depth and camera intrinsics so a part's bounding-box dimensions (mm) can cheaply
pre-filter the catalog before the network runs — but see [Project status](#project-status): that
gate is not yet wired into the retrieval scripts.

---

## Pipeline

```
CAD (.stl) + slicer supports
        │  make_catalog.py
        ▼
   catalog.json  (class_id, bbox_mm, support paths)
        │  render_pyrender.py
        ▼
 dataset_pyrender/  RGB + 16-bit metric depth + 0/1/2 mask + meta
        │  build_gallery.py        (frozen DINO encoder, the slow step)
        ▼
 gallery_*  +  heldout_*  embeddings
        │  build_index.py
        ▼
   index.faiss  (exact cosine / inner-product)
        ├─ evaluate.py   held-out synthetic top-1/top-5 (sanity check only)
        └─ query.py      predict part for real photos
                 ▲
                 │  segment.py (SAM / SAM 2 masks for real photos)
```

1. **Catalog.** `make_catalog.py` scans a folder of STLs and computes every geometric field with
   trimesh — class IDs, axis-aligned `bbox_mm`, watertightness, face count, volume.
2. **Render.** `render_pyrender.py` generates domain-randomized synthetic views (pose, camera
   distance/elevation, lighting, background, filament color/finish), saving RGB, metric depth, a
   part/support segmentation mask, and per-frame meta. Rendering — not training — is the slow pole.
3. **Baseline first.** `build_gallery.py` embeds every render with a **frozen** DINOv2 (or DINOv3)
   encoder; `build_index.py` builds a FAISS index. Zero training. If the parts are visually
   distinct enough, this may be sufficient on its own.
4. **Evaluate and query.** `evaluate.py` checks held-out synthetic accuracy; `query.py` (with
   `segment.py` for masks) predicts parts from real photos — the number that actually matters.
5. **Fine-tune only if needed** (not in repo yet): a cross-entropy head for a fixed catalog, or a
   metric-learning loss (ArcFace / triplet) feeding FAISS so parts can be added without retraining.

---

## Repository layout

```
. 
├── render
│   ├── render_pyrender.py     # catalog.json + STLs       -> dataset_pyrender/    (pyrender + EGL)
├── preprocess
│   ├── segment.py             # real photos               -> 0/1 masks            (SAM / SAM 2)
├── utils/
│   ├── common.py          # dataset IO, masked_crop, DinoEmbedder, FAISS + voting helpers
│   ├── make_catalog.py        # parts/*.stl              -> parts/catalog.json   (trimesh)
├── build/
│   ├── build_gallery.py   # renders   -> gallery_*/heldout_* embeddings  (the slow step)
│   ├── build_index.py     # gallery_* -> index.faiss
├── eval/
│   ├── evaluate.py        # held-out synthetic top-1/top-5 + confusion (sanity check)
│   └── query.py           # predict catalog part for real photos
├── parts/                 # input STLs + generated catalog.json
└── dataset_pyrender/      # generated: rgb/  depth/  mask/  meta/
```

> The four retrieval scripts import `utils.common` and add the project root to `sys.path`, so keep
> `utils/` at the project root and run those scripts from one level below it (the code comments
> assume a `build/` directory). `make_catalog.py`, `render_pyrender.py`, and `segment.py` are
> self-contained and don't import `utils.common`. Adjust the tree to your real layout — only the
> `utils/` ↔ root relationship is load-bearing.

### Dataset format (`render_pyrender.py` output)

```
dataset_pyrender/
  rgb/<part>_<view:03d>.png     8-bit RGB
  depth/<part>_<view:03d>.png   16-bit PNG, depth in 0.1 mm units (background = 0)
  mask/<part>_<view:03d>.png    8-bit: 0=background  1=part  2=support
  meta/<part>_<view:03d>.json   part, class_id, view, intrinsics, cam_pose, bbox_mm, ...
```

Part names contain underscores, so labels are read from each meta JSON's `part` field, never
parsed from the filename.

---

## Requirements

- Python 3.9+
- An NVIDIA GPU is strongly recommended (rendering and embedding are both GPU-accelerated).
  Developed against an RTX 30/40-series card. EGL is used for headless rendering.
- Intel RealSense D436 (1 MP global-shutter RGB-D, active-IR stereo) for live capture.

Heavy dependencies (`torch`, `transformers`, `faiss`, `scipy`, SAM) are imported lazily, so the
pure-data utilities and `--help` work without them installed.

| Package | Used by | Notes |
|---------|---------|-------|
| `numpy`, `pillow` | all | core array + image IO |
| `trimesh` | `make_catalog.py`, `render_pyrender.py` | STL analysis, mesh transforms |
| `pyrender`, `PyOpenGL` | `render_pyrender.py` | headless render via EGL |
| `torch`, `transformers` | `utils.common.DinoEmbedder` | DINOv2 / DINOv3 from Hugging Face |
| `faiss-gpu` (or `faiss-cpu`) | indexing, eval, query | exact inner-product index |
| `scipy` | `segment.py` | optional mask clean-up (largest blob + hole fill) |
| SAM 2 / segment-anything | `segment.py` | optional, only for real-photo masks |

```bash
pip install numpy pillow trimesh pyrender PyOpenGL torch transformers faiss-gpu scipy
# faiss-cpu if you don't have a GPU FAISS build

# only if you need real-photo masks via segment.py:
pip install "git+https://github.com/facebookresearch/sam2.git"   # --backend sam2 (default)
# or: pip install segment-anything   (+ a local .pth, --backend sam)
```

> Pin a real manifest from your working environment (`pip freeze > requirements.txt`) before
> publishing — the table above is the minimal set implied by the imports, not a locked list.

---

## Usage

### 1. Build the catalog

```bash
python make_catalog.py --parts parts
python make_catalog.py --parts parts --supports supports --scale 25.4
```

Reads `parts/*.stl`, writes `parts/catalog.json`. Supports are matched by the
`<part>_support.stl` naming convention in the `--supports` folder.

**Units matter.** STL files carry no unit. The 3D-printing convention is millimeters and the depth
gate assumes mm; pass `--scale 25.4` for inch-exported STLs. Always sanity-check the printed
`bbox_mm` against a known physical dimension. `bbox_mm` is the axis-aligned `[dx, dy, dz]` extent
in mesh-axis order (not sorted). Non-watertight meshes are flagged and skip the volume field.

### 2. Render synthetic data

```bash
python render_pyrender.py --parts parts --out dataset_pyrender --views 50
```

| Flag | Default | Purpose |
|------|---------|---------|
| `--parts` | `parts` | folder holding `catalog.json` + STLs |
| `--out` | `dataset_pyrender` | output dataset root |
| `--views` | `50` | randomized views per part |
| `--res` | `512` | square render resolution |
| `--seed` | `0` | reproducible randomization |

Camera elevation is sampled 10°–80° to mirror an overhead-ish capture; distance is set from the
combined part+support extent so framing stays stable. Part and support share one filament material
(they print together) — the **mask**, not color, separates them.

### 3. Embed renders into a gallery (the slow step)

```bash
python build/build_gallery.py --dataset dataset_pyrender
```

Splits each part's views into a gallery and a held-out set, embeds both with a frozen encoder, and
writes `gallery_emb.npy` / `gallery_meta.json` and `heldout_emb.npy` / `heldout_meta.json`.

| Flag | Default | Purpose |
|------|---------|---------|
| `--model` | `facebook/dinov2-base` | encoder; e.g. `facebook/dinov3-vitb16-pretrain-lvd1689m` |
| `--pooling` | `cls` | `cls`, or `cls+patch` (concat patch-token mean; doubles dim) |
| `--holdout-frac` | `0.2` | per-part fraction held out for eval (`0` disables; each part keeps ≥1) |
| `--no-support` | off | exclude supports from the foreground crop |
| `--batch-size` | `32` | embedding batch size |
| `--seed` | `0` | controls the gallery / held-out split |

Building a gallery with and without `--no-support` is how you quantify the cost of supports.

### 4. Build the FAISS index

```bash
python build/build_index.py --gallery-prefix gallery --index-out index.faiss
```

Embeddings are L2-normalized, so an exact `IndexFlatIP` gives cosine similarity. Flat search is
exact and fast to ~1M vectors; only move to IVF/HNSW if it becomes a bottleneck.

### 5. Evaluate on held-out synthetic renders

```bash
python build/evaluate.py
python build/evaluate.py --k 30 --vote count
```

Reports top-1 / top-5, per-part accuracy, and the most-confused part pairs. `--vote score` (default)
sums cosine similarity per part across the `k` neighbours; `--vote count` is a plain vote.

> ⚠️ **Synthetic eval is a sanity check, not a ship signal.** High held-out accuracy proves the
> encoder can separate your parts *in principle*; it says nothing about the sim-to-real gap. The
> number that decides whether you ship is real-photo accuracy from step 6.

### 6. Query real photos

```bash
# first segment, so the object is cropped before embedding:
python segment.py real_photos/ --out-dir real_masks/ --preview-dir previews/
python build/query.py real_photos/ --mask-dir real_masks/
```

Use the **same** `--model` / `--pooling` / `--no-support` flags you used in `build_gallery.py`.
Without a mask, the whole frame is embedded and your randomized-background renders carry the
sim-to-real load — it still works, but expect lower accuracy.

`segment.py` (SAM 2 by default, classic SAM with `--backend sam --checkpoint …`) prompts with a
centre point unless you pass `--points "x,y;x,y"` or `--box`, and writes single-channel `0/1`
masks plus optional overlays. Note SAM only separates foreground from background — it cannot label
supports, so its masks contain only `0` and `1`. That's compatible with the pipeline
(`masked_crop` folds part and support together unless `--no-support`), but `--no-support` has no
effect on a SAM mask since there are no label-`2` pixels to drop.

---

## Design notes

- **Closed-set, not open-world.** The catalog is fully known in advance, which is what makes a
  frozen-encoder baseline a reasonable first cut.
- **RGB matches, depth gates.** Depth on this sensor is too noisy on small glossy parts to match
  on; the renderer stores metric depth + intrinsics + `bbox_mm` so a size pre-filter *can* shrink
  the candidate set, but that gate is not yet wired into the retrieval scripts.
- **Baseline before training.** Frozen DINOv2 + FAISS first. Fine-tuning is cheap (hours) on this
  GPU class; rendering is the longer pole.
- **Honest evaluation.** Holding out a per-part fraction of views keeps gallery and eval disjoint,
  and synthetic accuracy is treated only as a sanity check against real-photo results.

---

## Project status

**Implemented:** catalog generation, synthetic rendering (RGB + metric depth + part/support masks),
frozen-encoder embedding, gallery/held-out splitting, exact FAISS indexing, held-out evaluation,
real-photo segmentation, and real-photo querying.

**Not yet implemented:**
- The **metric-scale gate** (step 4 of the original plan). The renderer and meta already provide
  everything it needs — metric depth, intrinsics, and per-part `bbox_mm` — but no script measures an
  observed bounding box from depth or filters candidates by size before search.
- **Fine-tuning** (cross-entropy or ArcFace/triplet), pending whether the frozen baseline separates
  similar parts.
- **Robotic-arm integration** for grasping, sorting, and packaging. Intended decomposition: grasp
  synthesis (classical, from known CAD + pose), free-space reaching (classical or optional RL), and
  contact-rich execution (the strongest RL target, where sim-to-real on contact dynamics is the main
  difficulty).

---

## License

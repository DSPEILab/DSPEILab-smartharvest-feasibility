#!/usr/bin/env python3
"""
render_pyrender.py
------------------
Fast, pure-Python synthetic data generator for the part-ID pipeline.
Headless via EGL/OSMesa. Good for quick iteration and for the metric-depth
gate (step 4): the depth buffer is true metric distance, and we save the
exact camera intrinsics so depth -> mm point cloud is exact.

For each (part, view) it writes:
  rgb/<part>_<i>.png        8-bit color
  depth/<part>_<i>.png      16-bit PNG, depth in 0.1 mm units (see meta)
  meta/<part>_<i>.json      pose, intrinsics, label, bbox_mm

Domain randomization: pose, camera distance/elevation, light rig, background
color, filament base color + roughness/metallic (PLA/PETG/ABS-like finishes).

Usage:
  python render_pyrender.py --parts parts --out dataset_pyrender --views 12
"""
import os, json, argparse, glob
os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
os.environ["__GLIX_VENDOR_LIBRARY_NAME"] = "nvidia"
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")   # set BEFORE importing pyrender

import numpy as np
import trimesh
import pyrender
from PIL import Image

# 16-bit depth PNGs store depth in units of DEPTH_SCALE mm (0.1 mm -> 0..6553 mm).
DEPTH_SCALE = 0.1

# Segmentation label scheme (saved as 8-bit single-channel PNG):
#   0 = background, 1 = part, 2 = support
LABEL_BACKGROUND, LABEL_PART, LABEL_SUPPORT = 0, 1, 2

# A few plausible FDM filament colors (linear-ish sRGB triples).
FILAMENTS = [
    (0.85, 0.12, 0.12), (0.10, 0.30, 0.75), (0.05, 0.05, 0.05),
    (0.95, 0.95, 0.95), (0.10, 0.55, 0.20), (0.95, 0.65, 0.10),
    (0.55, 0.10, 0.65), (0.75, 0.75, 0.78),  # natural/grey
]


def look_at(eye, target, up=(0, 0, 1)):
    """Camera-to-world matrix (OpenGL convention: camera looks down -Z)."""
    eye, target, up = map(np.asarray, (eye, target, up))
    f = (target - eye); f = f / np.linalg.norm(f)
    s = np.cross(f, up); s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = s, u, -f, eye
    return m


def sample_pose_on_sphere(radius, rng):
    """Random camera position: full azimuth, elevation biased above the bed."""
    az = rng.uniform(0, 2 * np.pi)
    el = rng.uniform(np.deg2rad(10), np.deg2rad(80))
    x = radius * np.cos(el) * np.cos(az)
    y = radius * np.cos(el) * np.sin(az)
    z = radius * np.sin(el)
    return np.array([x, y, z])


def build_scene(mesh, support_mesh, color, rng, intr):
    """Build a randomized scene. `support_mesh` may be None.

    The part and the support are rigidly transformed by the SAME pose so they
    stay registered, then added as separate nodes so the segmentation render
    can label them independently. Returns the node handles for the SEG pass.
    """
    scene = pyrender.Scene(bg_color=[*rng.uniform(0.05, 0.95, 3), 1.0],
                           ambient_light=rng.uniform(0.02, 0.15, 3))
    # one filament material, shared by part and support (printed together, so
    # in reality they're the same color -- the mask is what separates them).
    mat = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[*color, 1.0],
        metallicFactor=float(rng.uniform(0.0, 0.25)),
        roughnessFactor=float(rng.uniform(0.35, 0.9)),
    )
    # one random pose; center on the PART centroid so framing is stable and
    # the support keeps its true offset relative to the part.
    R = trimesh.transformations.random_rotation_matrix(rng.random(3))
    part = mesh.copy(); part.apply_transform(R)
    t = -part.centroid
    part.apply_translation(t)
    part_node = scene.add(pyrender.Mesh.from_trimesh(part, material=mat, smooth=False))

    support_node = None
    all_bounds = [part.bounds]
    if support_mesh is not None:
        sup = support_mesh.copy(); sup.apply_transform(R); sup.apply_translation(t)
        support_node = scene.add(
            pyrender.Mesh.from_trimesh(sup, material=mat, smooth=False))
        all_bounds.append(sup.bounds)

    # size-aware camera distance using the combined (part+support) extent
    lo = np.min([b[0] for b in all_bounds], axis=0)
    hi = np.max([b[1] for b in all_bounds], axis=0)
    radius_mm = float(np.linalg.norm(hi - lo)) / 2
    dist = radius_mm / np.tan(intr["yfov"] / 2) * rng.uniform(1.4, 2.2)
    cam_pos = sample_pose_on_sphere(dist, rng)
    cam_pose = look_at(cam_pos, [0, 0, 0])
    cam = pyrender.PerspectiveCamera(yfov=intr["yfov"], znear=1.0, zfar=5000.0)
    scene.add(cam, pose=cam_pose)

    # 2-3 randomized lights (key + fills)
    for _ in range(rng.integers(2, 4)):
        light = pyrender.PointLight(color=rng.uniform(0.7, 1.0, 3),
                                    intensity=float(rng.uniform(2e5, 1.2e6)))
        scene.add(light, pose=look_at(sample_pose_on_sphere(dist * 1.5, rng),
                                      [0, 0, 0]))
    return scene, cam_pose, R, part_node, support_node


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="parts")
    ap.add_argument("--out", default="dataset_pyrender")
    ap.add_argument("--views", type=int, default=50)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    for sub in ("rgb", "depth", "mask", "meta"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)

    catalog = json.load(open(os.path.join(args.parts, "catalog.json")))
    rng = np.random.default_rng(args.seed)
    W = H = args.res
    yfov = np.deg2rad(45.0)
    fy = (H / 2) / np.tan(yfov / 2); fx = fy            # square pixels
    intr = {"yfov": yfov, "fx": fx, "fy": fy, "cx": W / 2, "cy": H / 2, "w": W, "h": H}

    renderer = pyrender.OffscreenRenderer(W, H)
    n = 0
    for name, info in catalog.items():
        mesh = trimesh.load(info["stl"])
        support_path = info.get("support")
        support_mesh = trimesh.load(support_path) if support_path else None
        for i in range(args.views):
            color = FILAMENTS[rng.integers(len(FILAMENTS))]
            scene, cam_pose, R, part_node, support_node = build_scene(
                mesh, support_mesh, color, rng, intr)

            # beauty pass: RGB + metric depth
            rgb, depth = renderer.render(scene)

            # segmentation pass: flat-color each node, then map color -> label.
            # We encode the class id in the red channel (1=part, 2=support).
            seg_map = {part_node: (LABEL_PART, 0, 0)}
            if support_node is not None:
                seg_map[support_node] = (LABEL_SUPPORT, 0, 0)
            seg, _ = renderer.render(scene, flags=pyrender.RenderFlags.SEG,
                                     seg_node_map=seg_map)
            mask = seg[:, :, 0].astype(np.uint8)        # 0 bg / 1 part / 2 support

            Image.fromarray(rgb).save(f"{args.out}/rgb/{name}_{i:03d}.png")
            d16 = np.clip(depth / DEPTH_SCALE, 0, 65535).astype(np.uint16)
            Image.fromarray(d16).save(f"{args.out}/depth/{name}_{i:03d}.png")
            Image.fromarray(mask).save(f"{args.out}/mask/{name}_{i:03d}.png")
            json.dump({
                "part": name, "class_id": info["class_id"], "view": i,
                "intrinsics": {k: float(v) for k, v in intr.items()},
                "cam_pose": cam_pose.tolist(),
                "object_rotation": R.tolist(),
                "depth_scale_mm": DEPTH_SCALE,
                "has_support": support_node is not None,
                "mask_labels": {"background": LABEL_BACKGROUND,
                                "part": LABEL_PART, "support": LABEL_SUPPORT},
                "bbox_mm": info["bbox_mm"],
            }, open(f"{args.out}/meta/{name}_{i:03d}.json", "w"), indent=1)
            n += 1
    renderer.delete()
    print(f"Rendered {n} frames ({len(catalog)} parts x {args.views} views) -> {args.out}/")


if __name__ == "__main__":
    main()

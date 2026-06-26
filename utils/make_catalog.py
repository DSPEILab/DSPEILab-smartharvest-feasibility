#!/usr/bin/env python3
"""
make_catalog.py
---------------
Scan a folder of .stl files and emit the catalog.json that render_pyrender.py
expects. Every geometric field is computed from the mesh by trimesh; you don't
fill anything in by hand.

IMPORTANT — units: STL files carry no unit, just raw numbers. The 3D-printing
convention is millimeters, and the renderer's depth gate assumes mm. If your
STLs are in another unit, pass --scale to convert (e.g. 25.4 for inches->mm).

Usage:
  python make_catalog.py --parts parts
  python make_catalog.py --parts parts --supports supports --scale 25.4
"""
import os, json, glob, argparse
import trimesh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="parts", help="folder containing .stl files")
    ap.add_argument("--out", default=None, help="output path (default <parts>/catalog.json)")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiply coords by this to reach mm (25.4 if STL is in inches)")
    ap.add_argument("--supports", default=None,
                    help="optional folder of support meshes named <part>_support.stl")
    args = ap.parse_args()

    out = args.out or os.path.join(args.parts, "catalog.json")
    stls = sorted(glob.glob(os.path.join(args.parts, "*.stl")))
    if not stls:
        raise SystemExit(f"No .stl files found in {args.parts!r}")

    catalog = {}
    for class_id, path in enumerate(stls):
        name = os.path.splitext(os.path.basename(path))[0]
        mesh = trimesh.load(path, force="mesh")     # concatenate multi-body STLs
        if args.scale != 1.0:
            mesh.apply_scale(args.scale)

        dx, dy, dz = (float(v) for v in mesh.extents)   # AABB side lengths in mm
        watertight = bool(mesh.is_watertight)

        entry = {
            "stl": path,
            "class_id": class_id,                       # 0..N-1 label for the classifier
            "bbox_mm": [dx, dy, dz],                     # physical size, used by step-4 gate
            "watertight": watertight,                    # QA only
            "n_faces": int(len(mesh.faces)),             # QA / perf only
            "volume": float(mesh.volume) if watertight else None,  # QA / optional gate signal
        }

        if args.supports:
            sup = os.path.join(args.supports, f"{name}_support.stl")
            if os.path.exists(sup):
                entry["support"] = sup

        catalog[name] = entry
        warn = "" if watertight else "   [NOT watertight: volume omitted, may render with holes]"
        print(f"{name:24s} bbox_mm={[round(x,2) for x in (dx,dy,dz)]}  faces={entry['n_faces']}{warn}")

    json.dump(catalog, open(out, "w"), indent=2)
    print(f"\nWrote {len(catalog)} part(s) -> {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
generate_test_parts.py
----------------------
Procedurally generate a small catalog of distinct mechanical test parts as
watertight STL meshes, at realistic FDM-printed scale (millimetres).

The goal is a *test* catalog for the part-ID pipeline: parts that are
- visually distinct (helps the DINOv2 baseline in step 2), and
- metrically distinct (helps the depth/bounding-box gate in step 4).

Dependencies: trimesh, manifold3d (boolean backend), shapely, numpy.
Each STL is exported in millimetres (1 Blender/!unit == 1 mm convention is
handled at render time; here we just keep vertex coords in mm).
"""

import os
import json
import numpy as np
import trimesh
from trimesh.creation import box, cylinder, annulus, extrude_polygon
from shapely.geometry import Polygon, box as shp_box
from shapely.affinity import rotate as shp_rotate, translate as shp_translate
from shapely.ops import unary_union

# Use the manifold engine for robust, watertight boolean ops.
ENGINE = "manifold"
OUT_DIR = "parts"


def _diff(a, b):
    return trimesh.boolean.difference([a, b], engine=ENGINE)


def _union(meshes):
    return trimesh.boolean.union(meshes, engine=ENGINE)


# ----------------------------------------------------------------------------
# Individual parts.  Each returns a single watertight Trimesh in mm.
# ----------------------------------------------------------------------------

def l_bracket():
    """L-shaped mounting bracket with two through-holes, ~40x40x30 mm."""
    t = 4.0                      # wall thickness
    a, b, depth = 40.0, 30.0, 30.0
    vert = box((t, depth, b)); vert.apply_translation((t / 2, depth / 2, b / 2))
    horiz = box((a, depth, t)); horiz.apply_translation((a / 2, depth / 2, t / 2))
    part = _union([vert, horiz])
    # mounting holes through the horizontal flange (drill along Z)
    for cx in (18.0, 30.0):
        hole = cylinder(radius=3.0, height=t * 3, sections=48)
        hole.apply_translation((cx, depth / 2, t / 2))
        part = _diff(part, hole)
    # a fillet-ish chamfer cut on the outer corner for visual character
    cham = box((10, depth * 1.2, 10))
    cham.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 4, [0, 1, 0]))
    cham.apply_translation((a, depth / 2, b))
    part = _diff(part, cham)
    return part


def hex_standoff():
    """Hexagonal standoff with an axial bore, ~14 mm across flats x 25 mm."""
    af = 14.0                    # across-flats
    R = af / np.sqrt(3)          # circumradius
    h = 25.0
    ang = np.deg2rad(np.arange(6) * 60 + 30)
    poly = Polygon(np.c_[R * np.cos(ang), R * np.sin(ang)])
    part = extrude_polygon(poly, height=h)
    bore = cylinder(radius=4.0, height=h * 3, sections=48)
    return _diff(part, bore)


def flanged_tube():
    """Pipe flange: thin disc flange + tubular neck, bored through. ~50 mm dia."""
    flange = cylinder(radius=25.0, height=5.0, sections=64)
    flange.apply_translation((0, 0, 2.5))
    neck = cylinder(radius=12.0, height=30.0, sections=64)
    neck.apply_translation((0, 0, 30.0 / 2 + 5.0))
    part = _union([flange, neck])
    bore = cylinder(radius=8.0, height=120.0, sections=64)
    part = _diff(part, bore)
    # bolt circle of 4 holes in the flange
    for k in range(4):
        th = np.deg2rad(45 + 90 * k)
        hole = cylinder(radius=2.5, height=20.0, sections=32)
        hole.apply_translation((19 * np.cos(th), 19 * np.sin(th), 2.5))
        part = _diff(part, hole)
    return part


def stepped_shaft():
    """Two coaxial cylinders of different diameter (a turned shaft). ~60 mm."""
    big = cylinder(radius=10.0, height=25.0, sections=64)
    big.apply_translation((0, 0, 12.5))
    small = cylinder(radius=6.0, height=35.0, sections=64)
    small.apply_translation((0, 0, 25.0 + 35.0 / 2))
    part = _union([big, small])
    # a flat (D-cut) on the small end so orientation is recoverable
    flat = box((20, 20, 35.0))
    flat.apply_translation((10 + 4.5, 0, 25.0 + 35.0 / 2))
    return _diff(part, flat)


def spur_gear(teeth=12, module=2.0, width=8.0, bore=5.0):
    """A simple spur gear: pitch disc + trapezoidal teeth, central bore."""
    pitch_r = module * teeth / 2.0
    root_r = pitch_r - 1.25 * module
    add_r = pitch_r + 1.0 * module
    base = Polygon(np.c_[root_r * np.cos(np.linspace(0, 2 * np.pi, 96)),
                         root_r * np.sin(np.linspace(0, 2 * np.pi, 96))])
    tooth_w = np.pi * module * 0.55          # tooth thickness at root
    geoms = [base]
    for k in range(teeth):
        th = 360.0 / teeth * k
        # a radial trapezoid spanning root_r -> add_r
        tip_w = tooth_w * 0.6
        t = Polygon([(-tooth_w / 2, root_r - 0.5), (tooth_w / 2, root_r - 0.5),
                     (tip_w / 2, add_r), (-tip_w / 2, add_r)])
        t = shp_rotate(t, th, origin=(0, 0))
        geoms.append(t)
    profile = unary_union(geoms).buffer(0)
    part = extrude_polygon(profile, height=width)
    hole = cylinder(radius=bore, height=width * 3, sections=48)
    return _diff(part, hole)


def u_channel():
    """Extruded U-profile channel / clip, ~30 x 20 x 40 mm."""
    outer = shp_box(0, 0, 30, 20)
    inner = shp_box(4, 4, 26, 20.1)          # open top
    profile = outer.difference(inner)
    part = extrude_polygon(profile, height=40.0)
    # a couple of slots in the back wall
    for z in (12, 28):
        slot = box((20, 8, 6))
        slot.apply_translation((15, 2, z))
        part = _diff(part, slot)
    return part


def vented_enclosure():
    """A small box enclosure with a grid of vent holes on the lid. ~45 mm."""
    outer = box((45, 35, 20)); outer.apply_translation((0, 0, 10))
    inner = box((39, 29, 18)); inner.apply_translation((0, 0, 10))
    part = _diff(outer, inner)
    vents = []
    for ix in np.linspace(-15, 15, 5):
        for iy in np.linspace(-10, 10, 3):
            v = cylinder(radius=1.6, height=10, sections=20)
            v.apply_translation((ix, iy, 20))
            vents.append(v)
    return _diff(part, _union(vents))


def knurled_knob():
    """A round knob with a fluted (knurled-ish) rim and a blind bore. ~30 mm."""
    body = cylinder(radius=15.0, height=14.0, sections=96)
    body.apply_translation((0, 0, 7))
    flutes = []
    for k in range(16):
        th = np.deg2rad(360 / 16 * k)
        f = cylinder(radius=2.0, height=20, sections=16)
        f.apply_translation((15 * np.cos(th), 15 * np.sin(th), 7))
        flutes.append(f)
    part = _diff(body, _union(flutes))
    bore = cylinder(radius=4.0, height=10, sections=32)
    bore.apply_translation((0, 0, 3))
    return _diff(part, bore)


PARTS = {
    "l_bracket": l_bracket,
    "hex_standoff": hex_standoff,
    "flanged_tube": flanged_tube,
    "stepped_shaft": stepped_shaft,
    "spur_gear": spur_gear,
    "u_channel": u_channel,
    "vented_enclosure": vented_enclosure,
    "knurled_knob": knurled_knob,
}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    catalog = {}
    for i, (name, fn) in enumerate(PARTS.items()):
        mesh = fn()
        # recenter on XY, sit on Z=0 (the print-bed convention)
        mesh.apply_translation((-mesh.centroid[0], -mesh.centroid[1],
                                -mesh.bounds[0][2]))
        path = os.path.join(OUT_DIR, f"{name}.stl")
        mesh.export(path)
        ext = (mesh.bounds[1] - mesh.bounds[0])
        catalog[name] = {
            "class_id": i,
            "stl": path,
            "watertight": bool(mesh.is_watertight),
            "n_faces": int(len(mesh.faces)),
            "bbox_mm": [round(float(x), 2) for x in ext],
            "volume_mm3": round(float(mesh.volume), 1),
        }
        print(f"{name:18s} watertight={mesh.is_watertight!s:5s} "
              f"faces={len(mesh.faces):6d} bbox(mm)={ext.round(1)}")
    with open(os.path.join(OUT_DIR, "catalog.json"), "w") as f:
        json.dump(catalog, f, indent=2)
    print(f"\nWrote {len(PARTS)} STL files + catalog.json to ./{OUT_DIR}/")


if __name__ == "__main__":
    main()

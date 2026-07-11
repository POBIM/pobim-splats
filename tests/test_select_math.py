# bpy-free tests for select_math. Run: python3 tests/test_select_math.py

import importlib.util
import math
import os

import numpy as np

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    'select_math', os.path.join(_root, 'pobim_splats', 'select_math.py'))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
points_in_polygon = _mod.points_in_polygon
points_near_polyline = _mod.points_near_polyline
points_in_sphere = _mod.points_in_sphere
points_in_box = _mod.points_in_box


def _star(cx, cy, R, r, n=5, rot=math.pi / 2):
    """A concave n-point star (2n vertices, outer/inner radius alternating)."""
    verts = []
    for i in range(2 * n):
        ang = rot + math.pi * i / n
        rad = R if i % 2 == 0 else r
        verts.append((cx + rad * math.cos(ang), cy + rad * math.sin(ang)))
    return verts


def test_polygon_concave_l_shape():
    # Concave L-shape (same shape as the measure test's area-3 polygon).
    ell = [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]

    inside = np.array([
        [0.5, 0.5],   # bottom bar
        [1.5, 0.5],   # bottom-right of the bar
        [0.5, 1.5],   # up the left arm
        [1.9, 0.5],   # near the right edge, still inside
    ], np.float32)
    outside = np.array([
        [1.5, 1.5],   # the concave notch — outside
        [2.1, 0.5],   # just past the right edge
        [3.0, 3.0],   # far away
        [-1.0, -1.0],  # far away, other side
        [0.5, 2.5],   # above the shape
    ], np.float32)

    for winding in (ell, list(reversed(ell))):
        assert points_in_polygon(inside, winding).all()
        assert not points_in_polygon(outside, winding).any()


def test_polygon_star():
    R, r = 1.0, 0.4
    star = _star(0.0, 0.0, R, r)
    star_arr = np.asarray(star, np.float32)

    # Tips are the even-indexed (outer) vertices; pull them slightly inward.
    tips_in = star_arr[0::2] * 0.95
    # Inner vertices are odd-indexed; push them radially outward into the
    # concave notch between two points — that region is outside the star.
    notch_out = star_arr[1::2] * 1.5

    inside = np.vstack([np.array([[0.0, 0.0]], np.float32), tips_in])  # center + tips
    outside = np.vstack([notch_out, np.array([[2.0, 2.0]], np.float32)])  # notches + far

    for winding in (star, list(reversed(star))):
        assert points_in_polygon(inside, winding).all()
        assert not points_in_polygon(outside, winding).any()


def test_polyline_chunk_boundary_vs_bruteforce():
    rng = np.random.default_rng(1234)
    # Dense diagonal stroke from (0,0) to (100,100).
    t = np.linspace(0.0, 1.0, 200)
    stroke = np.stack([t * 100.0, t * 100.0], axis=1).astype(np.float32)
    radius = 2.0

    # N > 250k chunk size, so the chunk loop runs more than once.
    n = 300_000
    px = rng.uniform(-10.0, 110.0, size=(n, 2)).astype(np.float32)

    mask = points_near_polyline(px, stroke, radius)
    assert mask.shape == (n,)
    assert mask.dtype == bool

    # Brute-force reference on a 1k subsample (nearest stroke *point*).
    sub_idx = rng.choice(n, size=1000, replace=False)
    sub = px[sub_idx]
    d = np.sqrt(((sub[:, None, :] - stroke[None, :, :]) ** 2).sum(-1)).min(axis=1)
    ref = d <= radius
    assert np.array_equal(mask[sub_idx], ref)


def test_sphere_basics():
    world = np.array([
        [0.0, 0.0, 0.0],   # center — inside
        [0.5, 0.5, 0.5],   # dist ~0.866 — inside
        [0.0, 0.0, 1.0],   # on the boundary — inside (<=)
        [1.0, 1.0, 1.0],   # dist ~1.732 — outside
        [2.0, 0.0, 0.0],   # outside
    ], np.float32)
    mask = points_in_sphere(world, (0.0, 0.0, 0.0), 1.0)
    assert list(mask) == [True, True, True, False, False]


def test_box_basics():
    world = np.array([
        [0.5, 0.5, 0.5],    # inside
        [0.0, 0.0, 0.0],    # min corner — inside
        [1.0, 1.0, 1.0],    # max corner — inside
        [-0.1, 0.5, 0.5],   # outside (below x)
        [0.5, 1.1, 0.5],    # outside (above y)
    ], np.float32)
    expected = [True, True, True, False, False]
    assert list(points_in_box(world, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))) == expected
    # Corners given in swapped order must give the same result.
    assert list(points_in_box(world, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))) == expected


def test_empty_inputs():
    ell = [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]
    stroke = np.array([[0.0, 0.0], [1.0, 1.0]], np.float32)

    r = points_in_polygon(np.zeros((0, 2), np.float32), ell)
    assert r.shape == (0,) and r.dtype == bool

    r = points_near_polyline(np.zeros((0, 2), np.float32), stroke, 1.0)
    assert r.shape == (0,) and r.dtype == bool

    r = points_in_sphere(np.zeros((0, 3), np.float32), (0, 0, 0), 1.0)
    assert r.shape == (0,) and r.dtype == bool

    r = points_in_box(np.zeros((0, 3), np.float32), (0, 0, 0), (1, 1, 1))
    assert r.shape == (0,) and r.dtype == bool

    # Empty stroke -> all-False, matching the point count.
    px = np.array([[0.0, 0.0], [5.0, 5.0]], np.float32)
    r = points_near_polyline(px, np.zeros((0, 2), np.float32), 1.0)
    assert r.shape == (2,) and not r.any()


def main():
    test_polygon_concave_l_shape()
    test_polygon_star()
    test_polyline_chunk_boundary_vs_bruteforce()
    test_sphere_basics()
    test_box_basics()
    test_empty_inputs()
    print('all select_math tests passed')


if __name__ == '__main__':
    main()

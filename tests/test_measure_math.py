# bpy-free tests for measure_math. Run: python3 tests/test_measure_math.py

import importlib.util
import os

import numpy as np

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    'measure_math', os.path.join(_root, 'pobim_splats', 'measure_math.py'))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
project_to_pixels = _mod.project_to_pixels
pick_nearest = _mod.pick_nearest
scale_about_point_matrix = _mod.scale_about_point_matrix
unproject_pixel = _mod.unproject_pixel


def perspective(f=2.0):
    """Simple GL-style perspective matrix looking down -Z."""
    return np.array([
        [f, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, -1.02, -2.02],
        [0, 0, -1, 0]], np.float32)


def main():
    persp = perspective()
    w, h = 800, 600

    # point on the view axis projects to the pixel center
    px, _, valid = project_to_pixels(persp, np.array([[0.0, 0.0, -5.0]]), w, h)
    assert valid[0]
    assert np.allclose(px[0], (w / 2, h / 2), atol=1e-3)

    # point behind the camera is invalid
    _, _, valid = project_to_pixels(persp, np.array([[0.0, 0.0, 5.0]]), w, h)
    assert not valid[0]

    # pick_nearest: nearest to the mouse within radius wins over far ones
    points = np.array([
        [0.0, 0.0, -5.0],    # center
        [1.0, 0.0, -5.0],    # off to the right
    ], np.float32)
    idx = pick_nearest(persp, points, w, h, (w / 2, h / 2), radius_px=25)
    assert idx == 0, idx

    # front-most wins when two points project to the same pixel
    points = np.array([
        [0.0, 0.0, -10.0],   # far
        [0.0, 0.0, -3.0],    # near
    ], np.float32)
    idx = pick_nearest(persp, points, w, h, (w / 2, h / 2), radius_px=25)
    assert idx == 1, idx

    # nothing in range -> -1
    idx = pick_nearest(persp, points, w, h, (0, 0), radius_px=10)
    assert idx == -1

    # unproject roundtrip: project a point, unproject with its window depth
    pt = np.array([[0.7, -0.4, -6.0]], np.float32)
    px, ndc_z, valid = project_to_pixels(persp, pt, w, h)
    assert valid[0]
    depth01 = (float(ndc_z[0]) + 1.0) * 0.5
    back = unproject_pixel(persp, float(px[0, 0]), float(px[0, 1]), depth01, w, h)
    assert np.allclose(back, pt[0], atol=1e-3), back

    # scale about pivot: pivot fixed, distances scale by factor
    pivot = np.array([1.0, 2.0, 3.0])
    m = scale_about_point_matrix(pivot, 2.5)
    hp = m @ np.append(pivot, 1.0)
    assert np.allclose(hp[:3], pivot, atol=1e-12)
    q = np.array([2.0, 2.0, 3.0])
    hq = m @ np.append(q, 1.0)
    assert np.allclose(np.linalg.norm(hq[:3] - pivot),
                       2.5 * np.linalg.norm(q - pivot), atol=1e-9)

    print('all measure_math tests passed')


if __name__ == '__main__':
    main()

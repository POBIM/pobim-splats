# bpy-free tests for transform_math. Run: python3 tests/test_transform_math.py

import importlib.util
import os

import numpy as np

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    'transform_math', os.path.join(_root, 'pobim_splats', 'transform_math.py'))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
quat_mul_wxyz = _mod.quat_mul_wxyz
mat3_to_quat_wxyz = _mod.mat3_to_quat_wxyz
rotation_matrix = _mod.rotation_matrix
make_transform_about_pivot = _mod.make_transform_about_pivot


def _quat_to_mat3(q):
    """(w,x,y,z) -> 3x3, matching ply_loader._quat_scale_to_cov6's convention."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], np.float64)


def test_quat_mul_identity_and_compose():
    ident = np.array([1.0, 0, 0, 0])
    q = np.array([0.5, 0.5, 0.5, 0.5])
    q /= np.linalg.norm(q)
    assert np.allclose(quat_mul_wxyz(q, ident), q)
    assert np.allclose(quat_mul_wxyz(ident, q), q)

    # composing two 90-degree z-rotations = one 180-degree z-rotation
    h = np.sqrt(0.5)
    qz90 = np.array([h, 0, 0, h])          # 90 deg about +z
    qz180 = quat_mul_wxyz(qz90, qz90)
    # apply to a vector via the matrix and compare to a direct 180 rotation
    v = np.array([1.0, 0.0, 0.0])
    got = _quat_to_mat3(qz180 / np.linalg.norm(qz180)) @ v
    assert np.allclose(got, [-1.0, 0.0, 0.0], atol=1e-6)


def test_quat_mul_batched_broadcast():
    h = np.sqrt(0.5)
    dq = np.array([h, 0, 0, h])            # single delta
    qs = np.array([[1.0, 0, 0, 0],
                   [0, 1.0, 0, 0],
                   [h, h, 0, 0]])
    out = quat_mul_wxyz(dq, qs)
    assert out.shape == (3, 3 + 1)
    # first row: dq * identity == dq
    assert np.allclose(out[0], dq)


def test_mat3_to_quat_roundtrip():
    rng = np.random.default_rng(5)
    for _ in range(200):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        if q[0] < 0:
            q = -q                         # canonical hemisphere
        R = _quat_to_mat3(q)
        back = mat3_to_quat_wxyz(R)
        if back[0] < 0:
            back = -back
        assert np.allclose(back, q, atol=1e-6), (q, back)

    # known 90-deg-about-x rotation matrix
    Rx = rotation_matrix('x', np.pi / 2)
    qx = mat3_to_quat_wxyz(Rx)
    if qx[0] < 0:
        qx = -qx
    assert np.allclose(qx, [np.sqrt(0.5), np.sqrt(0.5), 0, 0], atol=1e-6)

    # batched
    stack = np.stack([_quat_to_mat3(np.array([1.0, 0, 0, 0])), Rx])
    qb = mat3_to_quat_wxyz(stack)
    assert qb.shape == (2, 4)


def test_rotate_about_pivot_fixes_pivot():
    pivot = np.array([1.0, 2.0, -0.5])
    M = make_transform_about_pivot('rotate', 'z', np.pi / 2, pivot)
    assert M.shape == (4, 4)
    # pivot is a fixed point
    ph = np.append(pivot, 1.0)
    assert np.allclose(M @ ph, ph, atol=1e-6)
    # a point at pivot + x rotates to pivot + y (90 deg about +z)
    p = np.append(pivot + np.array([1.0, 0, 0]), 1.0)
    got = M @ p
    assert np.allclose(got[:3], pivot + np.array([0, 1.0, 0]), atol=1e-6)


def test_scale_about_pivot():
    pivot = np.array([0.5, 0.5, 0.5])
    M = make_transform_about_pivot('scale', None, 2.0, pivot)
    ph = np.append(pivot, 1.0)
    assert np.allclose(M @ ph, ph, atol=1e-6)          # pivot fixed
    p = np.append(pivot + np.array([1.0, 0, 0]), 1.0)
    assert np.allclose((M @ p)[:3], pivot + np.array([2.0, 0, 0]), atol=1e-6)
    # column norms of the 3x3 block are the per-axis scale factors
    assert np.allclose(np.linalg.norm(M[:3, :3], axis=0), [2.0, 2.0, 2.0])

    # per-axis with an axis lock: only that axis scales
    M2 = make_transform_about_pivot('scale', 'y', 3.0, (0, 0, 0))
    assert np.allclose(np.diag(M2[:3, :3]), [1.0, 3.0, 1.0])


def test_move():
    M = make_transform_about_pivot('move', None, np.array([1.0, -2.0, 3.0]),
                                   (9, 9, 9))
    assert np.allclose(M[:3, :3], np.eye(3))
    assert np.allclose(M[:3, 3], [1.0, -2.0, 3.0])   # pivot irrelevant

    # scalar amount along an axis
    Mx = make_transform_about_pivot('move', 'x', 5.0, None)
    assert np.allclose(Mx[:3, 3], [5.0, 0, 0])

    # vector amount locked to an axis keeps only that component
    Ml = make_transform_about_pivot('move', 'z', np.array([1.0, 1.0, 4.0]), None)
    assert np.allclose(Ml[:3, 3], [0, 0, 4.0])


def main():
    test_quat_mul_identity_and_compose()
    test_quat_mul_batched_broadcast()
    test_mat3_to_quat_roundtrip()
    test_rotate_about_pivot_fixes_pivot()
    test_scale_about_pivot()
    test_move()
    print('all transform_math tests passed')


if __name__ == '__main__':
    main()

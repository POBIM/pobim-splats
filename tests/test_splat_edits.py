# bpy-free tests for splat_edits. Run: python3 tests/test_splat_edits.py

import importlib.util
import os
import sys
import types

import numpy as np

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# splat_edits imports `.transform_math`, so build a package shim (as in
# test_splat_export) and load the bpy-free submodules under dotted names.
_pkg_dir = os.path.join(_root, 'pobim_splats')
_pkg = types.ModuleType('pobim_splats')
_pkg.__path__ = [_pkg_dir]
sys.modules['pobim_splats'] = _pkg


def _load(name):
    spec = importlib.util.spec_from_file_location(
        f'pobim_splats.{name}', os.path.join(_pkg_dir, f'{name}.py'))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f'pobim_splats.{name}'] = mod
    spec.loader.exec_module(mod)
    return mod


transform_math = _load('transform_math')
ply_loader = _load('ply_loader')
splat_edits = _load('splat_edits')
SplatEdits = splat_edits.SplatEdits
make_transform_about_pivot = transform_math.make_transform_about_pivot


def _cov6(quats, scales_log):
    q = np.ascontiguousarray(quats, np.float32)
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    scales = np.exp(np.asarray(scales_log, np.float32))
    out = np.empty((q.shape[0], 6), np.float32)
    ply_loader._quat_scale_to_cov6(q, scales, out)
    return out


def _sigma(cov6):
    n = cov6.shape[0]
    s = np.empty((n, 3, 3))
    s[:, 0, 0], s[:, 0, 1], s[:, 0, 2] = cov6[:, 0], cov6[:, 1], cov6[:, 2]
    s[:, 1, 0], s[:, 1, 1], s[:, 1, 2] = cov6[:, 1], cov6[:, 3], cov6[:, 4]
    s[:, 2, 0], s[:, 2, 1], s[:, 2, 2] = cov6[:, 2], cov6[:, 4], cov6[:, 5]
    return s


def _make_base(n, seed=1):
    rng = np.random.default_rng(seed)
    pos = rng.normal(size=(n, 3)).astype(np.float32)
    q = rng.normal(size=(n, 4)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    sl = np.log(rng.uniform(0.01, 0.05, (n, 3))).astype(np.float32)
    return pos, q, sl


def test_lazy_init():
    ed = SplatEdits(10)
    assert ed.positions is None and ed.quats is None and ed.scales_log is None
    assert not ed.initialized and not ed.dirty.any()


def test_apply_translate():
    pos, q, sl = _make_base(50)
    ed = SplatEdits(50)
    idx = np.array([1, 5, 9])
    M = make_transform_about_pivot('move', None, np.array([1.0, 2.0, -3.0]), None)
    payload = ed.apply_matrix(idx, M, pos, q, sl)
    assert payload is not None
    assert np.allclose(ed.positions[idx], pos[idx] + [1.0, 2.0, -3.0], atol=1e-5)
    # translation leaves rotation and scale untouched
    assert np.allclose(ed.quats[idx], q[idx], atol=1e-6)
    assert np.allclose(ed.scales_log[idx], sl[idx], atol=1e-6)
    # untouched splats unchanged; dirty only on edited
    assert np.allclose(ed.positions[0], pos[0])
    assert ed.dirty[idx].all() and ed.dirty.sum() == idx.size
    assert ed.version == 1


def test_apply_rotate_covariance():
    pos, q, sl = _make_base(40, seed=2)
    ed = SplatEdits(40)
    idx = np.arange(40)
    pivot = np.array([0.3, -0.2, 0.1])
    angle = 0.7
    M = make_transform_about_pivot('rotate', 'z', angle, pivot)
    ed.apply_matrix(idx, M, pos, q, sl)

    # positions match the pure matrix transform
    ph = np.c_[pos, np.ones(len(pos))]
    expect = (M @ ph.T).T[:, :3]
    assert np.allclose(ed.positions, expect, atol=1e-5)

    # scale is preserved by a pure rotation
    assert np.allclose(ed.scales_log, sl, atol=1e-5)

    # the recomputed covariance equals R * Sigma_orig * R^T (the whole point of
    # tracking raw quats: cov transforms correctly through the quat compose)
    R = M[:3, :3]
    sig_new = _sigma(_cov6(ed.quats, ed.scales_log))
    sig_old = _sigma(_cov6(q, sl))
    sig_expect = R @ sig_old @ R.T
    assert np.abs(sig_new - sig_expect).max() < 1e-4


def test_apply_scale():
    pos, q, sl = _make_base(30, seed=3)
    ed = SplatEdits(30)
    idx = np.array([0, 2, 4])
    M = make_transform_about_pivot('scale', None, 2.0, (0, 0, 0))
    ed.apply_matrix(idx, M, pos, q, sl)
    assert np.allclose(ed.positions[idx], pos[idx] * 2.0, atol=1e-5)
    # uniform scale adds log(2) to every axis' log-scale
    assert np.allclose(ed.scales_log[idx], sl[idx] + np.log(2.0), atol=1e-5)
    assert np.allclose(ed.quats[idx], q[idx], atol=1e-5)


def test_restore_undo():
    pos, q, sl = _make_base(20, seed=4)
    ed = SplatEdits(20)
    idx = np.array([3, 7, 11])
    M = make_transform_about_pivot('rotate', 'x', 0.5, (0, 0, 0))
    _, before, after = ed.apply_matrix(idx, M, pos, q, sl)
    assert np.allclose(ed.positions[idx], after['positions'])
    # undo: restore the 'before' payload
    ed.restore(idx, before['positions'], before['quats'], before['scales_log'])
    assert np.allclose(ed.positions[idx], pos[idx], atol=1e-6)
    assert np.allclose(ed.quats[idx], q[idx], atol=1e-6)
    # redo
    ed.restore(idx, after['positions'], after['quats'], after['scales_log'])
    assert np.allclose(ed.positions[idx], after['positions'])


def test_serialize_roundtrip():
    pos, q, sl = _make_base(200, seed=6)
    ed = SplatEdits(200)
    idx = np.array([0, 50, 199, 123])
    M = make_transform_about_pivot('rotate', 'y', 0.9,
                                   np.array([0.1, 0.2, 0.3]))
    ed.apply_matrix(idx, M, pos, q, sl)

    s = ed.serialize()
    assert isinstance(s, str)

    back = SplatEdits.deserialize(s, 200)
    assert not back.initialized                 # dense arrays still lazy
    assert set(np.nonzero(back.dirty)[0]) == set(idx.tolist())
    # materialize against the same base geometry -> dense arrays match
    back.ensure(pos, q, sl)
    assert np.allclose(back.positions, ed.positions, atol=1e-6)
    assert np.allclose(back.quats, ed.quats, atol=1e-6)
    assert np.allclose(back.scales_log, ed.scales_log, atol=1e-6)
    # untouched splats equal the base (only dirty ones were overridden)
    untouched = np.setdiff1d(np.arange(200), idx)
    assert np.allclose(back.positions[untouched], pos[untouched], atol=1e-6)


def test_serialize_empty():
    ed = SplatEdits(10)                          # never edited
    back = SplatEdits.deserialize(ed.serialize(), 10)
    assert not back.dirty.any() and not back.initialized


def test_deserialize_guards():
    import base64
    import zlib

    pos, q, sl = _make_base(64, seed=7)
    ed = SplatEdits(64)
    ed.apply_matrix([1, 2, 3], make_transform_about_pivot('move', 'x', 1.0, None),
                    pos, q, sl)
    s = ed.serialize()

    # stale count must raise (would corrupt the export otherwise)
    for wrong in (63, 65, 0, 1000):
        try:
            SplatEdits.deserialize(s, wrong)
        except ValueError:
            pass
        else:
            raise AssertionError(f'count mismatch {wrong} did not raise')

    # corrupt payloads raise ValueError
    good = (64).to_bytes(8, 'little')
    for bad in ('!!!not-base64!!!',
                base64.b64encode(b'\x01').decode(),               # truncated
                base64.b64encode(good + b'junk').decode(),        # bad zlib
                base64.b64encode(good + zlib.compress(b'\x00')).decode()):
        try:
            SplatEdits.deserialize(bad, 64)
        except ValueError:
            pass
        else:
            raise AssertionError(f'corrupt payload {bad!r} did not raise')


def main():
    test_lazy_init()
    test_apply_translate()
    test_apply_rotate_covariance()
    test_apply_scale()
    test_restore_undo()
    test_serialize_roundtrip()
    test_serialize_empty()
    test_deserialize_guards()
    print('all splat_edits tests passed')


if __name__ == '__main__':
    main()

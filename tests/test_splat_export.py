# bpy-free tests for splat_export. Run: python3 tests/test_splat_export.py

import importlib.util
import os
import sys
import tempfile
import types

import numpy as np

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# The package __init__ imports bpy (absent outside Blender), so we build a
# minimal 'pobim_splats' package shim and load only the bpy-free submodules by
# path under their proper dotted names, so relative imports resolve.
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


ply_loader = _load('ply_loader')
splat_export = _load('splat_export')
load_raw_ply = ply_loader.load_raw_ply
load_gaussian_ply = ply_loader.load_gaussian_ply
export_ply = splat_export.export_ply

from make_test_ply import (
    make_torus_splats, write_compressed_gaussian_ply, write_gaussian_ply)


def test_roundtrip_keep_mask(tmp):
    """Standard ply -> export with a keep mask -> surviving rows byte-identical."""
    path = os.path.join(tmp, 'src.ply')
    out = os.path.join(tmp, 'out.ply')
    pos, scales, quat, sh0, opacity = make_torus_splats(2000)
    rng = np.random.default_rng(11)
    sh_rest = rng.uniform(-2.0, 2.0, (2000, 9)).astype(np.float32)
    write_gaussian_ply(path, pos, scales, quat, sh0, opacity, sh_rest)

    keep = np.zeros(2000, bool)
    keep[::2] = True                       # keep every other splat

    n = export_ply(path, out, keep_mask=keep)
    assert n == int(keep.sum()), n

    src = load_raw_ply(path)
    got = load_raw_ply(out)
    assert got['count'] == int(keep.sum()), got['count']
    assert got['dtype'] == src['dtype'], 'dtype/property order must be preserved'
    assert got['vertex'].tobytes() == src['vertex'][keep].tobytes(), \
        'surviving rows must be byte-identical'


def test_keep_mask_none_exports_all(tmp):
    path = os.path.join(tmp, 'src2.ply')
    out = os.path.join(tmp, 'out2.ply')
    pos, scales, quat, sh0, opacity = make_torus_splats(1500)
    write_gaussian_ply(path, pos, scales, quat, sh0, opacity)

    n = export_ply(path, out)              # keep_mask=None
    assert n == 1500, n
    src = load_raw_ply(path)
    got = load_raw_ply(out)
    assert got['vertex'].tobytes() == src['vertex'].tobytes()


def test_subsample_mapping(tmp):
    """source_indices maps mask positions to original file rows."""
    path = os.path.join(tmp, 'src3.ply')
    out = os.path.join(tmp, 'out3.ply')
    pos, scales, quat, sh0, opacity = make_torus_splats(2000)
    write_gaussian_ply(path, pos, scales, quat, sh0, opacity)

    # simulate a subsampled import of 5 loaded rows mapping to these file rows
    source_indices = np.array([5, 10, 20, 30, 40], np.int64)
    keep = np.array([True, False, True, True, False])
    expected = source_indices[keep]        # -> file rows 5, 20, 30

    n = export_ply(path, out, keep_mask=keep, source_indices=source_indices)
    assert n == expected.shape[0], n

    src = load_raw_ply(path)
    got = load_raw_ply(out)
    assert got['vertex'].tobytes() == src['vertex'][expected].tobytes()

    # None keep_mask with source_indices exports every loaded row
    out_all = os.path.join(tmp, 'out3_all.ply')
    n_all = export_ply(path, out_all, source_indices=source_indices)
    assert n_all == source_indices.shape[0]
    got_all = load_raw_ply(out_all)
    assert got_all['vertex'].tobytes() == src['vertex'][source_indices].tobytes()


def test_compressed_source(tmp):
    """Compressed ply -> export -> re-load matches the decoded originals."""
    cpath = os.path.join(tmp, 'src.compressed.ply')
    out = os.path.join(tmp, 'out_c.ply')
    pos, scales, quat, sh0, opacity = make_torus_splats(2000)
    write_compressed_gaussian_ply(cpath, pos, scales, quat, sh0, opacity)

    # canonical raw dict has the expected shape
    raw = load_raw_ply(cpath)
    assert raw['kind'] == 'canonical', raw['kind']
    assert raw['positions'].shape == (2000, 3)
    assert raw['scales_log'].shape == (2000, 3)
    assert raw['quat_wxyz'].shape == (2000, 4)
    assert raw['f_dc'].shape == (2000, 3)
    assert raw['opacity_logit'].shape == (2000,)

    n = export_ply(cpath, out)             # keep_mask=None
    dec_src = load_gaussian_ply(cpath)     # decode compressed directly
    dec_out = load_gaussian_ply(out)       # decode the exported standard ply
    assert n == dec_src.count == dec_out.count, n
    assert np.abs(dec_out.positions - dec_src.positions).max() < 1e-5
    assert np.abs(dec_out.colors - dec_src.colors).max() < 1e-5


def main():
    with tempfile.TemporaryDirectory() as tmp:
        test_roundtrip_keep_mask(tmp)
        test_keep_mask_none_exports_all(tmp)
        test_subsample_mapping(tmp)
        test_compressed_source(tmp)
    print('all splat_export tests passed')


if __name__ == '__main__':
    main()

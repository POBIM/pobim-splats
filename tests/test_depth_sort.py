# Unit tests for pobim_splats.depth_sort.compute_order (bpy-free).
#
# Verifies the uint16-bin radix order against a float-argsort reference:
# the quantized order must be bin-monotonic (an exact property) and stay
# within a small rank error of the exact depth order, and it must degrade
# gracefully on all-equal / empty / single-element inputs.
#
# Run: python3 tests/test_depth_sort.py

import importlib.util
import os

import numpy as np

# load depth_sort.py directly (bypassing the package __init__, which imports bpy)
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    'depth_sort', os.path.join(_root, 'pobim_splats', 'depth_sort.py'))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
compute_order = _mod.compute_order


def _bins(d):
    dmin, dmax = float(d.min()), float(d.max())
    if dmax <= dmin:
        return np.zeros(d.shape[0], np.uint16)
    return np.clip((d - dmin) * (65535.0 / (dmax - dmin)), 0.0, 65535).astype(np.uint16)


def test_return_shape_and_dtype():
    rng = np.random.default_rng(0)
    pos = rng.random((1000, 3), np.float32)
    row = rng.random(3).astype(np.float32)
    order, behind = compute_order(pos, row)
    assert order.dtype == np.int32, order.dtype
    assert order.shape == (1000,)
    # a permutation of 0..n-1
    assert np.array_equal(np.sort(order), np.arange(1000))
    assert isinstance(behind, int) or np.isscalar(behind)
    print('  return shape/dtype/permutation OK')


def test_bin_monotonic():
    # the depths visited in draw order must be non-decreasing in bin space —
    # this is the exact correctness property of the quantized sort
    rng = np.random.default_rng(1)
    for n in (2, 17, 10_000, 250_000):
        pos = rng.random((n, 3), np.float32) * 4.0 - 2.0
        row = rng.random(3).astype(np.float32)
        order, _ = compute_order(pos, row)
        d = pos @ row
        b = _bins(d)
        bo = b[order]
        assert np.all(np.diff(bo.astype(np.int64)) >= 0), \
            f'bins not monotonic in draw order at n={n}'
    print('  bin-monotonicity OK')


def test_rank_error_bound():
    # compared to the exact float depth order, the quantized order may swap
    # near-equal-depth elements; the mean absolute rank displacement must stay
    # tiny (spec bound: < 500 at 1M random)
    rng = np.random.default_rng(2)
    n = 1_000_000
    pos = rng.random((n, 3), np.float32) * 2.0 - 1.0
    row = rng.random(3).astype(np.float32)
    order, _ = compute_order(pos, row)
    ref = np.argsort(pos @ row, kind='stable')

    rank_new = np.empty(n, np.int64)
    rank_new[order] = np.arange(n)
    rank_ref = np.empty(n, np.int64)
    rank_ref[ref] = np.arange(n)
    mean_err = float(np.abs(rank_new - rank_ref).mean())
    max_err = int(np.abs(rank_new - rank_ref).max())
    assert mean_err < 500.0, f'mean rank error {mean_err} exceeds 500'
    print(f'  rank error at 1M: mean={mean_err:.1f} max={max_err} (bound 500) OK')


def test_all_equal_depths():
    # every splat at the same depth: any order is valid — expect identity,
    # and no crash from the zero-range guard
    pos = np.ones((100, 3), np.float32)
    row = np.array([1.0, 1.0, 1.0], np.float32)  # d = 3.0 for every row
    order, behind = compute_order(pos, row)
    assert np.array_equal(order, np.arange(100, dtype=np.int32)), order[:5]
    assert order.dtype == np.int32
    assert behind == 0
    print('  all-equal depths -> identity order OK')


def test_degenerate_sizes():
    row = np.array([0.3, -0.7, 0.2], np.float32)
    # n = 0
    o0, b0 = compute_order(np.zeros((0, 3), np.float32), row)
    assert o0.shape == (0,) and o0.dtype == np.int32 and b0 == 0
    # n = 1
    o1, b1 = compute_order(np.array([[1.0, 2.0, 3.0]], np.float32), row)
    assert np.array_equal(o1, np.array([0], np.int32)) and b1 == 0
    print('  n=0 / n=1 degenerate sizes OK')


def test_does_not_mutate_inputs():
    rng = np.random.default_rng(3)
    pos = rng.random((5000, 3), np.float32)
    row = rng.random(3).astype(np.float32)
    pos_copy = pos.copy()
    row_copy = row.copy()
    compute_order(pos, row)
    assert np.array_equal(pos, pos_copy), 'compute_order mutated positions'
    assert np.array_equal(row, row_copy), 'compute_order mutated row'
    print('  inputs not mutated OK')


def main():
    test_return_shape_and_dtype()
    test_bin_monotonic()
    test_rank_error_bound()
    test_all_equal_depths()
    test_degenerate_sizes()
    test_does_not_mutate_inputs()
    print('all depth_sort tests passed')


if __name__ == '__main__':
    main()

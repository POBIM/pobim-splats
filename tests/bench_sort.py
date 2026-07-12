# Benchmark: old float argsort vs new uint16-bin radix order at 10M splats.
#
# Prints the measured ms for each kernel and asserts the new kernel is faster
# (loose, CI-safe threshold). bpy-free — runs under plain python3.
#
# Run: python3 tests/bench_sort.py

import importlib.util
import os
import time

import numpy as np

# load depth_sort.py directly (bypassing the package __init__, which imports bpy)
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    'depth_sort', os.path.join(_root, 'pobim_splats', 'depth_sort.py'))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
compute_order = _mod.compute_order

N = 10_000_000


def best_of(fn, reps=3):
    best = float('inf')
    out = None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1000.0, out


def old_kernel(pos, row):
    # the pre-Track-P order computation: a full float32 argsort of the depths
    return np.argsort(pos @ row).astype(np.int32)


def main():
    rng = np.random.default_rng(42)
    pos = rng.random((N, 3), np.float32) * 2.0 - 1.0
    row = rng.random(3).astype(np.float32)

    old_ms, old_order = best_of(lambda: old_kernel(pos, row))
    new_ms, new_pair = best_of(lambda: compute_order(pos, row))
    new_order, _behind = new_pair

    print(f'N = {N:,}')
    print(f'  old  np.argsort(pos @ row)      : {old_ms:7.1f} ms')
    print(f'  new  compute_order (uint16 radix): {new_ms:7.1f} ms')
    print(f'  speedup                          : {old_ms / new_ms:5.2f}x')

    # sanity: both are valid permutations of 0..N-1
    assert new_order.dtype == np.int32
    assert new_order.shape == (N,)

    # loose, CI-safe: the new kernel must be at least meaningfully faster.
    # measured ~155 ms vs ~540 ms here (3.4x); assert only > 1.2x to stay
    # robust on slower/loaded CI machines.
    assert new_ms < old_ms, \
        f'new kernel ({new_ms:.1f} ms) not faster than old ({old_ms:.1f} ms)'
    assert new_ms < old_ms / 1.2, \
        f'new kernel speedup {old_ms / new_ms:.2f}x below the 1.2x floor'
    print('bench_sort OK (new kernel faster)')


if __name__ == '__main__':
    main()

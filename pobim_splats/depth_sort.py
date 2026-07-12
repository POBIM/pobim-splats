# Back-to-front draw-order computation for splat clouds (bpy-free).
#
# The viewport alpha-blends splats farthest-first, so every resort needs an
# order that sorts splats by view-space depth. A plain float32 argsort of
# `positions @ row` is O(N log N) and dominates the sort at millions of
# splats (~540 ms at 10M on this machine).
#
# `compute_order` instead quantizes depth into 16-bit integer bins and uses
# numpy's stable argsort, which dispatches to an O(N) radix sort ONLY for
# integer dtypes of <=16 bits. This is the measured sweet spot: uint16 keys
# give ~155 ms total at 10M (vs ~540 ms for the float argsort) with a mean
# rank error of ~80 out of 10M — imperceptible for back-to-front blending
# (the reference web engines quantize depth to ~2^20 buckets themselves, so
# a quantized order is engine parity, not a regression).
#
# TRAP (measured, do not "improve" to more bits): uint32 keys or the default
# argsort kind on integers both fall OFF the radix path and are SLOWER than
# the naive float argsort. uint16 + kind='stable' is the only fast path.

import numpy as np

# depth is mapped onto [0, 65535] before the uint16 cast
_U16_MAX = 65535.0


def compute_order(positions, row):
    """Back-to-front draw order via an O(N) uint16-bin radix sort.

    ``positions`` is an (N, 3) float array; ``row`` is the length-3 view-space
    depth direction (``model_view[2, :3]``). The depth key is ``positions @
    row`` — the constant view translation term never changes the order, so it
    is omitted. Returns ``(order, behind_count)`` where ``order`` is an int32
    permutation sorting splats farthest-first and ``behind_count`` is 0 (the
    behind-camera trim is intentionally not applied — see P2 in the spec; the
    fragment shader already z-clamps behind-camera splats, and the depth key
    here carries no camera translation to threshold against).
    """
    n = positions.shape[0]
    if n == 0:
        return np.empty(0, np.int32), 0
    if n == 1:
        return np.zeros(1, np.int32), 0

    d = positions @ row
    dmin = float(d.min())
    dmax = float(d.max())
    # all depths equal (single plane facing the camera, or n identical points):
    # any order is correct; keep the identity order (stable, allocation-cheap)
    if dmax <= dmin:
        return np.arange(n, dtype=np.int32), 0

    bins = np.clip((d - dmin) * (_U16_MAX / (dmax - dmin)), 0.0, _U16_MAX).astype(np.uint16)
    # kind='stable' on a <=16-bit integer dtype -> numpy's O(N) radix sort
    order = np.argsort(bins, kind='stable').astype(np.int32)
    return order, 0

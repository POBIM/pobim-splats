# Pure-numpy math for the selection tools (lasso/polygon/brush/sphere/box)
# — bpy-free and unit-testable.

import numpy as np


def points_in_polygon(px, poly):
    """Vectorized even-odd (ray-casting) point-in-polygon test.

    px:   (N,2) float32 pixel positions.
    poly: list[(x,y)] with len >= 3 — the closed polygon (last vertex is
          joined back to the first automatically).
    Returns a bool (N,) mask, True where the point is inside.

    The even-odd rule is winding-independent (clockwise and
    counter-clockwise polygons give the same result) and horizontal edges
    contribute no crossings, so they are handled correctly. N == 0 yields an
    empty mask.
    """
    px = np.asarray(px, np.float32)
    n = px.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=bool)

    poly = np.asarray(poly, np.float32)
    if poly.ndim != 2 or poly.shape[0] < 3:
        return np.zeros((n,), dtype=bool)

    x = px[:, 0]
    y = px[:, 1]

    x1 = poly[:, 0]
    y1 = poly[:, 1]
    x2 = np.roll(x1, -1)  # next vertex x (wraps around)
    y2 = np.roll(y1, -1)  # next vertex y (wraps around)

    inside = np.zeros(n, dtype=bool)
    for ax, ay, bx, by in zip(x1, y1, x2, y2):
        denom = by - ay
        if denom == 0.0:
            # Horizontal edge: a horizontal ray never crosses it.
            continue
        # The ray (from the point toward +x) crosses this edge when the
        # edge straddles the point's y and the crossing x is to the right.
        straddles = (ay > y) != (by > y)
        xint = (bx - ax) * (y - ay) / denom + ax
        inside ^= straddles & (x < xint)
    return inside


def points_near_polyline(px, stroke, radius):
    """True where a point is within ``radius`` of ANY stroke point.

    px:     (N,2) float32 pixel positions.
    stroke: (S,2) ordered pixel points (S is capped by the caller, <= 512).
    radius: scalar screen-space radius.

    Distance is measured to the nearest stroke *point* (not segment); the
    caller keeps stroke points dense so this is accurate enough. The N x S
    distance matrix is bounded by processing px in chunks of 250k rows.
    Returns a bool (N,) mask. N == 0 or an empty stroke yields all-False.
    """
    px = np.asarray(px, np.float32)
    n = px.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=bool)

    stroke = np.asarray(stroke, np.float32)
    if stroke.ndim != 2 or stroke.shape[0] == 0:
        return np.zeros((n,), dtype=bool)

    r2 = float(radius) * float(radius)
    sx = stroke[:, 0][None, :]
    sy = stroke[:, 1][None, :]

    out = np.zeros(n, dtype=bool)
    CHUNK = 250_000
    for s in range(0, n, CHUNK):
        e = min(s + CHUNK, n)
        cx = px[s:e, 0][:, None]
        cy = px[s:e, 1][:, None]
        d2 = (cx - sx) ** 2 + (cy - sy) ** 2  # (rows, S)
        out[s:e] = (d2 <= r2).any(axis=1)
    return out


def points_in_sphere(world, center, radius):
    """world (N,3) float32; True where |point - center| <= radius."""
    world = np.asarray(world, np.float32)
    n = world.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=bool)
    c = np.asarray(center, np.float32)
    d2 = ((world - c) ** 2).sum(axis=1)
    return d2 <= float(radius) * float(radius)


def points_in_box(world, bmin, bmax):
    """world (N,3) float32; axis-aligned box test in the SAME space as
    ``world`` (the caller passes splat-local positions with local corners
    for an object-aligned box). The corners may be given in any order —
    they are normalized to min/max internally. Returns a bool (N,) mask.
    """
    world = np.asarray(world, np.float32)
    n = world.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=bool)
    a = np.asarray(bmin, np.float32)
    b = np.asarray(bmax, np.float32)
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    return np.all((world >= lo) & (world <= hi), axis=1)

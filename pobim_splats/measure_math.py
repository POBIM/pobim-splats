# Pure-numpy math for the Measure & Scale tool — bpy-free and unit-testable.

import numpy as np


def project_to_pixels(persp, points, width, height):
    """Project world points through a 4x4 perspective matrix to pixels.

    persp: (4,4) row-major numpy matrix (Blender's rv3d.perspective_matrix).
    points: (N,3) world positions.
    Returns (px (N,2), ndc_z (N,), valid (N,) bool) — valid means in front
    of the camera and inside clip bounds with margin.
    """
    hom = points @ persp[:3, :3].T + persp[:3, 3]
    w = points @ persp[3, :3] + persp[3, 3]
    valid = w > 1e-9
    w_safe = np.where(valid, w, 1.0)
    ndc = hom / w_safe[:, None]
    px = np.empty((points.shape[0], 2), np.float32)
    px[:, 0] = (ndc[:, 0] * 0.5 + 0.5) * width
    px[:, 1] = (ndc[:, 1] * 0.5 + 0.5) * height
    valid &= (np.abs(ndc[:, 0]) <= 1.2) & (np.abs(ndc[:, 1]) <= 1.2)
    return px, ndc[:, 2].astype(np.float32), valid


def pick_nearest(persp, points, width, height, mouse_xy, radius_px=25.0):
    """Pick the front-most point whose projection is within radius of the mouse.

    Front-most (smallest ndc z) matches how users expect to pick surfaces —
    the visible splat wins over ones hidden behind it.
    Returns index into points, or -1 when nothing is in range.
    """
    px, ndc_z, valid = project_to_pixels(persp, points, width, height)
    d2 = ((px - np.asarray(mouse_xy, np.float32)) ** 2).sum(axis=1)
    candidates = valid & (d2 <= radius_px * radius_px)
    if not candidates.any():
        return -1
    z = np.where(candidates, ndc_z, np.inf)
    return int(np.argmin(z))


def unproject_pixel(persp, mx, my, depth01, width, height):
    """Pixel + window depth (gl_FragCoord.z, 0..1) -> world position."""
    ndc = np.array([
        2.0 * mx / width - 1.0,
        2.0 * my / height - 1.0,
        2.0 * depth01 - 1.0,
        1.0], np.float64)
    h = np.linalg.inv(np.asarray(persp, np.float64)) @ ndc
    return (h[:3] / h[3]).astype(np.float32)


def polygon_area(points):
    """0.5 * |sum (pj-p0) x (pj+1-p0)| — same fan-cross as POBIMStudio."""
    points = np.asarray(points, np.float64)
    if len(points) < 3:
        return 0.0
    p0 = points[0]
    acc = np.zeros(3, np.float64)
    for j in range(1, len(points) - 1):
        acc += np.cross(points[j] - p0, points[j + 1] - p0)
    return float(np.linalg.norm(acc) * 0.5)


def polygon_perimeter(points, closed=True):
    points = np.asarray(points, np.float64)
    n = len(points)
    if n < 2:
        return 0.0
    total = sum(float(np.linalg.norm(points[i + 1] - points[i]))
                for i in range(n - 1))
    if closed and n > 2:
        total += float(np.linalg.norm(points[0] - points[-1]))
    return total


def box_corners(pmin, pmax):
    """8 corners of an axis-aligned box, z-major then y then x."""
    return [np.array([x, y, z], np.float32)
            for z in (pmin[2], pmax[2])
            for y in (pmin[1], pmax[1])
            for x in (pmin[0], pmax[0])]


BOX_EDGES = ((0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7),
             (0, 4), (1, 5), (2, 6), (3, 7))


def scale_about_point_matrix(pivot, factor):
    """4x4 world matrix that scales uniformly by factor, keeping pivot fixed."""
    m = np.eye(4, dtype=np.float64)
    m[0, 0] = m[1, 1] = m[2, 2] = factor
    p = np.asarray(pivot, np.float64)
    m[:3, 3] = p - factor * p
    return m

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


def scale_about_point_matrix(pivot, factor):
    """4x4 world matrix that scales uniformly by factor, keeping pivot fixed."""
    m = np.eye(4, dtype=np.float64)
    m[0, 0] = m[1, 1] = m[2, 2] = factor
    p = np.asarray(pivot, np.float64)
    m[:3, 3] = p - factor * p
    return m

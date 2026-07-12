# Transform math helpers for editing splat geometry (Phase 3, Track T).
#
# Deliberately bpy-free (numpy only) so it can be unit-tested outside Blender
# and reused by the edit tools. All matrices are SPLAT-LOCAL 4x4 transforms;
# quaternions are (w, x, y, z) to match the PLY rot_0..3 layout and
# ply_loader._quat_scale_to_cov6.
#
# NOTE (documented MVP limitation): rotating a splat here does NOT rotate its
# SH coefficients, so rotated splats with SH bands >= 1 show a slight
# view-dependent color error. Fixing SH rotation is a Phase 3 non-goal.

import numpy as np

_AXIS_INDEX = {'x': 0, 'y': 1, 'z': 2}


def _axis_vec(axis):
    """Resolve an axis spec ('x'/'y'/'z' or a length-3 vector) to a unit vector.
    Returns None when axis is None."""
    if axis is None:
        return None
    if isinstance(axis, str):
        return {
            'x': np.array([1.0, 0.0, 0.0]),
            'y': np.array([0.0, 1.0, 0.0]),
            'z': np.array([0.0, 0.0, 1.0]),
        }[axis.lower()]
    v = np.asarray(axis, np.float64).ravel()
    n = np.linalg.norm(v)
    return v / n if n > 1e-30 else v


def quat_mul_wxyz(a, b):
    """Hamilton product of two (w,x,y,z) quaternions.

    Accepts single (4,) or batched (N,4) arrays and broadcasts a single against
    a batch. The result is a rotation that applies ``b`` first, then ``a``.
    """
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    single = a.ndim == 1 and b.ndim == 1
    a2 = np.atleast_2d(a)
    b2 = np.atleast_2d(b)
    aw, ax, ay, az = a2[:, 0], a2[:, 1], a2[:, 2], a2[:, 3]
    bw, bx, by, bz = b2[:, 0], b2[:, 1], b2[:, 2], b2[:, 3]
    out = np.empty((max(a2.shape[0], b2.shape[0]), 4), np.float64)
    out[:, 0] = aw * bw - ax * bx - ay * by - az * bz
    out[:, 1] = aw * bx + ax * bw + ay * bz - az * by
    out[:, 2] = aw * by - ax * bz + ay * bw + az * bx
    out[:, 3] = aw * bz + ax * by - ay * bx + az * bw
    return out[0] if single else out


def mat3_to_quat_wxyz(m):
    """Rotation matrix -> unit (w,x,y,z) quaternion.

    Inverse of the quaternion->matrix convention in
    ply_loader._quat_scale_to_cov6. Accepts a single (3,3) or batched (N,3,3)
    array (Shepperd's method, numerically stable branch selection).
    """
    m = np.asarray(m, np.float64)
    single = m.ndim == 2
    M = np.ascontiguousarray(m).reshape(-1, 3, 3)
    n = M.shape[0]
    m00, m01, m02 = M[:, 0, 0], M[:, 0, 1], M[:, 0, 2]
    m10, m11, m12 = M[:, 1, 0], M[:, 1, 1], M[:, 1, 2]
    m20, m21, m22 = M[:, 2, 0], M[:, 2, 1], M[:, 2, 2]
    tr = m00 + m11 + m22
    q = np.zeros((n, 4), np.float64)

    c0 = tr > 0.0
    c1 = (~c0) & (m00 >= m11) & (m00 >= m22)
    c2 = (~c0) & (~c1) & (m11 >= m22)
    c3 = ~(c0 | c1 | c2)

    if c0.any():
        s = np.sqrt(tr[c0] + 1.0) * 2.0
        q[c0, 0] = 0.25 * s
        q[c0, 1] = (m21[c0] - m12[c0]) / s
        q[c0, 2] = (m02[c0] - m20[c0]) / s
        q[c0, 3] = (m10[c0] - m01[c0]) / s
    if c1.any():
        s = np.sqrt(1.0 + m00[c1] - m11[c1] - m22[c1]) * 2.0
        q[c1, 0] = (m21[c1] - m12[c1]) / s
        q[c1, 1] = 0.25 * s
        q[c1, 2] = (m01[c1] + m10[c1]) / s
        q[c1, 3] = (m02[c1] + m20[c1]) / s
    if c2.any():
        s = np.sqrt(1.0 + m11[c2] - m00[c2] - m22[c2]) * 2.0
        q[c2, 0] = (m02[c2] - m20[c2]) / s
        q[c2, 1] = (m01[c2] + m10[c2]) / s
        q[c2, 2] = 0.25 * s
        q[c2, 3] = (m12[c2] + m21[c2]) / s
    if c3.any():
        s = np.sqrt(1.0 + m22[c3] - m00[c3] - m11[c3]) * 2.0
        q[c3, 0] = (m10[c3] - m01[c3]) / s
        q[c3, 1] = (m02[c3] + m20[c3]) / s
        q[c3, 2] = (m12[c3] + m21[c3]) / s
        q[c3, 3] = 0.25 * s

    q /= (np.linalg.norm(q, axis=1, keepdims=True) + 1e-30)
    return q[0] if single else q


def rotation_matrix(axis, angle):
    """3x3 rotation about a unit ``axis`` (length-3 or 'x'/'y'/'z') by ``angle``
    radians (Rodrigues' formula)."""
    a = _axis_vec(axis)
    if a is None:
        a = np.array([0.0, 0.0, 1.0])
    x, y, z = a
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1.0 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [x * y * C + z * s, c + y * y * C, y * z * C - x * s],
        [x * z * C - y * s, y * z * C + x * s, c + z * z * C],
    ], np.float64)


def make_transform_about_pivot(mode, axis, amount, pivot):
    """Build a SPLAT-LOCAL 4x4 transform about ``pivot``.

    mode:
      'move'   — translate. ``amount`` is a scalar distance along ``axis``
                 (axis required) or a length-3 delta vector (locked to ``axis``
                 when one is given). ``pivot`` is irrelevant to a translation.
      'rotate' — rotate ``amount`` radians about local ``axis`` (default +Z)
                 through ``pivot``.
      'scale'  — scale by ``amount`` (uniform scalar, or per-axis length-3)
                 about ``pivot``; ``axis`` locks scaling to that single axis.

    Rotation/scale are returned as ``T(pivot) @ core @ T(-pivot)`` so ``pivot``
    is a fixed point of the transform.
    """
    pivot = (np.zeros(3) if pivot is None
             else np.asarray(pivot, np.float64).ravel())

    if mode == 'move':
        amt = np.asarray(amount, np.float64)
        if amt.ndim == 0:
            av = _axis_vec(axis)
            if av is None:
                raise ValueError("move with a scalar amount needs an axis")
            t = av * float(amt)
        else:
            t = amt.astype(np.float64).ravel().copy()
            av = _axis_vec(axis)
            if av is not None:
                t = av * float(t @ av)   # lock the translation to the axis
        M = np.eye(4)
        M[:3, 3] = t
        return M

    core = np.eye(4)
    if mode == 'rotate':
        core[:3, :3] = rotation_matrix(axis if axis is not None else 'z',
                                       float(amount))
    elif mode == 'scale':
        s = np.asarray(amount, np.float64)
        if s.ndim == 0:
            if axis is None:
                svec = np.full(3, float(s))
            else:
                svec = np.ones(3)
                svec[_AXIS_INDEX[axis.lower()]] = float(s)
        else:
            svec = s.astype(np.float64).ravel().copy()
            if axis is not None:
                idx = _AXIS_INDEX[axis.lower()]
                locked = np.ones(3)
                locked[idx] = svec[idx]
                svec = locked
        core[:3, :3] = np.diag(svec)
    else:
        raise ValueError(f"unknown transform mode: {mode!r}")

    tp = np.eye(4)
    tp[:3, 3] = pivot
    tn = np.eye(4)
    tn[:3, 3] = -pivot
    return tp @ core @ tn

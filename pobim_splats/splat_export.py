# Lossless / canonical .ply export of surviving splats.
#
# Deliberately bpy-free so it can be unit-tested outside Blender. Re-reads the
# SOURCE file (no edited splat values are stored in RAM) and writes only the
# rows that survive an edit's keep mask:
#   - standard .ply sources are re-emitted byte-for-byte (verbatim rows,
#     original dtype/property order preserved);
#   - compressed .ply and SOG sources cannot be reproduced exactly, so they are
#     decoded to canonical arrays and written as a standard float32 3DGS .ply.

import numpy as np

from .ply_loader import load_raw_ply

# numpy (kind, itemsize) -> PLY property type name; inverse of ply_loader._PLY_TYPES
_NP_TO_PLY = {
    ('f', 4): 'float', ('f', 8): 'double',
    ('i', 1): 'char', ('u', 1): 'uchar',
    ('i', 2): 'short', ('u', 2): 'ushort',
    ('i', 4): 'int', ('u', 4): 'uint',
}


def _ply_type_name(dt):
    key = (dt.kind, dt.itemsize)
    if key not in _NP_TO_PLY:
        raise ValueError(f'ไม่รองรับชนิดข้อมูล PLY: {dt}')
    return _NP_TO_PLY[key]


def _load_raw(source_path):
    """Dispatch on extension; import the SOG loader lazily (it needs bpy)."""
    lower = source_path.lower()
    if lower.endswith('.sog') or lower.endswith('.json'):
        from .sog_loader import load_raw_sog
        return load_raw_sog(source_path)
    return load_raw_ply(source_path)


def _kept_rows(count, keep_mask, source_indices):
    """Resolve the ORIGINAL file-row indices to write.

    keep_mask indexes the LOADED cloud (which may be a subsample). When the
    import was subsampled, source_indices maps each loaded row to its file row;
    rows that were never loaded are simply absent from source_indices and are
    therefore dropped from the export.
    """
    if source_indices is not None:
        src = np.asarray(source_indices)
        if keep_mask is None:
            return src
        return src[np.asarray(keep_mask, bool)]

    if keep_mask is None:
        return np.arange(count)
    return np.nonzero(np.asarray(keep_mask, bool))[0]


def _write_ply(out_path, fields, dtype, rows):
    """Write a binary_little_endian PLY vertex element from a structured array."""
    n = rows.shape[0]
    header = 'ply\nformat binary_little_endian 1.0\n'
    header += f'element vertex {n}\n'
    for name in fields:
        header += f'property {_ply_type_name(dtype.fields[name][0])} {name}\n'
    header += 'end_header\n'
    with open(out_path, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(rows.tobytes())


def _write_canonical_ply(out_path, raw, kept):
    """Synthesize a standard float32 3DGS .ply from canonical arrays."""
    pos = raw['positions'][kept]
    f_dc = raw['f_dc'][kept]
    scales_log = raw['scales_log'][kept]
    quat = raw['quat_wxyz'][kept]
    opacity = raw['opacity_logit'][kept]
    sh = raw['sh']
    sh_sel = sh[kept] if sh is not None else None
    rest_n = 0 if sh_sel is None else sh_sel.shape[1]

    fields = (
        ['x', 'y', 'z', 'nx', 'ny', 'nz'] +
        [f'f_dc_{i}' for i in range(3)] +
        [f'f_rest_{i}' for i in range(rest_n)] +
        ['opacity'] +
        [f'scale_{i}' for i in range(3)] +
        [f'rot_{i}' for i in range(4)]
    )

    n = pos.shape[0]
    data = np.zeros((n, len(fields)), np.float32)
    data[:, 0:3] = pos
    # nx, ny, nz left zero
    data[:, 6:9] = f_dc
    if rest_n:
        data[:, 9:9 + rest_n] = sh_sel
    data[:, 9 + rest_n] = opacity
    data[:, 10 + rest_n:13 + rest_n] = scales_log
    data[:, 13 + rest_n:17 + rest_n] = quat

    header = 'ply\nformat binary_little_endian 1.0\n'
    header += f'element vertex {n}\n'
    header += ''.join(f'property float {f}\n' for f in fields)
    header += 'end_header\n'
    with open(out_path, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(np.ascontiguousarray(data, '<f4').tobytes())
    return n


def _resolve_edit_rows(edits, source_indices, count):
    """Map an edits payload's LOADED-cloud indices to ORIGINAL file rows.

    Returns (file_rows int64, positions, quats, scales_log) with the same
    length, or None when there is nothing to patch. Indices are mapped through
    ``source_indices`` (loaded-row -> file-row) exactly like the keep mask.
    """
    if not edits:
        return None
    eidx = np.asarray(edits.get('indices', []), np.int64).ravel()
    if eidx.size == 0:
        return None
    file_rows = (np.asarray(source_indices)[eidx] if source_indices is not None
                 else eidx)
    return (np.asarray(file_rows, np.int64),
            np.asarray(edits['positions'], np.float32),
            np.asarray(edits['quats'], np.float32),
            np.asarray(edits['scales_log'], np.float32))


def _patch_ply_rows(rows, kept, resolved):
    """Overwrite x/y/z, rot_0..3, scale_0..2 of the surviving edited rows.

    ``rows`` are the written rows (in ``kept`` order); untouched rows are left
    byte-identical. Edited splats that were deleted (absent from ``kept``) are
    silently skipped.
    """
    file_rows, pos, quats, scales_log = resolved
    count = kept.max() + 1 if kept.size else 0
    lut_len = max(int(count), int(file_rows.max()) + 1 if file_rows.size else 0)
    row_to_out = np.full(lut_len, -1, np.int64)
    row_to_out[kept] = np.arange(kept.shape[0])
    out_pos = row_to_out[file_rows]
    valid = out_pos >= 0
    if not valid.any():
        return
    op = out_pos[valid]
    rows['x'][op] = pos[valid, 0]
    rows['y'][op] = pos[valid, 1]
    rows['z'][op] = pos[valid, 2]
    for i in range(4):
        rows[f'rot_{i}'][op] = quats[valid, i]
    for i in range(3):
        rows[f'scale_{i}'][op] = scales_log[valid, i]


def export_ply(source_path, out_path, keep_mask=None, source_indices=None,
               edits=None):
    """Export surviving splats from source_path to a 3DGS .ply at out_path.

    keep_mask: optional bool array over the LOADED cloud; None exports all rows.
    source_indices: optional map from loaded-row -> original file-row for
        subsampled imports (SplatCloud.source_indices). Effective kept file rows
        are source_indices[keep_mask]; splats that were never loaded are DROPPED.
    edits: optional geometry-override payload, a dict with 'indices' (into the
        LOADED order), 'positions' (n,3), 'quats' (n,4 wxyz) and 'scales_log'
        (n,3). The edited splats that survive get their position/rotation/scale
        patched; every UNtouched surviving row stays byte-identical.

    Standard .ply sources keep their surviving rows byte-identical (original
    dtype and property order). Compressed .ply and SOG sources are re-emitted as
    a standard float32 3DGS .ply. Returns the number of splats written.
    """
    raw = _load_raw(source_path)
    kept = _kept_rows(raw['count'], keep_mask, source_indices)
    resolved = _resolve_edit_rows(edits, source_indices, raw['count'])

    if raw['kind'] == 'ply':
        dtype = raw['dtype']
        rows = raw['vertex'][kept].copy()   # a copy so patches never touch source
        if resolved is not None:
            _patch_ply_rows(rows, kept, resolved)
        _write_ply(out_path, list(dtype.names), dtype, rows)
        return rows.shape[0]

    if raw['kind'] == 'canonical':
        if resolved is not None:
            file_rows, pos, quats, scales_log = resolved
            raw['positions'] = raw['positions'].copy()
            raw['quat_wxyz'] = raw['quat_wxyz'].copy()
            raw['scales_log'] = raw['scales_log'].copy()
            raw['positions'][file_rows] = pos
            raw['quat_wxyz'][file_rows] = quats
            raw['scales_log'][file_rows] = scales_log
        return _write_canonical_ply(out_path, raw, kept)

    raise ValueError(f'ไม่รู้จักชนิดข้อมูลต้นทาง: {raw.get("kind")}')

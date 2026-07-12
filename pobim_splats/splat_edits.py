# Sparse per-splat GEOMETRY overrides (Phase 3, Track T1).
#
# Mirrors SplatState (selection/hidden/deleted flags) but for edited geometry:
# position, rotation quaternion and log-scale in SPLAT-LOCAL space. Dense
# override arrays are allocated lazily on the first edit (a copy of the source
# cloud geometry), and only the DIRTY splats are serialized to the .blend.
#
# Deliberately bpy-free (numpy only) so it can be unit-tested outside Blender.

import base64
import zlib

import numpy as np

from .transform_math import mat3_to_quat_wxyz, quat_mul_wxyz


class SplatEdits:
    """Editable per-splat geometry for one cloud.

    The dense arrays (``positions``/``quats``/``scales_log``) stay ``None`` until
    the first edit (or an explicit :meth:`ensure`), at which point they are
    copied from the source cloud geometry. ``dirty`` marks splats that differ
    from the source; only those survive serialization. Every mutation bumps
    ``version`` (the GPU/commit re-upload trigger).
    """

    def __init__(self, count):
        self.count = int(count)
        self.positions = None    # (N,3) f32, splat-local
        self.quats = None        # (N,4) f32 (w,x,y,z)
        self.scales_log = None   # (N,3) f32
        self.dirty = np.zeros(self.count, bool)
        self.version = 0
        # sparse overrides loaded by deserialize, applied on the next ensure()
        # (deserialize has no access to the base cloud geometry)
        self._pending = None

    @property
    def initialized(self):
        return self.positions is not None

    def ensure(self, base_positions, base_quats, base_scales_log):
        """Materialize the dense override arrays from the source geometry.

        Idempotent: a no-op once initialized. When the state was loaded from a
        serialized payload, the pending sparse overrides are applied here.
        """
        if self.positions is not None:
            return
        self.positions = np.array(base_positions, np.float32)      # copy
        self.quats = np.array(base_quats, np.float32)
        self.scales_log = np.array(base_scales_log, np.float32)
        if self._pending is not None:
            idx, pos, quats, sl = self._pending
            if idx.size:
                self.positions[idx] = pos
                self.quats[idx] = quats
                self.scales_log[idx] = sl
                self.dirty[idx] = True
                self.version += 1
            self._pending = None

    def apply_matrix(self, indices, mat4_local,
                     base_positions, base_quats, base_scales_log):
        """Apply a 4x4 LOCAL-space transform (about an arbitrary pivot) to
        ``indices``.

        positions <- M @ p;  quats <- quat(M_rot) * q;  scales_log +=
        log(per-axis scale factors) taken from the column norms of M's 3x3
        block (uniform and per-axis scale both work). Marks the indices dirty
        and bumps ``version``.

        Returns a history payload ``(indices, before, after)`` where ``before``
        and ``after`` are dicts of the changed splats' ``positions``/``quats``/
        ``scales_log`` (copies), or ``None`` when ``indices`` is empty.
        """
        self.ensure(base_positions, base_quats, base_scales_log)
        idx = np.asarray(indices, np.int64).ravel()
        if idx.size == 0:
            return None

        M = np.asarray(mat4_local, np.float64)
        R = M[:3, :3]
        t = M[:3, 3]

        before = {
            'positions': self.positions[idx].copy(),
            'quats': self.quats[idx].copy(),
            'scales_log': self.scales_log[idx].copy(),
        }

        # positions: M @ p (row-vector form: p @ R^T + t)
        p = self.positions[idx].astype(np.float64)
        self.positions[idx] = (p @ R.T + t).astype(np.float32)

        # decompose the 3x3 into rotation * per-axis scale via column norms
        col_norms = np.linalg.norm(R, axis=0)
        safe = np.where(col_norms < 1e-12, 1.0, col_norms)
        Rn = R / safe[None, :]                      # orthonormal rotation part
        dq = mat3_to_quat_wxyz(Rn)                  # single (4,) delta quat

        q = quat_mul_wxyz(dq, self.quats[idx].astype(np.float64))
        q /= (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
        self.quats[idx] = q.astype(np.float32)

        self.scales_log[idx] += np.log(safe).astype(np.float32)[None, :]

        self.dirty[idx] = True
        self.version += 1

        after = {
            'positions': self.positions[idx].copy(),
            'quats': self.quats[idx].copy(),
            'scales_log': self.scales_log[idx].copy(),
        }
        return (idx, before, after)

    def restore(self, indices, positions, quats, scales_log):
        """Overwrite the geometry at ``indices`` (undo/redo restore)."""
        idx = np.asarray(indices, np.int64).ravel()
        if idx.size == 0 or self.positions is None:
            return
        self.positions[idx] = positions
        self.quats[idx] = quats
        self.scales_log[idx] = scales_log
        self.dirty[idx] = True
        self.version += 1

    def export_payload(self):
        """Sparse override payload for ``export_ply(edits=...)``:
        {'indices','positions','quats','scales_log'} covering the dirty
        splats, or None when there are no geometry edits.

        Works in BOTH states: materialized dense arrays (live editing) and
        pending overrides straight from :meth:`deserialize` (no base geometry
        needed — the pending payload already holds the final override values).
        Export paths must use this instead of reading ``positions`` directly,
        otherwise a deserialized-but-never-drawn cloud exports as unedited.
        """
        if self.positions is not None:
            idx = np.nonzero(self.dirty)[0].astype(np.int64)
            if idx.size == 0:
                return None
            return {'indices': idx,
                    'positions': self.positions[idx].copy(),
                    'quats': self.quats[idx].copy(),
                    'scales_log': self.scales_log[idx].copy()}
        if self._pending is not None:
            idx, pos, quats, sl = self._pending
            if idx.size == 0:
                return None
            return {'indices': idx.copy(), 'positions': pos.copy(),
                    'quats': quats.copy(), 'scales_log': sl.copy()}
        return None

    # -- persistence ----------------------------------------------------
    def serialize(self):
        """base64(count header + zlib(sparse dirty payload)).

        Like SplatState, the payload starts with the splat count as an 8-byte
        little-endian uint64 so :meth:`deserialize` can reject stale overrides
        after the cloud is re-imported with a different Max Splats or
        re-pointed — silently decoding into garbage would corrupt the export.
        The compressed body is: num_dirty (8-byte LE) + int64 indices + f32
        positions (n,3) + f32 quats (n,4) + f32 scales_log (n,3).
        """
        if self.positions is not None:
            idx = np.nonzero(self.dirty)[0].astype(np.int64)
            pos, quats, sl = (self.positions[idx], self.quats[idx],
                              self.scales_log[idx])
        elif self._pending is not None:
            # deserialized but never materialized: re-serialize the pending
            # overrides verbatim so a no-op session cannot wipe saved edits
            idx, pos, quats, sl = self._pending
        else:
            idx = np.zeros(0, np.int64)
            pos = quats = sl = None
        m = int(idx.size)
        parts = [m.to_bytes(8, 'little'), np.ascontiguousarray(idx, '<i8').tobytes()]
        if m:
            parts.append(np.ascontiguousarray(pos, '<f4').tobytes())
            parts.append(np.ascontiguousarray(quats, '<f4').tobytes())
            parts.append(np.ascontiguousarray(sl, '<f4').tobytes())
        body = b''.join(parts)
        payload = self.count.to_bytes(8, 'little') + zlib.compress(body, 6)
        return base64.b64encode(payload).decode('ascii')

    @staticmethod
    def deserialize(s, count):
        """Restore serialized overrides. Raises ValueError on a count mismatch
        (stale state) or a corrupt payload — callers must catch, drop the stale
        property and continue with fresh edits. The dense arrays stay lazy: the
        overrides are applied on the next :meth:`ensure` (which has the base
        cloud geometry)."""
        ed = SplatEdits(count)
        if not s:
            return ed
        try:
            payload = base64.b64decode(s)
        except Exception as e:
            raise ValueError(f'splat edits not valid base64: {e}')
        if len(payload) < 8:
            raise ValueError('splat edits payload truncated')
        stored = int.from_bytes(payload[:8], 'little')
        if stored != count:
            raise ValueError(
                f'splat edits count mismatch: stored {stored}, cloud has {count}')
        try:
            body = zlib.decompress(payload[8:])
        except Exception as e:
            raise ValueError(f'splat edits payload corrupt: {e}')
        if len(body) < 8:
            raise ValueError('splat edits payload too short')
        m = int.from_bytes(body[:8], 'little')
        need = 8 + m * 8 + m * (3 + 4 + 3) * 4
        if len(body) < need:
            raise ValueError('splat edits payload too short for dirty count')
        off = 8
        idx = np.frombuffer(body, np.int64, count=m, offset=off).copy()
        off += m * 8
        if m and (idx.min() < 0 or idx.max() >= count):
            raise ValueError('splat edits index out of range')
        pos = np.frombuffer(body, '<f4', count=m * 3, offset=off).reshape(m, 3).copy()
        off += m * 12
        quats = np.frombuffer(body, '<f4', count=m * 4, offset=off).reshape(m, 4).copy()
        off += m * 16
        sl = np.frombuffer(body, '<f4', count=m * 3, offset=off).reshape(m, 3).copy()
        ed._pending = (idx, pos, quats, sl)
        ed.dirty[idx] = True
        return ed

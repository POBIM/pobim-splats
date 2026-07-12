# Per-splat editing state (selection / hidden / deleted) with a tool-local
# undo stack. Mirrors the SuperSplat editor convention: a Uint8 array of bit
# flags plus an EditHistory of {indices, before, after} ops.
#
# This module is deliberately bpy-free (numpy only) so it can be unit tested
# outside Blender and reused for serialization to/from the .blend file.

import base64
import zlib

import numpy as np


class State:
    """Per-splat bit flags (mirrors the TS editor convention)."""
    SELECTED = 1
    HIDDEN = 2
    DELETED = 4


# uint8 masks that clear a single bit (~bit within a byte)
_CLR_SELECTED = np.uint8(0xFF ^ State.SELECTED)
_CLR_HIDDEN = np.uint8(0xFF ^ State.HIDDEN)


def serialize_rows(rows):
    """Serialize absolute source-file row indices (Duplicate/Separate subsets).

    Format mirrors the other payloads: an 8-byte little-endian uint64 count
    header (= number of indices) + zlib(level 6) of the int64 row bytes, then
    base64 ascii. bpy-free + unit-testable.
    """
    rows = np.asarray(rows, dtype=np.int64).ravel()
    body = np.ascontiguousarray(rows, '<i8').tobytes()
    payload = int(rows.size).to_bytes(8, 'little') + zlib.compress(body, 6)
    return base64.b64encode(payload).decode('ascii')


def deserialize_rows(s):
    """Restore serialized row indices as an int64 array.

    Raises ValueError on a corrupt payload or when the stored count does not
    match the decoded index count — callers drop the stale property and
    continue. An empty/None string yields an empty array.
    """
    if not s:
        return np.zeros(0, np.int64)
    try:
        payload = base64.b64decode(s)
    except Exception as e:
        raise ValueError(f'splat subset not valid base64: {e}')
    if len(payload) < 8:
        raise ValueError('splat subset payload truncated')
    stored = int.from_bytes(payload[:8], 'little')
    try:
        body = zlib.decompress(payload[8:])
    except Exception as e:
        raise ValueError(f'splat subset payload corrupt: {e}')
    rows = np.frombuffer(body, '<i8')
    if rows.size != stored:
        raise ValueError(
            f'splat subset count mismatch: header {stored}, decoded {rows.size}')
    return rows.copy()


class SplatState:
    """Editable per-splat state for one cloud.

    Every mutator bumps ``self.version`` (the GPU re-upload trigger) and
    returns the ``np.ndarray`` of CHANGED indices — the caller records these
    for undo. A no-op mutation bumps nothing and returns an empty array.
    """

    def __init__(self, count):
        self.flags = np.zeros(count, np.uint8)
        self.version = 0

    # -- internal -------------------------------------------------------
    def _commit(self, new_flags):
        """Adopt ``new_flags`` (full-length uint8), returning changed indices."""
        changed = np.nonzero(new_flags != self.flags)[0].astype(np.int64)
        if changed.size:
            self.flags[:] = new_flags
            self.version += 1
        return changed

    # -- selection ------------------------------------------------------
    def select_indices(self, indices, op='set'):
        """op: 'set' (replace), 'add', or 'remove'."""
        indices = np.asarray(indices, dtype=np.int64).ravel()
        new = self.flags.copy()
        if op == 'set':
            new &= _CLR_SELECTED
            if indices.size:
                new[indices] |= State.SELECTED
        elif op == 'add':
            if indices.size:
                new[indices] |= State.SELECTED
        elif op == 'remove':
            if indices.size:
                new[indices] &= _CLR_SELECTED
        else:
            raise ValueError(f'unknown select op: {op!r}')
        return self._commit(new)

    def select_all(self):
        """Select every non-deleted splat."""
        new = self.flags.copy()
        new[(new & State.DELETED) == 0] |= State.SELECTED
        return self._commit(new)

    def select_none(self):
        new = self.flags.copy()
        new &= _CLR_SELECTED
        return self._commit(new)

    def select_invert(self):
        """Invert selection among non-deleted splats."""
        new = self.flags.copy()
        mask = (new & State.DELETED) == 0
        new[mask] ^= State.SELECTED
        return self._commit(new)

    # -- visibility -----------------------------------------------------
    def hide_selected(self):
        new = self.flags.copy()
        new[(new & State.SELECTED) != 0] |= State.HIDDEN
        return self._commit(new)

    def unhide_all(self):
        new = self.flags.copy()
        new &= _CLR_HIDDEN
        return self._commit(new)

    # -- deletion -------------------------------------------------------
    def delete_selected(self):
        """Soft-delete selected splats and clear their SELECTED bit."""
        new = self.flags.copy()
        sel = (new & State.SELECTED) != 0
        new[sel] |= State.DELETED
        new[sel] &= _CLR_SELECTED
        return self._commit(new)

    # -- undo support ---------------------------------------------------
    def set_flags_raw(self, indices, values):
        """Overwrite flags at ``indices`` with ``values`` (undo restore)."""
        indices = np.asarray(indices, dtype=np.int64).ravel()
        values = np.asarray(values, dtype=np.uint8).ravel()
        new = self.flags.copy()
        if indices.size:
            new[indices] = values
        return self._commit(new)

    # -- queries --------------------------------------------------------
    @property
    def num_selected(self):
        return int(((self.flags & State.SELECTED) != 0).sum())

    @property
    def num_selected_exact(self):
        """Count of splats flagged EXACTLY State.SELECTED (no HIDDEN/DELETED).

        This is the Duplicate/Separate eligibility count: those operators
        mirror SuperSplat's strict ``state == State.selected`` filter, so a
        selected-but-hidden/deleted gaussian is not eligible to split off.
        Enable/label the split buttons on THIS, not ``num_selected`` (the
        SELECTED bit), so the buttons never enable on an all-hidden selection
        that the operator would then reject.
        """
        return int((self.flags == State.SELECTED).sum())

    @property
    def num_hidden(self):
        return int(((self.flags & State.HIDDEN) != 0).sum())

    @property
    def num_deleted(self):
        return int(((self.flags & State.DELETED) != 0).sum())

    def visible_mask(self):
        """Bool array: not hidden and not deleted."""
        return (self.flags & (State.HIDDEN | State.DELETED)) == 0

    def keep_mask(self):
        """Bool array of splats that survive export (not deleted)."""
        return (self.flags & State.DELETED) == 0

    # -- persistence ----------------------------------------------------
    def serialize(self):
        """base64(count header + zlib(packbits per flag)) for .blend
        persistence.

        The payload starts with the splat count as an 8-byte little-endian
        uint64 so ``deserialize`` can reject stale state after the cloud is
        re-imported with a different Max Splats or re-pointed to another
        file — silently decoding into garbage flags would make Export PLY
        drop the WRONG rows. Each of the three bit planes is packed
        independently, concatenated, then zlib-compressed (level 6: this
        runs on every committed edit).
        """
        parts = [np.packbits((self.flags & b) != 0)
                 for b in (State.SELECTED, State.HIDDEN, State.DELETED)]
        raw = np.concatenate(parts).astype(np.uint8).tobytes() if parts else b''
        payload = len(self.flags).to_bytes(8, 'little') + zlib.compress(raw, 6)
        return base64.b64encode(payload).decode('ascii')

    @staticmethod
    def deserialize(s, count):
        """Restore a serialized state. Raises ValueError when the stored
        count does not match ``count`` (stale state from a different import)
        or the payload is corrupt — callers must catch, drop the stale
        property and continue with fresh state."""
        st = SplatState(count)
        if not s:
            return st
        try:
            payload = base64.b64decode(s)
        except Exception as e:
            raise ValueError(f'splat state not valid base64: {e}')
        if len(payload) < 8:
            raise ValueError('splat state payload truncated')
        stored = int.from_bytes(payload[:8], 'little')
        if stored != count:
            raise ValueError(
                f'splat state count mismatch: stored {stored}, cloud has {count}')
        if count == 0:
            return st
        try:
            buf = np.frombuffer(zlib.decompress(payload[8:]), np.uint8)
        except Exception as e:
            raise ValueError(f'splat state payload corrupt: {e}')
        per = (count + 7) // 8
        if buf.size < 3 * per:
            raise ValueError('splat state payload too short for count')
        flags = np.zeros(count, np.uint8)
        for i, b in enumerate((State.SELECTED, State.HIDDEN, State.DELETED)):
            plane = buf[i * per:(i + 1) * per]
            bits = np.unpackbits(plane)[:count].astype(bool)
            flags[bits] |= b
        st.flags = flags
        return st


class EditHistory:
    """Tool-local undo/redo (Blender's global undo can't see numpy state).

    Each op is a dict with a ``kind`` selecting how undo/redo applies it:

    - ``'flags'`` (the default when absent): a selection/visibility/delete edit
      of the form {'label', 'indices', 'before' u8, 'after' u8}. Undo/redo
      restore directly via ``SplatState.set_flags_raw`` — the legacy behavior.
    - ``'transform'`` (Phase 3): a geometry edit whose 'before'/'after' are
      SplatEdits payloads. EditHistory does NOT touch these (it has no
      SplatEdits handle): undo/redo return ``(direction, op)`` and the CALLER
      applies the restore via ``SplatEdits.restore`` + a GPU update.

    ``undo``/``redo`` always return ``(direction, op)`` (or ``None`` when the
    stack end is reached); flags callers may ignore the return value, which
    keeps the existing API backward compatible.

    The stack is bounded (``max_ops``, ring behavior): a select-all op on a
    multi-million cloud stores three full-length arrays, so an unbounded
    history would grow by ~tens of MB per op.
    """

    def __init__(self, max_ops=64):
        self.max_ops = max(1, int(max_ops))
        self.ops = []
        self.cursor = 0   # number of currently-applied ops

    def push(self, op):
        # a new op invalidates any redo tail
        del self.ops[self.cursor:]
        self.ops.append(op)
        self.cursor += 1
        # ring behavior: drop the oldest entries beyond the cap; those edits
        # simply become non-undoable
        overflow = len(self.ops) - self.max_ops
        if overflow > 0:
            del self.ops[:overflow]
            self.cursor -= overflow

    @property
    def can_undo(self):
        return self.cursor > 0

    @property
    def can_redo(self):
        return self.cursor < len(self.ops)

    def undo(self, state):
        if not self.can_undo:
            return None
        self.cursor -= 1
        op = self.ops[self.cursor]
        if op.get('kind', 'flags') == 'flags':
            state.set_flags_raw(op['indices'], op['before'])
        return ('undo', op)

    def redo(self, state):
        if not self.can_redo:
            return None
        op = self.ops[self.cursor]
        self.cursor += 1
        if op.get('kind', 'flags') == 'flags':
            state.set_flags_raw(op['indices'], op['after'])
        return ('redo', op)

    def clear(self):
        self.ops = []
        self.cursor = 0

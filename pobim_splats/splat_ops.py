# Duplicate / Separate selection operators (Phase 4, Track F).
#
# Mirrors SuperSplat/POBIMStudio `select.duplicate` / `select.separate`:
# a strictly-SELECTED subset of gaussians is split off into a NEW splat empty
# that references the SAME source .ply but persists only the chosen absolute
# source rows (obj['pobim_splat_subset']). Separate additionally soft-deletes
# those rows on the source (keeping the SELECTED bit — deleted rows end up
# selected|deleted, TS parity).
#
# The subset object starts with a FRESH zero flag state; only intersecting
# TRANSFORM edits are carried over (re-indexed into the subset's local rows).

import uuid

import bpy
import numpy as np

from . import splat_gpu
from .splat_state import State, serialize_rows


def selected_exact_rows(state):
    """Rows whose flags are EXACTLY State.SELECTED (no HIDDEN/DELETED bits).

    Mirrors SuperSplat's `state == State.selected`: a gaussian that is also
    hidden or deleted is excluded from Duplicate/Separate.
    """
    if state is None:
        return np.zeros(0, np.int64)
    return np.nonzero(state.flags == State.SELECTED)[0].astype(np.int64)


def _source_rows(entry, sel_rows):
    """Map LOADED-cloud local rows -> absolute source-file rows via the entry's
    kept ``source_indices`` (survives ensure_gpu array freeing)."""
    cloud = getattr(entry, 'cloud', None)
    src = getattr(cloud, 'source_indices', None) if cloud is not None else None
    if src is None:
        return np.ascontiguousarray(sel_rows.astype(np.int64))
    return np.ascontiguousarray(np.asarray(src, np.int64)[sel_rows])


def _carry_edits_payload(entry, sel_rows, subset_count):
    """Serialized SplatEdits payload (count == subset_count) for the source
    transform edits whose rows intersect ``sel_rows``, re-indexed into the
    subset's local ordering, or None when there is nothing to carry.

    ``sel_rows`` is sorted ascending; the subset's local row k corresponds to
    the source's sel_rows[k], so a searchsorted lookup gives the remap.
    """
    edits = getattr(entry, 'edits', None)
    if edits is None:
        return None
    try:
        payload = edits.export_payload()
    except Exception as e:
        print(f'[pobim_splats] could not read source edits for carry-over: {e}')
        return None
    if not payload:
        return None
    eidx = np.asarray(payload['indices'], np.int64).ravel()
    if eidx.size == 0 or sel_rows.size == 0:
        return None

    pos = np.searchsorted(sel_rows, eidx)
    inb = pos < sel_rows.size
    match = np.zeros(eidx.size, bool)
    match[inb] = sel_rows[pos[inb]] == eidx[inb]
    if not match.any():
        return None

    new_local = pos[match].astype(np.int64)
    try:
        from .splat_edits import SplatEdits  # lazy: bpy-free, may lag
    except Exception:
        return None
    ed = SplatEdits(int(subset_count))
    ed._pending = (
        new_local,
        np.ascontiguousarray(payload['positions'][match], np.float32),
        np.ascontiguousarray(payload['quats'][match], np.float32),
        np.ascontiguousarray(payload['scales_log'][match], np.float32))
    ed.dirty[new_local] = True
    return ed.serialize()


def _make_subset_object(context, src, name, subset_payload, edits_payload):
    """Create a new splat empty mirroring ``src``'s import props, positioned
    identically (same parent + matrices), carrying the subset/edits payloads."""
    obj = bpy.data.objects.new(name, None)
    obj.empty_display_type = 'PLAIN_AXES'
    obj.empty_display_size = 0.5
    obj.pobim_splat_uid = uuid.uuid4().hex
    obj.pobim_splat_file = src.pobim_splat_file
    obj.pobim_splat_max = 0                       # subset loads full then slices
    obj.pobim_splat_srgb = src.pobim_splat_srgb
    obj.pobim_splat_shmax = src.pobim_splat_shmax
    # display props (do not carry per-gaussian flag state — fresh copy)
    obj.pobim_splat_sh_view = src.pobim_splat_sh_view
    obj.pobim_splat_scale = src.pobim_splat_scale
    obj.pobim_splat_opacity = src.pobim_splat_opacity

    obj['pobim_splat_subset'] = subset_payload
    if edits_payload is not None:
        obj['pobim_splat_edits'] = edits_payload

    # link into the source's collections (fall back to the active collection)
    cols = list(getattr(src, 'users_collection', []) or [])
    if not cols:
        cols = [context.collection]
    for col in cols:
        try:
            col.objects.link(obj)
        except Exception:
            pass

    # replicate the source transform exactly (single-source case)
    obj.parent = src.parent
    obj.matrix_parent_inverse = src.matrix_parent_inverse.copy()
    obj.matrix_basis = src.matrix_basis.copy()
    return obj


def perform_split(context, src, entry, separate, report=None):
    """Core Duplicate/Separate. Returns (new_obj, changed_indices) or
    (None, None) when there is nothing selected.

    ``changed_indices`` is the source rows that Separate soft-deleted (for the
    modal to record an undo op); None for Duplicate.
    """
    sel_rows = selected_exact_rows(getattr(entry, 'state', None))
    if sel_rows.size == 0:
        if report:
            report({'ERROR'}, 'ไม่มี splat ที่เลือกไว้ (เลือกก่อนแล้วลองใหม่)')
        return None, None

    rows_abs = _source_rows(entry, sel_rows)
    subset_payload = serialize_rows(rows_abs)
    edits_payload = _carry_edits_payload(entry, sel_rows, sel_rows.size)

    new_obj = _make_subset_object(
        context, src, f'{src.name} Selection', subset_payload, edits_payload)

    # load the subset immediately so it draws this session
    try:
        splat_gpu.load_entry_for_object(new_obj)
    except Exception as e:
        bpy.data.objects.remove(new_obj)
        if report:
            report({'ERROR'}, f'สร้าง subset ไม่สำเร็จ: {e}')
        return None, None

    changed = None
    if separate:
        state = entry.state
        # keep the SELECTED bit; add DELETED (selected|deleted, TS parity)
        vals = (state.flags[sel_rows] | State.DELETED).astype(np.uint8)
        changed = state.set_flags_raw(sel_rows, vals)
        # persist + cache the payload so an undo of this Separate re-syncs the
        # source's in-memory state from the reverted property (resync_states).
        splat_gpu.persist_state(src, entry)

    # select + activate the new object, deselect the source
    for other in list(context.selected_objects):
        try:
            other.select_set(False)
        except Exception:
            pass
    try:
        src.select_set(False)
        new_obj.select_set(True)
        context.view_layer.objects.active = new_obj
    except Exception:
        pass

    splat_gpu.purge_orphans()
    splat_gpu.redraw_viewports()
    n = new_obj.pobim_splat_count
    if report:
        verb = 'แยก' if separate else 'ทำสำเนา'
        report({'INFO'}, f'{verb} {n:,} splats → {new_obj.name}')
    return new_obj, changed


class _SplitBase(bpy.types.Operator):
    bl_options = {'REGISTER', 'UNDO'}
    uid: bpy.props.StringProperty()

    _separate = False

    def execute(self, context):
        src = None
        if self.uid:
            for obj in bpy.data.objects:
                if obj.pobim_splat_uid == self.uid:
                    src = obj
                    break
        else:
            obj = context.active_object
            if obj is not None and getattr(obj, 'pobim_splat_uid', ''):
                src = obj
        if src is None:
            self.report({'ERROR'}, 'ไม่พบ object ของ splat')
            return {'CANCELLED'}
        entry = splat_gpu.REGISTRY.get(src.pobim_splat_uid)
        if entry is None:
            self.report({'ERROR'}, 'splat ยังไม่ได้โหลด — กด Reload ก่อน')
            return {'CANCELLED'}
        new_obj, _changed = perform_split(
            context, src, entry, self._separate, self.report)
        if new_obj is None:
            return {'CANCELLED'}
        return {'FINISHED'}


class POBIM_OT_duplicate_selection(_SplitBase):
    """Duplicate the selected splats into a new object (source unchanged)"""
    bl_idname = 'pobim_splats.duplicate_selection'
    bl_label = 'Duplicate Selection'
    _separate = False


class POBIM_OT_separate_selection(_SplitBase):
    """Separate the selected splats into a new object, deleting them here"""
    bl_idname = 'pobim_splats.separate_selection'
    bl_label = 'Separate Selection'
    _separate = True


CLASSES = (POBIM_OT_duplicate_selection, POBIM_OT_separate_selection)

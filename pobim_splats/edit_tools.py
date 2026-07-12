# Per-splat editing: a modal selection tool with a tool-local undo stack,
# plus a .ply export of the surviving splats. Mirrors POBIMStudio's editing
# model (State bit flags + EditHistory) and copies measure.py's modal
# lifecycle (running guard, cancel(), status text, POST_PIXEL overlay,
# PASS_THROUGH viewport navigation).
#
# Phase 1: rect select. Phase 2 adds SuperSplat-style tools — Lasso (L),
# Polygon (P), Brush (B), Sphere (S) and Box (C). Rect (R) stays the
# default. Every selection tool commits through the same _apply path: ONE
# EditHistory op per commit, state persisted, viewport redrawn.
#
# Phase 3 (Track U): an in-viewport clickable HUD chip toolbar, managed
# radius (scene props + F interactive resize + Alt+Wheel + drag slider),
# a two-stage Box/Sphere "preview" gesture with grabbable corners, and
# Move/Rotate/Scale transform modes that drive Track T's transform core
# (splat_edits.SplatEdits + splat_gpu preview/commit). Track T is imported
# defensively — the modal still runs (selection + HUD) if it lags, with
# the transform modes gracefully no-op'ing.

import bpy
import blf
import gpu
import numpy as np
from bpy.props import StringProperty
from bpy_extras.io_utils import ExportHelper
from gpu_extras.batch import batch_for_shader

from . import ply_loader, splat_export, splat_gpu
from .measure_math import (
    BOX_EDGES as _BOX_EDGES, box_corners, pick_nearest, project_to_pixels,
    unproject_pixel)
from .splat_state import EditHistory, SplatState, State

# select_math is written in parallel; import defensively so a missing module
# still lets the addon register (only the commit paths would then no-op).
try:
    from . import select_math as _select_math
except Exception as _e:  # pragma: no cover - only when the sibling module lags
    print(f'[pobim_splats] select_math unavailable: {_e}')
    _select_math = None

# Track T's transform core — imported defensively (same pattern as
# select_math). When absent, transform MODES enter cleanly but no-op on
# confirm and draw no GPU preview; selection + HUD + radius stay fully live.
try:
    from . import splat_edits as _splat_edits
except Exception as _e:  # pragma: no cover - only when Track T lags
    print(f'[pobim_splats] splat_edits unavailable: {_e}')
    _splat_edits = None
try:
    from . import transform_math as _transform_math
except Exception as _e:  # pragma: no cover - only when Track T lags
    print(f'[pobim_splats] transform_math unavailable: {_e}')
    _transform_math = None

_STATE_PROP = 'pobim_splat_state'
_EDIT_PROP = 'pobim_splat_edits'

# rect overlay palette (POBIMStudio rect-select: dark under-stroke + white line)
_WHITE = (1.0, 1.0, 1.0, 1.0)
_DARK = (0.0, 0.0, 0.0, 0.8)
_ORANGE = (1.0, 0.647, 0.0, 1.0)          # #ffa500 POBIM active

# HUD chip palette
_CHIP_BG = (0.10, 0.10, 0.10, 0.86)
_CHIP_BG_HOVER = (0.24, 0.24, 0.24, 0.92)
_CHIP_BG_ACTIVE = (1.0, 0.647, 0.0, 0.95)
_CHIP_TEXT = (1.0, 1.0, 1.0, 1.0)
_CHIP_TEXT_ACTIVE = (0.08, 0.08, 0.08, 1.0)
_CHIP_TEXT_DISABLED = (0.5, 0.5, 0.5, 1.0)

# project at most this many rows per matmul (bounds the temporary world buffer)
_PROJ_CHUNK = 1_000_000

# subsample used ONLY for the nearest-center pick fallback (selection commits
# always run on the full cloud); mirrors measure.py's PICK_SUBSAMPLE.
PICK_SUBSAMPLE = 400_000
PICK_RADIUS = 25.0

_BRUSH_DEFAULT = 40.0     # px
_BRUSH_MIN = 4.0
_BRUSH_MAX = 400.0
_SPHERE_DEFAULT = 0.25    # world units
_SPHERE_MIN = 1e-3
_SPHERE_MAX = 100.0
_POLY_CLOSE_PX = 8.0      # click within this of the first vertex closes
_LASSO_MIN_SPACING = 4.0  # px between appended lasso points
_LASSO_CAP = 512          # decimate the stroke beyond this many points
_CORNER_HIT_PX = 12.0     # grab radius for box/sphere preview handles

_TOOL_ITEMS = ('RECT', 'LASSO', 'POLYGON', 'BRUSH', 'SPHERE', 'BOX')
_TOOL_KEYS = {'R': 'RECT', 'L': 'LASSO', 'P': 'POLYGON',
              'B': 'BRUSH', 'S': 'SPHERE', 'C': 'BOX'}
_XFORM_ITEMS = ('MOVE', 'ROTATE', 'SCALE')
# 1/2/3 (TS convention) + G alias for Move
_XFORM_KEYS = {'ONE': 'MOVE', 'TWO': 'ROTATE', 'THREE': 'SCALE', 'G': 'MOVE'}
_TOOL_LABELS = {
    'RECT': 'กรอบ Rect', 'LASSO': 'บ่วง Lasso', 'POLYGON': 'หลายเหลี่ยม Polygon',
    'BRUSH': 'พู่กัน Brush', 'SPHERE': 'ทรงกลม Sphere', 'BOX': 'กล่อง Box',
    'MOVE': 'ย้าย Move', 'ROTATE': 'หมุน Rotate', 'SCALE': 'ย่อ/ขยาย Scale'}
_TOOL_HINTS = {
    'RECT': 'ลาก=เลือกกรอบ',
    'LASSO': 'ลากวาดเส้น=เลือก',
    'POLYGON': 'คลิกเพิ่มจุด · Enter/คลิกจุดแรก=ปิด · BkSp=ลบจุด',
    'BRUSH': 'ลากทาสี · F/[ ]/Alt+ล้อ ปรับขนาด',
    'SPHERE': 'คลิกวาง · ลากจุดกลาง · F/[ ] รัศมี · Enter=ยืนยัน',
    'BOX': 'คลิกสองมุม · ลากจุดมุมปรับ · Enter=ยืนยัน · Esc=ยกเลิก',
    'MOVE': 'ลาก=ย้าย · X/Y/Z ล็อกแกน · Shift ละเอียด · Enter/ปล่อย=ยืนยัน',
    'ROTATE': 'ลาก=หมุนรอบแกนมอง · X/Y/Z ล็อกแกน · Enter/ปล่อย=ยืนยัน',
    'SCALE': 'ลาก=ย่อ/ขยาย · X/Y/Z ต่อแกน · Enter/ปล่อย=ยืนยัน'}

# --- HUD chip toolbar layout ------------------------------------------------
_HUD_SEP = ('SEP', None)
_HUD_ROW1 = [
    ('tool:RECT', 'Rect'), ('tool:LASSO', 'Lasso'), ('tool:POLYGON', 'Poly'),
    ('tool:BRUSH', 'Brush'), ('tool:SPHERE', 'Sphere'), ('tool:BOX', 'Box'),
    _HUD_SEP,
    ('xform:MOVE', 'Move'), ('xform:ROTATE', 'Rotate'), ('xform:SCALE', 'Scale'),
    _HUD_SEP,
    ('undo', 'Undo'), ('redo', 'Redo'),
    _HUD_SEP,
    ('done', 'Done'),
]


def _find_splat_object(context, uid):
    if uid:
        for obj in bpy.data.objects:
            if obj.pobim_splat_uid == uid:
                return obj
        return None
    obj = context.active_object
    if obj is not None and getattr(obj, 'pobim_splat_uid', ''):
        return obj
    splats = [o for o in context.scene.objects if o.pobim_splat_uid]
    return splats[0] if len(splats) == 1 else None


def _circle(cx, cy, r, segments=32):
    t = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=True)
    return [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in t]


# --- transform matrix helpers (numpy, self-contained) -----------------------
# Kept local so the transform PREVIEW gesture works even if transform_math
# lags. The commit path (SplatEdits.apply_matrix) is Track T's; it receives
# the same 4x4 LOCAL matrix built here.

def _rot_axis_angle(axis, angle):
    a = np.asarray(axis, np.float64)
    n = np.linalg.norm(a)
    if n < 1e-12:
        return np.eye(3)
    a = a / n
    c, s = np.cos(angle), np.sin(angle)
    x, y, z = a
    C = 1.0 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C]], np.float64)


def _about_pivot(rot3, pivot):
    """4x4 LOCAL matrix applying 3x3 ``rot3`` about ``pivot`` (local space)."""
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = rot3
    m[:3, 3] = np.asarray(pivot, np.float64) - rot3 @ np.asarray(pivot, np.float64)
    return m


def _translate4(t):
    m = np.eye(4, dtype=np.float64)
    m[:3, 3] = np.asarray(t, np.float64)
    return m


class POBIM_OT_edit_splats(bpy.types.Operator):
    """แก้ไข splat: เลือก / ซ่อน / ลบ / ย้าย-หมุน-ย่อขยาย ทีละจุด
(R กรอบ | L บ่วง | P หลายเหลี่ยม | B พู่กัน | S ทรงกลม | C กล่อง | 1/2/3/G ย้าย/หมุน/สเกล | F ปรับรัศมีสด | Shift = เพิ่ม | Ctrl = ลบออก | A = เลือกทั้งหมด | Shift+A/Alt+A = ไม่เลือก | Ctrl+I = สลับเลือก | H = ซ่อน | Alt+H = คืน | X/Del = ลบ | Ctrl+Z = undo | Ctrl+Shift+Z = redo | Esc/คลิกขวา = ออก)"""
    bl_idname = 'pobim_splats.edit_splats'
    bl_label = 'Edit Splats'

    uid: StringProperty()

    _running = False

    # --- lifecycle --------------------------------------------------------

    def invoke(self, context, event):
        if POBIM_OT_edit_splats._running:
            self.report({'WARNING'}, 'เครื่องมือแก้ไขกำลังทำงานอยู่แล้ว')
            return {'CANCELLED'}
        if context.area is None or context.area.type != 'VIEW_3D':
            self.report({'ERROR'}, 'ต้องเรียกจาก 3D Viewport')
            return {'CANCELLED'}

        obj = _find_splat_object(context, self.uid)
        if obj is None:
            self.report({'ERROR'}, 'เลือก splat ก่อน (คลิกที่ Empty ของ splat)')
            return {'CANCELLED'}
        entry = splat_gpu.REGISTRY.get(obj.pobim_splat_uid)
        positions = None
        if entry is not None:
            if entry.gpu is not None:
                positions = entry.gpu.positions
            elif entry.cloud is not None:
                positions = entry.cloud.positions
        if positions is None:
            self.report({'ERROR'}, 'splat ยังไม่ได้โหลด — กด Reload ก่อน')
            return {'CANCELLED'}

        count = positions.shape[0]
        # lazily create the edit state; restore a serialized one when counts
        # match — on mismatch/corruption drop the stale property (decoding it
        # into garbage flags would make Export PLY drop the wrong rows)
        if entry.state is None:
            s = obj.get(_STATE_PROP)
            if s:
                try:
                    entry.state = SplatState.deserialize(s, count)
                except Exception as e:
                    print(f'[pobim_splats] discarding stale edit state: {e}')
                    try:
                        del obj[_STATE_PROP]
                    except Exception:
                        pass
                    entry.state = SplatState(count)
            else:
                entry.state = SplatState(count)

        self._entry = entry
        self._state = entry.state
        self._history = EditHistory()
        self._obj_name = obj.name
        self._local = positions
        self._count = count

        # geometry-edit overrides (Track T). Reuse a live entry.edits if
        # present; else restore from the object property (count-guarded like
        # the flag state) or start fresh. Absent Track T -> stays None and
        # every transform path no-ops.
        self._edits = self._restore_edits(obj, entry, count)

        # subsample for the nearest-center pick fallback only. The permutation
        # and the gpu geometry_version are kept so the subsampled COPY can be
        # refreshed after update_splats (transform commit / undo / redo)
        # rewrites positions in place — see _refresh_pick_local.
        if count > PICK_SUBSAMPLE:
            self._pick_sel = np.random.default_rng(0).permutation(count)[:PICK_SUBSAMPLE]
            self._pick_local = positions[self._pick_sel]
        else:
            self._pick_sel = None
            self._pick_local = positions
        self._pick_geom_version = getattr(entry.gpu, 'geometry_version', 0) \
            if entry.gpu is not None else 0
        self._world = None
        self._world_matrix = None

        # active selection tool + optional transform mode (mutually exclusive)
        self._tool = getattr(context.scene, 'pobim_splat_edit_tool', 'RECT')
        self._xform = None            # None | 'MOVE' | 'ROTATE' | 'SCALE'
        self._mouse = (event.mouse_region_x, event.mouse_region_y)
        self._shift_state = bool(event.shift)
        # radii persist in scene props (read on invoke, written on change)
        self._brush_radius = float(np.clip(
            getattr(context.scene, 'pobim_splat_brush_radius', _BRUSH_DEFAULT),
            _BRUSH_MIN, _BRUSH_MAX))
        self._sphere_radius = float(np.clip(
            getattr(context.scene, 'pobim_splat_sphere_radius', _SPHERE_DEFAULT),
            _SPHERE_MIN, _SPHERE_MAX))

        # HUD state
        self._hud_chips = []          # cached rects for hover draw + hit-test
        self._hud_hover = None        # chip id under the cursor
        self._hud_drag = None         # ('radius', start_x, start_value)

        # radius interactive-resize (F) sub-mode
        self._resizing = False
        self._resize_before = None
        self._resize_anchor = None

        # RECT
        self._dragging = False
        self._drag_start = None
        self._drag_end = None
        # LASSO
        self._lassoing = False
        self._lasso_pts = []
        self._lasso_spacing = _LASSO_MIN_SPACING
        # POLYGON
        self._poly_pts = []
        # BRUSH
        self._painting = False
        self._brush_op = 'add'
        self._brush_snapshot = None
        self._stroke_px = None
        self._stroke_valid = None
        self._stroke_persp = None
        self._last_paint_pt = None
        # SPHERE (v2: place -> preview -> confirm)
        self._sphere_center = None    # world
        self._sphere_stage = None     # None | 'PREVIEW'
        self._sphere_grab = False
        # BOX (corners kept in splat-LOCAL space; v2 preview stage)
        self._box_c1 = None
        self._box_c2 = None
        self._box_hover = None        # world hover before corner2
        self._box_stage = None        # None | 'CORNER2' | 'PREVIEW'
        self._box_grab = None         # 0 or 1 while dragging a corner
        # TRANSFORM gesture
        self._xform_sel = None
        self._xform_centroid = None   # local space
        self._xform_dragging = False
        self._xform_start = None
        self._xform_axis = None       # None | 0 | 1 | 2 (local axis lock)
        self._xform_matrix = None     # current preview 4x4 (local)
        # depth-pick offscreen (owned here; freed in _finish)
        self._depth_offs = None
        self._depth_persp = None

        self._shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        self._handle = bpy.types.SpaceView3D.draw_handler_add(
            self._draw_overlay, (context,), 'WINDOW', 'POST_PIXEL')
        POBIM_OT_edit_splats._running = True
        self._set_status(context)
        context.window_manager.modal_handler_add(self)
        context.area.tag_redraw()
        return {'RUNNING_MODAL'}

    def _restore_edits(self, obj, entry, count):
        """Reuse/restore/create the SplatEdits override store (Track T)."""
        edits = getattr(entry, 'edits', None)
        if edits is not None:
            return edits
        if _splat_edits is None:
            return None
        s = obj.get(_EDIT_PROP)
        if s:
            try:
                edits = _splat_edits.SplatEdits.deserialize(s, count)
            except Exception as e:
                print(f'[pobim_splats] discarding stale edit overrides: {e}')
                try:
                    del obj[_EDIT_PROP]
                except Exception:
                    pass
                edits = None
        if edits is None:
            try:
                edits = _splat_edits.SplatEdits(count)
            except Exception as e:
                print(f'[pobim_splats] SplatEdits unavailable: {e}')
                return None
        try:
            entry.edits = edits
        except Exception:
            pass
        return edits

    def _active(self):
        """Identifier of the current tool/mode ('RECT'.. or 'MOVE'..)."""
        return self._xform if self._xform is not None else self._tool

    def _set_status(self, context):
        n_sel = self._state.num_selected
        n_hid = self._state.num_hidden
        n_del = self._state.num_deleted
        count = self._count
        active = self._active()
        tool = _TOOL_LABELS.get(active, active)
        hint = _TOOL_HINTS.get(active, '')
        if active in {'BRUSH', 'SPHERE'}:
            r = (f'{self._brush_radius:.0f} px' if active == 'BRUSH'
                 else f'{self._sphere_radius:.3f} m')
            hint = f'รัศมี {r} · {hint}'
        if self._resizing:
            hint = 'กำลังปรับรัศมี — LMB/Enter ยืนยัน · Esc/RMB ยกเลิก'
        context.workspace.status_text_set(
            f'Edit Splats [{tool}] — Selected {n_sel:,} / {count:,} · '
            f'Hidden {n_hid:,} · Deleted {n_del:,} | {hint} | '
            f'R/L/P/B/S/C เครื่องมือ · 1/2/3/G ย้าย/หมุน/สเกล · Shift เพิ่ม Ctrl ลบออก | '
            f'A/Shift+A/Ctrl+I | H ซ่อน Alt+H คืน | X ลบ | Ctrl+Z undo | Esc ออก')

    def _finish(self, context):
        POBIM_OT_edit_splats._running = False
        # drop any live transform preview so the GPU doesn't keep drawing it
        self._clear_preview()
        if getattr(self, '_handle', None) is not None:
            bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
            self._handle = None
        if getattr(self, '_depth_offs', None) is not None:
            self._depth_offs.free()
            self._depth_offs = None
        context.workspace.status_text_set(None)
        if context.area:
            context.area.tag_redraw()

    def cancel(self, context):
        self._finish(context)

    # --- editing helpers --------------------------------------------------

    def _persist(self, context):
        """Serialize flag state to the object and refresh status + viewport."""
        obj = bpy.data.objects.get(self._obj_name)
        if obj is not None:
            obj[_STATE_PROP] = self._state.serialize()
        self._set_status(context)
        splat_gpu.redraw_viewports()
        if context.area:
            context.area.tag_redraw()

    def _apply(self, context, label, mutator):
        """Run a state mutator, record an undo op, persist and redraw."""
        snapshot = self._state.flags.copy()      # before-flags of the whole cloud
        changed = mutator()
        if changed is None or changed.size == 0:
            self._set_status(context)
            if context.area:
                context.area.tag_redraw()
            return
        self._history.push({
            'label': label,
            'indices': changed,
            'before': snapshot[changed].copy(),
            'after': self._state.flags[changed].copy(),
        })
        self._persist(context)

    def _eligible_mask(self):
        """Splats that a selection may touch: not hidden and not deleted."""
        return (self._state.flags & (State.HIDDEN | State.DELETED)) == 0

    @staticmethod
    def _op_from_event(event):
        return 'add' if event.shift else ('remove' if event.ctrl else 'set')

    def _project_all(self, context):
        """Project every splat center to region pixels (chunked matmul)."""
        obj = bpy.data.objects.get(self._obj_name)
        rv3d = context.region_data
        region = context.region
        if obj is None or rv3d is None or region is None:
            return None, None
        m = np.array(obj.matrix_world, np.float64)
        rot = m[:3, :3].T
        trans = m[:3, 3]
        persp = np.array(rv3d.perspective_matrix, np.float32)
        local = self._local
        n = local.shape[0]
        px = np.empty((n, 2), np.float32)
        valid = np.empty(n, bool)
        for i in range(0, n, _PROJ_CHUNK):
            j = min(i + _PROJ_CHUNK, n)
            world = (local[i:j].astype(np.float64) @ rot + trans).astype(np.float32)
            p, _z, v = project_to_pixels(persp, world, region.width, region.height)
            px[i:j] = p
            valid[i:j] = v
        return px, valid

    def _sphere_mask(self, context, center, radius):
        """Chunked points_in_sphere over the full cloud transformed to world."""
        if _select_math is None:
            return None
        obj = bpy.data.objects.get(self._obj_name)
        if obj is None:
            return None
        m = np.array(obj.matrix_world, np.float64)
        rot = m[:3, :3].T
        trans = m[:3, 3]
        local = self._local
        n = local.shape[0]
        mask = np.empty(n, bool)
        for i in range(0, n, _PROJ_CHUNK):
            j = min(i + _PROJ_CHUNK, n)
            world = (local[i:j].astype(np.float64) @ rot + trans).astype(np.float32)
            mask[i:j] = _select_math.points_in_sphere(world, center, radius)
        return mask

    # --- transforms / picking --------------------------------------------

    def _matrix3(self):
        obj = bpy.data.objects.get(self._obj_name)
        if obj is None:
            return np.eye(3), np.eye(3)
        m = np.array(obj.matrix_world, np.float64)
        inv = np.linalg.inv(m)
        return m[:3, :3], inv[:3, :3]

    def _to_local(self, world):
        obj = bpy.data.objects.get(self._obj_name)
        if obj is None:
            return np.asarray(world, np.float32)
        m = np.linalg.inv(np.array(obj.matrix_world, np.float64))
        return (m[:3, :3] @ np.asarray(world, np.float64) + m[:3, 3]).astype(np.float32)

    def _to_world(self, local):
        obj = bpy.data.objects.get(self._obj_name)
        if obj is None:
            return np.asarray(local, np.float32)
        m = np.array(obj.matrix_world, np.float64)
        return (m[:3, :3] @ np.asarray(local, np.float64) + m[:3, 3]).astype(np.float32)

    def _refresh_pick_local(self):
        """Rebuild the pick subsample when geometry changed on the GPU.

        update_splats (transform commit / undo / redo) rewrites self._local in
        place and bumps gpu.geometry_version. The full-array _pick_local IS
        self._local (already fresh), but the subsampled copy and the cached
        _world points go stale — refresh both when the version moved."""
        gpu_obj = getattr(self._entry, 'gpu', None)
        ver = getattr(gpu_obj, 'geometry_version', 0) if gpu_obj is not None else 0
        if ver == self._pick_geom_version:
            return
        self._pick_geom_version = ver
        if self._pick_sel is not None:
            self._pick_local = self._local[self._pick_sel]
        self._world = None   # cached world points are stale either way

    def _world_points(self, obj):
        self._refresh_pick_local()
        matrix = np.array(obj.matrix_world, np.float32)
        if self._world is None or not np.array_equal(matrix, self._world_matrix):
            self._world_matrix = matrix
            self._world = self._pick_local @ matrix[:3, :3].T + matrix[:3, 3]
        return self._world

    def _ensure_depth_map(self, context):
        rv3d = context.region_data
        region = context.region
        persp = np.array(rv3d.perspective_matrix, np.float32)
        if (self._depth_offs is not None and self._depth_persp is not None and
                np.array_equal(persp, self._depth_persp)):
            return
        if self._depth_offs is not None:
            self._depth_offs.free()
            self._depth_offs = None
        view = np.array(rv3d.view_matrix, np.float32)
        proj = np.array(rv3d.window_matrix, np.float32)
        try:
            self._depth_offs = splat_gpu.render_depth_map(
                view, proj, region.width, region.height)
            self._depth_persp = persp
        except Exception as e:
            print(f'[pobim_splats] depth pick render failed: {e}')
            self._depth_offs = None

    def _pick_world(self, context, mx, my):
        """Surface-pick a world point under (mx, my); nearest-center fallback."""
        obj = bpy.data.objects.get(self._obj_name)
        rv3d = context.region_data
        region = context.region
        if obj is None or rv3d is None or region is None:
            return None
        persp = np.array(rv3d.perspective_matrix, np.float32)

        self._ensure_depth_map(context)
        depth = splat_gpu.read_depth_pixel(self._depth_offs, mx, my)
        if depth is not None:
            return unproject_pixel(persp, mx, my, depth, region.width, region.height)

        world = self._world_points(obj)
        idx = pick_nearest(persp, world, region.width, region.height,
                           (mx, my), PICK_RADIUS)
        return None if idx < 0 else world[idx].copy()

    def _sphere_pixel_radius(self, context, center, radius):
        """Screen radius (px) of a world sphere centred at ``center``."""
        rv3d = context.region_data
        region = context.region
        if rv3d is None or region is None:
            return None
        proj = np.array(rv3d.window_matrix, np.float64)
        focal = proj[1, 1] * 0.5 * region.height
        if rv3d.is_perspective:
            view = np.array(rv3d.view_matrix, np.float64)
            c = view[:3, :3] @ np.asarray(center, np.float64) + view[:3, 3]
            depth = -c[2]
            if depth <= 1e-6:
                return None
            return float(radius * focal / depth)
        return float(radius * focal)

    def _world_per_pixel(self, context, world_pt):
        """World-space size of one pixel at ``world_pt`` (inverse of above)."""
        ppw = self._sphere_pixel_radius(context, world_pt, 1.0)
        if ppw is None or ppw < 1e-9:
            return None
        return 1.0 / ppw

    # --- gesture bookkeeping ---------------------------------------------

    def _reset_gesture(self):
        """Discard any in-progress gesture (switch/cancel). A live brush
        stroke restores the pre-stroke flags so nothing is committed."""
        if self._painting and self._brush_snapshot is not None:
            if not np.array_equal(self._state.flags, self._brush_snapshot):
                self._state.flags[:] = self._brush_snapshot
                self._state.version += 1
                # parity with _persist: other viewports show the live-painted
                # flags and would stay stale without a full redraw
                splat_gpu.redraw_viewports()
        self._dragging = False
        self._drag_start = self._drag_end = None
        self._lassoing = False
        self._lasso_pts = []
        self._lasso_spacing = _LASSO_MIN_SPACING
        self._poly_pts = []
        self._painting = False
        self._brush_snapshot = None
        self._stroke_px = self._stroke_valid = None
        self._stroke_persp = None
        self._last_paint_pt = None
        self._box_c1 = self._box_c2 = None
        self._box_hover = None
        self._box_stage = None
        self._box_grab = None
        self._sphere_center = None
        self._sphere_stage = None
        self._sphere_grab = False
        self._resizing = False
        self._cancel_transform()

    def _switch_tool(self, context, tool):
        if tool == self._tool and self._xform is None:
            return
        self._reset_gesture()
        # only SPHERE/BOX surface-pick: release the depth offscreen early
        # (polish — _finish frees it anyway)
        if self._tool in {'SPHERE', 'BOX'} and tool not in {'SPHERE', 'BOX'} \
                and self._depth_offs is not None:
            self._depth_offs.free()
            self._depth_offs = None
            self._depth_persp = None
        self._xform = None
        self._tool = tool
        context.scene.pobim_splat_edit_tool = tool
        self._set_status(context)
        if context.area:
            context.area.tag_redraw()

    # --- per-tool commits -------------------------------------------------

    def _commit_rect(self, context, op):
        (x0, y0), (x1, y1) = self._drag_start, self._drag_end
        xmin, xmax = (x0, x1) if x0 <= x1 else (x1, x0)
        ymin, ymax = (y0, y1) if y0 <= y1 else (y1, y0)
        # ignore near-degenerate rects: a plain click would otherwise commit
        # an empty 'set' rect and surprise-deselect everything
        if (xmax - xmin) < 3 and (ymax - ymin) < 3:
            return
        px, valid = self._project_all(context)
        if px is None:
            return
        inside = (valid &
                  (px[:, 0] >= xmin) & (px[:, 0] <= xmax) &
                  (px[:, 1] >= ymin) & (px[:, 1] <= ymax))
        inside &= self._eligible_mask()
        indices = np.nonzero(inside)[0]
        self._apply(context, 'Rect Select',
                    lambda: self._state.select_indices(indices, op))

    def _commit_polyshape(self, context, poly, op, label):
        """Shared LASSO/POLYGON commit: point-in-polygon over projected centers."""
        if _select_math is None or len(poly) < 3:
            return
        px, valid = self._project_all(context)
        if px is None:
            return
        inside = _select_math.points_in_polygon(px, list(poly))
        inside &= valid & self._eligible_mask()
        indices = np.nonzero(inside)[0]
        self._apply(context, label,
                    lambda: self._state.select_indices(indices, op))

    def _commit_sphere(self, context, op):
        if self._sphere_center is None:
            return
        mask = self._sphere_mask(context, self._sphere_center, self._sphere_radius)
        if mask is None:
            return
        mask &= self._eligible_mask()
        indices = np.nonzero(mask)[0]
        self._apply(context, 'Sphere Select',
                    lambda: self._state.select_indices(indices, op))

    def _commit_box(self, context, first_local, second_local, op):
        if _select_math is None:
            return
        bmin = np.minimum(first_local, second_local)
        bmax = np.maximum(first_local, second_local)
        # entry positions are already splat-local — no transform needed
        mask = _select_math.points_in_box(self._local, bmin, bmax)
        mask &= self._eligible_mask()
        indices = np.nonzero(mask)[0]
        self._apply(context, 'Box Select',
                    lambda: self._state.select_indices(indices, op))

    # --- brush stroke -----------------------------------------------------

    @staticmethod
    def _current_persp(context):
        rv3d = context.region_data
        return None if rv3d is None else np.array(rv3d.perspective_matrix,
                                                  np.float32)

    def _start_stroke(self, context, event):
        # without select_math the stroke can't paint anything — starting one
        # would still run select_none() below and commit a clear-only op
        if _select_math is None:
            return
        self._painting = True
        self._brush_snapshot = self._state.flags.copy()
        self._brush_op = 'remove' if event.ctrl else 'add'
        # plain (no modifier) = set-style: clear the current selection first,
        # then paint additively — all captured in one snapshot diff at release
        if not event.ctrl and not event.shift:
            self._state.select_none()
        # project once per stroke; _paint re-projects if the view changes
        # (wheel/trackpad zoom PASSes THROUGH even mid-drag)
        self._stroke_px, self._stroke_valid = self._project_all(context)
        self._stroke_persp = self._current_persp(context)
        self._last_paint_pt = None
        self._paint(context, *self._mouse)

    def _paint(self, context, x, y):
        if self._stroke_px is None or _select_math is None:
            return
        # viewport navigation passes through mid-stroke: refresh the cached
        # projection when the perspective matrix changed, else we test
        # against pre-zoom pixels and paint the wrong splats
        persp = self._current_persp(context)
        if (persp is None or self._stroke_persp is None or
                not np.array_equal(persp, self._stroke_persp)):
            self._stroke_px, self._stroke_valid = self._project_all(context)
            self._stroke_persp = persp
            if self._stroke_px is None:
                return
        if self._last_paint_pt is not None:
            dx = x - self._last_paint_pt[0]
            dy = y - self._last_paint_pt[1]
            if dx * dx + dy * dy < (self._brush_radius * 0.3) ** 2:
                return
        self._last_paint_pt = (x, y)
        stroke = np.array([[x, y]], np.float32)
        near = _select_math.points_near_polyline(
            self._stroke_px, stroke, self._brush_radius)
        near &= self._stroke_valid & self._eligible_mask()
        indices = np.nonzero(near)[0]
        if indices.size:
            self._state.select_indices(indices, self._brush_op)
            splat_gpu.redraw_viewports()

    def _commit_stroke(self, context):
        snapshot = self._brush_snapshot
        self._painting = False
        self._stroke_px = self._stroke_valid = None
        self._stroke_persp = None
        self._last_paint_pt = None
        if snapshot is None:
            return
        changed = np.nonzero(self._state.flags != snapshot)[0].astype(np.int64)
        self._brush_snapshot = None
        if changed.size == 0:
            self._set_status(context)
            if context.area:
                context.area.tag_redraw()
            return
        self._history.push({
            'label': 'Brush Select',
            'indices': changed,
            'before': snapshot[changed].copy(),
            'after': self._state.flags[changed].copy(),
        })
        self._persist(context)

    # --- lasso / polygon point management ---------------------------------

    def _append_lasso(self, x, y):
        if self._lasso_pts:
            lx, ly = self._lasso_pts[-1]
            if (x - lx) ** 2 + (y - ly) ** 2 < self._lasso_spacing ** 2:
                return
        self._lasso_pts.append((x, y))
        if len(self._lasso_pts) > _LASSO_CAP:
            # keep long strokes working: decimate and widen the spacing
            self._lasso_pts = self._lasso_pts[::2]
            self._lasso_spacing *= 2.0

    # --- radius management -------------------------------------------------

    def _write_radius(self, context):
        """Persist current radii back to the scene props."""
        sc = context.scene
        try:
            sc.pobim_splat_brush_radius = int(round(self._brush_radius))
            sc.pobim_splat_sphere_radius = float(self._sphere_radius)
        except Exception:
            pass

    def _scale_radius(self, factor):
        if self._active() == 'BRUSH':
            self._brush_radius = float(np.clip(
                self._brush_radius * factor, _BRUSH_MIN, _BRUSH_MAX))
        elif self._active() == 'SPHERE':
            self._sphere_radius = float(np.clip(
                self._sphere_radius * factor, _SPHERE_MIN, _SPHERE_MAX))

    def _start_resize(self, context):
        """Begin F interactive resize (Blender brush convention)."""
        active = self._active()
        if active not in {'BRUSH', 'SPHERE'}:
            return False
        mx, my = self._mouse
        if active == 'BRUSH':
            self._resize_before = self._brush_radius
            # anchor left of the cursor so the initial distance = current radius
            self._resize_anchor = (mx - self._brush_radius, my)
        else:
            if self._sphere_center is None:
                return False
            c = self._px(context, self._sphere_center)
            if c is None:
                return False
            self._resize_before = self._sphere_radius
            self._resize_anchor = c
        self._resizing = True
        return True

    def _update_resize(self, context, x, y):
        ax, ay = self._resize_anchor
        dist = float(np.hypot(x - ax, y - ay))
        if self._active() == 'BRUSH':
            self._brush_radius = float(np.clip(dist, _BRUSH_MIN, _BRUSH_MAX))
        else:
            wpp = self._world_per_pixel(context, self._sphere_center)
            if wpp is not None:
                self._sphere_radius = float(np.clip(
                    dist * wpp, _SPHERE_MIN, _SPHERE_MAX))

    def _end_resize(self, context, commit):
        if not commit and self._resize_before is not None:
            if self._active() == 'BRUSH':
                self._brush_radius = self._resize_before
            else:
                self._sphere_radius = self._resize_before
        self._resizing = False
        self._resize_before = None
        self._resize_anchor = None
        if commit:
            self._write_radius(context)
        self._set_status(context)

    # --- transform modes (Track T) ----------------------------------------

    def _enter_transform(self, context, mode):
        sel = np.nonzero((self._state.flags & State.SELECTED) != 0)[0]
        if sel.size == 0:
            self.report({'INFO'}, 'เลือก splat ก่อนจึงจะย้าย/หมุน/สเกลได้')
            self._set_status(context)
            return
        self._reset_gesture()
        self._xform = mode
        self._xform_sel = sel.astype(np.int64)
        self._xform_centroid = self._local[self._xform_sel].mean(axis=0).astype(np.float64)
        self._xform_axis = None
        self._xform_dragging = False
        self._xform_start = None
        self._xform_matrix = None
        self._set_status(context)
        if context.area:
            context.area.tag_redraw()

    def _start_transform_drag(self, context, x, y):
        if self._xform_sel is None:
            return
        self._xform_dragging = True
        self._xform_start = (x, y)
        self._xform_matrix = np.eye(4)
        self._set_preview(np.eye(4, dtype=np.float32))

    def _update_transform_drag(self, context, x, y, shift):
        if not self._xform_dragging or self._xform_start is None:
            return
        M = self._build_transform_matrix(context, x, y, shift)
        if M is None:
            return
        self._xform_matrix = M
        self._set_preview(M.astype(np.float32))
        if context.area:
            context.area.tag_redraw()

    def _build_transform_matrix(self, context, x, y, shift):
        """Local-space 4x4 preview matrix about the selection centroid."""
        rv3d = context.region_data
        if rv3d is None or self._xform_centroid is None:
            return None
        _, invM3 = self._matrix3()
        centroid_local = self._xform_centroid
        centroid_world = self._to_world(centroid_local).astype(np.float64)
        sx, sy = self._xform_start
        dx = float(x - sx)
        dy = float(y - sy)
        axis = self._xform_axis
        e = None if axis is None else np.eye(3)[axis]

        if self._xform == 'MOVE':
            wpp = self._world_per_pixel(context, centroid_world)
            if wpp is None:
                return None
            view = np.array(rv3d.view_matrix, np.float64)
            right = view[0, :3]
            up = view[1, :3]
            world_delta = right * (dx * wpp) + up * (dy * wpp)
            if shift:
                world_delta *= 0.1
            local_delta = invM3 @ world_delta
            if e is not None:
                local_delta = e * float(local_delta @ e)
            return _translate4(local_delta)

        # both ROTATE and SCALE need the centroid's screen position
        c_px = self._px(context, centroid_world)
        if c_px is None:
            return None
        if self._xform == 'ROTATE':
            a0 = np.arctan2(sy - c_px[1], sx - c_px[0])
            a1 = np.arctan2(y - c_px[1], x - c_px[0])
            angle = float(a1 - a0)
            if shift:
                angle *= 0.1
            view = np.array(rv3d.view_matrix, np.float64)
            axis_world = -view[2, :3]           # camera forward, in world
            if e is not None:
                axis_world = self._matrix3()[0] @ e     # local axis -> world
            axis_local = invM3 @ axis_world
            n = np.linalg.norm(axis_local)
            if n < 1e-9:
                return None
            axis_local /= n
            R3 = _rot_axis_angle(axis_local, angle)
            return _about_pivot(R3, centroid_local)

        # SCALE
        d0 = float(np.hypot(sx - c_px[0], sy - c_px[1]))
        d1 = float(np.hypot(x - c_px[0], y - c_px[1]))
        ratio = d1 / d0 if d0 > 1e-3 else 1.0
        if shift:
            ratio = 1.0 + (ratio - 1.0) * 0.1
        ratio = float(np.clip(ratio, 1e-3, 1e3))
        if e is None:
            s = np.array([ratio, ratio, ratio], np.float64)
        else:
            s = np.ones(3, np.float64)
            s[axis] = ratio
        return _about_pivot(np.diag(s), centroid_local)

    def _set_preview(self, mat4):
        obj = bpy.data.objects.get(self._obj_name)
        uid = obj.pobim_splat_uid if obj is not None else None
        if uid and hasattr(splat_gpu, 'set_preview'):
            try:
                splat_gpu.set_preview(uid, mat4)
                splat_gpu.redraw_viewports()
            except Exception as e:
                print(f'[pobim_splats] set_preview failed: {e}')

    def _clear_preview(self):
        obj = bpy.data.objects.get(getattr(self, '_obj_name', ''))
        uid = obj.pobim_splat_uid if obj is not None else None
        if uid and hasattr(splat_gpu, 'clear_preview'):
            try:
                splat_gpu.clear_preview(uid)
                splat_gpu.redraw_viewports()
            except Exception as e:
                print(f'[pobim_splats] clear_preview failed: {e}')

    def _cancel_transform(self):
        """Drop an in-progress transform gesture and its preview."""
        self._xform_dragging = False
        self._xform_start = None
        self._xform_matrix = None
        self._clear_preview()

    def _commit_transform(self, context):
        """Bake the preview matrix into SplatEdits + GPU + history (Track T)."""
        M = self._xform_matrix
        self._xform_dragging = False
        self._xform_start = None
        if M is None or self._edits is None or self._xform_sel is None:
            self._cancel_transform()
            return
        if np.allclose(M, np.eye(4)):
            self._cancel_transform()
            return
        cloud = self._entry.cloud
        base_quat = getattr(cloud, 'quats', None)
        base_slog = getattr(cloud, 'scales_log', None)
        if base_quat is None or base_slog is None:
            # keep_geometry was not stashed — cannot rotate/scale quats
            print('[pobim_splats] transform needs cloud.quats/scales_log '
                  '(keep_geometry); skipping commit')
            self._cancel_transform()
            return
        # re-read the CURRENT selection at commit time: a mid-drag undo/redo
        # can change it, and the GPU preview only ever moved SELECTED splats
        # (the shader samples the live state texture), so committing to the
        # frozen _xform_sel would move splats the user never saw move.
        sel = np.nonzero((self._state.flags & State.SELECTED) != 0)[0].astype(np.int64)
        if sel.size == 0:
            self._cancel_transform()
            self._set_status(context)
            return
        self._xform_sel = sel
        self._xform_centroid = self._local[sel].mean(axis=0).astype(np.float64)
        try:
            result = self._edits.apply_matrix(
                sel, M, self._local, base_quat, base_slog)
            if result is None:
                self._cancel_transform()
                return
            idx, before, after = result
            self._push_geometry(idx, after)
            self._history.push({
                'kind': 'transform',
                'label': f'{self._xform.title()} Selected',
                'indices': np.asarray(idx, np.int64),
                'before': before,
                'after': after,
            })
            self._persist_edits(context)
            # _local was updated in place by the GPU commit — refresh the
            # centroid so the NEXT drag pivots about the moved selection
            self._xform_centroid = self._local[sel].mean(axis=0).astype(np.float64)
        except Exception as e:
            print(f'[pobim_splats] transform commit failed: {e}')
        self._xform_matrix = None
        self._clear_preview()
        self._set_status(context)
        if context.area:
            context.area.tag_redraw()

    def _push_geometry(self, idx, payload):
        """Recompute cov6 for ``payload`` (dict of pos/quat/scales_log) and
        upload the edited splats to the GPU data texture."""
        gpu_obj = self._entry.gpu
        if gpu_obj is None or not hasattr(gpu_obj, 'update_splats'):
            return
        quat = payload['quats']
        scales_log = payload['scales_log']
        pos = payload['positions']
        try:
            if hasattr(splat_gpu, 'recompute_cov6'):
                cov6 = splat_gpu.recompute_cov6(quat, scales_log)
            else:
                scales_lin = np.exp(np.asarray(scales_log, np.float32))
                cov6 = np.empty((len(idx), 6), np.float32)
                ply_loader._quat_scale_to_cov6(
                    np.asarray(quat, np.float32), scales_lin, cov6)
            gpu_obj.update_splats(np.asarray(idx, np.int64),
                                  np.asarray(pos, np.float32), cov6)
        except Exception as e:
            print(f'[pobim_splats] GPU geometry update failed: {e}')

    def _persist_edits(self, context):
        obj = bpy.data.objects.get(self._obj_name)
        if obj is not None and self._edits is not None:
            try:
                obj[_EDIT_PROP] = self._edits.serialize()
            except Exception as e:
                print(f'[pobim_splats] edit serialize failed: {e}')
        self._set_status(context)
        splat_gpu.redraw_viewports()
        if context.area:
            context.area.tag_redraw()

    def _restore_transform(self, op, which):
        if self._edits is None:
            return
        idx = op['indices']
        payload = op[which]
        try:
            self._edits.restore(idx, payload['positions'], payload['quats'],
                                payload['scales_log'])
            self._push_geometry(idx, payload)
        except Exception as e:
            print(f'[pobim_splats] transform restore failed: {e}')

    # --- undo / redo dispatch (peeks kind; transform never hits set_flags) --

    def _do_undo(self, context):
        h = self._history
        if not h.can_undo:
            return
        op = h.ops[h.cursor - 1]
        if op.get('kind') == 'transform':
            h.cursor -= 1
            self._restore_transform(op, 'before')
            self._persist_edits(context)
        else:
            h.undo(self._state)
            self._persist(context)

    def _do_redo(self, context):
        h = self._history
        if not h.can_redo:
            return
        op = h.ops[h.cursor]
        if op.get('kind') == 'transform':
            h.cursor += 1
            self._restore_transform(op, 'after')
            self._persist_edits(context)
        else:
            h.redo(self._state)
            self._persist(context)

    # --- HUD chip toolbar -------------------------------------------------

    def _build_hud(self, region):
        """Compute chip rects (id,label,rect,state) for draw + hit-test."""
        if region is None:
            return []
        font = 0
        blf.size(font, 13.0)
        pad = 9.0
        gap = 6.0
        sep = 12.0
        h = 24.0
        chips = []

        def measure(specs):
            items, width = [], 0.0
            for cid, label in specs:
                if cid == 'SEP':
                    items.append(('SEP', None, sep))
                    width += sep + gap
                    continue
                tw = blf.dimensions(font, label)[0]
                w = tw + 2 * pad
                items.append((cid, label, w))
                width += w + gap
            return items, max(0.0, width - gap)

        def place(items, total, y0):
            x = (region.width - total) * 0.5
            for cid, label, w in items:
                if cid == 'SEP':
                    x += w + gap
                    continue
                chips.append({
                    'id': cid, 'label': label,
                    'x0': x, 'y0': y0, 'x1': x + w, 'y1': y0 + h,
                    'active': self._chip_active(cid),
                    'disabled': self._chip_disabled(cid),
                })
                x += w + gap

        items1, total1 = measure(_HUD_ROW1)
        y1 = region.height - 10.0 - h
        place(items1, total1, y1)

        row2 = self._hud_row2()
        if row2:
            items2, total2 = measure(row2)
            place(items2, total2, y1 - gap - h)

        self._hud_chips = chips
        return chips

    def _hud_row2(self):
        """Contextual second row: radius slider / dims / apply chips."""
        active = self._active()
        specs = []
        if active == 'BRUSH':
            specs.append(('radius', f'Radius: {self._brush_radius:.0f} px'))
        elif active == 'SPHERE':
            specs.append(('radius', f'Radius: {self._sphere_radius:.3f} m'))
            if self._sphere_stage == 'PREVIEW':
                specs.append(('apply', 'Apply'))
        elif active == 'BOX':
            dims = self._box_dims_label()
            specs.append(('hint', dims if dims else 'Box: คลิกสองมุม'))
            if self._box_stage == 'PREVIEW':
                specs.append(('apply', 'Apply'))
        elif active in _XFORM_ITEMS:
            lock = ('X', 'Y', 'Z')[self._xform_axis] if self._xform_axis is not None else '—'
            specs.append(('hint', f'{active.title()} · axis {lock}'))
            if self._xform_dragging:
                specs.append(('apply', 'Apply'))
        return specs

    def _box_dims_label(self):
        if self._box_c1 is None:
            return None
        c2 = self._box_c2
        if c2 is None:
            if self._box_hover is None:
                return None
            c2 = self._to_local(self._box_hover)
        d = np.abs(np.asarray(c2, np.float64) - np.asarray(self._box_c1, np.float64))
        return f'Box: {d[0]:.3f} × {d[1]:.3f} × {d[2]:.3f} m'

    def _chip_active(self, cid):
        if cid == f'tool:{self._tool}' and self._xform is None:
            return True
        if self._xform is not None and cid == f'xform:{self._xform}':
            return True
        return False

    def _chip_disabled(self, cid):
        if cid == 'undo':
            return not self._history.can_undo
        if cid == 'redo':
            return not self._history.can_redo
        return False

    def _hud_hit(self, region, mx, my):
        for c in self._build_hud(region):
            if c['disabled']:
                continue
            if c['x0'] <= mx <= c['x1'] and c['y0'] <= my <= c['y1']:
                return c
        return None

    def _hud_click(self, context, event, chip):
        """Handle a chip click. Returns 'finish' to exit the modal."""
        cid = chip['id']
        if cid.startswith('tool:'):
            self._switch_tool(context, cid.split(':', 1)[1])
        elif cid.startswith('xform:'):
            self._enter_transform(context, cid.split(':', 1)[1])
        elif cid == 'undo':
            self._do_undo(context)
        elif cid == 'redo':
            self._do_redo(context)
        elif cid == 'done':
            return 'finish'
        elif cid == 'apply':
            self._confirm_active(context, event)
        elif cid == 'radius':
            before = (self._brush_radius if self._active() == 'BRUSH'
                      else self._sphere_radius)
            self._hud_drag = ('radius', event.mouse_region_x, before)
        return None

    def _update_hud_drag(self, context, x):
        kind, sx, before = self._hud_drag
        if kind != 'radius':
            return
        dx = float(x - sx)
        if self._active() == 'BRUSH':
            self._brush_radius = float(np.clip(before + dx, _BRUSH_MIN, _BRUSH_MAX))
        else:
            # 1px drag ≈ 0.005 m; keeps the whole range reachable in one sweep
            self._sphere_radius = float(np.clip(
                before + dx * 0.005, _SPHERE_MIN, _SPHERE_MAX))
        if context.area:
            context.area.tag_redraw()

    # --- confirm dispatch (Enter / Apply chip / drag-release) -------------

    def _confirm_active(self, context, event):
        """Commit the active preview gesture (box / sphere / transform)."""
        active = self._active()
        if active == 'BOX' and self._box_stage == 'PREVIEW':
            self._commit_box(context, self._box_c1, self._box_c2,
                             self._op_from_event(event))
            self._box_c1 = self._box_c2 = None
            self._box_hover = None
            self._box_stage = None
            self._box_grab = None
        elif active == 'SPHERE' and self._sphere_stage == 'PREVIEW':
            self._commit_sphere(context, self._op_from_event(event))
            self._sphere_center = None
            self._sphere_stage = None
            self._sphere_grab = False
        elif active in _XFORM_ITEMS and self._xform_dragging:
            self._commit_transform(context)
        if context.area:
            context.area.tag_redraw()

    # --- input dispatch ---------------------------------------------------

    def _on_mousemove(self, context):
        x, y = self._mouse
        self._hud_hover = None
        region = context.region
        if region is not None:
            hit = self._hud_hit(region, x, y)
            self._hud_hover = hit['id'] if hit else None
        if self._hud_drag is not None:
            self._update_hud_drag(context, x)
            return
        if self._resizing:
            self._update_resize(context, x, y)
            self._set_status(context)
            return
        active = self._active()
        if active in _XFORM_ITEMS:
            if self._xform_dragging:
                self._update_transform_drag(context, x, y, self._shift_state)
            return
        t = self._tool
        if t == 'RECT':
            if self._dragging:
                self._drag_end = (x, y)
        elif t == 'LASSO':
            if self._lassoing:
                self._append_lasso(x, y)
        elif t == 'BRUSH':
            if self._painting:
                self._paint(context, x, y)
        elif t == 'SPHERE':
            if self._sphere_grab or self._sphere_stage is None:
                w = self._pick_world(context, x, y)
                if w is not None:
                    self._sphere_center = w
        elif t == 'BOX':
            if self._box_grab is not None:
                w = self._pick_world(context, x, y)
                if w is not None:
                    if self._box_grab == 0:
                        self._box_c1 = self._to_local(w)
                    else:
                        self._box_c2 = self._to_local(w)
            elif self._box_stage == 'CORNER2':
                self._box_hover = self._pick_world(context, x, y)

    def _corner_px(self, context):
        """Screen positions of the two box corners (PREVIEW stage)."""
        if self._box_c1 is None or self._box_c2 is None:
            return None, None
        return (self._px(context, self._to_world(self._box_c1)),
                self._px(context, self._to_world(self._box_c2)))

    def _box_corner_under(self, context, x, y):
        p0, p1 = self._corner_px(context)
        best, bd = None, _CORNER_HIT_PX ** 2
        for i, p in ((0, p0), (1, p1)):
            if p is None:
                continue
            d = (p[0] - x) ** 2 + (p[1] - y) ** 2
            if d <= bd:
                best, bd = i, d
        return best

    def _sphere_center_under(self, context, x, y):
        if self._sphere_center is None:
            return False
        c = self._px(context, self._sphere_center)
        if c is None:
            return False
        return (c[0] - x) ** 2 + (c[1] - y) ** 2 <= _CORNER_HIT_PX ** 2

    def _on_leftmouse(self, context, event):
        x, y = event.mouse_region_x, event.mouse_region_y
        self._mouse = (x, y)
        press = event.value == 'PRESS'
        release = event.value == 'RELEASE'

        # radius drag-slider release
        if self._hud_drag is not None:
            if release:
                self._hud_drag = None
                self._write_radius(context)
                self._set_status(context)
            if context.area:
                context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        # F interactive resize: LMB confirms
        if self._resizing:
            if press:
                self._end_resize(context, commit=True)
            if context.area:
                context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        # HUD hit-test runs BEFORE any gesture logic
        if press:
            chip = self._hud_hit(context.region, x, y)
            if chip is not None:
                if self._hud_click(context, event, chip) == 'finish':
                    self._finish(context)
                    return {'FINISHED'}
                if context.area:
                    context.area.tag_redraw()
                return {'RUNNING_MODAL'}

        active = self._active()

        # transform gesture: press starts drag, release confirms
        if active in _XFORM_ITEMS:
            if press:
                self._start_transform_drag(context, x, y)
            elif release and self._xform_dragging:
                self._commit_transform(context)
            if context.area:
                context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        t = self._tool
        if t == 'RECT':
            if press:
                self._dragging = True
                self._drag_start = (x, y)
                self._drag_end = self._drag_start
            elif release and self._dragging:
                self._dragging = False
                self._drag_end = (x, y)
                self._commit_rect(context, self._op_from_event(event))
        elif t == 'LASSO':
            if press:
                self._lassoing = True
                self._lasso_pts = [(x, y)]
                self._lasso_spacing = _LASSO_MIN_SPACING
            elif release and self._lassoing:
                self._lassoing = False
                self._commit_polyshape(context, self._lasso_pts,
                                       self._op_from_event(event), 'Lasso Select')
                self._lasso_pts = []
        elif t == 'POLYGON':
            if press:
                if (len(self._poly_pts) >= 3 and
                        (x - self._poly_pts[0][0]) ** 2 +
                        (y - self._poly_pts[0][1]) ** 2 <= _POLY_CLOSE_PX ** 2):
                    self._commit_polyshape(context, self._poly_pts,
                                           self._op_from_event(event), 'Polygon Select')
                    self._poly_pts = []
                else:
                    self._poly_pts.append((x, y))
        elif t == 'BRUSH':
            if press:
                self._start_stroke(context, event)
            elif release and self._painting:
                self._commit_stroke(context)
        elif t == 'SPHERE':
            if press:
                if self._sphere_grab:
                    # drop the grabbed centre
                    self._sphere_grab = False
                elif self._sphere_stage == 'PREVIEW' and \
                        self._sphere_center_under(context, x, y):
                    self._sphere_grab = True
                else:
                    w = self._pick_world(context, x, y)
                    if w is not None:
                        self._sphere_center = w
                        self._sphere_stage = 'PREVIEW'
        elif t == 'BOX':
            if press:
                if self._box_stage == 'PREVIEW':
                    if self._box_grab is not None:
                        self._box_grab = None      # drop the grabbed corner
                    else:
                        c = self._box_corner_under(context, x, y)
                        if c is not None:
                            self._box_grab = c     # grab a corner to drag
                else:
                    w = self._pick_world(context, x, y)
                    if w is not None:
                        if self._box_c1 is None:
                            self._box_c1 = self._to_local(w)
                            self._box_hover = w
                            self._box_stage = 'CORNER2'
                        else:
                            self._box_c2 = self._to_local(w)
                            self._box_hover = None
                            self._box_stage = 'PREVIEW'
        if context.area:
            context.area.tag_redraw()
        return {'RUNNING_MODAL'}

    # --- modal ------------------------------------------------------------

    def modal(self, context, event):
        # Alt+Wheel adjusts the active radius; plain wheel stays navigation
        if event.type in {'WHEELUPMOUSE', 'WHEELDOWNMOUSE'} and event.alt:
            self._scale_radius(1.1 if event.type == 'WHEELUPMOUSE' else 1.0 / 1.1)
            self._write_radius(context)
            self._set_status(context)
            if context.area:
                context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        if (event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE',
                           'TRACKPADPAN', 'TRACKPADZOOM', 'MOUSEROTATE'} or
                (event.type.startswith('NUMPAD') and event.type != 'NUMPAD_ENTER')):
            return {'PASS_THROUGH'}  # keep viewport navigation working

        # remember modifier state for the mouse-move driven transform preview
        self._shift_state = event.shift

        if event.type == 'MOUSEMOVE':
            self._mouse = (event.mouse_region_x, event.mouse_region_y)
            self._on_mousemove(context)
            if context.area:
                context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        if event.type == 'LEFTMOUSE':
            return self._on_leftmouse(context, event)

        if event.value == 'PRESS':
            et = event.type

            # transform mode axis-lock keys (X/Y/Z) take priority over
            # delete/etc while a transform mode is active
            if self._xform is not None and et in {'X', 'Y', 'Z'}:
                axis = {'X': 0, 'Y': 1, 'Z': 2}[et]
                self._xform_axis = None if self._xform_axis == axis else axis
                if self._xform_dragging:
                    self._update_transform_drag(context, *self._mouse,
                                                self._shift_state)
                self._set_status(context)
                if context.area:
                    context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            # F: interactive radius resize (brush/sphere)
            if et == 'F' and not self._resizing:
                if self._start_resize(context):
                    self._set_status(context)
                    if context.area:
                        context.area.tag_redraw()
                return {'RUNNING_MODAL'}
            if self._resizing and et in {'RET', 'NUMPAD_ENTER'}:
                self._end_resize(context, commit=True)
                if context.area:
                    context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            # transform mode keys 1/2/3 + G (needs a selection)
            if et in _XFORM_KEYS and not event.ctrl:
                self._enter_transform(context, _XFORM_KEYS[et])
                return {'RUNNING_MODAL'}

            # selection tool switch keys — cancel any in-progress gesture
            if et in _TOOL_KEYS and not event.ctrl:
                self._switch_tool(context, _TOOL_KEYS[et])
                return {'RUNNING_MODAL'}

            # brush / sphere radius bracket keys
            if et in {'LEFT_BRACKET', 'RIGHT_BRACKET'}:
                self._scale_radius(0.8 if et == 'LEFT_BRACKET' else 1.25)
                if self._active() in {'BRUSH', 'SPHERE'}:
                    self._write_radius(context)
                    self._set_status(context)
                    if context.area:
                        context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            # Enter confirms preview gestures / closes the polygon
            if et in {'RET', 'NUMPAD_ENTER'}:
                active = self._active()
                if active == 'POLYGON' and len(self._poly_pts) >= 3:
                    self._commit_polyshape(context, self._poly_pts,
                                           self._op_from_event(event), 'Polygon Select')
                    self._poly_pts = []
                    if context.area:
                        context.area.tag_redraw()
                else:
                    self._confirm_active(context, event)
                return {'RUNNING_MODAL'}

            # remove the last polygon vertex
            if et == 'BACK_SPACE':
                if self._tool == 'POLYGON' and self._poly_pts:
                    self._poly_pts.pop()
                    if context.area:
                        context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            if et == 'A':
                if event.shift or event.alt:
                    self._apply(context, 'Select None', self._state.select_none)
                else:
                    self._apply(context, 'Select All', self._state.select_all)
                return {'RUNNING_MODAL'}
            if et == 'I' and event.ctrl:
                self._apply(context, 'Invert Selection', self._state.select_invert)
                return {'RUNNING_MODAL'}
            if et == 'H':
                if event.alt:
                    self._apply(context, 'Unhide All', self._state.unhide_all)
                else:
                    self._apply(context, 'Hide Selected', self._state.hide_selected)
                return {'RUNNING_MODAL'}
            if et in {'X', 'DEL'} and self._xform is None:
                self._apply(context, 'Delete Selected', self._state.delete_selected)
                return {'RUNNING_MODAL'}
            if et == 'Z' and event.ctrl:
                if event.shift:
                    self._do_redo(context)
                else:
                    self._do_undo(context)
                return {'RUNNING_MODAL'}

            if et in {'ESC', 'RIGHTMOUSE'}:
                return self._on_cancel(context)

        return {'RUNNING_MODAL'}

    def _on_cancel(self, context):
        """Esc/RMB: cancel an in-progress gesture first; exit only when idle."""
        if self._resizing:
            self._end_resize(context, commit=False)
            if context.area:
                context.area.tag_redraw()
            return {'RUNNING_MODAL'}
        if self._xform is not None:
            if self._xform_dragging:
                self._cancel_transform()          # revert preview, stay in mode
                self._set_status(context)
                if context.area:
                    context.area.tag_redraw()
                return {'RUNNING_MODAL'}
            # idle in a transform mode -> back to the selection tool
            self._xform = None
            self._set_status(context)
            if context.area:
                context.area.tag_redraw()
            return {'RUNNING_MODAL'}
        if self._painting:
            self._reset_gesture()
            if context.area:
                context.area.tag_redraw()
            return {'RUNNING_MODAL'}
        if self._tool == 'POLYGON' and self._poly_pts:
            self._poly_pts = []
            if context.area:
                context.area.tag_redraw()
            return {'RUNNING_MODAL'}
        if self._tool == 'BOX' and (self._box_c1 is not None or
                                    self._box_stage is not None):
            self._box_c1 = self._box_c2 = None
            self._box_hover = None
            self._box_stage = None
            self._box_grab = None
            if context.area:
                context.area.tag_redraw()
            return {'RUNNING_MODAL'}
        if self._tool == 'SPHERE' and self._sphere_stage is not None:
            self._sphere_center = None
            self._sphere_stage = None
            self._sphere_grab = False
            if context.area:
                context.area.tag_redraw()
            return {'RUNNING_MODAL'}
        if self._tool == 'LASSO' and self._lassoing:
            self._lassoing = False
            self._lasso_pts = []
            if context.area:
                context.area.tag_redraw()
            return {'RUNNING_MODAL'}
        self._finish(context)
        return {'FINISHED'}

    # --- overlay ----------------------------------------------------------

    def _lines(self, coords, color, width, strip=True):
        gpu.state.line_width_set(width)
        batch = batch_for_shader(
            self._shader, 'LINE_STRIP' if strip else 'LINES', {'pos': coords})
        self._shader.uniform_float('color', color)
        batch.draw(self._shader)
        gpu.state.line_width_set(1.0)

    def _fill(self, coords, color):
        batch = batch_for_shader(self._shader, 'TRI_FAN', {'pos': coords})
        self._shader.uniform_float('color', color)
        batch.draw(self._shader)

    def _stroke2(self, coords, strip=True):
        """dark under-stroke + crisp white line (the rect-select look)."""
        self._lines(coords, _DARK, 2.0, strip)
        self._lines(coords, _WHITE, 1.5, strip)

    def _dot(self, p, active):
        r = 6.0 if active else 4.5
        ring = _circle(p[0], p[1], r)
        self._fill([p] + ring, _ORANGE if active else _WHITE)
        self._lines(ring, _DARK, 2.0)

    def _px(self, context, world_point):
        rv3d = context.region_data
        region = context.region
        if rv3d is None or region is None:
            return None
        persp = np.array(rv3d.perspective_matrix, np.float32)
        px, _, valid = project_to_pixels(
            persp, np.asarray(world_point, np.float32)[None, :],
            region.width, region.height)
        return (float(px[0, 0]), float(px[0, 1])) if valid[0] else None

    def _draw_overlay(self, context):
        try:
            gpu.state.blend_set('ALPHA')
            self._shader.bind()
            if self._resizing:
                self._draw_resize(context)
            else:
                active = self._active()
                if active == 'RECT':
                    self._draw_rect()
                elif active == 'LASSO':
                    self._draw_lasso()
                elif active == 'POLYGON':
                    self._draw_polygon()
                elif active == 'BRUSH':
                    self._draw_brush()
                elif active == 'SPHERE':
                    self._draw_sphere(context)
                elif active == 'BOX':
                    self._draw_box(context)
                elif active in _XFORM_ITEMS:
                    self._draw_transform(context)
            self._draw_hud(context)
            gpu.state.blend_set('NONE')
        except Exception as e:
            print(f'[pobim_splats] edit overlay error: {e}')

    def _draw_resize(self, context):
        if self._active() == 'BRUSH':
            cx, cy = self._mouse
            self._lines(_circle(cx, cy, self._brush_radius), _DARK, 2.0)
            self._lines(_circle(cx, cy, self._brush_radius), _ORANGE, 1.5)
        else:
            self._draw_sphere(context, color=_ORANGE)

    def _draw_rect(self):
        if not self._dragging or self._drag_start is None or self._drag_end is None:
            return
        (x0, y0), (x1, y1) = self._drag_start, self._drag_end
        coords = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
        self._stroke2(coords)

    def _draw_lasso(self):
        if not self._lassoing or len(self._lasso_pts) < 2:
            return
        coords = [(float(x), float(y)) for x, y in self._lasso_pts]
        coords = coords + [coords[0]]
        self._stroke2(coords)

    def _draw_polygon(self):
        pts = [(float(x), float(y)) for x, y in self._poly_pts]
        if pts:
            rubber = pts + [(float(self._mouse[0]), float(self._mouse[1]))]
            if len(pts) >= 2:
                self._stroke2(pts)
            self._stroke2(rubber)
            self._lines(_circle(pts[0][0], pts[0][1], _POLY_CLOSE_PX), _DARK, 2.0)
            self._lines(_circle(pts[0][0], pts[0][1], _POLY_CLOSE_PX), _WHITE, 1.5)

    def _draw_brush(self):
        cx, cy = self._mouse
        self._lines(_circle(cx, cy, self._brush_radius), _DARK, 2.0)
        self._lines(_circle(cx, cy, self._brush_radius), _WHITE, 1.5)

    def _draw_sphere(self, context, color=_WHITE):
        if self._sphere_center is None:
            return
        c = self._px(context, self._sphere_center)
        if c is None:
            return
        r = self._sphere_pixel_radius(context, self._sphere_center,
                                      self._sphere_radius)
        if r is None or r < 1.0:
            r = 1.0
        self._lines(_circle(c[0], c[1], r), _DARK, 2.0)
        self._lines(_circle(c[0], c[1], r), color, 1.5)
        if self._sphere_stage == 'PREVIEW':
            grab = self._sphere_grab or self._sphere_center_under(context, *self._mouse)
            self._dot(c, grab)

    def _draw_box(self, context):
        if self._box_c1 is None:
            return
        obj = bpy.data.objects.get(self._obj_name)
        if obj is None:
            return
        m = np.array(obj.matrix_world, np.float64)
        if self._box_c2 is not None:
            c2 = self._box_c2
        elif self._box_hover is not None:
            c2 = self._to_local(self._box_hover)
        else:
            return
        bmin = np.minimum(self._box_c1, c2)
        bmax = np.maximum(self._box_c1, c2)
        cpx = []
        for c in box_corners(bmin, bmax):
            w = m[:3, :3] @ np.asarray(c, np.float64) + m[:3, 3]
            cpx.append(self._px(context, w))
        for e0, e1 in _BOX_EDGES:
            if cpx[e0] is not None and cpx[e1] is not None:
                self._stroke2([cpx[e0], cpx[e1]])
        if self._box_stage == 'PREVIEW':
            hover = self._box_corner_under(context, *self._mouse)
            p0, p1 = self._corner_px(context)
            for i, p in ((0, p0), (1, p1)):
                if p is not None:
                    self._dot(p, self._box_grab == i or hover == i)

    def _draw_transform(self, context):
        if self._xform_centroid is None:
            return
        c = self._px(context, self._to_world(self._xform_centroid))
        if c is None:
            return
        # centroid cross-hair (orange while dragging)
        col = _ORANGE if self._xform_dragging else _WHITE
        self._lines([(c[0] - 9, c[1]), (c[0] + 9, c[1])], _DARK, 2.0, strip=False)
        self._lines([(c[0], c[1] - 9), (c[0], c[1] + 9)], _DARK, 2.0, strip=False)
        self._lines([(c[0] - 8, c[1]), (c[0] + 8, c[1]),
                     (c[0], c[1] - 8), (c[0], c[1] + 8)], col, 1.5, strip=False)

    def _draw_hud(self, context):
        chips = self._build_hud(context.region)
        if not chips:
            return
        font = 0
        blf.size(font, 13.0)
        for c in chips:
            if c['active']:
                bg, txt = _CHIP_BG_ACTIVE, _CHIP_TEXT_ACTIVE
            elif c['id'] == self._hud_hover and not c['disabled']:
                bg, txt = _CHIP_BG_HOVER, _CHIP_TEXT
            else:
                bg = _CHIP_BG
                txt = _CHIP_TEXT_DISABLED if c['disabled'] else _CHIP_TEXT
            x0, y0, x1, y1 = c['x0'], c['y0'], c['x1'], c['y1']
            self._fill([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], bg)
            self._lines([(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)],
                        _DARK, 1.0)
            tw, th = blf.dimensions(font, c['label'])
            blf.position(font, (x0 + x1) / 2 - tw / 2,
                         (y0 + y1) / 2 - th / 2 + 1.0, 0)
            blf.color(font, *txt)
            blf.draw(font, c['label'])


class POBIM_OT_export_ply(bpy.types.Operator, ExportHelper):
    """ส่งออก splat ที่เหลือเป็นไฟล์ .ply (ตัด splat ที่ลบออกไปแล้ว)"""
    bl_idname = 'pobim_splats.export_ply'
    bl_label = 'Export PLY'

    filename_ext = '.ply'
    filter_glob: StringProperty(default='*.ply', options={'HIDDEN'})

    uid: StringProperty()

    def invoke(self, context, event):
        obj = _find_splat_object(context, self.uid)
        if obj is not None and not self.filepath:
            self.filepath = (obj.name or 'splats') + '.ply'
        return ExportHelper.invoke(self, context, event)

    def execute(self, context):
        obj = _find_splat_object(context, self.uid)
        if obj is None:
            self.report({'ERROR'}, 'ไม่พบ object ของ splat')
            return {'CANCELLED'}
        entry = splat_gpu.REGISTRY.get(obj.pobim_splat_uid)
        if entry is None:
            self.report({'ERROR'}, 'splat ยังไม่ได้โหลด — กด Reload ก่อน')
            return {'CANCELLED'}

        # keep_mask indexes the LOADED cloud; source_indices maps loaded rows
        # back to original file rows for subsampled imports (survives the GPU
        # build, which only frees the cloud's heavy value arrays).
        keep_mask = entry.state.keep_mask() if entry.state is not None else None
        source_indices = entry.cloud.source_indices if entry.cloud is not None else None
        source_path = bpy.path.abspath(obj.pobim_splat_file)

        # geometry-edit payload (Track T): pass patched fields for the splats
        # that were transformed so the export reflects Move/Rotate/Scale edits.
        # NEVER export silently without them: a payload failure aborts.
        try:
            edits_payload = self._edits_payload(entry)
        except Exception as e:
            self.report({'ERROR'}, f'อ่านข้อมูลแก้ไข geometry ไม่สำเร็จ: {e}')
            return {'CANCELLED'}

        # explicit capability probe (an `except TypeError` fallback would also
        # swallow genuine TypeErrors from inside export_ply)
        import inspect
        supports_edits = 'edits' in inspect.signature(
            splat_export.export_ply).parameters
        if edits_payload is not None and not supports_edits:
            self.report({'ERROR'},
                        'export_ply รุ่นนี้ไม่รองรับ geometry edits — '
                        'มีการแก้ไข Move/Rotate/Scale ที่จะสูญหาย จึงยกเลิกการส่งออก')
            return {'CANCELLED'}
        try:
            if supports_edits:
                n = splat_export.export_ply(
                    source_path, self.filepath, keep_mask, source_indices,
                    edits=edits_payload)
            else:
                n = splat_export.export_ply(
                    source_path, self.filepath, keep_mask, source_indices)
        except Exception as e:
            self.report({'ERROR'}, f'ส่งออกไม่สำเร็จ: {e}')
            return {'CANCELLED'}
        self.report({'INFO'}, f'ส่งออก {n:,} splats → {self.filepath}')
        return {'FINISHED'}

    @staticmethod
    def _edits_payload(entry):
        """Build the {'indices','positions','quats','scales_log'} dict for the
        dirty splats, or None when there are no geometry edits.

        Raises when dirty edits exist but cannot be read (the caller aborts
        the export instead of silently dropping the user's transforms).
        SplatEdits.export_payload handles both the materialized case and the
        deserialized-but-pending case (fresh .blend reload, no draw yet)."""
        edits = getattr(entry, 'edits', None)
        if edits is None:
            return None
        if hasattr(edits, 'export_payload'):
            return edits.export_payload()
        # legacy SplatEdits without export_payload: dense arrays only
        dirty = getattr(edits, 'dirty', None)
        if dirty is None or not bool(np.any(dirty)):
            return None
        if edits.positions is None:
            raise RuntimeError('geometry edits present but not materialized')
        idx = np.nonzero(dirty)[0]
        return {
            'indices': idx,
            'positions': edits.positions[idx],
            'quats': edits.quats[idx],
            'scales_log': edits.scales_log[idx],
        }


CLASSES = (POBIM_OT_edit_splats, POBIM_OT_export_ply)

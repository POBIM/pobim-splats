# Per-splat editing: a modal selection tool with a tool-local undo stack,
# plus a .ply export of the surviving splats. Mirrors POBIMStudio's editing
# model (State bit flags + EditHistory) and copies measure.py's modal
# lifecycle (running guard, cancel(), status text, POST_PIXEL overlay,
# PASS_THROUGH viewport navigation).
#
# Phase 1: rect select. Phase 2 adds SuperSplat-style tools — Lasso (L),
# Polygon (P), Brush (B), Sphere (S) and Box (C). Rect (R) stays the
# default. Every tool commits through the same _apply path: ONE EditHistory
# op per commit, state persisted, viewport redrawn.

import bpy
import gpu
import numpy as np
from bpy.props import StringProperty
from bpy_extras.io_utils import ExportHelper
from gpu_extras.batch import batch_for_shader

from . import splat_export, splat_gpu
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

_STATE_PROP = 'pobim_splat_state'

# rect overlay palette (POBIMStudio rect-select: dark under-stroke + white line)
_WHITE = (1.0, 1.0, 1.0, 1.0)
_DARK = (0.0, 0.0, 0.0, 0.8)

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
_POLY_CLOSE_PX = 8.0      # click within this of the first vertex closes
_LASSO_MIN_SPACING = 4.0  # px between appended lasso points
_LASSO_CAP = 512          # decimate the stroke beyond this many points

_TOOL_ITEMS = ('RECT', 'LASSO', 'POLYGON', 'BRUSH', 'SPHERE', 'BOX')
_TOOL_KEYS = {'R': 'RECT', 'L': 'LASSO', 'P': 'POLYGON',
              'B': 'BRUSH', 'S': 'SPHERE', 'C': 'BOX'}
_TOOL_LABELS = {
    'RECT': 'กรอบ Rect', 'LASSO': 'บ่วง Lasso', 'POLYGON': 'หลายเหลี่ยม Polygon',
    'BRUSH': 'พู่กัน Brush', 'SPHERE': 'ทรงกลม Sphere', 'BOX': 'กล่อง Box'}
_TOOL_HINTS = {
    'RECT': 'ลาก=เลือกกรอบ',
    'LASSO': 'ลากวาดเส้น=เลือก',
    'POLYGON': 'คลิกเพิ่มจุด · Enter/คลิกจุดแรก=ปิด · BkSp=ลบจุด',
    'BRUSH': 'ลากทาสี · [ ] ปรับขนาด',
    'SPHERE': 'คลิก=เลือกทรงกลม · [ ] ปรับรัศมี',
    'BOX': 'คลิกสองมุม=กล่อง (local space)'}


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


class POBIM_OT_edit_splats(bpy.types.Operator):
    """แก้ไข splat: เลือก / ซ่อน / ลบ ทีละจุด
(R กรอบ | L บ่วง | P หลายเหลี่ยม | B พู่กัน | S ทรงกลม | C กล่อง | Shift = เพิ่ม | Ctrl = ลบออก | A = เลือกทั้งหมด | Shift+A/Alt+A = ไม่เลือก | Ctrl+I = สลับเลือก | H = ซ่อน | Alt+H = คืน | X/Del = ลบ | Ctrl+Z = undo | Ctrl+Shift+Z = redo | Esc/คลิกขวา = ออก)"""
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

        # subsample for the nearest-center pick fallback only
        if count > PICK_SUBSAMPLE:
            sel = np.random.default_rng(0).permutation(count)[:PICK_SUBSAMPLE]
            self._pick_local = positions[sel]
        else:
            self._pick_local = positions
        self._world = None
        self._world_matrix = None

        # active tool + per-gesture state
        self._tool = getattr(context.scene, 'pobim_splat_edit_tool', 'RECT')
        self._mouse = (event.mouse_region_x, event.mouse_region_y)
        self._brush_radius = _BRUSH_DEFAULT
        self._sphere_radius = _SPHERE_DEFAULT

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
        # SPHERE
        self._sphere_center = None
        # BOX (corners kept in splat-LOCAL space)
        self._box_first = None
        self._box_hover = None
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

    def _set_status(self, context):
        n_sel = self._state.num_selected
        n_hid = self._state.num_hidden
        n_del = self._state.num_deleted
        count = self._count
        tool = _TOOL_LABELS.get(self._tool, self._tool)
        hint = _TOOL_HINTS.get(self._tool, '')
        context.workspace.status_text_set(
            f'Edit Splats [{tool}] — Selected {n_sel:,} / {count:,} · '
            f'Hidden {n_hid:,} · Deleted {n_del:,} | {hint} | '
            f'R/L/P/B/S/C เครื่องมือ · Shift เพิ่ม Ctrl ลบออก | '
            f'A/Shift+A/Ctrl+I | H ซ่อน Alt+H คืน | X ลบ | Ctrl+Z undo | Esc ออก')

    def _finish(self, context):
        POBIM_OT_edit_splats._running = False
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
        """Serialize state to the object and refresh status + viewport."""
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

    def _to_local(self, world):
        obj = bpy.data.objects.get(self._obj_name)
        if obj is None:
            return np.asarray(world, np.float32)
        m = np.linalg.inv(np.array(obj.matrix_world, np.float64))
        return (m[:3, :3] @ np.asarray(world, np.float64) + m[:3, 3]).astype(np.float32)

    def _world_points(self, obj):
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
        self._box_first = None
        self._box_hover = None

    def _switch_tool(self, context, tool):
        if tool == self._tool:
            return
        self._reset_gesture()
        # only SPHERE/BOX surface-pick: release the depth offscreen early
        # (polish — _finish frees it anyway)
        if self._tool in {'SPHERE', 'BOX'} and tool not in {'SPHERE', 'BOX'} \
                and self._depth_offs is not None:
            self._depth_offs.free()
            self._depth_offs = None
            self._depth_persp = None
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

    # --- input dispatch ---------------------------------------------------

    def _on_mousemove(self, context):
        x, y = self._mouse
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
            self._sphere_center = self._pick_world(context, x, y)
        elif t == 'BOX':
            if self._box_first is not None:
                self._box_hover = self._pick_world(context, x, y)

    def _on_leftmouse(self, context, event):
        x, y = event.mouse_region_x, event.mouse_region_y
        self._mouse = (x, y)
        t = self._tool
        press = event.value == 'PRESS'
        release = event.value == 'RELEASE'

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
                self._sphere_center = self._pick_world(context, x, y)
                self._commit_sphere(context, self._op_from_event(event))
        elif t == 'BOX':
            if press:
                w = self._pick_world(context, x, y)
                if w is not None:
                    if self._box_first is None:
                        self._box_first = self._to_local(w)
                        self._box_hover = w
                    else:
                        self._commit_box(context, self._box_first,
                                         self._to_local(w), self._op_from_event(event))
                        self._box_first = None
                        self._box_hover = None
        if context.area:
            context.area.tag_redraw()
        return {'RUNNING_MODAL'}

    # --- modal ------------------------------------------------------------

    def modal(self, context, event):
        if (event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE',
                           'TRACKPADPAN', 'TRACKPADZOOM', 'MOUSEROTATE'} or
                (event.type.startswith('NUMPAD') and event.type != 'NUMPAD_ENTER')):
            return {'PASS_THROUGH'}  # keep viewport navigation working

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

            # tool switch keys — cancel any in-progress gesture
            if et in _TOOL_KEYS and not event.ctrl:
                self._switch_tool(context, _TOOL_KEYS[et])
                return {'RUNNING_MODAL'}

            # brush / sphere radius (matches the TS bracket keys)
            if et in {'LEFT_BRACKET', 'RIGHT_BRACKET'}:
                f = 0.8 if et == 'LEFT_BRACKET' else 1.25
                if self._tool == 'BRUSH':
                    self._brush_radius = float(np.clip(
                        self._brush_radius * f, _BRUSH_MIN, _BRUSH_MAX))
                    if context.area:
                        context.area.tag_redraw()
                    return {'RUNNING_MODAL'}
                if self._tool == 'SPHERE':
                    self._sphere_radius = max(self._sphere_radius * f, _SPHERE_MIN)
                    if context.area:
                        context.area.tag_redraw()
                    return {'RUNNING_MODAL'}

            # close the polygon with Enter
            if et in {'RET', 'NUMPAD_ENTER'}:
                if self._tool == 'POLYGON' and len(self._poly_pts) >= 3:
                    self._commit_polyshape(context, self._poly_pts,
                                           self._op_from_event(event), 'Polygon Select')
                    self._poly_pts = []
                    if context.area:
                        context.area.tag_redraw()
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
            if et in {'X', 'DEL'}:
                self._apply(context, 'Delete Selected', self._state.delete_selected)
                return {'RUNNING_MODAL'}
            if et == 'Z' and event.ctrl:
                if event.shift:
                    self._history.redo(self._state)
                else:
                    self._history.undo(self._state)
                self._persist(context)
                return {'RUNNING_MODAL'}

            if et in {'ESC', 'RIGHTMOUSE'}:
                # cancel an in-progress gesture first; only exit when idle
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
                if self._tool == 'BOX' and self._box_first is not None:
                    self._box_first = None
                    self._box_hover = None
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

        return {'RUNNING_MODAL'}

    # --- overlay ----------------------------------------------------------

    def _lines(self, coords, color, width, strip=True):
        gpu.state.line_width_set(width)
        batch = batch_for_shader(
            self._shader, 'LINE_STRIP' if strip else 'LINES', {'pos': coords})
        self._shader.uniform_float('color', color)
        batch.draw(self._shader)
        gpu.state.line_width_set(1.0)

    def _stroke2(self, coords, strip=True):
        """dark under-stroke + crisp white line (the rect-select look)."""
        self._lines(coords, _DARK, 2.0, strip)
        self._lines(coords, _WHITE, 1.5, strip)

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
            t = self._tool
            if t == 'RECT':
                self._draw_rect()
            elif t == 'LASSO':
                self._draw_lasso()
            elif t == 'POLYGON':
                self._draw_polygon()
            elif t == 'BRUSH':
                self._draw_brush()
            elif t == 'SPHERE':
                self._draw_sphere(context)
            elif t == 'BOX':
                self._draw_box(context)
            gpu.state.blend_set('NONE')
        except Exception as e:
            print(f'[pobim_splats] edit overlay error: {e}')

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
        # hint the closing edge back to the start
        coords = coords + [coords[0]]
        self._stroke2(coords)

    def _draw_polygon(self):
        pts = [(float(x), float(y)) for x, y in self._poly_pts]
        if pts:
            rubber = pts + [(float(self._mouse[0]), float(self._mouse[1]))]
            if len(pts) >= 2:
                self._stroke2(pts)
            self._stroke2(rubber)
            # first-vertex ring so the user can see the close target
            self._lines(_circle(pts[0][0], pts[0][1], _POLY_CLOSE_PX),
                        _DARK, 2.0)
            self._lines(_circle(pts[0][0], pts[0][1], _POLY_CLOSE_PX),
                        _WHITE, 1.5)

    def _draw_brush(self):
        cx, cy = self._mouse
        self._lines(_circle(cx, cy, self._brush_radius), _DARK, 2.0)
        self._lines(_circle(cx, cy, self._brush_radius), _WHITE, 1.5)

    def _draw_sphere(self, context):
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
        self._lines(_circle(c[0], c[1], r), _WHITE, 1.5)

    def _draw_box(self, context):
        if self._box_first is None or self._box_hover is None:
            return
        obj = bpy.data.objects.get(self._obj_name)
        if obj is None:
            return
        m = np.array(obj.matrix_world, np.float64)
        second_local = self._to_local(self._box_hover)
        bmin = np.minimum(self._box_first, second_local)
        bmax = np.maximum(self._box_first, second_local)
        cpx = []
        for c in box_corners(bmin, bmax):
            w = m[:3, :3] @ np.asarray(c, np.float64) + m[:3, 3]
            cpx.append(self._px(context, w))
        for e0, e1 in _BOX_EDGES:
            if cpx[e0] is not None and cpx[e1] is not None:
                self._stroke2([cpx[e0], cpx[e1]])


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
        try:
            n = splat_export.export_ply(
                source_path, self.filepath, keep_mask, source_indices)
        except Exception as e:
            self.report({'ERROR'}, f'ส่งออกไม่สำเร็จ: {e}')
            return {'CANCELLED'}
        self.report({'INFO'}, f'ส่งออก {n:,} splats → {self.filepath}')
        return {'FINISHED'}


CLASSES = (POBIM_OT_edit_splats, POBIM_OT_export_ply)

# Measure suite: Distance / Area / Volume on splats, mirroring POBIMStudio's
# measure tools (src/tools/measure-*.ts) in behavior, math and styling.
#
# UX (matches POBIMStudio):
# - left click adds a point (or grabs an existing one to move it)
# - RIGHT CLICK finishes the current chain/polygon but STAYS in the tool
# - Esc exits (finishing anything in progress); measurements PERSIST on the
#   object (stored in local space in a JSON custom property, so they follow
#   the splat and survive tool re-entry and .blend save/load)
# - X/Delete removes the point under the cursor; Enter/S opens the
#   scale-to-real-distance dialog for the last distance segment
# - D/A/V switches Distance/Area/Volume; M switches Surface/Centers picking
#
# Math mirrors the main project: polygon area = 0.5*|Σ (pj−p0)×(pj+1−p0)|
# with perimeter; volume boxes are the bounds of two picked corners with
# volume = extents × world scale.

import json

import blf
import bpy
import gpu
import numpy as np
from bpy.props import FloatProperty, FloatVectorProperty, StringProperty
from gpu_extras.batch import batch_for_shader
from mathutils import Matrix

from . import splat_gpu
from .measure_math import (
    BOX_EDGES as _BOX_EDGES, box_corners, pick_nearest, polygon_area,
    polygon_perimeter, project_to_pixels, scale_about_point_matrix,
    unproject_pixel)

PICK_SUBSAMPLE = 400_000
PICK_RADIUS = 25.0
POINT_HIT_RADIUS = 10.0

# POBIMStudio measure palette (src/ui/scss/tool.scss)
_ORANGE = (1.0, 0.647, 0.0, 1.0)          # #ffa500 active
_WHITE = (1.0, 1.0, 1.0, 1.0)
_DARK = (0.0, 0.0, 0.0, 0.8)
_DARK_ACTIVE = (0.0, 0.0, 0.0, 0.9)
_CHIP_BG = (1.0, 0.647, 0.0, 0.95)
_CHIP_TEXT = (0.08, 0.08, 0.08, 1.0)
_FILL = (1.0, 0.647, 0.0, 0.18)

_KINDS = ('DISTANCE', 'AREA', 'VOLUME')
_MEASURE_PROP = 'pobim_measures'


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


def _circle(cx, cy, r, segments=24):
    t = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=True)
    return [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in t]


class MeasureStore:
    """Persisted measurements on a splat object, in LOCAL space.

    Resolved by splat uid (not name) so renaming the object mid-session
    cannot silently drop measurements on save.
    """

    def __init__(self, obj):
        self.uid = obj.pobim_splat_uid
        try:
            data = json.loads(obj.get(_MEASURE_PROP, '') or '{}')
        except Exception:
            data = {}
        self.chains = [[np.array(p, np.float32) for p in c]
                       for c in data.get('distance', [])]
        self.polygons = [[np.array(p, np.float32) for p in c]
                         for c in data.get('area', [])]
        self.boxes = [[np.array(p, np.float32) for p in c]
                      for c in data.get('volume', [])]

    def save(self):
        obj = next((o for o in bpy.data.objects
                    if o.pobim_splat_uid == self.uid), None)
        if obj is None:
            return
        obj[_MEASURE_PROP] = json.dumps({
            'distance': [[[float(v) for v in p] for p in c] for c in self.chains],
            'area': [[[float(v) for v in p] for p in c] for c in self.polygons],
            'volume': [[[float(v) for v in p] for p in c] for c in self.boxes],
        })

    def all_points(self):
        """(kind, item_index, point_index, local_point) for every point."""
        for ci, chain in enumerate(self.chains):
            for pi, p in enumerate(chain):
                yield ('DISTANCE', ci, pi, p)
        for ci, poly in enumerate(self.polygons):
            for pi, p in enumerate(poly):
                yield ('AREA', ci, pi, p)
        for ci, box in enumerate(self.boxes):
            for pi, p in enumerate(box):
                yield ('VOLUME', ci, pi, p)


class POBIM_OT_measure_scale(bpy.types.Operator):
    """เครื่องมือวัดบน splat: ระยะ / พื้นที่ / ปริมาตร
(คลิกซ้าย = เพิ่ม/ย้ายจุด | คลิกขวา = จบชุดที่กำลังวัด | D/A/V = สลับชนิด | M = โหมดจับจุด | X = ลบจุด | Enter/S = ปรับสเกล | Esc = ออก)"""
    bl_idname = 'pobim_splats.measure_scale'
    bl_label = 'Measure'

    uid: StringProperty()

    _running = False

    # --- lifecycle --------------------------------------------------------

    def invoke(self, context, event):
        if POBIM_OT_measure_scale._running:
            self.report({'WARNING'}, 'เครื่องมือวัดกำลังทำงานอยู่แล้ว')
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

        self._obj_name = obj.name
        n = positions.shape[0]
        if n > PICK_SUBSAMPLE:
            sel = np.random.default_rng(0).permutation(n)[:PICK_SUBSAMPLE]
            self._local = positions[sel]
        else:
            self._local = positions
        self._world = None
        self._world_matrix = None

        self._store = MeasureStore(obj)
        self._kind = context.scene.pobim_splat_measure_kind
        self._mode = context.scene.pobim_splat_measure_mode
        self._current = []         # in-progress points (world space)
        self._hover = None
        self._hover_hit = None     # (kind, item, index) of grabbable point
        self._grab = None          # (kind, item, index, original_local)
        self._mouse = (event.mouse_region_x, event.mouse_region_y)

        self._depth_offs = None
        self._depth_persp = None

        self._shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        self._handle = bpy.types.SpaceView3D.draw_handler_add(
            self._draw_overlay, (context,), 'WINDOW', 'POST_PIXEL')
        POBIM_OT_measure_scale._running = True
        self._set_status(context)
        context.window_manager.modal_handler_add(self)
        context.area.tag_redraw()
        return {'RUNNING_MODAL'}

    def _set_status(self, context):
        kind = {'DISTANCE': 'ระยะ', 'AREA': 'พื้นที่', 'VOLUME': 'ปริมาตร'}[self._kind]
        mode = 'Surface' if self._mode == 'SURFACE' else 'Centers'
        context.workspace.status_text_set(
            f'วัด{kind} [{mode}] — คลิกซ้าย: เพิ่ม/ย้ายจุด | คลิกขวา: จบชุดนี้ | '
            f'D/A/V: ชนิด | M: โหมด | X: ลบจุด | Enter/S: ปรับสเกล | Esc: ออก')

    def _finish(self, context):
        self._commit_current()
        self._store.save()
        POBIM_OT_measure_scale._running = False
        if self._handle is not None:
            bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
            self._handle = None
        if self._depth_offs is not None:
            self._depth_offs.free()
            self._depth_offs = None
        context.workspace.status_text_set(None)
        if context.area:
            context.area.tag_redraw()

    def cancel(self, context):
        self._finish(context)

    # --- transforms -------------------------------------------------------

    def _matrix(self):
        obj = bpy.data.objects.get(self._obj_name)
        if obj is None:
            return np.eye(4, dtype=np.float64)
        return np.array(obj.matrix_world, np.float64)

    def _to_world(self, local):
        m = self._matrix()
        return (m[:3, :3] @ np.asarray(local, np.float64) + m[:3, 3]).astype(np.float32)

    def _to_local(self, world):
        m = np.linalg.inv(self._matrix())
        return (m[:3, :3] @ np.asarray(world, np.float64) + m[:3, 3]).astype(np.float32)

    def _world_scale(self):
        m = self._matrix()
        return np.array([np.linalg.norm(m[:3, k]) for k in range(3)], np.float64)

    # --- picking ----------------------------------------------------------

    def _world_points(self, obj):
        matrix = np.array(obj.matrix_world, np.float32)
        if self._world is None or not np.array_equal(matrix, self._world_matrix):
            self._world_matrix = matrix
            self._world = self._local @ matrix[:3, :3].T + matrix[:3, 3]
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

    def _pick_world(self, context):
        """Pick a world point under the mouse with the current mode."""
        obj = bpy.data.objects.get(self._obj_name)
        rv3d = context.region_data
        region = context.region
        if obj is None or rv3d is None or region is None:
            return None
        persp = np.array(rv3d.perspective_matrix, np.float32)
        mx, my = self._mouse

        if self._mode == 'SURFACE':
            self._ensure_depth_map(context)
            depth = splat_gpu.read_depth_pixel(self._depth_offs, mx, my)
            if depth is not None:
                return unproject_pixel(
                    persp, mx, my, depth, region.width, region.height)

        world = self._world_points(obj)
        idx = pick_nearest(persp, world, region.width, region.height,
                           (mx, my), PICK_RADIUS)
        return None if idx < 0 else world[idx].copy()

    def _hit_test_points(self, context):
        """Find a committed point near the cursor -> (kind, item, index)."""
        rv3d = context.region_data
        region = context.region
        if rv3d is None or region is None:
            return None
        keys = []
        locals_ = []
        for kind, ci, pi, local in self._store.all_points():
            keys.append((kind, ci, pi))
            locals_.append(local)
        if not keys:
            return None

        m = self._matrix()
        world = (np.asarray(locals_, np.float64) @ m[:3, :3].T + m[:3, 3]).astype(np.float32)
        persp = np.array(rv3d.perspective_matrix, np.float32)
        px, _, valid = project_to_pixels(persp, world, region.width, region.height)
        d2 = ((px - np.asarray(self._mouse, np.float32)) ** 2).sum(axis=1)
        d2[~valid] = np.inf
        best = int(np.argmin(d2))
        if d2[best] > POINT_HIT_RADIUS * POINT_HIT_RADIUS:
            return None
        return keys[best]

    def _update_hover(self, context):
        self._hover = self._pick_world(context)
        # no grabbing while a new chain/polygon is in progress — a click
        # near an old point must add the intended vertex, not hijack it
        self._hover_hit = (None if (self._grab or self._current)
                           else self._hit_test_points(context))

    # --- editing ----------------------------------------------------------

    def _commit_current(self):
        pts = [self._to_local(p) for p in self._current]
        if self._kind == 'DISTANCE' and len(pts) >= 2:
            self._store.chains.append(pts)
        elif self._kind == 'AREA' and len(pts) >= 3:
            self._store.polygons.append(pts)
        # VOLUME commits immediately at 2 points in _add_point
        self._current = []

    def _add_point(self, world):
        self._current.append(world.copy())
        if self._kind == 'VOLUME' and len(self._current) == 2:
            self._store.boxes.append([self._to_local(p) for p in self._current])
            self._current = []
            self._store.save()

    def _grab_point(self, hit):
        kind, ci, pi = hit
        items = {'DISTANCE': self._store.chains,
                 'AREA': self._store.polygons,
                 'VOLUME': self._store.boxes}[kind]
        self._grab = (kind, ci, pi, items[ci][pi].copy())

    def _set_grabbed(self, local):
        kind, ci, pi, _orig = self._grab
        items = {'DISTANCE': self._store.chains,
                 'AREA': self._store.polygons,
                 'VOLUME': self._store.boxes}[kind]
        items[ci][pi] = local

    def _delete_hit(self, hit):
        kind, ci, pi = hit
        if kind == 'DISTANCE':
            chain = self._store.chains[ci]
            chain.pop(pi)
            if len(chain) < 2:
                self._store.chains.pop(ci)
        elif kind == 'AREA':
            poly = self._store.polygons[ci]
            poly.pop(pi)
            if len(poly) < 3:
                self._store.polygons.pop(ci)
        else:
            self._store.boxes.pop(ci)
        self._store.save()

    def _last_segment(self):
        """Last distance segment (current chain first, then committed)."""
        if len(self._current) >= 2 and self._kind == 'DISTANCE':
            return self._current[-2], self._current[-1]
        if self._store.chains:
            chain = self._store.chains[-1]
            if len(chain) >= 2:
                return self._to_world(chain[-2]), self._to_world(chain[-1])
        return None

    def _open_scale_dialog(self, context):
        seg = self._last_segment()
        if seg is None:
            self.report({'WARNING'}, 'ยังไม่มีช่วงระยะให้ปรับสเกล')
            return False
        p1, p2 = seg
        measured = float(np.linalg.norm(np.asarray(p2) - np.asarray(p1)))
        if measured < 1e-9:
            self.report({'ERROR'}, 'ช่วงล่าสุดสั้นเกินไป วัดระยะไม่ได้')
            return False
        obj = bpy.data.objects.get(self._obj_name)
        uid = obj.pobim_splat_uid if obj else ''
        self._finish(context)
        try:
            bpy.ops.pobim_splats.apply_scale(
                'INVOKE_DEFAULT', uid=uid,
                measured=measured, target=measured,
                pivot=tuple(float(v) for v in p1))
        except Exception as e:
            self.report({'ERROR'}, f'เปิดหน้าต่างปรับสเกลไม่สำเร็จ '
                                   f'(ระยะที่วัดได้ {measured:.3f} m): {e}')
        return True

    # --- modal ------------------------------------------------------------

    def modal(self, context, event):
        if event.type == 'MOUSEMOVE':
            self._mouse = (event.mouse_region_x, event.mouse_region_y)
            self._update_hover(context)
            if self._grab is not None and self._hover is not None:
                self._set_grabbed(self._to_local(self._hover))  # live preview
            if context.area:
                context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        if (event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE',
                           'TRACKPADPAN', 'TRACKPADZOOM', 'MOUSEROTATE'} or
                (event.type.startswith('NUMPAD') and event.type != 'NUMPAD_ENTER')):
            return {'PASS_THROUGH'}  # keep viewport navigation working

        if event.value == 'PRESS' and event.type in {'D', 'A', 'V'}:
            self._commit_current()
            self._kind = {'D': 'DISTANCE', 'A': 'AREA', 'V': 'VOLUME'}[event.type]
            context.scene.pobim_splat_measure_kind = self._kind
            self._set_status(context)
            if context.area:
                context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        if event.type == 'M' and event.value == 'PRESS':
            self._mode = 'CENTERS' if self._mode == 'SURFACE' else 'SURFACE'
            context.scene.pobim_splat_measure_mode = self._mode
            self._update_hover(context)
            self._set_status(context)
            if context.area:
                context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        if event.value == 'PRESS' and event.type in {'X', 'DEL'}:
            if self._hover_hit is not None:
                self._delete_hit(self._hover_hit)
                self._hover_hit = None
                if context.area:
                    context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            if self._grab is not None:
                # drop the grabbed point at the picked position
                if self._hover is not None:
                    self._set_grabbed(self._to_local(self._hover))
                    self._store.save()
                self._grab = None
            elif self._hover_hit is not None:
                self._grab_point(self._hover_hit)
            elif self._hover is not None:
                self._add_point(self._hover)
            if context.area:
                context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        if event.type in {'RET', 'NUMPAD_ENTER', 'S'} and event.value == 'PRESS':
            if self._open_scale_dialog(context):
                return {'FINISHED'}
            return {'RUNNING_MODAL'}

        if event.type == 'RIGHTMOUSE' and event.value == 'PRESS':
            # finish the current chain/polygon/grab but stay in the tool
            if self._grab is not None:
                kind, ci, pi, orig = self._grab
                self._grab = None
                self._set_grabbed_restore(kind, ci, pi, orig)
            else:
                self._commit_current()
                self._store.save()
            self._set_status(context)
            if context.area:
                context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        if event.type == 'ESC' and event.value == 'PRESS':
            if self._grab is not None:
                kind, ci, pi, orig = self._grab
                self._grab = None
                self._set_grabbed_restore(kind, ci, pi, orig)
                if context.area:
                    context.area.tag_redraw()
                return {'RUNNING_MODAL'}
            self._finish(context)
            return {'FINISHED'}

        return {'RUNNING_MODAL'}

    def _set_grabbed_restore(self, kind, ci, pi, orig):
        items = {'DISTANCE': self._store.chains,
                 'AREA': self._store.polygons,
                 'VOLUME': self._store.boxes}[kind]
        try:
            items[ci][pi] = orig
        except Exception:
            pass

    # --- overlay ----------------------------------------------------------

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

    def _segment(self, a, b, active):
        self._lines([a, b], _DARK_ACTIVE if active else _DARK, 8.0 if active else 7.0)
        self._lines([a, b], _ORANGE if active else _WHITE, 4.0 if active else 2.5)

    def _dot(self, p, active):
        r = 7.0 if active else 5.0
        ring = _circle(p[0], p[1], r)
        self._fill([p] + ring, _ORANGE if active else _WHITE)
        self._lines(ring, _DARK_ACTIVE if active else _DARK, 2.5 if active else 2.0)

    def _chip(self, x, y, text):
        font = 0
        blf.size(font, 13.0)
        tw, th = blf.dimensions(font, text)
        pad = 6.0
        x0, y0 = x - tw / 2 - pad, y - th / 2 - pad * 0.7
        x1, y1 = x + tw / 2 + pad, y + th / 2 + pad * 0.7
        self._fill([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], _CHIP_BG)
        blf.position(font, x - tw / 2, y - th / 2 + 1.0, 0)
        blf.color(font, *_CHIP_TEXT)
        blf.draw(font, text)

    def _draw_polyline(self, context, world_pts, active, closed=False, fill=False):
        pts_px = [self._px(context, p) for p in world_pts]
        visible = [p for p in pts_px if p is not None]
        if fill and len(visible) >= 3:
            self._fill(visible, _FILL)
        for i in range(len(world_pts) - 1):
            if pts_px[i] is not None and pts_px[i + 1] is not None:
                self._segment(pts_px[i], pts_px[i + 1], active)
        if closed and len(world_pts) > 2 and pts_px[0] is not None and pts_px[-1] is not None:
            self._segment(pts_px[-1], pts_px[0], active)
        return pts_px

    def _draw_overlay(self, context):
        try:
            gpu.state.blend_set('ALPHA')
            self._shader.bind()

            hover_px = None if self._hover is None else self._px(context, self._hover)
            grab_active = self._grab is not None

            # committed distance chains
            for ci, chain in enumerate(self._store.chains):
                world = [self._to_world(p) for p in chain]
                pts_px = self._draw_polyline(context, world, False)
                for i in range(len(world) - 1):
                    a, b = pts_px[i], pts_px[i + 1]
                    if a is None or b is None:
                        continue
                    d = float(np.linalg.norm(world[i + 1] - world[i]))
                    self._chip((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 18,
                               f'{d:.2f} m' if d >= 1 else f'{d:.3f} m')
                for p in pts_px:
                    if p is not None:
                        self._dot(p, False)

            # committed polygons (area)
            for poly in self._store.polygons:
                world = [self._to_world(p) for p in poly]
                pts_px = self._draw_polyline(context, world, False, closed=True, fill=True)
                for p in pts_px:
                    if p is not None:
                        self._dot(p, False)
                pts = np.asarray(world, np.float64)
                area = polygon_area(pts)
                peri = polygon_perimeter(pts)
                center = self._px(context, pts.mean(axis=0))
                if center is not None:
                    self._chip(center[0], center[1],
                               f'{area:.3f} m²  |  P {peri:.3f} m')

            # committed volume boxes
            for box in self._store.boxes:
                a = self._to_world(box[0])
                b = self._to_world(box[1])
                lmin = np.minimum(box[0], box[1])
                lmax = np.maximum(box[0], box[1])
                corners = [self._to_world(c) for c in box_corners(lmin, lmax)]
                cpx = [self._px(context, c) for c in corners]
                for e0, e1 in _BOX_EDGES:
                    if cpx[e0] is not None and cpx[e1] is not None:
                        self._segment(cpx[e0], cpx[e1], False)
                for p in (self._px(context, a), self._px(context, b)):
                    if p is not None:
                        self._dot(p, False)
                extents = (lmax - lmin) * self._world_scale()
                vol = float(abs(extents[0] * extents[1] * extents[2]))
                center = self._px(context, (np.asarray(a) + np.asarray(b)) / 2)
                if center is not None:
                    self._chip(center[0], center[1], f'{vol:.3f} m³')

            # in-progress chain/polygon + rubber band
            if self._current:
                is_area = self._kind == 'AREA'
                cur_px = self._draw_polyline(
                    context, self._current, True,
                    closed=is_area and len(self._current) > 2 and hover_px is None,
                    fill=is_area and len(self._current) >= 3)
                if hover_px is not None and cur_px and cur_px[-1] is not None:
                    self._segment(cur_px[-1], hover_px, True)
                    d = float(np.linalg.norm(self._hover - self._current[-1]))
                    self._chip((cur_px[-1][0] + hover_px[0]) / 2,
                               (cur_px[-1][1] + hover_px[1]) / 2 + 18,
                               f'{d:.2f} m' if d >= 1 else f'{d:.3f} m')
                for p in cur_px:
                    if p is not None:
                        self._dot(p, True)

            # hover indicator: grabbable point ring or pick dot
            if self._hover_hit is not None and not grab_active:
                kind, ci, pi = self._hover_hit
                items = {'DISTANCE': self._store.chains,
                         'AREA': self._store.polygons,
                         'VOLUME': self._store.boxes}[kind]
                p = self._px(context, self._to_world(items[ci][pi]))
                if p is not None:
                    self._lines(_circle(p[0], p[1], 11.0), _ORANGE, 2.5)
            elif hover_px is not None:
                self._dot(hover_px, True)

            gpu.state.blend_set('NONE')
        except Exception as e:
            print(f'[pobim_splats] measure overlay error: {e}')


class POBIM_OT_apply_scale(bpy.types.Operator):
    """ปรับสเกล splat ให้ระยะที่วัดตรงกับระยะจริง (สเกลรอบจุดแรกของช่วงที่วัด)"""
    bl_idname = 'pobim_splats.apply_scale'
    bl_label = 'Set Real Distance'
    bl_options = {'REGISTER', 'UNDO'}

    uid: StringProperty()
    measured: FloatProperty(name='ระยะที่วัดได้ (m)', precision=4)
    target: FloatProperty(name='ระยะจริง (m)', precision=4, min=1e-6, default=1.0)
    pivot: FloatVectorProperty(size=3)

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context):
        layout = self.layout
        row = layout.row()
        row.enabled = False
        row.prop(self, 'measured')
        layout.prop(self, 'target')
        if self.measured > 0:
            layout.label(text=f'สเกล × {self.target / self.measured:.4f}')

    def execute(self, context):
        obj = _find_splat_object(context, self.uid)
        if obj is None:
            self.report({'ERROR'}, 'ไม่พบ object ของ splat')
            return {'CANCELLED'}
        if self.measured <= 0:
            self.report({'ERROR'}, 'ระยะที่วัดไม่ถูกต้อง')
            return {'CANCELLED'}

        factor = self.target / self.measured
        m = scale_about_point_matrix(self.pivot, factor)
        obj.matrix_world = Matrix([tuple(row) for row in m]) @ obj.matrix_world
        splat_gpu.redraw_viewports()
        self.report({'INFO'}, f'ปรับสเกล × {factor:.4f} '
                              f'({self.measured:.3f} m → {self.target:.3f} m)')
        return {'FINISHED'}


class POBIM_OT_clear_measures(bpy.types.Operator):
    """ลบการวัดทั้งหมดของ splat นี้"""
    bl_idname = 'pobim_splats.clear_measures'
    bl_label = 'Clear Measurements'
    bl_options = {'REGISTER', 'UNDO'}

    uid: StringProperty()

    def execute(self, context):
        for obj in bpy.data.objects:
            if obj.pobim_splat_uid == self.uid and _MEASURE_PROP in obj:
                del obj[_MEASURE_PROP]
        splat_gpu.redraw_viewports()
        return {'FINISHED'}


CLASSES = (POBIM_OT_measure_scale, POBIM_OT_apply_scale, POBIM_OT_clear_measures)

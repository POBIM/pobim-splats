# Measure & Scale: click points on a splat as a chain of segments, read the
# measured distances, then scale the splat so the last segment matches a
# typed real-world distance (scaled about that segment's first point).
#
# Two pick modes (M toggles while measuring):
# - SURFACE (default): depth pick against the rendered splat surface — the
#   same idea as POBIMStudio's camera.intersect. Splats are drawn once per
#   view into an offscreen depth map and the pixel under the cursor is
#   unprojected.
# - CENTERS: snap to the nearest gaussian center within PICK_RADIUS px
#   (subsampled to PICK_SUBSAMPLE for speed).
#
# Overlay styling mirrors POBIMStudio's measure tool: a dark under-stroke
# with a white line (orange #ffa500 for the active segment), white/orange
# endpoint dots with dark outlines, and orange distance chips at midpoints.

import blf
import bpy
import gpu
import numpy as np
from bpy.props import FloatProperty, FloatVectorProperty, StringProperty
from gpu_extras.batch import batch_for_shader
from mathutils import Matrix

from . import splat_gpu
from .measure_math import (
    pick_nearest, project_to_pixels, scale_about_point_matrix, unproject_pixel)

PICK_SUBSAMPLE = 400_000
PICK_RADIUS = 25.0

# POBIMStudio measure palette (src/ui/scss/tool.scss)
_ORANGE = (1.0, 0.647, 0.0, 1.0)          # #ffa500 active
_WHITE = (1.0, 1.0, 1.0, 1.0)
_DARK = (0.0, 0.0, 0.0, 0.8)
_DARK_ACTIVE = (0.0, 0.0, 0.0, 0.9)
_CHIP_BG = (1.0, 0.647, 0.0, 0.95)
_CHIP_TEXT = (0.08, 0.08, 0.08, 1.0)


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


class POBIM_OT_measure_scale(bpy.types.Operator):
    """วัดระยะบน splat แบบต่อเนื่อง แล้วปรับสเกลตามระยะจริง
(คลิกซ้าย = เพิ่มจุด | Enter/S = ปรับสเกลจากช่วงล่าสุด | M = สลับโหมด Surface/Centers | คลิกขวา/Esc = จบ)"""
    bl_idname = 'pobim_splats.measure_scale'
    bl_label = 'Measure & Scale'

    uid: StringProperty()

    _running = False

    # --- lifecycle --------------------------------------------------------

    def invoke(self, context, event):
        if POBIM_OT_measure_scale._running:
            self.report({'WARNING'}, 'Measure & Scale กำลังทำงานอยู่แล้ว')
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

        self._mode = context.scene.pobim_splat_measure_mode
        self._points = []          # chained world-space points
        self._hover = None
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
        mode = 'Surface' if self._mode == 'SURFACE' else 'Centers'
        total = self._chain_length()
        total_txt = f' | รวม {total:.3f} m' if total > 0 else ''
        context.workspace.status_text_set(
            f'Measure & Scale [{mode}]{total_txt} — คลิกซ้าย: เพิ่มจุด | '
            f'Enter/S: ปรับสเกลช่วงล่าสุด | M: สลับโหมด | คลิกขวา/Esc: จบ')

    def _finish(self, context):
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
        # Blender terminates modals without an event on file open, undo,
        # area changes and quit — without this the draw handler leaks
        self._finish(context)

    # --- picking ----------------------------------------------------------

    def _chain_length(self):
        return sum(
            float(np.linalg.norm(self._points[i + 1] - self._points[i]))
            for i in range(len(self._points) - 1))

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

    def _update_hover(self, context):
        obj = bpy.data.objects.get(self._obj_name)
        rv3d = context.region_data
        region = context.region
        if obj is None or rv3d is None or region is None:
            self._hover = None
            return
        persp = np.array(rv3d.perspective_matrix, np.float32)
        mx, my = self._mouse

        if self._mode == 'SURFACE':
            self._ensure_depth_map(context)
            depth = splat_gpu.read_depth_pixel(self._depth_offs, mx, my)
            if depth is not None:
                self._hover = unproject_pixel(
                    persp, mx, my, depth, region.width, region.height)
                return
            # background under cursor: fall through to center snapping

        world = self._world_points(obj)
        idx = pick_nearest(persp, world, region.width, region.height,
                           (mx, my), PICK_RADIUS)
        self._hover = None if idx < 0 else world[idx].copy()

    # --- modal ------------------------------------------------------------

    def _open_scale_dialog(self, context):
        if len(self._points) < 2:
            self.report({'WARNING'}, 'ต้องมีอย่างน้อย 2 จุดก่อนปรับสเกล')
            return False
        p1, p2 = self._points[-2], self._points[-1]
        measured = float(np.linalg.norm(p2 - p1))
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

    def modal(self, context, event):
        if event.type == 'MOUSEMOVE':
            self._mouse = (event.mouse_region_x, event.mouse_region_y)
            self._update_hover(context)
            if context.area:
                context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        if (event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE',
                           'TRACKPADPAN', 'TRACKPADZOOM', 'MOUSEROTATE'} or
                (event.type.startswith('NUMPAD') and event.type != 'NUMPAD_ENTER')):
            return {'PASS_THROUGH'}  # keep viewport navigation working

        if event.type == 'M' and event.value == 'PRESS':
            self._mode = 'CENTERS' if self._mode == 'SURFACE' else 'SURFACE'
            context.scene.pobim_splat_measure_mode = self._mode
            self._update_hover(context)
            self._set_status(context)
            if context.area:
                context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            if self._hover is not None:
                self._points.append(self._hover.copy())
                self._set_status(context)
                if context.area:
                    context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        if event.type in {'RET', 'NUMPAD_ENTER', 'S'} and event.value == 'PRESS':
            if self._open_scale_dialog(context):
                return {'FINISHED'}
            return {'RUNNING_MODAL'}

        if event.type in {'RIGHTMOUSE', 'ESC'}:
            self._finish(context)
            return {'CANCELLED'}

        return {'RUNNING_MODAL'}

    # --- overlay ----------------------------------------------------------

    def _px(self, context, world_point):
        rv3d = context.region_data
        region = context.region
        if rv3d is None or region is None:
            return None
        persp = np.array(rv3d.perspective_matrix, np.float32)
        px, _, valid = project_to_pixels(
            persp, world_point[None, :], region.width, region.height)
        return (float(px[0, 0]), float(px[0, 1])) if valid[0] else None

    def _lines(self, coords, color, width):
        gpu.state.line_width_set(width)
        batch = batch_for_shader(self._shader, 'LINE_STRIP', {'pos': coords})
        self._shader.uniform_float('color', color)
        batch.draw(self._shader)
        gpu.state.line_width_set(1.0)

    def _fill(self, coords, color):
        batch = batch_for_shader(self._shader, 'TRI_FAN', {'pos': coords})
        self._shader.uniform_float('color', color)
        batch.draw(self._shader)

    def _segment(self, a, b, active):
        # POBIMStudio look: dark under-stroke + colored top line
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

    def _draw_overlay(self, context):
        try:
            gpu.state.blend_set('ALPHA')
            self._shader.bind()

            pts_px = [self._px(context, p) for p in self._points]
            hover_px = None if self._hover is None else self._px(context, self._hover)

            # committed segments; the rubber-band to the hover point is the
            # active one, otherwise the last committed segment is active
            rubber = hover_px is not None and len(self._points) >= 1
            chips = []
            for i in range(len(self._points) - 1):
                a, b = pts_px[i], pts_px[i + 1]
                if a is None or b is None:
                    continue
                active = (not rubber) and i == len(self._points) - 2
                self._segment(a, b, active)
                dist = float(np.linalg.norm(self._points[i + 1] - self._points[i]))
                chips.append(((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 18, dist, active))

            if rubber and pts_px and pts_px[-1] is not None:
                self._segment(pts_px[-1], hover_px, True)
                dist = float(np.linalg.norm(self._hover - self._points[-1]))
                chips.append(((pts_px[-1][0] + hover_px[0]) / 2,
                              (pts_px[-1][1] + hover_px[1]) / 2 + 18, dist, True))

            for p, is_first in [(pp, i == 0) for i, pp in enumerate(pts_px)]:
                if p is not None:
                    self._dot(p, not rubber and p is pts_px[-1])

            if hover_px is not None:
                self._dot(hover_px, True)

            for cx, cy, dist, active in chips:
                text = f'{dist:.2f} m' if dist >= 1 else f'{dist:.3f} m'
                self._chip(cx, cy, text)

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


CLASSES = (POBIM_OT_measure_scale, POBIM_OT_apply_scale)

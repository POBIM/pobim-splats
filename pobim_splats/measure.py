# Measure & Scale: pick two points on a splat, read the measured distance,
# type the real-world distance, and the splat is scaled uniformly about the
# first picked point.
#
# Picking has no mesh to raycast, so it snaps to gaussian centers: a random
# subsample (≤ PICK_SUBSAMPLE) of the cloud is projected to screen space with
# numpy on every mouse move and the front-most center within PICK_RADIUS px
# of the cursor wins.

import blf
import bpy
import gpu
import numpy as np
from bpy.props import FloatProperty, FloatVectorProperty, StringProperty
from gpu_extras.batch import batch_for_shader
from mathutils import Matrix

from . import splat_gpu
from .measure_math import pick_nearest, project_to_pixels, scale_about_point_matrix

PICK_SUBSAMPLE = 400_000
PICK_RADIUS = 25.0

_ACCENT = (0.15, 0.7, 1.0, 1.0)
_WHITE = (1.0, 1.0, 1.0, 1.0)


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


def _circle_points(cx, cy, r, segments=24):
    t = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    return [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in t]


class POBIM_OT_measure_scale(bpy.types.Operator):
    """วัดระยะจากจุดสองจุดบน splat แล้วปรับสเกลตามระยะจริง (คลิกซ้าย = เลือกจุด, คลิกขวา/Esc = ยกเลิก)"""
    bl_idname = 'pobim_splats.measure_scale'
    bl_label = 'Measure & Scale'

    uid: StringProperty()

    # one instance at a time — a second concurrent modal would stack draw
    # handlers and orphan one of them
    _running = False

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

        self._points = []       # picked world-space points (np arrays)
        self._hover = None      # current snapped world point
        self._mouse = (event.mouse_region_x, event.mouse_region_y)

        self._shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        self._handle = bpy.types.SpaceView3D.draw_handler_add(
            self._draw_overlay, (context,), 'WINDOW', 'POST_PIXEL')
        POBIM_OT_measure_scale._running = True
        context.workspace.status_text_set(
            'Measure & Scale: คลิกซ้าย = เลือกจุด (2 จุด) | คลิกขวา/Esc = ยกเลิก')
        context.window_manager.modal_handler_add(self)
        context.area.tag_redraw()
        return {'RUNNING_MODAL'}

    def _finish(self, context):
        POBIM_OT_measure_scale._running = False
        if self._handle is not None:
            bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
            self._handle = None
        context.workspace.status_text_set(None)
        if context.area:
            context.area.tag_redraw()

    def cancel(self, context):
        # Blender terminates modals without an event on file open, undo,
        # area changes and quit — without this the draw handler leaks
        self._finish(context)

    def _world_points(self, obj):
        matrix = np.array(obj.matrix_world, np.float32)
        if self._world is None or not np.array_equal(matrix, self._world_matrix):
            self._world_matrix = matrix
            self._world = self._local @ matrix[:3, :3].T + matrix[:3, 3]
        return self._world

    def _update_hover(self, context):
        obj = bpy.data.objects.get(self._obj_name)
        rv3d = context.region_data
        region = context.region
        if obj is None or rv3d is None or region is None:
            self._hover = None
            return
        persp = np.array(rv3d.perspective_matrix, np.float32)
        world = self._world_points(obj)
        idx = pick_nearest(persp, world, region.width, region.height,
                           self._mouse, PICK_RADIUS)
        self._hover = None if idx < 0 else world[idx].copy()

    def modal(self, context, event):
        if event.type == 'MOUSEMOVE':
            self._mouse = (event.mouse_region_x, event.mouse_region_y)
            self._update_hover(context)
            if context.area:
                context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        if (event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE',
                           'TRACKPADPAN', 'TRACKPADZOOM', 'MOUSEROTATE'} or
                event.type.startswith('NUMPAD')):
            return {'PASS_THROUGH'}  # keep viewport navigation working

        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            if self._hover is not None:
                self._points.append(self._hover.copy())
                if len(self._points) == 2:
                    p1, p2 = self._points
                    measured = float(np.linalg.norm(p2 - p1))
                    self._finish(context)
                    if measured < 1e-9:
                        self.report({'ERROR'}, 'จุดสองจุดซ้อนกัน วัดระยะไม่ได้')
                        return {'CANCELLED'}
                    obj = bpy.data.objects.get(self._obj_name)
                    uid = obj.pobim_splat_uid if obj else ''
                    try:
                        bpy.ops.pobim_splats.apply_scale(
                            'INVOKE_DEFAULT', uid=uid,
                            measured=measured, target=measured,
                            pivot=tuple(float(v) for v in p1))
                    except Exception as e:
                        self.report({'ERROR'}, f'เปิดหน้าต่างปรับสเกลไม่สำเร็จ '
                                               f'(ระยะที่วัดได้ {measured:.3f} m): {e}')
                    return {'FINISHED'}
            return {'RUNNING_MODAL'}

        if event.type in {'RIGHTMOUSE', 'ESC'}:
            self._finish(context)
            return {'CANCELLED'}

        return {'RUNNING_MODAL'}

    # --- overlay ---------------------------------------------------------

    def _px(self, context, world_point):
        rv3d = context.region_data
        region = context.region
        if rv3d is None or region is None:
            return None
        persp = np.array(rv3d.perspective_matrix, np.float32)
        px, _, valid = project_to_pixels(
            persp, world_point[None, :], region.width, region.height)
        return (float(px[0, 0]), float(px[0, 1])) if valid[0] else None

    def _draw_lines(self, coords, color, width=2.0):
        gpu.state.line_width_set(width)
        batch = batch_for_shader(self._shader, 'LINES', {'pos': coords})
        self._shader.uniform_float('color', color)
        batch.draw(self._shader)
        gpu.state.line_width_set(1.0)

    def _draw_overlay(self, context):
        try:
            gpu.state.blend_set('ALPHA')
            self._shader.bind()

            anchors = [self._px(context, p) for p in self._points]
            hover_px = None if self._hover is None else self._px(context, self._hover)

            if hover_px is not None:
                ring = _circle_points(hover_px[0], hover_px[1], 8.0)
                segments = [p for i in range(len(ring))
                            for p in (ring[i], ring[(i + 1) % len(ring)])]
                self._draw_lines(segments, _ACCENT)

            for a in anchors:
                if a is not None:
                    cross = [(a[0] - 6, a[1]), (a[0] + 6, a[1]),
                             (a[0], a[1] - 6), (a[0], a[1] + 6)]
                    self._draw_lines(cross, _WHITE)

            # rubber-band line: p1 -> (p2 | hover)
            endpoint_world = None
            if len(self._points) >= 1:
                endpoint_world = self._hover
            if len(self._points) >= 1 and endpoint_world is not None and anchors[0] is not None:
                end_px = self._px(context, endpoint_world)
                if end_px is not None:
                    self._draw_lines([anchors[0], end_px], _ACCENT)
                    dist = float(np.linalg.norm(endpoint_world - self._points[0]))
                    mx = (anchors[0][0] + end_px[0]) * 0.5
                    my = (anchors[0][1] + end_px[1]) * 0.5
                    font = 0
                    blf.size(font, 14.0)
                    blf.color(font, 1.0, 1.0, 1.0, 1.0)
                    blf.position(font, mx + 8, my + 8, 0)
                    blf.draw(font, f'{dist:.3f} m')

            gpu.state.blend_set('NONE')
        except Exception as e:
            print(f'[pobim_splats] measure overlay error: {e}')


class POBIM_OT_apply_scale(bpy.types.Operator):
    """ปรับสเกล splat ให้ระยะที่วัดตรงกับระยะจริง (สเกลรอบจุดแรกที่เลือก)"""
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

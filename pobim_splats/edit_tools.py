# Per-splat editing: a modal rect-select tool with a tool-local undo stack,
# plus a .ply export of the surviving splats. Mirrors POBIMStudio's editing
# model (State bit flags + EditHistory) and copies measure.py's modal
# lifecycle (running guard, cancel(), status text, POST_PIXEL overlay,
# PASS_THROUGH viewport navigation).

import bpy
import gpu
import numpy as np
from bpy.props import StringProperty
from bpy_extras.io_utils import ExportHelper
from gpu_extras.batch import batch_for_shader

from . import splat_export, splat_gpu
from .measure_math import project_to_pixels
from .splat_state import EditHistory, SplatState, State

_STATE_PROP = 'pobim_splat_state'

# rect overlay palette (POBIMStudio rect-select: dark under-stroke + white line)
_WHITE = (1.0, 1.0, 1.0, 1.0)
_DARK = (0.0, 0.0, 0.0, 0.8)

# project at most this many rows per matmul (bounds the temporary world buffer)
_PROJ_CHUNK = 1_000_000


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


class POBIM_OT_edit_splats(bpy.types.Operator):
    """แก้ไข splat: เลือก / ซ่อน / ลบ ทีละจุด
(ลากเมาส์ = เลือกกรอบ | Shift = เพิ่ม | Ctrl = ลบออกจากที่เลือก | A = เลือกทั้งหมด | Shift+A/Alt+A = ไม่เลือก | Ctrl+I = สลับเลือก | H = ซ่อน | Alt+H = คืน | X/Del = ลบ | Ctrl+Z = undo | Ctrl+Shift+Z = redo | Esc/คลิกขวา = ออก)"""
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

        self._dragging = False
        self._drag_start = None
        self._drag_end = None

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
        context.workspace.status_text_set(
            f'Edit Splats — Selected {n_sel:,} / {count:,} · Hidden {n_hid:,} · '
            f'Deleted {n_del:,} | ลาก=เลือก Shift=เพิ่ม Ctrl=ลบออก | '
            f'A/Shift+A/Ctrl+I | H ซ่อน Alt+H คืน | X ลบ | Ctrl+Z undo | Esc ออก')

    def _finish(self, context):
        POBIM_OT_edit_splats._running = False
        if getattr(self, '_handle', None) is not None:
            bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
            self._handle = None
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
        # never (re)select hidden or deleted splats
        inside &= (self._state.flags & (State.HIDDEN | State.DELETED)) == 0
        indices = np.nonzero(inside)[0]
        self._apply(context, 'Rect Select',
                    lambda: self._state.select_indices(indices, op))

    # --- modal ------------------------------------------------------------

    def modal(self, context, event):
        if (event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE',
                           'TRACKPADPAN', 'TRACKPADZOOM', 'MOUSEROTATE'} or
                (event.type.startswith('NUMPAD') and event.type != 'NUMPAD_ENTER')):
            return {'PASS_THROUGH'}  # keep viewport navigation working

        if event.type == 'MOUSEMOVE':
            if self._dragging:
                self._drag_end = (event.mouse_region_x, event.mouse_region_y)
                if context.area:
                    context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        if event.type == 'LEFTMOUSE':
            if event.value == 'PRESS':
                self._dragging = True
                self._drag_start = (event.mouse_region_x, event.mouse_region_y)
                self._drag_end = self._drag_start
            elif event.value == 'RELEASE' and self._dragging:
                self._dragging = False
                self._drag_end = (event.mouse_region_x, event.mouse_region_y)
                op = 'add' if event.shift else ('remove' if event.ctrl else 'set')
                self._commit_rect(context, op)
                if context.area:
                    context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        if event.value == 'PRESS':
            et = event.type
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
                self._finish(context)
                return {'FINISHED'}

        return {'RUNNING_MODAL'}

    # --- overlay ----------------------------------------------------------

    def _draw_overlay(self, context):
        if not self._dragging or self._drag_start is None or self._drag_end is None:
            return
        try:
            (x0, y0), (x1, y1) = self._drag_start, self._drag_end
            coords = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
            gpu.state.blend_set('ALPHA')
            self._shader.bind()
            # dark under-stroke first, then the crisp white line on top
            gpu.state.line_width_set(2.0)
            batch = batch_for_shader(self._shader, 'LINE_STRIP', {'pos': coords})
            self._shader.uniform_float('color', _DARK)
            batch.draw(self._shader)
            gpu.state.line_width_set(1.5)
            batch = batch_for_shader(self._shader, 'LINE_STRIP', {'pos': coords})
            self._shader.uniform_float('color', _WHITE)
            batch.draw(self._shader)
            gpu.state.line_width_set(1.0)
            gpu.state.blend_set('NONE')
        except Exception as e:
            print(f'[pobim_splats] edit overlay error: {e}')


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

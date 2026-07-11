import math
import os
import uuid

import bpy
from bpy.props import BoolProperty, IntProperty, StringProperty
from bpy_extras.io_utils import ImportHelper

from . import splat_gpu


def _load_into_entry(obj, report=None):
    cloud = splat_gpu.load_entry_for_object(obj)
    if report:
        report({'INFO'}, f'โหลด {cloud.count:,} splats จาก '
                         f'{os.path.basename(obj.pobim_splat_file)}')


class POBIM_OT_import_splat(bpy.types.Operator, ImportHelper):
    """Import a 3D Gaussian Splat (.ply / .compressed.ply / .sog / meta.json)"""
    bl_idname = 'pobim_splats.import_ply'
    bl_label = 'Import Splat'
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = '.ply'
    filter_glob: StringProperty(default='*.ply;*.sog;*.json', options={'HIDDEN'})

    max_splats: IntProperty(
        name='Max Splats',
        description='จำกัดจำนวน splat (สุ่มลดตอนโหลด), 0 = โหลดทั้งหมด',
        default=0, min=0, soft_max=10_000_000)
    srgb_to_linear: BoolProperty(
        name='Convert Color to Linear',
        description='แปลงสีเป็น linear เพื่อให้ view transform แบบ Standard แสดงสีตรงกับ web viewer',
        default=True)
    orient_z_up: BoolProperty(
        name='Rotate to Z-up',
        description='หมุน -90° รอบแกน X (ไฟล์ 3DGS ส่วนใหญ่เป็นแบบ Y-down)',
        default=True)

    def execute(self, context):
        name = os.path.splitext(os.path.basename(self.filepath))[0]

        obj = bpy.data.objects.new(name, None)
        obj.empty_display_type = 'PLAIN_AXES'
        obj.empty_display_size = 0.5
        obj.pobim_splat_uid = uuid.uuid4().hex
        obj.pobim_splat_file = self.filepath
        obj.pobim_splat_max = self.max_splats
        obj.pobim_splat_srgb = self.srgb_to_linear
        if self.orient_z_up:
            obj.rotation_euler = (-math.pi / 2.0, 0.0, 0.0)
        context.collection.objects.link(obj)

        try:
            _load_into_entry(obj, self.report)
        except Exception as e:
            bpy.data.objects.remove(obj)
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}

        for other in context.selected_objects:
            other.select_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj

        splat_gpu.purge_orphans()
        splat_gpu.redraw_viewports()
        return {'FINISHED'}


class POBIM_OT_reload_splat(bpy.types.Operator):
    """Reload this splat from its .ply file"""
    bl_idname = 'pobim_splats.reload'
    bl_label = 'Reload Splat'

    uid: StringProperty()

    def execute(self, context):
        for obj in bpy.data.objects:
            if obj.pobim_splat_uid == self.uid:
                try:
                    _load_into_entry(obj, self.report)
                except Exception as e:
                    self.report({'ERROR'}, str(e))
                    return {'CANCELLED'}
                splat_gpu.redraw_viewports()
                return {'FINISHED'}
        self.report({'ERROR'}, 'ไม่พบ object ของ splat นี้แล้ว')
        return {'CANCELLED'}


class POBIM_OT_remove_splat(bpy.types.Operator):
    """Remove this splat and its object"""
    bl_idname = 'pobim_splats.remove'
    bl_label = 'Remove Splat'
    bl_options = {'REGISTER', 'UNDO'}

    uid: StringProperty()

    def execute(self, context):
        splat_gpu.REGISTRY.pop(self.uid, None)
        for obj in list(bpy.data.objects):
            if obj.pobim_splat_uid == self.uid:
                bpy.data.objects.remove(obj)
        splat_gpu.purge_orphans()
        splat_gpu.redraw_viewports()
        return {'FINISHED'}


CLASSES = (
    POBIM_OT_import_splat,
    POBIM_OT_reload_splat,
    POBIM_OT_remove_splat,
)

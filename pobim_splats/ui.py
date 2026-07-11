import bpy

from . import splat_gpu


class POBIM_PT_splats(bpy.types.Panel):
    bl_label = 'POBIM Splats'
    bl_idname = 'POBIM_PT_splats'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = '3DGS'

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        layout.operator('pobim_splats.import_ply', icon='IMPORT')

        row = layout.row(align=True)
        row.prop(scene, 'pobim_splats_enabled', text='Show Splats', toggle=True)
        layout.prop(scene, 'pobim_splat_sort_interval')

        splats = [obj for obj in scene.objects if obj.pobim_splat_uid]
        if not splats:
            layout.label(text='ยังไม่มี splat — กด Import', icon='INFO')
            return

        for obj in splats:
            entry = splat_gpu.REGISTRY.get(obj.pobim_splat_uid)
            box = layout.box()

            row = box.row(align=True)
            row.label(text=obj.name, icon='OUTLINER_OB_POINTCLOUD')
            op = row.operator('pobim_splats.reload', text='', icon='FILE_REFRESH')
            op.uid = obj.pobim_splat_uid
            op = row.operator('pobim_splats.remove', text='', icon='X')
            op.uid = obj.pobim_splat_uid

            if entry is None:
                box.label(text='ยังไม่ได้โหลด — กด Reload', icon='ERROR')
                continue
            if entry.error:
                box.label(text=entry.error, icon='ERROR')
                continue

            box.label(text=f'{obj.pobim_splat_count:,} splats')
            box.prop(obj, 'pobim_splat_scale')
            box.prop(obj, 'pobim_splat_opacity')


CLASSES = (POBIM_PT_splats,)

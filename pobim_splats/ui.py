import bpy

from . import splat_gpu


class POBIM_PT_splats(bpy.types.Panel):
    bl_label = 'POBIM Splats'
    bl_idname = 'POBIM_PT_splats'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'POBIM3DGS'

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        layout.operator('pobim_splats.import_ply', icon='IMPORT')

        row = layout.row(align=True)
        row.prop(scene, 'pobim_splats_enabled', text='Show Splats', toggle=True)
        layout.prop(scene, 'pobim_splat_sort_interval')
        layout.prop(scene, 'pobim_splats_aa', text='Energy-Conserving AA')

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

            sh = obj.pobim_splat_sh_loaded
            box.label(text=f'{obj.pobim_splat_count:,} splats'
                           + (f' · SH band {sh}' if sh else ''))
            box.prop(obj, 'pobim_splat_scale')
            box.prop(obj, 'pobim_splat_opacity')
            if sh:
                box.prop(obj, 'pobim_splat_sh_view')
            row = box.row(align=True)
            op = row.operator('pobim_splats.measure_scale', icon='DRIVER_DISTANCE')
            op.uid = obj.pobim_splat_uid
            row.prop(scene, 'pobim_splat_measure_kind', text='')
            row.prop(scene, 'pobim_splat_measure_mode', text='')
            if obj.get('pobim_measures'):
                op = box.operator('pobim_splats.clear_measures',
                                  icon='TRASH', text='Clear Measurements')
                op.uid = obj.pobim_splat_uid


CLASSES = (POBIM_PT_splats,)

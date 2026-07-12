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

        # --- Display: scene-wide render options -----------------------------
        disp = layout.box()
        disp.label(text='Display', icon='RESTRICT_VIEW_OFF')
        disp.prop(scene, 'pobim_splats_enabled', text='Show Splats', toggle=True)
        disp.prop(scene, 'pobim_splats_aa', text='Energy-Conserving AA')
        disp.prop(scene, 'pobim_splats_near_cull')
        disp.prop(scene, 'pobim_splat_sort_interval')

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

            # --- Splat: counts + per-splat appearance -----------------------
            sub = box.box()
            sh = obj.pobim_splat_sh_loaded
            sub.label(text=f'{obj.pobim_splat_count:,} splats'
                           + (f' · SH band {sh}' if sh else ''),
                      icon='OUTLINER_OB_POINTCLOUD')
            sub.prop(obj, 'pobim_splat_scale')
            sub.prop(obj, 'pobim_splat_opacity')
            if sh:
                sub.prop(obj, 'pobim_splat_sh_view')

            # --- Measure ----------------------------------------------------
            meas = box.box()
            meas.label(text='Measure', icon='DRIVER_DISTANCE')
            row = meas.row(align=True)
            op = row.operator('pobim_splats.measure_scale', text='Measure',
                              icon='DRIVER_DISTANCE')
            op.uid = obj.pobim_splat_uid
            row.prop(scene, 'pobim_splat_measure_kind', text='')
            row.prop(scene, 'pobim_splat_measure_mode', text='')
            if obj.get('pobim_measures'):
                op = meas.operator('pobim_splats.clear_measures',
                                   icon='TRASH', text='Clear Measurements')
                op.uid = obj.pobim_splat_uid

            # --- Edit -------------------------------------------------------
            edit = box.box()
            edit.label(text='Edit', icon='EDITMODE_HLT')
            row = edit.row(align=True)
            op = row.operator('pobim_splats.edit_splats',
                              icon='EDITMODE_HLT', text='Edit Splats')
            op.uid = obj.pobim_splat_uid
            op = row.operator('pobim_splats.export_ply',
                              icon='EXPORT', text='Export PLY')
            op.uid = obj.pobim_splat_uid
            edit.prop(scene, 'pobim_splat_edit_tool', text='Tool')
            edit.prop(scene, 'pobim_splat_brush_radius')
            edit.prop(scene, 'pobim_splat_sphere_radius')
            state = entry.state
            if state is not None and (state.num_selected or state.num_hidden
                                      or state.num_deleted):
                edit.label(text=f'เลือก {state.num_selected:,} · '
                                f'ซ่อน {state.num_hidden:,} · '
                                f'ลบ {state.num_deleted:,}', icon='CHECKMARK')


CLASSES = (POBIM_PT_splats,)

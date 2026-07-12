bl_info = {
    'name': 'POBIM Splats — 3D Gaussian Splatting Viewer',
    'author': 'POBIM',
    'version': (0, 8, 0),
    'blender': (4, 2, 0),
    'location': 'View3D > Sidebar (N) > POBIM3DGS',
    'description': 'Import and display 3D Gaussian Splatting .ply files with a real GPU splat renderer',
    'category': '3D View',
}

import bpy
from bpy.app.handlers import persistent
from bpy.props import (
    BoolProperty, EnumProperty, FloatProperty, IntProperty, StringProperty)

from . import edit_tools, measure, operators, splat_gpu, splat_ops, ui


def _redraw(self, context):
    splat_gpu.redraw_viewports()


@persistent
def _on_load_post(_dummy):
    """Rebuild splats referenced by the newly opened file."""
    splat_gpu.REGISTRY.clear()
    splat_gpu.reconcile()


@persistent
def _on_depsgraph_update(scene, depsgraph=None):
    splat_gpu.on_depsgraph_update(scene, depsgraph)


@persistent
def _on_undo_redo(*_args):
    """After global undo/redo, re-sync in-memory SplatState from the reverted
    property (Blender rolls back the ID custom property but not our session
    state / GPU texture — see splat_gpu.resync_states)."""
    splat_gpu.resync_states()


def register():
    bpy.types.Object.pobim_splat_uid = StringProperty(default='')
    bpy.types.Object.pobim_splat_file = StringProperty(subtype='FILE_PATH', default='')
    bpy.types.Object.pobim_splat_count = IntProperty(default=0)
    bpy.types.Object.pobim_splat_max = IntProperty(default=0, min=0)
    bpy.types.Object.pobim_splat_srgb = BoolProperty(default=True, update=_redraw)
    bpy.types.Object.pobim_splat_shmax = IntProperty(
        name='Max SH Bands', default=3, min=0, max=3)
    bpy.types.Object.pobim_splat_sh_loaded = IntProperty(default=0)
    bpy.types.Object.pobim_splat_sh_view = IntProperty(
        name='View SH Bands',
        description='จำนวน SH bands ที่ใช้แสดงผล (view-dependent color)',
        default=3, min=0, max=3, update=_redraw)
    bpy.types.Object.pobim_splat_scale = FloatProperty(
        name='Splat Size', default=1.0, min=0.05, max=10.0, update=_redraw)
    bpy.types.Object.pobim_splat_opacity = FloatProperty(
        name='Opacity', default=1.0, min=0.0, max=2.0, update=_redraw)
    bpy.types.Scene.pobim_splats_enabled = BoolProperty(
        name='Show Splats', default=True, update=_redraw)
    bpy.types.Scene.pobim_splat_measure_kind = EnumProperty(
        name='Measure Kind',
        description='ชนิดการวัด (กด D/A/V สลับได้ระหว่างวัด)',
        items=(
            ('DISTANCE', 'Distance', 'วัดระยะเป็นช่วงต่อเนื่อง'),
            ('AREA', 'Area', 'วัดพื้นที่รูปหลายเหลี่ยม + เส้นรอบรูป'),
            ('VOLUME', 'Volume', 'วัดปริมาตรกล่องจากมุมสองจุด'),
        ),
        default='DISTANCE')
    bpy.types.Scene.pobim_splat_measure_mode = EnumProperty(
        name='Pick Mode',
        description='วิธีจับจุดของเครื่องมือวัด (กด M สลับได้ระหว่างวัด)',
        items=(
            ('SURFACE', 'Surface', 'จับจุดบนพื้นผิว splat ที่เรนเดอร์จริง (แบบ POBIMStudio)'),
            ('CENTERS', 'Splat Centers', 'Snap เข้าหาศูนย์กลาง splat ที่ใกล้ที่สุด'),
        ),
        default='SURFACE')
    bpy.types.Scene.pobim_splat_edit_tool = EnumProperty(
        name='Select Tool',
        description='เครื่องมือเลือก splat ในโหมดแก้ไข (กด R/L/P/B/S/C สลับได้ระหว่างแก้ไข)',
        items=(
            ('RECT', 'Rect', 'เลือกด้วยกรอบสี่เหลี่ยม (ลากเมาส์)'),
            ('LASSO', 'Lasso', 'เลือกด้วยบ่วงวาดอิสระ'),
            ('POLYGON', 'Polygon', 'เลือกด้วยรูปหลายเหลี่ยม (คลิกทีละจุด)'),
            ('BRUSH', 'Brush', 'ทาสีเลือกด้วยพู่กันวงกลม'),
            ('SPHERE', 'Sphere', 'เลือกด้วยทรงกลมในปริภูมิ 3 มิติ'),
            ('BOX', 'Box', 'เลือกด้วยกล่องจากสองมุม (พิกัด local ของ splat)'),
        ),
        default='RECT')
    bpy.types.Scene.pobim_splats_aa = BoolProperty(
        name='Antialiasing (energy conserving)',
        description='ชดเชยความสว่างของ splat เล็กแบบ Mip-Splatting — ภาพไกลคมขึ้น '
                    'แต่พื้นผิวจะโปร่งกว่าโหมดปกติ (ปิด = look แบบ PlayCanvas/SuperSplat)',
        default=False, update=_redraw)
    bpy.types.Scene.pobim_splats_near_cull = FloatProperty(
        name='Near Cull (m)',
        description='ซ่อน splat ที่อยู่ใกล้กล้องกว่าระยะนี้ — สแกน indoor มัก'
                    'มีก้อนลอย (floater) ตามเส้นทางถ่ายที่บังจอเป็นแผ่นเบลอ '
                    'เพิ่มค่านี้เพื่อตัดทิ้งแบบเดียวกับ near plane ของ web viewer',
        default=0.1, min=0.0, soft_max=2.0, max=20.0, update=_redraw)
    bpy.types.Scene.pobim_splat_sort_interval = FloatProperty(
        name='Sort Interval (s)',
        description='ช่วงเวลาขั้นต่ำระหว่างการเรียงลำดับความลึก (ทำงานเบื้องหลัง ไม่บล็อก viewport '
                    'และเรียงเฉพาะเมื่อกล้องหมุนเกิน ~1°)',
        default=0.2, min=0.0, max=5.0)
    bpy.types.Scene.pobim_splat_brush_radius = IntProperty(
        name='Brush Radius (px)',
        description='รัศมีพู่กันเลือก splat เป็นพิกเซล (ในโหมดแก้ไข: กด F ปรับสด · '
                    'Alt+ลูกกลิ้ง ±10% · [ ] ปรับทีละขั้น · หรือลากแถบ Radius บน HUD)',
        default=40, min=4, max=400)
    bpy.types.Scene.pobim_splat_sphere_radius = FloatProperty(
        name='Sphere Radius (m)',
        description='รัศมีทรงกลมเลือก splat เป็นเมตร (ในโหมดแก้ไข: กด F ปรับสด · '
                    'Alt+ลูกกลิ้ง ±10% · [ ] ปรับทีละขั้น · หรือลากแถบ Radius บน HUD)',
        default=0.25, min=1e-3, max=100.0)

    for cls in (operators.CLASSES + measure.CLASSES + edit_tools.CLASSES
                + splat_ops.CLASSES + ui.CLASSES):
        bpy.utils.register_class(cls)

    splat_gpu.register_draw_handler()
    if _on_load_post not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_on_load_post)
    if _on_depsgraph_update not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph_update)
    if _on_undo_redo not in bpy.app.handlers.undo_post:
        bpy.app.handlers.undo_post.append(_on_undo_redo)
    if _on_undo_redo not in bpy.app.handlers.redo_post:
        bpy.app.handlers.redo_post.append(_on_undo_redo)


def unregister():
    if _on_undo_redo in bpy.app.handlers.redo_post:
        bpy.app.handlers.redo_post.remove(_on_undo_redo)
    if _on_undo_redo in bpy.app.handlers.undo_post:
        bpy.app.handlers.undo_post.remove(_on_undo_redo)
    if _on_depsgraph_update in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_on_depsgraph_update)
    if _on_load_post in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_on_load_post)
    splat_gpu.unregister_draw_handler()

    for cls in reversed(operators.CLASSES + measure.CLASSES + edit_tools.CLASSES
                        + splat_ops.CLASSES + ui.CLASSES):
        bpy.utils.unregister_class(cls)

    del bpy.types.Object.pobim_splat_uid
    del bpy.types.Object.pobim_splat_file
    del bpy.types.Object.pobim_splat_count
    del bpy.types.Object.pobim_splat_max
    del bpy.types.Object.pobim_splat_srgb
    del bpy.types.Object.pobim_splat_shmax
    del bpy.types.Object.pobim_splat_sh_loaded
    del bpy.types.Object.pobim_splat_sh_view
    del bpy.types.Object.pobim_splat_scale
    del bpy.types.Object.pobim_splat_opacity
    del bpy.types.Scene.pobim_splats_enabled
    del bpy.types.Scene.pobim_splats_aa
    del bpy.types.Scene.pobim_splat_measure_kind
    del bpy.types.Scene.pobim_splat_measure_mode
    del bpy.types.Scene.pobim_splat_edit_tool
    del bpy.types.Scene.pobim_splats_near_cull
    del bpy.types.Scene.pobim_splat_sort_interval
    del bpy.types.Scene.pobim_splat_brush_radius
    del bpy.types.Scene.pobim_splat_sphere_radius

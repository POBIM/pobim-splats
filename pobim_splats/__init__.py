bl_info = {
    'name': 'POBIM Splats — 3D Gaussian Splatting Viewer',
    'author': 'POBIM',
    'version': (0, 1, 0),
    'blender': (4, 2, 0),
    'location': 'View3D > Sidebar (N) > 3DGS',
    'description': 'Import and display 3D Gaussian Splatting .ply files with a real GPU splat renderer',
    'category': '3D View',
}

import bpy
from bpy.app.handlers import persistent
from bpy.props import BoolProperty, FloatProperty, IntProperty, StringProperty

from . import operators, splat_gpu, ui


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


def register():
    bpy.types.Object.pobim_splat_uid = StringProperty(default='')
    bpy.types.Object.pobim_splat_file = StringProperty(subtype='FILE_PATH', default='')
    bpy.types.Object.pobim_splat_count = IntProperty(default=0)
    bpy.types.Object.pobim_splat_max = IntProperty(default=0, min=0)
    bpy.types.Object.pobim_splat_srgb = BoolProperty(default=True)
    bpy.types.Object.pobim_splat_scale = FloatProperty(
        name='Splat Size', default=1.0, min=0.05, max=10.0, update=_redraw)
    bpy.types.Object.pobim_splat_opacity = FloatProperty(
        name='Opacity', default=1.0, min=0.0, max=2.0, update=_redraw)
    bpy.types.Scene.pobim_splats_enabled = BoolProperty(
        name='Show Splats', default=True, update=_redraw)
    bpy.types.Scene.pobim_splat_sort_interval = FloatProperty(
        name='Sort Interval (s)',
        description='ยิ่งต่ำยิ่งเรียงลำดับความลึกถี่ (ถูกต้องขึ้นตอนหมุนกล้อง แต่กระตุกขึ้นในฉากใหญ่)',
        default=0.5, min=0.05, max=5.0)

    for cls in operators.CLASSES + ui.CLASSES:
        bpy.utils.register_class(cls)

    splat_gpu.register_draw_handler()
    if _on_load_post not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_on_load_post)
    if _on_depsgraph_update not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph_update)


def unregister():
    if _on_depsgraph_update in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_on_depsgraph_update)
    if _on_load_post in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_on_load_post)
    splat_gpu.unregister_draw_handler()

    for cls in reversed(operators.CLASSES + ui.CLASSES):
        bpy.utils.unregister_class(cls)

    del bpy.types.Object.pobim_splat_uid
    del bpy.types.Object.pobim_splat_file
    del bpy.types.Object.pobim_splat_count
    del bpy.types.Object.pobim_splat_max
    del bpy.types.Object.pobim_splat_srgb
    del bpy.types.Object.pobim_splat_scale
    del bpy.types.Object.pobim_splat_opacity
    del bpy.types.Scene.pobim_splats_enabled
    del bpy.types.Scene.pobim_splat_sort_interval

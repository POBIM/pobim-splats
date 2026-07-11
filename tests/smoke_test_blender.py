# Headless smoke test — run inside Blender:
#   blender -b --factory-startup --python tests/smoke_test_blender.py
#
# Verifies: addon registers, PLY imports through the operator, properties
# exist, and (when a GPU context is available) the shader compiles and
# GPU resources build.

import os
import sys
import tempfile

import bpy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tests'))

FAILURES = []


def check(name, fn):
    try:
        fn()
        print(f'  OK   {name}')
    except Exception as e:
        FAILURES.append((name, e))
        print(f'  FAIL {name}: {type(e).__name__}: {e}')


def main():
    import pobim_splats
    from pobim_splats import splat_gpu
    from make_test_ply import make_torus_splats, write_gaussian_ply

    check('register addon', pobim_splats.register)

    tmp = tempfile.mkdtemp()
    ply_path = os.path.join(tmp, 'torus.ply')
    write_gaussian_ply(ply_path, *make_torus_splats(100_000))

    def do_import():
        result = bpy.ops.pobim_splats.import_ply(filepath=ply_path)
        assert result == {'FINISHED'}, result
    check('import operator', do_import)

    def check_object():
        obj = bpy.data.objects['torus']
        assert obj.pobim_splat_uid, 'uid missing'
        assert obj.pobim_splat_count == 100_000, obj.pobim_splat_count
        assert obj.pobim_splat_uid in splat_gpu.REGISTRY, 'registry entry missing'
        assert abs(obj.rotation_euler.x + 1.5708) < 1e-3, 'z-up rotation not applied'
    check('object + registry state', check_object)

    # GPU part — background mode may have no GPU context; report but tolerate
    gpu_ok = True
    try:
        import gpu  # noqa: F401
        splat_gpu.get_shader()
        print('  OK   shader compiles')
    except Exception as e:
        gpu_ok = False
        print(f'  SKIP shader compile (no GPU context in background mode?): {e}')

    if gpu_ok:
        def build_gpu():
            entry = next(iter(splat_gpu.REGISTRY.values()))
            splat_gpu.SplatGPU(entry.cloud)
        check('GPU resources build', build_gpu)

        def offscreen_draw():
            import gpu
            import numpy as np
            entry = next(iter(splat_gpu.REGISTRY.values()))
            sg = splat_gpu.SplatGPU(entry.cloud)
            offs = gpu.types.GPUOffScreen(256, 256)
            with offs.bind():
                fb = gpu.state.active_framebuffer_get()
                fb.clear(color=(0.0, 0.0, 0.0, 0.0), depth=1.0)
                view = np.eye(4, dtype=np.float32)
                view[2, 3] = -8.0  # camera looking down -z at origin
                f = 2.0
                proj = np.array([
                    [f, 0, 0, 0],
                    [0, f, 0, 0],
                    [0, 0, -1.02, -2.02],
                    [0, 0, -1, 0]], np.float32)
                sg.sort_if_needed(view, 0.0)
                params = np.array([256, 256, 1.0, 1.0, 0, 0, 0, 0], np.float32)
                ubo_data = np.concatenate([view.T.ravel(), proj.T.ravel(), params])
                ubo = gpu.types.GPUUniformBuf(
                    splat_gpu._np_buffer('FLOAT', ubo_data))
                shader = splat_gpu.get_shader()
                gpu.state.blend_set('ALPHA')
                shader.bind()
                shader.uniform_block('u', ubo)
                shader.uniform_sampler('dataTex', sg.data_tex)
                shader.uniform_sampler('orderTex', sg.order_tex)
                sg.batch.draw(shader)
                gpu.state.blend_set('NONE')
                pixels = np.array(fb.read_color(0, 0, 256, 256, 4, 0, 'FLOAT').to_list())
            offs.free()
            covered = (pixels[..., 3] > 0.01).mean()
            assert covered > 0.05, f'splats cover only {covered:.1%} of test render'
            print(f'       splat coverage in test render: {covered:.1%}')
        check('offscreen draw renders splats', offscreen_draw)

    def reconcile_duplicate():
        obj = bpy.data.objects['torus']
        copy = obj.copy()
        bpy.context.collection.objects.link(copy)
        assert copy.pobim_splat_uid == obj.pobim_splat_uid, 'expected copied uid'
        splat_gpu.reconcile()
        assert copy.pobim_splat_uid != obj.pobim_splat_uid, 'uid collision not fixed'
        assert copy.pobim_splat_uid in splat_gpu.REGISTRY, 'copy has no registry entry'
        assert obj.pobim_splat_uid in splat_gpu.REGISTRY, 'original entry lost'
        bpy.data.objects.remove(copy)
        splat_gpu.reconcile()
    check('reconcile: duplicated object gets own entry', reconcile_duplicate)

    def reconcile_delete_and_restore():
        obj = bpy.data.objects['torus']
        uid = obj.pobim_splat_uid
        filepath = obj.pobim_splat_file

        # keyboard-delete: object gone -> entry purged (no VRAM leak)
        bpy.data.objects.remove(obj)
        splat_gpu.reconcile()
        assert uid not in splat_gpu.REGISTRY, 'orphan entry not purged'

        # undo-of-remove equivalent: object exists, entry missing -> rebuilt
        restored = bpy.data.objects.new('torus', None)
        restored.pobim_splat_uid = uid
        restored.pobim_splat_file = filepath
        bpy.context.collection.objects.link(restored)
        splat_gpu.reconcile()
        assert uid in splat_gpu.REGISTRY, 'missing entry not rebuilt'
        assert restored.pobim_splat_count == 100_000
    check('reconcile: delete purges, restore rebuilds', reconcile_delete_and_restore)

    def apply_scale():
        import numpy as np
        obj = bpy.data.objects['torus']
        before = np.array(obj.matrix_world, np.float64)
        result = bpy.ops.pobim_splats.apply_scale(
            uid=obj.pobim_splat_uid, measured=2.0, target=4.0,
            pivot=(1.0, 0.5, -0.25))
        assert result == {'FINISHED'}
        after = np.array(obj.matrix_world, np.float64)
        delta = after @ np.linalg.inv(before)
        pivot_h = np.array([1.0, 0.5, -0.25, 1.0])
        assert np.allclose(delta @ pivot_h, pivot_h, atol=1e-5), 'pivot moved'
        assert abs(np.linalg.det(delta[:3, :3]) - 8.0) < 1e-3, 'scale factor wrong'
    check('apply_scale keeps pivot, scales x2', apply_scale)

    def import_compressed():
        from make_test_ply import write_compressed_gaussian_ply
        cpath = os.path.join(tmp, 'torus.compressed.ply')
        write_compressed_gaussian_ply(cpath, *make_torus_splats(50_000))
        result = bpy.ops.pobim_splats.import_ply(filepath=cpath)
        assert result == {'FINISHED'}
        obj = bpy.data.objects['torus.compressed']
        assert obj.pobim_splat_count == 50_000
        assert bpy.ops.pobim_splats.remove(uid=obj.pobim_splat_uid) == {'FINISHED'}
    check('import compressed.ply via operator', import_compressed)

    def measure_store_roundtrip():
        import numpy as np
        from pobim_splats.measure import MeasureStore
        obj = bpy.data.objects['torus']
        store = MeasureStore(obj)
        assert store.chains == [] and store.polygons == [] and store.boxes == []
        store.chains.append([np.array([0, 0, 0], np.float32),
                             np.array([1, 2, 3], np.float32)])
        store.polygons.append([np.array([0, 0, 0], np.float32),
                               np.array([1, 0, 0], np.float32),
                               np.array([1, 1, 0], np.float32)])
        store.boxes.append([np.array([0, 0, 0], np.float32),
                            np.array([2, 2, 2], np.float32)])
        store.save()
        again = MeasureStore(obj)
        assert len(again.chains) == 1 and len(again.chains[0]) == 2
        assert np.allclose(again.chains[0][1], (1, 2, 3))
        assert len(again.polygons) == 1 and len(again.boxes) == 1
        # clear operator wipes the property
        result = bpy.ops.pobim_splats.clear_measures(uid=obj.pobim_splat_uid)
        assert result == {'FINISHED'}
        assert not obj.get('pobim_measures')
    check('measure store persists and clears', measure_store_roundtrip)

    def edit_state_persist():
        import numpy as np
        from pobim_splats.splat_state import SplatState
        obj = bpy.data.objects['torus']
        entry = splat_gpu.REGISTRY[obj.pobim_splat_uid]
        count = entry.cloud.count
        state = SplatState(count)
        state.select_indices(np.arange(0, count, 2))   # select half
        selected = state.num_selected
        state.delete_selected()
        deleted = state.num_deleted
        assert deleted == selected, (deleted, selected)
        entry.state = state
        obj['pobim_splat_state'] = state.serialize()
        # a fresh (re)load must rebuild the entry AND restore the edit state
        splat_gpu.load_entry_for_object(obj)
        restored = splat_gpu.REGISTRY[obj.pobim_splat_uid]
        assert restored.state is not None, 'edit state not restored on reload'
        assert restored.state.flags.size == count
        assert restored.state.num_deleted == deleted, \
            (restored.state.num_deleted, deleted)
    check('edit state persists through reload', edit_state_persist)

    def export_survivors():
        from pobim_splats.ply_loader import load_gaussian_ply
        obj = bpy.data.objects['torus']
        entry = splat_gpu.REGISTRY[obj.pobim_splat_uid]
        deleted = entry.state.num_deleted if entry.state is not None else 0
        count = obj.pobim_splat_count
        out = os.path.join(tmp, 'export.ply')
        result = bpy.ops.pobim_splats.export_ply(
            filepath=out, uid=obj.pobim_splat_uid)
        assert result == {'FINISHED'}, result
        cloud = load_gaussian_ply(out)
        assert cloud.count == count - deleted, (cloud.count, count, deleted)
    check('export_ply writes surviving splats', export_survivors)

    def edit_modal_registered():
        assert hasattr(bpy.ops.pobim_splats, 'edit_splats'), 'operator not registered'
        from pobim_splats.edit_tools import POBIM_OT_edit_splats
        obj = bpy.data.objects['torus']
        # background mode: no window/area for a modal op -> Blender short-circuits
        # to PASS_THROUGH, or invoke's `area is None` guard returns CANCELLED.
        # Either way it must not raise and must not leave the tool running.
        result = bpy.ops.pobim_splats.edit_splats(
            'INVOKE_DEFAULT', uid=obj.pobim_splat_uid)
        assert result in ({'CANCELLED'}, {'PASS_THROUGH'}), result
        assert POBIM_OT_edit_splats._running is False, 'tool left running'
    check('edit_splats modal registers, cancels in background', edit_modal_registered)

    def reload_and_remove():
        obj = bpy.data.objects['torus']
        uid = obj.pobim_splat_uid
        assert bpy.ops.pobim_splats.reload(uid=uid) == {'FINISHED'}
        assert bpy.ops.pobim_splats.remove(uid=uid) == {'FINISHED'}
        assert uid not in splat_gpu.REGISTRY
        assert 'torus' not in bpy.data.objects
    check('reload + remove operators', reload_and_remove)

    check('unregister addon', pobim_splats.unregister)

    print()
    if FAILURES:
        print(f'{len(FAILURES)} FAILURE(S)')
        sys.exit(1)
    print('smoke test passed')


main()

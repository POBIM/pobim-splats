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
                cam4 = np.array([0, 0, 8.0, 0], np.float32)
                sel4 = np.zeros(4, np.float32)   # a=0 skips the state fetch
                prev4 = np.eye(4, dtype=np.float32).ravel()
                misc = np.zeros(4, np.float32)
                ubo_data = np.concatenate([view.T.ravel(), proj.T.ravel(),
                                           params, cam4, sel4, prev4, misc])
                ubo = gpu.types.GPUUniformBuf(
                    splat_gpu._np_buffer('FLOAT', ubo_data))
                shader = splat_gpu.get_shader()
                gpu.state.blend_set('ALPHA')
                shader.bind()
                shader.uniform_block('u', ubo)
                shader.uniform_sampler('dataTex', sg.data_tex)
                shader.uniform_sampler('orderTex', sg.order_tex)
                shader.uniform_sampler('shTex', sg.sh_tex if sg.sh_tex else sg.data_tex)
                shader.uniform_sampler('stateTex', sg.data_tex)
                sg.batch.draw_instanced(shader, instance_count=sg.count)
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

    def transform_edits_persist_and_export():
        # The Track T<->U seam that once silently lost data: persisted
        # transform edits must survive a .blend reload (registry rebuild) AND
        # flow into the export. Steps: (1) create + persist edits, (2) simulate
        # reload via a fresh load_entry_for_object, (3) assert entry.edits
        # restored with the dirty rows, (4) export and re-parse, asserting the
        # EDITED positions were written (not the originals).
        import numpy as np
        from pobim_splats.splat_edits import SplatEdits
        from pobim_splats.ply_loader import load_gaussian_ply
        obj = bpy.data.objects['torus']
        uid = obj.pobim_splat_uid
        entry = splat_gpu.REGISTRY[uid]
        cloud = entry.cloud
        count = cloud.count
        assert cloud.quats is not None and cloud.scales_log is not None, \
            'keep_geometry raw arrays missing from the cloud'

        # odd rows survive edit_state_persist's delete (it deleted the evens)
        idx = np.array([1, 3, 5], np.int64)
        ed = SplatEdits(count)
        M = np.eye(4)
        M[:3, 3] = [10.0, 20.0, 30.0]
        payload = ed.apply_matrix(idx, M, cloud.positions, cloud.quats,
                                  cloud.scales_log)
        assert payload is not None
        expected = payload[2]['positions'].copy()
        obj['pobim_splat_edits'] = ed.serialize()

        # simulate .blend reload: fresh registry entry from object properties
        splat_gpu.REGISTRY.pop(uid, None)
        splat_gpu.load_entry_for_object(obj)
        entry2 = splat_gpu.REGISTRY[uid]
        assert getattr(entry2, 'edits', None) is not None, \
            'geometry edits not restored on reload'
        assert set(np.nonzero(entry2.edits.dirty)[0].tolist()) == set(idx.tolist()), \
            'restored dirty rows wrong'

        # export must carry the edited positions through the reloaded entry
        out = os.path.join(tmp, 'export_edited.ply')
        result = bpy.ops.pobim_splats.export_ply(filepath=out, uid=uid)
        assert result == {'FINISHED'}, result
        exported = load_gaussian_ply(out)
        keep = (entry2.state.keep_mask() if entry2.state is not None
                else np.ones(count, bool))
        kept_rows = np.nonzero(keep)[0]
        out_rows = np.searchsorted(kept_rows, idx)
        assert (kept_rows[out_rows] == idx).all(), 'edited splats were dropped'
        got = exported.positions[out_rows]
        assert np.allclose(got, expected, atol=1e-4), \
            f'exported positions are NOT the edited ones: {got[0]} vs {expected[0]}'
    check('transform edits persist through reload into export',
          transform_edits_persist_and_export)

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

    def edit_overrides_restore():
        # U4: obj['pobim_splat_edits'] restore path. When Track T's SplatEdits
        # is importable, exercise serialize->reload->restore end to end; when
        # it lags, assert the defensive path leaves the modal creatable anyway.
        import numpy as np
        from pobim_splats import edit_tools
        obj = bpy.data.objects['torus']
        entry = splat_gpu.REGISTRY[obj.pobim_splat_uid]
        count = entry.cloud.count
        try:
            from pobim_splats.splat_edits import SplatEdits
        except Exception:
            SplatEdits = None
        if SplatEdits is not None:
            edits = SplatEdits(count)
            idx = np.arange(0, min(count, 10), dtype=np.int64)
            mat = np.eye(4, dtype=np.float32)
            mat[:3, 3] = (0.5, -0.25, 0.1)      # a pure translation
            cloud = entry.cloud
            base_quat = getattr(cloud, 'quats', None)
            base_slog = getattr(cloud, 'scales_log', None)
            if base_quat is None:
                base_quat = np.tile(np.array([1, 0, 0, 0], np.float32), (count, 1))
            if base_slog is None:
                base_slog = np.zeros((count, 3), np.float32)
            edits.apply_matrix(idx, mat, cloud.positions, base_quat, base_slog)
            obj['pobim_splat_edits'] = edits.serialize()
            entry.edits = None                  # force a fresh restore
            restored = SplatEdits.deserialize(obj['pobim_splat_edits'], count)
            assert bool(np.any(restored.dirty)), 'restored edits lost dirty set'
            entry.edits = restored
        # the modal invoke path must survive the property either way
        result = bpy.ops.pobim_splats.edit_splats(
            'INVOKE_DEFAULT', uid=obj.pobim_splat_uid)
        assert result in ({'CANCELLED'}, {'PASS_THROUGH'}), result
        assert edit_tools.POBIM_OT_edit_splats._running is False, 'tool left running'
        if 'pobim_splat_edits' in obj.keys():
            del obj['pobim_splat_edits']
        entry.edits = None
    check('edit overrides serialize + restore path exercised', edit_overrides_restore)

    def duplicate_and_separate():
        # Track F end-to-end: Duplicate splits a SELECTED subset into a new
        # object that references the same file but persists only the chosen
        # absolute rows; intersecting transform edits are carried over and
        # survive a simulated reload. Separate additionally soft-deletes the
        # rows on the source, and the source export then drops them.
        import numpy as np
        from pobim_splats.splat_state import SplatState, deserialize_rows
        from pobim_splats.splat_edits import SplatEdits
        from pobim_splats.ply_loader import load_gaussian_ply

        obj = bpy.data.objects['torus']
        uid = obj.pobim_splat_uid
        entry = splat_gpu.REGISTRY[uid]
        cloud = entry.cloud
        count = cloud.count

        # a deterministic selection on non-deleted rows + edits on a subset
        sel = np.array([1, 3, 5, 7, 9, 11], np.int64)
        state = SplatState(count)
        state.select_indices(sel, 'set')
        entry.state = state
        obj['pobim_splat_state'] = state.serialize()

        ed = SplatEdits(count)
        M = np.eye(4)
        M[:3, 3] = [7.0, 8.0, 9.0]
        ed.apply_matrix(np.array([1, 3], np.int64), M,
                        cloud.positions, cloud.quats, cloud.scales_log)
        entry.edits = ed
        obj['pobim_splat_edits'] = ed.serialize()

        # --- Duplicate ---------------------------------------------------
        assert bpy.ops.pobim_splats.duplicate_selection(uid=uid) == {'FINISHED'}
        dup = bpy.data.objects.get('torus Selection')
        assert dup is not None, 'duplicate object not created'
        assert dup.pobim_splat_count == sel.size, dup.pobim_splat_count
        rows = deserialize_rows(dup['pobim_splat_subset'])
        assert np.array_equal(rows, sel), rows
        # source per-gaussian flags untouched by Duplicate (TS parity)
        assert entry.state.num_deleted == 0
        assert entry.state.num_selected == sel.size

        # carried transform edits: rows 1,3 -> subset-local indices 0,1
        dup_entry = splat_gpu.REGISTRY[dup.pobim_splat_uid]
        assert dup_entry.edits is not None, 'carried edits missing'
        assert set(np.nonzero(dup_entry.edits.dirty)[0].tolist()) == {0, 1}, \
            np.nonzero(dup_entry.edits.dirty)[0].tolist()

        # simulated save/reload of the subset object rebuilds from properties
        splat_gpu.REGISTRY.pop(dup.pobim_splat_uid, None)
        splat_gpu.load_entry_for_object(dup)
        re = splat_gpu.REGISTRY[dup.pobim_splat_uid]
        assert re.cloud.count == sel.size, re.cloud.count
        # source_indices of the reloaded subset are the absolute file rows
        assert np.array_equal(re.cloud.source_indices, sel)
        assert re.edits is not None and \
            set(np.nonzero(re.edits.dirty)[0].tolist()) == {0, 1}, \
            'carried edits lost on reload'

        # --- Separate ----------------------------------------------------
        state2 = SplatState(count)
        sep_sel = np.array([7, 9, 11], np.int64)
        state2.select_indices(sep_sel, 'set')
        entry.state = state2
        obj['pobim_splat_state'] = state2.serialize()
        entry.edits = None
        if 'pobim_splat_edits' in obj.keys():
            del obj['pobim_splat_edits']

        assert bpy.ops.pobim_splats.separate_selection(uid=uid) == {'FINISHED'}
        # deleted rows keep SELECTED and gain DELETED (selected|deleted)
        assert entry.state.num_deleted == sep_sel.size, entry.state.num_deleted
        for r in sep_sel:
            f = int(entry.state.flags[r])
            assert f & 1 and f & 4, f
        # source export drops the separated rows
        out = os.path.join(tmp, 'export_after_sep.ply')
        assert bpy.ops.pobim_splats.export_ply(
            filepath=out, uid=uid) == {'FINISHED'}
        exported = load_gaussian_ply(out)
        assert exported.count == count - sep_sel.size, \
            (exported.count, count, sep_sel.size)

        # tidy up the two subset objects so later checks see only 'torus'
        for name in ('torus Selection', 'torus Selection.001'):
            o = bpy.data.objects.get(name)
            if o is not None:
                splat_gpu.REGISTRY.pop(o.pobim_splat_uid, None)
                bpy.data.objects.remove(o)
        # restore a clean source state for the remaining checks
        entry.state = None
        for k in ('pobim_splat_state', 'pobim_splat_edits'):
            if k in obj.keys():
                del obj[k]
        splat_gpu.load_entry_for_object(obj)
    check('duplicate + separate selection end-to-end', duplicate_and_separate)

    def undo_resync():
        # MAJOR regression (audit): Blender's global undo reverts the
        # obj['pobim_splat_state'] ID custom property but NOT the Python-session
        # entry.state / GPU texture. After a non-modal Separate + Ctrl+Z the
        # source would keep showing the separated gaussians as deleted until a
        # manual Reload. resync_states() (the undo_post/redo_post handler) must
        # rebuild the in-memory state from whatever the reverted property holds.
        # This test drives resync_states() directly; without the fix (a no-op /
        # absent handler) the assertions below fail — entry.state keeps the
        # deletion / stays populated.
        import numpy as np
        from pobim_splats.splat_state import SplatState, State
        obj = bpy.data.objects['torus']
        uid = obj.pobim_splat_uid
        entry = splat_gpu.REGISTRY[uid]
        count = entry.cloud.count

        # Case A: property reverts to an OLDER value (undo of a Separate).
        sel = np.array([2, 4, 6], np.int64)
        pre = SplatState(count)
        pre.select_indices(sel, 'set')
        pre_payload = pre.serialize()          # what undo restores the prop to

        post = SplatState(count)
        post.select_indices(sel, 'set')
        post.set_flags_raw(sel, (post.flags[sel] | State.DELETED).astype(np.uint8))
        entry.state = post
        splat_gpu.persist_state(obj, entry)    # prop + cache == post
        assert entry.state.num_deleted == sel.size
        # simulate Blender rolling the ID property back to the pre-Separate value
        obj['pobim_splat_state'] = pre_payload
        splat_gpu.resync_states()
        assert entry.state.num_deleted == 0, \
            f'resync did not clear deletion: {entry.state.num_deleted}'
        assert entry.state.num_selected == sel.size, entry.state.num_selected
        assert entry.state_payload == pre_payload, 'cache not updated after resync'

        # Case B: property reverts to ABSENT (undo back before any edit existed).
        splat_gpu.persist_state(obj, entry)    # prop + cache present again
        del obj['pobim_splat_state']
        splat_gpu.resync_states()
        assert entry.state is None, 'resync should drop state when prop is gone'
        assert entry.state_payload is None

        # Case C: in-sync objects are a cheap no-op (no exceptions, state kept).
        entry.state = None
        splat_gpu.resync_states()
        assert entry.state is None

        # leave the source clean for later checks
        entry.state = None
        entry.state_payload = None
        for k in ('pobim_splat_state', 'pobim_splat_edits'):
            if k in obj.keys():
                del obj[k]
    check('undo/redo resync rebuilds source state', undo_resync)

    def reload_and_remove():
        obj = bpy.data.objects['torus']
        uid = obj.pobim_splat_uid
        assert bpy.ops.pobim_splats.reload(uid=uid) == {'FINISHED'}
        assert bpy.ops.pobim_splats.remove(uid=uid) == {'FINISHED'}
        assert uid not in splat_gpu.REGISTRY
        assert 'torus' not in bpy.data.objects
    check('reload + remove operators', reload_and_remove)

    def edit_tool_enum():
        scene = bpy.context.scene
        prop = scene.bl_rna.properties['pobim_splat_edit_tool']
        items = [it.identifier for it in prop.enum_items]
        assert items == ['RECT', 'LASSO', 'POLYGON', 'BRUSH', 'SPHERE', 'BOX'], items
        assert prop.default == 'RECT', prop.default
        scene.pobim_splat_edit_tool = 'BRUSH'
        assert scene.pobim_splat_edit_tool == 'BRUSH', scene.pobim_splat_edit_tool
        scene.pobim_splat_edit_tool = 'RECT'
    check('edit tool enum registers with 6 items, round-trips', edit_tool_enum)

    def radius_props():
        # U2: the two radius scene props register with the documented ranges
        # and round-trip (the modal reads them on invoke, writes on change).
        scene = bpy.context.scene
        bp = scene.bl_rna.properties['pobim_splat_brush_radius']
        assert bp.default == 40, bp.default
        assert bp.hard_min == 4 and bp.hard_max == 400, (bp.hard_min, bp.hard_max)
        sp = scene.bl_rna.properties['pobim_splat_sphere_radius']
        assert abs(sp.default - 0.25) < 1e-6, sp.default
        assert sp.hard_max == 100.0, sp.hard_max
        scene.pobim_splat_brush_radius = 123
        assert scene.pobim_splat_brush_radius == 123
        scene.pobim_splat_sphere_radius = 1.5
        assert abs(scene.pobim_splat_sphere_radius - 1.5) < 1e-6
        # clamping honours the registered bounds
        scene.pobim_splat_brush_radius = 9999
        assert scene.pobim_splat_brush_radius == 400
        scene.pobim_splat_brush_radius = 40
        scene.pobim_splat_sphere_radius = 0.25
    check('radius scene props register + round-trip + clamp', radius_props)

    check('unregister addon', pobim_splats.unregister)

    print()
    if FAILURES:
        print(f'{len(FAILURES)} FAILURE(S)')
        sys.exit(1)
    print('smoke test passed')


main()

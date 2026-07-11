# GPU-context test — must run in FOREGROUND mode (WSLg/X11 display needed):
#   blender --factory-startup --python tests/gpu_test_blender.py
#
# Compiles the splat shader, builds GPU resources for a synthetic cloud and
# draws it into an offscreen buffer, checking splats actually rasterize.
# Results are written to tests/gpu_test_result.txt, then Blender quits.

import os
import sys
import tempfile
import traceback

import bpy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tests'))

RESULT_PATH = os.path.join(ROOT, 'tests', 'gpu_test_result.txt')
LINES = []


def log(msg):
    print(msg)
    LINES.append(msg)


def run_tests():
    import numpy as np
    import gpu
    import pobim_splats
    from pobim_splats import splat_gpu
    from make_test_ply import make_torus_splats, write_gaussian_ply
    from pobim_splats.ply_loader import load_gaussian_ply

    pobim_splats.register()

    tmp = tempfile.mkdtemp()
    ply_path = os.path.join(tmp, 'torus.ply')
    write_gaussian_ply(ply_path, *make_torus_splats(100_000))
    cloud = load_gaussian_ply(ply_path)

    shader = splat_gpu.get_shader()
    log('OK shader compiles')

    sg = splat_gpu.SplatGPU(cloud)
    log(f'OK GPU resources build ({sg.count:,} splats)')

    offs = gpu.types.GPUOffScreen(256, 256)
    with offs.bind():
        fb = gpu.state.active_framebuffer_get()
        fb.clear(color=(0.0, 0.0, 0.0, 0.0), depth=1.0)

        view = np.eye(4, dtype=np.float32)
        view[2, 3] = -8.0
        f = 2.0
        proj = np.array([
            [f, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, -1.02, -2.02],
            [0, 0, -1, 0]], np.float32)

        # sorting is async now: launch, wait for the worker, pick up result
        import time as _time
        sg.sort_if_needed(view, 0.0)
        deadline = _time.monotonic() + 10.0
        while sg._sort_result is None and _time.monotonic() < deadline:
            _time.sleep(0.01)
        sg.sort_if_needed(view, 0.0)
        assert sg._applied_dir is not None, 'async sort did not complete'
        log('OK async depth sort completes and applies')

        params = np.array([256, 256, 1.0, 1.0, 0, 0, 0, 0], np.float32)
        ubo_data = np.concatenate([view.T.ravel(), proj.T.ravel(), params])
        ubo = gpu.types.GPUUniformBuf(splat_gpu._np_buffer('FLOAT', ubo_data))

        gpu.state.blend_set('ALPHA')
        gpu.state.depth_test_set('LESS_EQUAL')
        gpu.state.depth_mask_set(False)
        shader.bind()
        shader.uniform_block('u', ubo)
        shader.uniform_sampler('dataTex', sg.data_tex)
        shader.uniform_sampler('orderTex', sg.order_tex)
        sg.batch.draw(shader)
        gpu.state.depth_mask_set(True)
        gpu.state.depth_test_set('NONE')
        gpu.state.blend_set('NONE')

        raw = fb.read_color(0, 0, 256, 256, 4, 0, 'FLOAT')
        pixels = np.array(raw.to_list(), np.float32).reshape(256, 256, 4)
    offs.free()

    covered = float((pixels[..., 3] > 0.01).mean())
    max_alpha = float(pixels[..., 3].max())
    log(f'OK offscreen draw: coverage={covered:.1%} maxAlpha={max_alpha:.3f}')
    if covered < 0.05:
        raise AssertionError(f'coverage too low: {covered:.1%} — splats not rendering')

    # colored pixels present (rainbow torus, so all channels should appear)
    lit = pixels[pixels[..., 3] > 0.1]
    log(f'OK color check: mean rgb of lit pixels = '
        f'({lit[:, 0].mean():.2f}, {lit[:, 1].mean():.2f}, {lit[:, 2].mean():.2f})')

    pobim_splats.unregister()
    log('PASSED')


def main():
    try:
        run_tests()
    except Exception:
        LINES.append('FAILED')
        LINES.append(traceback.format_exc())
        print(traceback.format_exc())
    with open(RESULT_PATH, 'w') as f:
        f.write('\n'.join(LINES) + '\n')
    bpy.ops.wm.quit_blender()


# run after the window/GPU context is fully initialized
bpy.app.timers.register(main, first_interval=0.5)

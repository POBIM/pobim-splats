# Render the synthetic torus through the splat pipeline and save a PNG for
# visual inspection. Foreground mode:
#   blender --factory-startup --python tests/render_preview_blender.py

import os
import sys
import tempfile
import traceback

import bpy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tests'))

OUT_PNG = os.path.join(ROOT, 'tests', 'preview.png')
SIZE = 512


def render():
    import numpy as np
    import gpu
    from pobim_splats import splat_gpu
    from pobim_splats.ply_loader import load_gaussian_ply
    from make_test_ply import make_torus_splats, write_gaussian_ply

    tmp = tempfile.mkdtemp()
    ply_path = os.path.join(tmp, 'torus.ply')
    write_gaussian_ply(ply_path, *make_torus_splats(300_000))
    cloud = load_gaussian_ply(ply_path)

    sg = splat_gpu.SplatGPU(cloud)
    shader = splat_gpu.get_shader()

    # tilted view so the torus reads as 3D: rotate 60deg about x, then back off
    a = np.radians(60.0)
    rx = np.array([
        [1, 0, 0, 0],
        [0, np.cos(a), -np.sin(a), 0],
        [0, np.sin(a), np.cos(a), 0],
        [0, 0, 0, 1]], np.float32)
    tz = np.eye(4, dtype=np.float32)
    tz[2, 3] = -7.0
    view = tz @ rx

    f = 2.4
    proj = np.array([
        [f, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, -1.02, -2.02],
        [0, 0, -1, 0]], np.float32)

    offs = gpu.types.GPUOffScreen(SIZE, SIZE)
    with offs.bind():
        fb = gpu.state.active_framebuffer_get()
        fb.clear(color=(0.08, 0.08, 0.1, 1.0), depth=1.0)

        import time as _time
        sg.sort_if_needed(view, 0.0)
        deadline = _time.monotonic() + 10.0
        while sg._sort_result is None and _time.monotonic() < deadline:
            _time.sleep(0.01)
        sg.sort_if_needed(view, 0.0)

        params = np.array([SIZE, SIZE, 1.0, 1.0, 0, 0, 0, 0], np.float32)
        cam = np.linalg.inv(view)[:3, 3]
        cam4 = np.array([cam[0], cam[1], cam[2], 0.0], np.float32)
        ubo_data = np.concatenate([view.T.ravel(), proj.T.ravel(), params, cam4])
        ubo = gpu.types.GPUUniformBuf(splat_gpu._np_buffer('FLOAT', ubo_data))

        gpu.state.blend_set('ALPHA')
        gpu.state.depth_test_set('LESS_EQUAL')
        gpu.state.depth_mask_set(False)
        shader.bind()
        shader.uniform_block('u', ubo)
        shader.uniform_sampler('dataTex', sg.data_tex)
        shader.uniform_sampler('orderTex', sg.order_tex)
        shader.uniform_sampler('shTex', sg.sh_tex if sg.sh_tex else sg.data_tex)
        sg.batch.draw(shader)
        gpu.state.depth_mask_set(True)
        gpu.state.depth_test_set('NONE')
        gpu.state.blend_set('NONE')

        raw = fb.read_color(0, 0, SIZE, SIZE, 4, 0, 'FLOAT')
        pixels = np.array(raw.to_list(), np.float32).reshape(SIZE, SIZE, 4)
    offs.free()

    img = bpy.data.images.new('preview', SIZE, SIZE, alpha=False)
    img.pixels = pixels.ravel().tolist()
    img.filepath_raw = OUT_PNG
    img.file_format = 'PNG'
    img.save()
    print(f'saved {OUT_PNG}')


def main():
    try:
        render()
    except Exception:
        print(traceback.format_exc())
    bpy.ops.wm.quit_blender()


bpy.app.timers.register(main, first_interval=0.5)

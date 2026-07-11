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

    def make_ubo(view, proj, size, sh_bands=0, sh_width=0, cam=None):
        params = np.array([size, size, 1.0, 1.0, 0.0,
                           sh_bands, sh_width, 0.0], np.float32)
        if cam is None:
            cam = np.linalg.inv(view)[:3, 3]
        cam4 = np.array([cam[0], cam[1], cam[2], 0.0], np.float32)
        data = np.concatenate([view.T.ravel(), proj.T.ravel(), params, cam4])
        return gpu.types.GPUUniformBuf(splat_gpu._np_buffer('FLOAT', data))

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

        ubo = make_ubo(view, proj, 256)

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

    # SH view dependence: band-1 coefficient on the -C1*x basis makes the
    # red channel differ between +x and -x viewpoints
    pos, scales, quat, sh0, opacity = make_torus_splats(50_000)
    sh_rest = np.zeros((50_000, 9), np.float32)
    sh_rest[:, 2] = 2.0  # R channel, basis[2] = -0.4886 * dir.x
    sh_ply = os.path.join(tmp, 'sh.ply')
    write_gaussian_ply(sh_ply, pos, scales, quat, sh0, opacity, sh_rest)
    sh_cloud = load_gaussian_ply(sh_ply)
    assert sh_cloud.sh_bands == 1
    sg2 = splat_gpu.SplatGPU(sh_cloud)
    assert sg2.sh_tex is not None, 'SH texture not built'

    def render_mean_red(cam_x):
        f = 2.0
        proj = np.array([
            [f, 0, 0, 0], [0, f, 0, 0],
            [0, 0, -1.02, -2.02], [0, 0, -1, 0]], np.float32)
        # camera at (cam_x, 0, 0) looking at the origin along -x or +x
        s = 1.0 if cam_x > 0 else -1.0
        view = np.array([
            [0, 0, -s, 0],
            [0, 1, 0, 0],
            [s, 0, 0, -abs(cam_x)],
            [0, 0, 0, 1]], np.float32)
        offs2 = gpu.types.GPUOffScreen(128, 128)
        with offs2.bind():
            fb2 = gpu.state.active_framebuffer_get()
            fb2.clear(color=(0.0, 0.0, 0.0, 0.0), depth=1.0)
            ubo2 = make_ubo(view, proj, 128,
                            sh_bands=1, sh_width=sg2.sh_width)
            gpu.state.blend_set('ALPHA')
            shader.bind()
            shader.uniform_block('u', ubo2)
            shader.uniform_sampler('dataTex', sg2.data_tex)
            shader.uniform_sampler('orderTex', sg2.order_tex)
            shader.uniform_sampler('shTex', sg2.sh_tex)
            sg2.batch.draw(shader)
            gpu.state.blend_set('NONE')
            raw2 = fb2.read_color(0, 0, 128, 128, 4, 0, 'FLOAT')
            px2 = np.array(raw2.to_list(), np.float32).reshape(128, 128, 4)
        offs2.free()
        lit2 = px2[px2[..., 3] > 0.1]
        return float(lit2[:, 0].mean())

    red_px = render_mean_red(8.0)   # dir.x ~ -1 -> basis positive -> red up
    red_nx = render_mean_red(-8.0)  # dir.x ~ +1 -> red down
    log(f'OK SH view dependence: red +x={red_px:.3f} vs -x={red_nx:.3f}')
    assert red_px - red_nx > 0.3, 'SH did not change color with view direction'

    # depth pick: unproject the pixel under the torus center-ring and check
    # the recovered world point sits on the torus (radius ~2 in xz)
    from pobim_splats.measure_math import unproject_pixel
    pick_shader = splat_gpu.get_pick_shader()
    log('OK pick shader compiles')

    W = H = 256
    offs3 = gpu.types.GPUOffScreen(W, H, format='RGBA32F')
    view3 = np.eye(4, dtype=np.float32)
    view3[2, 3] = -8.0
    f = 2.0
    proj3 = np.array([
        [f, 0, 0, 0], [0, f, 0, 0],
        [0, 0, -1.02, -2.02], [0, 0, -1, 0]], np.float32)
    with offs3.bind():
        fb3 = gpu.state.active_framebuffer_get()
        fb3.clear(color=(1.0, 0.0, 0.0, 0.0), depth=1.0)
        gpu.state.blend_set('NONE')
        gpu.state.depth_test_set('LESS_EQUAL')
        gpu.state.depth_mask_set(True)
        ubo3 = make_ubo(view3, proj3, W)
        pick_shader.bind()
        pick_shader.uniform_block('u', ubo3)
        pick_shader.uniform_sampler('dataTex', sg.data_tex)
        pick_shader.uniform_sampler('orderTex', sg.order_tex)
        pick_shader.uniform_sampler('shTex', sg.data_tex)
        sg.batch.draw(pick_shader)
        gpu.state.depth_mask_set(True)
        gpu.state.depth_test_set('NONE')
        # pixel on the torus ring: ~0.5 ndc to the right of center
        px_x, px_y = int(W * 0.75), int(H * 0.5)
        raw3 = fb3.read_color(px_x, px_y, 1, 1, 4, 0, 'FLOAT')
        z = float(raw3.to_list()[0][0][0])
    offs3.free()
    assert z < 1.0 - 1e-6, 'depth pick pixel is background'
    persp3 = proj3 @ view3
    world = unproject_pixel(persp3, px_x + 0.5, px_y + 0.5, z, W, H)
    ring_r = float(np.hypot(world[0], world[2]))
    log(f'OK depth pick: z={z:.4f} world=({world[0]:.2f},{world[1]:.2f},{world[2]:.2f}) '
        f'ring radius={ring_r:.2f}')
    assert 1.2 < ring_r < 2.8, f'picked point not on torus surface (r={ring_r})'

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

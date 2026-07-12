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

    def make_ubo(view, proj, size, sh_bands=0, sh_width=0, cam=None, sel4=None,
                 preview=None):
        params = np.array([size, size, 1.0, 1.0, 0.0,
                           sh_bands, sh_width, 0.0], np.float32)
        if cam is None:
            cam = np.linalg.inv(view)[:3, 3]
        cam4 = np.array([cam[0], cam[1], cam[2], 0.0], np.float32)
        # selColor.a defaults to 0 -> shader skips the state fetch, so every
        # existing test renders exactly as before with a dummy stateTex bound
        if sel4 is None:
            sel4 = np.zeros(4, np.float32)
        else:
            sel4 = np.asarray(sel4, np.float32)
        # previewMatrix defaults to identity + misc.x = 0 (inactive), keeping
        # all existing analytic tests byte-for-byte unchanged. Passing a 4x4
        # `preview` uploads it column-major and flags misc.x = 1.
        if preview is None:
            prev4 = np.eye(4, dtype=np.float32).ravel()
            misc = np.zeros(4, np.float32)
        else:
            prev4 = np.asarray(preview, np.float32).reshape(4, 4).T.ravel()
            misc = np.array([1.0, 0.0, 0.0, 0.0], np.float32)
        data = np.concatenate([view.T.ravel(), proj.T.ravel(),
                               params, cam4, sel4, prev4, misc])
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
        shader.uniform_sampler('stateTex', sg.data_tex)
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
            shader.uniform_sampler('stateTex', sg2.data_tex)
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
        pick_shader.uniform_sampler('stateTex', sg.data_tex)
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

    # --- analytic gaussian profile -------------------------------------
    # Render ONE isotropic splat and compare the alpha falloff against the
    # true gaussian. Catches size/falloff calibration bugs — e.g. the
    # missing px->NDC factor 2 that drew every splat at half size.
    from pobim_splats.ply_loader import build_cloud

    sigma_w = 0.5
    z_dist = 8.0
    f_ndc = 2.0
    SIZE = 256
    f_px = f_ndc * SIZE / 2.0
    sigma_px = f_px * sigma_w / z_dist   # = 16 px

    one = build_cloud(
        np.zeros((1, 3), np.float32),
        np.full((1, 3), sigma_w, np.float32),
        np.array([[1, 0, 0, 0]], np.float32),
        np.ones((1, 3), np.float32),
        np.ones(1, np.float32))
    sg1 = splat_gpu.SplatGPU(one)

    view1 = np.eye(4, dtype=np.float32)
    view1[2, 3] = -z_dist
    proj1 = np.array([
        [f_ndc, 0, 0, 0], [0, f_ndc, 0, 0],
        [0, 0, -1.02, -2.02], [0, 0, -1, 0]], np.float32)

    def render_single(sgx, view_m, proj_m):
        offsx = gpu.types.GPUOffScreen(SIZE, SIZE)
        with offsx.bind():
            fbx = gpu.state.active_framebuffer_get()
            fbx.clear(color=(0.0, 0.0, 0.0, 0.0), depth=1.0)
            gpu.state.blend_set('ALPHA')
            shader.bind()
            # keep the UBO referenced until the draw call — passing the
            # temporary inline lets Python free it before batch.draw
            ubox = make_ubo(view_m, proj_m, SIZE)
            shader.uniform_block('u', ubox)
            shader.uniform_sampler('dataTex', sgx.data_tex)
            shader.uniform_sampler('orderTex', sgx.order_tex)
            shader.uniform_sampler('shTex', sgx.data_tex)
            shader.uniform_sampler('stateTex', sgx.data_tex)
            sgx.batch.draw(shader)
            gpu.state.blend_set('NONE')
            rawx = fbx.read_color(0, 0, SIZE, SIZE, 4, 0, 'FLOAT')
            out = np.array(rawx.to_list(), np.float32).reshape(SIZE, SIZE, 4)
        offsx.free()
        return out

    px1 = render_single(sg1, view1, proj1)
    lit1 = np.argwhere(px1[..., 3] > 0.01)
    log(f'       single splat: lit={lit1.shape[0]} '
        f'bbox={(lit1.min(0).tolist(), lit1.max(0).tolist()) if lit1.shape[0] else None}')
    cx = SIZE // 2
    exp4 = float(np.exp(-4.0))

    def expected_alpha(r_px):
        g = float(np.exp(-(r_px ** 2) / (2.0 * sigma_px ** 2)))
        return max((g - exp4) / (1.0 - exp4), 0.0)

    a0 = float(px1[cx, cx, 3])
    a1 = float(px1[cx, cx + int(sigma_px), 3])
    a2 = float(px1[cx, cx + int(2 * sigma_px), 3])
    log(f'OK gaussian profile: a(0)={a0:.3f} a(σ)={a1:.3f}/{expected_alpha(sigma_px):.3f} '
        f'a(2σ)={a2:.3f}/{expected_alpha(2 * sigma_px):.3f}')
    assert a0 > 0.95, f'center alpha {a0}'
    assert abs(a1 - expected_alpha(sigma_px)) < 0.08, \
        f'alpha at 1σ = {a1} expected {expected_alpha(sigma_px)} — size/falloff miscalibrated'
    assert abs(a2 - expected_alpha(2 * sigma_px)) < 0.06, \
        f'alpha at 2σ = {a2} expected {expected_alpha(2 * sigma_px)}'

    # --- edge stretch bounded at wide FOV -------------------------------
    # A splat near the screen edge of a ~90° lens must not smear into a
    # radial streak (INRIA Jacobian clamp keeps its footprint bounded).
    def footprint(x_world):
        cl = build_cloud(
            np.array([[x_world, 0.0, 0.0]], np.float32),
            np.full((1, 3), 0.15, np.float32),
            np.array([[1, 0, 0, 0]], np.float32),
            np.ones((1, 3), np.float32),
            np.ones(1, np.float32))
        sgx = splat_gpu.SplatGPU(cl)
        vw = np.eye(4, dtype=np.float32)
        vw[2, 3] = -6.0
        pw = np.array([
            [1.0, 0, 0, 0], [0, 1.0, 0, 0],
            [0, 0, -1.02, -2.02], [0, 0, -1, 0]], np.float32)
        img = render_single(sgx, vw, pw)
        lit_idx = np.argwhere(img[..., 3] > 0.05)
        if lit_idx.shape[0] == 0:
            return 0.0
        return float(max(lit_idx[:, 0].max() - lit_idx[:, 0].min(),
                         lit_idx[:, 1].max() - lit_idx[:, 1].min()))

    center_ext = footprint(0.0)
    edge_ext = footprint(5.2)   # ndc x ≈ 0.87 at z=6 with f=1 (~90° lens)
    log(f'OK edge stretch: center={center_ext:.0f}px edge={edge_ext:.0f}px '
        f'ratio={edge_ext / max(center_ext, 1.0):.2f}')
    assert center_ext > 4, 'center splat did not render'
    assert edge_ext > 0, 'edge splat was wrongly culled'
    assert edge_ext < 3.5 * center_ext, \
        f'edge splat smeared into a streak ({edge_ext:.0f}px vs {center_ext:.0f}px)'

    # --- perspective tilt terms present ---------------------------------
    # A needle elongated along the VIEW axis is a dot at screen center but
    # must smear radially at the screen edge (the -f*x/z^2 Jacobian terms).
    # A transposed Jacobian (textbook row-major layout fed to the
    # column-major mat3 constructor) silently drops these terms: the edge
    # needle stays a dot and grazing surfaces at wide-lens screen edges
    # render wrong. This asserts the tilt terms actually contribute.
    def needle_footprint(x_world):
        cl = build_cloud(
            np.array([[x_world, 0.0, 0.0]], np.float32),
            np.array([[0.02, 0.02, 0.6]], np.float32),
            np.array([[1, 0, 0, 0]], np.float32),
            np.ones((1, 3), np.float32),
            np.ones(1, np.float32))
        sgx = splat_gpu.SplatGPU(cl)
        vw = np.eye(4, dtype=np.float32)
        vw[2, 3] = -6.0
        pw = np.array([
            [1.0, 0, 0, 0], [0, 1.0, 0, 0],
            [0, 0, -1.02, -2.02], [0, 0, -1, 0]], np.float32)
        img = render_single(sgx, vw, pw)
        lit_idx = np.argwhere(img[..., 3] > 0.05)
        if lit_idx.shape[0] == 0:
            return 0.0
        return float(max(lit_idx[:, 0].max() - lit_idx[:, 0].min(),
                         lit_idx[:, 1].max() - lit_idx[:, 1].min()))

    center_needle = needle_footprint(0.0)
    edge_needle = needle_footprint(5.2)
    log(f'OK perspective tilt: z-needle center={center_needle:.0f}px '
        f'edge={edge_needle:.0f}px')
    assert edge_needle > max(2.5 * center_needle, 12.0), \
        'no radial smear at edge — perspective tilt terms missing (transposed J?)'

    # --- per-splat edit state (select / hide / delete) ------------------
    # Select the first half of a torus, tint them, and verify the selection
    # tint raises red; then delete (and separately hide) the selection and
    # verify the drawn coverage drops the same way for both.
    from pobim_splats.splat_state import SplatState

    sgS = splat_gpu.SplatGPU(cloud)
    nS = sgS.count
    stS = SplatState(nS)
    # a SPATIAL half (one side of the ring) so deleting it clears a
    # contiguous screen region — a random index-half just thins the dense
    # torus without opening holes, so coverage would barely move
    half = np.nonzero(sgS.positions[:, 0] > 0.0)[0].astype(np.int64)
    ch = stS.select_indices(half, 'set')
    assert ch.size == half.size, 'select_indices did not report all changes'

    SEL = (1.0, 0.5, 0.0, 1.0)   # orange highlight for a clear red shift

    def render_state(sgx, state_tex, sel4):
        vw = np.eye(4, dtype=np.float32)
        vw[2, 3] = -8.0
        f = 2.0
        pj = np.array([
            [f, 0, 0, 0], [0, f, 0, 0],
            [0, 0, -1.02, -2.02], [0, 0, -1, 0]], np.float32)
        sgx.sort_if_needed(vw, 0.0)
        dl = _time.monotonic() + 10.0
        while sgx._sort_result is None and _time.monotonic() < dl:
            _time.sleep(0.01)
        sgx.sort_if_needed(vw, 0.0)
        offsx = gpu.types.GPUOffScreen(256, 256)
        with offsx.bind():
            fbx = gpu.state.active_framebuffer_get()
            fbx.clear(color=(0.0, 0.0, 0.0, 0.0), depth=1.0)
            gpu.state.blend_set('ALPHA')
            gpu.state.depth_test_set('LESS_EQUAL')
            gpu.state.depth_mask_set(False)
            ubox = make_ubo(vw, pj, 256, sel4=sel4)
            shader.bind()
            shader.uniform_block('u', ubox)
            shader.uniform_sampler('dataTex', sgx.data_tex)
            shader.uniform_sampler('orderTex', sgx.order_tex)
            shader.uniform_sampler('shTex', sgx.data_tex)
            shader.uniform_sampler('stateTex', state_tex if state_tex else sgx.data_tex)
            sgx.batch.draw(shader)
            gpu.state.depth_mask_set(True)
            gpu.state.depth_test_set('NONE')
            gpu.state.blend_set('NONE')
            rawx = fbx.read_color(0, 0, 256, 256, 4, 0, 'FLOAT')
            out = np.array(rawx.to_list(), np.float32).reshape(256, 256, 4)
        offsx.free()
        return out

    sgS.upload_state(stS.flags)

    # same geometry rendered with and without the tint (a=0 skips the fetch)
    base = render_state(sgS, sgS.state_tex, (0.0, 0.0, 0.0, 0.0))
    tint = render_state(sgS, sgS.state_tex, SEL)
    red_base = float(base[base[..., 3] > 0.1][:, 0].mean())
    red_tint = float(tint[tint[..., 3] > 0.1][:, 0].mean())
    log(f'OK selection tint: mean red {red_base:.3f} -> {red_tint:.3f}')
    assert red_tint > red_base + 0.02, 'selection tint did not raise red'

    cov_sel = float((tint[..., 3] > 0.01).mean())   # nothing deleted yet

    # delete the selection and re-upload: hidden/deleted are culled (a>0.5)
    stS.delete_selected()
    sgS.upload_state(stS.flags)
    deleted = render_state(sgS, sgS.state_tex, SEL)
    cov_del = float((deleted[..., 3] > 0.01).mean())
    log(f'OK delete coverage: {cov_sel:.1%} -> {cov_del:.1%}')
    assert cov_del < cov_sel * 0.9, \
        f'coverage did not drop after delete ({cov_sel:.1%} -> {cov_del:.1%})'

    # hiding the same splats must look like deleting them
    stH = SplatState(nS)
    stH.select_indices(half, 'set')
    stH.hide_selected()
    sgS.upload_state(stH.flags)
    hidden = render_state(sgS, sgS.state_tex, SEL)
    cov_hid = float((hidden[..., 3] > 0.01).mean())
    log(f'OK hidden coverage: {cov_hid:.1%} (delete was {cov_del:.1%})')
    assert abs(cov_hid - cov_del) < 0.03, \
        f'hidden ({cov_hid:.1%}) does not match deleted ({cov_del:.1%})'

    # --- transform preview + commit (Phase 3, Track T) ------------------
    # A SELECTED splat must move under the GPU preview matrix (no texture
    # re-upload), and update_splats must move it PERMANENTLY, with the async
    # sort still functioning after the in-place position update.
    from pobim_splats.splat_gpu import recompute_cov6

    one_t = build_cloud(
        np.zeros((1, 3), np.float32),
        np.full((1, 3), 0.5, np.float32),
        np.array([[1, 0, 0, 0]], np.float32),
        np.ones((1, 3), np.float32),
        np.ones(1, np.float32))
    sgT = splat_gpu.SplatGPU(one_t)
    stT = SplatState(1)
    stT.select_indices([0], 'set')   # preview only moves SELECTED splats
    sgT.upload_state(stT.flags)

    viewT = np.eye(4, dtype=np.float32)
    viewT[2, 3] = -8.0
    projT = np.array([
        [2.0, 0, 0, 0], [0, 2.0, 0, 0],
        [0, 0, -1.02, -2.02], [0, 0, -1, 0]], np.float32)

    def centroid_x(preview=None):
        offsx = gpu.types.GPUOffScreen(256, 256)
        with offsx.bind():
            fbx = gpu.state.active_framebuffer_get()
            fbx.clear(color=(0.0, 0.0, 0.0, 0.0), depth=1.0)
            gpu.state.blend_set('ALPHA')
            # SEL (a>0.5) binds the state tex so stFlags SELECTED is read,
            # which the preview requires; identity preview when None
            ubox = make_ubo(viewT, projT, 256, sel4=SEL, preview=preview)
            shader.bind()
            shader.uniform_block('u', ubox)
            shader.uniform_sampler('dataTex', sgT.data_tex)
            shader.uniform_sampler('orderTex', sgT.order_tex)
            shader.uniform_sampler('shTex', sgT.data_tex)
            shader.uniform_sampler('stateTex', sgT.state_tex)
            sgT.batch.draw(shader)
            gpu.state.blend_set('NONE')
            rawx = fbx.read_color(0, 0, 256, 256, 4, 0, 'FLOAT')
            out = np.array(rawx.to_list(), np.float32).reshape(256, 256, 4)
        offsx.free()
        a = out[..., 3]
        ys, xs = np.nonzero(a > 0.05)
        if xs.size == 0:
            return None
        return float((xs * a[ys, xs]).sum() / a[ys, xs].sum())

    cx_base = centroid_x(None)
    tx = np.eye(4, dtype=np.float32)
    tx[0, 3] = 2.0   # translate +2 in local x
    cx_prev = centroid_x(tx)
    log(f'OK transform preview: centroid x {cx_base:.1f} -> {cx_prev:.1f}')
    assert cx_base is not None and cx_prev is not None, 'splat did not render'
    assert cx_prev - cx_base > 15.0, 'preview did not move the selected splat'

    # commit: update_splats moves it permanently; a render WITHOUT preview now
    # shows the new location, and the sort still completes afterwards.
    new_pos = np.array([[2.0, 0.0, 0.0]], np.float32)
    new_cov = recompute_cov6(np.array([[1, 0, 0, 0]], np.float32),
                             np.log(np.full((1, 3), 0.5, np.float32)))
    sgT.update_splats([0], new_pos, new_cov)
    cx_commit = centroid_x(None)   # no preview -> geometry itself moved
    log(f'OK transform commit: centroid x -> {cx_commit:.1f}')
    assert cx_commit - cx_base > 15.0, 'commit did not move the splat permanently'
    assert abs(cx_commit - cx_prev) < 20.0, 'commit location disagrees with preview'

    sgT.sort_if_needed(viewT, 0.0)
    dlT = _time.monotonic() + 10.0
    while sgT._sort_result is None and _time.monotonic() < dlT:
        _time.sleep(0.01)
    sgT.sort_if_needed(viewT, 0.0)
    assert sgT._applied_dir is not None, 'sort did not complete after commit'
    log('OK sort completes after transform commit')

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

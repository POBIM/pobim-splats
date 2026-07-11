# Viewport renderer for 3D Gaussian Splats.
#
# Architecture (same as web splat viewers, incl. PlayCanvas/SuperSplat):
# - per-splat data (center, covariance, color, opacity) lives in a float
#   texture uploaded once
# - a single static triangle batch draws 2 triangles per splat; the vertex
#   shader looks up its splat through an order texture and builds a
#   screen-space ellipse from the projected 3D covariance (EWA splatting)
# - splats are alpha-blended back-to-front; the draw order is a numpy argsort
#   of view-space depth, refreshed (throttled) when the camera or the object
#   moves, and re-uploaded as a small R32I texture
#
# One draw call per splat object — no per-splat Blender objects, no geometry
# nodes, which is what keeps this light at millions of splats.

import threading
import time

import bpy
import gpu
import numpy as np

# resort when the object-space view direction rotates more than ~1 degree.
# depth order depends only on that direction — translation and zoom never
# change it — so panning/zooming costs zero sorts.
SORT_COS_THRESHOLD = 0.99985

# data texture: 3 texels per splat, rows hold a whole number of splats
DATA_TEX_WIDTH = 2046
TEXELS_PER_SPLAT = 3
SPLATS_PER_ROW = DATA_TEX_WIDTH // TEXELS_PER_SPLAT
ORDER_TEX_WIDTH = 2048
MAX_TEX_HEIGHT = 16384
MAX_SPLATS = SPLATS_PER_ROW * MAX_TEX_HEIGHT  # ~11.1M

VERT_SRC = """
const vec2 kCorners[4] = vec2[4](
    vec2(-2.0, -2.0), vec2(2.0, -2.0), vec2(2.0, 2.0), vec2(-2.0, 2.0));

void emitDegenerate()
{
    gl_Position = vec4(0.0, 0.0, 2.0, 0.0);
    vColor = vec4(0.0);
    vQuad = vec2(0.0);
}

void main()
{
    int quad = int(quadId + 0.5);
    int corner = int(cornerId + 0.5);

    // order texture is R32F: float32 holds integers exactly up to 2^24,
    // above our MAX_SPLATS cap (Blender's Python API can only upload FLOAT buffers)
    int splat = int(texelFetch(orderTex, ivec2(quad % 2048, quad / 2048), 0).r + 0.5);

    int base = splat * 3;
    vec4 d0 = texelFetch(dataTex, ivec2(base % 2046, base / 2046), 0);
    vec4 d1 = texelFetch(dataTex, ivec2((base + 1) % 2046, (base + 1) / 2046), 0);
    vec4 d2 = texelFetch(dataTex, ivec2((base + 2) % 2046, (base + 2) / 2046), 0);

    vec4 cam = u.modelView * vec4(d0.xyz, 1.0);
    vec4 pos2d = u.projection * cam;

    float clipw = 1.2 * pos2d.w;
    if (pos2d.z < -clipw ||
        pos2d.x < -clipw || pos2d.x > clipw ||
        pos2d.y < -clipw || pos2d.y > clipw) {
        emitDegenerate();
        return;
    }

    vec2 vp = u.params.xy;
    float fx = u.projection[0][0] * vp.x * 0.5;
    float fy = u.projection[1][1] * vp.y * 0.5;

    // Jacobian of the projection at the splat center (EWA splatting).
    // Orthographic projections have a constant Jacobian.
    mat3 J;
    if (u.projection[3][3] == 1.0) {
        J = mat3(fx, 0.0, 0.0,
                 0.0, fy, 0.0,
                 0.0, 0.0, 0.0);
    } else {
        float iz = 1.0 / cam.z;
        J = mat3(fx * iz, 0.0, 0.0,
                 0.0, fy * iz, 0.0,
                 -fx * cam.x * iz * iz, -fy * cam.y * iz * iz, 0.0);
    }

    mat3 Vrk = mat3(d1.x, d1.y, d1.z,
                    d1.y, d2.x, d2.y,
                    d1.z, d2.y, d2.z);

    // includes model rotation/scale: cov2d = J * A * Vrk * A^T * J^T
    mat3 W = transpose(mat3(u.modelView));
    mat3 T = W * J;
    mat3 cov2dm = transpose(T) * Vrk * T;

    // +0.3px dilation as in the reference 3DGS rasterizer (antialias floor)
    float cxx = cov2dm[0][0] + 0.3;
    float cxy = cov2dm[0][1];
    float cyy = cov2dm[1][1] + 0.3;

    float mid = 0.5 * (cxx + cyy);
    float radius = length(vec2(0.5 * (cxx - cyy), cxy));
    float lambda1 = mid + radius;
    float lambda2 = mid - radius;
    if (lambda2 < 0.0) {
        emitDegenerate();
        return;
    }

    // guard the eigenvector against (0,0) when the ellipse is axis-aligned
    // with cxx >= cyy (or a perfect circle); x-axis is the correct fallback
    vec2 dv = vec2(cxy, lambda1 - cxx);
    vec2 diagv = dot(dv, dv) > 1e-12 ? normalize(dv) : vec2(1.0, 0.0);
    float sizeMul = u.params.z;
    vec2 majorAxis = min(sqrt(2.0 * lambda1), 1024.0) * diagv * sizeMul;
    vec2 minorAxis = min(sqrt(2.0 * lambda2), 1024.0) * vec2(diagv.y, -diagv.x) * sizeMul;

    uint pc = floatBitsToUint(d1.w);
    vec3 rgb = vec3(float((pc >> 16u) & 255u),
                    float((pc >> 8u) & 255u),
                    float(pc & 255u)) / 255.0;

    vColor = vec4(rgb, d0.w * u.params.w);
    vec2 c = kCorners[corner];
    vQuad = c;

    vec2 center = pos2d.xy / pos2d.w;
    gl_Position = vec4(
        center + c.x * majorAxis / vp + c.y * minorAxis / vp,
        pos2d.z / pos2d.w,
        1.0);
}
"""

FRAG_SRC = """
void main()
{
    float a = -dot(vQuad, vQuad);
    if (a < -4.0) {
        discard;
    }
    float alpha = exp(a) * vColor.a;
    if (alpha < 0.004) {
        discard;
    }
    FragColor = vec4(vColor.rgb, alpha);
}
"""

_shader = None


def get_shader():
    global _shader
    if _shader is None:
        info = gpu.types.GPUShaderCreateInfo()
        info.typedef_source(
            'struct SplatUniforms {'
            '  mat4 modelView;'
            '  mat4 projection;'
            '  vec4 params;'    # viewport w, viewport h, size multiplier, opacity multiplier
            '  vec4 params2;'   # reserved
            '};')
        info.uniform_buf(0, 'SplatUniforms', 'u')
        info.sampler(0, 'FLOAT_2D', 'dataTex')
        info.sampler(1, 'FLOAT_2D', 'orderTex')
        info.vertex_in(0, 'FLOAT', 'quadId')
        info.vertex_in(1, 'FLOAT', 'cornerId')
        iface = gpu.types.GPUStageInterfaceInfo('pobim_splat_iface')
        iface.smooth('VEC4', 'vColor')
        iface.smooth('VEC2', 'vQuad')
        info.vertex_out(iface)
        info.fragment_out(0, 'VEC4', 'FragColor')
        info.vertex_source(VERT_SRC)
        info.fragment_source(FRAG_SRC)
        _shader = gpu.shader.create_from_info(info)
    return _shader


def _np_buffer(btype, arr):
    """gpu.types.Buffer from a numpy array, with a slow-path fallback."""
    arr = np.ascontiguousarray(arr)
    try:
        return gpu.types.Buffer(btype, arr.shape[0], arr)
    except Exception:
        return gpu.types.Buffer(btype, arr.shape[0], arr.tolist())


def _attr_fill(vbo, attr_id, arr):
    try:
        vbo.attr_fill(id=attr_id, data=arr)
    except Exception:
        vbo.attr_fill(id=attr_id, data=arr.tolist())


class SplatGPU:
    """GPU resources for one splat cloud."""

    def __init__(self, cloud):
        n = cloud.count
        if n > MAX_SPLATS:
            raise ValueError(f'มากเกินไป ({n:,} splats, สูงสุด {MAX_SPLATS:,}) — ใช้ Max Splats ตอน import')

        self.count = n
        self.positions = cloud.positions  # kept for depth sorting

        rows = (n + SPLATS_PER_ROW - 1) // SPLATS_PER_ROW
        texels = np.zeros((rows * DATA_TEX_WIDTH, 4), np.float32)

        c8 = np.clip(np.round(cloud.colors * 255.0), 0, 255).astype(np.uint32)
        packed = ((c8[:, 0] << 16) | (c8[:, 1] << 8) | c8[:, 2]).view(np.float32)

        base = np.arange(n, dtype=np.int64) * TEXELS_PER_SPLAT
        texels[base, :3] = cloud.positions
        texels[base, 3] = cloud.opacities
        texels[base + 1, :3] = cloud.cov6[:, :3]
        texels[base + 1, 3] = packed
        texels[base + 2, :3] = cloud.cov6[:, 3:]

        self.data_tex = gpu.types.GPUTexture(
            (DATA_TEX_WIDTH, rows), format='RGBA32F',
            data=_np_buffer('FLOAT', texels.ravel()))

        fmt = gpu.types.GPUVertFormat()
        fmt.attr_add(id='quadId', comp_type='F32', len=1, fetch_mode='FLOAT')
        fmt.attr_add(id='cornerId', comp_type='F32', len=1, fetch_mode='FLOAT')
        vbo = gpu.types.GPUVertBuf(fmt, n * 6)
        _attr_fill(vbo, 'quadId', np.repeat(np.arange(n, dtype=np.float32), 6))
        _attr_fill(vbo, 'cornerId', np.tile(np.array([0, 1, 2, 0, 2, 3], np.float32), n))
        self.vbo = vbo
        self.batch = gpu.types.GPUBatch(type='TRIS', buf=vbo)

        self.order_rows = (n + ORDER_TEX_WIDTH - 1) // ORDER_TEX_WIDTH
        self.order_tex = None
        self.last_sort_time = 0.0
        self._applied_dir = None      # object-space view dir of applied order
        self._sort_pending = False    # a worker thread is running
        self._sort_result = None      # (order, dir) ready for pickup
        self._upload_order(np.arange(n, dtype=np.int32))

    def _upload_order(self, order):
        padded = np.zeros(self.order_rows * ORDER_TEX_WIDTH, np.float32)
        padded[:self.count] = order.astype(np.float32)
        self.order_tex = gpu.types.GPUTexture(
            (ORDER_TEX_WIDTH, self.order_rows), format='R32F',
            data=_np_buffer('FLOAT', padded))

    def sort_if_needed(self, model_view, interval):
        """Back-to-front ordering, computed off the draw thread.

        The draw handler stays fast: it only picks up finished results
        (GPU upload) and decides whether to launch a new argsort worker.
        numpy's sort releases the GIL, so the viewport keeps drawing.
        """
        result = self._sort_result
        if result is not None:
            self._sort_result = None
            order, direction = result
            self._upload_order(order)
            self._applied_dir = direction

        if self._sort_pending:
            return

        direction = model_view[2, :3].astype(np.float64)
        norm = np.linalg.norm(direction)
        if norm == 0.0:
            return
        direction /= norm
        if (self._applied_dir is not None and
                float(direction @ self._applied_dir) > SORT_COS_THRESHOLD):
            return
        now = time.monotonic()
        if now - self.last_sort_time < interval:
            return

        self.last_sort_time = now
        self._sort_pending = True
        row = model_view[2, :3].copy()
        positions = self.positions

        def job():
            try:
                # view-space z (negative in front); ascending = farthest first.
                # the constant translation term never changes the order.
                order = np.argsort(positions @ row).astype(np.int32)
                self._sort_result = (order, direction)
            except Exception as e:
                print(f'[pobim_splats] sort failed: {e}')
            finally:
                self._sort_pending = False
                if _draw_handle is not None:  # skip after addon unregister
                    try:
                        # timers.register is the documented thread-safe way
                        # to poke the main thread for a redraw
                        bpy.app.timers.register(_redraw_once, first_interval=0.0)
                    except Exception:
                        pass

        threading.Thread(target=job, daemon=True).start()


def _redraw_once():
    redraw_viewports()
    return None


class SplatEntry:
    """Registry entry: parsed cloud + lazily-built GPU resources.

    GPU objects are created inside the draw handler, where a GPU context is
    guaranteed, so imports and file-open reloads never touch the GPU.
    """

    def __init__(self, uid, cloud):
        self.uid = uid
        self.cloud = cloud
        self.gpu = None
        self.error = None


# uid -> SplatEntry (session state; rebuilt from object properties on file open)
REGISTRY = {}

_draw_handle = None


def redraw_viewports():
    wm = bpy.context.window_manager
    if wm is None:
        return
    for window in wm.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


def _splat_objects():
    """Map uid -> object for all splat empties in the current file."""
    result = {}
    for obj in bpy.data.objects:
        uid = getattr(obj, 'pobim_splat_uid', '')
        if uid:
            result[uid] = obj
    return result


def _draw_callback():
    context = bpy.context
    region = context.region
    rv3d = context.region_data
    if region is None or rv3d is None or not REGISTRY:
        return

    scene = context.scene
    if not getattr(scene, 'pobim_splats_enabled', True):
        return
    interval = getattr(scene, 'pobim_splat_sort_interval', 0.5)

    objects = _splat_objects()
    view = np.array(rv3d.view_matrix, np.float32)
    proj = np.array(rv3d.window_matrix, np.float32)

    # gather visible entries with their model-view, farthest object first so
    # inter-object blending is approximately correct
    draw_list = []
    for uid, entry in REGISTRY.items():
        obj = objects.get(uid)
        if obj is None or entry.error is not None:
            continue
        try:
            if not obj.visible_get():
                continue
        except Exception:
            continue
        model = np.array(obj.matrix_world, np.float32)
        mv = view @ model
        draw_list.append((mv[2, 3], entry, obj, mv))
    if not draw_list:
        return
    draw_list.sort(key=lambda item: item[0])

    try:
        shader = get_shader()
    except Exception as e:
        _fail_all(f'shader compile failed: {e}')
        return

    gpu.state.blend_set('ALPHA')
    gpu.state.depth_test_set('LESS_EQUAL')
    gpu.state.depth_mask_set(False)
    try:
        for _, entry, obj, mv in draw_list:
            if entry.gpu is None:
                try:
                    entry.gpu = SplatGPU(entry.cloud)
                    entry.cloud.cov6 = entry.cloud.colors = entry.cloud.opacities = None
                except Exception as e:
                    entry.error = str(e)
                    print(f'[pobim_splats] GPU build failed for {obj.name}: {e}')
                    continue

            sg = entry.gpu
            sg.sort_if_needed(mv, interval)

            params = np.array([
                region.width, region.height,
                getattr(obj, 'pobim_splat_scale', 1.0),
                getattr(obj, 'pobim_splat_opacity', 1.0),
                0.0, 0.0, 0.0, 0.0], np.float32)
            ubo_data = np.concatenate([mv.T.ravel(), proj.T.ravel(), params])
            ubo = gpu.types.GPUUniformBuf(_np_buffer('FLOAT', ubo_data))

            shader.bind()
            shader.uniform_block('u', ubo)
            shader.uniform_sampler('dataTex', sg.data_tex)
            shader.uniform_sampler('orderTex', sg.order_tex)
            sg.batch.draw(shader)
    finally:
        gpu.state.depth_mask_set(True)
        gpu.state.depth_test_set('NONE')
        gpu.state.blend_set('NONE')


def _fail_all(message):
    for entry in REGISTRY.values():
        if entry.error is None:
            entry.error = message
    print(f'[pobim_splats] {message}')


def purge_orphans():
    """Drop registry entries whose object no longer exists."""
    objects = _splat_objects()
    for uid in [uid for uid in REGISTRY if uid not in objects]:
        del REGISTRY[uid]


def load_entry_for_object(obj):
    """(Re)load the splat file referenced by a splat empty into the registry.

    Dispatches on extension: .ply (standard or compressed), .sog bundle,
    or an unbundled SOG meta.json.
    """
    import os

    filepath = bpy.path.abspath(obj.pobim_splat_file)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f'ไม่พบไฟล์: {filepath}')

    lower = filepath.lower()
    if lower.endswith('.sog') or lower.endswith('.json'):
        from .sog_loader import load_sog as loader
    else:
        from .ply_loader import load_gaussian_ply as loader

    cloud = loader(
        filepath,
        max_splats=obj.pobim_splat_max,
        srgb_to_linear=obj.pobim_splat_srgb)
    obj.pobim_splat_count = cloud.count
    REGISTRY[obj.pobim_splat_uid] = SplatEntry(obj.pobim_splat_uid, cloud)
    return cloud


# --- lifecycle reconciliation -------------------------------------------
#
# The registry is Python session state; Blender's object lifecycle is not.
# Three workflows desync them:
# - Shift+D/Alt+D copies the object INCLUDING pobim_splat_uid (registered
#   properties are copied), so two objects collide on one registry entry
# - undo of Remove restores the object but not the popped registry entry
# - deleting the empty with X leaves the entry (and its VRAM) orphaned
#
# A depsgraph handler detects membership changes and schedules a reconcile
# on a one-shot timer (depsgraph callbacks must not write ID data directly).

_last_signature = None
_reconcile_scheduled = False


def _state_signature():
    return tuple(sorted(
        (obj.name, obj.pobim_splat_uid)
        for obj in bpy.data.objects if obj.pobim_splat_uid))


def reconcile():
    """Fix duplicate uids, purge orphans, rebuild missing entries."""
    global _last_signature
    import uuid

    seen = set()
    for obj in bpy.data.objects:
        if not obj.pobim_splat_uid:
            continue
        if obj.pobim_splat_uid in seen:
            obj.pobim_splat_uid = uuid.uuid4().hex  # duplicated object
        seen.add(obj.pobim_splat_uid)

    purge_orphans()

    for obj in bpy.data.objects:
        uid = obj.pobim_splat_uid
        if uid and uid not in REGISTRY and obj.pobim_splat_file:
            try:
                load_entry_for_object(obj)
            except Exception as e:
                print(f'[pobim_splats] reload failed for {obj.name}: {e}')

    _last_signature = _state_signature()
    redraw_viewports()


def _reconcile_timer():
    global _reconcile_scheduled
    _reconcile_scheduled = False
    try:
        reconcile()
    except Exception as e:
        print(f'[pobim_splats] reconcile failed: {e}')
    return None


def on_depsgraph_update(_scene, _depsgraph=None):
    global _reconcile_scheduled
    if _reconcile_scheduled:
        return
    if _state_signature() == _last_signature:
        return
    _reconcile_scheduled = True
    bpy.app.timers.register(_reconcile_timer, first_interval=0.0)


def register_draw_handler():
    global _draw_handle
    if _draw_handle is None:
        _draw_handle = bpy.types.SpaceView3D.draw_handler_add(
            _draw_callback, (), 'WINDOW', 'POST_VIEW')


def unregister_draw_handler():
    global _draw_handle
    if _draw_handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_draw_handle, 'WINDOW')
        _draw_handle = None
    REGISTRY.clear()

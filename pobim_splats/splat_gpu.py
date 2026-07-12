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

# tint applied to selected splats — POBIMStudio's default selected colour
# (DEFAULT_SELECTED_CLR in the web editor's scene-config.ts: pure yellow)
SELECTED_COLOR = (1.0, 1.0, 0.0)

VERT_SRC = """
const vec2 kCorners[4] = vec2[4](
    vec2(-2.0, -2.0), vec2(2.0, -2.0), vec2(2.0, 2.0), vec2(-2.0, 2.0));

void emitDegenerate()
{
    gl_Position = vec4(0.0, 0.0, 2.0, 0.0);
    vColor = vec4(0.0);
    vQuad = vec2(0.0);
}

// SH texture words for the current splat (filled before evalSH)
uvec4 g_shw0;
uvec4 g_shw1;
uvec4 g_shw2;

// byte-packed SH coefficient i (4 per float, 16 per texel), range +-4
float shCoef(int i)
{
    int t = i >> 4;
    uvec4 w = t == 0 ? g_shw0 : (t == 1 ? g_shw1 : g_shw2);
    uint b = (w[(i >> 2) & 3] >> uint(8 * (i & 3))) & 255u;
    return (float(b) / 255.0 - 0.5) * 8.0;
}

// view-dependent color offset from SH bands 1..3 (INRIA basis and layout)
vec3 evalSH(int bands, int coeffsN, vec3 dir)
{
    float x = dir.x, y = dir.y, z = dir.z;
    float xx = x * x, yy = y * y, zz = z * z;
    float xy = x * y, yz = y * z, xz = x * z;

    float basis[15];
    basis[0] = -0.48860251 * y;
    basis[1] = 0.48860251 * z;
    basis[2] = -0.48860251 * x;
    if (bands >= 2) {
        basis[3] = 1.09254843 * xy;
        basis[4] = -1.09254843 * yz;
        basis[5] = 0.31539157 * (2.0 * zz - xx - yy);
        basis[6] = -1.09254843 * xz;
        basis[7] = 0.54627421 * (xx - yy);
    }
    if (bands >= 3) {
        basis[8] = -0.59004359 * y * (3.0 * xx - yy);
        basis[9] = 2.89061144 * xy * z;
        basis[10] = -0.45704580 * y * (4.0 * zz - xx - yy);
        basis[11] = 0.37317633 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy);
        basis[12] = -0.45704580 * x * (4.0 * zz - xx - yy);
        basis[13] = 1.44530572 * z * (xx - yy);
        basis[14] = -0.59004359 * x * (xx - yy);
    }

    vec3 result = vec3(0.0);
    for (int ch = 0; ch < 3; ch++) {
        float acc = 0.0;
        for (int k = 0; k < coeffsN; k++) {
            acc += basis[k] * shCoef(ch * coeffsN + k);
        }
        result[ch] = acc;
    }
    return result;
}

void main()
{
    int quad = int(quadId + 0.5);
    int corner = int(cornerId + 0.5);

    // order texture is R32F: float32 holds integers exactly up to 2^24,
    // above our MAX_SPLATS cap (Blender's Python API can only upload FLOAT buffers)
    int splat = int(texelFetch(orderTex, ivec2(quad % 2048, quad / 2048), 0).r + 0.5);

    // per-splat edit state (selection / hidden / deleted). selColor.a is 0
    // when no state texture is bound, so untouched clouds pay zero cost.
    int stFlags = 0;
    if (u.selColor.a > 0.5) {
        float st = texelFetch(stateTex, ivec2(splat % 2048, splat / 2048), 0).r;
        stFlags = int(st + 0.5);
        if ((stFlags & 6) != 0) {   // HIDDEN (2) | DELETED (4)
            emitDegenerate();
            return;
        }
    }

    int base = splat * 3;
    vec4 d0 = texelFetch(dataTex, ivec2(base % 2046, base / 2046), 0);
    vec4 d1 = texelFetch(dataTex, ivec2((base + 1) % 2046, (base + 1) / 2046), 0);
    vec4 d2 = texelFetch(dataTex, ivec2((base + 2) % 2046, (base + 2) / 2046), 0);

    // live drag preview (no per-frame texture re-upload): apply previewMatrix
    // in SPLAT-LOCAL space, before modelView. It only fires when misc.x > 0.5
    // AND the splat is SELECTED (bit 0 of stFlags). stFlags is 0 unless the
    // state texture was sampled (selColor.a > 0.5), so preview REQUIRES an
    // active edit state — an untouched cloud never previews.
    bool preview = (u.misc.x > 0.5 && (stFlags & 1) != 0);
    vec3 localCenter = d0.xyz;
    if (preview) {
        localCenter = (u.previewMatrix * vec4(localCenter, 1.0)).xyz;
    }

    vec4 cam = u.modelView * vec4(localCenter, 1.0);
    bool ortho = u.projection[3][3] == 1.0;

    // behind the camera, and closer than the near-cull distance
    // (u.camPos.w): indoor scans carry floater gaussians along the capture
    // path that smear across the whole screen when the camera walks through
    // them — web viewers hide these behind a larger camera near plane
    if (!ortho && cam.z > -u.camPos.w) {
        emitDegenerate();
        return;
    }

    vec4 pos2d = u.projection * cam;
    // never clip against near/far: clamp depth instead (engine behavior),
    // so walking through a scan doesn't pop splats at the near plane
    pos2d.z = clamp(pos2d.z, -abs(pos2d.w), abs(pos2d.w));

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
        // INRIA reference rasterizer: clamp the point used for the Jacobian
        // to 1.3x the frustum edge (guards splats outside the frustum).
        float limx = 1.3 / u.projection[0][0];
        float limy = 1.3 / u.projection[1][1];
        float iz = 1.0 / cam.z;
        float tx = clamp(cam.x * iz, -limx, limx) * cam.z;
        float ty = clamp(cam.y * iz, -limy, limy) * cam.z;
        // LAYOUT MATTERS: the perspective tilt terms belong in the BOTTOM
        // ROW (third component of columns 0/1), matching the engine and the
        // INRIA CUDA rasterizer. With cov = J^T * Sigma * J, putting them in
        // the right column (the textbook row-major form) silently drops the
        // foreshortening — center splats render fine but periphery splats
        // on wide lenses bloat into radial streaks.
        J = mat3(fx * iz, 0.0, -fx * tx * iz * iz,
                 0.0, fy * iz, -fy * ty * iz * iz,
                 0.0, 0.0, 0.0);
    }

    mat3 Vrk = mat3(d1.x, d1.y, d1.z,
                    d1.y, d2.x, d2.y,
                    d1.z, d2.y, d2.z);

    // preview covariance: conjugate by the previewMatrix rotation/scale block.
    // mat3(previewMatrix) is the column-major upper-left 3x3 (uploaded m.T so
    // it reads correctly here). Vrk' = P3 * Vrk * P3^T is exact for
    // rotation+uniform scale and an accepted approximation for per-axis scale.
    if (preview) {
        mat3 P3 = mat3(u.previewMatrix);
        Vrk = P3 * Vrk * transpose(P3);
    }

    // includes model rotation/scale: cov2d = J * A * Vrk * A^T * J^T
    mat3 W = transpose(mat3(u.modelView));
    mat3 T = W * J;
    mat3 cov2dm = transpose(T) * Vrk * T;

    // Kernel handling matched to the PlayCanvas engine (gsplatCorner.js):
    // +0.3px dilation on the diagonal and a minimum minor eigenvalue of 0.1
    // so thin/sub-pixel splats never collapse below ~1px — this is what makes
    // surfaces read as continuous at Splat Size 1. The optional AA factor
    // (params2.x) is the engine's GSPLAT_AA energy compensation: sharper and
    // physically correct in the distance, but more translucent.
    float sizeMul = u.params.z;
    float sizeSq = sizeMul * sizeMul;
    float cxx = cov2dm[0][0] * sizeSq;
    float cxy = cov2dm[0][1] * sizeSq;
    float cyy = cov2dm[1][1] * sizeSq;

    float aaFactor = 1.0;
    if (u.params2.x > 0.5) {
        float detOrig = max(cxx * cyy - cxy * cxy, 0.0);
        float detBlur = (cxx + 0.3) * (cyy + 0.3) - cxy * cxy;
        aaFactor = sqrt(max(detOrig / max(detBlur, 1e-12), 0.0));
    }
    cxx += 0.3;
    cyy += 0.3;

    float mid = 0.5 * (cxx + cyy);
    float radius = length(vec2(0.5 * (cxx - cyy), cxy));
    float lambda1 = mid + radius;
    float lambda2 = max(mid - radius, 0.1);

    // guard the eigenvector against (0,0) when the ellipse is axis-aligned
    // with cxx >= cyy (or a perfect circle); x-axis is the correct fallback
    vec2 dv = vec2(cxy, lambda1 - cxx);
    vec2 diagv = dot(dv, dv) > 1e-12 ? normalize(dv) : vec2(1.0, 0.0);
    // half of the engine's on-screen cap (the quad is scaled 2x below)
    float vmin = 0.5 * min(1024.0, min(vp.x, vp.y));
    float l1 = min(sqrt(2.0 * lambda1), vmin);
    float l2 = min(sqrt(2.0 * lambda2), vmin);
    vec2 majorAxis = l1 * diagv;
    vec2 minorAxis = l2 * vec2(diagv.y, -diagv.x);

    // extent-aware frustum cull (engine gsplatCorner.js): reject only when
    // the whole ellipse is outside — center-based culling pops large splats
    // at the screen edges with wide lenses
    float extentNdc = 4.0 * max(l1, l2) / min(vp.x, vp.y);
    if (any(greaterThan(abs(pos2d.xy) - vec2(extentNdc) * abs(pos2d.w),
                        vec2(abs(pos2d.w))))) {
        emitDegenerate();
        return;
    }

    uint pc = floatBitsToUint(d1.w);
    vec3 rgb = vec3(float((pc >> 16u) & 255u),
                    float((pc >> 8u) & 255u),
                    float(pc & 255u)) / 255.0;

    // view-dependent color: SH bands evaluated in object space
    int shBands = int(u.params2.y + 0.5);
    if (shBands > 0) {
        int coeffsN = shBands == 1 ? 3 : (shBands == 2 ? 8 : 15);
        int tps = (3 * coeffsN + 15) / 16;
        int shW = int(u.params2.z + 0.5);
        int sbase = splat * tps;
        g_shw0 = floatBitsToUint(
            texelFetch(shTex, ivec2(sbase % shW, sbase / shW), 0));
        g_shw1 = tps > 1 ? floatBitsToUint(
            texelFetch(shTex, ivec2((sbase + 1) % shW, (sbase + 1) / shW), 0)) : uvec4(0u);
        g_shw2 = tps > 2 ? floatBitsToUint(
            texelFetch(shTex, ivec2((sbase + 2) % shW, (sbase + 2) / shW), 0)) : uvec4(0u);
        vec3 dir = normalize(d0.xyz - u.camPos.xyz);
        rgb += evalSH(shBands, coeffsN, dir);
    }

    // colors live in SH space up to here; convert to scene-linear at the end
    // so Blender's Standard view transform reproduces web-viewer colors
    if (u.params2.w > 0.5) {
        rgb = clamp(rgb, vec3(0.0), vec3(1.0));
        rgb = mix(rgb / 12.92,
                  pow((rgb + vec3(0.055)) / 1.055, vec3(2.4)),
                  step(vec3(0.04045), rgb));
    }

    // tint selected splats — applied after the sRGB conversion so the
    // highlight colour lands in the same space Blender then displays
    if ((stFlags & 1) != 0) {
        rgb = mix(rgb, u.selColor.rgb, 0.55);
    }

    vColor = vec4(rgb, d0.w * u.params.w * aaFactor);
    vec2 c = kCorners[corner];
    vQuad = c;

    // The 2.0 factor converts the pixel half-extent to NDC (NDC spans 2
    // units across the viewport). The antimatter15-derived shader this was
    // ported from omits it, drawing every splat at HALF its true screen
    // size — the root cause of thin/fuzzy surfaces (and why Splat Size ~2
    // "looked right"). With it, vQuad t in ±2 maps to r = t*sqrt(2)*sigma,
    // so exp(-t^2) in the fragment is exactly the true gaussian
    // exp(-r^2 / (2 sigma^2)) reaching exp(-4) at the quad edge.
    vec2 center = pos2d.xy / pos2d.w;
    gl_Position = vec4(
        center + (c.x * majorAxis + c.y * minorAxis) * 2.0 / vp,
        pos2d.z / pos2d.w,
        1.0);
}
"""

# normalized gaussian falloff (engine gsplat.js frag): rescaled so alpha
# reaches exactly zero at the kernel edge — soft ellipse rims instead of a
# visible cutoff at exp(-4), which reads as hard-edged blobs on wide lenses
_FALLOFF_GLSL = """
const float EXP4 = 0.0183156389;

float falloff(vec2 quad)
{
    float t2 = dot(quad, quad);
    return (exp(-t2) - EXP4) / (1.0 - EXP4);
}
"""

FRAG_SRC = _FALLOFF_GLSL + """
void main()
{
    float f = falloff(vQuad);
    float alpha = f * vColor.a;
    if (f <= 0.0 || alpha < 0.004) {
        discard;
    }
    FragColor = vec4(vColor.rgb, alpha);
}
"""

# depth-pick variant: opaque threshold + fragment depth in the red channel,
# used by the measure tool's Surface mode (front-most splat surface wins)
PICK_FRAG_SRC = _FALLOFF_GLSL + """
void main()
{
    float alpha = falloff(vQuad) * vColor.a;
    if (alpha < 0.2) {
        discard;
    }
    FragColor = vec4(gl_FragCoord.z, 0.0, 0.0, 1.0);
}
"""

_shader = None
_pick_shader = None


def _build_shader(frag_src, iface_name):
    info = gpu.types.GPUShaderCreateInfo()
    info.typedef_source(
        'struct SplatUniforms {'
        '  mat4 modelView;'
        '  mat4 projection;'
        '  vec4 params;'    # viewport w, viewport h, size multiplier, opacity multiplier
        '  vec4 params2;'   # aa flag, sh bands, sh tex width, srgb flag
        '  vec4 camPos;'    # camera position in object space (for SH)
        '  vec4 selColor;'  # selected-splat tint rgb; a>0.5 = state tex bound
        '  mat4 previewMatrix;'  # live drag preview transform (splat-local)
        '  vec4 misc;'      # x = preview active flag (>0.5); yzw spare
        '};')
    info.uniform_buf(0, 'SplatUniforms', 'u')
    info.sampler(0, 'FLOAT_2D', 'dataTex')
    info.sampler(1, 'FLOAT_2D', 'orderTex')
    info.sampler(2, 'FLOAT_2D', 'shTex')
    info.sampler(3, 'FLOAT_2D', 'stateTex')
    info.vertex_in(0, 'FLOAT', 'quadId')
    info.vertex_in(1, 'FLOAT', 'cornerId')
    iface = gpu.types.GPUStageInterfaceInfo(iface_name)
    iface.smooth('VEC4', 'vColor')
    iface.smooth('VEC2', 'vQuad')
    info.vertex_out(iface)
    info.fragment_out(0, 'VEC4', 'FragColor')
    info.vertex_source(VERT_SRC)
    info.fragment_source(frag_src)
    return gpu.shader.create_from_info(info)


def get_shader():
    global _shader
    if _shader is None:
        _shader = _build_shader(FRAG_SRC, 'pobim_splat_iface')
    return _shader


def get_pick_shader():
    global _pick_shader
    if _pick_shader is None:
        _pick_shader = _build_shader(PICK_FRAG_SRC, 'pobim_splat_pick_iface')
    return _pick_shader


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

        # keep the CPU texel array alive so a transform commit can rewrite the
        # edited splats' texels and re-upload (update_splats). Memory cost: the
        # same as the data texture itself (rows*W*4 f32) — documented.
        self._texels = texels
        self._rows = rows
        self.data_tex = gpu.types.GPUTexture(
            (DATA_TEX_WIDTH, rows), format='RGBA32F',
            data=_np_buffer('FLOAT', texels.ravel()))

        # SH texture: coefficients byte-packed 4-per-float / 16-per-texel in
        # an RGBA32F texture (the Python API only uploads FLOAT buffers, and
        # raw bit reinterpretation keeps memory at 1 byte per coefficient)
        self.sh_bands = 0
        self.sh_tex = None
        self.sh_width = 0
        if cloud.sh is not None and cloud.sh_bands > 0:
            n_bytes = cloud.sh.shape[1]
            tps = (n_bytes + 15) // 16
            spr = 4096 // tps
            self.sh_width = spr * tps
            sh_rows = (n + spr - 1) // spr
            if sh_rows <= MAX_TEX_HEIGHT:
                # chunked packing keeps the temporary byte buffers bounded
                # (~50MB) on multi-million-splat band-3 clouds
                sh_texels = np.zeros((sh_rows * self.sh_width, 4), np.float32)
                step = 1_000_000
                for i in range(0, n, step):
                    j = min(i + step, n)
                    padded = np.zeros((j - i, tps * 16), np.uint8)
                    padded[:, :n_bytes] = cloud.sh[i:j]
                    packed = padded.view(np.uint32).view(np.float32)
                    sh_base = np.arange(i, j, dtype=np.int64) * tps
                    for t in range(tps):
                        sh_texels[sh_base + t] = packed[:, t * 4:(t + 1) * 4]
                self.sh_tex = gpu.types.GPUTexture(
                    (self.sh_width, sh_rows), format='RGBA32F',
                    data=_np_buffer('FLOAT', sh_texels.ravel()))
                self.sh_bands = cloud.sh_bands
            else:
                print('[pobim_splats] SH texture too tall, skipping SH data')

        fmt = gpu.types.GPUVertFormat()
        fmt.attr_add(id='quadId', comp_type='F32', len=1, fetch_mode='FLOAT')
        fmt.attr_add(id='cornerId', comp_type='F32', len=1, fetch_mode='FLOAT')
        vbo = gpu.types.GPUVertBuf(fmt, n * 6)
        _attr_fill(vbo, 'quadId', np.repeat(np.arange(n, dtype=np.float32), 6))
        _attr_fill(vbo, 'cornerId', np.tile(np.array([0, 1, 2, 0, 2, 3], np.float32), n))
        self.vbo = vbo
        self.batch = gpu.types.GPUBatch(type='TRIS', buf=vbo)

        # bumped by update_splats; lets pickers/caches (e.g. the edit tool's
        # subsampled _pick_local) detect that positions changed and refresh
        self.geometry_version = 0

        self.order_rows = (n + ORDER_TEX_WIDTH - 1) // ORDER_TEX_WIDTH
        self.order_tex = None
        # per-splat edit-state texture (R32F, one texel/splat), uploaded
        # lazily from a SplatState and refreshed on version bumps
        self.state_tex = None
        self.state_version = -1
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

    def upload_state(self, flags):
        """Upload per-splat edit flags into an R32F texture (value =
        float(flags)), addressed the same way as the order texture."""
        padded = np.zeros(self.order_rows * ORDER_TEX_WIDTH, np.float32)
        padded[:self.count] = flags.astype(np.float32)
        self.state_tex = gpu.types.GPUTexture(
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

    def update_splats(self, indices, positions, cov6_rows):
        """Commit an edit: rewrite the data texture for ``indices`` in place.

        Rewrites the position texel (t0.xyz) and the two covariance texels
        (t1.xyz, t2.xyz) of the edited splats in the maintained CPU texel array
        and re-creates the data texture (one full re-upload per commit — the
        drag itself uses the GPU preview, not this). Also updates
        ``self.positions`` (used by sorting/picking) in place and invalidates
        the applied sort so the next frame re-orders.
        """
        idx = np.asarray(indices, np.int64).ravel()
        if idx.size == 0:
            return
        positions = np.ascontiguousarray(positions, np.float32)
        cov6_rows = np.ascontiguousarray(cov6_rows, np.float32)
        base = idx * TEXELS_PER_SPLAT
        self._texels[base, :3] = positions
        self._texels[base + 1, :3] = cov6_rows[:, :3]
        self._texels[base + 2, :3] = cov6_rows[:, 3:]
        self.data_tex = gpu.types.GPUTexture(
            (DATA_TEX_WIDTH, self._rows), format='RGBA32F',
            data=_np_buffer('FLOAT', self._texels.ravel()))
        self.positions[idx] = positions
        self._applied_dir = None   # geometry moved -> force a re-sort
        self.geometry_version += 1


def recompute_cov6(quats_wxyz, scales_log):
    """Covariance (N,6) from edited quats (w,x,y,z) and log-scales, reusing the
    loader's Sigma = R S S^T R^T builder. Used to feed SplatGPU.update_splats
    after a transform edit."""
    from .ply_loader import _quat_scale_to_cov6
    q = np.ascontiguousarray(quats_wxyz, np.float32)
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    scales = np.exp(np.asarray(scales_log, np.float32)).astype(np.float32)
    out = np.empty((q.shape[0], 6), np.float32)
    if q.shape[0]:
        _quat_scale_to_cov6(q, scales, out)
    return out


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
        self.state = None   # SplatState, created lazily on first edit
        self.edits = None   # SplatEdits geometry overrides (Phase 3 Track T)


# uid -> SplatEntry (session state; rebuilt from object properties on file open)
REGISTRY = {}

# uid -> (np.ndarray 4x4 mat, active bool): live transform preview set by the
# edit tool during a drag. The draw callback uploads it into previewMatrix and
# raises misc.x; the vertex shader moves only SELECTED splats, so the drag
# costs zero texture uploads. Absent uid -> identity/inactive.
PREVIEW = {}

_draw_handle = None


def set_preview(uid, mat):
    """Set the live preview transform (splat-local 4x4) for ``uid`` and redraw."""
    PREVIEW[uid] = (np.asarray(mat, np.float32).reshape(4, 4), True)
    redraw_viewports()


def clear_preview(uid):
    """Drop the live preview for ``uid`` (back to identity) and redraw."""
    if PREVIEW.pop(uid, None) is not None:
        redraw_viewports()


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


def _apply_persisted_edits(entry):
    """Re-apply restored geometry overrides to a freshly built SplatGPU.

    Runs once, right after the GPU build inside the draw handler (GPU context
    guaranteed). Ordering matters: ``edits.ensure`` snapshots the base geometry
    BEFORE ``update_splats`` mutates ``sg.positions`` in place — and at build
    time ``sg.positions`` ARE the freshly loaded ORIGINAL cloud positions, the
    exact base SplatEdits expects. A reload within the same session creates a
    fresh entry + fresh cloud, so this can never double-apply; re-running on
    the same edits would write identical override values anyway (idempotent).
    """
    edits = getattr(entry, 'edits', None)
    if edits is None:
        return
    try:
        has_pending = getattr(edits, '_pending', None) is not None
        if not has_pending and not edits.dirty.any():
            return
        cloud = entry.cloud
        sg = entry.gpu
        if getattr(cloud, 'quats', None) is None or \
                getattr(cloud, 'scales_log', None) is None:
            print('[pobim_splats] cannot re-apply geometry edits: '
                  'raw quats/scales missing (keep_geometry)')
            return
        edits.ensure(sg.positions, cloud.quats, cloud.scales_log)
        idx = np.nonzero(edits.dirty)[0]
        if idx.size:
            sg.update_splats(
                idx, edits.positions[idx],
                recompute_cov6(edits.quats[idx], edits.scales_log[idx]))
    except Exception as e:
        print(f'[pobim_splats] geometry edit re-apply failed: {e}')


def ensure_gpu(entry, obj_name=''):
    """Build GPU resources for an entry inside a GPU context. Returns SplatGPU or None."""
    if entry.gpu is None and entry.error is None:
        try:
            entry.gpu = SplatGPU(entry.cloud)
            # persisted transform edits (restored by load_entry_for_object)
            # are sparse/pending until now: bake them into the data texture
            _apply_persisted_edits(entry)
            # free the GPU-resident arrays, but KEEP cloud.quats / scales_log:
            # transform edits need the raw rotation+scale (cov6 is not
            # invertible back to quat+scale).
            entry.cloud.cov6 = entry.cloud.colors = None
            entry.cloud.opacities = entry.cloud.sh = None
        except Exception as e:
            entry.error = str(e)
            print(f'[pobim_splats] GPU build failed for {obj_name}: {e}')
    return entry.gpu


def _gather_draw_list(view):
    """Visible splat entries with their model-view matrices, farthest first."""
    objects = _splat_objects()
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
    draw_list.sort(key=lambda item: item[0])
    return draw_list


def render_depth_map(view, proj, width, height):
    """Render visible splats' front-surface depth (gl_FragCoord.z in the red
    channel) into an RGBA32F offscreen for the measure tool's Surface pick.
    Caller owns the returned GPUOffScreen (call .free()). Returns None when
    nothing is drawable."""
    draw_list = _gather_draw_list(view)
    if not draw_list:
        return None

    shader = get_pick_shader()
    offs = gpu.types.GPUOffScreen(max(width, 1), max(height, 1), format='RGBA32F')
    try:
        with offs.bind():
            fb = gpu.state.active_framebuffer_get()
            fb.clear(color=(1.0, 0.0, 0.0, 0.0), depth=1.0)
            gpu.state.blend_set('NONE')
            gpu.state.depth_test_set('LESS_EQUAL')
            gpu.state.depth_mask_set(True)
            try:
                for _, entry, obj, mv in draw_list:
                    sg = ensure_gpu(entry, obj.name)
                    if sg is None:
                        continue
                    params = np.array([
                        width, height,
                        getattr(obj, 'pobim_splat_scale', 1.0),
                        getattr(obj, 'pobim_splat_opacity', 1.0),
                        0.0, 0.0, 0.0, 0.0], np.float32)
                    near_cull = getattr(
                        bpy.context.scene, 'pobim_splats_near_cull', 0.1)
                    cam4 = np.array([0.0, 0.0, 0.0, near_cull], np.float32)

                    # honour hidden/deleted edits in the depth pick too
                    if (entry.state is not None and
                            entry.state.version != sg.state_version):
                        sg.upload_state(entry.state.flags)
                        sg.state_version = entry.state.version
                    has_state = entry.state is not None and sg.state_tex is not None
                    sel4 = np.array([SELECTED_COLOR[0], SELECTED_COLOR[1],
                                     SELECTED_COLOR[2], 1.0 if has_state else 0.0],
                                    np.float32)

                    # depth pick never previews: identity previewMatrix + zero
                    # misc keep the UBO the shader's expected 68 floats.
                    prev4 = np.eye(4, dtype=np.float32).ravel()
                    misc = np.zeros(4, np.float32)
                    ubo = gpu.types.GPUUniformBuf(_np_buffer(
                        'FLOAT',
                        np.concatenate([mv.T.ravel(), proj.T.ravel(),
                                        params, cam4, sel4, prev4, misc])))
                    shader.bind()
                    shader.uniform_block('u', ubo)
                    shader.uniform_sampler('dataTex', sg.data_tex)
                    shader.uniform_sampler('orderTex', sg.order_tex)
                    shader.uniform_sampler('shTex', sg.data_tex)
                    shader.uniform_sampler('stateTex', sg.state_tex if sg.state_tex else sg.data_tex)
                    sg.batch.draw(shader)
            finally:
                gpu.state.depth_mask_set(True)
                gpu.state.depth_test_set('NONE')
                gpu.state.blend_set('NONE')
    except Exception:
        offs.free()
        raise
    return offs


def read_depth_pixel(offs, x, y):
    """Read one depth value (0..1) from a render_depth_map offscreen.
    Returns None when the pixel is background."""
    if offs is None:
        return None
    x = int(min(max(x, 0), offs.width - 1))
    y = int(min(max(y, 0), offs.height - 1))
    with offs.bind():
        fb = gpu.state.active_framebuffer_get()
        raw = fb.read_color(x, y, 1, 1, 4, 0, 'FLOAT')
    z = float(raw.to_list()[0][0][0])
    return None if z >= 1.0 - 1e-7 else z


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

    view = np.array(rv3d.view_matrix, np.float32)
    proj = np.array(rv3d.window_matrix, np.float32)

    # farthest object first so inter-object blending is approximately correct
    draw_list = _gather_draw_list(view)
    if not draw_list:
        return

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
            sg = ensure_gpu(entry, obj.name)
            if sg is None:
                continue

            sg.sort_if_needed(mv, interval)

            # actual framebuffer viewport in device pixels — region.width is
            # in UI points and differs under display scaling, which would
            # skew the pixel-space dilation/AA terms in the shader
            try:
                vx, vy, vw, vh = gpu.state.viewport_get()
            except Exception:
                vw, vh = region.width, region.height
            if vw <= 0 or vh <= 0:
                vw, vh = region.width, region.height

            sh_bands = min(sg.sh_bands, getattr(obj, 'pobim_splat_sh_view', 3))
            cam_obj = np.linalg.inv(mv)[:3, 3]

            params = np.array([
                vw, vh,
                getattr(obj, 'pobim_splat_scale', 1.0),
                getattr(obj, 'pobim_splat_opacity', 1.0),
                1.0 if getattr(scene, 'pobim_splats_aa', False) else 0.0,
                sh_bands,
                sg.sh_width,
                1.0 if getattr(obj, 'pobim_splat_srgb', True) else 0.0], np.float32)
            near_cull = getattr(scene, 'pobim_splats_near_cull', 0.1)
            cam4 = np.array([cam_obj[0], cam_obj[1], cam_obj[2], near_cull],
                            np.float32)

            # refresh the edit-state texture when the SplatState changed;
            # selColor.a flags the shader to sample it (0 = untouched cloud)
            if entry.state is not None and entry.state.version != sg.state_version:
                sg.upload_state(entry.state.flags)
                sg.state_version = entry.state.version
            has_state = entry.state is not None and sg.state_tex is not None
            sel4 = np.array([SELECTED_COLOR[0], SELECTED_COLOR[1],
                             SELECTED_COLOR[2], 1.0 if has_state else 0.0],
                            np.float32)

            # live transform preview for this cloud (identity when inactive).
            # uploaded column-major (m.T) like modelView/projection.
            prev = PREVIEW.get(entry.uid)
            if prev is not None and prev[1]:
                prev4 = np.asarray(prev[0], np.float32).reshape(4, 4).T.ravel()
                misc = np.array([1.0, 0.0, 0.0, 0.0], np.float32)
            else:
                prev4 = np.eye(4, dtype=np.float32).ravel()
                misc = np.zeros(4, np.float32)

            ubo_data = np.concatenate([mv.T.ravel(), proj.T.ravel(),
                                       params, cam4, sel4, prev4, misc])
            ubo = gpu.types.GPUUniformBuf(_np_buffer('FLOAT', ubo_data))

            shader.bind()
            shader.uniform_block('u', ubo)
            shader.uniform_sampler('dataTex', sg.data_tex)
            shader.uniform_sampler('orderTex', sg.order_tex)
            shader.uniform_sampler('shTex', sg.sh_tex if sg.sh_tex else sg.data_tex)
            shader.uniform_sampler('stateTex', sg.state_tex if sg.state_tex else sg.data_tex)
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
        max_sh_bands=obj.pobim_splat_shmax)
    obj.pobim_splat_count = cloud.count
    obj.pobim_splat_sh_loaded = cloud.sh_bands
    entry = SplatEntry(obj.pobim_splat_uid, cloud)
    # restore serialized per-splat edit state; a count mismatch (Max Splats
    # changed, file re-pointed) or corrupt payload raises — drop the stale
    # property instead of decoding garbage flags into the export mask
    s = obj.get('pobim_splat_state')
    if s:
        try:
            from .splat_state import SplatState  # lazy: avoids import cycle
            entry.state = SplatState.deserialize(s, cloud.count)
        except Exception as e:
            print(f'[pobim_splats] discarding stale edit state for {obj.name}: {e}')
            try:
                del obj['pobim_splat_state']
            except Exception:
                pass
    # restore serialized geometry overrides (transform edits) the same way.
    # SplatEdits.deserialize keeps them sparse/pending: they are baked into
    # the GPU data texture on the first draw (ensure_gpu ->
    # _apply_persisted_edits) and read directly by the export payload —
    # without this restore, saved Move/Rotate/Scale edits are silently lost
    # on .blend reload in both the viewport and the export.
    s = obj.get('pobim_splat_edits')
    if s:
        try:
            from .splat_edits import SplatEdits  # lazy: avoids import cycle
            entry.edits = SplatEdits.deserialize(s, cloud.count)
        except Exception as e:
            print(f'[pobim_splats] discarding stale geometry edits for {obj.name}: {e}')
            try:
                del obj['pobim_splat_edits']
            except Exception:
                pass
    REGISTRY[obj.pobim_splat_uid] = entry
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

# 3D Gaussian Splatting PLY loader.
#
# Deliberately bpy-free so it can be unit-tested outside Blender.
# Parses the standard INRIA 3DGS binary PLY layout (x/y/z, opacity,
# scale_0..2, rot_0..3, f_dc_0..2) and precomputes per-splat data the
# viewport shader needs: world covariance (6 floats) and SH0 color.

import numpy as np

SH_C0 = 0.28209479177387814

# numpy dtype strings for PLY property types (little-endian binary)
_PLY_TYPES = {
    'float': '<f4', 'float32': '<f4',
    'double': '<f8', 'float64': '<f8',
    'char': 'i1', 'int8': 'i1',
    'uchar': 'u1', 'uint8': 'u1',
    'short': '<i2', 'int16': '<i2',
    'ushort': '<u2', 'uint16': '<u2',
    'int': '<i4', 'int32': '<i4',
    'uint': '<u4', 'uint32': '<u4',
}

_REQUIRED = (
    'x', 'y', 'z', 'opacity',
    'scale_0', 'scale_1', 'scale_2',
    'rot_0', 'rot_1', 'rot_2', 'rot_3',
    'f_dc_0', 'f_dc_1', 'f_dc_2',
)


class SplatCloud:
    """Parsed splat data, ready for GPU upload.

    Colors stay in SH color space (no sRGB conversion) — the shader adds the
    view-dependent SH contribution first and converts to linear afterwards.
    """

    def __init__(self, positions, cov6, colors, opacities, sh=None, sh_bands=0):
        self.positions = positions    # (N, 3) float32, object space
        self.cov6 = cov6              # (N, 6) float32: xx, xy, xz, yy, yz, zz
        self.colors = colors          # (N, 3) float32, 0..1
        self.opacities = opacities    # (N,)   float32, 0..1
        self.sh = sh                  # (N, 3C) uint8 channel-major, or None
        self.sh_bands = sh_bands      # 0..3

    @property
    def count(self):
        return self.positions.shape[0]


# coefficients per channel for SH bands 1..3
SH_COEFFS = {0: 0, 1: 3, 2: 8, 3: 15}
SH_BANDS_FROM_COEFFS = {0: 0, 3: 1, 8: 2, 15: 3}


def quantize_sh(coeffs):
    """float SH coefficients -> uint8 with the ±4 range used by compressed PLY."""
    out = np.empty(coeffs.shape, np.uint8)
    step = 1_000_000
    for i in range(0, coeffs.shape[0], step):
        j = min(i + step, coeffs.shape[0])
        out[i:j] = np.clip(
            np.round((coeffs[i:j] / 8.0 + 0.5) * 255.0), 0, 255).astype(np.uint8)
    return out


def truncate_sh(coeffs, src_c, dst_c):
    """Reduce channel-major (N, 3*src_c) coefficients to dst_c per channel."""
    if dst_c >= src_c:
        return coeffs
    idx = [ch * src_c + k for ch in range(3) for k in range(dst_c)]
    return np.ascontiguousarray(coeffs[:, idx])


def _parse_header(filepath):
    """Parse a binary PLY header.

    Returns (elements, data_offset) where elements is an ordered list of
    (name, count, numpy_dtype) — one entry per PLY element.
    """
    with open(filepath, 'rb') as f:
        head = f.read(65536)

    end_tag = b'end_header\n'
    end = head.find(end_tag)
    if end < 0:
        raise ValueError('ไม่พบ PLY header (ไฟล์อาจไม่ใช่ .ply หรือ header ใหญ่ผิดปกติ)')

    lines = head[:end].decode('ascii', errors='replace').splitlines()
    lines = [line.strip() for line in lines if line.strip()]
    if not lines or lines[0] != 'ply':
        raise ValueError('ไฟล์นี้ไม่ใช่ PLY')

    fmt = next((line for line in lines if line.startswith('format ')), '')
    if 'binary_little_endian' not in fmt:
        raise ValueError(
            'รองรับเฉพาะ PLY แบบ binary — ถ้าเป็น .splat / .spz '
            'ให้แปลงก่อนด้วย: npx @playcanvas/splat-transform input output.ply')

    elements = []          # (name, count, [(prop, dtype), ...])
    for line in lines:
        if line.startswith('element '):
            parts = line.split()
            elements.append((parts[1], int(parts[2]), []))
        elif line.startswith('property '):
            if not elements:
                continue
            parts = line.split()
            if parts[1] == 'list':
                raise ValueError('PLY มี list property — ไม่ใช่ไฟล์ 3DGS')
            elements[-1][2].append((parts[2], _PLY_TYPES[parts[1]]))

    result = [(name, count, np.dtype(props)) for name, count, props in elements]
    return result, end + len(end_tag)


def _read_elements(filepath, elements, data_offset):
    """Read every PLY element into a dict name -> structured array."""
    out = {}
    offset = data_offset
    for name, count, dtype in elements:
        data = np.fromfile(filepath, dtype=dtype, count=count, offset=offset)
        if data.shape[0] < count:
            raise ValueError(f'ไฟล์สั้นกว่าที่ header ระบุ (element {name})')
        out[name] = data
        offset += count * dtype.itemsize
    return out


def _quat_scale_to_cov6(quat, scales, out):
    """Sigma = R S S^T R^T for a chunk; writes 6 unique elements into out."""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    n = quat.shape[0]
    rot = np.empty((n, 3, 3), np.float32)
    rot[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rot[:, 0, 1] = 2 * (x * y - w * z)
    rot[:, 0, 2] = 2 * (x * z + w * y)
    rot[:, 1, 0] = 2 * (x * y + w * z)
    rot[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rot[:, 1, 2] = 2 * (y * z - w * x)
    rot[:, 2, 0] = 2 * (x * z - w * y)
    rot[:, 2, 1] = 2 * (y * z + w * x)
    rot[:, 2, 2] = 1 - 2 * (x * x + y * y)

    m = rot * scales[:, None, :]          # R @ diag(s)
    sigma = m @ m.transpose(0, 2, 1)

    out[:, 0] = sigma[:, 0, 0]
    out[:, 1] = sigma[:, 0, 1]
    out[:, 2] = sigma[:, 0, 2]
    out[:, 3] = sigma[:, 1, 1]
    out[:, 4] = sigma[:, 1, 2]
    out[:, 5] = sigma[:, 2, 2]


def build_cloud(positions, scales, quat, colors, opacities,
                max_splats=0, sh=None):
    """Assemble a SplatCloud from raw arrays (shared by all format decoders).

    positions (N,3); scales (N,3) LINEAR (already exp'd); quat (N,4) w,x,y,z;
    colors (N,3) 0..1 in SH color space; opacities (N,) 0..1;
    sh: channel-major SH coefficients (N, 3C) as float32 or pre-quantized
    uint8, or None.
    """
    n = positions.shape[0]
    if max_splats and n > max_splats:
        sel = np.random.default_rng(12345).permutation(n)[:max_splats]
        positions, scales, quat = positions[sel], scales[sel], quat[sel]
        colors, opacities = colors[sel], opacities[sel]
        if sh is not None:
            sh = sh[sel]
        n = max_splats

    quat = quat / (np.linalg.norm(quat, axis=1, keepdims=True) + 1e-12)

    # chunked so the (N,3,3) temporaries stay bounded on multi-million scenes
    cov6 = np.empty((n, 6), np.float32)
    step = 1_000_000
    for i in range(0, n, step):
        j = min(i + step, n)
        _quat_scale_to_cov6(quat[i:j], scales[i:j], cov6[i:j])

    colors = np.clip(colors, 0.0, 1.0).astype(np.float32)

    sh_bands = 0
    if sh is not None:
        c = sh.shape[1] // 3
        sh_bands = SH_BANDS_FROM_COEFFS.get(c, 0)
        if sh_bands == 0:
            sh = None
        elif sh.dtype != np.uint8:
            sh = quantize_sh(np.asarray(sh, np.float32))

    return SplatCloud(
        np.ascontiguousarray(positions, dtype=np.float32), cov6,
        colors, np.ascontiguousarray(opacities, dtype=np.float32),
        sh=sh, sh_bands=sh_bands)


def _unpack_111011(packed):
    """uint32 -> three unorm floats (11, 10, 11 bits, high to low)."""
    x = ((packed >> 21) & 0x7FF).astype(np.float32) / 2047.0
    y = ((packed >> 11) & 0x3FF).astype(np.float32) / 1023.0
    z = (packed & 0x7FF).astype(np.float32) / 2047.0
    return x, y, z


CHUNK = 256  # splats per chunk in the compressed PLY format


def _decode_compressed(parts, max_splats, max_sh_bands):
    """Decode SuperSplat .compressed.ply (chunk min/max + packed uint32 vertex).

    Mirrors splat-transform's decompress-ply.ts, vectorized with numpy.
    """
    chunks = parts['chunk']
    verts = parts['vertex']
    n = verts.shape[0]
    if n <= 0:
        raise ValueError('compressed PLY ไม่มี splat')
    if (n + CHUNK - 1) // CHUNK != chunks.shape[0]:
        raise ValueError('compressed PLY: จำนวน chunk ไม่ตรงกับจำนวน splat')

    ci = np.arange(n, dtype=np.int64) // CHUNK

    def lerp_chunk(lo_name, hi_name, t):
        lo = chunks[lo_name][ci].astype(np.float32)
        hi = chunks[hi_name][ci].astype(np.float32)
        return lo * (1.0 - t) + hi * t

    px, py, pz = _unpack_111011(verts['packed_position'])
    positions = np.stack([
        lerp_chunk('min_x', 'max_x', px),
        lerp_chunk('min_y', 'max_y', py),
        lerp_chunk('min_z', 'max_z', pz)], axis=1)

    sx, sy, sz = _unpack_111011(verts['packed_scale'])
    scales = np.exp(np.stack([
        lerp_chunk('min_scale_x', 'max_scale_x', sx),
        lerp_chunk('min_scale_y', 'max_scale_y', sy),
        lerp_chunk('min_scale_z', 'max_scale_z', sz)], axis=1))

    # smallest-three rotation: top 2 bits tag which component was largest
    pr = verts['packed_rotation']
    norm = np.float32(np.sqrt(2.0))
    a = (((pr >> 20) & 0x3FF).astype(np.float32) / 1023.0 - 0.5) * norm
    b = (((pr >> 10) & 0x3FF).astype(np.float32) / 1023.0 - 0.5) * norm
    c = ((pr & 0x3FF).astype(np.float32) / 1023.0 - 0.5) * norm
    m = np.sqrt(np.maximum(0.0, 1.0 - (a * a + b * b + c * c))).astype(np.float32)
    which = (pr >> 30).astype(np.uint8)
    quat = np.empty((n, 4), np.float32)             # (w, x, y, z) = rot_0..3
    for w_idx, cols in enumerate(([1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2])):
        mask = which == w_idx
        quat[mask, w_idx] = m[mask]
        for src, dst in zip((a, b, c), cols):
            quat[mask, dst] = src[mask]

    pc = verts['packed_color']
    colors = np.stack([
        ((pc >> 24) & 0xFF).astype(np.float32) / 255.0,
        ((pc >> 16) & 0xFF).astype(np.float32) / 255.0,
        ((pc >> 8) & 0xFF).astype(np.float32) / 255.0], axis=1)
    if 'min_r' in (chunks.dtype.names or ()):
        for k, (lo, hi) in enumerate((('min_r', 'max_r'), ('min_g', 'max_g'),
                                      ('min_b', 'max_b'))):
            colors[:, k] = lerp_chunk(lo, hi, colors[:, k])
    opacities = (pc & 0xFF).astype(np.float32) / 255.0

    # optional 'sh' element: uint8 f_rest_i columns, channel-major
    sh = None
    sh_elem = parts.get('sh')
    if sh_elem is not None and max_sh_bands > 0:
        names = [f for f in (sh_elem.dtype.names or ()) if f.startswith('f_rest_')]
        c = len(names) // 3
        if c in SH_BANDS_FROM_COEFFS and c > 0:
            raw = np.stack(
                [sh_elem[f'f_rest_{i}'] for i in range(3 * c)], axis=1)
            # decode: 0 -> 0, 255 -> 1, else (v+0.5)/256; coef = (n-0.5)*8
            norm = np.where(raw == 0, 0.0,
                            np.where(raw == 255, 1.0,
                                     (raw.astype(np.float32) + 0.5) / 256.0))
            coeffs = ((norm - 0.5) * 8.0).astype(np.float32)
            dst_c = min(c, SH_COEFFS[max_sh_bands])
            sh = truncate_sh(coeffs, c, dst_c)

    return build_cloud(positions, scales, quat, colors, opacities,
                       max_splats, sh=sh)


def load_gaussian_ply(filepath, max_splats=0, max_sh_bands=3):
    """Load a 3DGS PLY file (standard or SuperSplat-compressed) into a SplatCloud.

    max_splats: 0 = load all; otherwise randomly subsample down to this count.
    max_sh_bands: highest spherical-harmonics band to keep (0 = SH0 only).
    """
    elements, offset = _parse_header(filepath)
    names = [name for name, _, _ in elements]

    if 'chunk' in names and 'vertex' in names:
        parts = _read_elements(filepath, elements, offset)
        return _decode_compressed(parts, max_splats, max_sh_bands)

    if not elements or elements[0][0] != 'vertex' or elements[0][1] <= 0:
        raise ValueError('PLY ไม่มี vertex element นำหน้า — ไม่ใช่ไฟล์ 3DGS ที่รองรับ')
    _, count, dtype = elements[0]

    missing = [r for r in _REQUIRED if r not in (dtype.names or ())]
    if missing:
        raise ValueError(f'ไม่ใช่ไฟล์ 3DGS PLY (ขาด property: {", ".join(missing)})')

    data = np.fromfile(filepath, dtype=dtype, count=count, offset=offset)
    if data.shape[0] < count:
        raise ValueError(f'ไฟล์สั้นกว่าที่ header ระบุ ({data.shape[0]}/{count} splats)')

    positions = np.stack([data['x'], data['y'], data['z']], axis=1).astype(np.float32)
    opacities = (1.0 / (1.0 + np.exp(-data['opacity'].astype(np.float32)))).astype(np.float32)
    scales = np.exp(
        np.stack([data['scale_0'], data['scale_1'], data['scale_2']], axis=1).astype(np.float32))
    quat = np.stack([data['rot_0'], data['rot_1'], data['rot_2'], data['rot_3']],
                    axis=1).astype(np.float32)
    colors = 0.5 + SH_C0 * np.stack(
        [data['f_dc_0'], data['f_dc_1'], data['f_dc_2']], axis=1).astype(np.float32)

    # higher SH bands: f_rest_0..f_rest_{3C-1}, channel-major
    sh = None
    if max_sh_bands > 0:
        rest = [f for f in (dtype.names or ()) if f.startswith('f_rest_')]
        c = len(rest) // 3
        if c in SH_BANDS_FROM_COEFFS and c > 0:
            coeffs = np.stack(
                [data[f'f_rest_{i}'] for i in range(3 * c)], axis=1).astype(np.float32)
            dst_c = min(c, SH_COEFFS[max_sh_bands])
            sh = truncate_sh(coeffs, c, dst_c)

    return build_cloud(positions, scales, quat, colors, opacities,
                       max_splats, sh=sh)

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
    """Parsed splat data, ready for GPU upload."""

    def __init__(self, positions, cov6, colors, opacities):
        self.positions = positions    # (N, 3) float32, object space
        self.cov6 = cov6              # (N, 6) float32: xx, xy, xz, yy, yz, zz
        self.colors = colors          # (N, 3) float32, 0..1
        self.opacities = opacities    # (N,)   float32, 0..1

    @property
    def count(self):
        return self.positions.shape[0]


def _parse_header(filepath):
    """Return (vertex_dtype, vertex_count, data_offset)."""
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
            'รองรับเฉพาะ PLY แบบ binary_little_endian — ถ้าเป็น .compressed.ply / .sog / .spz '
            'ให้แปลงก่อนด้วย: npx @playcanvas/splat-transform input output.ply')

    count = 0
    props = []
    in_vertex = False
    for line in lines:
        if line.startswith('element '):
            parts = line.split()
            in_vertex = parts[1] == 'vertex'
            if in_vertex:
                count = int(parts[2])
        elif line.startswith('property ') and in_vertex:
            parts = line.split()
            if parts[1] == 'list':
                raise ValueError('PLY มี list property ใน vertex element — ไม่ใช่ไฟล์ 3DGS')
            props.append((parts[2], _PLY_TYPES[parts[1]]))

    names = [name for name, _ in props]
    missing = [r for r in _REQUIRED if r not in names]
    if missing:
        raise ValueError(f'ไม่ใช่ไฟล์ 3DGS PLY (ขาด property: {", ".join(missing)})')

    return np.dtype(props), count, end + len(end_tag)


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


def _srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)


def load_gaussian_ply(filepath, max_splats=0, srgb_to_linear=True):
    """Load a 3DGS PLY file into a SplatCloud.

    max_splats: 0 = load all; otherwise randomly subsample down to this count.
    srgb_to_linear: convert SH0 colors to linear so Blender's Standard view
        transform reproduces what web splat viewers show.
    """
    dtype, count, offset = _parse_header(filepath)
    if count <= 0:
        raise ValueError('PLY ไม่มี vertex data')

    data = np.fromfile(filepath, dtype=dtype, count=count, offset=offset)
    if data.shape[0] < count:
        raise ValueError(f'ไฟล์สั้นกว่าที่ header ระบุ ({data.shape[0]}/{count} splats)')

    if max_splats and count > max_splats:
        sel = np.random.default_rng(12345).permutation(count)[:max_splats]
        data = data[sel]

    n = data.shape[0]
    positions = np.stack([data['x'], data['y'], data['z']], axis=1).astype(np.float32)

    opacities = data['opacity'].astype(np.float32)
    opacities = (1.0 / (1.0 + np.exp(-opacities))).astype(np.float32)

    scales = np.exp(
        np.stack([data['scale_0'], data['scale_1'], data['scale_2']], axis=1).astype(np.float32))

    quat = np.stack([data['rot_0'], data['rot_1'], data['rot_2'], data['rot_3']],
                    axis=1).astype(np.float32)
    quat /= np.linalg.norm(quat, axis=1, keepdims=True) + 1e-12

    # chunked so the (N,3,3) temporaries stay bounded on multi-million scenes
    cov6 = np.empty((n, 6), np.float32)
    chunk = 1_000_000
    for i in range(0, n, chunk):
        j = min(i + chunk, n)
        _quat_scale_to_cov6(quat[i:j], scales[i:j], cov6[i:j])

    colors = 0.5 + SH_C0 * np.stack(
        [data['f_dc_0'], data['f_dc_1'], data['f_dc_2']], axis=1).astype(np.float32)
    np.clip(colors, 0.0, 1.0, out=colors)
    if srgb_to_linear:
        colors = _srgb_to_linear(colors)

    return SplatCloud(positions, cov6, colors, opacities)

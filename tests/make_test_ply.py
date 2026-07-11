# Generate a small synthetic 3DGS PLY for testing the addon without a real scan.
# Creates a colorful torus of gaussians. Usage:
#   python3 tests/make_test_ply.py [out.ply] [count]

import struct
import sys

import numpy as np


def make_torus_splats(n, major=2.0, minor=0.6, seed=7):
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 2 * np.pi, n)
    v = rng.uniform(0, 2 * np.pi, n)
    r = minor * np.sqrt(rng.uniform(0, 1, n))

    x = (major + r * np.cos(v)) * np.cos(u)
    y = r * np.sin(v)                      # y-down convention like real scans
    z = (major + r * np.cos(v)) * np.sin(u)
    pos = np.stack([x, y, z], 1).astype(np.float32)

    # log-scales around 2cm, slightly anisotropic
    scales = np.log(rng.uniform(0.01, 0.04, (n, 3))).astype(np.float32)

    quat = rng.normal(size=(n, 4))
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    quat = quat.astype(np.float32)

    # rainbow around the ring, stored as SH0 coefficients
    rgb = np.stack([
        0.5 + 0.5 * np.cos(u),
        0.5 + 0.5 * np.cos(u + 2.094),
        0.5 + 0.5 * np.cos(u + 4.188)], 1).astype(np.float32)
    sh0 = ((rgb - 0.5) / 0.28209479177387814).astype(np.float32)

    # opacity logits: mostly opaque
    opacity = np.full(n, 3.0, np.float32)
    return pos, scales, quat, sh0, opacity


def write_gaussian_ply(path, pos, scales, quat, sh0, opacity):
    n = pos.shape[0]
    fields = (
        ['x', 'y', 'z', 'nx', 'ny', 'nz'] +
        [f'f_dc_{i}' for i in range(3)] +
        ['opacity'] +
        [f'scale_{i}' for i in range(3)] +
        [f'rot_{i}' for i in range(4)]
    )
    header = 'ply\nformat binary_little_endian 1.0\n'
    header += f'element vertex {n}\n'
    header += ''.join(f'property float {f}\n' for f in fields)
    header += 'end_header\n'

    data = np.zeros((n, len(fields)), np.float32)
    data[:, 0:3] = pos
    data[:, 6:9] = sh0
    data[:, 9] = opacity
    data[:, 10:13] = scales
    data[:, 13:17] = quat

    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(data.tobytes())
    print(f'wrote {path}: {n:,} splats')


def write_compressed_gaussian_ply(path, pos, scales, quat, sh0, opacity):
    """Encode splats into SuperSplat .compressed.ply (inverse of the decoder).

    quat is (w, x, y, z) = rot_0..3; scales/opacity in log/logit space as in
    the standard PLY.
    """
    n = pos.shape[0]
    num_chunks = (n + 255) // 256
    pad = num_chunks * 256 - n

    def padded(a):
        return np.concatenate([a, np.repeat(a[-1:], pad, axis=0)]) if pad else a

    p = padded(pos).reshape(num_chunks, 256, 3).astype(np.float32)
    s = padded(scales).reshape(num_chunks, 256, 3).astype(np.float32)
    q = padded(quat).reshape(-1, 4).astype(np.float32)
    color = np.clip(0.5 + 0.28209479177387814 * padded(sh0), 0, 1)
    alpha = 1.0 / (1.0 + np.exp(-padded(opacity)))

    pmin, pmax = p.min(1), p.max(1)
    smin, smax = s.min(1), s.max(1)

    def quantize(v, lo, hi, bits):
        rng = np.where(hi - lo == 0, 1, hi - lo)
        t = np.clip((v - lo[:, None, :]) / rng[:, None, :], 0, 1)
        return np.round(t * ((1 << bits) - 1)).astype(np.uint32)

    def pack_111011(v, lo, hi):
        x = quantize(v[..., 0:1], lo[:, 0:1], hi[:, 0:1], 11)[..., 0]
        y = quantize(v[..., 1:2], lo[:, 1:2], hi[:, 1:2], 10)[..., 0]
        z = quantize(v[..., 2:3], lo[:, 2:3], hi[:, 2:3], 11)[..., 0]
        return ((x << 21) | (y << 11) | z).reshape(-1)

    packed_position = pack_111011(p, pmin, pmax)
    packed_scale = pack_111011(s, smin, smax)

    # smallest-three: negate so the largest-|.| component is positive
    which = np.argmax(np.abs(q), axis=1)
    sign = np.sign(q[np.arange(q.shape[0]), which])
    q = q * np.where(sign == 0, 1, sign)[:, None]
    others = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])[which]
    rest = np.take_along_axis(q, others, axis=1)
    bits = np.clip(np.round((rest / np.sqrt(2.0) + 0.5) * 1023), 0, 1023).astype(np.uint32)
    packed_rotation = ((which.astype(np.uint32) << 30) |
                       (bits[:, 0] << 20) | (bits[:, 1] << 10) | bits[:, 2])

    c8 = np.round(color * 255).astype(np.uint32)
    a8 = np.round(alpha * 255).astype(np.uint32)
    packed_color = (c8[:, 0] << 24) | (c8[:, 1] << 16) | (c8[:, 2] << 8) | a8

    chunk_fields = ['min_x', 'min_y', 'min_z', 'max_x', 'max_y', 'max_z',
                    'min_scale_x', 'min_scale_y', 'min_scale_z',
                    'max_scale_x', 'max_scale_y', 'max_scale_z']
    header = 'ply\nformat binary_little_endian 1.0\n'
    header += f'element chunk {num_chunks}\n'
    header += ''.join(f'property float {f}\n' for f in chunk_fields)
    header += f'element vertex {n}\n'
    header += ''.join(f'property uint packed_{f}\n'
                      for f in ('position', 'rotation', 'scale', 'color'))
    header += 'end_header\n'

    chunk_data = np.concatenate([pmin, pmax, smin, smax], axis=1).astype('<f4')
    vert_data = np.stack([packed_position[:n], packed_rotation[:n],
                          packed_scale[:n], packed_color[:n]], axis=1).astype('<u4')

    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(chunk_data.tobytes())
        f.write(vert_data.tobytes())
    print(f'wrote {path}: {n:,} splats (compressed)')


if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else 'test_torus.ply'
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 100_000
    if out.endswith('.compressed.ply'):
        write_compressed_gaussian_ply(out, *make_torus_splats(count))
    else:
        write_gaussian_ply(out, *make_torus_splats(count))

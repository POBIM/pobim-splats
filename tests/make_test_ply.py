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


if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else 'test_torus.ply'
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 100_000
    write_gaussian_ply(out, *make_torus_splats(count))

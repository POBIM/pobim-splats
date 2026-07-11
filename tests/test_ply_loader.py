# bpy-free tests for ply_loader. Run: python3 tests/test_ply_loader.py

import importlib.util
import os
import sys
import tempfile

import numpy as np

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# load ply_loader directly by path: the package __init__ imports bpy,
# which does not exist outside Blender
_spec = importlib.util.spec_from_file_location(
    'ply_loader', os.path.join(_root, 'pobim_splats', 'ply_loader.py'))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
load_gaussian_ply = _mod.load_gaussian_ply

from make_test_ply import make_torus_splats, write_gaussian_ply


def main():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 't.ply')
        pos, scales, quat, sh0, opacity = make_torus_splats(50_000)
        write_gaussian_ply(path, pos, scales, quat, sh0, opacity)

        cloud = load_gaussian_ply(path, srgb_to_linear=False)
        assert cloud.count == 50_000, cloud.count
        assert np.allclose(cloud.positions, pos, atol=1e-6)

        # opacity = sigmoid(3.0)
        assert np.allclose(cloud.opacities, 1 / (1 + np.exp(-3.0)), atol=1e-5)

        # colors reconstruct the original rainbow rgb
        rgb = 0.5 + 0.28209479177387814 * sh0
        assert np.allclose(cloud.colors, np.clip(rgb, 0, 1), atol=1e-5)

        # covariance: symmetric PSD with det ~ prod(exp(scale)^2)
        s2 = np.exp(scales.astype(np.float64)) ** 2
        c = cloud.cov6.astype(np.float64)
        sigma = np.empty((cloud.count, 3, 3))
        sigma[:, 0, 0], sigma[:, 0, 1], sigma[:, 0, 2] = c[:, 0], c[:, 1], c[:, 2]
        sigma[:, 1, 0], sigma[:, 1, 1], sigma[:, 1, 2] = c[:, 1], c[:, 3], c[:, 4]
        sigma[:, 2, 0], sigma[:, 2, 1], sigma[:, 2, 2] = c[:, 2], c[:, 4], c[:, 5]
        det = np.linalg.det(sigma)
        expected = s2.prod(axis=1)
        assert np.allclose(det, expected, rtol=1e-2), 'covariance determinant mismatch'
        eig = np.linalg.eigvalsh(sigma)
        assert (eig > -1e-9).all(), 'covariance not PSD'

        # subsampling
        small = load_gaussian_ply(path, max_splats=10_000)
        assert small.count == 10_000

        # srgb conversion darkens midtones
        lin = load_gaussian_ply(path, srgb_to_linear=True)
        assert (lin.colors <= cloud.colors + 1e-6).all()

        # rejects non-3DGS ply
        bad = os.path.join(tmp, 'bad.ply')
        with open(bad, 'wb') as f:
            f.write(b'ply\nformat binary_little_endian 1.0\n'
                    b'element vertex 1\nproperty float x\nproperty float y\n'
                    b'property float z\nend_header\n' + b'\x00' * 12)
        try:
            load_gaussian_ply(bad)
            raise AssertionError('should have rejected non-3DGS ply')
        except ValueError:
            pass

    print('all ply_loader tests passed')


if __name__ == '__main__':
    main()

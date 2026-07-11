# SOG v2 decode math — bpy-free (works on raw RGBA byte arrays).
#
# Mirrors splat-transform's read-sog.ts. A SOG scene is meta.json plus WebP
# textures; gaussian g lives at texel (g % width, g // width) in top-down
# raster order:
#   means_l/means_u : 16-bit fixed-point positions, log-transformed, lerped
#                     between meta mins/maxs
#   quats           : smallest-three bytes + tag byte 252..255 (largest comp)
#   scales          : rgb bytes index a 256-entry log-space codebook
#   sh0             : rgb bytes index an SH0 codebook; alpha byte = opacity

import numpy as np

from .ply_loader import (
    SH_C0, SH_COEFFS, build_cloud, opacity_to_logit)

# for largest component m (0..3 = w,x,y,z), the three packed values fill
# the remaining quaternion slots in ascending index order
_QUAT_IDX = np.array([
    [1, 2, 3],
    [0, 2, 3],
    [0, 1, 3],
    [0, 1, 2]], dtype=np.int64)


def _inv_log_transform(v):
    return np.sign(v) * (np.exp(np.abs(v)) - 1.0)


def _texel_data(rgba, count, name):
    """Flatten an (H, W, 4) uint8 texture to (count, 4) texels."""
    flat = rgba.reshape(-1, 4)
    if flat.shape[0] < count:
        raise ValueError(f'SOG: texture {name} เล็กเกินไป ({flat.shape[0]} < {count})')
    return flat[:count]


def _decode_sog_arrays(meta, textures, max_sh_bands):
    """Decode SOG v2 into full-count canonical arrays (no subsampling).

    Returns (positions, scales, quat, colors, opacities, sh) where scales are
    LINEAR (already exp'd), colors are 0..1 in SH color space, opacities are
    0..1, and sh is float32 (N, 3C) channel-major or None.
    """
    version = meta.get('version')
    if version != 2:
        raise ValueError(f'รองรับเฉพาะ SOG v2 (ไฟล์นี้ version={version}) — '
                         'แปลงด้วย: npx @playcanvas/splat-transform input output.sog')

    for key in ('count', 'means', 'scales', 'quats', 'sh0'):
        if key not in meta:
            raise ValueError(f'SOG: meta.json ขาด key "{key}"')
    for key in ('mins', 'maxs', 'files'):
        if key not in meta['means']:
            raise ValueError(f'SOG: meta.json ขาด means.{key}')
    for section in ('scales', 'sh0'):
        if 'codebook' not in meta[section]:
            raise ValueError(f'SOG: meta.json ขาด {section}.codebook')

    count = int(meta['count'])
    if count <= 0:
        raise ValueError('SOG: count เป็นศูนย์')

    def tex(section, index=0):
        files = meta[section].get('files', [])
        if len(files) <= index:
            raise ValueError(f'SOG: meta.json ขาด {section}.files[{index}]')
        name = files[index]
        if name not in textures:
            raise ValueError(f'SOG: ไม่พบ texture {name}')
        return _texel_data(textures[name], count, name)

    # positions: 16-bit lerp of log-transformed bounds
    lo = tex('means', 0).astype(np.uint16)
    hi = tex('means', 1).astype(np.uint16)
    mins = np.asarray(meta['means']['mins'], np.float32)
    maxs = np.asarray(meta['means']['maxs'], np.float32)
    scale = np.where(maxs - mins == 0.0, 1.0, maxs - mins).astype(np.float32)
    fixed = (lo[:, :3] | (hi[:, :3] << 8)).astype(np.float32) / 65535.0
    positions = _inv_log_transform(mins + scale * fixed).astype(np.float32)

    # quats: smallest-three with largest-component tag
    q = tex('quats')
    n = count
    tag = q[:, 3].astype(np.int64)
    maxc = np.clip(tag - 252, 0, 3)
    abc = ((q[:, :3].astype(np.float32) / 255.0) * 2.0 - 1.0) / np.float32(np.sqrt(2.0))
    quat = np.zeros((n, 4), np.float32)             # (w, x, y, z)
    rows = np.arange(n)
    for k in range(3):
        quat[rows, _QUAT_IDX[maxc, k]] = abc[:, k]
    quat[rows, maxc] = np.sqrt(
        np.maximum(0.0, 1.0 - (abc * abc).sum(axis=1))).astype(np.float32)
    invalid = (tag < 252) | (tag > 255)
    if invalid.any():
        quat[invalid] = (1.0, 0.0, 0.0, 0.0)

    # scales: codebook holds log-space values
    s_code = np.asarray(meta['scales']['codebook'], np.float32)
    s = tex('scales')
    scales = np.exp(s_code[s[:, :3]])

    # sh0: codebook holds SH coefficients; alpha channel is plain opacity
    c_code = np.asarray(meta['sh0']['codebook'], np.float32)
    c = tex('sh0')
    colors = 0.5 + SH_C0 * c_code[c[:, :3]]
    opacities = c[:, 3].astype(np.float32) / 255.0

    # shN (optional): 16-bit palette label -> centroids texture -> codebook
    sh = None
    shn = meta.get('shN')
    if shn and max_sh_bands > 0:
        bands = int(shn.get('bands', 0))
        coeffs_n = SH_COEFFS.get(min(bands, max_sh_bands), 0)
        src_c = SH_COEFFS.get(bands, 0)
        if coeffs_n > 0 and src_c > 0:
            sh_code = np.asarray(shn['codebook'], np.float32)
            labels = tex('shN', 1)
            label = labels[:, 0].astype(np.int64) | (labels[:, 1].astype(np.int64) << 8)
            cent_name = shn['files'][0]
            if cent_name not in textures:
                raise ValueError(f'SOG: ไม่พบ texture {cent_name}')
            cent = textures[cent_name]          # (H, 64*src_c, 4)
            ch, cw = cent.shape[0], cent.shape[1]
            palette_count = int(shn.get('count', ch * 64))
            valid = label < min(palette_count, ch * 64)
            label_safe = np.where(valid, label, 0)
            cy = label_safe // 64
            cx = (label_safe % 64) * src_c      # column base, one col per coeff
            # gather (N, coeffs_n, 3) centroid bytes -> codebook floats
            cols = cx[:, None] + np.arange(coeffs_n)[None, :]
            bytes_rgb = cent[cy[:, None], cols, :3]
            coeffs = sh_code[bytes_rgb].astype(np.float32)
            coeffs[~valid] = 0.0
            # channel-major (N, 3*coeffs_n)
            sh = np.concatenate(
                [coeffs[:, :, 0], coeffs[:, :, 1], coeffs[:, :, 2]], axis=1)

    return positions, scales, quat, colors, opacities, sh


def decode_sog(meta, textures, max_splats=0, max_sh_bands=3):
    """Decode SOG v2 into a SplatCloud.

    meta: parsed meta.json dict.
    textures: dict filename -> (H, W, 4) uint8 array in TOP-DOWN raster order.
    max_sh_bands: highest SH band to decode from the optional shN section.
    """
    positions, scales, quat, colors, opacities, sh = _decode_sog_arrays(
        meta, textures, max_sh_bands)
    return build_cloud(positions, scales, quat, colors, opacities,
                       max_splats, sh=sh)


def decode_sog_canonical(meta, textures, max_sh_bands=3):
    """Decode SOG v2 into the canonical dict used for re-export.

    Same shape as ply_loader.load_raw_ply's 'canonical' result:
    scales stored as log, colors as f_dc SH0 coefficients, opacity as logit.
    """
    positions, scales, quat, colors, opacities, sh = _decode_sog_arrays(
        meta, textures, max_sh_bands)
    return {
        'kind': 'canonical',
        'positions': positions,
        'scales_log': np.log(np.maximum(scales, 1e-30)).astype(np.float32),
        'quat_wxyz': quat,
        'f_dc': ((colors - 0.5) / SH_C0).astype(np.float32),
        'opacity_logit': opacity_to_logit(opacities),
        'sh': sh,
        'count': int(positions.shape[0]),
    }

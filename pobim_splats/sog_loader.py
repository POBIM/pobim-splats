# SOG file loading for Blender: unzip bundles, decode WebP textures through
# bpy.data.images (Blender ships a WebP codec; no external deps), then hand
# raw byte arrays to the bpy-free decoder in sog_format.py.

import json
import os
import shutil
import tempfile
import zipfile

import bpy
import numpy as np

from .sog_format import decode_sog


def _load_webp_rgba(filepath):
    """Load an image file to an (H, W, 4) uint8 array in top-down order."""
    img = bpy.data.images.load(filepath, check_existing=False)
    try:
        # data textures: no color transform, no alpha association
        img.colorspace_settings.name = 'Non-Color'
        img.alpha_mode = 'CHANNEL_PACKED'
        w, h = img.size
        if w == 0 or h == 0:
            raise ValueError(f'อ่านภาพไม่ได้: {os.path.basename(filepath)}')
        px = np.empty(w * h * 4, np.float32)
        img.pixels.foreach_get(px)
    finally:
        bpy.data.images.remove(img)

    rgba = np.round(px * 255.0).astype(np.uint8).reshape(h, w, 4)
    return rgba[::-1]  # Blender rows are bottom-up; SOG raster is top-down


def _load_from_dir(meta_path, max_splats, max_sh_bands):
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    base = os.path.dirname(meta_path)
    sections = ['means', 'scales', 'quats', 'sh0']
    if max_sh_bands > 0:
        sections.append('shN')
    textures = {}
    for section in sections:
        for name in meta.get(section, {}).get('files', []):
            if name not in textures:
                textures[name] = _load_webp_rgba(os.path.join(base, name))

    return decode_sog(meta, textures, max_splats, max_sh_bands)


def load_sog(filepath, max_splats=0, max_sh_bands=3):
    """Load a SOG scene from a bundled .sog (zip) or an unbundled meta.json."""
    if filepath.lower().endswith('.json'):
        return _load_from_dir(filepath, max_splats, max_sh_bands)

    if not zipfile.is_zipfile(filepath):
        raise ValueError('ไฟล์ .sog ไม่ใช่ zip bundle ที่ถูกต้อง')

    tmp = tempfile.mkdtemp(prefix='pobim_sog_')
    try:
        with zipfile.ZipFile(filepath) as z:
            z.extractall(tmp)
        meta_path = os.path.join(tmp, 'meta.json')
        if not os.path.exists(meta_path):
            raise ValueError('.sog bundle ไม่มี meta.json')
        return _load_from_dir(meta_path, max_splats, max_sh_bands)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

# POBIM Splats — 3D Gaussian Splatting for Blender

[![Release](https://img.shields.io/github/v/release/POBIM/pobim-splats)](https://github.com/POBIM/pobim-splats/releases)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](LICENSE)
[![Blender 4.2+](https://img.shields.io/badge/Blender-4.2%2B-orange)](https://www.blender.org/)
[![Sponsor](https://img.shields.io/badge/Sponsor-%E2%9D%A4-ff69b4)](https://github.com/sponsors/POBIM)

View 3D Gaussian Splats inside the Blender viewport with a **real GPU splat
renderer** — the same architecture as web viewers like SuperSplat: per-splat
data lives in GPU textures, drawn as screen-space ellipses (EWA splatting) in
**one draw call per cloud**. No per-splat objects, no geometry nodes — it stays
light and pretty at millions of splats.

![preview](tests/preview.png)

## Features

- **Real splat rendering** — projected 3D covariance with the same kernel
  behavior as the PlayCanvas engine (dilation + minimum kernel size), so
  surfaces read continuous at Splat Size 1; optional energy-conserving AA
- **View-dependent color (SH bands 1–3)** — loaded from `.ply` `f_rest`,
  compressed-PLY `sh`, and SOG `shN`; per-splat byte-packed on the GPU with
  a per-object View SH Bands control
- **Import without conversion** — standard `.ply`, SuperSplat
  `.compressed.ply`, and SOG v2 (`.sog` bundle or `meta.json`)
- **Fast** — depth sorting runs on a background thread (the viewport never
  stalls) and only re-sorts when the view direction rotates more than ~1°;
  panning and zooming cost zero sorts
- **Measure & Scale** — chained measurements with two pick modes: Surface
  (depth pick on the rendered splats) or Splat Centers; type the real-world
  distance and the scan snaps to real-world size (undoable)
- **Behaves like a Blender object** — each splat cloud is parented to an
  Empty: move, rotate, scale, hide, duplicate as usual; reloads automatically
  when you reopen the `.blend`

## Installation

1. Download `pobim_splats.zip` from the
   [latest release](https://github.com/POBIM/pobim-splats/releases/latest)
2. In Blender: `Edit > Preferences > Add-ons > (top-right menu) > Install from Disk…`
3. Enable **POBIM Splats**
4. Open the panel: press **N** in the 3D viewport → **POBIM3DGS** tab

Requires **Blender 4.2+** (tested on 4.5 LTS).

## Usage

Click **Import Splat** and pick a file:

| Format | Status |
|---|---|
| `.ply` (INRIA 3DGS, binary) | ✓ |
| `.compressed.ply` (SuperSplat) | ✓ decoded in-addon |
| `.sog` / `meta.json` (SOG v2) | ✓ decoded in-addon (Blender's WebP codec) |
| `.splat`, `.spz`, `.ksplat`, SOG v1 | ✗ convert first with [splat-transform](https://github.com/playcanvas/splat-transform): `npx @playcanvas/splat-transform input output.ply` |

Import options: **Max Splats** (random subsample for huge scenes; 0 = all,
hard cap ≈ 11.1M) and **Rotate to Z-up** (most scans are Y-down).

Per-splat controls in the panel: **Splat Size**, **Opacity**, **Reload**,
**Remove**, and **Measure & Scale**.

### Measure: Distance / Area / Volume

Open the **Measure** tool from the splat's panel box (tab **POBIM3DGS**;
the dropdowns next to the button pre-select kind and pick mode).
Measurements live on the splat object — they follow its transform and
persist across tool sessions and .blend save/load.

- **Left click** adds a point — or grabs an existing point to move it
- **Right click** finishes the current chain/polygon and **stays in the tool**
- **D / A / V** switches kind: Distance (chained segments), Area
  (polygon m² + perimeter), Volume (box m³ from two opposite corners)
- **M** toggles picking: **Surface** (depth pick on the rendered splats,
  like POBIMStudio) or **Splat Centers** (snap to nearest gaussian)
- **X** deletes the point under the cursor; **Esc** exits
- **Enter / S** opens the scale dialog: type the real-world length of the
  last segment and the cloud rescales about that segment's first point
  (undoable). **Clear Measurements** in the panel wipes everything.

### Edit Splats: select / hide / delete / export

Click **Edit Splats** in the splat's panel box to enter the editing tool
(SuperSplat-style state flags, selection shown in yellow):

- **Drag** a rectangle to select — **Shift** adds, **Ctrl** removes
- **A** select all · **Shift+A** none · **Ctrl+I** invert
- **H** hide selected · **Alt+H** unhide all
- **X / Del** delete selected (soft — undoable)
- **Ctrl+Z / Ctrl+Shift+Z** undo/redo (tool-local history)
- **Esc / right-click** exits; edit state persists in the .blend

**Export PLY** writes the surviving (non-deleted) splats. From a standard
`.ply` source the export is **lossless** — surviving rows are copied
byte-for-byte from the original file (all attributes, including full SH).
From `.compressed.ply`/`.sog` sources a standard `.ply` is synthesized from
the decoded values. Subsampled imports (Max Splats) export only the loaded
subset.

### Color tip

Blender's default AgX view transform washes out splat colors. For colors that
match web splat viewers, set
`Render Properties > Color Management > View Transform = Standard`
(the addon already converts colors to linear on import).

## Current limitations

- Viewport only — F12 renders (EEVEE/Cycles) don't include splats yet;
  use Viewport Render Image/Animation, or see the
  [roadmap](docs/ROADMAP.md)
- Orthographic cameras use an approximate (constant) Jacobian
- Center-mode picking subsamples to 400k gaussians for speed

## Development

```bash
python3 tests/test_ply_loader.py     # loaders + compressed roundtrip (no Blender needed)
python3 tests/test_measure_math.py   # pick/scale math (no Blender needed)
blender -b --factory-startup --python tests/smoke_test_blender.py   # operators + registry
blender --factory-startup --python tests/gpu_test_blender.py        # shader + GPU (needs display)
blender --factory-startup --python tests/render_preview_blender.py  # renders tests/preview.png
python3 tests/make_test_ply.py torus.ply 500000                     # generate test data
```

Module map: `ply_loader.py` (PLY + compressed decode, bpy-free) ·
`sog_format.py` / `sog_loader.py` (SOG v2) · `splat_gpu.py` (shader, draw
handler, threaded sort) · `measure_math.py` / `measure.py` (Measure & Scale) ·
`operators.py` / `ui.py` (UI). Details in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), plans in
[docs/ROADMAP.md](docs/ROADMAP.md).

**📚 [Notes on Implementing a 3D Gaussian Splatting Rasterizer](docs/SPLAT-RENDERING-NOTES.md)**
— general, implementation-agnostic lessons from building this renderer:
the projection math and its classic pitfalls (transposed Jacobian,
pixel→NDC factor), kernel conditioning, sorting theory, and how to test a
splat renderer against closed-form math. Useful even if you never touch
Blender.

## Support ❤

Free and open source under **GPL-3.0** (see [LICENSE](LICENSE) and
[THIRD_PARTY.md](THIRD_PARTY.md)) — free for personal and commercial use.

If this addon saves you time, consider
**[sponsoring @POBIM](https://github.com/sponsors/POBIM)** — sponsorships fund
the roadmap: F12 rendering, crop box, and view-dependent color (SH bands).

---

## ภาษาไทย (สรุปย่อ)

แสดงผล 3D Gaussian Splats ใน Blender ด้วย splat renderer จริง — เบาและสวย
ระดับหลายล้าน splats รองรับ `.ply`, `.compressed.ply`, `.sog` โดยไม่ต้องแปลงไฟล์

**ติดตั้ง**: โหลด zip จาก [Releases](https://github.com/POBIM/pobim-splats/releases/latest)
→ `Edit > Preferences > Add-ons > Install from Disk…` → เปิดใช้งาน →
กด **N** ใน viewport → แท็บ **POBIM3DGS**

**Measure & Scale**: กดปุ่มในแผง → คลิกจุดสองจุดบน splat → ใส่ระยะจริง →
โมเดลถูกปรับสเกลให้ตรงหน่วยจริง (Ctrl+Z ย้อนได้)

**สีให้ตรงกับ web viewer**: ตั้ง `Color Management > View Transform = Standard`

ถ้า addon นี้มีประโยชน์ สนับสนุนได้ที่
[github.com/sponsors/POBIM](https://github.com/sponsors/POBIM) ครับ 🙏

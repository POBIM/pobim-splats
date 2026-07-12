# Phase 3 Spec: Transform Selected Splats + Edit-Tool UX Overhaul

Two tracks. Track T (transform core) and Track U (UX/HUD + gesture
integration). Track U owns ALL edit_tools.py changes; Track T must not
touch edit_tools.py — it exposes APIs that U wires in.

## Track T — transform core

### T1. Edit overrides (`pobim_splats/splat_edits.py`, NEW, bpy-free)

Sparse per-splat geometry overrides in SPLAT-LOCAL space:

```python
class SplatEdits:
    def __init__(self, count):
        # dense override arrays allocated lazily on first edit:
        self.positions = None   # (N,3) f32 copy-on-first-edit of cloud positions
        self.quats = None       # (N,4) f32 (w,x,y,z) — needs raw quats, see T4
        self.scales_log = None  # (N,3) f32
        self.dirty = np.zeros(count, bool)   # which splats differ from source
        self.version = 0
    def apply_matrix(self, indices, mat4_local, base_positions, ...):
        """Apply a 4x4 LOCAL-space transform about an arbitrary pivot to
        `indices`: positions = M @ p; quats = quat(M_rot) * q; scales_log +=
        log(scale factors) (per-axis from M's column norms — uniform and
        per-axis both work). Marks dirty, bumps version. Returns
        (indices, before, after) payloads for history (positions/quats/
        scales of changed indices)."""
    def restore(self, indices, positions, quats, scales)   # for undo
    def serialize(self) / deserialize(s, count)  # dirty-sparse: count header +
        # packed indices + f32 payloads, zlib+base64 (same style as SplatState)
```

History integration: EditHistory ops are already dicts — add a 'kind'
field: 'flags' (existing, default) or 'transform' (indices + before/after
position/quat/scale arrays). splat_state.EditHistory.undo/redo dispatch on
kind: flags → set_flags_raw as now; transform → callback the caller
provides (keep EditHistory generic: give it an `apply_fn(op, direction)`
registry or let undo/redo return the op and the CALLER applies non-flags
kinds — choose the simplest that keeps existing tests green).

### T2. GPU preview + commit (`pobim_splats/splat_gpu.py`)

- Interactive drag must NOT re-upload textures per frame. Add to the UBO
  (48 → 68 floats): `mat4 previewMatrix; vec4 misc;` misc.x = preview
  active flag. Vertex shader: when misc.x > 0.5 AND splat SELECTED:
  center = (previewMatrix * vec4(center,1)).xyz BEFORE the modelView
  transform, and Vrk = P3 * Vrk * transpose(P3) with P3 = mat3(previewMatrix)
  (covariance transform; correct for rotation+uniform scale, acceptable
  for per-axis scale MVP). All UBO writers updated (draw, render_depth_map,
  tests' make_ubo — identity matrix + zeros default keeps old tests green).
- Draw callback reads preview from module-level `PREVIEW` dict {uid:
  (mat4 ndarray, active)} set by the tool; identity when absent.
- Commit path: `SplatGPU.refresh_geometry(cloud_positions, cov6, indices)`
  — NO: simplest correct MVP = rebuild the data texture from a maintained
  CPU texel array. Keep `self._texels` (the (rows*W,4) array built in
  __init__) alive; add `update_splats(indices, positions, cov6)` that
  rewrites texels[base+0].xyz / [base+1].xyz / [base+2].xyz for those
  indices and re-creates the data texture (full re-upload, once per
  commit — acceptable). Memory cost of keeping _texels: same as the
  texture; document. Positions used by sorting/picking (`sg.positions`)
  must ALSO be updated in place.
- After commit also invalidate the sort (`self._applied_dir = None`).

### T3. Export with edits (`pobim_splats/splat_export.py`)

`export_ply(..., edits=None)` where edits = dict with 'indices' (into the
LOADED order), 'positions' (n,3), 'quats' (n,4 wxyz), 'scales_log' (n,3):
- kind 'ply': after masking rows, patch fields x/y/z, rot_0..3,
  scale_0..2 of the surviving edited rows (map loaded indices → file rows
  via source_indices, then → positions within the surviving subset).
  Untouched rows stay byte-identical.
- kind 'canonical': overwrite the arrays at those indices before writing.
- Tests: patched roundtrip — export with edits, re-load, edited splats
  match new values, untouched rows byte-identical.

### T4. Raw geometry for transforms (`pobim_splats/ply_loader.py`)

Transforming quats/scales requires the RAW values (cov6 alone is not
invertible to quat+scale). Add `keep_geometry=False` param to build_cloud:
when True, stash `cloud.quats` (N,4 f32 normalized) and
`cloud.scales_log` (N,3 f32) BEFORE cov computation (subsample-aligned).
Importer passes True (memory +28B/splat — fine). ensure_gpu must NOT free
these two (they're needed for later transforms) but MAY free the rest as
today. SOG/compressed paths pass their decoded quat/scales through
build_cloud already — verify keep_geometry works for all three formats.

### T5. Transform math helper (`pobim_splats/transform_math.py`, NEW, bpy-free)

quat multiply (wxyz), mat3→quat, build local transform matrix about pivot
from (mode, axis_lock, mouse delta params) — plus unit tests. Rotating a
splat does NOT rotate its SH coefficients in this MVP (document: slight
view-dependent color error on rotated splats with bands ≥ 1).

## Track U — UX overhaul + gesture integration (edit_tools.py, ui.py, __init__.py)

### U1. In-viewport HUD (the tools are too many for hotkeys alone)

Clickable chip toolbar drawn top-center of the viewport during the modal
(POST_PIXEL, same style language: dark rounded rect + white text, active
chip in POBIM orange #ffa500):
`[Rect] [Lasso] [Poly] [Brush] [Sphere] [Box] │ [Move] [Rotate] [Scale] │ [↶] [↷] │ [✓ Done]`
- Hit-test on LEFTMOUSE PRESS before any gesture logic; hovering a chip
  highlights it and suppresses tool gestures under it.
- Second row (contextual): for BRUSH/SPHERE a radius readout chip
  `Radius: 42 px` / `0.25 m` that acts as a DRAG SLIDER (click-drag left/
  right adjusts, like Blender's number drag), for BOX a hint chip.
- Undo/redo chips call the history; Done exits cleanly (= Esc when idle).

### U2. Radius management (the user reports [ ]/sizing is unusable)

- Keep [ ] but add: **F** = interactive resize (Blender brush convention:
  mouse distance from gesture start sets radius live, LMB/Enter confirm,
  Esc/RMB revert) for BRUSH (px) and SPHERE (world radius via the same
  screen-projected circle).
- **Alt+Wheel** = radius ±10% (do NOT consume plain wheel — navigation).
- Radii live in Scene props (`pobim_splat_brush_radius` px 4..400 default
  40, `pobim_splat_sphere_radius` m 0.001..100 default 0.25) registered in
  __init__.py, shown as sliders in the panel Edit section, read/written by
  the modal so they persist across sessions.
- Status bar always shows the current radius for the active tool.

### U3. Box gesture v2 (currently "unmanageable")

Box becomes adjustable before applying: corner1 click → corner2 click →
PREVIEW stage: wireframe + the two corner handles drawn as grabbable dots
(reuse measure.py's grab pattern: hover ring, click-drag moves a corner
via surface pick), radius chips show box dimensions; ENTER or a click on
the HUD `[Apply]` chip commits the selection; Esc/RMB cancels the box.
Sphere similarly gets a PREVIEW stage: click places it, center is then
draggable, F/slider adjusts radius, ENTER applies.

### U4. Transform modes (wire Track T)

- Keys **1/2/3** (TS convention) or HUD chips = Move/Rotate/Scale mode,
  **G** = alias for Move. Requires a non-empty selection (status hint if
  empty).
- MOVE: mouse drag translates selection in the view plane; X/Y/Z lock to
  a LOCAL axis; SHIFT slows (precision).
- ROTATE: drag rotates about the view axis through the selection centroid;
  X/Y/Z lock to local axes.
- SCALE: drag distance ratio from centroid, uniform; X/Y/Z per-axis lock.
- Preview via splat_gpu.PREVIEW (live, no texture uploads); LMB/ENTER
  confirms → SplatEdits.apply_matrix + SplatGPU.update_splats + history
  op kind 'transform' + persist edits (obj['pobim_splat_edits']); ESC/RMB
  cancels preview cleanly.
- Undo/redo of transform ops: restore via SplatEdits.restore + GPU update.
- Export operator passes the edits payload to export_ply.

### U5. Panel reorganization (ui.py)

The panel is crowded. Restructure into labeled sub-boxes: **Display**
(Show/AA/Near Cull/Sort), per-splat box with sections **Splat** (counts,
size, opacity, SH), **Measure** (button+dropdowns+clear), **Edit**
(Edit/Export buttons, tool dropdown, radius sliders, edit counts). Keep
every existing prop reachable; no behavior changes.

## Tests

- T: test_transform_math.py, test_splat_edits.py (apply/restore/serialize
  roundtrip), extended test_splat_export.py (patched rows), gpu test:
  preview matrix identity default keeps all analytic tests green + one new
  check (preview translation moves the rendered splat; commit path
  update_splats moves it permanently and sort still works).
- U: smoke additions (radius scene props register/roundtrip; HUD constants
  sane). Modal interactions remain manual-test items — list them in the
  report.

## Non-goals

SH rotation on transform (documented limitation), numeric input, multi
splat-object transforms, gizmo widgets, flood select.

# Phase 1 Spec: Splat Editing Foundation

Goal: per-splat selection/hidden/deleted state, rect-select edit tool with
its own undo stack, and lossless .ply export of surviving splats. Mirrors
the SuperSplat-style editing model (state bit flags + EditHistory).

## Module: `pobim_splats/splat_state.py` (bpy-free where possible)

```python
class State:            # bit flags, mirrors the TS editor convention
    SELECTED = 1
    HIDDEN   = 2
    DELETED  = 4

class SplatState:
    def __init__(self, count): self.flags = np.zeros(count, np.uint8); self.version = 0
    # every mutator bumps self.version (GPU re-upload trigger) and returns
    # the np.ndarray of CHANGED indices (for undo ops); no-op -> empty array
    def select_indices(self, indices, op='set')      # op: set|add|remove
    def select_all(self) / select_none(self) / select_invert(self)
    def hide_selected(self) / unhide_all(self)
    def delete_selected(self)                        # soft delete; also clears their SELECTED
    def set_flags_raw(self, indices, values)         # for undo restore
    @property num_selected / num_hidden / num_deleted
    def visible_mask(self)   # ~(HIDDEN|DELETED) as bool array
    def keep_mask(self)      # ~DELETED  (for export)
    def serialize(self) -> str    # base64(zlib(packbits per flag)) for .blend persistence
    @staticmethod deserialize(s, count) -> SplatState

class EditHistory:       # tool-local undo (Blender undo can't see our numpy state)
    def push(self, op)   # op = {'label': str, 'indices': ndarray, 'before': ndarray(u8), 'after': ndarray(u8)}
    def undo(self, state) / redo(self, state)   # apply via state.set_flags_raw
    can_undo / can_redo / clear()
```

## GPU integration (`splat_gpu.py`)

- `SplatEntry` gains `.state: SplatState | None` (created lazily on first
  edit; `None` means untouched → zero overhead).
- `SplatGPU` gains a **state texture**: R32F, 1 texel/splat, same
  `%width` addressing pattern as the order texture (width 2048), value =
  `float(flags)`. New method `upload_state(flags)`; draw callback compares
  `entry.state.version` with `sg.state_version` and re-uploads when stale.
- Shader (both FRAG paths share the VERTEX): fetch `float st =
  texelFetch(stateTex, ...)`; bits via `int(st+0.5)`:
  - DELETED or HIDDEN → `emitDegenerate()`
  - SELECTED → tint: `rgb = mix(rgb, uSelColr, 0.55)` AFTER SH + srgb
    conversion. Selection color passed in the spare `u.params2` slot? All 4
    used → use `u.camPos` spare? none. **Extend UBO by one vec4
    `selColor`** (48 floats) — update ALL writers (draw, depth pick,
    tests' make_ubo) and the typedef. selColor.a = 1.0 when a state
    texture is bound, 0.0 when not (shader skips state fetch → zero cost
    for non-edited splats; bind data_tex as dummy stateTex then).
  - Selection color: POBIMStudio default selected color (check
    `src/scene-config.ts` in the sibling repo /home/pobimgroup/SplatsRender
    — do NOT copy code, just the RGB constant).
- Depth-pick pass automatically skips hidden/deleted (shared vertex path).
- `render_depth_map` binds the state texture the same way.

## Persistence

- On every committed edit: `obj['pobim_splat_state'] = state.serialize()`
  (packbits+zlib+base64 ≈ 200KB per 1.5M splats). `load_entry_for_object`
  restores it when count matches.

## Export (`pobim_splats/splat_export.py`)

`export_ply(obj, entry, filepath)`:
- Re-reads the SOURCE file (`obj.pobim_splat_file`) — no edited values are
  stored in RAM, so export is LOSSLESS for .ply sources: reuse
  `ply_loader._parse_header/_read_elements`, take the `vertex` structured
  array, apply the keep mask, write binary_little_endian .ply with the
  ORIGINAL dtype/fields (including f_rest, normals — verbatim rows).
- Subsampled imports (`pobim_splat_max`): cloud must record
  `source_indices` (the permutation indices) so the mask maps back to file
  rows; splats NOT loaded are DROPPED from export (document this).
- compressed.ply / SOG sources: decode via the existing loaders, then
  synthesize a standard .ply: x/y/z, f_dc (from color: `(c-0.5)/SH_C0`),
  opacity logit, scales log, rot quat, f_rest from the quantized u8 sh
  (dequantize (v/255-0.5)*8). Requires loaders to OPTIONALLY return raw
  quat/scales — add `keep_raw=False` param to build_cloud that stashes
  `cloud.raw = {'scales_log', 'quat'}`... NO — simpler: exporter calls the
  format decoder itself with a new `raw=True` mode returning canonical
  arrays. Keep the API minimal: `ply_loader.load_raw(filepath)` /
  `sog_loader.load_raw(filepath)` returning dict of canonical arrays.
- Operator `pobim_splats.export_ply` with `showSaveFilePicker`-style
  ImportHelper (filename_ext='.ply'), reports count written.

## Edit tool (`pobim_splats/edit_tools.py`) — modal operator

- `pobim_splats.edit_splats` (uid) — modal like measure.py (copy its
  lifecycle: `_running` guard, `cancel()`, status text, POST_PIXEL overlay).
- **Rect select**: LEFTMOUSE drag draws a rect (POBIMStudio style: thin
  white line + dark under-stroke); on release select splats whose projected
  centers fall inside. Modifiers: none=set, SHIFT=add, CTRL=remove.
  Projection via `measure_math.project_to_pixels` on ALL splat positions
  (full count, chunked matmul if > 2M).
- Keymap (match TS editor where sensible):
  - `A` select all, `Alt+A`/`Shift+A` none, `Ctrl+I` invert
  - `H` hide selected, `Alt+H` unhide all
  - `X`/`DEL` delete selected (soft)
  - `Ctrl+Z` undo, `Ctrl+Shift+Z` redo (tool-local EditHistory)
  - `Esc`/`RIGHTMOUSE` exit tool
- Every mutation: build op {indices, before, after}, push to history,
  bump version, save serialized state to the object, tag_redraw.
- Header status: `Selected 12,345 / 1,485,076 · Hidden 0 · Deleted 3,210`.
- Panel: Edit Splats button + Export PLY button + counts row.

## Tests

- bpy-free: `tests/test_splat_state.py` — flags ops, changed-indices
  return values, serialize roundtrip, EditHistory undo/redo chains.
- bpy-free: export roundtrip — write synthetic ply → load → delete mask →
  export → re-parse → row count and byte-identical surviving rows.
- smoke (blender -b): operators registered, export operator writes a file,
  state persists through save→`load_entry_for_object`.
- GPU (foreground): selected splats tint (render torus, select half by
  index, assert mean color shift in selected region), hidden/deleted not
  drawn (coverage drops), state texture re-upload on version bump.

## Non-goals (later phases)

Lasso/polygon/brush/sphere/box selection; transform of selected gaussians;
locked flag; export of compressed formats.

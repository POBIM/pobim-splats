# Phase 4 — Duplicate/Separate selection + 10M-splat performance (v0.8.0)

Two independent tracks. Track F (features) and Track P (performance) touch
disjoint code regions except where noted in "Seams" — read that section first.

Reference semantics come from SuperSplat/POBIMStudio (`select.duplicate` /
`select.separate` in editor.ts `performSelectionFunc`) and playcanvas@2.20.6.
Do NOT read the SplatsRender repo; everything needed is in this spec.

---

## Track F — Duplicate / Separate selection, export-selected, HUD select ops

### F1. SplatCloud.take(rows) helper (`ply_loader.py`)

`SplatCloud.take(rows: np.ndarray) -> SplatCloud` — new cloud containing the
given rows (int64 indices into THIS cloud). Slices every non-None array:
positions, cov6, colors, opacities, sh (2-D), quats, scales_log, and composes
`source_indices = self.source_indices[rows]` (absolute rows in the source
file — this keeps lossless export working). Copies scalars (sh_bands, count).
Must not mutate self. Arrays are copied (`np.ascontiguousarray`), not views —
the new cloud outlives the old one.

### F2. Subset persistence (`splat_gpu.py` + loader path)

New persisted property `obj['pobim_splat_subset']`: serialized absolute
source-file row indices, format identical to the other payloads —
8-byte LE count header (= number of indices) + zlib(level 6) of int64 bytes +
base64 ascii. Add `serialize_rows(rows) / deserialize_rows(payload)` helpers
in `splat_state.py` (bpy-free, unit-testable; ValueError on corrupt).

`load_entry_for_object`: after loading the cloud, if the property exists:
- decode rows; the subset load MUST ignore `pobim_splat_max` (load with
  `max_splats=0`) because random subsampling would not contain the rows.
  Enforce: when `pobim_splat_subset` is set, pass `max_splats=0` to the
  loader, then `cloud = cloud.take(np.searchsorted-free direct slice)` —
  with a full load `source_indices` is identity, so the absolute rows ARE
  the slice indices. Validate `rows.max() < cloud.count` else discard the
  property with a console warning (same stale-payload pattern as state/edits).
- `obj.pobim_splat_count` must reflect the subset count.
State/edits payload restore then proceeds unchanged against the subset count
(they are stored per-duplicate-object and already sized to the subset).

### F3. Duplicate / Separate operators (`edit_tools.py` or new `splat_ops.py`)

`POBIM_OT_duplicate_selection(uid)` — "Duplicate Selection":
1. Guard: entry exists, `entry.state` has rows with flags EXACTLY == SELECTED
   (mirror SuperSplat: `state == State.selected` — a gaussian that is also
   HIDDEN or DELETED is excluded). Report `{'CANCELLED'}` + error if none.
2. `rows_abs = entry.cloud_source_rows[sel_rows]` — absolute source rows via
   the entry's kept `source_indices` (this survives ensure_gpu array freeing).
3. Create a new Empty like POBIM_OT_import_splat does (PLAIN_AXES, size 0.5,
   fresh uuid4 uid, same `pobim_splat_file`, `pobim_splat_max = 0`,
   same `pobim_splat_shmax` / `pobim_splat_srgb` / display props). Name:
   `f'{source.name} Selection'` (deviation from TS's verbatim name reuse —
   deliberate, Blender auto-suffixes .001 anyway). Copy `matrix_world` from
   the source empty and copy its parent, so the subset appears exactly where
   the source's gaussians are (single-source case; multi-source duplicate is
   out of scope — one source object per operation).
4. Persist `obj['pobim_splat_subset'] = serialize_rows(rows_abs)`.
5. Carry over TRANSFORM EDITS: if `entry.edits` has edited rows intersecting
   `sel_rows`, re-index those rows into the subset's local indexing and write
   `obj['pobim_splat_edits']` for the new object (same payload format,
   count = subset count). Do NOT carry selection/hidden/deleted state — the
   copy starts with a fresh zero state (TS parity).
6. Load the new entry immediately via `load_entry_for_object(new_obj)` so it
   draws this session (reuse existing reload operator flow), select + make
   active the new object, deselect the source object at the Blender level.
   The SOURCE's per-gaussian SELECTED flags stay untouched (TS parity).
7. `bl_options = {'REGISTER', 'UNDO'}` so Blender undo removes the object.

`POBIM_OT_separate_selection(uid)` — "Separate Selection":
Steps 1–7 identical, PLUS: set DELETED on `sel_rows` of the SOURCE entry
(`entry.state` mutator, keep SELECTED bit set — TS parity: deleted rows end
up selected|deleted), push one 'flags' op onto the source `EditHistory`,
re-upload the source state texture, bump versions, tag redraw. Known
deviation from TS: undo is two-step here (Blender object undo + our flags
undo) instead of one atomic MultiOp — document it in the README.

Both operators must work while the edit modal is running on the source
(they are invoked from HUD chips) AND from the N-panel when the modal is not
running. When invoked from the modal, use the modal's live state object.

### F4. Export selected only (`splat_export.py` + export operator)

`POBIM_OT_export_ply` gains a checkbox `selected_only` (default False,
enabled only when a selection exists): keep_mask additionally requires the
SELECTED bit (mask = not DELETED and (not selected_only or SELECTED)).
`export_ply` itself needs no change if the operator builds the mask —
verify keep_mask + source_indices + edits still compose (they do today for
delete; add a bpy-free test for the selected_only mask path).

### F5. HUD + panel UI (`edit_tools.py` HUD, `ui.py` panel)

- HUD gets a "Select" chip group: **All · None · Invert** (fire the existing
  A / Shift+A / Ctrl+I actions) and an "Object" chip group:
  **Duplicate · Separate** (call the new operators). Reuse the existing chip
  drawing/hit-test code; keep orange active styling; chips disabled (dim)
  when there is no selection.
- N-panel Edit box gets the same two buttons (Duplicate / Separate) with the
  selection count in the label when available.

### F6. Tests (Track F owns these)

- `tests/test_splat_state.py`: serialize_rows/deserialize_rows roundtrip +
  corrupt/mismatch ValueError.
- `tests/test_ply_loader.py`: `SplatCloud.take` slices every array, composes
  source_indices, leaves the source cloud untouched; take-of-take composes.
- `tests/test_splat_export.py`: selected_only mask; duplicate-object export
  (subset source_indices) stays byte-identical for untouched rows.
- `tests/smoke_test_blender.py`: end-to-end — import torus, select half by
  predicate, run duplicate op → new object exists with correct count, subset
  payload persists; save/reload path (`load_entry_for_object` on a fresh
  object dict) restores the subset AND carried transform edits; separate op
  soft-deletes on the source and the source export drops those rows.

---

## Track P — 10M-splat performance (sort + geometry + uploads)

GOAL: 10M splats interactive on desktop with ZERO image-quality change —
the fragment/vertex math and kernel constants must not change. Only the
order computation, geometry feed, and upload strategy may change.

All numbers below were MEASURED on this machine (Blender 4.5, numpy 2.5.1,
N = 10,000,000) by a probe agent, and the engine behavior was read from
playcanvas@2.20.6 sources. Treat them as ground truth; do not re-derive.

### P1. O(N) integer-key sort (`depth_sort.py`, new bpy-free module)

Replace `np.argsort(positions @ row)` in `SplatGPU.sort_if_needed`'s worker
with a new bpy-free function `compute_order(positions, row) -> (order i32,
behind_count int)` in `pobim_splats/depth_sort.py`:

```python
d = positions @ row                       # 13 ms at 10M
dmin = float(d.min()); dmax = float(d.max())
if dmax <= dmin: return arange, 0
bins = np.clip((d - dmin) * (65535.0 / (dmax - dmin)), 0, 65535).astype(np.uint16)
order = np.argsort(bins, kind='stable')   # 106 ms — numpy radix path
```

MEASURED: 521 ms → ~158 ms at 10M. TRAPS (verified): `kind='stable'` hits
numpy's O(N) radix ONLY for int dtypes ≤ 16 bits — uint32 keys (807 ms) or
default kind on ints (729 ms) are both SLOWER than the naive float argsort.
Use uint16, never "more bits for accuracy" (measured mean rank error 80 of
10M — imperceptible for back-to-front blending; the playcanvas engine
itself quantizes to ~2^20 buckets, so quantized ordering is engine parity).
Keep the existing threading/gating structure (worker thread, >1° direction
gate, interval throttle, _sort_result pickup) — only the kernel changes.

### P2. Instanced draw — kill the N*6 vertex VBO

Blender 4.5 `GPUBatch.draw_instanced(program, *, instance_start=0,
instance_count=0)` EXISTS and `gl_InstanceID` is available as a plain GLSL
builtin in GPUShaderCreateInfo shaders (proven by an offscreen test that
round-tripped 16 distinct instance IDs through pixels).

- Replace the per-splat VBO (currently `n*6` vertices × 2 F32 attrs,
  480 MB + 553 ms build at 10M) with ONE static 4-vertex quad VBO
  (a single `vec2 corner` attr holding the 4 corner signs) + a
  `gpu.types.GPUIndexBuf` of 2 triangles, built once in `SplatGPU.__init__`
  (0.03 ms, constant size).
- Vertex shader: delete `quadId`/`cornerId` vertex_ins; splat index =
  `gl_InstanceID` (replaces quadId everywhere incl. the order-texture
  fetch); the corner offset comes directly from the `corner` attribute.
  NO other shader math may change — kernel, falloff, culls, UBO layout
  (68 floats) are all frozen; renders must stay visually identical.
- EVERY `batch.draw(shader)` call site becomes
  `batch.draw_instanced(shader, instance_count=sg.count)`:
  the main draw handler, `render_depth_map`, `tests/gpu_test_blender.py`,
  and `tests/render_preview_blender.py`. Grep for `.draw(` to catch all.
- Optional free win (do it, it's small): `sort_if_needed` already sorts
  ascending along view z (farthest first; behind-camera splats sort to the
  tail on the + side). Have `compute_order` also return
  `draw_count = np.searchsorted(bins[order], behind_threshold_bin)`-style
  trim ONLY IF trivially correct with the existing z-clamp shader behavior;
  otherwise skip and keep drawing `sg.count` instances — correctness first.

### P3. Texture strategy — keep recreation (measured: not a bottleneck)

Blender 4.5 Python has NO in-place GPUTexture update (verified:
GPUTexture exposes only clear/format/height/read/width). Full recreation
of a 40 MB R32F 10M-texel texture measured at ~13 ms — fine off the hot
path. Keep `_upload_order` / `upload_state` / `update_splats` recreation
as-is; just avoid needless numpy copies (e.g. reuse the padded scratch
array across sorts as an instance attribute — allocate once).

### P4. Do NOT expand scope

No mixed-precision/packed data textures this phase (half-precision cov6
would violate the zero-quality-change rule), no LOD, no sub-pixel budget
mask, no UBO changes. Note them in the spec's Future-work margin only.

### P5. Tests (Track P owns these)

- New bpy-free `tests/test_depth_sort.py`: compute_order vs
  `np.argsort(pos @ row)` reference — assert bin-monotonicity of the
  result, rank-error bound (< 500 at 1M random), identical behavior on
  degenerate inputs (all-equal depths, n=0, n=1).
- New bpy-free `tests/bench_sort.py`: times old vs new kernel at 10M,
  prints ms, asserts new < old (loose threshold, CI-safe).
- `tests/gpu_test_blender.py`: switch harness draws to draw_instanced and
  confirm ALL existing analytic checks still pass unchanged (profile
  α(σ)≈0.580/0.599, edge/center ratio 1.27, z-needle 3px→41px, selection
  tint, delete/hidden, transform preview/commit) — these ARE the
  image-quality guard. Add one new check: instanced draw with
  instance_count=count renders the same nonzero-pixel coverage as before
  (record the expected coverage constant from a pre-change run if needed).
- `tests/render_preview_blender.py`: update draw call; preview.png must
  still render the torus.

### Seams (both tracks read this)

- Track F touches: ply_loader (take), splat_state (row serializers),
  new operators, HUD chips, export operator, load_entry_for_object (subset
  branch). Track P touches: SplatGPU internals (init/batch/sort/_upload_*),
  shader sources, draw handler, render_depth_map, gpu_test harness.
  SHARED FILE: `splat_gpu.py` — the tracks run IN PARALLEL, so on this file
  use the Edit tool ONLY (string replacement with unique anchors), NEVER
  Write/full-rewrite. Track F edits ONLY inside `load_entry_for_object`;
  Track P must NOT touch `load_entry_for_object` or `_apply_persisted_edits`.
  Do not reformat outside your region.
- Track F's duplicate calls `load_entry_for_object` → `ensure_gpu` — it must
  keep working when Track P changes SplatGPU's constructor internals; the
  constructor SIGNATURE `SplatGPU(cloud)` is frozen.
- `update_splats`, `upload_state`, `geometry_version`, `_texels` are API
  used by edit_tools/measure — Track P may change their internals but not
  their names/signatures/observable behavior.
- The 68-float UBO layout is FROZEN in this phase.

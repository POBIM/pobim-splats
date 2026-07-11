# Phase 2 Spec: Selection Tools (Lasso / Polygon / Brush / Sphere / Box)

Extends the Phase-1 edit modal (`pobim_splats.edit_splats`) with the
remaining SuperSplat-style selection tools. Rect stays the default. All
tools honor the same modifier semantics at commit time: none=set,
SHIFT=add, CTRL=remove. Every commit is ONE EditHistory op (snapshot
before-flags of changed indices), persists state, redraws.

## Module: `pobim_splats/select_math.py` (NEW, bpy-free, numpy only)

```python
def points_in_polygon(px, poly):
    """px (N,2) float32 pixels; poly list[(x,y)] len>=3.
    Vectorized ray-casting (even-odd). Returns bool (N,)."""

def points_near_polyline(px, stroke, radius):
    """px (N,2); stroke (S,2) ordered points; True where min distance from
    px to ANY stroke point <= radius. S is capped by the caller (<=512);
    compute in chunks of 250k px rows x S to bound memory. Segment-accurate
    distance NOT required (stroke points are dense); point distance is fine."""

def points_in_sphere(world, center, radius):
    """world (N,3) float32; bool mask."""

def points_in_box(world, bmin, bmax):
    """world (N,3); axis-aligned min/max in the SAME space as `world`
    (caller passes splat-local positions + local corners for an
    object-aligned box). bool mask."""
```

Tests `tests/test_select_math.py` (bpy-free): polygon = concave L-shape and
star (points inside/outside/near edges), clockwise AND counter-clockwise
winding both work; polyline = diagonal stroke radius test incl. chunk
boundary (N > chunk size); sphere/box basics + empty results.

## Edit tool changes (`pobim_splats/edit_tools.py`)

Tool mode state `self._tool` in the modal: RECT (default) | LASSO |
POLYGON | BRUSH | SPHERE | BOX. Key switching (PRESS, consume): R/L/P/B/S
match the TS editor; box uses **C** (X is taken by delete). Switching
tools cancels any in-progress gesture. Status text shows the active tool
and its hints (keep the Thai style of the existing string; add tool name).

Scene enum `pobim_splat_edit_tool` mirrors the mode (updated on switch, read
on invoke) so the panel can pre-select it — register in `__init__.py`,
show as a dropdown next to the Edit Splats button in `ui.py`.

### LASSO (L)
LEFTMOUSE press starts a freehand path; MOUSEMOVE appends points >= 4px
apart (cap 512 points: if exceeded, decimate by dropping every other point
and doubling the spacing threshold — keeps long strokes working); release
closes the path (>=3 points, else ignore) and commits via
points_in_polygon on projected centers (reuse the Phase-1 chunked
projection helper; exclude HIDDEN|DELETED).

### POLYGON (P)
Clicks add vertices; MOUSEMOVE shows a rubber-band edge; RET/NUMPAD_ENTER
or a click within 8px of the FIRST vertex closes and commits (>=3 vertices);
BACK_SPACE removes the last vertex; RIGHTMOUSE/ESC cancels the polygon
in progress (tool stays active; ESC with no polygon exits the modal as
before — preserve Phase-1 exit semantics for RECT).

### BRUSH (B)
Circle cursor of radius `self._brush_radius` px (default 40; '['=×0.8,
']'=×1.25, clamp 4..400, matches the TS bracket keys). LEFTMOUSE press
snapshots flags and starts a stroke; MOUSEMOVE appends stroke points >=
brush_radius*0.3 px apart AND live-applies selection incrementally (apply
add/remove per new point using points_near_polyline on JUST the new
point(s) for responsiveness); release commits ONE history op via diff
against the stroke-start snapshot (indices = where flags changed).
Modifier read at stroke START: none/SHIFT=add-style painting (set uses
add during the stroke but clears selection first at stroke start),
CTRL=remove painting.

### SPHERE (S)
Mouse position continuously surface-picks a 3D center (copy the depth-map
pattern from measure.py: render_depth_map on view change +
read_depth_pixel + unproject_pixel; fall back to nearest-center pick).
`self._sphere_radius` world units (default 0.25; '['/']' scale ×0.8/×1.25).
Overlay: project the center and draw a circle whose pixel radius =
radius * (pixel focal)/|view z| (compute from rv3d matrices). LEFTMOUSE
click commits: mask = points_in_sphere(world_positions, center, radius)
minus HIDDEN|DELETED.

### BOX (C)
Two surface-picked clicks = opposite corners in SPLAT-LOCAL space
(transform picked world points by inv(matrix_world)); axis-aligned box in
local space (object-aligned in world). Preview wireframe after the first
click (reuse measure.py's box_corners/BOX_EDGES via measure_math import).
Second click commits points_in_box on the LOCAL positions (entry
positions are already local — no transform needed). RIGHTMOUSE/ESC cancels
the first corner.

### Shared plumbing
- Projection of centers for LASSO/POLYGON/BRUSH: reuse the existing
  Phase-1 helper (`_project_all` or equivalent) — do not duplicate.
- Depth-pick helpers: extract measure.py's `_ensure_depth_map`/pick logic
  into small private copies inside edit_tools (do NOT modify measure.py).
- Overlay: reuse the drawing-style constants; add: lasso path (white on
  dark, like measure lines but 1.5/2px), polygon edges + first-vertex ring,
  brush circle (white, dark under-stroke), sphere circle, box wireframe.
- All commits go through the existing `_apply` (history + persist +
  status update). Empty masks = no-op (no history entry).

## Tests

- `tests/test_select_math.py` per above (bpy-free).
- smoke: scene enum registers; simulate tool selection via scene prop.
- GPU test additions are NOT required (selection application is already
  covered); do not touch gpu_test_blender.py.

## Non-goals

Flood select, moving selected splats (phase 3), locked flag, oriented
(rotated) box.

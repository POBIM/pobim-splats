# Notes on Implementing a 3D Gaussian Splatting Rasterizer

Lessons learned building a splat renderer from scratch (on Blender's `gpu`
API, but nothing here is Blender-specific unless marked). Every item below
was a real bug or a real design decision in this project, verified against
closed-form math and the open-source reference implementations:

- the [INRIA reference CUDA rasterizer](https://github.com/graphdeco-inria/gaussian-splatting)
- the [PlayCanvas engine](https://github.com/playcanvas/engine) gsplat pipeline
- [antimatter15/splat](https://github.com/antimatter15/splat) (WebGL)

They differ from each other in subtle ways. When porting, pick ONE and match
it exactly — mixing conventions is where the bugs below came from.

---

## 1. The projection math

### 1.1 EWA splatting in one paragraph

A 3D gaussian with covariance `Σ = R·S·Sᵀ·Rᵀ` (from quaternion `R`, scale
`S`) is projected to a screen-space 2D gaussian by linearizing the
perspective projection at the splat center:

```
cov2D = J · A · Σ · Aᵀ · Jᵀ            (take the top-left 2×2)
```

where `A` is the 3×3 model-view rotation/scale block and `J` is the Jacobian
of the projection evaluated at the view-space center `t = (x, y, z)`:

```
J = | f/z   0    -f·x/z² |
    | 0    f/z   -f·y/z² |
    | 0     0        0   |        (f = pixel focal length = proj00 · width/2)
```

The `-f·x/z²` terms are the **perspective tilt**: they make splats
foreshorten and skew toward the screen periphery. At screen center they are
zero — which is exactly why bugs in them are invisible until you test a wide
lens.

### 1.2 Pitfall: column-major constructors transpose your Jacobian

GLSL (and GLM) `mat3(a,b,c, d,e,f, g,h,i)` fills **columns**, not rows.
The reference implementations write:

```glsl
mat3 J = mat3(f/z,      0.0,      -f·x/z²,     // column 0
              0.0,      f/z,      -f·y/z²,     // column 1
              0.0,      0.0,       0.0);       // column 2
```

so the tilt terms land in the constructor's *first two triplets* — the math
matrix's **bottom row**. They then compute `cov = Tᵀ·Vrk·T` with `T = W·J`,
and the combination is exactly the `J·Σview·Jᵀ` above.

If you instead copy the textbook row-major matrix into the constructor
(tilt terms in the third triplet), you get `Jᵀ`, and in `Tᵀ·Vrk·T` the tilt
terms **cancel out of the visible 2×2 block entirely** — no compiler error,
no NaN, screen centers pixel-identical. The only symptom: grazing-angle
surfaces near the edges of wide lenses render bloated and straw-like.

**Regression test that catches it** (§5.3): a needle elongated along the
view axis must smear radially at the screen edge and stay a dot at center.

### 1.3 Pitfall: the pixel → NDC factor of 2

NDC spans **2 units** across the viewport. Converting a pixel-space
half-extent to an NDC offset is `px * 2 / viewport`, not `px / viewport`.
Omitting the 2 draws every splat at **half** its true screen size.

This bug hides brutally well: dense scans still look plausible (small
splats + a minimum-kernel floor fill the surface), and users quietly
compensate by doubling the global size multiplier. The giveaways:

- sparse/grazing surfaces (walls seen at an angle, screen edges) look
  thin, fuzzy, straw-like;
- everyone runs your renderer with "splat size ≈ 2".

Beware when reading other implementations: some engines define
`focal = proj00 · width` (2× the pixel focal) and compensate with a
`w/viewport` conversion — their size constants (dilation, clamps) are then
in *different units* than yours. Derive one end-to-end example numerically
before copying any constant.

### 1.4 Kernel conditioning

Standard tricks, in pixel² units (verify your units first — see 1.3):

- **Dilation**: add `0.3` to both diagonal entries of `cov2D`. Prevents
  sub-pixel splats from undersampling into holes.
- **Minimum eigenvalue**: `λ2 = max(mid - radius, 0.1)`. Keeps razor-thin
  splats at least ~1px wide — this is what makes surfaces read as
  *continuous* at native size.
- **Energy-conserving antialiasing** (optional, "Mip-Splatting" style):
  `alpha *= sqrt(det(cov2D) / det(cov2D + 0.3·I))`. Physically correct —
  small splats get dimmer instead of over-bright — but surfaces become more
  translucent. Reference viewers commonly ship with this OFF; make it a
  toggle.
- **Jacobian input clamp** (INRIA): clamp `x/z, y/z` to `±1.3·tan(fov/2)`
  before building `J`. Guards splats outside the frustum from producing
  absurd covariances.

### 1.5 The falloff must reach zero

Evaluating the raw gaussian and discarding at the quad edge leaves a visible
step (the edge sits at `exp(-4) ≈ 0.018`, clearly visible on large splats).
Normalize the falloff so it hits zero exactly at the kernel boundary:

```glsl
alpha = (exp(-r²/2σ²) - exp(-4.0)) / (1.0 - exp(-4.0))
```

This is the difference between "soft airbrushed ellipses" and "hard-edged
blobs", most visible on close-ups and wide lenses.

### 1.6 Culling: cull by ellipse extent, clamp depth, add a near-cull

- **Frustum cull by extent, not center**: a splat whose *center* is
  off-screen can still cover visible pixels (large splats, wide lenses).
  Cull only when `|ndc| - extent > 1`. Center-margin culling makes edge
  splats pop in and out during camera pans.
- **Don't clip at near/far — clamp**: `clip.z = clamp(clip.z, -|w|, |w|)`
  so walking the camera through a scan doesn't slice splats at the near
  plane.
- **Do add a user-facing near-cull distance** (hide splats with
  `view.z > -d`): real scans carry "floater" gaussians along the capture
  path. Web viewers hide them implicitly behind a relatively large camera
  near plane; a viewport with `clip start = 0.01` shows them as giant
  blurry sheets. A default of ~0.1 m with a slider works well.

---

## 2. Sorting

Alpha blending needs back-to-front ordering, approximated by sorting splat
centers by view-space depth (`z = dot(view_row_z, position)`).

**Key theorem worth exploiting**: the *order* of `dot(d, pᵢ) + c` is
invariant to the constant `c` and to positive scaling of `d`. Therefore the
sort order depends **only on the view direction**, never on camera position
or zoom. Consequences:

- panning, dollying and zooming never require a re-sort;
- re-sort only when the view direction rotates past a threshold (~1°);
- the sort key is `positions @ direction` — a single matrix-vector product.

Run the argsort on a worker thread (NumPy releases the GIL) and have the
render loop pick up finished results; never block the frame on a sort.
Upload the order as an index texture and draw quads in fixed order — then a
re-sort is a small texture upload, not a geometry rebuild.

---

## 3. Color

- **Spherical harmonics**: evaluate in the splat's object space with
  `dir = normalize(splat_pos - camera_pos_object_space)`; coefficients are
  stored channel-major (`R₀..R_C, G₀..G_C, B₀..B_C`) in the standard `.ply`
  layout. Quantizing coefficients to uint8 over `[-4, 4]` (the compressed-PLY
  convention) is visually lossless and cuts memory 4×.
- **Do color-space conversion AFTER the SH sum**: SH deltas live in the same
  space as the DC color. Convert sRGB→linear (or whatever your target needs)
  as the final step in the shader, not in the loader, or view-dependent
  colors will be wrong.
- **Blending space matters**: web viewers blend in the sRGB-encoded
  framebuffer; a linear-workflow viewport blends in linear. Colors match if
  you linearize correctly, but soft edge *transitions* will always differ
  slightly. Don't chase pixel-identical blends across the two.

---

## 4. Picking without geometry

Splats have no mesh to raycast. Two complementary modes:

- **Surface pick**: render splats to a small offscreen with an opaque alpha
  threshold (~0.2), depth writes ON, and the fragment outputting
  `gl_FragCoord.z` into a float channel. Read the pixel under the cursor and
  unproject through the inverse view-projection. Gives true points on the
  *rendered* surface. Re-render only when the view changes.
- **Center snap**: project all (or a subsample of) splat centers with one
  NumPy matmul and pick the front-most center within a screen radius.
  Deterministic, great for corner-to-corner measuring.

---

## 5. Testing a renderer against closed-form math

Screenshots lie; write tests that compare against formulas. Three that
caught real bugs here:

### 5.1 Analytic gaussian profile

Render ONE isotropic splat of known world σ at known depth with a known
projection; `σ_px = f·σ_world/|z|` is exact. Sample the rendered alpha at
`r = σ_px` and `r = 2σ_px` and compare with the closed-form (normalized)
falloff. Catches: size miscalibration (the ×2 bug), falloff-shape errors,
unit mistakes. Tolerances of ±0.06–0.08 absorb dilation and pixel sampling.

### 5.2 Sub-pixel continuity

A wall of sub-pixel splats at fixed density; measure the fraction of
interior pixels darker than a threshold ("hole fraction"). Catches missing
dilation / eigenvalue floors. A healthy renderer is ≈ 0%.

### 5.3 Perspective tilt presence

A needle scaled along the view axis (`scale = (ε, ε, L)`), rendered at
screen center and near the screen edge of a ~90° lens. Center must be a dot;
the edge must smear radially (the `-f·x/z²` terms project the needle's depth
extent into screen space). If edge ≈ center, the tilt terms are missing —
the transposed-Jacobian bug (§1.2). Also assert an *isotropic* splat's
edge/center footprint ratio is mildly > 1 (foreshortening) but bounded.

---

## 6. Blender `gpu` API notes (host-specific)

- `GPUTexture` accepts only `Buffer('FLOAT')` data. Store integer data
  (sort indices) in `R32F` (exact to 2²⁴) or bit-pack bytes into `RGBA32F`
  and decode with `floatBitsToUint`.
- The `gpu` module is unavailable in `blender -b`; create GPU resources
  lazily inside draw handlers, and run GPU tests in foreground mode.
- **Hold a Python reference to `GPUUniformBuf` until after `batch.draw()`**
  — passing a temporary straight into `uniform_block()` lets Python free it
  before the draw, which then reads garbage and renders nothing.
- `image.pixels` rows are bottom-up; flip before treating them as raster
  data. Set `colorspace = 'Non-Color'` and `alpha_mode = 'CHANNEL_PACKED'`
  for data textures.
- Use `gpu.state.viewport_get()` (device pixels) for pixel-space shader
  constants; `region.width` is in UI points and differs under display
  scaling.
- Modal operators that add draw handlers MUST implement `cancel()` —
  file-open/undo/quit terminate modals without any event, leaking the
  handler otherwise.

---

*From the [POBIM Splats](https://github.com/POBIM/pobim-splats) project
(GPL-3.0). Corrections welcome — file an issue.*

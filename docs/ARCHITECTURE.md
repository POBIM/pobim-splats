# สถาปัตยกรรม POBIM Splats

## เป้าหมายการออกแบบ

แสดง 3DGS ใน Blender viewport ให้ "สวยเท่า web viewer และเบา" — ปัญหาของ addon
รุ่นเก่าคือใช้ Geometry Nodes สร้าง instance ต่อ splat ทำให้ Blender ต้องจัดการ
object นับล้าน ที่นี่ใช้แนวทางเดียวกับ SuperSplat/PlayCanvas แทน:
ข้อมูลทั้งหมดอยู่ใน texture, วาดครั้งเดียวต่อ splat cloud ด้วย shader เดียว

## Data flow

```
.ply ── ply_loader.py ──▶ SplatCloud (numpy)
                             │ positions (N,3)  ← เก็บไว้ใช้ sort
                             │ cov6 (N,6)       ← R S Sᵀ Rᵀ precomputed
                             │ colors, opacities
                             ▼
                          SplatGPU (สร้าง lazy ใน draw handler ตัวแรก
                             │        เพื่อการันตีว่ามี GPU context)
                             │
        data texture RGBA32F (3 texels/splat):
          t0 = center.xyz, opacity
          t1 = cov.xx,xy,xz, packed RGB (uint bits ใน float)
          t2 = cov.yy,yz,zz, (ว่าง)
        order texture R32F (1 texel/splat) = ลำดับวาด back-to-front
        vertex buffer static: quadId + cornerId, 6 จุดยอด/splat
```

## Render loop (`_draw_callback`, POST_VIEW)

1. รวบรวม splat objects ที่มองเห็น เรียง object ไกล→ใกล้
2. ต่อ object: `sort_if_needed(modelView)` — ถ้ากล้อง/objectขยับ และพ้น throttle
   interval ให้ argsort ความลึก view-space ด้วย numpy แล้วอัปโหลด order texture ใหม่
   (4MB/ล้าน splat)
3. อัด uniform ผ่าน UBO (`GPUUniformBuf`): modelView, projection, viewport, ปรับแต่ง
4. `blend=ALPHA, depth_test=LESS_EQUAL, depth_mask=off` — splat โดนวัตถุ mesh
   ในฉากบังถูกต้อง และไม่เขียน depth ทับกันเอง

## Vertex shader (EWA splatting)

ต่อจุดยอด: อ่าน splat index จาก order texture → อ่านข้อมูลจาก data texture →
แปลง center เป็น view space → คำนวณ Jacobian ของ projection (มี branch แยก
orthographic) → `cov2D = J·A·Σ·Aᵀ·Jᵀ` (A = mat3(modelView) รวม scale ของ object แล้ว)
→ หา eigenvalues ได้แกน major/minor ของวงรีบนจอ → ขยับมุม quad ตาม ±2σ
สี unpack จาก uint bits, fragment shader ตัดที่ `exp(-4)` (รัศมี 2σ)

ค่า +0.3px dilation ตาม reference rasterizer ของ INRIA กันวงรีเล็กกว่า pixel แตก

## บทเรียนจากการทดสอบจริง (Blender 4.5.2)

- `GPUTexture` จาก Python รับเฉพาะ `Buffer('FLOAT')` — order texture จึงเป็น
  R32F ไม่ใช่ R32I (float32 เก็บ int แม่นถึง 2²⁴ = 16.7M เกินเพดาน 11.1M ที่
  ผูกกับ texture height 16384)
- `gpu` module ใช้ไม่ได้ใน background mode (`blender -b`) — จึงต้องแยกเทสต์
  GPU เป็น foreground script และ addon สร้าง GPU resources แบบ lazy ใน draw
  handler เท่านั้น
- `GPUShaderCreateInfo` + `typedef_source` + `uniform_buf` ใช้ส่ง struct UBO
  ได้จริง (เลี่ยงข้อจำกัด push constant 128 bytes)
- numpy array ส่งเข้า `Buffer`/`attr_fill` ผ่าน buffer protocol ได้ มี fallback
  `.tolist()` กันไว้

## Roadmap

1. **SH bands 1–3** — view-dependent color: เพิ่ม texel ต่อ splat + ประเมิน SH
   ใน vertex shader ตามทิศกล้อง (ทำได้ทันทีเพราะ loader อ่าน f_rest ได้ ถ้าเพิ่ม)
2. **Threaded sort** — ย้าย argsort ไป thread + double-buffer order texture
   เพื่อฆ่าอาการกระตุกในฉาน >3M splats
3. **F12 render** — ทางเลือก:
   - `bpy.types.RenderEngine` custom engine ที่ rasterize splat เข้า Render Result
   - หรือ hook `render_post` composite ภาพจาก offscreen render (มีโค้ดอยู่แล้วใน
     `tests/render_preview_blender.py` เป็นต้นแบบ)
4. **Edit sync กับ POBIMStudio** — export ply จาก POBIMStudio → auto-reload
   (มี Reload อยู่แล้ว; เพิ่ม file-watch timer ได้)
5. **Frustum culling / LOD** — ข้าม splat นอกจอใน shader แล้วอยู่, เพิ่ม
   subsample ตามระยะสำหรับฉากใหญ่

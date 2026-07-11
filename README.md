# POBIM Splats — 3D Gaussian Splatting Viewer for Blender

แสดงผลไฟล์ 3D Gaussian Splatting (`.ply`) ใน Blender viewport ด้วย splat renderer จริง
(EWA splatting + depth sorting) — สถาปัตยกรรมเดียวกับ web viewer อย่าง SuperSplat
ไม่ใช่การสร้าง object ต่อ splat จึงเบากว่า addon แบบ Geometry Nodes มาก

ทดสอบแล้วกับ **Blender 4.5 LTS** (ต้องเป็น 4.2 ขึ้นไป)

![preview](tests/preview.png)

## ติดตั้ง

1. สร้างไฟล์ zip ของ addon:
   ```bash
   ./scripts/build_zip.sh        # ได้ pobim_splats.zip
   ```
2. ใน Blender: `Edit > Preferences > Add-ons > (ลูกศรมุมขวาบน) > Install from Disk…`
   เลือก `pobim_splats.zip` แล้วติ๊กเปิดใช้งาน
3. แผงควบคุมอยู่ที่ **View3D > กด N > แท็บ "3DGS"**

## ใช้งาน

1. กด **Import Splat** เลือกไฟล์ได้ 4 แบบ:
   - `.ply` — 3DGS มาตรฐาน
   - `.compressed.ply` — SuperSplat compressed (decode ในตัว ไม่ต้องแปลงก่อน)
   - `.sog` — SOG v2 bundle (zip)
   - `meta.json` — SOG v2 แบบแยกไฟล์ (webp อยู่โฟลเดอร์เดียวกัน)
   - **Max Splats** — จำกัดจำนวน splat สำหรับฉากใหญ่มาก (0 = ทั้งหมด, เพดาน ~11.1 ล้าน)
   - **Rotate to Z-up** — ไฟล์สแกนส่วนใหญ่เป็นแกน Y-down จะหมุนให้ตั้งตรงอัตโนมัติ
2. จะได้ Empty หนึ่งตัวแทน splat — ย้าย/หมุน/สเกลได้ตามปกติ splat จะตามไปด้วย
3. ปรับ **Splat Size** และ **Opacity** ได้จากแผงด้านข้าง
4. เซฟไฟล์ .blend ได้ตามปกติ — เปิดกลับมา addon จะโหลด splat จากไฟล์เดิมให้อัตโนมัติ

### Measure & Scale (ปรับสเกลให้ตรงหน้างานจริง)

สแกน 3DGS มักมีสเกลไม่ตรงหน่วยจริง เครื่องมือนี้แก้ให้:

1. กดปุ่ม **Measure & Scale** ในกล่องของ splat นั้น
2. คลิกซ้ายจุดแรก → คลิกซ้ายจุดที่สอง (เคอร์เซอร์จะ snap เข้าหา splat ใกล้สุด
   มีวงแหวนแสดง และมีเส้น + ระยะโชว์ระหว่างลาก; คลิกขวา/Esc = ยกเลิก)
3. ใส่ **ระยะจริง** ในหน้าต่างที่เด้งขึ้น → กด OK
4. Splat จะถูกสเกลรอบจุดแรกที่คลิก (จุดแรกอยู่กับที่) — Ctrl+Z ย้อนได้

### ประสิทธิภาพ

การเรียงลำดับความลึก (depth sort) รันใน**เธรดแยก** ไม่บล็อก viewport และเรียงเฉพาะเมื่อ
ทิศกล้องหมุนเกิน ~1° — การแพน/ซูมไม่ต้องเรียงใหม่เลย ฉากระดับหลายล้าน splat
จึงหมุนได้ลื่นโดยลำดับการ blend ตามทันภายในเสี้ยววินาที (ปรับ **Sort Interval** ได้)

### สีให้ตรงกับ web viewer

Blender เริ่มต้นใช้ view transform แบบ AgX/Filmic ซึ่งจะทำให้สีจางลง
ให้ตั้ง `Render Properties > Color Management > View Transform = Standard`
(addon แปลงสีเป็น linear ให้แล้วตอน import)

### ไฟล์ที่รองรับ

| ฟอร์แมต | สถานะ |
|---|---|
| `.ply` (INRIA 3DGS, binary) | ✓ |
| `.compressed.ply` (SuperSplat) | ✓ decode ในตัว |
| `.sog` / `meta.json` (SOG v2) | ✓ decode ในตัว (ใช้ WebP codec ของ Blender) |
| `.splat`, `.spz`, `.ksplat`, SOG v1 | ✗ แปลงก่อนด้วย [splat-transform](https://github.com/playcanvas/splat-transform): `npx @playcanvas/splat-transform input output.ply` |

## ข้อจำกัดปัจจุบัน

- แสดงเฉพาะสี SH band 0 (ยังไม่มี view-dependent color; ข้อมูล shN ใน SOG ถูกข้าม)
- เห็นเฉพาะใน **viewport** — การกด F12 render ด้วย EEVEE/Cycles จะยังไม่เห็น splat
  (ใช้ Viewport Render Image/Animation ได้ หรือดู roadmap ใน docs/ARCHITECTURE.md)
- กล้อง orthographic รองรับแบบประมาณ (Jacobian คงที่)
- Measure & Scale จับจุดที่ศูนย์กลาง gaussian (สุ่มตัวอย่างสูงสุด 400k จุดตอน pick)

## ทดสอบ

```bash
python3 tests/test_ply_loader.py     # loader + compressed roundtrip (ไม่ต้องมี Blender)
python3 tests/test_measure_math.py   # คณิต pick/scale (ไม่ต้องมี Blender)
blender -b --factory-startup --python tests/smoke_test_blender.py # operators + registry
blender --factory-startup --python tests/gpu_test_blender.py     # shader + GPU (ต้องมีจอ)
blender --factory-startup --python tests/render_preview_blender.py  # ออกภาพ tests/preview.png
python3 tests/make_test_ply.py torus.ply 500000                   # สร้างไฟล์ทดสอบ
python3 tests/make_test_ply.py torus.compressed.ply 500000        # แบบ compressed
```

## โครงสร้าง

- `pobim_splats/ply_loader.py` — อ่าน PLY มาตรฐาน + decode compressed.ply (bpy-free)
- `pobim_splats/sog_format.py` — คณิต decode SOG v2 (bpy-free)
- `pobim_splats/sog_loader.py` — โหลด .sog/meta.json + decode webp ผ่าน Blender
- `pobim_splats/splat_gpu.py` — shader, GPU resources, draw handler, threaded depth sort
- `pobim_splats/measure_math.py` — คณิต pick/scale (bpy-free)
- `pobim_splats/measure.py` — modal operator Measure & Scale + overlay
- `pobim_splats/operators.py` — import / reload / remove
- `pobim_splats/ui.py` — แผง N-panel
- `docs/ARCHITECTURE.md` — สถาปัตยกรรมโดยละเอียด
- `docs/ROADMAP.md` — แผนสู่เวอร์ชันขายจริง + เรื่อง license/ช่องทางขาย

## License

GNU GPL v3 (ดู `LICENSE`) — เป็นข้อกำหนดของ Blender สำหรับ addon ที่ใช้ `bpy`
การขายเชิงพาณิชย์ทำได้ตามปกติ (โมเดลเดียวกับ Blender Market) —
รายละเอียดใน `docs/ROADMAP.md` ส่วน "เรื่อง License" และดู `THIRD_PARTY.md`
สำหรับเครดิตสเปค/อัลกอริทึมที่อ้างอิง

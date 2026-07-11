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

1. กด **Import 3DGS PLY** เลือกไฟล์ `.ply` (ฟอร์แมต 3DGS มาตรฐาน)
   - **Max Splats** — จำกัดจำนวน splat สำหรับฉากใหญ่มาก (0 = ทั้งหมด, เพดาน ~11.1 ล้าน)
   - **Rotate to Z-up** — ไฟล์สแกนส่วนใหญ่เป็นแกน Y-down จะหมุนให้ตั้งตรงอัตโนมัติ
2. จะได้ Empty หนึ่งตัวแทน splat — ย้าย/หมุน/สเกลได้ตามปกติ splat จะตามไปด้วย
3. ปรับ **Splat Size** และ **Opacity** ได้จากแผงด้านข้าง
4. เซฟไฟล์ .blend ได้ตามปกติ — เปิดกลับมา addon จะโหลด splat จากไฟล์ .ply เดิมให้อัตโนมัติ

### สีให้ตรงกับ web viewer

Blender เริ่มต้นใช้ view transform แบบ AgX/Filmic ซึ่งจะทำให้สีจางลง
ให้ตั้ง `Render Properties > Color Management > View Transform = Standard`
(addon แปลงสีเป็น linear ให้แล้วตอน import)

### ไฟล์ที่รองรับ

รองรับเฉพาะ `.ply` แบบ binary (INRIA 3DGS layout) — ถ้ามี `.compressed.ply`, `.sog`,
`.spz`, `.splat` ให้แปลงก่อนด้วย [splat-transform](https://github.com/playcanvas/splat-transform):

```bash
npx @playcanvas/splat-transform input.compressed.ply output.ply
```

## ข้อจำกัดของ MVP นี้

- แสดงเฉพาะสี SH band 0 (ยังไม่มี view-dependent color)
- เห็นเฉพาะใน **viewport** — การกด F12 render ด้วย EEVEE/Cycles จะยังไม่เห็น splat
  (ใช้ Viewport Render Image/Animation ได้ หรือดู roadmap ใน docs/ARCHITECTURE.md)
- การเรียงลำดับความลึก (depth sort) ทำงานเป็นช่วงๆ ตามค่า Sort Interval —
  ฉากหลายล้าน splat อาจกระตุกสั้นๆ ระหว่างหมุนกล้อง ปรับ interval เพิ่มได้
- กล้อง orthographic รองรับแบบประมาณ (Jacobian คงที่)

## ทดสอบ

```bash
python3 tests/test_ply_loader.py                                  # loader (ไม่ต้องมี Blender)
blender -b --factory-startup --python tests/smoke_test_blender.py # operators + registry
blender --factory-startup --python tests/gpu_test_blender.py     # shader + GPU (ต้องมีจอ)
blender --factory-startup --python tests/render_preview_blender.py  # ออกภาพ tests/preview.png
python3 tests/make_test_ply.py torus.ply 500000                   # สร้างไฟล์ทดสอบ
```

## โครงสร้าง

- `pobim_splats/ply_loader.py` — อ่าน PLY + คำนวณ covariance (bpy-free, เทสต์นอก Blender ได้)
- `pobim_splats/splat_gpu.py` — shader, GPU resources, draw handler, depth sorting
- `pobim_splats/operators.py` — import / reload / remove
- `pobim_splats/ui.py` — แผง N-panel
- `docs/ARCHITECTURE.md` — สถาปัตยกรรมโดยละเอียด + roadmap

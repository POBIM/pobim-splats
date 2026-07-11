# Roadmap สู่เวอร์ชันขายจริง ($9 บน X)

สถานะปัจจุบัน: v0.2 — viewport renderer (ply/compressed.ply/sog), threaded sort,
Measure & Scale ผ่านการทดสอบบนงานสแกนจริงแล้ว

หลักคิด: คนจ่าย $9 เพราะ "เอา splat เข้า Blender แล้ว**ใช้งานได้จบ**" — ดูสวย
ตัดส่วนเกิน วัดสเกล และ**เรนเดอร์ออกมาได้** ฟีเจอร์เรียงตามผลต่อการตัดสินใจซื้อ

## v0.3 — ฟีเจอร์ปิดการขาย

### P0.1 Crop Box (ความคุ้ม effort สูงสุด — ทำก่อน)
กล่องตัด splat: แสดงเฉพาะภายใน/ภายนอกกล่อง
- เทคนิค: เพิ่ม box inverse-matrix เข้า UBO แล้ว `discard` ใน shader — งานเล็ก
- UI: ปุ่ม "Add Crop Box" สร้าง Empty (Cube) ผูกกับ splat, ย้าย/หมุน/สเกลได้
- ตอบโจทย์งานจริงที่สุด: สแกนทุกไฟล์มีขอบรุงรัง ผู้ใช้ตัดใน Blender ได้เลย

### P0.2 F12 Render (จุดขายหลัก)
เฟส A — **splat-only RenderEngine**: custom `bpy.types.RenderEngine`
rasterize splat ด้วยกล้อง scene ที่ความละเอียด render จริง → Render Result
รองรับ animation (กล้อง fly-through) + transparent background
(โค้ด offscreen มีแล้วใน `tests/render_preview_blender.py`)
เฟส B (หลังขาย) — composite กับ mesh ผ่าน Z-depth ใน compositor

### P0.3 Drag & Drop + ความลื่นตอน import
- Blender 4.1+ มี `bpy.types.FileHandler` รองรับลากไฟล์ลง viewport
- progress cursor ระหว่างโหลดไฟล์ใหญ่ + error popup ที่อ่านรู้เรื่อง

## v0.4 — คุณภาพภาพ
- **SH band 1–3**: view-dependent color (SOG มีข้อมูล shN แล้ว) — เพิ่ม texel
  ต่อ splat + ประเมิน SH ใน vertex shader; มี toggle ปิดเพื่อประหยัดแรม
- ตัวเลือก exposure/สี ให้ match กับ POBIMStudio และ web viewer

## v1.0 — แพ็กเกจขาย
- แปลงเป็น **Blender Extension** format (`blender_manifest.toml`) — ติดตั้งง่าย
  รองรับ 4.2 LTS+ อย่างเป็นทางการ
- เอกสาร/README ภาษาอังกฤษ + วิดีโอเดโม 60 วิ สำหรับโพสต์ X
- หน้า Gumroad: ชื่อ "POBIM Splats — Gaussian Splats in Blender",
  ราคา $9 ขายขาด, ส่ง zip อัตโนมัติ, changelog ต่อเวอร์ชัน

## เรื่อง License (สำคัญ — อ่านก่อนขาย)

**ข้อเท็จจริง**: addon ที่ import `bpy` ถือเป็น derivative work ของ Blender
ตามการตีความของ Blender Foundation → ต้องเผยแพร่เป็น **GPL-3.0**
(repo นี้ใส่ `LICENSE` + `THIRD_PARTY.md` แล้ว)

**ขายได้ปกติ** — GPL ไม่ห้ามขาย ตลาด Blender Market ทั้งตลาดขาย GPL addon
แบบเดียวกันนี้ สิ่งที่ลูกค้าจ่ายคือ: ไฟล์พร้อมใช้ + อัปเดต + ซัพพอร์ต
สิ่งที่ต้องรู้: ผู้ซื้อได้สิทธิ์ GPL เต็ม (แจกจ่ายต่อได้ถูกกฎหมาย) — เป็นความเสี่ยง
มาตรฐานของตลาดนี้ และราคา $9 ต่ำพอที่คนเลือกจ่ายมากกว่าหาโหลดเถื่อน

**ช่องทางขายจาก X**: โพสต์เดโม + ลิงก์ Gumroad (แนะนำ — ตั้งราคา $9,
จัดการจ่ายเงิน/VAT/ส่งไฟล์อัตโนมัติ, มี license key ให้ใช้ได้ถ้าต้องการ)
ทางเลือก: Lemon Squeezy, Ko-fi, หรือ Blender Market (ค่าธรรมเนียมสูงกว่า
แต่มีฐานลูกค้า Blender ตรงกลุ่ม)

## แผนที่ยังไม่ทำ (v1.x)
- Mixed-scene F12 composite (Z-depth กับ EEVEE/Cycles)
- LOD / streaming สำหรับฉาก 20M+
- Sync อัตโนมัติกับ POBIMStudio (file watch)
- Measure หลายช่วง + snap แบบ median

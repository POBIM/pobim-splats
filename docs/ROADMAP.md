# Roadmap (โมเดล: โอเพนซอร์สฟรี + Donate)

สถานะปัจจุบัน: v0.2 — viewport renderer (ply/compressed.ply/sog), threaded sort,
Measure & Scale ผ่านการทดสอบบนงานสแกนจริงแล้ว

โมเดลการแจกจ่าย: **แจกฟรีบน GitHub (GPL-3.0) + รับสนับสนุนผ่าน GitHub
Sponsors / Ko-fi** — โปรโมทผ่าน X ด้วยวิดีโอเดโม ลิงก์ไป repo
ยิ่งคนใช้เยอะยิ่งได้ดาว/แชร์/donate และเป็นเครดิตให้แบรนด์ POBIM ด้วย

หลักคิดเดิมยังใช้ได้: คนจะแชร์/สปอนเซอร์เมื่อ "เอา splat เข้า Blender แล้ว
**ใช้งานได้จบ**" — ดูสวย ตัดส่วนเกิน วัดสเกล และ**เรนเดอร์ออกมาได้**

## v0.3 — ฟีเจอร์เรือธง

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

## v1.0 — เผยแพร่สาธารณะ
- สร้าง GitHub repo สาธารณะ + เปิด GitHub Sponsors / Ko-fi
  (แก้ `.github/FUNDING.yml` ให้เป็น username จริง → ปุ่ม Sponsor ขึ้นเอง)
- แปลงเป็น **Blender Extension** format (`blender_manifest.toml`) แล้วส่งขึ้น
  **extensions.blender.org** — ช่องทางแจกทางการของ Blender คนค้นเจอเอง
  (รับเฉพาะ addon โอเพนซอร์ส — ตรงกับโมเดลเราพอดี)
- README อังกฤษ (มีแล้วส่วนต้น) + GIF/วิดีโอเดโม 60 วิ + Releases พร้อม zip
- โพสต์เปิดตัวบน X: วิดีโอสแกนจริง + ลิงก์ repo

## โมเดล License / รายได้ (สรุป)

- addon ที่ import `bpy` ต้องเป็น **GPL-3.0** ตามข้อกำหนด Blender →
  โอเพนซอร์สคือทางที่เข้ากับ license อยู่แล้วโดยธรรมชาติ ไม่มีแรงเสียดทาน
- **GitHub Sponsors รองรับประเทศไทย** (payout ผ่าน Stripe ผูกบัญชีธนาคาร
  ตอนสมัคร) — มีนักพัฒนาไทยใช้จริง เช่น ผู้ดูแล PyThaiNLP
- **Ko-fi** เป็นทางสำรองที่สมัครง่ายกว่า (payout ผ่าน PayPal)
- ประโยชน์ทางอ้อม: ดาว/ผู้ใช้/เครดิตแบรนด์ POBIM + ฟีดแบ็กและ bug report ฟรี

## แผนที่ยังไม่ทำ (v1.x)
- Mixed-scene F12 composite (Z-depth กับ EEVEE/Cycles)
- LOD / streaming สำหรับฉาก 20M+
- Sync อัตโนมัติกับ POBIMStudio (file watch)
- Measure หลายช่วง + snap แบบ median

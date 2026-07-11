# Third-Party Notices

POBIM Splats ไม่ได้ฝังโค้ดจากโปรเจกต์อื่นโดยตรง แต่พัฒนาโดยอ้างอิง
สเปคฟอร์แมตและอัลกอริทึมสาธารณะต่อไปนี้:

- **3D Gaussian Splatting** (INRIA / GRAPHDECO) — อัลกอริทึม EWA splatting
  และโครงสร้างข้อมูล 3DGS PLY
  https://github.com/graphdeco-inria/gaussian-splatting
- **PlayCanvas splat-transform** (MIT License) — สเปคฟอร์แมต
  `.compressed.ply` และ SOG v2 (ศึกษาจากซอร์สเพื่อเขียน decoder ใหม่ด้วย numpy)
  https://github.com/playcanvas/splat-transform
- **antimatter15/splat** (MIT License) — แนวทาง shader โปรเจกต์ covariance
  เป็นวงรี 2D บนจอ (เขียนใหม่เป็น GLSL สำหรับ Blender GPU module)
  https://github.com/antimatter15/splat

ตัว addon ทำงานร่วมกับ Blender ผ่าน Python API (`bpy`) จึงเผยแพร่ภายใต้
GNU GPL v3 ตามข้อกำหนด license ของ Blender — ดูไฟล์ `LICENSE`

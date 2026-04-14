Dưới đây là bảng đối chiếu (mapping) giữa các ký hiệu điểm Landmark trong file JSON của bạn và các ký hiệu tương ứng được sử dụng trong luận án của TS. Hoàng Thị Đợi để tính toán các chỉ số hài hòa người Việt.

### **1\. Bảng chuyển đổi các điểm Landmark**

| Ký hiệu JSON | Tên đầy đủ (JSON) | Ký hiệu trong Paper | Tên tiếng Việt trong Paper | Chỉ số liên quan |
| :---- | :---- | :---- | :---- | :---- |
| **S** | Sella | **S** | Điểm S (Điểm hố yên) | SNA, SNB |
| **N** | Nasion | **N** | Điểm N (Điểm gốc mũi) | SNA, SNB, ANB, U1-NA, L1-NB |
| **A** | A-point | **A** | Điểm A | SNA, ANB, U1-NA |
| **B** | B-point | **B** | Điểm B | SNB, ANB, L1-NB |
| **Pn** | Pronasale | **Pn** | Đỉnh mũi | Đường E, Đường S, Góc mũi mặt |
| **Sn** | Subnasale | **Sn** | Điểm dưới mũi | Góc mũi môi, Góc lồi mặt |
| **Ls** | Labrale superius | **ls** | Môi trên | Ls-E, Ls-S, Góc mũi môi |
| **Li** | Labrale inferius | **li** | Môi dưới | Li-E, Li-S, Góc Z |
| **Pog\`** | Soft Tissue Pogonion | **Pg'** | Cằm mềm | Đường E, Đường S, Góc Z, Góc lồi mặt |
| **UIT** / **UIA** | Upper Incisor Tip/Apex | **I** | Răng cửa trên | I-NA (U1-NA), I/Pal |
| **LIT** / **LIA** | Lower Incisor Tip/Apex | **i** | Răng cửa dưới | i-NB (L1-NB), i/MP (IMPA) |
| **Po** | Porion | **Po** | Điểm Porion | Mặt phẳng FH (FH Plane) |
| **Or** | Orbitale | **Or** | Điểm Orbitale | Mặt phẳng FH |
| **Go** | Gonion | **go** | Điểm góc hàm | Mặt phẳng hàm dưới (MP) |
| **Me** / **Gn** | Menton / Gnathion | **Me / Gn** | Điểm cằm / Gnathion | Mặt phẳng hàm dưới (MP) |

### ---

**2\. Công thức tổ hợp điểm để tính chỉ số (Mapping Logic)**

Dựa trên bảng trên, đây là cách bạn lập trình để tính các chỉ số hài hòa:

| Chỉ số hài hòa | Các điểm cần dùng từ JSON | Logic tính toán |
| :---- | :---- | :---- |
| **SNA** | S, N, A | Góc giữa vector NS và NA |
| **SNB** | S, N, B | Góc giữa vector NS và NB |
| **ANB** | A, N, B | Góc giữa vector NA và NB (hoặc SNA \- SNB) |
| **U1-NA (mm)** | UIT, N, A | Khoảng cách từ UIT đến đường thẳng đi qua N và A |
| **L1-NB (mm)** | LIT, N, B | Khoảng cách từ LIT đến đường thẳng đi qua N và B |
| **IMPA (i/MP)** | LIT, LIA, Go, Me | Góc giữa trục răng cửa dưới (LIT-LIA) và mặt phẳng hàm dưới (Go-Me) |
| **Ls-E (mm)** | Ls, Pn, Pog\` | Khoảng cách từ Ls đến đường thẩm mỹ Ricketts (Pn-Pog\`) |
| **Li-E (mm)** | Li, Pn, Pog\` | Khoảng cách từ Li đến đường thẩm mỹ Ricketts (Pn-Pog\`) |
| **Góc Z** | Li, Pog\`, Po, Or | Góc giữa đường Profile (Li-Pog\`) và mặt phẳng ngang FH (Po-Or) |

### ---

**⚠️ Lưu ý quan trọng về các điểm thiếu:**

1. **Góc mũi môi (Nasolabial Angle):** Luận án tính bằng góc **Cm-Sn-ls**.  
   * Trong JSON bạn đã có **Sn** và **Ls**.  
   * **Thiếu điểm Cm (Columella):** Bạn cần bổ sung điểm này (nằm ở phần thấp nhất của vách ngăn mũi) để tính được góc này một cách chuẩn xác.  
2. **Đường thẩm mỹ S (Steiner):** Luận án cũng sử dụng đường S (nối từ trung điểm cánh mũi đến cằm mềm).  
   * File JSON hiện tại thiếu điểm **trung điểm cánh mũi**. Nếu bạn muốn tính thêm các chỉ số phụ như Ls-S hay Li-S trong paper, bạn cần bổ sung điểm này.

**Lời khuyên:** Với bộ Landmark hiện có trong JSON, bạn đã có thể tính được **ANB, SNA, SNB, IMPA, Ls-E, Li-E và Góc Z**. Đây là những chỉ số "xương sống" để đánh giá một khuôn mặt Việt có hài hòa hay không.  

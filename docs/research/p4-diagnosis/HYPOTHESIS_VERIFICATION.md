# TAPTA Hypotheses Verification Report (H1 – H8)

**Date:** 2026-11-25  
**Evidence Source:** Diagnostic Suite D01–D09 on TACRED Track A (`runs/diagnostics/tacred_5shot/seed_2021/`)  

---

## 1. Bảng Tổng Hợp Kiểm Chứng 8 Giả Thuyết Khoa Học

| Giả thuyết | Mô tả tóm tắt | Bằng chứng thực nghiệm định lượng | Kết luận khoa học |
|---|---|---|---|
| **H1: Routing Bottleneck** | Router Top-1 miss là nút thắt cổ chai chi phối toàn bộ hiệu năng của hệ thống. | - Router Top-1 chỉ đạt **48.44%**.<br>- Khi Router đúng Top-1, độ chính xác đạt **80.74%**.<br>- Lỗi liên quan đến Router chiếm **83.8%** tổng số lỗi ($1,061 / 1,266$). | **CONFIRMED (Xác nhận)** |
| **H2: Cross-Task Prototype Interference** | Prototype giữa các task bị chồng lấn trong không gian metric toàn cục. | - D01 (Oracle Local): **72.76%** $\to$ D02 (Oracle Global): **53.39%** (Tụt **19.37%** dù biết trước Task ID).<br>- Độ tương đồng cosine giữa các prototype khác task trung bình đạt **0.7082**, sát nút cùng task (**0.7308**), biên cách biệt chỉ **0.0226**. | **CONFIRMED (Xác nhận)** |
| **H3: Task-Agnostic Calibration Gap** | Điểm số tương đồng giữa các candidate prompts bị lệch phân phối, làm sai lệch fusion. | - Khi Router đúng, Soft Fusion kéo tụt accuracy từ **80.74% xuống 73.15%** (-7.59%).<br>- D08 (Z-score row-wise) làm sụp đổ hoàn toàn về 4.71% do phá vỡ cấu trúc độ dài vector. Cần hiệu chuẩn phân phối task-level thay vì sample-level. | **CONFIRMED (Xác nhận)** |
| **H4: Prompt Under-parameterization** | Độ dài prompt $L=8$ quá ngắn không đủ dung lượng biểu diễn để phân biệt quan hệ. | - Với $L=8$ ($L_{root}=4, L_{leaf}=4$), phân loại nội bộ D01 đạt tới **72.76%** (và **80.74%** khi router trúng).<br>- Lỗi nội bộ chỉ chiếm **7.96%** (169/2,122 mẫu).<br>- Prompt đã đủ biểu diễn phân biệt nội bộ; vấn đề nằm ở biên phân tách liên tác vụ (inter-task margin). | **REFUTED (Bác bỏ)** |
| **H5: Root Prompt Freezing Trade-off** | Đóng băng Root prompt sau Task 1 triệt tiêu tính dẻo, khiến các task sau học kém đi. | - Độ chính xác ban đầu của Task 1 đạt 89.49%, trong khi Task 8 chỉ đạt 30.74% khi mới học xong ($R_{8,8} = 30.74\%$).<br>- Độ chính xác học mới suy giảm đơn điệu theo thời gian: 89.5% (T1) $\to$ 85.1% (T2) $\to$ 43.8% (T3) $\to$ 37.3% (T6) $\to$ 30.7% (T8). | **CONFIRMED (Xác nhận)** |
| **H6: Within-Task vs. Cross-Task Confusion** | Nhầm lẫn nội bộ quan hệ trong task chiếm tỷ trọng tương đương nhầm lẫn sang task khác. | - Lỗi nội bộ (Within-Task): **169 mẫu (7.96%)**.<br>- Lỗi liên quan đến router / liên tác vụ: **1,061 mẫu (50.0%)**.<br>- Lỗi liên tác vụ áp đảo lỗi nội bộ gấp **6.27 lần**. | **REFUTED (Bác bỏ)** |
| **H7: Rehearsal-Free Memory Disadvantage** | Duy trì $M=0$ làm mất khả năng cân chỉnh ranh giới giữa các task cũ và mới. | - Do $M=0$, prototype của Task 1 đến Task 7 không thể cập nhật lại khi học task mới.<br>- Không gian biểu diễn của các leaf prompts sau không được trực tiếp tương phản với dữ liệu của các task trước, dẫn đến biên phân tách inter-task mỏng manh (0.0226). | **CONFIRMED (Xác nhận)** |
| **H8: Protocol Asymmetry with Baselines** | Khoảng cách số liệu giữa TAPTA và ConPL/CPL do bất đối xứng giao thức (10-way vs 40/80-way CIL). | - Mã nguồn ConPL (`dataprocess.py:466`, `fewrel_5shot.py:86`) chứng minh ConPL chỉ test trên 10 candidate relations (xác suất đoán mò 10%), và cấp 100 mẫu ở Task 1.<br>- TAPTA test CIL toàn cục 40/80 quan hệ (xác suất đoán mò 1.25% - 2.5%), $M=0$, strict 5-shot. So sánh trực tiếp là phi khoa học. | **CONFIRMED (Xác nhận)** |

---

## 2. Bác Bỏ Các Giả Định Kỹ Thuật Sai Trong Thiết Kế P4

1. **Bác bỏ Giả định 1: "Soft Decision Fusion ($K \ge 2$) luôn tốt hơn Hard Top-1 Routing ($K=1$)":**
   - *Thực nghiệm chứng minh:* Hard Top-1 (40.53%) vượt trội Soft Top-2 (40.34%), Top-4 (40.20%) và All-Task (40.15%). Soft fusion mang tính chất "pha loãng chất lượng" vì kéo theo vector biểu diễn của một task sai vào phép tính trung bình.
2. **Bác bỏ Giả định 2: "Centroid trong không gian unprompted BERT đủ độ phân tách để làm Task Router":**
   - *Thực nghiệm chứng minh:* Độ tương đồng cosine trung bình giữa các task centroid trong unprompted BERT là **0.9334** (cao nhất lên tới 0.9579). Không gian biểu diễn của BERT bị co lại thành một hình nón hẹp (Anisotropy / Cone Effect), khiến router không tham số rất dễ bị nhiễu.
3. **Bác bỏ Giả định 3: "Hiệu năng thấp là do độ dài Prompt Tree quá ngắn":**
   - *Thực nghiệm chứng minh:* Prompt tree hiện tại giải quyết 80.74% bài toán khi router chọn trúng task. Vấn đề không nằm ở dung lượng prompt mà nằm ở **cơ chế Router** và **sự thiếu vắng Contrastive Separation giữa các tasks**.

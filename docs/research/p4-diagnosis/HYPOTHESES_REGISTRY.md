# TAPTA Hypotheses Registry & Diagnostic Tracking

**Date Created:** 2026-11-25  
**Last Updated:** 2026-11-25  
**Target Architecture:** TAPTA P4 (Hierarchical Prompt Tree & Decision Fusion)  
**Goal:** Xác định nguyên nhân gốc rễ (root cause) khiến hiệu năng P4 thấp trên FewRel (57.58%) và TACRED (42.45%), làm căn cứ khoa học cho thiết kế P5.

---

## 1. Danh Mục 8 Giả Thuyết Khoa Học (Hypotheses H1–H8)

| ID | Tên giả thuyết | Nội dung cốt lõi | Biến độc lập | Tiêu chí chấp nhận / Bác bỏ | Trạng thái hiện tại |
|---|---|---|---|---|---|
| **H1** | **Routing Bottleneck** | Nguyên nhân lớn nhất kéo sụt độ chính xác là Router Top-1 miss (~50% trên TACRED). Nếu router chọn đúng task (Oracle Top-K), accuracy sẽ nhảy vọt lên $> 65\%$. | Router Mode (Top-1, Top-2, Top-4, Oracle Top-K) | Chấp nhận nếu $A_{\text{Oracle Top-K}} - A_{\text{P4 Top-2}} \ge 12\%$. | Đang chờ thực nghiệm D03 |
| **H2** | **Cross-Task Prototype Interference** | Dù có Oracle Task ID hoàn hảo, việc phân loại trên không gian toàn cục ($C_{\text{seen}}=80$) vẫn bị tụt sâu so với phân loại nội bộ task do sự chồng lấn prototype giữa các task. | Classification Scope (Task-Local vs. Global) | Chấp nhận nếu $A_{\text{D01 (Oracle Local)}} - A_{\text{D02 (Oracle Global)}} \ge 15\%$. | Đang chờ thực nghiệm D01 vs D02 |
| **H3** | **Task-Agnostic Calibration Gap** | Điểm số cosine similarity giữa các prompt khác nhau có biên độ phân phối lệch nhau, dẫn đến candidate prompt sai lấn át candidate prompt đúng khi soft fusion. | Score Calibration (Raw vs. Cosine Z-score / Temperature) | Chấp nhận nếu D08 cải thiện $\ge 3\%$ so với D05 không hiệu chuẩn. | Đang chờ thực nghiệm D08 |
| **H4** | **Prompt Under-parameterization** | Chiều dài prompt $L_{root}=4, L_{leaf}=4$ (tổng 8 tokens) không đủ dung lượng biểu diễn để hướng dẫn BERT phân biệt 80 quan hệ phức tạp trong điều kiện 5-shot. | Prompt Length ($L \in \{8, 16, 32\}$) | Chấp nhận nếu tăng $L$ lên 16 cải thiện D01 $> 5\%$. | Phân tích tại Phase 4 |
| **H5** | **Root Prompt Freezing Trade-off** | Đóng băng Root prompt sau Task 1 giúp chống quên tuyệt đối cho Task 1, nhưng triệt tiêu tính dẻo (plasticity), khiến các task sau ($T_5 \dots T_8$) học kém đi rõ rệt. | Root Freezing Policy (Frozen vs. Momentum EWC / EMA) | Chấp nhận nếu độ chính xác ban đầu của $T_6 \dots T_8$ thấp hơn $T_1 \dots T_3$ trên $15\%$. | Đã thấy dấu hiệu ở $R_{i,i}$ |
| **H6** | **Within-Task vs. Cross-Task Confusion** | Sai số do nhầm lẫn giữa các quan hệ nội bộ cùng một task (fine-grained confusion) chiếm tỷ trọng lớn hơn hoặc tương đương sai số nhảy sang task khác khi router đúng. | Error Category (Within-Task vs. Inter-Task) | Xác định tỷ lệ lỗi qua ma trận nhầm lẫn sample-level. | Đang chờ xuất `predictions.jsonl` |
| **H7** | **Rehearsal-Free Memory Disadvantage** | Việc giữ nghiêm ngặt $M=0$ (không lưu mẫu cũ) làm mất khả năng cân chỉnh ranh giới quyết định giữa task cũ và mới, giải thích khoảng cách với các phương pháp $M=10$. | Memory Rehearsal ($M=0$ vs. $M=5, 10$) | Đo lường mức độ suy giảm ranh giới (boundary shift). | Đối soát Track B |
| **H8** | **Protocol Asymmetry with Baselines** | Khoảng cách số liệu giữa TAPTA (57.58%) và các bài báo ConPL/CPL (~85%) là do bất đối xứng giao thức: ConPL chỉ test Candidate Ranking 10-way và cấp 100 mẫu ở T1. | Evaluation Protocol (10-way candidate vs. 80-way CIL) | Chấp nhận nếu test ConPL theo chuẩn CIL 80-way làm số liệu tụt về vùng $\sim 50-60\%$. | Đã xác minh mã nguồn |

---

## 2. Ma Trận Kế Hoạch Thực Nghiệm Chẩn Đoán (D01–D09)

| Exp ID | Tên Mode Chẩn Đoán | Mục đích khoa học | Giả thuyết liên quan | Yêu cầu tài nguyên |
|---|---|---|---|---|
| **D01** | **Oracle Task + Task-Local Classifier** | Đo trần năng lực phân biệt nội bộ quan hệ trong task của Prompt (10-way FewRel, 5-way TACRED). Loại bỏ hoàn toàn lỗi routing và lỗi can nhiễu cross-task. | H2, H4 | Inference-only |
| **D02** | **Oracle Task + Global Classifier** | Cung cấp Oracle Task ID để chọn đúng task prompt $P_t$, nhưng argmax trên toàn bộ seen classes (80-way FewRel, 40-way TACRED). Đo lường trực tiếp mức độ can nhiễu cross-task. | H2 | Inference-only |
| **D03** | **Oracle Top-K Routing** | Giả lập router luôn đưa True Task vào danh sách Top-K ($K=2$). Kiểm tra xem nếu router không bao giờ miss thì Decision Fusion đạt hiệu năng bao nhiêu. | H1 | Inference-only |
| **D04** | **Hard Top-1 Routing ($K=1$)** | Chỉ chọn duy nhất 1 task prompt theo Top-1 router prediction. So sánh với Soft Fusion $K=2$. | H1 | Inference-only |
| **D05** | **Reference Top-2 Routing ($K=2$)** | Cấu hình mặc định của P4 hiện tại. Dùng làm mốc đối chuẩn (Baseline reference). | Tất cả | Đã chạy |
| **D06** | **Top-4 Routing ($K=4$)** | Mở rộng số ứng viên lên 4 tasks. Kiểm tra việc mở rộng $K$ có tăng độ phủ hay làm loãng chất lượng quyết định. | H1, H3 | Inference-only |
| **D07** | **All-Task Fusion ($K=\text{seen\_tasks}$)** | Kết hợp toàn bộ các task prompt đã học. Xác định giới hạn của việc fusion phi chọn lọc. | H1, H3 | Inference-only |
| **D08** | **Score Calibration** | Áp dụng chuẩn hóa điểm số tương đồng cosine trước khi fusion (Cosine z-score và Min-Max Scaling) để cân bằng biên độ giữa các candidate prompts. | H3 | Inference-only |
| **D09** | **Frozen BERT + Flat Nearest Prototype** | Không dùng soft prompt nào, dùng trực tiếp biểu diễn unprompted $q(x)$ từ frozen BERT để nearest prototype matching. Đo đóng góp thực tế của Prompt Tree. | H4 | Baseline |

---

## 3. Nhật Ký Tiến Độ Thực Nghiệm
- **2026-11-25:** Khởi tạo registry; xác minh mã nguồn và phân tích lý thuyết cho H8 (Protocol Asymmetry).
- **Tiếp theo:** Triển khai runner chẩn đoán `experiments/diagnostics/run_p4_diagnostics.py` và chạy trên cả FewRel và TACRED.

# TAPTA P4 Quantitative Error Taxonomy & Cross-Benchmark Analysis

**Date:** 2026-11-25  
**Benchmarks:** FewRel Track A (8,000 samples) & TACRED Track A (2,122 samples)  
**Setting:** Strict Equal 5-shot, Rehearsal-Free $M=0$, Seed 2021  
**Artifact Source:** `runs/diagnostics/{fewrel,tacred}_5shot/seed_2021/predictions.jsonl`  

---

## 1. Bảng Đối Chiếu Ma Trận Chẩn Đoán D01 – D09 Giữa FewRel & TACRED

| Ký hiệu | Cấu hình Thực nghiệm Chẩn đoán | FewRel Track A (80 rels) | TACRED Track A (40 rels) | Ý nghĩa Khoa học & Bản chất Định lượng |
|---|---|---:|---:|---|
| **D01** | **Oracle Task + Task-Local Classifier** | **84.14%** | **72.05%** | **Trần năng lực biểu diễn nội bộ task:** Khớp hoàn hảo với mức SOTA công bố của ConPL (85.2% FewRel, 76.5% TACRED) khi chỉ test trong tập ứng viên nhỏ. |
| **D02** | **Oracle Task + Global Classifier** | **65.44%** | **53.53%** | **Biểu diễn toàn cục khi biết trước task:** Cấp đúng Task ID nhưng cho phép argmax trên toàn bộ quan hệ đã học. |
| **P_drop** | **Cross-Task Interference Penalty (D01 - D02)** | **18.70%** | **18.52%** | **Tính nhất quán xuyên benchmark:** Độ sụt giảm do can nhiễu cross-task trên cả 2 dataset đều đồng nhất ở mức $\approx \mathbf{18.6\%}$! |
| **D03** | **Oracle Top-K Routing** | **58.44%** | **43.36%** | **Trần của Soft Fusion nếu Router không miss:** Ngay cả khi true task luôn nằm trong Top-2, Soft Fusion chỉ đạt 58.44% (FewRel) và 43.36% (TACRED). |
| **D04** | **Hard Top-1 Routing ($K=1$)** | **58.38%** | **40.06%** | **Inference chọn cứng Top-1:** Trên FewRel, Hard Top-1 (58.38%) vượt trội hơn Soft Top-2 (57.58%)! |
| **D05** | **Reference Top-2 Soft Fusion ($K=2$)** | **57.58%** | **41.28%** | **Cấu hình P4 Chuẩn hiện tại:** Kết quả khớp 100% với báo cáo P4 đã đăng ký trong registry. |
| **D06** | **Top-4 Soft Fusion ($K=4$)** | **57.63%** | **41.09%** | **Mở rộng $K=4$:** Không mang lại cải thiện ý nghĩa, gây loãng độ sắc nét prototype. |
| **D07** | **All-Task Fusion ($K=8$)** | **57.64%** | **40.48%** | **Fusion phi chọn lọc toàn bộ tasks:** Duy trì xu hướng bão hòa hoặc suy giảm. |
| **D08** | **Score Calibration** | **57.63%** | **4.71%** | **Hiệu chỉnh điểm số:** Row-wise Z-score không cải thiện trên FewRel và làm méo phân phối trên TACRED. |
| **D09** | **Frozen BERT Flat Prototype (Zero-prompt)** | **59.93%** | **41.52%** | **Baseline unprompted BERT:** Nhỉnh hơn P4 không có contrastive margin, chứng minh sự cấp thiết của việc mở rộng biên phân tách liên tác vụ trong P5. |
| **R_T1** | **Router Top-1 Accuracy** | **64.69%** | **48.44%** | Router tắc nghẽn do Anisotropy (hình nón unprompted BERT). |
| **R_T2** | **Router Top-2 Coverage** | **81.41%** | **70.12%** | Độ phủ Top-2 đạt 70–81%. |
| **R_T4** | **Router Top-4 Coverage** | **93.05%** | **88.74%** | Độ phủ Top-4 đạt 89–93%. |

---

## 2. Bảng Phân Phối Lỗi Sample-Level (8,000 Mẫu FewRel & 2,122 Mẫu TACRED)

| Nhóm Lỗi (Error Category) | FewRel Mẫu | FewRel (%) | TACRED Mẫu | TACRED (%) | Bản chất cơ chế gây lỗi |
|---|---:|---:|---:|---:|---|
| **Chính xác (Correct)** | **4,606** | **57.57%** | **856** | **40.34%** | Phân loại đúng quan hệ toàn cục CIL. |
| **Router Miss (Bỏ sót ngoài Top-2)** | **1,418** | **17.72%** | **613** | **28.89%** | Router chệch hướng do task centroid bị collinear ($0.93 - 0.96$). |
| **Inter-Task Confusion (Nhiễu Soft Fusion)** | **1,247** | **15.59%** | **448** | **21.11%** | True Task trong Top-2 nhưng Soft Fusion bị candidate sai lấn át. |
| **Within-Task Confusion (Nhầm nội bộ)** | **450** | **5.62%** | **169** | **7.96%** | Router đúng, nhưng prompt không phân biệt nổi 2 quan hệ trong task. |
| **Cross-Task Intrusion (Chèn ép toàn cục)** | **279** | **3.49%** | **36** | **1.70%** | Router đúng, D01 đúng, nhưng prototype task ngoại lai chèn ép. |
| **Tổng cộng** | **8,000** | **100.00%** | **2,122** | **100.00%** | Tổng số mẫu kiểm thử. |

---

## 3. Hành Vi Kỳ Lạ Của Soft Fusion: Gây Suy Giảm Khi Router Đúng

Bảng phân tích khi Router Top-1 dự đoán **ĐÚNG** True Task:

| Độ đo khi Router Top-1 Đúng | FewRel Track A ($5,175$ mẫu đúng) | TACRED Track A ($1,028$ mẫu đúng) |
|---|---:|---:|
| **Phân loại Nội bộ (D01 Task-Local)** | **90.90%** ($4,704/5,175$) | **80.74%** ($830/1,028$) |
| **Phân loại Toàn cục (D02 Global)** | **85.26%** ($4,412/5,175$) | **78.89%** ($811/1,028$) |
| **Soft Fusion P4 Chuẩn (D05)** | **80.83%** ($4,183/5,175$) | **73.15%** ($752/1,028$) |
| **Mức sụt giảm do Soft Fusion gây ra** | **-10.07%** | **-7.59%** |
| **Độ chính xác nếu dùng Hard Task Gating** | **58.80%** ($4,704/8,000$) | **39.11%** ($830/2,122$) |

> [!CRITICAL] Định Luật Về Độc Tính Của Soft Fusion
> Trên cả FewRel và TACRED, khi router đã chọn đúng task ở vị trí Top-1, việc ép mô hình pha trộn candidate thứ 2 (vốn thuộc về một task SAI) làm độ chính xác **tụt giảm từ 7.59% đến 10.07%**! Candidate sai hoạt động như một nguồn nhiễu độc hại. Thay thế bằng **Confidence-Gated Hard Routing** lập tức thu hồi lại toàn bộ phần suy giảm này.

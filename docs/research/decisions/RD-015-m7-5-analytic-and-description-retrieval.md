# RD-015 — M7.5 Improved Retrieval Space (Analytic Classifier, Random Projection & Description Prototypes)

- **Ngày:** 2026-11-28
- **Trạng thái:** Accepted
- **Liên quan:** M7.5 (Section 8), RD-013

---

## 1. Thứ tự triển khai tuần tự theo Section 8
Ưu tiên phương pháp giải tích đóng không replay và ít tham số trước khi cân nhắc trainable query adapter (chỉ thử query adapter nếu Bước 1–3 chưa đạt mục tiêu hoặc cần làm ablation):
1. Identity Incremental Analytic Classifier (`DT-C0.5a`)
2. Frozen Random Projection + Analytic Classifier (`DT-C0.5b`, $M=4096$ mặc định; cấu hình được)
3. Relation Description-Enhanced Prototype (`DT-C0.5c`, $\beta = 0.25$)
4. Tích hợp với Flat (`DT-C2.5`) và DynaTree (`DT-C4.5`) Candidate Reranking.

## 2. Chính sách chống Test Leakage cho $\lambda_{\text{ridge}}$ và $\beta_{\text{desc}}$
- Theo `RD-013`, cấm tune $\lambda_{\text{ridge}}$ hoặc $\beta_{\text{desc}}$ trên `valid_0` hoặc `test_0`.
- $\lambda_{\text{ridge}}$ được đặt theo mặc định chuẩn RanPAC/ridge ($\lambda_{\text{ridge}} = 10.0$ hoặc chọn bằng **LOOCV / GCV giải tích chỉ trên ma trận thống kê $G, C$ của dữ liệu train hiện tại**, hoàn toàn không chạm tập test).
- Tương tự, mọi chọn lọc cấu hình trước multi-seed (`2022–2025`) phải chốt trước trên seed `2021` (dev).

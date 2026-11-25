# Kế Hoạch Triển Khai Chi Tiết (Implementation Plan)
**Dự án:** Continual Relation Extraction with Task-Aware Prompt Adaptation (TAPTA)  
**Tài liệu:** `docs/research/tapta-scalability/implementation-plan.md`  
**Phương pháp luận:** Spec-Driven Development (SDD) & Test-Driven Development (TDD)

---

## 1. Lộ Trình Triển Khai Theo Giai Đoạn (Phased Roadmap)

### Giai đoạn 1: Dataset Pipeline & Nested Sampling (TDD)
- **Mục tiêu:** Xây dựng module nạp dữ liệu sạch `src/data/scalable_fewrel_loader.py` hỗ trợ mọi cỡ mẫu (5, 10, 50, 100, Full).
- **Nguyên tắc TDD:** Viết bộ test `tests/test_scalable_loader.py` trước khi viết hàm xử lý chính.
- **Tiêu chí hoàn thành:** 100% test pass, xác nhận zero data leakage và nested sampling.

### Giai đoạn 2: Cấu Hình Thực Nghiệm Chuẩn Hóa
- **Mục tiêu:** Thiết lập thư mục cấu hình `configs/scalability/` cho 5 mức dữ liệu (DS-05, DS-10, DS-50, DS-100, DS-FULL).
- **Tiêu chí hoàn thành:** Tích hợp đầy đủ schema, validation không cho phép fallback âm thầm.

### Giai đoạn 3: Runner Nghiên Cứu Mở Rộng & Chẩn Đoán Oracle
- **Mục tiêu:** Tạo script `experiments/run_scalability_study.py` kế thừa mô hình P4, bổ sung đo đạc chi phí tài nguyên (training time, GPU peak memory, persistent buffer bytes) và chế độ chẩn đoán Oracle Routing.
- **Tiêu chí hoàn thành:** Hỗ trợ lưu trữ đầy đủ artifacts và dataset manifest.

### Giai đoạn 4: Kiểm Thử Khói & Pilot Run (Smoke Test & Pilot Stage)
- **Mục tiêu:**
  - Chạy smoke test 2 task với dữ liệu giả lập/cắt ngắn để xác nhận không lỗi runtime.
  - Chạy pilot run 8 task trên môi trường cho phép (MPS / CPU / Colab) để đo đạc thời gian huấn luyện thực tế.
- **Tiêu chí hoàn thành:** Checkpoint và metrics sinh ra đúng format.

### Giai đoạn 5: Tổng Hợp Kết Quả & Báo Cáo Luận Văn
- **Mục tiêu:** Xây dựng công cụ tổng hợp đa seed và lập báo cáo `final-report.md`.
- **Tiêu chí hoàn thành:** Bảng kết quả tổng hợp 5 mức dữ liệu kèm biểu đồ phân tích RQ1–RQ5.

---

## 2. Ranh Giới Kiểm Soát Môi Trường (Colab GPU vs Local Mac)
- **MacBook cá nhân:** Chỉ dùng cho việc lập trình code, viết unit tests, xác thực cấu hình và chạy smoke test mô phỏng nhẹ (< 1 phút).
- **Google Colab GPU (CUDA):** Thực hiện toàn bộ các phiên huấn luyện và đánh giá chính thức (DS-05, DS-10, DS-50, DS-100, DS-FULL).

# Danh Sách Công Việc Cần Triển Khai (Tasks & Checklist)
**Dự án:** Continual Relation Extraction with Task-Aware Prompt Adaptation (TAPTA)  
**Tài liệu:** `docs/research/tapta-scalability/tasks.md`

---

## Danh Mục Công Việc (WBS - Work Breakdown Structure)

- [x] **TASK-001:** Kiểm toán repository và xác minh cấu trúc dữ liệu (`docs/research/tapta-scalability/repository-audit.md`)
- [x] **TASK-002:** Kiểm chứng benchmark từ các paper gốc và phân tích tính công bằng (`docs/research/tapta-scalability/benchmark-comparison.md`)
- [x] **TASK-003:** Soạn thảo đặc tả kỹ thuật và ranh giới nghiên cứu (`docs/research/tapta-scalability/spec.md`)
- [x] **TASK-004:** Lập kế hoạch triển khai chi tiết (`docs/research/tapta-scalability/implementation-plan.md`)
- [ ] **TASK-005:** Soạn thảo tiêu chuẩn chấp thuận và ma trận thí nghiệm (`acceptance-criteria.md`, `experiment-matrix.md`, `risk-register.md`)
- [ ] **TASK-006:** Viết bộ unit tests cho Scalable Dataset Loader (`tests/test_scalable_loader.py`)
- [ ] **TASK-007:** Triển khai module `src/data/scalable_fewrel_loader.py` hỗ trợ nested sampling & zero leakage
- [ ] **TASK-008:** Chạy kiểm thử pytest đảm bảo 100% tests của loader pass
- [ ] **TASK-009:** Tạo bộ cấu hình thí nghiệm `configs/scalability/` cho 5 mức dữ liệu
- [ ] **TASK-010:** Triển khai experiment runner `experiments/run_scalability_study.py` kèm theo tracking tài nguyên và Oracle Diagnostic
- [ ] **TASK-011:** Chạy Smoke Test kiểm chứng toàn bộ luồng huấn luyện, checkpoint và evaluator
- [ ] **TASK-012:** Đóng gói Colab Execution Script / Notebook phục vụ chạy thực nghiệm chính thức trên GPU
- [ ] **TASK-013:** Cập nhật Research Track Log trong Vault và tổng kết tiến độ Phase 0–4

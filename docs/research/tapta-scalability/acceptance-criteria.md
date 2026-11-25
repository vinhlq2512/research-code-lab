# Tiêu Chuẩn Chấp Thuận & Ranh Giới Nghiệm Thu (Acceptance Criteria)
**Dự án:** Continual Relation Extraction with Task-Aware Prompt Adaptation (TAPTA)  
**Tài liệu:** `docs/research/tapta-scalability/acceptance-criteria.md`

---

## 1. Tiêu Chuẩn Dataset & Data Leakage (AC-DATA)
- **AC-DATA-01:** `ScalableFewRelDataset` nạp chính xác số mẫu train theo cấu hình:
  - `train_shot = 5` $\implies$ Đúng 5 mẫu/quan hệ $\times$ 10 quan hệ = 50 mẫu/task (400 mẫu tổng).
  - `train_shot = 10` $\implies$ Đúng 10 mẫu/quan hệ $\times$ 10 quan hệ = 100 mẫu/task (800 mẫu tổng).
  - `train_shot = 50` $\implies$ Đúng 50 mẫu/quan hệ $\times$ 10 quan hệ = 500 mẫu/task (4,000 mẫu tổng).
  - `train_shot = 100` $\implies$ Đúng 100 mẫu/quan hệ $\times$ 10 quan hệ = 1,000 mẫu/task (8,000 mẫu tổng).
  - `train_shot = "full"` $\implies$ Toàn bộ mẫu trong train pool sạch (~420 mẫu/quan hệ $\times$ 10 = ~4,200 mẫu/task).
- **AC-DATA-02 (Zero Leakage):** $\text{Train Set} \cap \text{Test Set} = \emptyset$. Không có bất kỳ câu nào trong tập kiểm thử xuất hiện trong tập huấn luyện ở bất kỳ shot nào.
- **AC-DATA-03 (Nested Sampling):** Với cùng một seed, danh sách mẫu của shot nhỏ hơn phải là tập con thực sự của shot lớn hơn:
  $$S_5 \subset S_{10} \subset S_{50} \subset S_{100} \subset S_{\text{Full}}$$
- **AC-DATA-04 (Deterministic Reproducibility):** Khởi tạo cùng seed sinh ra đúng 100% thứ tự câu và token giống nhau trên mọi lần chạy.

---

## 2. Tiêu Chuẩn Mô Hình & Continual Learning (AC-MODEL)
- **AC-MODEL-01 (Frozen Backbone):** 100% trọng số của BERT (`bert-base-uncased`) có `requires_grad = False` trong suốt 8 tác vụ.
- **AC-MODEL-02 (Hierarchical Prompt Tree):**
  - Task 0: Cập nhật $P^{\text{root}}$ ($L=4$) và $P_0^{\text{leaf}}$ ($L=4$).
  - Task 1..7: $P^{\text{root}}$ đóng băng vĩnh viễn, chỉ cập nhật $P_t^{\text{leaf}}$ ($L=4$, đúng 3,072 tham số) kèm mất mát trực giao $\mathcal{L}_{\text{orth}}$.
- **AC-MODEL-03 (Rehearsal-Free M=0):** Hoàn toàn không lưu hoặc phát lại văn bản thô từ các task cũ.
- **AC-MODEL-04 (Task-Agnostic Inference):** Toàn bộ đánh giá chính thức sử dụng bộ định tuyến phi tham số và Decision-Level Soft Fusion, không sử dụng Ground-Truth Task ID.

---

## 3. Tiêu Chuẩn Thử Nghiệm & Tái Lập (AC-EXP)
- **AC-EXP-01:** Cung cấp file cấu hình YAML độc lập cho từng experiment (DS-05, DS-10, DS-50, DS-100, DS-FULL).
- **AC-EXP-02:** Mỗi run tạo đầy đủ artifact: `metrics.json`, `performance_matrix.csv`, `resource_usage.json`, `dataset_manifest.json`, `summary.json`.
- **AC-EXP-03:** Cung cấp chế độ chạy chẩn đoán Oracle Task ID trên cùng checkpoint để tính `Routing Gap`.
- **AC-EXP-04:** Tất cả unit tests trong `tests/test_scalable_loader.py` đạt 100% pass trước khi triển khai training.

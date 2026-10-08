---
type: project
created: 2026-10-08
updated: 2026-10-08
---

# Upper Bound: Joint Training (Multitask) Phase Execution Notes (What & Why)

Tài liệu này lưu vết chi tiết việc thiết kế, triển khai và nghiệm thu Cận trên (Upper Bound) cho FewRel Track A (80 quan hệ, 5-shot, seed 2021). Mỗi phần đều ghi rõ **ĐÃ LÀM GÌ (What)** và **TẠI SAO PHẢI LÀM (Why)**.

---

## Phase 1: Protocol Alignment & Design Clarification (Grill-Me)

### 1. Đã làm gì (What)
- Thực hiện phỏng vấn đối soát qua `/grill-me` để chốt trọn vẹn 7 nhánh quyết định:
  1. **Dataset & Seed:** Đồng bộ 100% với Baseline B0 (FewRel Track A, 80 relations, 5-shot train/val, 140 test, seed 2021).
  2. **Data Assembly:** Gom chung 400 mẫu train của 80 quan hệ và shuffle đồng nhất (multitask joint training).
  3. **Kiến trúc mô hình:** Tái sử dụng `BERTRelationClassifier` (`bert-base-uncased`, entity markers `[E1]`, `[E2]`, linear head 80 classes), mở toàn bộ 80 classes (`seen_classes_count = 80`).
  4. **Ngân sách huấn luyện:** 15 epochs (batch size 16, LR $2\times 10^{-5}$, AdamW, linear warmup), kiểm tra trên 400 mẫu validation mỗi epoch để lưu checkpoint tốt nhất (`best_model.pt`).
  5. **Đánh giá hai tầng:** Đánh giá tổng thể 80 lớp trên toàn bộ 11,200 mẫu test, kết hợp phân rã độ chính xác theo 8 task ($T_1 \dots T_8$) tương ứng với B0.
  6. **Thiết bị:** Chạy trên Apple Silicon GPU (`mps`).
  7. **Quản lý & Registry:** Đăng ký mã thực nghiệm `E002` trong registry trung tâm.
- Tạo quyết định kiến trúc: [`docs/adr/0002-upper-bound-joint-training.md`](../../docs/adr/0002-upper-bound-joint-training.md).
- Cập nhật từ điển miền: [`GLOSSARY.md`](../../GLOSSARY.md) bổ sung thuật ngữ `Joint Training (Upper Bound)`.

### 2. Tại sao phải làm (Why)
- **Đảm bảo tính công bằng khoa học (Benchmark Parity):** Để Upper Bound làm mốc trần chuẩn mực đối sánh với Lower Bound B0, toàn bộ tiền xử lý, split dữ liệu, backbone BERT và ánh xạ nhãn bắt buộc phải giống hệt B0, chỉ khác duy nhất cơ chế huấn luyện (học cùng lúc toàn bộ thay vì học tuần tự chia task).
- **Tránh Overfitting trong Few-Shot:** Trong bối cảnh 5-shot (400 mẫu / 80 lớp), việc lưu checkpoint dựa trên Validation accuracy (`best_model.pt`) đảm bảo mô hình chọn được trọng số có khả năng tổng quát hóa tốt nhất trên tập test.

---

## Phase 2: Configuration & Runner Implementation

### 1. Đã làm gì (What)
- Tạo file cấu hình chính thức:
  - [`configs/upper_bound_joint_fewrel_5shot_seed2021.json`](../../configs/upper_bound_joint_fewrel_5shot_seed2021.json)
  - [`configs/upper_bound_joint_fewrel_5shot_seed2021.yaml`](../../configs/upper_bound_joint_fewrel_5shot_seed2021.yaml)
- Xây dựng runner hoàn chỉnh: [`experiments/run_joint_training.py`](../../experiments/run_joint_training.py):
  - Tự động nạp FewRel dataset và TaskOrder seed 2021.
  - Gom toàn bộ train/val/test samples qua 8 task.
  - Vòng lặp huấn luyện 15 epochs với log streaming ra `training_log.jsonl`.
  - Checkpoint manager tự động lưu `best_model.pt` khi đạt đỉnh validation accuracy.
  - Đánh giá trên tập test (11,200 mẫu) và xuất `per_task_breakdown.csv`, `per_task_breakdown.json`, `summary.json`, và `conclusion.md`.

### 2. Tại sao phải làm (Why)
- **Tự động hóa hoàn toàn (Zero Manual Steps):** Script runner độc lập giúp thực nghiệm có thể chạy lại bất cứ lúc nào với 1 dòng lệnh, bảo đảm tính tái lập 100%.

---

## Phase 3: Real Execution & Empirical Results

### 1. Đã làm gì (What)
- Kích hoạt chạy thực nghiệm trên Apple Silicon GPU (`mps`):
  ```bash
  .venv/bin/python3 experiments/run_joint_training.py --device mps
  ```
- Quá trình chạy hoàn tất trong ~7 phút (từ Epoch 1 đến Epoch 15 và đánh giá toàn bộ 55,200 lượt test samples).
- Kết quả ghi nhận tại `results/fewrel/5shot/Upper_Bound_joint/seed_2021/`:
  - **Best Validation Epoch:** Epoch 15 ($53.25\%$).
  - **Overall 80-Way Test Accuracy:** **$52.12\%$**
  - **Overall 80-Way Test Macro-F1:** **$48.62\%$**
  - **Khoảng cách vượt trội so với Lower Bound (B0):**
    - Accuracy Gain: **$+43.35\%$** ($52.12\%$ vs $8.76\%$)
    - Macro-F1 Gain: **$+38.93\%$** ($48.62\%$ vs $9.69\%$)
  - **Phân rã per-task trên test set:**
    - Task 1: $51.41\%$ (B0 cuối cùng là $0.00\%$)
    - Task 2: $51.12\%$ (B0 cuối cùng là $0.00\%$)
    - Task 3: $60.41\%$ (B0 cuối cùng là $6.64\%$)
    - Task 4: $51.54\%$ (B0 cuối cùng là $1.54\%$)
    - Task 5: $58.90\%$ (B0 cuối cùng là $3.13\%$)
    - Task 6: $63.49\%$ (B0 cuối cùng là $14.96\%$)
    - Task 7: $38.55\%$ (B0 cuối cùng là $6.81\%$)
    - Task 8: $41.25\%$ (B0 cuối cùng là $37.04\%$)
- Đăng ký chính thức thực nghiệm **`E002`** trong `experiments/registry.yaml` và `registry.json` với `baseline_role: upper_bound`, `status: completed_real`.

### 2. Tại sao phải làm (Why)
- **Thiết lập chuẩn trần hiệu năng (Ceiling Metric):** Con số $52.12\%$ và khoảng cách $+43.35\%$ phản ánh rõ rệt mức trần mà các phương pháp Continual Learning tiên tiến (như ConPL, CPL, TAPTA) có thể hướng tới nhằm thu hẹp khoảng cách giữa Lower Bound và Upper Bound.

---
type: project
created: 2026-09-21
updated: 2026-09-21
---

# B0 Baseline Phase Execution Notes (What & Why)

Tài liệu này lưu vết chi tiết mọi bước triển khai baseline B0 (Sequential Fine-Tuning) cho FewRel Track A (8 tasks × 10 relations, 5-shot, seed 2021). Mỗi phase đều ghi rõ **ĐÃ LÀM GÌ (What)** và **TẠI SAO PHẢI LÀM (Why)**.

---

## Phase 0: Codebase Audit & Component Precheck

### 1. Đã làm gì (What)
- Thực hiện rà soát toàn bộ 17 hạng mục kiểm tra trong repository `research-code-lab`.
- Xác nhận dữ liệu thô FewRel Track A (56,000 mẫu, 80 quan hệ) và bộ nạp chuẩn `FewRelDataset` tại `dataset-pipelines/continual-relation-extraction/src/datasets/fewrel.py`.
- Tạo và xác thực thứ tự task cho seed 2021 tại `task-orders/fewrel/order_seed_2021.json` gồm 8 tasks, mỗi task đúng 10 quan hệ không trùng lặp.
- Kiểm tra tính bất biến của ánh xạ nhãn: 80 quan hệ được gán ID từ $0 \dots 79$ theo thứ tự bảng chữ cái và giữ nguyên vẹn qua mọi task.
- Xác nhận các thành phần còn thiếu: Runner huấn luyện nơ-ron tuần tự, các độ đo mở rộng ($BWT, AIA$) và môi trường PyTorch trong `.venv`.
- Tạo báo cáo deliverable: [`B0_PRECHECK.md`](../../B0_PRECHECK.md) tại root và copy vào `docs/B0_PRECHECK.md`.

### 2. Tại sao phải làm (Why)
- **Tránh sai lệch benchmark (Benchmark Integrity):** Nếu không kiểm tra trước, mô hình có thể vô tình dùng lại nhãn cục bộ (local re-indexing) hoặc chia sai số task (ví dụ 10 task thay vì 8 task chuẩn của Track A), làm hỏng toàn bộ kết quả so sánh.
- **Bảo toàn tính bất biến (Label Invariant):** Cần đảm bảo relation $X$ có ID $k$ thì ở Task 1 hay Task 8 nhãn của nó vẫn là $k$, tránh việc mô hình dự đoán nhầm lớp cũ do label mapping bị trôi.
- **Tiết kiệm tài nguyên (Fail-Fast):** Xác định sớm những gì đã có sẵn để tái sử dụng tối đa, tránh viết lại code thừa hoặc chạy thử nghiệm tốn kém khi evaluator chưa sẵn sàng.

---

## Phase 1: Freeze Experimental Protocol & Config

### 1. Đã làm gì (What)
- Tạo file cấu hình đóng băng thực nghiệm:
  - [`configs/b0_sequential_ft_fewrel_5shot_seed2021.yaml`](../../configs/b0_sequential_ft_fewrel_5shot_seed2021.yaml)
  - [`configs/b0_sequential_ft_fewrel_5shot_seed2021.json`](../../configs/b0_sequential_ft_fewrel_5shot_seed2021.json)
- Khóa toàn bộ các nhóm thông số:
  - **Dữ liệu:** FewRel Track A, 8 task, 10 relation/task, 5-shot train (50 mẫu/task), 5-shot val (50 mẫu/task), toàn bộ 140 mẫu test/relation (1,400 mẫu/task), `seed = 2021`.
  - **Mô hình:** `bert-base-uncased`, `max_seq_length: 128`, global classification head (80 classes), `seen_classes` masking.
  - **Khóa độ tinh khiết (Lower Bound Purity):** `memory_size: 0`, `replay: false`, `prompt: false`, `prototype: false`, `router: false`, `knowledge_distillation: false`, `ema_teacher: false`, `regularization: false`.
  - **Huấn luyện:** LR = $2.0 \times 10^{-5}$, `AdamW` (weight decay = 0.01), `linear` scheduler (warmup = 0.1), batch size = 16, epochs/task = 5.
- Xác thực cú pháp và tính toàn vẹn bằng script Python độc lập.

### 2. Tại sao phải làm (Why)
- **Đảm bảo tính công bằng khoa học (Scientific Fairness):** B0 là cận dưới (Lower Bound). Mọi siêu tham số phải cố định đồng nhất giữa các task $T_1 \dots T_8$. Không được phép tinh chỉnh riêng cho từng task để lấy điểm cao.
- **Loại bỏ hoàn toàn rò rỉ cơ chế (Zero-Leakage):** B0 phải hoàn toàn sạch bóng các thành phần của hướng nghiên cứu chính (TAPTA) như prototype, tree, prompt, router hay memory replay. Nếu để sót bất kỳ cơ chế nào, cận dưới sẽ bị "nhiễm bẩn", làm giảm giá trị chứng minh hiệu quả của TAPTA sau này.
- **Tự động hóa và tái lập (Reproducibility):** Lưu cấu hình thành file chuẩn giúp runner tự động đọc tham số mà không bị hardcode trong code, giúp bất kỳ ai cũng có thể tái lập chính xác 100% thí nghiệm.

---

## Các Phase tiếp theo (Đang chờ thực hiện)
- **Phase 3 (Evaluator Gate):** Bổ sung $BWT$, $AIA$, Old/New breakdown vào `metrics.py` và viết unit test ma trận $A_{t,j}$ để đảm bảo bộ đo lường hoàn toàn chính xác trước khi train.
- **Phase 4 (Sequential Trainer):** Xây dựng vòng lặp fine-tune tuần tự $T_1 \to \dots \to T_8$ với cùng một model BERT liên tục cập nhật trọng số.
- **Phase 9 (2-Task Smoke Test):** Chạy thử 2 task đầu tiên để kiểm tra $A_{1,1}, A_{2,1}, A_{2,2}$ trước khi chạy full 8 task.
- **Phase 10 (Full T1→T8 Run):** Huấn luyện toàn bộ 8 task, tạo đủ 36 cell $A_{t,j}$.
- **Phase 11 (Verification & E001):** Chạy `validate_b0_results.py`, vẽ biểu đồ và đăng ký kết quả vào registry.

---
type: project
created: 2026-10-09
updated: 2026-10-09
---

# Frozen BERT + Linear Head (Sequential Probing) Phase Execution Notes (What & Why)

Tài liệu này lưu vết chi tiết việc thiết kế, tối ưu hóa và nghiệm thu thực nghiệm **E003: Frozen BERT + Linear Head (Sequential Probing)** cho FewRel Track A (80 quan hệ, 8 tasks, 5-shot, seed 2021). Mỗi phần đều ghi rõ **ĐÃ LÀM GÌ (What)** và **TẠI SAO PHẢI LÀM (Why)**.

---

## Phase 1: Protocol Alignment & Design Clarification (/grill-me)

### 1. Đã làm gì (What)
- Thực hiện phỏng vấn đối soát qua `/grill-me` để chốt trọn vẹn 7 nhánh quyết định:
  1. **Mục tiêu & Chế độ:** Huấn luyện tuần tự qua 8 tasks ($T_1 \to T_8$) tương tự B0 nhưng đóng băng BERT để đo mức độ quên ở riêng classifier head.
  2. **Phạm vi đóng băng:** Đóng băng 100% BERT encoder (12 layers + embeddings). Vector biểu diễn là $[h_{e1}; h_{e2}] \in \mathbb{R}^{1536}$.
  3. **Cấu trúc Head:** Shared Single Global Head (80 classes) với `seen_classes` masking, nhận gradient tuần tự qua từng task.
  4. **Siêu tham số tối ưu:** Tối ưu riêng cho Linear Probing (LR $1\times 10^{-3}$, AdamW weight decay 0.01, batch size 16, 15 epochs/task).
  5. **Chiến lược tính toán:** Feature Caching trong RAM (forward qua BERT 1 lần cho toàn bộ dataset, sau đó train/eval trên cached vectors).
  6. **Định danh & Lưu trữ:** Mã thực nghiệm `E003`, role `representation_probing`, lưu tại `results/fewrel/5shot/Frozen_BERT_linear_head/seed_2021/`.
  7. **Quy trình nghiệm thu:** Tạo branch `feat/frozen-bert-linear-head`, lập ADR 0003, chạy Smoke Test 2 task, chạy Full 8 tasks, kiểm tra DoD 8/8 và đăng ký Registry.
- Tạo quyết định kiến trúc: [`docs/adr/0003-frozen-bert-linear-head.md`](../../docs/adr/0003-frozen-bert-linear-head.md).
- Cập nhật từ điển miền: [`GLOSSARY.md`](../../GLOSSARY.md) bổ sung thuật ngữ `Sequential Linear Probing (Frozen Backbone)`.

### 2. Tại sao phải làm (Why)
- **Cô lập nguyên nhân gây quên (Decoupling Forgetting Mechanisms):** Trong Continual Learning, câu hỏi then chốt là: Catastrophic Forgetting bắt nguồn từ sự méo mó biểu diễn không gian ẩn của BERT (Representation Drift) hay do các lớp mới ghi đè tham số của Classification Head (Classifier Interference)? Việc đóng băng 100% BERT cung cấp câu trả lời thực nghiệm chính xác.
- **Tiết kiệm tài nguyên khoa học (RAM Feature Caching):** Do weights của BERT hoàn toàn bất biến, trích xuất 55,600 vector 1 lần vào RAM giúp rút ngắn thời gian chạy từ 12 phút xuống dưới 3 phút mà kết quả toán học tương đương 100%.

---

## Phase 2: Configuration & Dedicated Branch

### 1. Đã làm gì (What)
- Tạo dedicated branch `feat/frozen-bert-linear-head`.
- Tạo file cấu hình:
  - [`configs/frozen_bert_linear_head_fewrel_5shot_seed2021.json`](../../configs/frozen_bert_linear_head_fewrel_5shot_seed2021.json)
  - [`configs/frozen_bert_linear_head_fewrel_5shot_seed2021.yaml`](../../configs/frozen_bert_linear_head_fewrel_5shot_seed2021.yaml)

### 2. Tại sao phải làm (Why)
- **Bảo toàn quy ước dự án:** Mọi thay đổi lớn đều phải nằm trên branch riêng theo `project-conventions.md`.
- **Cấu hình có thể tái lập:** Đóng băng toàn bộ tham số vào JSON/YAML giúp chạy lại thí nghiệm 1 dòng lệnh.

---

## Phase 3: Runner & Feature Caching Implementation

### 1. Đã làm gì (What)
- Xây dựng runner hoàn chỉnh tại [`experiments/run_frozen_bert.py`](../../experiments/run_frozen_bert.py):
  - Khởi tạo `BERTRelationClassifier` và thiết lập `requires_grad = False` cho 100% encoder (109.5M tham số), chỉ unfreeze 122.9K tham số ở linear head.
  - Trích xuất và cache toàn bộ train/test feature vectors qua các task.
  - Vòng lặp huấn luyện 15 epochs per task với AdamW và linear warmup.
  - Checkpoint manager lưu `model.pt` đầy đủ (418 MB) sau mỗi task.
  - Đánh giá 36 ô ma trận tam giác dưới và streaming kết quả ra `metrics.jsonl`, `summary.json`, và `conclusion.md`.
- Cập nhật [`experiments/generate_plots_svg.py`](../../experiments/generate_plots_svg.py) hỗ trợ `--results-dir` để xuất biểu đồ SVG, PNG và `dashboard.html`.
- Xây dựng DoD validator [`experiments/validate_frozen_bert_results.py`](../../experiments/validate_frozen_bert_results.py) với 8 chốt chặn độc lập.

### 2. Tại sao phải làm (Why)
- **Đóng gói trọn vẹn mô hình (Full Artifact Parity):** Mặc dù BERT được freeze, mỗi checkpoint `after_T{t}/model.pt` vẫn là một mô hình PyTorch hoàn chỉnh, cho phép tái sử dụng hoặc probe biểu diễn bất cứ lúc nào.
- **Tự động hóa toàn diện:** Không cần can thiệp thủ công giữa các bước train, eval, plot và validate.

---

## Phase 4: 2-Task Smoke Test (T1 → T2 Verification)

### 1. Đã làm gì (What)
- Kích hoạt chạy Smoke Test 2 task:
  ```bash
  .venv/bin/python3 experiments/run_frozen_bert.py \
    --config configs/frozen_bert_linear_head_fewrel_5shot_seed2021.json \
    --num-tasks 2 \
    --output-dir results/fewrel/5shot/Frozen_BERT_smoke_test/seed_2021 \
    --device mps
  ```
- Thời gian chạy ~40 giây (caching 13,900 vectors trong 36s, train & eval trong 4s).
- Kết quả thu được:
  - Stage 1 (T1): Acc $82.87\%$, Macro-F1 $81.55\%$.
  - Stage 2: T2 đạt Acc $86.22\%$, nhưng T1 sụt giảm xuống $0.13\%$!
  - Kiểm tra 2 checkpoint `after_T1/model.pt` và `after_T2/model.pt` đều đạt 418 MB.

### 2. Tại sao phải làm (Why)
- **Xác thực đường ống (Pipeline Sanity Check):** Bảo đảm cơ chế Feature Caching, tính toán ma trận, và định dạng checkpoint hoạt động chính xác trước khi khởi chạy toàn bộ 8 task.
- **Phát hiện sớm hiện tượng classifier forgetting:** Ngay tại Task 2, độ sụt giảm từ $82.87\% \to 0.13\%$ chứng minh Classifier Interference là cực kỳ nghiêm trọng ngay cả khi encoder không bị drift.

---

## Phase 5: Full 8-Task Execution & Benchmark Triangulation

### 1. Đã làm gì (What)
- Kích hoạt chạy thực nghiệm toàn diện 8 task trên Apple Silicon GPU (`mps`):
  ```bash
  .venv/bin/python3 experiments/run_frozen_bert.py \
    --config configs/frozen_bert_linear_head_fewrel_5shot_seed2021.json \
    --device mps
  ```
- Hoàn thành toàn bộ quá trình trong 163 giây (feature caching 148s, train & eval 36 ô trong 15s).
- Kết quả thực nghiệm chính thức:
  - **Final Average Accuracy ($AA_8$):** **$10.33\%$**
  - **Final Macro-F1:** **$10.56\%$**
  - **Average Incremental Accuracy ($AIA$):** **$29.96\%$**
  - **Average Catastrophic Forgetting ($AF$):** **$86.27\%$**
  - **Backward Transfer ($BWT$):** **$-86.27\%$**
  - **Most Forgotten Task:** $T_4$ (sụt giảm tuyệt đối $92.38\%$, từ $92.38\%$ về $0.00\%$).
- Xác thực cổng DoD tự động qua [`experiments/validate_frozen_bert_results.py`](../../experiments/validate_frozen_bert_results.py):
  - **Kết quả:** `ALL 8 CHECKS PASSED: DEFINITION OF DONE SATISFIED FOR E003!`
- Chạy toàn bộ 69 test cases trong repo qua pytest đều vượt qua 100%.
- Đăng ký chính thức thực nghiệm **`E003`** trong `experiments/registry.yaml` và `registry.json` với `baseline_role: representation_probing`, `status: completed_real`.

### 2. Đối chiếu khoa học 3 mốc thực nghiệm (Triangulation):

| Chỉ số | E001 (B0 Full Sequential FT) | E003 (Frozen BERT + Linear Head) | E002 (Upper Bound Joint FT) |
| :--- | :---: | :---: | :---: |
| **Backbone Mode** | Fine-tuned theo từng task | **100% Frozen** | Fine-tuned đồng thời 80 lớp |
| **Final Average Accuracy ($AA_8$)** | $8.76\%$ | **$10.33\%$** | $52.12\%$ |
| **Final Macro-F1** | $9.69\%$ | **$10.56\%$** | $48.62\%$ |
| **Average Incremental Accuracy ($AIA$)** | $18.98\%$ | **$29.96\%$** | N/A (Joint) |
| **Average Catastrophic Forgetting ($AF$)** | $33.62\%$ | **$86.27\%$** | N/A |
| **Backward Transfer ($BWT$)** | $-33.62\%$ | **$-86.27\%$** | N/A |
| **Most Forgotten Task Drop** | T1 ($-41.45\%$) | **T4 ($-92.38\%$)** | N/A |

### 3. Ý nghĩa khoa học cốt lõi (Scientific Takeaway)
1. **Classifier Interference là thủ phạm chủ đạo gây xóa sạch trí nhớ cũ:** Khi BERT được freeze, classifier head đạt độ chính xác tức thời rất cao trên task mới ($New_t \approx 78\% - 92\%$), nhưng các hàng trọng số của các lớp cũ lập tức bị bias về các lớp mới khiến độ chính xác trên task cũ sụt gần như về $0.00\%$ ($Old_t < 1\%$).
2. **Khoảng cách so với Upper Bound ($52.12\%$):** Cho thấy cả B0 ($8.76\%$) và Frozen BERT Probing ($10.33\%$) đều thất bại thảm hại trong việc giữ lại tri thức quá khứ nếu không có cơ chế memory replay hoặc prototype alignment. Đây là minh chứng mạnh mẽ cho sự cần thiết của các phương pháp Continual Relation Extraction tiên tiến như TAPTA.

---
type: project
created: 2026-10-10
updated: 2026-10-10
---

# Frozen BERT + Nearest Prototype (Cosine NCM) Phase Execution Notes (What & Why)

Tài liệu này lưu vết chi tiết việc thiết kế, tối ưu hóa và nghiệm thu thực nghiệm **E004: Frozen BERT + Nearest Prototype (Cosine NCM)** cho FewRel Track A (80 quan hệ, 8 tasks, 5-shot, seed 2021). Mỗi phần đều ghi rõ **ĐÃ LÀM GÌ (What)** và **TẠI SAO PHẢI LÀM (Why)**.

---

## Phase 1: Protocol Alignment & Design Clarification (/grill-me)

### 1. Đã làm gì (What)
- Thực hiện phỏng vấn đối soát qua `/grill-me` để chốt trọn vẹn 7 nhánh quyết định:
  1. **Vai trò Khoa học:** Diagnostic baseline `E004` (`non_parametric_probing`) với 100% frozen BERT encoder và bộ phân loại nearest prototype phi tham số (non-parametric centroid).
  2. **Độ đo Tương đồng:** Cosine Similarity ($\arg\max_k \cos(f(x), c_k)$) trên vector marker thực thể $[h_{e1}; h_{e2}] \in \mathbb{R}^{1536}$ để loại bỏ ảnh hưởng của biên độ vector.
  3. **Cơ chế Prototype:** Strictly Static Centroid ($c_k = \frac{1}{|S_k|} \sum_{x \in S_k} f(x)$) tính một lần duy nhất từ 5 mẫu training khi quan hệ $k$ xuất hiện tại Task $t$, đóng băng vĩnh viễn không cập nhật.
  4. **Cấu trúc Checkpoint:** Đạt chuẩn Full Artifact Parity: mỗi task lưu `after_T{t}/model.pt` (418 MB, chứa BERT encoder + prototype bank tensor) và file nhẹ `after_T{t}/prototypes.pt` (~500 KB).
  5. **Chiến lược Tính toán:** Dual-mode Feature Caching (RAM + Disk tại `data/cache/features_fewrel_seed2021_frozen_bert.pt`) cho phép chạy lại thực nghiệm trong < 2 giây.
  6. **Định danh & Lưu trữ:** Mã thực nghiệm `E004`, role `non_parametric_probing`, lưu tại `results/fewrel/5shot/Frozen_BERT_nearest_prototype/seed_2021/`.
  7. **Chốt chặn Nghiệm thu:** 8 tiêu chí kiểm định tự động trong `validate_nearest_prototype_results.py`.
- Tạo quyết định kiến trúc: [`docs/adr/0004-frozen-bert-nearest-prototype.md`](../../docs/adr/0004-frozen-bert-nearest-prototype.md).
- Cập nhật từ điển miền: [`GLOSSARY.md`](../../GLOSSARY.md) bổ sung `Nearest Prototype Classifier (Frozen Backbone)` và `Cross-Task Prototype Ambiguity`.

### 2. Tại sao phải làm (Why)
- **Mổ xẻ nguyên nhân gây quên ở tầng sâu nhất:** E001 (B0) bị ảnh hưởng bởi cả Representation Drift và Classifier Interference ($AF = 33.62\%$). E003 (Frozen Probing) cô lập Classifier Interference ($AF = 86.27\%$). E004 loại bỏ hoàn toàn cả 2 yếu tố trên bằng cách dùng non-parametric centroid, từ đó đo lường chính xác sự suy giảm hiệu năng do sự chồng lấn hình học (Cross-Task Prototype Ambiguity) trong không gian vector tĩnh.

---

## Phase 2: Configuration & Dedicated Branch

### 1. Đã làm gì (What)
- Tạo dedicated branch `feat/frozen-bert-nearest-prototype`.
- Tạo file cấu hình:
  - [`configs/frozen_bert_nearest_prototype_fewrel_5shot_seed2021.json`](../../configs/frozen_bert_nearest_prototype_fewrel_5shot_seed2021.json)
  - [`configs/frozen_bert_nearest_prototype_fewrel_5shot_seed2021.yaml`](../../configs/frozen_bert_nearest_prototype_fewrel_5shot_seed2021.yaml)

### 2. Tại sao phải làm (Why)
- **Bảo toàn quy ước dự án:** Độc lập hóa các thay đổi lớn trên branch riêng theo `project-conventions.md`.
- **Cấu hình minh bạch, tái lập 100%:** Thiết lập contract đầy đủ về seed, backbone, purity boundary (`prototype: true`, `memory_size: 0`).

---

## Phase 3: Module, Runner & Validator Implementation

### 1. Đã làm gì (What)
- Xây dựng mô hình [`experiments/continual-relation-baselines/src/cl_re_baselines/prototype_model.py`](../../experiments/continual-relation-baselines/src/cl_re_baselines/prototype_model.py):
  - `BERTRelationPrototypeClassifier` quản lý buffer prototype $[80 \times 1536]$.
  - Hỗ trợ `compute_and_set_prototype`, `classify_features` với cosine similarity và seen-class masking ($-10^9$).
  - Tương thích giao thức `RelationPredictor` với hàm `predict()`.
- Xây dựng runner [`experiments/run_nearest_prototype.py`](../../experiments/run_nearest_prototype.py):
  - Tích hợp Dual-mode Feature Caching với lưu đĩa tự động.
  - Vòng lặp tích lũy prototype tuần tự qua từng task.
  - Đánh giá toàn bộ ma trận tam giác dưới và xuất metrics.
- Xây dựng DoD validator [`experiments/validate_nearest_prototype_results.py`](../../experiments/validate_nearest_prototype_results.py) với 8 chốt chặn độc lập.

### 2. Tại sao phải làm (Why)
- **Tách biệt kiến trúc & module sâu:** Tái sử dụng thành phần chuẩn của baseline suite mà không làm biến dạng mô hình huấn luyện.
- **Tốc độ thực thi tối đa:** Việc cache đĩa giúp bất kỳ lần test nào tiếp theo chạy trong vài giây.

---

## Phase 4: 2-Task Smoke Test (T1 → T2 Verification)

### 1. Đã làm gì (What)
- Kích hoạt chạy Smoke Test 2 task:
  ```bash
  .venv/bin/python3 experiments/run_nearest_prototype.py \
    --config configs/frozen_bert_nearest_prototype_fewrel_5shot_seed2021.json \
    --num-tasks 2 \
    --output-dir results/fewrel/5shot/Frozen_BERT_nearest_prototype_smoke/seed_2021 \
    --device mps
  ```
- Kết quả thu được:
  - Stage 1 (T1): Acc $83.45\%$, Macro-F1 $82.92\%$.
  - Stage 2: T2 đạt Acc $78.06\%$, T1 chỉ giảm nhẹ từ $83.45\% \to 76.48\%$ ($AF = 6.97\%$).
  - Checkpoint và đồ thị được tạo đầy đủ.

### 2. Tại sao phải làm (Why)
- **Xác thực logic toán học trước khi scale:** Khẳng định ngay rằng non-parametric prototype hoàn toàn không bị sụp đổ như linear head ở E003 ($82.87\% \to 0.13\%$).

---

## Phase 5: Full 8-Task Execution & Benchmark Triangulation

### 1. Đã làm gì (What)
- Kích hoạt chạy thực nghiệm toàn diện 8 task trên Apple Silicon GPU (`mps`):
  ```bash
  .venv/bin/python3 experiments/run_nearest_prototype.py \
    --config configs/frozen_bert_nearest_prototype_fewrel_5shot_seed2021.json \
    --device mps
  ```
- Kết quả thực nghiệm chính thức:
  - **Final Average Accuracy ($AA_8$):** **$59.63\%$**
  - **Final Macro-F1:** **$69.35\%$**
  - **Average Incremental Accuracy ($AIA$):** **$70.63\%$**
  - **Average Catastrophic Forgetting ($AF$):** **$10.45\%$**
  - **Backward Transfer ($BWT$):** **$-10.45\%$**
  - **Most Forgotten Task:** $T_1$ (sụt giảm $23.96\%$, từ $83.45\%$ về $59.49\%$).
- Xác thực cổng DoD tự động qua [`experiments/validate_nearest_prototype_results.py`](../../experiments/validate_nearest_prototype_results.py):
  - **Kết quả:** `ALL 8 CHECKS PASSED: DEFINITION OF DONE SATISFIED FOR E004!`
- Chạy toàn bộ 73 test cases trong repo qua pytest đều vượt qua 100%.
- Đăng ký chính thức thực nghiệm **`E004`** trong `experiments/registry.yaml` và `registry.json` với `baseline_role: non_parametric_probing`, `status: completed_real`.

### 2. Đối chiếu khoa học 4 mốc thực nghiệm (Triangulation Matrix):

| Chỉ số | E001 (B0 Full Sequential FT) | E003 (Frozen BERT + Linear Head) | E004 (Frozen BERT + Nearest Prototype) | E002 (Upper Bound Joint FT) |
| :--- | :---: | :---: | :---: | :---: |
| **Backbone Mode** | Fine-tuned tuần tự | 100% Frozen | **100% Frozen** | Fine-tuned đồng thời 80 lớp |
| **Phân loại (Classifier)** | Parametric Head ($W, b$) | Parametric Head ($W, b$) | **Cosine Centroid ($c_k$)** | Parametric Head ($W, b$) |
| **Final Average Accuracy ($AA_8$)** | $8.76\%$ | $10.33\%$ | **$59.63\%$** | $52.12\%$ |
| **Final Macro-F1** | $9.69\%$ | $10.56\%$ | **$69.35\%$** | $48.62\%$ |
| **Average Incremental Accuracy ($AIA$)** | $18.98\%$ | $29.96\%$ | **$70.63\%$** | N/A (Joint) |
| **Average Catastrophic Forgetting ($AF$)** | $33.62\%$ | $86.27\%$ | **$10.45\%$** | N/A |
| **Backward Transfer ($BWT$)** | $-33.62\%$ | $-86.27\%$ | **$-10.45\%$** | N/A |
| **Task 1 Final Score** | $0.00\%$ | $0.00\%$ | **$59.49\%$** | $62.00\%$ |

### 3. Ý nghĩa khoa học cốt lõi (Scientific Takeaway)
1. **Sức mạnh phi thường của Non-Parametric Prototypes trong Few-Shot:** Trong điều kiện 5-shot, Nearest Prototype trên pretrained BERT embeddings đạt $59.63\%$ $AA_8$, vượt qua cả Joint Training Upper Bound với Linear Head ($52.12\%$). Lý do là bộ phân loại tuyến tính tham số dễ bị overfit trên tập dữ liệu nhỏ (5 mẫu/lớp), trong khi centroid trung bình chuẩn hóa cosine duy trì ranh giới quyết định ổn định và khái quát hóa vượt trội.
2. **Chứng minh Forgetting chủ yếu do Classifier Overwriting:** Khi so sánh E003 ($AF = 86.27\%$) và E004 ($AF = 10.45\%$) trên cùng một backbone BERT đóng băng, mức quên giảm tới **$75.82\%$** tuyệt đối. Điều này chứng minh rằng thảm họa quên lãng trong probing chủ yếu đến từ việc cập nhật trọng số phân loại gradient làm biến dạng các lớp cũ, chứ không phải do không gian biểu diễn tĩnh của BERT không đủ sức chứa 80 quan hệ.

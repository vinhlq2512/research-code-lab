---
title: "[Tên Đề mục Thực nghiệm / Tính năng ML]"
task_id: "E00X" # hoặc mã feature: FEAT-XXX
type: "research_prd"
status: "draft" # draft | in_review | approved | in_progress | completed
author: "[Tên tác giả / PM / Researcher]"
created: "YYYY-MM-DD"
updated: "YYYY-MM-DD"
target_benchmark: "[Tên Benchmark, ví dụ: FewRel Track A, 8 tasks, 5-shot]"
assigned_agents:
  - "project-planner"
  - "backend-specialist"
  - "test-engineer"
related_adrs:
  - "docs/adr/000X-name.md"
---

# PRD: [Tên Đề mục Thực nghiệm / Tính năng ML]

> **Tài liệu Yêu cầu Sản phẩm & Thực nghiệm Khoa học (Research PRD)**  
> Bản đặc tả này đóng vai trò là hợp đồng kỹ thuật giữa nhóm nghiên cứu và các AI Coding Agent, đảm bảo tính chặt chẽ khoa học, khả năng tái lập 100% và tuân thủ các chốt chặn nghiệm thu (Definition of Done).

---

## 1. Tổng quan & Động lực nghiên cứu (Executive Summary & Problem Statement)

### 1.1 Bối cảnh & Vấn đề (Context & Problem)
- **Bối cảnh hiện tại:** [Mô tả ngắn gọn trạng thái hiện tại của codebase/dự án. Ví dụ: Đã có Baseline B0 (cận dưới) và Upper Bound E002, nhưng thiếu một mốc phân tích trung gian...]
- **Vấn đề cần giải quyết:** [Nêu rõ sự thiếu hụt hoặc nút thắt kỹ thuật. Tại sao phương pháp hoặc tính năng này lại cần thiết?]

### 1.2 Câu hỏi nghiên cứu then chốt (Core Research Question)
> *[Ví dụ: Nếu đóng băng 100% biểu diễn của BERT và chỉ huấn luyện linear head tuần tự, mức độ Catastrophic Forgetting xảy ra do đâu và có biên độ bao nhiêu so với Full Fine-Tuning?]*

### 1.3 Giả thuyết khoa học (Scientific Hypothesis)
- **Giả thuyết $H_1$:** [Dự đoán định lượng hoặc định tính về hành vi của mô hình. Ví dụ: Quên lãng phần lớn bắt nguồn từ Classifier Interference hơn là Representation Drift...]
- **Tiêu chí bác bỏ / chấp nhận:** [Điều kiện số liệu cụ thể để khẳng định hoặc bác bỏ giả thuyết].

---

## 2. Mục tiêu, Ranh giới & Ràng buộc Purity (Goals, Non-Goals & Invariants)

### 2.1 Mục tiêu chính (Primary Goals)
- [ ] **Goal 1:** [Xây dựng mô hình / đường ống dữ liệu / thuật toán X đáp ứng giao thức Y].
- [ ] **Goal 2:** [Thiết lập script runner độc lập hỗ trợ cấu hình qua file JSON/YAML].
- [ ] **Goal 3:** [Thu thập đủ ma trận kết quả thực nghiệm và vượt qua toàn bộ các cổng DoD].

### 2.2 Ngoài phạm vi (Non-Goals / Out-of-Scope)
- **Không thực hiện trong giai đoạn này:** [Liệt kê những gì cố tình KHÔNG làm để tránh lan man. Ví dụ: Chưa tối ưu hóa đa GPU; Chưa áp dụng memory buffer...]
- **Ranh giới công nghệ:** [Các kiến trúc hoặc dataset không thuộc phạm vi].

### 2.3 Ràng buộc Purity & Zero-Leakage (Strict Protocol Invariants)
> [!IMPORTANT]
> **Ràng buộc bất biến khoa học (Invariants):**
> 1. **Zero-Leakage Invariant:** [Ví dụ: Tuyệt đối không sử dụng mẫu dữ liệu từ các task quá khứ/tương lai trong quá trình huấn luyện task hiện tại ($M = 0$)].
> 2. **Label Space Invariant:** [ID của quan hệ/lớp $k$ phải giữ nguyên vẹn từ Task 1 đến Task $N$, không được re-index cục bộ].
> 3. **Model Instance Continuity:** [Sử dụng đúng 1 thể hiện mô hình truyền tiếp qua các task, không re-initialize trọng số].

---

## 3. Đặc tả Dữ liệu & Xử lý Đường ống (Data Requirements & Pipeline Spec)

### 3.1 Nguồn dữ liệu & Cấu trúc Phân chia (Dataset Splits)
| Thuộc tính | Quy định / Giá trị | Ghi chú kỹ thuật |
| :--- | :--- | :--- |
| **Dataset Name** | `[Tên dataset, ví dụ: FewRel]` | Đường dẫn: `data/raw/...` |
| **Track / Benchmark** | `[Ví dụ: Track A]` | Phân chia chuẩn theo tài liệu gốc |
| **Số lượng Task ($N$)** | `[Ví dụ: 8 tasks]` | Mỗi task gồm cố định $K$ lớp |
| **Số lớp mỗi Task ($K$)** | `[Ví dụ: 10 relations]` | Tổng số lớp: $N \times K = 80$ |
| **Train Split Budget** | `[Ví dụ: 5-shot (50 mẫu/task)]` | Lấy mẫu ngẫu nhiên có kiểm soát seed |
| **Val Split Budget** | `[Ví dụ: 5-shot (50 mẫu/task)]` | Dùng cho checkpointing / tuning |
| **Test Split Budget** | `[Ví dụ: 140 mẫu/quan hệ]` | Cố định xuyên suốt toàn bộ benchmark |
| **Random Seed** | `2021` (hoặc seed chỉ định) | Phải cô lập generator bằng `random.Random(seed)` |

### 3.2 Thứ tự Task & Ánh xạ Nhãn (Task Order & Label Space)
- **File thứ tự task:** `task-orders/[dataset]/order_seed_[seed].json`
- **Quy tắc phân chia:** Sắp xếp alphabet toàn bộ nhãn $\to$ Shuffle bằng isolated seed $\to$ Phân hoạch đều thành $N$ task không trùng lặp.
- **Ánh xạ toàn cục (Global Mapping):** `data/processed/[dataset]/relation_mapping.json`.

### 3.3 Tiền xử lý & Biểu diễn Thực thể (Representation / Feature Extraction)
- **Token Markers:** [Ví dụ: Chèn cặp entity markers `[E1]...[/E1]` và `[E2]...[/E2]` theo thứ tự vị trí giảm dần để không làm lệch index].
- **Feature Vector:** [Ví dụ: Concatenate hidden states tại vị trí `[E1]` và `[E2]`: $h_{rep} = [h_{e1}; h_{e2}] \in \mathbb{R}^{1536}$].

---

## 4. Kiến trúc Mô hình & Giao thức Thực nghiệm (Modeling & Protocol)

### 4.1 Cấu trúc Mạng nơ-ron
```text
[Input Sample: Tokens + Special Markers]
                 ↓
     [Backbone: Pretrained Encoder] (Frozen / Trainable)
                 ↓
[Representation Pooling: Markers / [CLS] / Mean Span]
                 ↓
         [Dropout (p = 0.1)]
                 ↓
[Classification Head: Linear Layer (Input_Dim -> Num_Classes)]
                 ↓
[Softmax with Seen-Class Masking: logits[:, seen_classes:] = -1e9]
```

### 4.2 Phạm vi Đóng băng & Huấn luyện (Freezing Scope)
| Thành phần | Trạng thái (Trainable / Frozen) | Số lượng tham số | Cơ chế thực thi |
| :--- | :--- | :--- | :--- |
| **Embedding Matrix** | `[Frozen / Trainable]` | `[e.g. 23.8M]` | `requires_grad = False` |
| **Transformer Layers** | `[Frozen / Trainable]` | `[e.g. 85.0M]` | `requires_grad = False`, `eval()` mode |
| **Classification Head** | `[Trainable]` | `[e.g. 122.9K]` | Tối ưu tuần tự bằng AdamW |

### 4.3 Mặt nạ Phân loại Tăng dần (Seen-Class Masking)
- Tại task thứ $t$ (0-indexed), số lượng lớp được phép dự đoán là:
  $$\text{seen\_classes} = (t + 1) \times K$$
- Toàn bộ logits của các lớp chưa xuất hiện ($j \ge \text{seen\_classes}$) bắt buộc phải gán giá trị $-10^9$ trước khi tính CrossEntropyLoss hoặc Softmax inference.

### 4.4 Siêu tham số Huấn luyện (Hyperparameter Specification)
```yaml
training:
  optimizer: "AdamW"
  learning_rate: 1.0e-3 # Tinh chỉnh theo loại kiến trúc
  weight_decay: 0.01
  batch_size: 16
  epochs_per_task: 15
  warmup_ratio: 0.1
  lr_scheduler: "linear"
  gradient_clip_norm: 1.0
  device: "auto" # mps, cuda, hoặc cpu
  feature_caching: true # true nếu backbone đóng băng 100%
```

---

## 5. Đo lường & Tiêu chuẩn Nghiệm thu Tự động (Evaluation & DoD Gate)

### 5.1 Các chỉ số khoa học bắt buộc (Metrics)
1. **Độ chính xác tích lũy giai đoạn $t$ ($AA_t$):**
   $$AA_t = \frac{1}{t+1} \sum_{j=0}^{t} A_{t,j}$$
2. **Final Average Accuracy ($FinalAA$):** Độ chính xác trung bình sau khi học xong toàn bộ $N$ task ($AA_{N-1}$).
3. **Average Incremental Accuracy ($AIA$):** Trung bình tích lũy qua tất cả các giai đoạn:
   $$AIA = \frac{1}{N} \sum_{t=0}^{N-1} AA_t$$
4. **Backward Transfer ($BWT$) / Quên lãng ($AF$):**
   $$BWT = \frac{1}{N-1} \sum_{j=0}^{N-2} (A_{N-1,j} - A_{j,j}), \quad AF = -BWT$$
5. **Độ suy giảm của từng Task ($F_j$) & Most Forgotten Task:**
   $$F_j = \max_{l \in \{j \dots N-1\}} A_{l,j} - A_{N-1,j}, \quad \text{Task bị quên nặng nhất} = \operatorname{argmax}_j F_j$$

### 5.2 Ràng buộc Ma trận Thực nghiệm ($A[t, j]$)
- Kích thước ma trận: đúng $N \times N$.
- Số ô tam giác dưới hợp lệ: đúng $\frac{N(N+1)}{2}$ ô có giá trị thực số trong $[0.0, 1.0]$ (ví dụ 8 task $\to$ đúng 36 ô).
- Số ô tương lai ($j > t$): đúng $\frac{N(N-1)}{2}$ ô bắt buộc phải là `null` / `None` (ví dụ 8 task $\to$ đúng 28 ô null).

### 5.3 Automated Definition of Done (DoD) Checklist
Script kiểm tra tự động phải xác thực đạt 100% (8/8 chốt chặn):
- [ ] **Check 1 (Config):** File config JSON/YAML khớp 100% protocol đã đóng băng.
- [ ] **Check 2 (Task Order):** Đủ $N$ task, phủ đủ tổng số lớp, các task rời rạc không giao nhau.
- [ ] **Check 3 (Matrix Validity):** Đủ các ô tam giác dưới trong $[0.0, 1.0]$, ô tương lai là `null`.
- [ ] **Check 4 (Streaming Log):** `metrics.jsonl` có đúng $\frac{N(N+1)}{2}$ dòng log hợp lệ.
- [ ] **Check 5 (Summary):** `summary.json` chứa đầy đủ $FinalAA, AIA, AF, BWT, MostForgotten$.
- [ ] **Check 6 (Conclusion):** File `conclusion.md` tồn tại và có số liệu phân tích thật.
- [ ] **Check 7 (Checkpoints):** Tồn tại đủ $N$ folder checkpoint kèm `stage_metadata.json` và file trọng số nhị phân `model.pt` ($> 10\text{ MB}$).
- [ ] **Check 8 (Plots):** Đủ 2 đồ thị PNG `accuracy_over_tasks.png` và `forgetting_curve.png` hợp lệ.

### 5.4 Lệnh kiểm thử & Nghiệm thu mẫu (CLI Commands)
```bash
# 1. Chạy Smoke Test (2 tasks)
.venv/bin/python3 experiments/run_[experiment_name].py --num-tasks 2 --output-dir results/.../smoke_test/

# 2. Chạy Full Benchmark (N tasks)
.venv/bin/python3 experiments/run_[experiment_name].py --config configs/[config_name].json --device [mps/cuda]

# 3. Chạy Automated DoD Gate
.venv/bin/python3 experiments/validate_[experiment_name]_results.py --results-dir results/.../

# 4. Chạy PyTest Regression Suite
.venv/bin/python3 -m pytest
```

---

## 6. Ngân sách Phần cứng & Tối ưu Tính toán (Compute Budget & Hardware)

| Hạng mục | Quy chuẩn đề xuất | Giới hạn tối đa (Budget Ceiling) |
| :--- | :--- | :--- |
| **Thiết bị chính** | Apple Silicon GPU (`mps`) hoặc CUDA | CPU fallback chỉ dùng khi CI/Test |
| **VRAM Consumption** | $\le 4.0\text{ GB}$ VRAM | Tránh OOM trên máy trạm cá nhân |
| **RAM Usage** | $\le 8.0\text{ GB}$ (khi dùng Feature Caching) | Giải phóng tensors sau khi cache |
| **Thời gian chạy Full Run**| $\le 15\text{ phút}$ (toàn bộ $N$ tasks) | Tối ưu bằng Feature Caching / Batching |
| **Dung lượng lưu trữ đĩa** | $\approx 3.5\text{ GB}$ ($N$ checkpoints `model.pt`) | Tự động loại trừ khỏi git bằng `.gitignore` |

---

## 7. Quản lý Artifacts & Đăng ký Sổ Thực nghiệm (Artifacts & Provenance)

### 7.1 Cấu trúc thư mục kết quả tiêu chuẩn
```text
results/[dataset]/[shot]shot/[Experiment_Dir]/seed_[seed]/
├── checkpoints/
│   ├── after_T1/ (stage_metadata.json, model.pt)
│   ├── ...
│   └── after_TN/ (stage_metadata.json, model.pt)
├── plots/
│   ├── accuracy_over_tasks.png (và .svg)
│   └── forgetting_curve.png (và .svg)
├── conclusion.md
├── dashboard.html
├── f1_performance_matrix.csv & .json
├── performance_matrix.csv & .json
├── metrics.jsonl
└── summary.json
```

### 7.2 Đăng ký Experiment Registry (`experiments/registry.yaml`)
Sau khi nghiệm thu DoD thành công, thêm bản ghi chính thức:
```yaml
experiments:
  - id: "E00X"
    title: "[Tên Đề mục Thực nghiệm]"
    method: "[method_code_name]"
    baseline_role: "[lower_bound | upper_bound | representation_probing | proposed_method]"
    execution_mode: "real_bert_mps"
    status: "completed_real"
    date: "YYYY-MM-DD"
    benchmark:
      dataset: "[FewRel Track A]"
      num_tasks: 8
      relations_per_task: 10
      total_relations: 80
      train_shot: 5
      seed: 2021
      backbone: "bert-base-uncased"
    results:
      final_average_accuracy: 0.XXXX
      average_incremental_accuracy: 0.XXXX
      backward_transfer: -0.XXXX
    validation:
      dod_validator: "experiments/validate_[name]_results.py"
      status: "PASSED"
```

---

## 8. Rủi ro, Giả định & Kế hoạch Dự phòng (Risks & Mitigations)

| Rủi ro kỹ thuật | Mức độ | Khả năng xảy ra | Giải pháp xử lý & Dự phòng (Mitigation) |
| :--- | :---: | :---: | :--- |
| **Hết bộ nhớ VRAM / MPS Memory Leak** | Cao | Thấp | Ép giải phóng bộ nhớ `torch.mps.empty_cache()` sau mỗi epoch/stage; sử dụng Feature Caching để không forward backbone nhiều lần. |
| **Underfitting trên Few-Shot (5-shot)** | Trung bình | Trung bình | Tinh chỉnh Learning Rate phù hợp (ví dụ $10^{-3}$ cho linear head thay vì $2\times 10^{-5}$ của full model); bổ sung warmup. |
| **Trôi dạt nhãn (Label Drift)** | Nghiêm trọng | Thấp | Khóa cố định bảng ánh xạ nhãn toàn cục `relation_mapping.json` và kiểm tra assertion trước khi huấn luyện. |
| **Lỗi crash gián đoạn thực nghiệm dài** | Trung bình | Trung bình | Áp dụng cơ chế Streaming Flush: sau mỗi task, ghi tức thời kết quả đánh giá ra `metrics.jsonl` và lưu checkpoint `model.pt`. |

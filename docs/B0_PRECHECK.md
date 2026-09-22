# B0 Precheck Audit Report (`B0_PRECHECK.md`)

> **Baseline:** B0 — Sequential Fine-Tuning  
> **Benchmark Protocol:** FewRel Track A, 8 tasks × 10 relations, 5-shot, seed = 2021  
> **Phase:** Phase 0 — Codebase Audit & Component Precheck  
> **Date:** 2026-09-21  
> **Assigned Agent:** `project-planner`  

---

## 1. Executive Summary & Component Status Table

Audit chi tiết codebase `research-code-lab` cho thấy: Hệ thống tiền xử lý dữ liệu và đánh giá liên tục (`dataset-pipelines/continual-relation-extraction`) đã được chuẩn hóa rất tốt, trong khi tầng mô hình nơ-ron và huấn luyện tuần tự (`experiments/continual-relation-baselines`) chưa tồn tại và cần được xây dựng mới hoàn toàn cho B0.

| Component | Status | Existing Location | Action for B0 Implementation |
| :--- | :--- | :--- | :--- |
| **FewRel Raw Data** | `EXISTING` | `dataset-pipelines/.../data/raw/fewrel/` | Tái sử dụng `train_wiki.json`, `val_wiki.json`, `pid2name.json` (700 samples/relation × 80 relations = 56,000 samples). |
| **Dataset Abstraction** | `EXISTING` | `src/datasets/base.py` (`ContinualRelationDataset`) | Tái sử dụng interface trừu tượng chuẩn. |
| **FewRel Loader** | `EXISTING` | `src/datasets/fewrel.py` (`FewRelDataset`) | Tái sử dụng loader với schema `RelationSample` (span nửa mở `[start, end)`). |
| **Relation Mapping** | `EXISTING` | `data/processed/fewrel/relation_mapping.json` | Cố định 80 relations $\to$ ID $0 \dots 79$ theo thứ tự bảng chữ cái. Bất biến qua mọi task. |
| **Task Order Schema** | `EXISTING` | `src/task_generation/task_order.py` | Tái sử dụng `TaskOrder` và hàm `create_task_order`. |
| **Task Order Seed 2021** | `CREATED` | `task-orders/fewrel/order_seed_2021.json` | Đã sinh và xác minh 8 tasks × 10 relations, disjoint, phủ đủ 80 relations. |
| **5-shot Sampler** | `EXISTING` | `src/cl_re_pipeline/build_tasks.py` | Đã có logic sample 5-shot train / 5-shot val; tích hợp vào task builder cho B0. |
| **Task Builder** | `EXISTING` | `src/task_generation/task_builder.py` | Tái sử dụng `ContinualTaskBuilder` để phân vùng train/val/test disjoint. |
| **Evaluator** | `REUSABLE` | `src/evaluation/evaluator.py` | Đạt chuẩn `RelationPredictor` protocol; tính Accuracy và Macro-F1 cho bất kỳ tập sample nào. |
| **Performance Matrix** | `EXISTING` | `src/evaluation/performance_matrix.py` | Quản lý ma trận $A[t, j]$ tam giác dưới, xuất/nạp CSV & JSON, phân biệt rõ `None` với `0.0`. |
| **Continual Metrics** | `PARTIALLY USABLE` | `src/evaluation/metrics.py` | Đã có $ACC_t$, $F_j$, $Avg\_F$. **Cần bổ sung:** $BWT$, $AIA$, ma trận Macro-F1, bảng Old/New task. |
| **Training Entrypoint** | `MISSING` | `experiments/` (hiện đang trống) | Cần tạo mới `experiments/run_sequential_ft.py`. |
| **Sequential FT Trainer** | `MISSING` | Chưa có trong repo | Cần xây dựng mới `sequential_trainer.py` (fine-tune tuần tự $T_1 \to T_8$, không reset backbone/classifier). |
| **Deep Learning Framework** | `MISSING IN VENV` | `.venv` (Python 3.11) | Cần cài đặt `torch`, `transformers`, `matplotlib` vào môi trường ảo. |
| **Checkpoint Manager** | `MISSING (MODEL)` | Chưa có cho PyTorch weights | Cần xây dựng module lưu/nạp checkpoint từng stage $T_1 \dots T_8$. |
| **Streaming Result Logger**| `MISSING` | Chưa có | Cần xây dựng logging trực tiếp ra `metrics.jsonl` và `summary.json`. |

---

## 2. Detailed Checklist Verification

### [x] 1. Tìm entrypoint training hiện tại
- **Kết quả:** **Chưa có (None)**. Thư mục `experiments/continual-relation-baselines/` hiện đang rỗng. Không có file huấn luyện nơ-ron nào trong toàn bộ repo.
- **Kế hoạch B0:** Xây dựng runner độc lập tại `experiments/run_sequential_ft.py`.

### [x] 2. Xác định dataset abstraction
- **Kết quả:** Lớp trừu tượng `ContinualRelationDataset` tại `dataset-pipelines/continual-relation-extraction/src/datasets/base.py`.
- **Phương thức chuẩn:**
  - `load_train() -> list[RelationSample]`
  - `load_validation() -> list[RelationSample]`
  - `load_test() -> list[RelationSample]`
  - `get_relations() -> list[str]`
  - `get_relation_to_id() -> dict[str, int]`
  - `load_split(split: str) -> list[RelationSample]`
- **Mẫu dữ liệu chuẩn:** `RelationSample` (`src/schemas/relation_sample.py`) đảm bảo span thực thể nửa mở $0 \le start < end \le len(tokens)$.

### [x] 3. Xác định FewRel loader hiện tại
- **Kết quả:** Lớp `FewRelDataset` tại `src/datasets/fewrel.py`.
- Đọc đồng thời `train_wiki.json` (64 quan hệ) và `val_wiki.json` (16 quan hệ) cùng `pid2name.json`.
- Tự động chuẩn hóa span vị trí `[start, end)` và bảo toàn metadata mô tả quan hệ Wikidata.

### [x] 4. Xác nhận Track A
- **Kết quả:** Đã xác nhận. FewRel Track A chuẩn mực sử dụng toàn bộ 80 quan hệ, không có lớp `no_relation`.

### [x] 5. Xác nhận số relation = 80
- **Kết quả:** Đã xác nhận. `len(dataset.get_relations()) == 80`.

### [x] 6. Xác nhận chia thành 8 task
- **Kết quả:** Đã xác nhận. Benchmark chuẩn là 8 task (khác với config cũ thử nghiệm 10 task).
- $80 \text{ relations} \div 8 \text{ tasks} = 10 \text{ relations/task}$.

### [x] 7. Mỗi task có 10 relation
- **Kết quả:** Đã xác nhận. Trong file `task-orders/fewrel/order_seed_2021.json` vừa sinh:
  - Task 0: 10 relations (`P39`, `P361`, `P150`, `P1344`, `P159`, `P1408`, `P551`, `P175`, `P466`, `P641`)
  - Task 1: 10 relations (`P931`, `P102`, `P495`, `P306`, `P22`, `P101`, `P178`, `P1346`, `P58`, `P86`)
  - Task 2: 10 relations (`P4552`, `P118`, `P59`, `P706`, `P40`, `P177`, `P1001`, `P800`, `P921`, `P974`)
  - Task 3: 10 relations (`P674`, `P264`, `P991`, `P400`, `P463`, `P3373`, `P206`, `P403`, `P449`, `P140`)
  - Task 4: 10 relations (`P137`, `P413`, `P176`, `P17`, `P3450`, `P6`, `P241`, `P1411`, `P407`, `P1877`)
  - Task 5: 10 relations (`P750`, `P276`, `P135`, `P127`, `P1923`, `P57`, `P412`, `P740`, `P105`, `P1435`)
  - Task 6: 10 relations (`P937`, `P364`, `P355`, `P136`, `P123`, `P156`, `P155`, `P131`, `P84`, `P27`)
  - Task 7: 10 relations (`P25`, `P31`, `P1303`, `P527`, `P460`, `P106`, `P2094`, `P26`, `P710`, `P410`)
  - Tổng số quan hệ: đúng 80, không có quan hệ nào bị lặp, không quan hệ nào bị thiếu.

### [x] 8. Xác định cách tạo 5-shot training set
- **Kết quả:** 
  - Mỗi relation trong FewRel có 700 samples.
  - Cần lấy chính xác **5 samples/relation** cho tập `train` (tổng 50 samples/task cho 10 relations).
  - Lấy **5 samples/relation** cho tập `val` (tổng 50 samples/task).
  - Có thể cấu hình trực tiếp qua `FewRelDataset(split_counts=(5, 5, 140))` hoặc sampling xác định theo seed 2021 trong task builder.

### [x] 9. Xác định validation/test set
- **Kết quả:**
  - `val`: 5 samples/relation (5-shot validation để early-stopping/model validation nếu cần).
  - `test`: 140 samples/relation (tổng 1,400 samples/task) theo chuẩn FewRel benchmark split, hoặc toàn bộ test split còn lại.

### [x] 10. Tìm file task order
- **Kết quả:** Đã sinh tại `dataset-pipelines/continual-relation-extraction/task-orders/fewrel/order_seed_2021.json`.
- Đã được validate bằng phương thức `TaskOrder.validate_against_expected_relations()` với 80 relations của FewRel.

### [x] 11. Xác định mapping `relation_name -> relation_id`
- **Kết quả:** Đã có tại `dataset-pipelines/continual-relation-extraction/data/processed/fewrel/relation_mapping.json`.
- Quy tắc: Sắp xếp 80 relation theo thứ tự chữ cái ($P1001 \to 0, P101 \to 1, \dots, P991 \to 79$).

### [x] 12. Kiểm tra mapping có cố định qua T1-T8 không
- **Kết quả:** **CỐ ĐỊNH HOÀN TOÀN (Invariant Guaranteed)**.
- Mapping là từ điển global duy nhất. Một relation $X$ có ID $k$ thì ở Task 1 hay Task 8 hay khi evaluate lại Task 1 ở stage 8, nhãn của $X$ luôn luôn là $k$. Không có local re-indexing.

### [x] 13. Tìm evaluator hiện tại
- **Kết quả:** Lớp `Evaluator` tại `src/evaluation/evaluator.py`.
- Triển khai phương thức:
  ```python
  evaluate(predictor: RelationPredictor, samples: Sequence[RelationSample]) -> EvaluationResult
  ```

### [x] 14. Kiểm tra evaluator có thể evaluate một task bất kỳ không
- **Kết quả:** **CÓ**. `evaluator.evaluate` nhận danh sách `RelationSample` bất kỳ.
- Sau stage $t$, hệ thống có thể gọi lặp qua $j \in \{0 \dots t\}$ với đầu vào là `task[j].test_samples`.

### [x] 15. Kiểm tra evaluator có lưu per-task accuracy/F1 không
- **Kết quả:**
  - `EvaluationResult` trả về đồng thời `accuracy`, `macro_f1`, `sample_count`, `predictions`, `labels`.
  - `PerformanceMatrix` (`src/evaluation/performance_matrix.py`) ghi nhận $A[t, j]$ cho từng task $j$ tại giai đoạn $t$.
  - Lưu ý: Cần khởi tạo 2 instance `PerformanceMatrix(metric="accuracy")` và `PerformanceMatrix(metric="macro_f1")` để lưu cả 2 ma trận song song.

### [x] 16. Kiểm tra seed propagation
- **Kết quả:**
  - Phần pipeline dữ liệu: Đã dùng `random.Random(seed)` cô lập trong `create_task_order` và `split_relation_order`.
  - Phần huấn luyện BERT sắp tới: Cần thiết lập hàm `set_seed(2021)` đồng bộ cho `random`, `numpy`, `torch` và `torch.backends.cudnn`.

### [x] 17. Kiểm tra checkpoint loading/saving
- **Kết quả:**
  - Dữ liệu ma trận & metric: Đã có sẵn `export_json`, `export_csv`, `load_json`, `load_csv` trong `PerformanceMatrix`.
  - Trọng số mô hình (PyTorch weights): Chưa có, sẽ được implement trong trainer của Phase 4 (`torch.save(model.state_dict(), ...)` và `torch.load(...)`).

---

## 3. Architecture Gap & Required Work for B0

Dựa trên kết quả kiểm tra, đây là các module cần xây dựng cho các Phase tiếp theo:

```text
research-code-lab/
├── dataset-pipelines/continual-relation-extraction/
│   ├── task-orders/fewrel/
│   │   └── order_seed_2021.json                   [READY]
│   └── src/evaluation/
│       └── metrics.py                             [EXTEND: BWT, AIA, Old/New]
├── experiments/
│   ├── configs/
│   │   └── b0_sequential_ft_seed2021.yaml         [PHASE 1]
│   ├── src/cl_re_baselines/
│   │   ├── model.py                               [PHASE 4: BERT + 80-class head]
│   │   ├── sequential_trainer.py                  [PHASE 4: Continual loop T1->T8]
│   │   ├── checkpoint.py                          [PHASE 5: Checkpoint manager]
│   │   ├── logger.py                              [PHASE 6: metrics.jsonl stream]
│   │   └── plotting.py                            [PHASE 7: Matplotlib plots]
│   ├── run_sequential_ft.py                       [PHASE 4: CLI Entrypoint]
│   ├── validate_b0_results.py                     [PHASE 8: Automated DoD validator]
│   └── registry.yaml                              [PHASE 11: E001 catalog]
```

---

## 4. Phase 0 Audit Conclusion

> **Kết luận:** Toàn bộ dữ liệu FewRel Track A (80 quan hệ), schema chuẩn `RelationSample`, thứ tự task `order_seed_2021.json` (8 tasks × 10 relations), và cấu trúc ma trận $A[t, j]$ đã sẵn sàng 100%. 
> 
> Hệ thống đủ điều kiện chuyển sang **Phase 1 (Freeze Experimental Protocol & Config)** và **Phase 3 (Evaluator Gate)**.

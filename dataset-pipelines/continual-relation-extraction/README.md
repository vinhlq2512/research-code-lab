# Continual Relation Extraction Dataset Pipeline & Evaluator

Hệ thống chuẩn bị luồng dữ liệu (task stream pipeline) và khung đánh giá học liên tục (Continual Learning Evaluation Suite) dùng chung, độc lập với kiến trúc mạng nơ-ron (model-agnostic) cho bài toán Continual Relation Extraction (CRE). 

Được thiết kế nhằm phục vụ việc tái lập và chuẩn hóa thực nghiệm cho các bài báo CRE như ConPL, CPL, WAVE-CRE, WAVE++, Continual-RE, và các baseline liên quan.

---

## 1. Kiến Trúc & Thiết Kế Tổng Thể

```
+-----------------------------------------------------------------------------------+
| 1. RAW DATASETS                                                                   |
|    - FewRel (train_wiki, val_wiki, pid2name) -> 56,000 samples, 80 relations      |
|    - TACRED (LDC licensed: train, dev, test) -> 42 relations, [subj/obj_start/end] |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| 2. CANONICAL SCHEMA & NORMALIZATION (src/schemas/relation_sample.py)               |
|    - RelationSample: tokens, relation, [start, end) token spans, head/tail types  |
|    - Half-open intervals: 0 <= start < end <= len(tokens)                          |
|    - Deterministic label mapping: Sorted alphabetically                           |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| 3. TASK GENERATION (src/task_generation/)                                         |
|    - TaskOrder (task_order.py): Seed-based disjoint relation partition            |
|    - ContinualTaskBuilder (task_builder.py): Split train/val/test per task        |
|    - Strict isolation check: No cross-task relation overlap                       |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| 4. MODEL-AGNOSTIC EVALUATOR & CONTINUAL METRICS (src/evaluation/)                 |
|    - Evaluator (evaluator.py): RelationPredictor Protocol -> Accuracy & Macro-F1  |
|    - PerformanceMatrix (performance_matrix.py): A[t, j] matrix (CSV/JSON export) |
|    - ContinualMetrics (metrics.py): ACC_t (Average Accuracy), F_j (Forgetting)    |
+-----------------------------------------------------------------------------------+
```

### Nguyên Tắc Thiết Kế Cốt Lõi
- **Zero Model Coupling**: Tuyệt đối không gắn chặt với mã nguồn PyTorch/HuggingFace/Transformers hoặc các thuật toán CL cụ thể (WAVE, ConPL). Bất kỳ predictor nào thỏa mãn giao thức `RelationPredictor` đều có thể cắm vào và đánh giá.
- **Strict Half-Open Token Spans**: Mọi entity span được chuẩn hóa về định dạng nửa mở $[start, end)$ chuẩn Python slice, kiểm tra chặt chẽ $0 \le start < end \le \text{len}(tokens)$.
- **Reproducibility**: Phân bổ quan hệ vào từng task dựa trên seed ngẫu nhiên độc lập (`random.Random(seed)`), xuất ra file task order JSON có thể versioning trên Git.
- **Explicit Missing Representation**: Ma trận hiệu năng $A[t, j]$ biểu diễn các ô chưa được đánh giá ($j > t$) tường minh dưới dạng `None` (`null` trong JSON, `""` rỗng trong CSV), tuyệt đối không gán `0.0` gây sai lệch thống kê.

---

## 2. Dữ Liệu Hỗ Trợ (Supported Datasets)

| Dataset | Số quan hệ | Số mẫu | Ghi chú tiền xử lý | Trạng thái |
|---------|------------|--------|---------------------|------------|
| **FewRel** | 80 relations | 56,000 | Chuẩn hóa span $[start, end + 1)$, gán nhãn `Pxxx` sang tên tự nhiên qua `pid2name.json`. | Sẵn sàng hoàn toàn |
| **TACRED** | 42 relations | 106,264 | Bản quyền LDC. Chuẩn hóa span từ inclusive sang $[start, end + 1)$. Có ~79.5% `no_relation`. | Đã kiểm định & có prototype loader |

### Cấu Trúc Thư Mục Dữ Liệu Khuyến Nghị
```text
dataset-pipelines/continual-relation-extraction/
├── data/
│   ├── raw/
│   │   ├── fewrel/
│   │   │   ├── train_wiki.json
│   │   │   ├── val_wiki.json
│   │   │   └── pid2name.json
│   │   └── tacred/
│   │       ├── train.json
│   │       ├── dev.json
│   │       └── test.json
│   └── processed/
│       ├── fewrel/
│       └── tacred/
```

> [!NOTE]
> Do vấn đề bản quyền học thuật, thư mục `data/raw/` và `data/processed/` được đưa vào `.gitignore`. Mã nguồn chỉ chứa các script tải công khai (cho FewRel) và kiểm định cấu trúc (cho TACRED).

---

## 3. Cài Đặt Môi Trường & Bắt Đầu Nhanh (Quick Start)

Yêu cầu môi trường: **Python $\ge$ 3.10** (khuyến nghị Python 3.11). Sử dụng 100% Python Standard Library, không yêu cầu cài thêm thư viện bên ngoài.

```bash
# Di chuyển vào thư mục subproject
cd dataset-pipelines/continual-relation-extraction

# Thiết lập PYTHONPATH chỉ vào src
export PYTHONPATH=src
```

### 3.1. Chạy Bộ Kiểm Thử Đơn Vị (Unit Tests)
```bash
python3 -m unittest discover -s tests -p "test_*.py"
```

### 3.2. Kiểm Tra & Thống Kê Dữ Liệu FewRel
```bash
python3 scripts/inspect_fewrel.py \
  --data-dir data/raw/fewrel \
  --split val_wiki.json
```

### 3.3. Sinh Task Order Ngẫu Nhiên Có Thể Tái Lập (Seed-based)
```bash
python3 scripts/generate_task_orders.py \
  --dataset fewrel \
  --data-dir data/raw/fewrel \
  --seeds 42 43 44 \
  --num-tasks 10 \
  --output-dir task-orders/fewrel
```

### 3.4. Kiểm Tra & Trực Quan Hóa Luồng Task (Task Stream Inspection)
```bash
python3 scripts/inspect_tasks.py \
  --dataset fewrel \
  --data-dir data/raw/fewrel \
  --task-order task-orders/fewrel/order_seed_42.json
```

### 3.5. Chạy Kiểm Định Toàn Trình (End-to-End Pipeline Validation)
```bash
python3 scripts/validate_pipeline.py \
  --dataset fewrel \
  --data-dir data/raw/fewrel \
  --task-order task-orders/fewrel/order_seed_42.json \
  --predictor perfect \
  --output-dir reports/continual-relation-extraction/fewrel/seed_42
```

### 3.6. Kiểm Tra Khả Thi & Kiểm Định Tiền Xử Lý TACRED
```bash
python3 scripts/inspect_tacred.py \
  --data-dir data/raw/tacred
```

---

## 4. Đặc Tả Ma Trận Hiệu Năng & Các Chỉ Số Continual Learning

### 4.1. Ma Trận Hiệu Năng $A[t, j]$

Ma trận $A[t, j]$ là chuẩn mực ghi nhận kết quả trong Continual Learning, biểu diễn hiệu năng của mô hình trên tập kiểm tra của **task $j$** ngay sau khi đã học xong **task $t$**.

```text
                     Task được kiểm tra (j)
                     Task 0      Task 1      Task 2      ...   Task T-1
Học xong Task 0 (t=0)   A[0,0]       None        None       ...     None
Học xong Task 1 (t=1)   A[1,0]      A[1,1]       None       ...     None
Học xong Task 2 (t=2)   A[2,0]      A[2,1]      A[2,2]      ...     None
...                     ...         ...         ...         ...     None
Học xong Task T-1       A[T-1,0]    A[T-1,1]    A[T-1,2]    ...    A[T-1,T-1]
```

- **Quy tắc nửa tam giác dưới (Lower Triangular)**: Tại bước $t$, mô hình chỉ đánh giá trên các task đã gặp ($j \le t$). Các ô ở tương lai ($j > t$) mặc định là `None`.
- **Cờ `allow_future_eval`**: Chỉ bật khi nghiên cứu Zero-shot Transfer sang các task tương lai.
- **Lưu trữ & Xuất file**:
  - `export_json()`: Các ô chưa đánh giá hiển thị dưới dạng `null`.
  - `export_csv()`: Các ô chưa đánh giá hiển thị dưới dạng chuỗi rỗng `""`.

### 4.2. Độ Chính Xác Trung Bình Tại Giai Đoạn $t$ ($ACC_t$)

Đo lường năng lực tổng quát trên toàn bộ các task từ $0$ đến $t$ sau khi học xong task $t$:
$$ACC_t = \frac{1}{t + 1} \sum_{j=0}^{t} A_{t, j}$$

Khi hoàn thành toàn bộ $T$ task, $ACC_{T-1}$ chính là **Final Average Accuracy** thường được báo cáo trong các bảng kết quả bài báo.

### 4.3. Độ Quên Của Từng Task ($F_j$)

Đo lường sự suy giảm hiệu năng lớn nhất của task $j$ sau khi mô hình đã học thêm các task tiếp theo:
$$F_j = \max_{l \in \{j, \dots, T-1\}} A_{l, j} - A_{T, j}$$

- Nếu $j = T - 1$ (task vừa học xong ở giai đoạn cuối): $F_{T-1} = 0.0$.
- $F_j \ge 0.0$: Giá trị càng nhỏ thể hiện mức độ duy trì tri thức càng cao (chống quên thảm khốc tốt).

### 4.4. Độ Quên Trung Bình ($\text{Avg\_F}$)

Đo độ quên trung bình trên toàn bộ $T$ task:
$$\text{Avg\_F} = \frac{1}{T} \sum_{j=0}^{T-1} F_j$$

---

## 5. Danh Mục File & Cấu Trúc Mã Nguồn

```text
dataset-pipelines/continual-relation-extraction/
├── src/
│   ├── schemas/
│   │   └── relation_sample.py        # Schema RelationSample (dataclass) với span validator
│   ├── datasets/
│   │   ├── base.py                   # ContinualRelationDataset Abstract Base Class
│   │   ├── fewrel.py                 # FewRelDataset loader & stable relation mapping
│   │   └── tacred.py                 # TACREDDataset loader & normalize_tacred_record
│   ├── task_generation/
│   │   ├── task_order.py             # Schema TaskOrder & hàm sinh create_task_order
│   │   └── task_builder.py           # ContinualTaskBuilder phân chia dữ liệu & cách ly task
│   └── evaluation/
│       ├── evaluator.py              # Evaluator & RelationPredictor protocol
│       ├── performance_matrix.py     # Lớp PerformanceMatrix A[t, j] & export CSV/JSON
│       ├── metrics.py                # Hàm tính ACC_t, F_j, Avg_F, summary dict
│       └── dummy_predictor.py        # Perfect, Constant, Random, Decaying Predictors
├── scripts/
│   ├── inspect_fewrel.py             # CLI kiểm tra thống kê FewRel
│   ├── inspect_tacred.py             # CLI kiểm tra & demo mapping TACRED
│   ├── generate_task_orders.py       # CLI sinh task order theo seed
│   ├── inspect_tasks.py              # CLI xem chi tiết phân bổ dữ liệu task stream
│   └── validate_pipeline.py          # Script xác thực kiểm định toàn trình (E2E)
├── task-orders/
│   └── fewrel/                       # File task order cố định (seed 42, 43, 44)
└── tests/
    ├── test_relation_sample.py       # Test xác thực span & bất biến dữ liệu
    ├── test_fewrel_loader.py         # Test nạp FewRel & mapping quan hệ
    ├── test_tacred_loader.py         # Test chuẩn hóa định dạng TACRED
    ├── test_task_order.py            # Test tính tất định của task order
    ├── test_task_builder.py          # Test cách ly nhãn giữa các task
    ├── test_performance_matrix.py    # Test lưu trữ & xuất ma trận A[t, j]
    └── test_metrics.py               # Test công thức toán học ACC_t & F_j
```

---

## 6. Ghi Chú Kỹ Thuật & Giới Hạn (Known Notes & Limitations)

1. **TACRED Licensing**:
   - Dữ liệu TACRED thuộc quyền sở hữu của LDC (Linguistic Data Consortium - LDC2018T24). Repo không phân phối sẵn file dữ liệu này. Người dùng cần đặt các file `train.json`, `dev.json`, `test.json` vào thư mục `data/raw/tacred/`.
   - Script `scripts/inspect_tacred.py` tự động phát hiện nếu dữ liệu thật bị thiếu và cung cấp mock data để kiểm thử toàn bộ luồng xử lý và span conversion.
2. **Xử lý nhãn `no_relation` trong TACRED**:
   - TACRED có nhãn âm tính `no_relation` chiếm ~79.5% tổng số mẫu.
   - Trong thiết lập Continual Relation Extraction chuẩn, các bài báo thường nghiên cứu sự xuất hiện tuần tự của các quan hệ dương tính (positive emerging relations). Mặc định `include_no_relation=False` trong `TACREDDataset` để tránh việc nhãn `no_relation` áp đảo phân phối của từng task. Người dùng có thể bật lại tùy chọn này khi thực nghiệm yêu cầu.
3. **Cơ chế tách k-shot**:
   - Đối với FewRel, mỗi quan hệ có đúng 700 mẫu (thường chia 420 train, 140 val, 140 test). Trong thiết lập k-shot (ví dụ 5-shot, 10-shot), `ContinualTaskBuilder` hỗ trợ lấy mẫu con ngẫu nhiên theo seed xác định.

---

## 7. Tài Liệu Kỹ Thuật Tham Khảo Thêm
- Tài liệu quy hoạch kiến trúc: [`docs/continual-re-pipeline.md`](../../docs/continual-re-pipeline.md)
- Báo cáo kiểm định TACRED: [`docs/datasets/tacred-preprocessing-check.md`](../../docs/datasets/tacred-preprocessing-check.md)
- Chi tiết ma trận $A[t, j]$: [`docs/notes/performance-matrix-notes.md`](../../docs/notes/performance-matrix-notes.md)
- Chi tiết các chỉ số Continual Learning: [`docs/notes/continual-metrics-and-dummy-predictors.md`](../../docs/notes/continual-metrics-and-dummy-predictors.md)
- Tổng quan bộ kiểm thử đơn vị: [`docs/notes/test-suite-overview.md`](../../docs/notes/test-suite-overview.md)

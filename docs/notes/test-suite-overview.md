# Tổng Quan Bộ Kiểm Thử (Phase 17 — FewRel & Continual Pipeline Tests)

**Ngày ghi nhận:** 2026-09-17  
**Vị trí lưu trữ:** `docs/notes/test-suite-overview.md`  
**Lệnh thực thi kiểm thử:**
```bash
cd dataset-pipelines/continual-relation-extraction
PYTHONPATH=src python3.11 -m unittest discover -s tests -p "test_*.py"
```

---

## 1. Mục Đích & Tiêu Chuẩn Nghiệm Thu (Definition of Done)

Bộ kiểm thử của Continual Relation Extraction pipeline được thiết kế với mục tiêu: **bất kỳ sự thay đổi mã nguồn nào trong tương lai (refactoring, thêm baseline, đổi tokenizer) đều không thể phá vỡ tính tái lập thực nghiệm hay ngữ nghĩa đánh giá khoa học.**

Tổng số test hiện tại: **46 unit tests**, chạy hoàn tất trong **~1.3 giây** với 0 dependency ngoài (chạy bằng thư viện chuẩn `unittest` trên Python 3.11).

---

## 2. Bản Đồ Phủ Kiểm Thử Chi Tiết (Test Coverage Matrix)

| Nhóm Kiểm Thử | File Test | Số Tests | Các Hành Vi Cốt Lõi Được Kiểm Chứng |
| :--- | :--- | :---: | :--- |
| **1. Canonical Schema** | [`tests/test_relation_sample.py`](../../dataset-pipelines/continual-relation-extraction/tests/test_relation_sample.py) | **7** | - Tính hợp lệ của span nửa mở $[start, end)$<br>- Chặn span âm, span đảo ngược ($start \ge end$), span vượt độ dài câu<br>- Chặn nhãn rỗng và relation ID âm<br>- Tính bất biến (`frozen=True`) và serialize `to_dict()` / `from_dict()` |
| **2. Base Dataset Interface** | [`tests/test_base_dataset.py`](../../dataset-pipelines/continual-relation-extraction/tests/test_base_dataset.py) | **4** | - Ngăn chặn khởi tạo lớp trừu tượng (ABC invariant)<br>- Đảm bảo các lớp con cài đặt đủ `load_train`, `load_val`, `load_test`<br>- Xử lý alias (`train`, `val`, `dev`, `test`) qua `load_split()`<br>- Đảo ngược mapping nhãn `get_id_to_relation()` |
| **3. FewRel Loader** | [`tests/test_fewrel_loader.py`](../../dataset-pipelines/continual-relation-extraction/tests/test_fewrel_loader.py) | **4** | - Parse chính xác dữ liệu thô FewRel thành `RelationSample`<br>- Chuẩn hóa đúng vị trí entity và Wikidata ID<br>- Ánh xạ 80 quan hệ sang ID xác định theo bảng chữ cái<br>- `sample_id` và `relation_mapping` giống nhau $100\%$ qua nhiều lần load độc lập<br>- Kiểm thử trên 56.000 mẫu FewRel thật (0 span lỗi) |
| **4. Task Order** | [`tests/test_task_order.py`](../../dataset-pipelines/continual-relation-extraction/tests/test_task_order.py) | **6** | - Cùng seed luôn sinh ra thứ tự task giống nhau $100\%$<br>- Khác seed sinh ra thứ tự khác nhau<br>- Xử lý chia dư số quan hệ cho số task đồng đều<br>- Tự động phát hiện trùng lặp quan hệ giữa các task<br>- Phát hiện thiếu hoặc thừa quan hệ so với tập quan hệ gốc<br>- Xuất và nạp JSON round-trip |
| **5. Task Builder** | [`tests/test_task_builder.py`](../../dataset-pipelines/continual-relation-extraction/tests/test_task_builder.py) | **4** | - Phân hoạch dữ liệu rời rạc theo từng quan hệ (Class-Incremental)<br>- Phát hiện và chặn đứng rò rỉ quan hệ (`Relation leakage`)<br>- Phát hiện task rỗng (`allow_empty_tasks=False`)<br>- Kiểm thử thực tế trên 8 task FewRel (mỗi task 10 quan hệ, 4.200 train, 1.400 val, 1.400 test) |
| **6. Generic Evaluator** | [`tests/test_evaluator.py`](../../dataset-pipelines/continual-relation-extraction/tests/test_evaluator.py) | **5** | - `PerfectPredictor` đạt đúng Accuracy = 1.0 và Macro F1 = 1.0<br>- Dự đoán không hoàn hảo khớp đúng số tính tay (Acc = 0.75, Macro F1 = 11/15)<br>- Xử lý an toàn tập test rỗng (không crash, không chia cho 0)<br>- Bắt lỗi lệch số lượng nhãn dự đoán (`predictions != labels`) |
| **7. Performance Matrix $A[t, j]$** | [`tests/test_performance_matrix.py`](../../dataset-pipelines/continual-relation-extraction/tests/test_performance_matrix.py) | **8** | - Khởi tạo ma trận $T \times T$ với giá trị `None`<br>- Phân biệt rõ ô chưa đo (`None` / `null` / `""`) với điểm số 0.0<br>- Chặn việc đánh giá task tương lai ($j > t$) trong continual learning<br>- Chặn các index ngoài biên<br>- Trích xuất hàng và cột<br>- Xuất và nạp round-trip cho cả CSV và JSON |
| **8. Continual Metrics & Dummies** | [`tests/test_metrics.py`](../../dataset-pipelines/continual-relation-extraction/tests/test_metrics.py) | **8** | - Kiểm thử toán học đối sánh với ma trận tính tay trong đặc tả:<br>  + $ACC_0 = 0.90, ACC_1 = 0.825, ACC_2 \approx 0.7933$<br>  + $F_0 = 0.20, F_1 = 0.05, F_2 = 0.0$<br>  + $\text{Avg\_F} = 0.125$<br>- Bắt lỗi khi ma trận bị thiếu cell cần thiết<br>- Kiểm thử hành vi của `PerfectPredictor`, `ConstantPredictor`, `DecayingPredictor` |

---

## 3. Cách Thức Chạy Kiểm Thử

### Chạy từ thư mục gốc của repository (`research-code-lab`):
```bash
PYTHONPATH=dataset-pipelines/continual-relation-extraction/src python3.11 -m unittest discover -s dataset-pipelines/continual-relation-extraction/tests -p "test_*.py"
```

### Chạy từ thư mục subproject:
```bash
cd dataset-pipelines/continual-relation-extraction
PYTHONPATH=src python3.11 -m unittest discover -s tests -p "test_*.py"
```

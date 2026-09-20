# TACRED Preprocessing Check

**Tài liệu kiểm định tính khả thi và tiền xử lý TACRED (Phase 20)**  
**Mã nguồn liên quan:**
- [`src/datasets/tacred.py`](../../dataset-pipelines/continual-relation-extraction/src/datasets/tacred.py)
- [`scripts/inspect_tacred.py`](../../dataset-pipelines/continual-relation-extraction/scripts/inspect_tacred.py)

---

## 1. Dataset Access

- **Trạng thái (Status):** Manual access required (Cần tải thủ công có bản quyền)
- **Nguồn dữ liệu (Source):** Linguistic Data Consortium (LDC) / The Stanford NLP Group
- **Mã bản quyền LDC:** LDC Catalog No. [LDC2018T24](https://catalog.ldc.upenn.edu/LDC2018T24)
- **Hướng dẫn đưa dữ liệu vào repo:** Sau khi có giấy phép LDC, tải các file JSON gốc và đặt vào thư mục:
  `dataset-pipelines/continual-relation-extraction/data/raw/tacred/`
  bao gồm: `train.json`, `dev.json`, `test.json`.
- **Lưu ý Git Hygiene:** Dữ liệu raw của TACRED đã được cấu hình trong `.gitignore` (`**/data/raw/`), đảm bảo tuyệt đối không vi phạm bản quyền khi commit code.

---

## 2. Raw Dataset Structure

TACRED được chia sẵn thành 3 phân tập chuẩn (fixed splits):

| Phân tập (Split) | Tên file | Số lượng mẫu (Samples) | Tỷ lệ (%) |
| :--- | :--- | :---: | :---: |
| **Train** | `train.json` | 68,124 | 64.1% |
| **Validation (Dev)** | `dev.json` | 22,631 | 21.3% |
| **Test** | `test.json` | 15,509 | 14.6% |
| **Tổng cộng** | | **106,264** | **100%** |

Khác với FewRel (80 quan hệ cân bằng, mỗi quan hệ 700 mẫu), TACRED là tập dữ liệu thực tế trích xuất từ tin tức báo chí, có độ lệch phân bố rất lớn (imbalanced distribution).

---

## 3. Sample Schema

Mỗi bản ghi trong file JSON gốc của TACRED có cấu trúc:

```json
{
  "id": "e77ee47f482d881e191b",
  "relation": "org:founded_by",
  "token": [
    "At", "the", "same", "time", ",", "Chief", "Executive", "Steve", "Jobs",
    "co-founded", "Apple", "Computer", "in", "1976", "."
  ],
  "subj_start": 10,
  "subj_end": 11,
  "subj_type": "ORGANIZATION",
  "obj_start": 7,
  "obj_end": 8,
  "obj_type": "PERSON",
  "stanford_ner": ["...", "PERSON", "ORGANIZATION", "..."],
  "stanford_pos": ["...", "NNP", "VBD", "..."]
}
```

- **`token`**: Danh sách các từ (tokens) đã được tokenizer của Stanford CoreNLP phân tách.
- **`subj_start`, `subj_end`**: Vị trí token của thực thể chủ ngữ (Subject / Head).
- **`obj_start`, `obj_end`**: Vị trí token của thực thể tân ngữ (Object / Tail).
- **`relation`**: Chuỗi định danh quan hệ (ví dụ: `org:founded_by`, `per:title`, `no_relation`).

---

## 4. Entity Span Semantics

### Điểm khác biệt chí mạng về chỉ số span:
- **Trong dữ liệu raw của TACRED:** Các chỉ số `subj_start`, `subj_end`, `obj_start`, `obj_end` là **khoảng đóng (inclusive interval) $[start, end]$**.
  *Ví dụ:* `subj_start = 10`, `subj_end = 11` đại diện cho cả 2 token ở vị trí 10 và 11 (`['Apple', 'Computer']`).
- **Trong Canonical Schema `RelationSample` của pipeline:** Toàn bộ hệ thống quy định sử dụng **khoảng nửa mở (half-open interval) $[start, end)$**.
- **Công thức ánh xạ chuẩn hóa:**
  $$\text{head\_start} = \text{subj\_start}$$
  $$\text{head\_end} = \text{subj\_end} + 1$$
  $$\text{tail\_start} = \text{obj\_start}$$
  $$\text{tail\_end} = \text{obj\_end} + 1$$
  *Sau khi ánh xạ:* $[10, 12)$ cho phép slice trực tiếp trong Python: `tokens[10:12] == ['Apple', 'Computer']`.

---

## 5. Xử Lý Nhãn `no_relation`

### Hiện trạng quan sát (Observed Behavior):
- TACRED có tổng cộng **42 nhãn quan hệ**: gồm 41 nhãn quan hệ thực sự (positive relations) và 1 nhãn đặc biệt là **`no_relation`** (âm tính - hai thực thể không có quan hệ trong ngữ cảnh).
- Nhãn `no_relation` chiếm tỷ trọng áp đảo: **~79.5% tổng số mẫu** (khoảng 84.400 / 106.264 mẫu).

### Quyết định thực nghiệm cho Continual Learning (Decision):
1. **Mặc định loại trừ `no_relation` khỏi Task Stream (`include_no_relation = False`):**
   Trong các bài báo Continual Relation Extraction (CRE) chuẩn như ConPL, CPL, WAVE-CRE, các task liên tục được mô hình hóa dựa trên các lớp quan hệ tích cực mới xuất hiện tuần tự (Emerging Positive Classes). Nếu đưa `no_relation` vào một task cụ thể, task đó sẽ bị áp đảo hoàn toàn và làm lệch phân bố continual learning.
2. **Cung cấp cờ cấu hình linh hoạt:**
   Loader `TACREDDataset` cung cấp tham số `include_no_relation: bool = False`. Nếu một nghiên cứu cụ thể yêu cầu bài toán phát hiện mở (open-world / out-of-distribution detection), có thể bật `include_no_relation=True` mà không cần sửa code.
3. **Tuyệt đối không gộp hoặc gán nhầm:**
   Không bao giờ tự ý gộp `no_relation` vào một nhãn quan hệ khác.

---

## 6. Canonical Conversion (TACRED $\to$ `RelationSample`)

Bộ chuyển đổi đã được triển khai hoàn chỉnh tại hàm `normalize_tacred_record()` trong [`src/datasets/tacred.py`](../../dataset-pipelines/continual-relation-extraction/src/datasets/tacred.py):

| Trường trong TACRED Raw | Trường trong Canonical `RelationSample` | Cơ chế chuyển đổi |
| :--- | :--- | :--- |
| `id` | `sample_id` | `tacred_{split}_{id}` (xác định, không dùng random UUID) |
| `token` | `tokens` | `list(item["token"])` |
| `subj_start`, `subj_end` | `head_start`, `head_end` | `[subj_start, subj_end + 1)` |
| Trích xuất từ tokens | `head_text` | `" ".join(tokens[subj_start : subj_end + 1])` |
| `obj_start`, `obj_end` | `tail_start`, `tail_end` | `[obj_start, obj_end + 1)` |
| Trích xuất từ tokens | `tail_text` | `" ".join(tokens[obj_start : obj_end + 1])` |
| `relation` | `relation` | Giữ nguyên chuỗi quan hệ chuẩn |
| Tra cứu từ bảng ánh xạ | `relation_id` | ID số nguyên cố định theo thứ tự Alphabet |
| `subj_type`, `obj_type`, `ner` | `metadata` | Giữ nguyên siêu dữ liệu gốc phục vụ phân tích sâu |

---

## 7. Khả Năng Tương Thích Với Continual Task Builder

- **Đánh giá mức độ hỗ trợ (Compatibility):** **YES (Hoàn toàn tương thích)**
- **Lý do:**
  - `TACREDDataset` kế thừa trực tiếp từ `ContinualRelationDataset` (cùng abstract base class với `FewRelDataset`).
  - Cung cấp đầy đủ các phương thức chuẩn: `load_train()`, `load_validation()`, `load_test()`, `get_relations()`, `get_relation_to_id()`.
  - Bộ sinh `ContinualTaskBuilder` ở Phase 9 có thể nhận `TACREDDataset` và một file `task-orders/tacred/order_seed_*.json` để chia các task học liên tục mà không cần sửa bất kỳ dòng code nào.

---

## 8. Các Vấn Đề Cần Lưu Ý Khi Triển Khai Thực Nghiệm (Open Issues)

1. **Mất cân bằng mẫu giữa các quan hệ (Imbalanced Class Problem):**
   Khác với FewRel có đúng 700 mẫu cho mỗi quan hệ, trong TACRED:
   - Có quan hệ phổ biến với hơn 2.000 mẫu (ví dụ `per:title`).
   - Có quan hệ hiếm chỉ có 20–30 mẫu (ví dụ `per:charges`, `org:dissolved`).
   - *Khuyến nghị:* Khi tạo continual task $K$-shot trên TACRED, cần cấu hình số lượng shot phù hợp (ví dụ $k=5$ hoặc $k=10$) và bật cờ `allow_undersampled` nếu số lượng mẫu của một số quan hệ hiếm không đủ.
2. **Kích thước tập Test:**
   Tập test của TACRED cố định (15.509 mẫu). Khi đánh giá continual learning trên các task có quan hệ hiếm, macro F1 sẽ phản ánh chính xác năng lực cân bằng giữa các lớp hơn là micro Accuracy.

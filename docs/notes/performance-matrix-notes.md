# Ghi Chú Kỹ Thuật: Ma Trận Hiệu Năng $A[t, j]$ & Xuất Báo Cáo (Phase 12 & Phase 13)

**Ngày ghi nhận:** 2026-09-15  
**Phạm vi:** Hệ thống đánh giá Continual Relation Extraction — `src/evaluation/performance_matrix.py`

---

## 1. Vấn Đề Cốt Lõi: Tại Sao Cần Ma Trận $A[t, j]$?

Trong học máy truyền thống, ta chỉ đánh giá mô hình một lần trên tập test và thu được một giá trị metric duy nhất (ví dụ: Accuracy tổng thể).

Tuy nhiên, trong **Continual Learning (Học liên tục / Học tăng cường)**:
- Mô hình phải tiếp nhận và học tuần tự qua nhiều giai đoạn: học xong **Task 0**, tiếp tục học **Task 1**, rồi **Task 2**, ..., cho tới Task $T-1$.
- **Thách thức cốt lõi:** Hiện tượng **Catastrophic Forgetting** (Quên thảm khốc) — khi mô hình tiếp thu kiến thức của task mới, trọng số mạng nơ-ron thay đổi khiến mô hình quên đi các kiến thức của các task cũ đã học trước đó.
- Do đó, sau khi kết thúc việc huấn luyện ở mỗi giai đoạn $t$, ta bắt buộc phải **kiểm tra lại mô hình trên TẤT CẢ các task đã từng gặp từ $0$ đến $t$**.

Tập hợp các kết quả đánh giá qua từng giai đoạn huấn luyện tạo thành một ma trận tam giác dưới, ký hiệu chuẩn mực trong các bài báo khoa học là **$A[t, j]$**.

---

## 2. Ý Nghĩa Toán Học & Cấu Trúc Của $A[t, j]$

- **$t$ (Chỉ số HÀNG):** Mô hình **đã hoàn thành huấn luyện đến task nào** (từ task $0$ đến task $t$).
- **$j$ (Chỉ số CỘT):** Task **đang được đem ra kiểm tra (test)**.
- **$A[t, j]$:** Điểm số đánh giá (Accuracy hoặc Macro F1) trên task $j$ sau khi mô hình đã học xong task $t$.

### Bảng minh họa trực quan:

```text
                     Task được test (j)
                     Task 0      Task 1      Task 2      Task 3
Học xong Task 0 (t=0)   0.84        ---         ---         ---
Học xong Task 1 (t=1)   0.79        0.86        ---         ---
Học xong Task 2 (t=2)   0.73        0.81        0.85        ---
Học xong Task 3 (t=3)   0.68        0.76        0.82        0.87
```

### Hai thông tin sống còn được trích xuất từ ma trận:
1. **Đọc theo từng HÀNG (ngang) — Average Accuracy ($ACC_t$):**
   Đo lường năng lực trung bình của mô hình trên toàn bộ các task đã học tính đến thời điểm $t$:
   $$ACC_t = \frac{1}{t + 1} \sum_{j=0}^{t} A_{t, j}$$
   *Ví dụ ở hàng $t=3$:* $ACC_3 = \frac{0.68 + 0.76 + 0.82 + 0.87}{4} = 0.7825$ (78.25%).

2. **Đọc theo từng CỘT (dọc) — Catastrophic Forgetting ($F_j$):**
   Theo dõi sự sụt giảm hiệu năng của task $j$ qua thời gian. Mức độ quên của task $j$ tại thời điểm kết thúc $T$ là độ chênh lệch giữa điểm số cao nhất từng đạt được và điểm số ở thời điểm cuối cùng:
   $$F_j = \max_{l \in \{j, \dots, T-1\}} A_{l, j} - A_{T, j}$$
   *Ví dụ tại cột Task 0 ($j=0$):* Điểm số rơi từ $0.84 \to 0.79 \to 0.73 \to 0.68$.
   Mức độ quên của Task 0 là: $0.84 - 0.68 = 0.16$ (quên 16%).

---

## 3. Những Việc Đã Triển Khai Trong Phase 12

File mã nguồn: [`src/evaluation/performance_matrix.py`](../../dataset-pipelines/continual-relation-extraction/src/evaluation/performance_matrix.py)

1. **Khởi tạo lớp `PerformanceMatrix`**:
   - Quản lý mảng 2 chiều kích thước $T \times T$ (với $T$ là số lượng task continual learning).
   - Khởi tạo toàn bộ các cell bằng `None`.

2. **Quy tắc bất biến: Dùng `None`, tuyệt đối không dùng số `0.0` cho ô chưa đo**:
   - Điểm `0.0` là một giá trị toán học thực tế (mô hình dự đoán sai toàn bộ).
   - Một ô chưa học hoặc task tương lai ($j > t$) là "chưa có dữ liệu" (Missing Value).
   - Nếu dùng `0.0` thay cho `None`, khi tính trung bình cộng hoặc độ quên sẽ bị sai lệch nghiêm trọng.

3. **Cơ chế chặn lỗi thực nghiệm (Sanity Check / Invariant Guard)**:
   - Trong Continual Learning chuẩn, mô hình không thể kiểm tra trên task tương lai mà nó chưa từng thấy ($j > t$).
   - Nếu phương thức `set_score(t, j, score)` nhận $j > t$ mà không có cờ cho phép rõ ràng (`allow_future_eval=True`), hệ thống sẽ ném ngoại lệ `ValueError` để bảo vệ tính hợp lệ khoa học của nghiên cứu.
   - Kiểm tra chặt chẽ các chỉ số index nằm trong khoảng $[0, T)$.

4. **API truy xuất dữ liệu**:
   - `get_row(t)`: Lấy mảng kết quả của toàn bộ các task tại thời điểm $t$.
   - `get_column(j)`: Lấy quỹ đạo điểm số của task $j$ qua các mốc huấn luyện để tính Forgetting.
   - `is_complete(t)`: Kiểm tra xem toàn bộ các cell tam giác dưới đã được ghi nhận đầy đủ chưa.

---

## 4. Những Việc Đã Triển Khai Trong Phase 13

Lưu trữ và xuất/nạp kết quả đánh giá ra các định dạng chuẩn phục vụ báo cáo khoa học và trực quan hóa dữ liệu:

1. **Xuất file CSV (`export_csv`, `to_csv_string`)**:
   - Các ô chưa học được biểu diễn dưới dạng chuỗi rỗng `""` (không phải số 0):
     ```csv
     trained_until,task_0,task_1,task_2,task_3
     0,0.840000,,,
     1,0.790000,0.860000,,
     2,0.730000,0.810000,0.850000,
     3,0.680000,0.760000,0.820000,0.870000
     ```
   - Định dạng này tương thích hoàn toàn với Excel, Google Sheets, Pandas, Seaborn và Matplotlib để vẽ biểu đồ nhiệt (Heatmap) hoặc đường cong quên lãng (Forgetting Curves).

2. **Xuất file JSON (`export_json`, `to_dict`)**:
   - Lưu trữ ma trận cùng đầy đủ siêu dữ liệu tái lập thí nghiệm:
     ```json
     {
       "dataset": "fewrel",
       "seed": 42,
       "num_tasks": 4,
       "metric": "accuracy",
       "task_order_path": "task-orders/fewrel/order_seed_42.json",
       "matrix": [
         [0.84, null, null, null],
         [0.79, 0.86, null, null],
         [0.73, 0.81, 0.85, null],
         [0.68, 0.76, 0.82, 0.87]
       ]
     }
     ```
   - Các ô chưa học được serialize chuẩn thành `null`.

3. **Cơ chế nạp ngược lại (`load_csv`, `load_json`)**:
   - Đảm bảo tính toàn vẹn 2 chiều (Round-trip integrity): có thể đọc lại các file đã lưu từ các lần chạy trước để so sánh giữa các thuật toán baseline khác nhau mà không cần chạy lại thực nghiệm từ đầu.

---

## 5. Kết Luận & Vai Trò Trong Toàn Bộ Pipeline

Ma trận $A[t, j]$ là **cầu nối trực tiếp** giữa:
- Quá trình thực thi mô hình (Model Execution / Evaluator ở Phase 11)
- Và các chỉ số tổng hợp cốt lõi của nghiên cứu (Continual Metrics: Average Accuracy & Forgetting ở Phase 14).

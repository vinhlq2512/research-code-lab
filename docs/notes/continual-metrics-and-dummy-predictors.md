# Ghi Chú Kỹ Thuật: Các Chỉ Số Continual Learning & Bộ Dự Đoán Giả Lập (Phase 14 & Phase 15)

**Ngày ghi nhận:** 2026-09-16  
**Vị trí lưu trữ:** `docs/notes/continual-metrics-and-dummy-predictors.md`  
**Mã nguồn liên quan:** 
- [`src/evaluation/metrics.py`](../../dataset-pipelines/continual-relation-extraction/src/evaluation/metrics.py)
- [`src/evaluation/dummy_predictor.py`](../../dataset-pipelines/continual-relation-extraction/src/evaluation/dummy_predictor.py)
- [`tests/test_metrics.py`](../../dataset-pipelines/continual-relation-extraction/tests/test_metrics.py)

---

## 1. Tổng Quan Mục Đích

Trong bài toán Continual Learning (Học liên tục / Học tăng cường), việc lưu ma trận kết quả $A[t, j]$ ở Phase 12–13 mới chỉ là bước trung gian. Để đánh giá một phương pháp Continual Relation Extraction (CRE) có hiệu quả hay không, cộng đồng nghiên cứu (như trong các paper ConPL, CPL, WAVE, Continual-RE) dựa trên 2 tiêu chuẩn đo lường cốt lõi:
1. **Khả năng duy trì tri thức tổng thể:** Mô hình làm tốt đến đâu trên tất cả các task đã học? $\to$ **Average Accuracy ($ACC$)**.
2. **Khả năng chống quên thảm khốc:** Mô hình bị suy giảm bao nhiêu phần trăm kiến thức cũ sau khi học thêm các task mới? $\to$ **Catastrophic Forgetting ($F$)**.

Đồng thời, ở **Phase 15**, để kiểm thử toàn bộ hệ thống đánh giá mà **không cần chờ huấn luyện mạng nơ-ron thật** (như BERT, RoBERTa tốn hàng giờ/hàng ngày), ta thiết kế các **Dummy Predictor** có hành vi toán học xác định.

---

## 2. Chi Tiết Các Chỉ Số Toán Học (Phase 14)

Giả sử quá trình Continual Learning gồm $T$ task được đánh giá theo thứ tự: $0, 1, \dots, T-1$.  
Ký hiệu $A[t, j]$ là hiệu năng (Accuracy hoặc Macro F1) trên task $j$ sau khi mô hình đã học xong task $t$ ($j \le t$).

```text
                     Task được test (j)
                     Task 0      Task 1      Task 2
Học xong Task 0 (t=0)   A[0,0]       ---         ---
Học xong Task 1 (t=1)   A[1,0]      A[1,1]       ---
Học xong Task 2 (t=2)   A[2,0]      A[2,1]      A[2,2]
```

---

### 2.1. Độ Chính Xác Trung Bình Tại Giai Đoạn $t$ ($ACC_t$ - Average Accuracy)

#### Ý nghĩa:
Đo lường năng lực trung bình của mô hình trên toàn bộ các task từ $0$ đến $t$ ngay sau khi vừa hoàn thành huấn luyện task $t$.

#### Công thức toán học:
$$ACC_t = \frac{1}{t + 1} \sum_{j=0}^{t} A_{t, j}$$

Trong đó:
- $t$: Chỉ số giai đoạn huấn luyện hiện tại ($t \in \{0, \dots, T-1\}$).
- $j$: Chỉ số task được đánh giá ($j \in \{0, \dots, t\}$).
- $t + 1$: Tổng số lượng task mà mô hình đã học tính đến thời điểm $t$.
- $A_{t, j}$: Điểm số trên task $j$ tại thời điểm $t$.

#### Đặc điểm quan trọng:
- $ACC_t$ là chỉ số theo dõi theo thời gian thực (real-time trajectory). Một phương pháp CL tốt sẽ giữ $ACC_t$ giảm rất ít khi $t$ tăng dần.
- $ACC_{T-1}$ (tại bước cuối cùng) được gọi là **Final Average Accuracy** — con số chính được đưa vào bảng so sánh kết quả trong các bài báo khoa học.

---

### 2.2. Độ Quên Của Từng Task ($F_j$ - Task Forgetting)

#### Ý nghĩa:
Đo lường sự sụt giảm hiệu năng lớn nhất của một task cụ thể $j$ sau khi mô hình tiếp tục học các task tiếp theo từ $j+1$ đến $T-1$.

#### Công thức toán học:
$$F_j = \max_{l \in \{j, \dots, T-1\}} A_{l, j} - A_{T, j}$$

Trong đó:
- $j$: Task cần đo mức độ quên ($j < T$).
- $l$: Các mốc thời gian từ lúc vừa học xong task $j$ ($l = j$) đến trước hoặc bằng thời điểm cuối ($l \le T-1$).
- $\max_{l \in \{j, \dots, T-1\}} A_{l, j}$: Hiệu năng đỉnh cao nhất mà task $j$ từng đạt được trong quá khứ.
- $A_{T, j}$: Hiệu năng của task $j$ tại thời điểm hiện tại / kết thúc $T$.

#### Lưu ý trường hợp biên:
- Nếu $j = T$ (task vừa mới học xong ở giai đoạn hiện tại), task này chưa trải qua giai đoạn học thêm nào tiếp theo, nên theo định nghĩa:
  $$F_T = 0.0$$

---

### 2.3. Độ Quên Trung Bình ($\text{Avg\_F}$ - Average Forgetting)

#### Ý nghĩa:
Tổng hợp mức độ quên của toàn bộ các task cũ thành một con số duy nhất để đánh giá thuật toán chống quên tốt đến đâu.

#### Công thức toán học:
$$\text{Avg\_F} = \frac{1}{T} \sum_{j=0}^{T-1} F_j$$

- Nếu $T = 1$ (mới chỉ học duy nhất 1 task): $\text{Avg\_F} = 0.0$.
- Giá trị $\text{Avg\_F}$ càng nhỏ (càng gần 0), thuật toán continual learning chống quên càng hoàn hảo.

---

## 3. Ví Dụ Tính Toán Bằng Tay Từng Bước (Worked Numerical Example)

Xét ma trận thực nghiệm chuẩn gồm 3 task ($T = 3$) trong file test [`tests/test_metrics.py`](../../dataset-pipelines/continual-relation-extraction/tests/test_metrics.py):

$$\text{Matrix } A = \begin{bmatrix} 
0.90 & \text{None} & \text{None} \\ 
0.80 & 0.85 & \text{None} \\ 
0.70 & 0.80 & 0.88 
\end{bmatrix}$$

### Bước 1: Tính Average Accuracy cho từng giai đoạn ($ACC_t$)
- **Giai đoạn $t = 0$:**
  $$ACC_0 = \frac{A[0, 0]}{1} = \frac{0.90}{1} = \mathbf{0.9000}$$
- **Giai đoạn $t = 1$:**
  $$ACC_1 = \frac{A[1, 0] + A[1, 1]}{2} = \frac{0.80 + 0.85}{2} = \frac{1.65}{2} = \mathbf{0.8250}$$
- **Giai đoạn $t = 2$ (Final Average Accuracy):**
  $$ACC_2 = \frac{A[2, 0] + A[2, 1] + A[2, 2]}{3} = \frac{0.70 + 0.80 + 0.88}{3} = \frac{2.38}{3} \approx \mathbf{0.7933}$$

---

### Bước 2: Tính Độ Quên của từng task ($F_j$) tại giai đoạn cuối ($T = 2$)
- **Task 0 ($j = 0$):**
  - Quá khứ task 0 đạt: $A[0, 0] = 0.90$ và $A[1, 0] = 0.80$.
  - Đỉnh cao nhất: $\max(0.90, 0.80) = 0.90$.
  - Hiện tại ở $T = 2$: $A[2, 0] = 0.70$.
  - Mức độ quên:
    $$F_0 = 0.90 - 0.70 = \mathbf{0.2000} \quad (20\%)$$
- **Task 1 ($j = 1$):**
  - Quá khứ task 1 đạt: $A[1, 1] = 0.85$.
  - Hiện tại ở $T = 2$: $A[2, 1] = 0.80$.
  - Mức độ quên:
    $$F_1 = 0.85 - 0.80 = \mathbf{0.0500} \quad (5\%)$$
- **Task 2 ($j = 2$):**
  - Vừa học xong tại stage 2:
    $$F_2 = \mathbf{0.0000}$$

---

### Bước 3: Tính Average Forgetting ($\text{Avg\_F}$)
Trung bình độ quên của các task cũ (Task 0 và Task 1):
$$\text{Avg\_F} = \frac{F_0 + F_1}{2} = \frac{0.20 + 0.05}{2} = \frac{0.25}{2} = \mathbf{0.1250} \quad (12.5\%)$$

Toàn bộ các con số trên được lập trình kiểm thử tự động trong `tests/test_metrics.py` và đều vượt qua với sai số $< 10^{-7}$.

---

## 4. Thiết Kế & Vai Trò Của Dummy Predictors (Phase 15)

File mã nguồn: [`src/evaluation/dummy_predictor.py`](../../dataset-pipelines/continual-relation-extraction/src/evaluation/dummy_predictor.py)

### 4.1. Tại sao phải xây dựng Dummy Predictor?
- Trong công nghệ phần mềm và nghiên cứu khoa học, **kiểm thử hạ tầng (Infrastructure Testing)** cần tách biệt hoàn toàn khỏi **chất lượng mô hình (Model Quality)**.
- Ta không thể đợi train xong một mô hình BERT (rất tốn tài nguyên và thời gian) rồi mới đi kiểm tra xem code tính ma trận và metrics có bị crash hay không.
- Dummy Predictor là các bộ phân loại mô phỏng, nhẹ, thực thi trong tích tắc và cho ra kết quả dự đoán có thể suy luận chính xác trước bằng toán học.

### 4.2. Các loại Dummy Predictor được triển khai:

1. **`PerfectPredictor` (Oracle / Ground-Truth Echo):**
   ```python
   class PerfectPredictor:
       def predict(self, samples: Sequence[RelationSample]) -> list[int]:
           return [sample.relation_id for sample in samples]
   ```
   - **Tác dụng:** Luôn trả về đúng $100\%$ nhãn thật của từng mẫu.
   - **Kỳ vọng:** Độ chính xác Accuracy = 1.0 và Macro F1 = 1.0. Dùng để xác nhận evaluator hoạt động hoàn hảo khi dữ liệu dự đoán hoàn toàn đúng.

2. **`ConstantPredictor`:**
   - Luôn dự đoán về một nhãn cố định duy nhất (ví dụ class 0).
   - Dùng để kiểm tra trường hợp phân lớp mất cân bằng hoặc dự đoán thiên lệch.

3. **`DecayingPredictor` (Mô phỏng hiện tượng quên theo thời gian):**
   ```python
   def predict_for_task(self, samples, trained_until, evaluated_task):
       lag = trained_until - evaluated_task  # Độ trễ thời gian
       target_acc = max(0.10, self.base_accuracy - (lag * self.decay_rate))
       ...
   ```
   - **Tác dụng:** Giả lập hành vi thực tế của mạng nơ-ron trong Continual Learning: khi khoảng cách giữa task hiện tại và task cũ càng xa ($lag = t - j$ càng lớn), xác suất dự đoán đúng càng giảm dần theo tỷ lệ `decay_rate`.
   - **Kỳ vọng:** Tạo ra một ma trận tam giác dưới $A[t, j]$ suy giảm thực tế để kiểm tra toàn trình đường ống validation ở Phase 16 mà không cần GPU.

---

## 5. Tóm Tắt Liên Kết Hệ Thống

```text
[Dataset: FewRel / TACRED]
          ↓
[TaskBuilder: Continual Tasks (Train / Val / Test)]
          ↓
[DummyPredictor / Real Neural Model]  <--- Tuân thủ Protocol RelationPredictor
          ↓
[Evaluator: Tính Accuracy & Macro F1 cho từng Task j]
          ↓
[PerformanceMatrix: Lưu vào ô A[t, j] & Xuất CSV / JSON]
          ↓
[Metrics: Tính Final ACC & Catastrophic Forgetting]
```
Mọi thành phần đều liên kết chặt chẽ qua các giao ước trừu tượng, độc lập và dễ dàng thay thế mà không gây lỗi dây chuyền.

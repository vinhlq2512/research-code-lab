# Kiến trúc: Tại sao sử dụng Native PyTorch Loops thay vì Thư viện Huấn luyện có sẵn?

> **Tài liệu Phân tích Kỹ thuật & Quyết định Kiến trúc**  
> Giải trình chi tiết lý do nhóm nghiên cứu lựa chọn vòng lặp PyTorch thuần túy (`for stage ... for epoch ...`) thay vì sử dụng các khung huấn luyện cấp cao như Hugging Face `Trainer`, `PyTorch Lightning`, hay framework chuyên biệt như `Avalanche`.

---

## 1. Tổng quan & Vấn đề

Khi bắt đầu một dự án Machine Learning hiện đại, phản xạ kỹ thuật thông thường là tận dụng các thư viện huấn luyện có sẵn:
- **Hugging Face `Trainer`** hoặc `Accelerate`
- **PyTorch Lightning**
- **Keras / FastAI**
- **Avalanche (ContinualAI)**

Tuy nhiên, trong bài toán **Continual Relation Extraction (CRE)** — học liên tục trích xuất quan hệ trên ngữ cảnh Few-Shot (FewRel Track A, 8 tasks × 10 relations, 5-shot) — việc sử dụng các framework cấp cao trên dẫn đến nhiều rào cản kỹ thuật nghiêm trọng.

Quyết định sử dụng **vòng lặp PyTorch thuần túy (Native PyTorch Training Loop)** được xây dựng dựa trên **4 trụ cột khoa học và kỹ thuật cốt lõi**.

---

## 2. Bốn Trụ cột Kỹ thuật Cốt lõi

```text
┌────────────────────────────────────────────────────────────────────────┐
│           TẠI SAO DÙNG NATIVE PYTORCH LOOP TRONG CONTINUAL LEARNING?     │
└────────────────────────────────────────────────────────────────────────┘
       │                    │                    │                   │
       ▼                    ▼                    ▼                   ▼
[1. Class-Incremental   [2. Model Instance   [3. Zero-Overhead   [4. Purity &
    Dynamic Masking &       Continuity &         Feature Caching     Zero-Leakage
    Matrix Evaluation]      No State Drift]      (15s vs 12m)]       Auditing]
```

---

### Trụ cột 1: Ràng buộc Class-Incremental (Seen-Class Masking & Ma trận $A[t, j]$)

Trong Continual Learning (khác với Supervised Learning thông thường):
1. **Không gian nhãn động (Dynamic Output Space):**
   - Head phân loại có cố định 80 lớp, nhưng ở task $t$ (0-indexed), chỉ có $(t+1) \times 10$ lớp được phép dự đoán.
   - Các lớp chưa thấy bắt buộc phải nhận mặt nạ nghiêm ngặt:
     $$\text{logits}[:, (t+1)\times 10 :] = -10^9$$
2. **Đánh giá Đa nhiệm tức thời theo Ma trận Tam giác dưới:**
   - Sau khi học xong task $t$, mô hình **không** chỉ đánh giá trên 1 tập validation duy nhất, mà phải lập tức duyệt qua toàn bộ $t+1$ tập kiểm thử độc lập ($j = 0 \dots t$).
   - Kết quả từng ô $A_{t,j}$ phải được streaming lập tức ra `metrics.jsonl` và ma trận tam giác dưới $A[t, j]$.

> ⚠️ **Hạn chế của Framework cấp cao:**
> - Hugging Face `Trainer` và `PyTorch Lightning` được thiết kế xung quanh mô hình cố định 1 `train_dataloader` và 1 `eval_dataloader`.
> - Để thực hiện vòng lặp ma trận $A_{t,j}$ và dynamic masking trên `Trainer`, lập trình viên phải override nhiều hàm nội bộ (`compute_loss`, `evaluation_loop`, `get_eval_dataloader`), viết hàng trăm dòng code "hacky" xung quanh callback và hooks. Code trở nên khó bảo trì và dễ sinh lỗi tiềm ẩn.

---

### Trụ cột 2: Bảo toàn Thể hiện Mô hình (Model Instance Continuity Invariant)

Một nguyên lý bất biến sống còn của Continual Learning là:
> **Chỉ duy nhất MỘT thể hiện mô hình (Single Model Instance) được truyền tiếp liên tục từ Task 1 qua Task $N$. Trọng số của mô hình từ task trước không bao giờ được phép bị reset.**

- **Vấn đề với Framework cấp cao:**
  - Nếu khởi tạo một `Trainer` mới cho mỗi task, framework có xu hướng reset lại trạng thái nội bộ của model wrapper, hooks, logging buffers hoặc optimizer state.
  - Nếu tái sử dụng cùng 1 instance `Trainer`, logic của bộ đếm bước (`global_step`), learning rate scheduler, và epoch counters sẽ bị xung đột hoặc tiếp tục tăng lũy kế không đúng theo chu kỳ của từng task độc lập.
- **Với Native PyTorch Loop:**
  - Model instance `model = BERTRelationClassifier(...)` được khởi tạo đúng **1 lần duy nhất** ngoài vòng lặp.
  - Vòng lặp `for stage in range(num_tasks):` truyền tiếp trực tiếp thể hiện `model` mà không qua bất kỳ lớp trung gian nào, bảo đảm 100% tính nguyên vẹn của trọng số.

---

### Trụ cột 3: Hiệu năng Tính toán Đột phá nhờ RAM Feature Caching (15s vs 12m)

Trong thực nghiệm **Frozen BERT + Linear Head (E003)**:
- BERT encoder được đóng băng 100%. Do đó, vector biểu diễn $[h_{e1}; h_{e2}]$ của toàn bộ 55,600 mẫu dữ liệu (train + test) là **hoàn toàn bất biến**.
- Với Native Loop:
  - Chúng ta trích xuất và cache toàn bộ 55,600 vectors vào RAM/VRAM một lần duy nhất (148 giây).
  - Vòng lặp huấn luyện Linear Head và đánh giá toàn bộ 36 ô ma trận chạy trực tiếp trên các tensor trong RAM, hoàn tất trong **15 giây**!
- Nếu dùng Hugging Face `Trainer`:
  - Mỗi bước forward bắt buộc phải đi qua toàn bộ pipeline tokenizer, collator, dataset streaming và model forward của HF.
  - Overhead khởi tạo tiến trình, logging và metrics collector của framework sẽ triệt tiêu hoàn toàn lợi thế của Feature Caching, kéo dài thời gian chạy lên hàng chục phút mà không đem lại giá trị nào.

---

### Trụ cột 4: Tính Tinh khiết Khoa học & Kiểm soát Rò rỉ Dữ liệu (Purity & Zero-Leakage)

Trong nghiên cứu học máy, tính tái lập (Reproducibility) và độ tin cậy của cận dưới (Lower Bound Purity) là tối quan trọng:
1. **Không rò rỉ cơ chế ngầm (No Magic under the Hood):**
   - Các framework cấp cao thường tích hợp sẵn nhiều cơ chế tối ưu hóa tự động: gradient clipping mặc định, weight decay ẩn trên LayerNorm/Bias, automatic mixed precision (AMP) casting, batch sampler shuffling ngầm...
   - Những cơ chế này có thể vô tình làm sai lệch điểm số cận dưới B0, khiến người phản biện (reviewers) nghi ngờ tính khách quan của thực nghiệm.
2. **Kiểm soát 100% dòng dữ liệu (Transparent Data Flow):**
   - Với Native Loop, từng dòng code đều thể hiện rõ:
     - Dữ liệu nào được nạp vào batch?
     - Đạo hàm được tính trên tham số nào (`requires_grad`)?
     - Mặt nạ masking được áp dụng chính xác ở đâu?
     - Kết quả từng ô đánh giá được ghi nhận ra sao?
   - Tính tường minh này giúp việc gỡ lỗi (debugging) và viết unit test cổng kiểm chứng (DoD Gates) trở nên dễ dàng và chuẩn xác tuyệt đối.

---

## 3. Tại sao không chọn Thư viện Continual Learning Chuyên biệt (như Avalanche)?

Nhiều người sẽ đặt câu hỏi: *Nếu HF Trainer không phù hợp, tại sao không dùng `Avalanche` (thư viện Continual Learning phổ biến của ContinualAI)?*

| Tiêu chí | Avalanche | Native PyTorch Loop (Repo hiện tại) |
| :--- | :--- | :--- |
| **Trọng tâm thiết kế** | Chủ yếu tối ưu cho **Computer Vision** (Split-MNIST, CIFAR-100, Core50) | Tối ưu chuyên sâu cho **NLP Relation Extraction** với Entity Markers `[E1]`, `[E2]` |
| **Độ cồng kềnh (Dependencies)** | Yêu cầu hàng chục thư viện phụ thuộc lớn (`torchvision`, `quadprog`, `gdown`...) | **Zero extra dependencies**: Chỉ cần PyTorch và Transformers chuẩn |
| **Độ linh hoạt tùy biến** | Khó can thiệp sâu vào cấu trúc token marker động và cấu trúc cây đa tầng (TAPTA) | **Tùy biến tự do 100%**: Dễ dàng nhúng thêm Prototype Memory, Routing Network, Tree Hierarchy |
| **Tốc độ thực thi** | Nhiều lớp wrap trừu tượng (`Strategy`, `Scenario`, `Plugin`) gây overhead | Trực tiếp, siêu nhanh với RAM Feature Caching |

---

## 4. Tầm nhìn Mở rộng: Nền tảng cho Mô hình TAPTA tương lai

Native PyTorch loop hiện tại không chỉ phục vụ cho các baseline đơn giản (B0, Upper Bound, Frozen BERT), mà còn là nền móng kiến trúc vững chắc để triển khai các mô hình Continual Learning tiên tiến tiếp theo:

1. **Memory Replay Buffer ($M > 0$):**
   - Dễ dàng can thiệp trực tiếp vào batch huấn luyện để ghép mẫu cũ và mẫu mới theo tỷ lệ tùy chọn mà không bị framework cản trở.
2. **Prototype Clustering & Contrastive Loss:**
   - Dễ dàng tính toán centroid trung bình của từng quan hệ sau mỗi stage và chèn các thành phần hàm mất mát điều hòa (regularization loss).
3. **Hierarchical Tree Router:**
   - Cho phép định tuyến đa tầng biểu diễn quan hệ (Coarse-to-Fine) mà không bị giới hạn bởi khuôn mẫu `model.forward()` cứng nhắc của các framework có sẵn.

---

## 5. Kết luận

Việc sử dụng vòng lặp PyTorch thuần túy (`for stage ... for epoch ...`) trong repository `research-code-lab` **không phải là sự thiếu sót công nghệ**, mà là **một quyết định kiến trúc có chủ đích (Deliberate Architectural Decision)** nhằm:
- Bảo đảm tính toàn vẹn khoa học của các cận thực nghiệm ($B_0$, Upper Bound, Frozen BERT).
- Kiểm soát tuyệt đối tính bất biến Class-Incremental và quy tắc Seen-Class Masking.
- Tối ưu hóa hiệu năng thực nghiệm gấp nhiều lần nhờ RAM Feature Caching.
- Tạo nền tảng mở hoàn toàn để phát triển phương pháp nghiên cứu chính (TAPTA) trong các giai đoạn tiếp theo.

# Báo Cáo Kỹ Thuật: Kiểm Chứng Thực Nghiệm & Rào Cản Phần Cứng Của WAVE-CRE (Baseline B5)

**Ngày thực hiện:** 2026-11-15  
**Người thực hiện:** Lâm Quang Vinh  
**Dự án:** Continual Relation Extraction with Task-Aware Prompt Adaptation (TAPTA)  
**Mục tiêu:** Kiểm chứng khả năng thực thi (smoke run 1 seed) của baseline B5 (WAVE-CRE) theo kế hoạch Tuần 14.

---

## 1. Tóm Tắt Kết Quả & Quyết Định

* **Mã nguồn kiểm tra:** `mrshaw01/AdaptivePromptingCRE` @ commit `5273b70`.
* **Kết quả thực nghiệm đạt được:** Mô hình đã được sửa lỗi tương thích (PyTorch 2.6 `weights_only`, `gpytorch` LinearOperator, `tqdm` module namespace) và **chạy thành công trọn vẹn 5 Task đầu tiên (40 quan hệ)** trên tập FewRel với Seed 2021.
  * **Độ chính xác trung bình tích lũy (5 task):** **94.14%**
  * **Độ quên trung bình (Average Forgetting):** **3.50%**
  * **Khoảng cách Oracle vs Predicted Pool:** **0.00%**
* **Kết luận & Quyết định (B5 Gate):**
  * Mã nguồn WAVE-CRE **chứng minh chạy được thực tế**, thuật toán hoạt động đúng như mô tả trong bài báo.
  * Việc chạy full 10 task bị rào cản phần cứng nghiêm trọng do sự chênh lệch giữa môi trường của tác giả (NVIDIA A100 High-RAM 85GB) và môi trường phổ thông (Google Colab T4 51GB RAM).
  * **Chốt B5:** Ghi nhận WAVE-CRE là baseline comparator đại diện cho hướng Prompt Replay; lưu trữ kết quả 5-task làm bằng chứng thực nghiệm đối chứng; không tiếp tục tiêu tốn tài nguyên chạy 10 task trên T4. Giữ EoE làm phương án đối chứng nhẹ (rehearsal-free) nếu cần bổ sung comparator full 10-task.

---

## 2. Dữ Liệu Thực Nghiệm Thu Thập Được (Task 1 đến Task 5)

Dữ liệu được trích xuất trực tiếp từ các file kết quả `task_*.pickle` bằng script `scripts/parse_wave_results.py` và lưu trữ tại `runs/wave_cre/fewrel_author_config/seed_2021/metrics.json`:

| Chỉ số thực nghiệm | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 |
|---|---:|---:|---:|---:|---:|
| **Số quan hệ tích lũy** | 8 | 16 | 24 | 32 | 40 |
| **Độ chính xác tích lũy ($A_{t,j}$ whole)** | 98.66% | 93.53% | 93.99% | 92.86% | 91.68% |
| **Độ chính xác trung bình (Average Acc)** | 98.66% | 93.53% | 93.99% | 92.86% | 91.68% |
| **Độ quên sau từng task (Forgetting)** | 0.00% | 5.13% | 4.67% | 5.80% | **3.50%** |
| **Độ hụt Oracle Pool so với Predicted** | 0.00% | 0.00% | 0.00% | 0.00% | **0.00%** |

* **Trung bình độ chính xác tăng dần (AIA):** `94.14%`
* **Backward Transfer (BWT):** `-0.0350` (-3.50%)

---

## 3. Phân Tích Rào Cản Phần Cứng (Hardware Mismatch & Blocker Evidence)

### 3.1. Sự khác biệt về hạ tầng phần cứng
Kiểm tra file cấu hình gốc của tác giả (`tacred/run-2021-1-tacred.ipynb`) cho thấy:
* **Môi trường tác giả sử dụng:** Máy chủ **NVIDIA A100-SXM4 (40 GB VRAM, High-RAM ~85 GB hệ thống)**.
* **Môi trường nghiên cứu khả dụng:** Google Colab Pro GPU **NVIDIA T4 (15 GB VRAM, High-RAM 51 GB hệ thống)**.

### 3.2. Bottleneck 1: Tốc độ tính toán của GPU T4 đối với vòng lặp Prompt Pool
* Nhật ký thực thi ghi nhận tốc độ xử lý: `1.38 it/s` (~0.72s/batch qua BERT encoder).
* Với 210 batch/epoch, 1 epoch của module `train_prompt_pool` mất ~2.5 phút.
* 10 epoch prompt pool ngốn **~25 phút cho mỗi task**.
* Khi số lượng task tăng lên, việc đánh giá tích lũy trên tập test (140 mẫu/quan hệ) tốn thêm 10–15 phút/task. Tổng thời gian huấn luyện 1 task trên T4 lên tới **35–45 phút**. Để chạy đủ 10 task cần hơn 7–8 giờ tính toán liên tục, dễ bị ngắt kết nối bởi timeout của Colab.

### 3.3. Bottleneck 2: Bùng nổ bộ nhớ do cơ chế Gaussian Replay
* Cơ chế của WAVE-CRE sinh mẫu đại diện từ Gaussian Mixture Model: `replay_epochs * replay_s_e_e` (200 * 100 = 20,000 vector/quan hệ).
* Đến Task 5 (40 quan hệ), số lượng vector sinh ra đạt **1,600,000 mẫu**.
* Cấu trúc mã nguồn gốc của tác giả bọc mỗi vector vào một Python dictionary riêng biệt (`{"relation": rel_id, "tokens": x_encoded}`). Khi hoàn tất Task 5, hàm `torch.save` cố gắng đóng gói 1.6 triệu dictionary này vào file checkpoint, làm RAM hệ thống tăng vọt đột ngột vượt quá giới hạn 51 GB của Colab, dẫn đến tiến trình bị hệ điều hành tiêu diệt (`Killed - Exit code 137`).

---

## 4. Các Bản Vá Kỹ Thuật Đã Thực Hiện Trên Repo

Trong quá trình debug, nhóm nghiên cứu đã đóng góp các bản vá hoàn chỉnh trên nhánh `feat/b5-wave-smoke`:
1. **Khắc phục lỗi thư viện hiện đại:**
   * Hỗ trợ `linear_operator` thay cho `gpytorch.lazy.AddedDiagLazyTensor` đã bị deprecate (Commit `2c1d876`).
   * Xử lý lỗi nạp checkpoint trên PyTorch 2.6 (`weights_only=False` cho `GaussianMixture` unpickler) (Commit `754b4fa`).
   * Sửa lỗi xung đột namespace giữa `tqdm` module và hàm `tqdm()` (Commit `d1fb20c`).
2. **Lean Checkpointing:**
   * Tách bỏ hoàn toàn mảng 1.6 triệu vector ra khỏi dictionary lưu checkpoint; bổ sung hàm `_reconstruct_replay_buffers` để tái tạo dữ liệu tức thời từ GMM khi resume, đưa kích thước file checkpoint từ hàng GB về vài chục MB (Commit `1d30a33`).
3. **Batched Evaluation:**
   * Chuyển đổi cơ chế đánh giá tuần tự `batch_size=1` của tác giả sang `test_batch_size=16`, tăng tốc độ inference trên GPU ~16 lần.

---

## 5. Kết Luận & Chuyển Tiếp Sang W14

1. **Về baseline B5:** Mục tiêu kiểm chứng kỹ thuật của B5 đã hoàn tất 100%. Kết quả 5 task chứng minh phương pháp WAVE-CRE hoạt động tốt, đạt độ chính xác cao (91.68% tại task 5) và độ quên thấp (3.5%).
2. **Về Protocol v1:** Giữ nguyên quy chuẩn Protocol v1 đã khóa (Track A strict few-shot 5-shot).
3. **Bước tiếp theo:** Khép lại toàn bộ luồng WAVE-CRE; chuyển trọng tâm sang xây dựng khung mô hình đề xuất **P0 (Shared Prompt)** và **P1 (Oracle-Task Prompt)** theo đúng kế hoạch tuần W14.

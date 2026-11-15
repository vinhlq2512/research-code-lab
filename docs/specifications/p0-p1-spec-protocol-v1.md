# Đặc Tả Kỹ Thuật: P0 (Shared Prompt) & P1 (Oracle-Task Prompt)

- **Mã thực nghiệm:** `P0` (Shared Prompt), `P1` (Oracle-Task Prompt)
- **Thuộc đề tài:** Continual Relation Extraction with Task-Aware Prompt Adaptation (TAPTA)
- **Giao thức chuẩn:** Protocol v1 — Track A (FewRel 8 tasks $\times$ 10 relations, 5-shot)
- **Ngày lập:** 2026-11-15
- **Tác giả:** Lâm Quang Vinh

---

## 1. Mục Đích & Định Vị Nghiên Cứu

Trong khuôn khổ đề tài TAPTA, **P0** và **P1** đóng vai trò là hai cực đối chứng (bounding baselines) bắt buộc trước khi bước vào thiết kế bộ định tuyến (Router P2/P3):

1. **P0 (Shared Prompt Baseline — Chặn dưới):**
   * Sử dụng một đoạn mồi nhắc mềm (soft prompt) dùng chung $P_{\text{shared}}$ cho tất cả các tác vụ trên nền mô hình ngôn ngữ đóng băng (Frozen BERT).
   * Đo lường hiện tượng quên lãng thảm khốc (Catastrophic Forgetting) khi không có bất kỳ cơ chế nhận biết tác vụ (Task-Awareness) hay định tuyến nào.
2. **P1 (Oracle-Task Prompt Baseline — Chặn trên):**
   * Sử dụng một tập các mồi nhắc riêng biệt cho từng tác vụ $\{P_1, P_2, \dots, P_T\}$.
   * Khi suy luận (inference), mô hình được cung cấp **Oracle Task ID** để chọn chính xác mồi nhắc $P_{t^*}$ của tác vụ đó.
   * Đại diện cho **mức trần hiệu năng lý tưởng (Upper Bound)** của kiến trúc Prompt khi bộ định tuyến đạt độ chính xác 100%.

$$\text{Task-Awareness Value} = \text{Accuracy}(P1) - \text{Accuracy}(P0)$$
$$\text{Routing Error Gap (TAPTA)} = \text{Accuracy}(P1) - \text{Accuracy}(P2)$$

---

## 2. Kiến Trúc Mô Hình Chi Tiết

### 2.1. Mã hóa đầu vào và chèn Entity Markers
Với câu đầu vào $x$ chứa hai thực thể $e_1$ và $e_2$, chèn 4 entity markers:
$$x_{\text{marked}} = [w_1, \dots, \mathbf{[E1]}, \dots, \mathbf{[/E1]}, \dots, \mathbf{[E2]}, \dots, \mathbf{[/E2]}, \dots, w_N]$$

Các token được đưa qua tầng embedding của BERT (đã bổ sung 4 special tokens):
$$E_{\text{tokens}} = \text{Embeddings}(x_{\text{marked}}) \in \mathbb{R}^{N \times d} \quad (d = 768)$$

### 2.2. Cơ chế Mồi nhắc Mềm (Soft Prompt Prepending)
Khởi tạo tham số mồi nhắc học được $P \in \mathbb{R}^{L \times d}$ với độ dài $L$ (mặc định $L = 8$):
$$E_{\text{prompted}} = [P; E_{\text{tokens}}] \in \mathbb{R}^{(L + N) \times d}$$
Mặt nạ chú ý (attention mask) được mở rộng tương ứng:
$$M_{\text{prompted}} = [\mathbf{1}_L; M_{\text{tokens}}] \in \mathbb{R}^{L + N}$$

### 2.3. Trích xuất biểu diễn quan hệ (Relation Representation)
Toàn bộ trọng số của BERT Encoder $\mathcal{E}$ được **đóng băng (Frozen)**:
$$H = \mathcal{E}(E_{\text{prompted}}, M_{\text{prompted}}) \in \mathbb{R}^{(L + N) \times d}$$
Do mồi nhắc có độ dài $L$ được chèn ở đầu, vị trí của các entity marker bị tịnh tiến $+L$:
$$h_{e1} = H[b, L + \text{pos}([E1])], \quad h_{e2} = H[b, L + \text{pos}([E2])]$$
Biểu diễn quan hệ được tạo thành bằng cách nối hai vector thực thể:
$$h_r = [h_{e1}; h_{e2}] \in \mathbb{R}^{2d} \quad (2d = 1536)$$

### 2.4. Tầng phân loại mở rộng lũy kế (Class-Incremental Linear Head)
Sau tác vụ $t$, không gian nhãn mở rộng gồm toàn bộ các quan hệ đã học:
$$\widetilde{R}_t = \bigcup_{i=1}^t R_i \quad (|\widetilde{R}_t| = 10 \times t)$$
Tầng phân loại tuyến tính tính phân phối xác suất:
$$z = W_{\text{seen}} h_r + b_{\text{seen}} \in \mathbb{R}^{|\widetilde{R}_t|}$$
$$P(y = r \mid x) = \frac{\exp(z_r)}{\sum_{r' \in \widetilde{R}_t} \exp(z_{r'})}$$

---

## 3. Khác Biệt Giữa P0 và P1

| Đặc tính kỹ thuật | P0 (Shared Prompt) | P1 (Oracle-Task Prompt) |
|---|---|---|
| **Số lượng Prompt** | Duy nhất 1 vector mồi nhắc $P_{\text{shared}} \in \mathbb{R}^{L \times d}$ | $T$ vector mồi nhắc riêng biệt $\{P_1, \dots, P_T\}$ |
| **Quy trình Huấn luyện** | Ở mỗi tác vụ $t$, $P_{\text{shared}}$ tiếp tục nhận gradient từ các quan hệ mới (và replay $M=1$ nếu có) | Ở tác vụ $t$, chỉ duy nhất $P_t$ được cập nhật gradient trên dữ liệu của task $t$; các $P_1..P_{t-1}$ cũ đóng băng |
| **Quy trình Suy luận** | Mọi mẫu kiểm thử đều dùng chung $P_{\text{shared}}$ | Mẫu kiểm thử thuộc nhãn $y \in R_k$ được chỉ định dùng chính xác mồi nhắc $P_k$ (Oracle) |
| **Thông tin Task ID** | Hoàn toàn không biết Task ID (Class-Incremental) | Được cung cấp Oracle Task ID khi kiểm tra |
| **Kỳ vọng hiệu năng** | Forgetting cao do một prompt phải gánh 80 quan hệ | Forgetting cực thấp, độ chính xác cao nhất (Upper Bound) |

---

## 4. Ngân Sách Bộ Nhớ (Memory Settings)

Để phục vụ so sánh công bằng theo Protocol v1:
* **Setting $M=0$ (Raw-data-free):** Hoàn toàn không lưu câu mẫu cũ.
* **Setting $M=1$ (1 exemplar/relation):** Lưu 1 câu văn bản gốc cho mỗi quan hệ đã học (giống ConPL). Khi học task $t$, trộn 10 mẫu mới/quan hệ với các mẫu nhớ cũ để huấn luyện tầng phân loại.

---

## 5. Kế Hoạch Đánh Giá & Ma Trận Hiệu Năng $A_{t,j}$

Sau mỗi tác vụ $t \in \{1, \dots, 8\}$, đánh giá trên toàn bộ tập test của các task $j \le t$:
1. Ma trận hiệu năng $A \in \mathbb{R}^{8 \times 8}$.
2. Độ chính xác tích lũy cuối cùng ($A_T$ Whole Accuracy).
3. Độ chính xác trung bình tăng dần (AIA).
4. Độ quên trung bình ($F_T$).
5. Khoảng cách chênh lệch hiệu năng giữa P1 và P0:
   $$\Delta_{\text{oracle}} = A_T(P1) - A_T(P0)$$

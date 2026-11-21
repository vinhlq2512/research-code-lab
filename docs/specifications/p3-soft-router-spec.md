# Đặc Tả Kỹ Thuật: P3 (Soft Top-K Prompt Routing & TAPTA Minimal)

- **Mã thực nghiệm:** `P3` (Soft Top-K Weighted Prompt Composition Router)
- **Thuộc đề tài:** Continual Relation Extraction with Task-Aware Prompt Adaptation (TAPTA)
- **Giao thức chuẩn:** Protocol v1 — Track A (FewRel 8 tasks $\times$ 10 relations, strict 5-shot, Rehearsal-Free $M=0$, seed 2021)
- **Hạ tầng thực thi chuẩn:** Google Colab (NVIDIA GPU CUDA)
- **Ngày lập:** 2026-11-21
- **Tác giả:** Lâm Quang Vinh

---

## 1. Động Cơ & Bài Toán Đặt Ra Sau P2

Ở tuần W15, thực nghiệm **P2a (Centroid Cosine Router)** đã chứng minh một phát hiện quan trọng:
- Trong điều kiện 5-shot ($M=0$), bộ định tuyến phi tham số dựa trên không gian truy vấn đóng băng bất biến `[E1]; [E2]` đạt **Router Accuracy = 64.69%** và **Routed Prototype Accuracy = 59.35%**, thu hẹp **37.69%** khoảng cách tới trần lý tưởng P1 Oracle (70.14%).
- Tuy nhiên, **P2a sử dụng cơ chế chọn cứng Top-1 (Hard Top-1 Routing):**
  $$\hat{t} = \arg\max_{t} s_t(x) \implies \text{kích hoạt duy nhất } P_{\hat{t}}$$
- **Nhược điểm chí tử của Hard Top-1 Routing:**
  Khi câu kiểm thử $x$ có ngữ nghĩa nằm ở biên phân tách hoặc gần nghĩa giữa hai tác vụ (ví dụ quan hệ *residence* ở Task 1 và *headquarters location* ở Task 2):
  - Nếu Router chọn nhầm Task 2 thay vì Task 1 (~35.3% mẫu bị chọn sai), câu $x$ bị ép ghép với mồi nhắc $P_2$.
  - Mồi nhắc $P_2$ được tối ưu hóa riêng cho các quan hệ của Task 2, khiến biểu diễn trích xuất $h_r(x; P_2)$ bị **bóp méo nghiêm trọng (Severe Representation Distortion)**, bị kéo dạt khỏi các nguyên mẫu quan hệ của Task 1.
  - Hậu quả: Độ chính xác phân loại của những mẫu bị route nhầm sụt giảm thảm hại xuống dưới 15%.

### Giải pháp của P3: Soft Top-K Prompt Composition
Thay vì chọn cứng duy nhất 1 prompt theo nguyên lý "được ăn cả ngã về không", **P3 tổng hợp một vector mồi nhắc mềm dành riêng cho từng mẫu $x$ ($P_{\text{soft}}(x)$)** thông qua tổ hợp tuyến tính có trọng số từ Top-$K$ ($K \in \{2, 3\}$) tác vụ ứng viên có điểm tương đồng cao nhất:
$$P_{\text{soft}}(x) = \sum_{k \in \text{Top-}K} w_k(x) \cdot P_k \in \mathbb{R}^{L \times d}$$

- Khi Router rất tự tin ($s_1 \gg s_2$), $w_1 \approx 1.0 \implies P_{\text{soft}}(x) \approx P_1$ (bảo toàn năng lực đặc thù tác vụ như P1/P2a).
- Khi Router phân vân giữa 2 tác vụ ($s_1 \approx s_2$), $P_{\text{soft}}(x)$ là sự hòa trộn trơn tru giữa $P_1$ và $P_2$, giữ cho biểu diễn ngữ cảnh không bị đứt gãy hay rơi vào không gian bất lợi.
- Khi $K=1$, P3 suy biến tự nhiên về P2a (Hard Top-1 Routing).

---

## 2. Công Thức Toán Học Của P3

### 2.1. Không Gian Truy Vấn Bất Biến & Chấm Điểm Tác Vụ (Kế Thừa P2a)
Với encoder BERT $\mathcal{E}_{\text{frozen}}$ đóng băng 100%:
1. Trích xuất vector truy vấn $q(x) = \text{Normalize}_{\ell_2}([h_{\text{[E1]}}^{(0)}; h_{\text{[E2]}}^{(0)}]) \in \mathbb{R}^{2d}$ ($2d = 1536$).
2. Với mỗi tác vụ đã học $t \in \{0, \dots, t_{\text{cur}}\}$, điểm tương đồng hợp nhất được tính theo tâm tác vụ $\mu_t$ và tâm con quan hệ $\nu_{t,c}$:
   $$s_t(x) = (1 - \alpha) \cdot (q(x)^\top \mu_t) + \alpha \cdot \max_{c \in R_t} (q(x)^\top \nu_{t,c})$$
   (Mặc định $\alpha = 0.7$).

### 2.2. Trọng Số Tổ Hợp Mềm Top-$K$ (Soft Top-K Attention Weights)
Xác định tập chỉ số Top-$K$ tác vụ có điểm cao nhất:
$$\mathcal{T}_K(x) = \text{Top-}K\left(\{s_t(x)\}_{t=0}^{t_{\text{cur}}}\right), \quad K = \min(K_{\text{target}}, t_{\text{cur}} + 1)$$

Trọng số softmax với hệ số nhiệt độ $\tau > 0$:
$$w_k(x) = \frac{\exp\left(\frac{s_k(x)}{\tau}\right)}{\sum_{j \in \mathcal{T}_K(x)} \exp\left(\frac{s_j(x)}{\tau}\right)}, \quad \forall k \in \mathcal{T}_K(x)$$
với $\sum_{k \in \mathcal{T}_K(x)} w_k(x) = 1.0$.

### 2.3. Tổng Hợp Mồi Nhắc Mềm & Trích Xuất Biểu Diễn
Vector mồi nhắc mềm per-sample:
$$P_{\text{soft}}(x) = \sum_{k \in \mathcal{T}_K(x)} w_k(x) \cdot P_k \in \mathbb{R}^{L \times d}$$

Biểu diễn quan hệ được trích xuất qua BERT đóng băng với tiền tố $P_{\text{soft}}(x)$:
$$h_r(x; P_{\text{soft}}(x)) = \text{ExtractEntityPair}\left(\mathcal{E}_{\text{frozen}}\left([P_{\text{soft}}(x); x_{\text{marked}}]\right)\right) \in \mathbb{R}^{2d}$$

### 2.4. Mất Mát Trực Giao Hóa Mồi Nhắc (Prompt Orthogonality Regularization)
Để tối ưu hóa hiệu quả của việc hòa trộn $\sum w_k P_k$, các mồi nhắc $P_0, \dots, P_T$ cần học các không gian ngữ cảnh bổ trợ nhau, tránh hiện tượng sụp đổ mồi nhắc (*Prompt Collapse / Redundancy*).
Khi huấn luyện mồi nhắc $P_t$ ở tác vụ $t \ge 1$:
$$\bar{P}_t = \text{Normalize}_{\ell_2}\left(\frac{1}{L} \sum_{l=1}^L P_t[l]\right) \in \mathbb{R}^d$$
$$\mathcal{L}_{\text{orth}}(P_t) = \frac{1}{t} \sum_{j=0}^{t-1} \left|\bar{P}_t^\top \bar{P}_j\right|$$

Hàm mục tiêu tổng thể khi huấn luyện tác vụ $t$:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}}(D_t; P_t) + \lambda_{\text{orth}} \mathcal{L}_{\text{orth}}(P_t)$$
(Mặc định $\lambda_{\text{orth}} = 0.05$).

---

## 3. Kiến Trúc Xử Lý Vector-Parallel (GPU Batch Friendly)

Để đảm bảo tốc độ suy luận siêu tốc tương đương với P2a (không lặp vòng for trên từng mẫu):
1. Gom toàn bộ prompt đã học thành tensor:
   $$\mathbf{P}_{\text{stack}} = \text{stack}([P_0, \dots, P_{t_{\text{cur}}}]) \in \mathbb{R}^{T_{\text{seen}} \times L \times d}$$
2. Với một mini-batch $B$ mẫu, tensor chỉ số Top-$K$ là $\mathbf{I}_{\text{topk}} \in \mathbb{R}^{B \times K}$ và trọng số $\mathbf{W}_{\text{topk}} \in \mathbb{R}^{B \times K}$.
3. Thu thập các prompt ứng viên bằng tensor indexing:
   $$\mathbf{P}_{\text{cand}} = \mathbf{P}_{\text{stack}}[\mathbf{I}_{\text{topk}}] \in \mathbb{R}^{B \times K \times L \times d}$$
4. Nhân trọng số và tính tổng trực tiếp trên chiều $K$:
   $$\mathbf{P}_{\text{batch\_soft}} = \sum_{k=1}^K \mathbf{W}_{\text{topk}}[:, k, \text{None}, \text{None}] \odot \mathbf{P}_{\text{cand}}[:, k, :, :] \in \mathbb{R}^{B \times L \times d}$$
5. Nối $\mathbf{P}_{\text{batch\_soft}}$ trực tiếp với word embeddings của batch và đưa qua một forward pass duy nhất của BERT.

---

## 4. Các Độ Đo Đánh Giá Của P3

1. **Top-1 Router Accuracy ($R_{\text{acc}}^{\text{top-1}}$):** Tỷ lệ mẫu có $\arg\max s_t(x) = t^*(y)$.
2. **Top-$K$ Router Coverage ($R_{\text{acc}}^{\text{top-k}}$):** Tỷ lệ mẫu có tác vụ thật $t^*(y) \in \mathcal{T}_K(x)$. (Kỳ vọng $\ge 85\%$ khi $K=2, 3$).
3. **Global Prototype Accuracy ($A_T^{\text{proto}}$):** Độ chính xác phân loại trên toàn bộ tập nhãn đã học $\widetilde{R}_T$ khi sử dụng $h_r(x; P_{\text{soft}}(x))$ so khớp cosine với tất cả nguyên mẫu quan hệ.
4. **Top-$K$ Gated Prototype Accuracy ($A_T^{\text{topk-gated}}$):** Độ chính xác khi giới hạn không gian nhãn trong tập quan hệ của các tác vụ ứng viên $\bigcup_{j \in \mathcal{T}_K(x)} R_j$.
5. **Oracle Gap Closed Percentage:**
   $$\text{Gap Closed (\%)} = \frac{A_T^{\text{proto}}(P3) - A_T(P0)}{A_T(P1) - A_T(P0)} \times 100\%$$

---

## 5. Tiêu Chí Nghiệm Thu (Acceptance Criteria)

- Hoàn thành cập nhật mã nguồn trong `src/models/prompt_bert.py` và `src/models/task_router.py`.
- Tạo config `configs/p3_soft_router_fewrel_5shot_seed2021.yaml`, runner `experiments/run_p3.py` và notebook Colab `notebooks/p3_soft_router_colab_seed2021.ipynb`.
- Kiểm thử cú pháp cục bộ thành công mà không chạy train trên MacBook.
- **Mục tiêu hiệu năng trên Colab GPU:**
  - Top-$K$ Router Coverage ($K=2$): $\ge 82.0\%$.
  - Whole Prototype Accuracy: **$\ge 62.0\%$** (kỳ vọng $63\% - 65\%$).
  - Oracle Gap Closed: **$\ge 50.0\%$**.

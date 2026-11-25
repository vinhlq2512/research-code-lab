# Đặc Tả Kỹ Thuật: P4 (TAPTA Hierarchical Prompt Tree & Decision Fusion)

- **Mã thực nghiệm:** `P4` (TAPTA Hierarchical Prompt Tree with Sub-Centroid Routing & Decision Fusion)
- **Thuộc đề tài:** Continual Relation Extraction with Task-Aware Prompt Adaptation (TAPTA)
- **Giao thức chuẩn:** Protocol v1 — Track A (FewRel 8 tasks $\times$ 10 relations, strict 5-shot, Rehearsal-Free $M=0$, seed 2021)
- **Hạ tầng thực thi chuẩn:** Google Colab (NVIDIA GPU CUDA)
- **Ngày lập:** 2026-11-23
- **Tác giả:** Lâm Quang Vinh

---

## 1. Động Cơ & Đột Phá Thiết Kế Sau Kết Quả P3

Tại các mốc thực nghiệm trước:
- **P0 (Shared Prompt — Sàn):** Dùng 1 mồi nhắc chia sẻ $L=8$ cho mọi tác vụ $\to$ Quên lãng thảm khốc ($19.20\%$, $A_{8,1} = 38.90\%$, Proto Acc = $52.83\%$).
- **P1 (Oracle Task Prompt — Trần):** Mỗi tác vụ dùng 1 mồi nhắc riêng $L=8$, có Oracle Task ID $\to$ Proto Acc = $70.14\%$, TIL = $82.75\%$.
- **P2a (Centroid Router):** Dùng nhân Sub-Centroid không tham số chọn cứng Top-1 $\to$ Router Acc = $64.69\%$, Proto Acc = $59.28\%$, Gated Acc = $59.35\%$. Tuy nhiên, ~35.3% mẫu bị gán sai prompt do chọn cứng đơn lẻ.
- **P3 (Soft Top-K Prompt Composition):** Hòa trộn mồi nhắc mềm trong không gian tham số ($P_{\text{soft}} = w_1 P_1 + w_2 P_2$) $\to$ **Top-2 Router Coverage tăng vọt lên 81.41% (AIA 90.58%)**, nhưng Global Proto Acc lại giảm còn $55.50\%$ do hiện tượng **Lệch pha Mồi nhắc - Nguyên mẫu (*Prompt-Prototype Discrepancy*)**: vector đặc trưng của mẫu kiểm thử bị kéo sang không gian nội suy, lệch khỏi các nguyên mẫu tĩnh $p_c$ được tính từ mồi nhắc thuần túy lúc huấn luyện.

### Giải pháp cốt lõi của P4: Cây Mồi nhắc Phân cấp & Tổ hợp Tầng Quyết định
P4 giải quyết triệt để sự đánh đổi này bằng 2 cải tiến kiến trúc:
1. **Phân rã Mồi nhắc Phân cấp 2 Tầng (Hierarchical Prompt Tree Decomposition):**
   Thay vì xem prompt là một khối phẳng $L=8$ tokens, P4 chia tách độ dài $L$ thành:
   - **Tầng Gốc (Root Prompt $P^{\text{root}} \in \mathbb{R}^{L_{\text{root}} \times d}$, $L_{\text{root}}=4$):** Chia sẻ chung cho mọi tác vụ, học ngữ cảnh quan hệ bất biến tổng quát, đóng vai trò mỏ neo ổn định không gian đặc trưng.
   - **Tầng Nhánh (Leaf Prompt $P^{\text{leaf}}_t \in \mathbb{R}^{L_{\text{leaf}} \times d}$, $L_{\text{leaf}}=4$):** Độc lập theo từng tác vụ $T_t$, học biên phân loại chi tiết cho 10 quan hệ mới.
   - Tổng chiều dài: $L = L_{\text{root}} + L_{\text{leaf}} = 4 + 4 = 8$ tokens (bảo toàn 100% ngân sách tính toán và tham số như P0/P1/P2/P3).
2. **Tổ hợp Mềm ở Tầng Quyết định (Decision-Level Soft Fusion):**
   Thay vì hòa trộn tham số mồi nhắc trước encoder (gây méo mó phân phối ở P3), P4 giữ nguyên biểu diễn thuần túy của các mồi nhắc cây và tổng hợp điểm số tương đồng nguyên mẫu theo trọng số tin cậy của Router trên Top-$K$ ứng viên:
   $$\mathcal{S}_{\text{fusion}}(x, c) = \sum_{k \in \mathcal{T}_K(x)} w_k(x) \cdot \cos\left(h_r(x; [P^{\text{root}}; P^{\text{leaf}}_k]), p_c\right)$$
   Triệt tiêu hoàn toàn hiện tượng lệch pha nguyên mẫu, đồng thời tận dụng trọn vẹn **81.41% Top-2 Router Coverage**!

---

## 2. Công Thức Toán Học Của P4

### 2.1. Kiến Trúc Cây Mồi Nhắc 2 Tầng
Mồi nhắc đầy đủ được đưa vào đầu chuỗi đầu vào của BERT cho tác vụ $t$:
$$P_t = [P^{\text{root}}; P^{\text{leaf}}_t] \in \mathbb{R}^{1 \times (L_{\text{root}} + L_{\text{leaf}}) \times d}$$
trong đó:
- $P^{\text{root}} \in \mathbb{R}^{1 \times 4 \times 768}$
- $P^{\text{leaf}}_t \in \mathbb{R}^{1 \times 4 \times 768}$ cho $t \in \{0, \dots, T-1\}$.

### 2.2. Huấn Luyện Tác Vụ Cô Lập
Tại tác vụ $T_t$:
- $P^{\text{root}}$ được huấn luyện ở $T_0$ và cập nhật với tốc độ học nhỏ $\eta_{\text{root}} = 0.1 \times \eta_{\text{leaf}}$ ở các tác vụ sau để bảo tồn tri thức chung.
- $P^{\text{leaf}}_t$ được tối ưu hóa bằng Cross-Entropy trên $D_t$ kết hợp mất mát trực giao $\mathcal{L}_{\text{orth}}$ so với các leaf prompt trước đó $\{P^{\text{leaf}}_0, \dots, P^{\text{leaf}}_{t-1}\}$:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}}(D_t; P_t) + \lambda_{\text{orth}} \mathcal{L}_{\text{orth}}(P^{\text{leaf}}_t)$$

### 2.3. Trích Xuất Nguyên Mẫu Quan Hệ
Nguyên mẫu $p_c$ cho quan hệ $c \in R_t$ được tính bằng trung bình các biểu diễn mẫu huấn luyện mang mồi nhắc cây chuẩn $P_t$:
$$p_c = \text{Normalize}_{\ell_2}\left(\frac{1}{|D_{t,c}|} \sum_{x \in D_{t,c}} h_r(x; [P^{\text{root}}; P^{\text{leaf}}_t])\right) \in \mathbb{R}^{2d}$$

### 2.4. Quy Trình Suy Luận Hai Tầng (Two-Tier Hierarchical Inference)
Với mỗi câu kiểm thử $x$ không có Task ID:
1. **Bước 1 (Định tuyến Sub-Centroid):**
   Trích xuất vector truy vấn bất biến $q(x) = \text{Normalize}_{\ell_2}([h_{\text{[E1]}}^{(0)}; h_{\text{[E2]}}^{(0)}])$ qua BERT đóng băng.
   Tính điểm tương đồng $s_t(x)$ và trích xuất Top-$K$ ($K=2$) tác vụ ứng viên $\mathcal{T}_K(x)$ cùng trọng số Softmax nhiệt độ $w_k(x)$ ($\tau = 0.2$).
2. **Bước 2 (Trích xuất Biểu diễn theo Mồi nhắc Cây):**
   - **Chế độ Top-1 Hierarchical:** Ghép mồi nhắc cây của tác vụ ứng viên số 1 $P_{\hat{t}} = [P^{\text{root}}; P^{\text{leaf}}_{\hat{t}}]$, trích xuất $h_r(x; P_{\hat{t}})$ và so khớp với các nguyên mẫu đã học (toàn cục hoặc khoanh vùng Top-$K$ candidate classes).
   - **Chế độ Decision-Level Soft Fusion:**
     Trích xuất biểu diễn cho từng ứng viên $k \in \mathcal{T}_K(x)$, tính ma trận tương đồng Cosine $\text{sim}_k(x) \in \mathbb{R}^{|\widetilde{R}|}$ và tổng hợp:
     $$\text{sim}_{\text{fused}}(x) = \sum_{k \in \mathcal{T}_K(x)} w_k(x) \cdot \text{sim}_k(x)$$
     $$\hat{y}(x) = \arg\max_{c \in \widetilde{R}_{t_{\text{cur}}}} \text{sim}_{\text{fused}}(x)[c]$$

---

## 3. Các Độ Đo Đánh Giá Của P4

1. **Hierarchical Top-1 Prototype Accuracy ($A_T^{\text{tree-top1}}$):** Độ chính xác toàn cục khi dùng mồi nhắc cây với Top-1 task selection.
2. **Top-$K$ Gated Tree Accuracy ($A_T^{\text{tree-topk-gated}}$):** Độ chính xác khi giới hạn không gian nguyên mẫu trong Top-$K$ tác vụ ứng viên.
3. **Decision Fusion Accuracy ($A_T^{\text{decision-fusion}}$):** Độ chính xác phân loại toàn cục khi tổng hợp điểm số ở tầng quyết định.
4. **Router Top-1 & Top-2 Coverage:** Theo dõi tính nhất quán của bộ định tuyến.
5. **Oracle Gap Closed Percentage:** Mức độ thu hẹp khoảng cách trần P1 so với sàn P0.

---

## 4. Tiêu Chí Nghiệm Thu (Acceptance Criteria)

- Triển khai chế độ `prompt_mode = "hierarchical_tree"` trong `PromptedBERTRelationClassifier`.
- Tạo cấu hình `configs/p4_hierarchical_prompt_tree_fewrel_5shot_seed2021.yaml`, runner `experiments/run_p4.py` và notebook `notebooks/p4_hierarchical_tree_colab_seed2021.ipynb`.
- **Mục tiêu định lượng trên Colab GPU:**
  - Vượt mốc Whole Accuracy của P2a ($59.28\%$).
  - Kỳ vọng đạt **$61.5\% - 64.0\%$**, thu hẹp $\ge 50\%$ khoảng cách trần - sàn Oracle.

# M7.4 Specification — Candidate-Aware Prompt Reranking (`CANDIDATE_RERANKING_SPEC.md`)

**Status:** ACTIVE (M7.4)  
**Parent Method:** `DynaTree-CRE` (within TAPTA Master's Thesis)  
**Động cơ từ M7.2–M7.3:** Retrieval trên Dual-Space đã đạt `Recall@3 = 80.70%`, `Recall@5 = 87.99%`, `Recall@10 = 94.29%` (FewRel) và `Recall@10 = 94.16%` (TACRED), nhưng Soft Top-3 Mixing làm phẳng trọng số (`top1_prob < 0.2` ở 100% mẫu) và gây nhiễu chéo giữa các prompt, kéo `Cond@3` xuống `~69%`.

---

## 1. Quy trình 6 bước tại Suy luận (Inference — Zero Label Leakage)
1. **Stage 1 — Retrieval:** Router truy xuất Top-$K$ ứng viên $\mathcal{C}_K(x) = \{r_1, \dots, r_K\}$ kèm điểm truy xuất thô $s_{\text{ret}}(x, r_k)$ (`flat_soft_topk` cho `DT-C2.1`; `hybrid_tree_routing` hoặc `size_corrected_hierarchical` cho `DT-C4.4`). Tuyệt đối không truy cập nhãn thật.
2. **Stage 2 — Candidate Prompt Composition:** Với mỗi ứng viên $r \in \mathcal{C}_K(x)$, tạo prompt path độc lập (không trộn giữa các ứng viên):
   $$P_r = P_{\text{root}} + P_{c(r)} + \Delta P_r \in \mathbb{R}^{L \times d}$$
3. **Stage 3 — Batched Prompted Representation:** Chạy forward gộp theo mini-batch (chunked để tránh OOM khi $K=10$):
   $$z_r = \text{FrozenBERT}(x, P_r) \in \mathbb{R}^d, \quad r \in \mathcal{C}_K(x)$$
4. **Stage 4 — Candidate Prototype Scoring:** So sánh $z_r$ với prototype $\mu^{\text{prompted}}_r$ của chính ứng viên $r$ đã được dựng trong không gian candidate-conditioned:
   $$s_{\text{prompt}}^{\text{NCM}}(x, r) = \cos(z_r, \mu^{\text{prompted}}_r), \qquad s_{\text{prompt}}^{\text{Cos}}(x, r) = \alpha \cos(z_r, w_r)$$
5. **Stage 5 — Score Fusion:** Chuẩn hóa điểm trong tập ứng viên $\mathcal{C}_K(x)$ (theo mẫu bằng `zscore` hoặc `minmax`):
   - `retrieval_only`: $\lambda = 1.0$
   - `prompted_only`: $\lambda = 0.0$
   - `fusion`: $S_r = \lambda \tilde{s}_{\text{ret}}(x, r) + (1 - \lambda)\tilde{s}_{\text{prompt}}(x, r)$ (chốt $\lambda = 0.5$ hoặc tune nội bộ trên train-internal validation, cấm dùng `test_0` / `valid_0`).
6. **Stage 6 — Final Prediction:** $\hat{r} = \arg\max_{r \in \mathcal{C}_K(x)} S_r$.

---

## 2. Huấn luyện Candidate-Aware & Tính nhất quán Prototype (Zero Drift)
- **Ràng buộc Section 7:** Không tái sử dụng NCM prototypes từ soft-mixed training cho candidate-conditioned inference.
- **Candidate-Aware Training:** Khi học task $t$, với mỗi mẫu $(x, r_{\text{true}})$ ($r_{\text{true}} \in R_t$):
  1. Tính $z_{\text{pos}} = \text{FrozenBERT}(x, P_{r_{\text{true}}})$ và loss phân loại toàn cục `CE(classifier(z_pos), r_true)` (bảo đảm mọi leaf mới luôn có gradient, `zero_grad_new_leaf_count == 0`).
  2. Tính candidate-conditioned contrastive CE loss trên tập ứng viên $\{r_{\text{true}}\} \cup \text{HardNeg}_{\text{Top-}(K-1)}(x)$ để `cos(z(x, P_{r_true}), w_{r_true})` vượt các ứng viên dễ nhầm (kể cả lớp cũ đóng băng).
- **Xây dựng `PromptedNCM`:** Dùng `use_effective_leaf_prompt=True` (`z = FrozenBERT(x, P_r)`) chỉ từ dữ liệu train của task hiện tại $D_r$ ($r \in R_t$). Do `freeze_old_root=True`, `freeze_old_clusters=True`, `freeze_old_leaves=True`, mọi $P_{r_{\text{old}}}$ bất biến tuyệt đối $\implies$ $\mu^{\text{prompted}}_{r_{\text{old}}}$ **không bao giờ bị drift**.

---

## 3. Biến thể & Chỉ số Bắt buộc
- **Biến thể chính:** `DT-C2.1` (Flat candidate reranking) và `DT-C4.4` (Tree candidate reranking) tại $K \in \{1, 3, 5, 10\}$ và `{retrieval_only, prompted_only, fusion}`.
- **Bất biến đại số:** Vì $\hat{r} \in \mathcal{C}_K(x)$, mọi lượt chạy reranking phải thỏa:
  $$\text{Final Accuracy} = \text{Candidate Recall@}K \times \text{Conditional Reranking Accuracy}$$
- **Hiệu năng:** Ghi nhận thêm `num_bert_forward_passes`, `peak_gpu_memory_bytes`, `persistent_memory_bytes`, và `inference_latency_ms_per_sample`.

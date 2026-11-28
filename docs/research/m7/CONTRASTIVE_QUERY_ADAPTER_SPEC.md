# M7.5 Step 4 Specification — Learned Contrastive Query Adapter (`CONTRASTIVE_QUERY_ADAPTER_SPEC.md`)

**Status:** ACTIVE (M7.5 Step 4)  
**Parent Method:** `DynaTree-CRE` (within TAPTA Master's Thesis)  
**Động cơ từ M7.5 Steps 1–3:** Do số mẫu 5-shot ($N=400 \ll d=1536$) khiến Analytic Ridge (`DT-C0.5`, `57.3%–58.1%`) thiếu hạng ma trận Gram và chưa vượt được `DT-C0 Raw` (`60.08%`). Bước 4 kiểm chứng `Learned Residual Contrastive Query Adapter` trên biểu diễn `bert-base-uncased` đóng băng 100%.

---

## 1. Kiến trúc `ResidualQueryAdapter` (Zero Backbone Fine-Tuning)
$$g_\theta(q) = \text{Norm}_2\Big(q + \alpha \cdot W_2\,\text{LayerNorm}\big(\text{GELU}(W_1 q + b_1)\big) + b_2\Big)$$
- `W_1` $\in \mathbb{R}^{d_{\text{rep}} \times h}$, `W_2` $\in \mathbb{R}^{h \times d_{\text{rep}}}$ (`h=256`, tổng ~0.79M params ≈ `3.0 MB`).
- Khởi tạo cận 0 ở `W_2, b_2` (`std=1e-3`) với hệ số thặng dư $\alpha = 0.5$ để tại thời điểm khởi đầu $g_\theta(q) \approx \text{Norm}_2(q)$, không phá vỡ hình học BERT.

---

## 2. Hai chế độ huấn luyện & Chính sách bộ nhớ (Section 8)

### Chế độ A — `t1_frozen` (`DT-C0.6a` / `DT-C4.6a`)
- **Memory Policy:** `rehearsal_free_strict_zero_pseudo_replay` (không dùng prototype cũ làm neo huấn luyện).
- **Cơ chế:** Chỉ huấn luyện $g_\theta$ trên `train_samples` của $T_1$ bằng Supervised Contrastive Loss (`SupCon`) + Prototype Cross-Entropy, sau đó **đóng băng vĩnh viễn** (`requires_grad=False`) từ $T_2$ trở đi.
- **Bất biến:** `representation_drift = 0.0`; mọi prototype mới ở $T_{2..8}$ biến đổi nhất quán qua $g_\theta$.

### Chế độ B — `continual_anchor_preserved` (`DT-C0.6b` / `DT-C4.6b`)
- **Memory Policy (phân loại riêng theo Section 8):** `rehearsal_free_raw_prototype_anchors_only` — **không** lưu bất kỳ instance embedding hay text cũ nào; chỉ sử dụng duy nhất `raw_prototype_vector` ($\mu^{\text{raw}}_r$, `1` vector/relation) đã lưu làm **prototype anchor**.
- **Hàm mục tiêu tại task $t$:**
  $$\mathcal{L}_{\text{adapter}} = \mathcal{L}_{\text{proto-CE}}\big(g_\theta(q_b), \{g_\theta(\mu^{\text{raw}}_r)\}_{r \in R_{\le t}}\big) + 0.5\,\mathcal{L}_{\text{SupCon}}(g_\theta(q_b)) + \lambda_{\text{drift}}\frac{1}{|R_{<t}|}\sum_{r \in R_{<t}}\|g_\theta(\mu^{\text{raw}}_r) - \text{sg}(g_{\theta_{t-1}}(\mu^{\text{raw}}_r))\|_2^2$$
- **Prototype Transformation Policy:** Sau khi kết thúc huấn luyện adapter ở task $t$, cập nhật vector truy xuất cho mọi quan hệ đã thấy qua phép biến đổi nhất quán $\mu^{\text{ret}}_r = g_{\theta_t}(\mu^{\text{raw}}_r)$. Cấu trúc cây phân cấp ở `t1_centered_cosine` giữ nguyên bất biến để không làm sụp đổ phân cụm.

---

## 3. Chỉ số kiểm chứng bắt buộc (Old vs New Retrieval & Drift)
Mỗi stage $t$ bắt buộc ghi nhận:
- `retrieval_all_top1_acc`, `retrieval_old_top1_acc` ($R_{<t}$), `retrieval_new_top1_acc` ($R_t$).
- `representation_drift_l2`: Trung bình $\|g_{\theta_t}(\mu^{\text{raw}}_r) - g_{\theta_{t-1}}(\mu^{\text{raw}}_r)\|_2$ trên các quan hệ cũ.
- So sánh công bằng: `DT-C0 Raw` (`60.08%`) vs `DT-C0.6a/b` (Retrieval Only) vs `DT-C2.6` (Flat Rerank) vs `DT-C4.6` (Tree Rerank).

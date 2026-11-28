# M7.5 Specification — Improved Retrieval Space (`IMPROVED_RETRIEVAL_SPEC.md`)

**Status:** ACTIVE (M7.5 — Steps 1, 2, 3)  
**Parent Method:** `DynaTree-CRE` (within TAPTA Master's Thesis)  
**Động cơ từ M7.4:** Không gian `raw_cosine` trên `[h_e1; h_e2]` (`1536`-d) của BERT đóng băng bị cộng phương sai nền (shared cone covariance), khiến retrieval Top-1 bị khóa trần ở `60.08%` trên FewRel và `41.75%` trên TACRED. Trong khi đó `DT-C4.4` đã chứng minh `Tree` vượt `Flat` `+6.3%–+11.0%` ở `prompted_only` và vượt `P4`/`DT-C0` trên TACRED (`42.55%` vs `41.85%` / `42.45%`).

---

## 1. Ba thành phần M7.5 (Ưu tiên đơn giản, Zero Instance Replay)

### Thành phần 1 — Frozen Random Projection (RanPAC-style)
$$\phi(q) = \text{Norm}_2\big(\text{ReLU}(\text{Norm}_2(q) W_{\text{rand}})\big) \in \mathbb{R}^{M}$$
- `W_rand` $\in \mathbb{R}^{d_{\text{rep}} \times M}$ khởi tạo xác định theo seed (`projection_seed`, chuẩn $\mathcal{N}(0, 1/M)$) và **đóng băng vĩnh viễn**.
- Hỗ trợ đối chứng `projection_mode = "identity"` ($M = d_{\text{rep}}$, không random projection) vs `"random_relu"` ($M = 2048$ hoặc $4096$).

### Thành phần 2 — Incremental Analytic Classifier (Closed-Form Ridge / LDA)
- **Bộ nhớ liên tục (Zero Replay — Không lưu bất kỳ instance embedding cũ nào):**
  1. Ma trận Gram tích lũy: $G = \sum_{i \in \mathcal{D}_{\le t}} \phi(q_i)^T \phi(q_i) \in \mathbb{R}^{M \times M}$.
  2. Vector tổng theo từng quan hệ đã thấy: $C_r = \sum_{x \in D_r} \phi(q(x)) \in \mathbb{R}^{M}, \forall r \in R_{\le t}$.
- **Giải tích đóng chính xác (float64 solve, không có future class):**
  $$W_{\text{analytic}} = (G + \lambda_{\text{ridge}} I_M)^{-1} C^T \in \mathbb{R}^{M \times |R_{\le t}|}, \qquad s_{\text{analytic}}(x) = \phi(q(x)) W_{\text{analytic}}$$
- **Tính chất:** Không có optimizer forgetting, không bị new-class weight magnitude bias, bất biến thứ tự task (`G` và `C` là phép cộng giao hoán).

### Thành phần 3 — Relation Description-Enhanced Prototype
- Sử dụng tên quan hệ sạch từ `ProtocolV1Task.relation_names` đưa vào mẫu ngữ cảnh đánh dấu thực thể:
  `"[E1] subject entity [/E1] has relation {relation_name} with [E2] object entity [/E2]"`
- Mã hóa bằng **cùng `StableQueryEncoder` đóng băng** để thu được $e_{\text{desc}}(r) \in \mathbb{R}^{d_{\text{rep}}}$ cùng không gian `[h_e1; h_e2]`, rồi kết hợp có trọng số $\beta \in [0, 1)$ với prototype 5-shot từ train:
  $$\bar{q}^{\text{raw}}_r \leftarrow \text{Norm}_2\big((1 - \beta)\bar{q}^{\text{sample}}_r + \beta\, e_{\text{desc}}(r)\big)$$
  và cập nhật tương ứng vào $C_r, G$ nếu bật mô tả ở retrieval/analytic.

---

## 2. Biến thể Đối chứng Bắt buộc (Section 8 & 13)
| Variant ID | Tree & Prompt | Retrieval Space / Head | Mục đích |
|---|---|---|---|
| **`DT-C0.5a`** | None (Control) | `Analytic` (`identity`, $M=1536$) | Tác dụng thuần của khử phương sai $(G+\lambda I)^{-1}$ |
| **`DT-C0.5b`** | None (Control) | `Analytic` (`random_relu`, $M=4096$) | Tác dụng của mở rộng phi tuyến RanPAC |
| **`DT-C0.5c`** | None (Control) | `Analytic` (`M=4096`) + `Relation Desc` | Tác dụng cộng thêm từ mô tả quan hệ 5-shot |
| **`DT-C2.5`** | Flat Rerank (`DT-C2.1`) | `Analytic` ($M=4096$ + Desc) + Candidate Rerank | Kiểm chứng lợi ích Flat Reranking trên Analytic |
| **`DT-C4.5`** | **Tree Rerank (`DT-C4.4`)** | **`Analytic` ($M=4096$ + Desc) + Candidate Rerank** | Kiểm chứng `Tree > Flat` và `DT-C4.5 > DT-C0.5` |

> **Ghi chú trung thực về bộ nhớ:** $G \in \mathbb{R}^{M \times M}$ (float32) chiếm $4M^2$ bytes (`~9 MB` ở $M=1536$, `~16 MB` ở $M=2048$, `~64 MB` ở $M=4096$) — sẽ được hạch toán đầy đủ trong `resource_metrics.json` (`persistent_analytic_bytes`).

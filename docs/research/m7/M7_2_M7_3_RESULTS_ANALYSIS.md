# M7.2 & M7.3 Empirical Results Analysis (Colab GPU, Seed 2021)

**Nguồn dữ liệu:** [`runs/m7_2_m7_3_colab_results/`](../../../runs/m7_2_m7_3_colab_results) (giải nén từ `m7_2_m7_3_colab_results.zip`, 12/12 runs hoàn tất trên CUDA).

---

## 1. M7.2 — Kết quả Ma trận Dual-Space 2×2 (FewRel Standard, Seed 2021)

| Run | Tree Space | Retrieval Space | NCM Acc | NCM F | Route@1 | R@3 | R@5 | R@10 | Clus@1 | Dead Leaves (Max/Task) | Clusters |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **M7.2-A** | `raw` (`0.75`) | `raw` | **55.59%** | 16.54% | 17.04% | 18.89% | 19.93% | 23.65% | 17.31% | **10/10** (60 zero steps) | 17 (65% sing.) |
| **M7.2-D** | `raw` (`0.75`) | `centered` | 52.41% | 16.97% | 23.04% | 29.44% | 33.26% | 44.34% | 25.02% | **3/10** | 17 (65% sing.) |
| **M7.2-B (`DT-C4`)** | `centered` | `centered` | **57.23%** | 16.71% | **56.97%** | **76.66%** | — | — | **86.54%** | **0** | 4 (0% sing.) |
| **M7.2-C** | `centered` | `raw` | 55.10% | **16.21%** | 48.90% | 66.42% | 74.76% | 85.09% | 68.60% | **0** | 4 (0% sing.) |

**Kết luận M7.2:**
1. **`raw_cosine` KHÔNG dùng để xây cây được:** Khi dựng cây bằng `raw_cosine`, cấu trúc cụm sụp đổ (65% singleton), làm tới `10/10` leaf prompts ở T8 bị chết gradient (`zero_grad_new_leaf_count = [0, 5, 4, 8, 7, 5, 8, 10]`), kéo `Route@1` xuống `17%–23%`.
2. **Vì sao `M7.2-C` (`eta=0`) bị tụt `Route@1` từ `56.97%` xuống `48.90%`:** Không gian `raw_cosine` có độ co cụm hình nón cao (`cos ~ 0.90–0.99`), nên khi dùng `hierarchical_soft` chưa hiệu chỉnh kích thước cụm ($\eta = 0$), hiện tượng cluster-size bias bị khuếch đại mạnh hơn.

---

## 2. M7.3 — Kết quả Sửa Cluster-Size Bias & Hybrid Tree Routing

### 2.1. Tổng hợp toàn bộ các phân phối (Seed 2021)

| Dataset / Dist | Variant | NCM Whole | NCM AA | NCM F | Route@1 | R@3 | R@5 | R@10 | Clus@1 | Cond@1 | Cond@3 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **FewRel Std** | `DT-C4` MVP | **57.23%** | 57.23% | 16.71% | 56.97% | 76.66% | — | — | 86.54% | — | — |
| **FewRel Std** | `SizeCorr (eta=1)` | 55.91% | 55.91% | 17.40% | 59.01% | 79.09% | 86.36% | 93.40% | 87.75% | 84.37% | **70.30%** |
| **FewRel Std** | `Hybrid (g=0.25)` | 55.35% | 55.35% | 19.20% | 59.88% | 80.25% | 87.90% | 94.21% | **89.72%** | 82.38% | 68.74% |
| **FewRel Std** | `FlatRaw (g=0.0)` | 56.46% | 56.46% | 17.07% | **60.08%** | **80.70%** | **87.99%** | **94.29%** | 89.51% | 84.04% | 69.59% |
| **Exp B (Homog.)** | `DT-C4` MVP | 53.92% | 53.92% | **12.43%** | 41.21% | 66.59% | — | — | 56.27% | — | — |
| **Exp B (Homog.)** | `SizeCorr (eta=1)` | **56.83%** | **56.83%** | 12.69% | 58.64% | 78.34% | 85.79% | 93.03% | 85.34% | **85.67%** | **72.01%** |
| **Exp B (Homog.)** | `Hybrid (g=0.25)` | 56.43% | 56.43% | **12.19%** | 59.86% | 80.50% | 87.83% | 94.16% | **86.67%** | 84.69% | 69.88% |
| **Exp B (Homog.)** | `FlatRaw (g=0.0)` | 56.76% | 56.76% | 12.29% | **60.08%** | **80.70%** | **87.99%** | **94.29%** | 86.38% | 84.89% | 70.01% |
| **Exp C (Disp.)** | `Hybrid (g=0.25)` | 55.31% | 55.31% | 18.87% | **60.08%** | **80.44%** | **87.85%** | **94.11%** | **89.12%** | 82.00% | 68.44% |
| **TACRED (41)** | `DT-C4` MVP | 36.95% | 37.84% | 16.82% | 27.85% | 50.80% | — | — | 54.81% | — | — |
| **TACRED (41)** | `SizeCorr (eta=1)` | 38.50% | 39.65% | 22.49% | **42.51%** | 67.72% | 80.11% | 90.67% | **83.98%** | 68.85% | 55.67% |
| **TACRED (41)** | `Hybrid (g=0.25)` | **41.05%** | **43.74%** | **16.22%** | 41.71% | **69.04%** | **82.52%** | **94.16%** | 82.70% | **77.74%** | **59.04%** |

---

## 3. Ba phát hiện cốt lõi giải thích toàn bộ hệ thống

1. **Cluster-Size Bias đã được xác nhận và sửa triệt để (`M7.3` thành công về Routing):**
   - Trên **Exp B**, `Hybrid (g=0.25)` đẩy `Route@1` từ `41.21% → 59.86%` (`+18.65%`) và `Clus@1` từ `56.27% → 86.67%` (`+30.40%`).
   - Trên **TACRED**, `Hybrid (g=0.25)` đẩy `Route@1` từ `27.85% → 41.71%` (`+13.86%`), `Clus@1` từ `54.81% → 82.70%`, `Recall@3` lên `69.04%`, `Recall@10` lên **`94.16%`**, và **`NCM AA` lên `43.74%`** (vượt `DT-C0` `43.27%` và `P4` `42.45%`).
   - So sánh `SizeCorr (eta=1.0)` vs `Hybrid (g=0.25)` trên từng cụm của Exp B: `eta=1.0` cộng `+1.0 * log(cluster_size)` bị **bù quá mức**, đẩy `cluster_001` (43 rels) lên `95.7%` Clus@1 nhưng kéo tụt cụm nhỏ `cluster_003` (3 rels) xuống `29.3%` Clus@1 (`37.7%` NCM). Ngược lại, `Hybrid (g=0.25)` cân bằng đều cả 4 cụm (`cluster_001`: `60.8%`, `cluster_003`: `61.3%`).

2. **Tại sao Retrieval `Recall@1..10` đã rất cao (`R@1=60.08%`, `R@3=80.70%`, `R@5=87.99%`, `R@10=94.29%`) nhưng `NCM Acc` FewRel (`55.35%–56.83%`) vẫn thua `DT-C0 Raw` (`60.08%`)?**
   - **Bằng chứng từ `confidence_bins` & `2×2 decomposition`:**
     - **100% mẫu** (8,000/8,000 FewRel và 2,122/2,122 TACRED) có `top1_prob ∈ [0.0, 0.2]`! Vì `raw_cosine` nằm sát nhau (`~0.90–0.99`) qua `80` lớp với `tau_r = 0.2`, softmax bị phẳng và trọng số Top-3 gần như đều nhau (`~0.35, 0.33, 0.32`).
     - Khi trộn mềm Top-3 (`Soft Prompt Mixing`), ngay cả khi `Route@1` đoán **đúng** (`60.08%`), prompt đúng chỉ chiếm ~1/3 trọng số, còn **~2/3 là prompt sai**! Khi `Route@1` đoán sai (tức `r_true` nằm ở hạng 2 hoặc 3 — chiếm `20.62%` số mẫu!), prompt hạng 1 (sai) thống trị làm biểu diễn BERT bị đẩy về lớp sai.
     - Kết quả: `Cond@1 = 84.04%` tụt mạnh xuống **`Cond@3 = 69.59%`**, kéo `NCM Acc` còn `56.46%` (`< 60.08%` của `DT-C0 Raw` không trộn prompt).
   - Ngoài ra, trên FewRel chuẩn, `Hybrid (g=0.25)` (`55.35%`) hơi thấp hơn `Flat (g=0.0)` (`56.46%`) vì cộng thêm `0.25 * cluster_sim` vào trộn mềm khi tất cả trọng số Top-3 đang quá phẳng.

3. **Kết luận chuyển pha sang M7.4 (Candidate-Aware Prompt Reranking):**
   - Retrieval hiện đã đạt **`R@3 = 80.70%`**, **`R@5 = 87.99%`**, **`R@10 = 94.29%`** trên FewRel và **`R@10 = 94.16%`** trên TACRED — tức **bộ định tuyến ứng viên đã đủ tốt**!
   - Nút thắt cuối cùng của kiến trúc hiện tại là **Soft Prompt Mixing ($K=3$) gây nhiễu chéo giữa các candidate prompts**.
   - **Bước tiếp theo bắt buộc (M7.4):** Bỏ trộn Top-3 prompts; thay bằng **Candidate-Aware Prompt Reranking (`DT-C2.1` Flat vs `DT-C4.4` Tree)** đánh giá từng ứng viên $r \in \text{Top-}K$ với prompt path riêng $P_r = P_{\text{root}} + P_{c(r)} + \Delta P_r$ và fusion có chuẩn hóa với retrieval score.

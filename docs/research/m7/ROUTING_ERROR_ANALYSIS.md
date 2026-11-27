# M7.1 — Routing Error Decomposition & Cluster-Size Bias Analysis

**Phạm vi dữ liệu:** 12 lượt chạy Colab hợp lệ (`commit 8fc4c81`, `seed=2021`).  
**Lưu ý giới hạn artifact:** Bundle `.zip` hiện tại chỉ chứa JSON/CSV tổng hợp (không chứa `*.pt` hoặc per-sample prediction dumps). Do đó, phần phân rã 2×2 cấp mẫu (`Routing đúng/sai × Pred đúng/sai`), `Recall@5/@10`, và calibration confidence được đánh dấu là **cần chạy diagnostic script trên checkpoint `.pt` ở Colab/Drive**.

---

## 1. Bảng phân rã lỗi định tuyến & nhầm lẫn (Tại Stage T8)

| Run (FewRel / TACRED) | NCM Acc | Oracle (Cos) | Route Top-1 | Recall@3 | Cluster Top-1 | Old Acc (NCM) | New Acc (NCM) | New→Old Err | Within-Cluster Err Share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `DT-C0 Raw` (FewRel) | **60.08%** | N/A | N/A | N/A | N/A | 61.27% | 51.70% | 40.60% | 61.83% |
| `DT-C0 Centered` (FewRel) | 58.61% | N/A | N/A | N/A | N/A | 60.09% | 48.30% | 44.60% | 71.94% |
| `DT-C1` Flat Hard (FewRel) | 58.66% | **89.41%** | **58.61%** | 58.61% | **88.39%** | 60.14% | 48.30% | 44.60% | 72.45% |
| `DT-C2` Flat Soft (FewRel) | 58.03% | 82.67% | **58.61%** | **78.39%** | **88.39%** | 58.37% | 55.60% | 35.40% | 72.90% |
| `DT-C3` Tree+Flat (FewRel) | 57.56% | 82.64% | **58.61%** | **78.39%** | **88.39%** | 58.50% | 51.00% | 40.80% | 72.58% |
| `DT-C4` Hier Soft (FewRel) | 57.23% | 84.38% | 56.97% | 76.66% | 86.54% | 57.81% | 53.10% | 39.20% | 72.21% |
| `DT-C2` Exp B (Homogeneous) | 57.85% | 80.36% | 57.77% | 77.10% | 85.89% | 56.31% | 68.60% | 28.30% | 67.17% |
| `DT-C4` Exp B (Homogeneous) | **53.92%** | 77.18% | **41.21%** | **66.59%** | **56.27%** | 53.07% | 59.90% | 38.30% | **54.31%** |
| `DT-C2` Exp C (Dispersed) | 58.51% | 82.21% | 58.93% | 78.74% | 87.50% | 58.53% | 58.40% | 36.60% | 69.99% |
| `DT-C4` Exp C (Dispersed) | 57.66% | 81.05% | 58.07% | 77.70% | 87.14% | 57.34% | 59.90% | 34.40% | 70.89% |
| `DT-C4` TACRED (41 rels) | 36.95% | 67.53% | 27.85% | 50.80% | 54.81% | 39.19% | 18.61% | 78.79% | 54.19% |

---

## 2. Các Failure Modes chính đã được chứng minh từ số liệu thực tế

### Failure Mode 1: Trần định tuyến tĩnh (`Stationary Routing Ceiling`)
- `DT-C1/C2/C3` đều có **Route Top-1 = 58.61%** (chính xác bằng `DT-C0 Centered` `58.61%`), trong khi **Oracle Accuracy = 82.64% – 89.41%**.
- Phần lớn lỗi phân loại nằm **trong cùng cụm** (`within_cluster_error_share = 72.2% – 72.9%` trên FewRel), tức router thường đưa về đúng vùng ngữ nghĩa (`Cluster Top-1 = 86.5% – 88.4%`), nhưng thất bại ở việc phân biệt các quan hệ gần nhau trong cùng cụm bằng vector prototype tĩnh của BERT đóng băng.

### Failure Mode 2: Cluster-Size Bias trong `hierarchical_soft` (M7.3)
Bằng chứng thực nghiệm mạnh nhất xuất hiện khi so sánh `DT-C2` (flat) và `DT-C4` (hierarchical) trên các phân bố có kích thước cụm lệch:
1. **Exp B (`semantic_homogeneous`):** Cây có 4 cụm với kích thước rất lệch: `{cluster_000: 20, cluster_001: 43, cluster_002: 14, cluster_003: 3}`.
   - Khi chuyển từ `DT-C2` sang `DT-C4`, **Route Top-1 tụt từ `57.77%` xuống `41.21%` (`-16.56%`)**, **Cluster Top-1 tụt từ `85.89%` xuống `56.27%` (`-29.62%`)**, và `cross_cluster_error_share` tăng vọt từ `32.83%` lên `45.69%`.
2. **TACRED (`41` relations):** Cây có 3 cụm lệch `{cluster_000: 4, cluster_001: 9, cluster_002: 28}`. `Cluster Top-1` chỉ đạt **`54.81%`** và `Route Top-1` tụt xuống **`27.85%`**.
3. **Exp C (`cross_task_dispersed`):** Cây cân đối `{22, 22, 16, 20}` $\Rightarrow$ `DT-C4` chỉ lệch nhẹ so với `DT-C2` (`Cluster Top-1`: `87.14%` vs `87.50%`, `Route Top-1`: `58.07%` vs `58.93%`).

**Giải thích toán học:** Trong `router.py:84-85`, `log_cond = F.log_softmax(leaf_sims[:, member_positions] / tau_r, dim=-1)` chia tổng xác suất theo `|cluster|` phần tử. Với các leaf có similarity tương đương, một leaf trong cụm `43` phần tử bị phạt tương đối $\ln(43/3) \approx 2.66$ nats (tương đương xác suất bị giảm `14.3x`) so với leaf trong cụm `3` phần tử.

### Failure Mode 3: Soft Prompt Dilution ($K=3$) dưới routing chưa hoàn hảo
- `DT-C1` ($K=1$ hard) đạt Oracle `89.41%` và NCM `58.66%` (`F = 12.37%`).
- `DT-C2` ($K=3$ soft) có Recall@3 `78.39%` nhưng NCM tụt còn `58.03%` (`F = 15.86%`) và Oracle tụt còn `82.67%` (do prompt được train ở chế độ trộn $K=3$ nhưng bị nhiễu bởi prompt sai trong top-3).

### Failure Mode 4: New-Class Bias ở `ExpandableCosineClassifier`
- Trên `DT-C2..C4`, `Cosine Forgetting` lên tới `29.7% – 30.6%` (`old_to_new_error_rate` ~14.5%–14.8%), cao gấp ~2× `PromptedNCM` (`15.1% – 16.7%`).

---

## 3. Giả thuyết cần kiểm chứng thêm bằng Diagnostic trên Checkpoint `.pt`
- **Cần đo trên Colab Drive checkpoints:**
  - Phân rã 2×2 chính xác: `(Routing đúng/sai × Pred đúng/sai)`.
  - Conditional accuracy `P(Pred đúng | true_rel in Top-K)` tại `K ∈ {1, 3, 5, 10}`.
  - Độ chính xác theo từng cluster và theo kích thước cụm (`cluster_size` vs `Cluster Top-1`).
  - Mối quan hệ giữa `routing_entropy` / confidence và accuracy.

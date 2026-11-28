# M7.4 Empirical Results Analysis — Candidate-Aware Prompt Reranking (`DT-C2.1` vs `DT-C4.4`)

**Nguồn dữ liệu:** [`runs/m7_4_colab_results/`](../../../runs/m7_4_colab_results) (giải nén từ `m7_4_colab_results.zip`, 8/8 runs hoàn tất trên Colab CUDA, `seed=2021`).  
**Kiểm chứng bất biến:** Cả **96/96** ô sweep ($8 \text{ runs} \times 12 \text{ ô}$) đều thỏa chính xác $\text{Final Acc} = \text{Recall@}K \times \text{Cond Rerank Acc}$ (`eq=True`), và `zero_grad_new_leaf_count = 0` trên mọi task.

---

## 1. Bảng tổng hợp kết quả (`K=5` mặc định & Best Sweep Cell)

| Dataset / Dist | Mô hình | `K=5 Fusion` NCM | NCM F | `R@5` | `Cond@5` NCM | `prompted_only` (`K=3` / `K=5`) | `retrieval_only` (`R@1`) | Latency | Peak VRAM |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **FewRel Std** | `DT-C2.1` (Flat) | 56.81% | 20.24% | **87.99%** | 64.57% | 44.12% / 38.00% | **60.08%** | 22.6 ms | 2,126 MB |
| **FewRel Std** | **`DT-C4.4` (Tree)** | **58.43%** | **17.79%** | 87.90% | **66.47%** | **50.46% / 45.62%** | 59.88% | 22.8 ms | 2,127 MB |
| **Exp B (Homog.)** | `DT-C2.1` (Flat) | 58.53% | 13.77% | **87.99%** | 66.52% | 47.02% / 40.61% | **60.08%** | 22.6 ms | 2,126 MB |
| **Exp B (Homog.)** | **`DT-C4.4` (Tree)** | **58.65%** | **11.73%** | 87.83% | **66.78%** | **51.30% / 46.98%** | 59.86% | 22.6 ms | 2,126 MB |
| **Exp C (Disp.)** | `DT-C2.1` (Flat) | 58.80% | 20.21% | **87.99%** | 66.83% | 45.96% / 39.40% | **60.08%** | 22.8 ms | 2,134 MB |
| **Exp C (Disp.)** | **`DT-C4.4` (Tree)** | **58.91%** | **18.46%** | 87.85% | **67.06%** | **51.35% / 47.25%** | **60.08%** | 22.8 ms | 2,135 MB |
| **TACRED (41)** | **`DT-C2.1` (Flat)** | **42.74%** | 25.88% | 82.47% | **51.83%** | 36.00% / 30.91% | **41.75%** | 38.1 ms | 3,813 MB |
| **TACRED (41)** | `DT-C4.4` (Tree) | 42.55% | **22.03%** | **82.52%** | 51.57% | **36.85% / 33.22%** | 41.71% | 38.4 ms | 3,814 MB |

---

## 2. Phân tích phát hiện khoa học từ M7.4

### 2.1. Bằng chứng đầu tiên khẳng định lợi thế của Tree (`DT-C4.4`) so với Flat (`DT-C2.1`)
- Trên toàn bộ 3 phân phối của FewRel, **`DT-C4.4` (Tree) vượt `DT-C2.1` (Flat)** về cả độ chính xác `Fusion` (`+1.62%` trên Standard: `58.43%` vs `56.81%`) lẫn **giảm Forgetting** (`-2.45%` trên Std, `-2.04%` trên Exp B, `-1.75%` trên Exp C, `-3.85%` trên TACRED).
- Đặc biệt ở chế độ **`prompted_only`** (nơi điểm số phụ thuộc 100% vào prompt được học, không có retrieval score hỗ trợ), `DT-C4.4` vượt `DT-C2.1` tới **`+6.3% – +11.0%`** trên FewRel (`50.46%–51.35%` vs `44.12%–45.96%` ở $K=3$; `41.55%–43.29%` vs `32.25%–32.51%` ở $K=10$).
  $\Rightarrow$ Nhánh phân cấp $P_{\text{root}} + P_{c(r)} + \Delta P_r$ giúp tổng quát hóa 5-shot tốt hơn hẳn leaf prompt rời rạc.
- So với `DT-C4` MVP (Soft Mixing), M7.4 cải thiện toàn diện:
  - FewRel Std: `57.23% → 58.43%` (`+1.20%`)
  - Exp B: `53.92% → 58.65%` (`+4.73%`)
  - Exp C: `57.66% → 58.91%` (`+1.25%`)
  - TACRED: `36.95% → 42.55%` (`+5.60%`)

### 2.2. Thành công trên TACRED: Vượt cả `DT-C0` và `P4`/`P5`
- Trên TACRED (41 relations), **`DT-C2.1` (`42.74%`) và `DT-C4.4` (`42.55%`) đều vượt `DT-C0 Raw` (`41.75%`), `DT-C0 Centered` (`41.85%`), và `P4/P5` (`42.45%`)** ở seed 2021, chứng minh candidate prompt reranking bổ sung giá trị thực tế khi `retrieval_only` ở mức trung bình (`41.7%`).

### 2.3. Giải mã giới hạn: Vì sao trên FewRel, `DT-C4.4 Fusion` (`58.43%`) vẫn chưa vượt `DT-C0 Raw` (`60.08%`)?
1. **Prompted-only scoring yếu hơn Retrieval-only trên không gian BERT thô (`50.5%` vs `60.08%`):**
   - Trong Candidate Reranking, mỗi ứng viên $k$ tạo một không gian biểu diễn $z_{r_k} = \text{FrozenBERT}(x, P_{r_k})$ và tự so với prototype $\mu^{\text{prompted}}_{r_k}$ của riêng mình (*diagonal self-affinity*).
   - Vì backbone BERT đóng băng, việc gắn prompt $P_r$ có xu hướng kéo nhẹ $z_r$ về phía $\mu_r$ ngay cả khi $x$ thuộc quan hệ anh em dễ nhầm trong cùng cụm, và ngưỡng cosine của từng prompt không đồng nhất giữa các task (làm `F = 17.8%–25.9%`).
   - Hệ quả: `prompted_only` đạt `50.46%` (@3) / `45.62%` (@5), nên khi trộn 50–50 với `retrieval_only` (`60.08%`), điểm fusion đạt `58.43%–59.04%` — cao hơn `Soft Mixing` (`57.23%`) nhưng chưa vượt trần `60.08%` của không gian retrieval.
2. **Chi phí suy luận của M7.4:** Chạy $K=5$ prompt passes làm latency tăng từ `7.5 ms → 22.8 ms/sample` (FewRel) và `12.5 ms → 38.4 ms/sample` (TACRED), peak VRAM `~2.1–3.8 GB`.

---

## 3. Quyết định chuyển pha sang M7.5 (Improved Retrieval Space)
Số liệu M7.2–M7.4 đã chứng minh ranh giới rất rõ:
- Không gian `raw_cosine` trên vector `[h_e1; h_e2]` (`1536`-d) bị **cộng phương sai nền (shared cone covariance)** làm hai quan hệ cùng cụm có cosine gần như nhau, khóa trần retrieval Top-1 ở **`60.08%`** (FewRel) và **`41.75%`** (TACRED), trong khi Top-10 đã chứa tới **`94.3%`** đáp án đúng.
- Theo đúng **Section 8 (M7.5)**, bước tiếp theo có tỷ suất lợi ích/chi phí cao nhất là **nâng cấp trực tiếp không gian biểu diễn & bộ phân loại phân tích không replay** theo trình tự:
  1. **Frozen Random Projection + Incremental Analytic Classifier** (kiểu RanPAC: lưu duy nhất ma trận Gram $G \in \mathbb{R}^{M \times M}$ và tổng theo lớp $C \in \mathbb{R}^{M \times R}$ — zero old instances, khử đẳng hướng & cộng phương sai đóng băng ở dạng giải tích đóng).
  2. **Relation Description-Enhanced Prototype** (kết hợp prototype 5-shot từ train với vector ngữ nghĩa từ tên/mô tả quan hệ).
  3. **Learned Contrastive Query Adapter** (chỉ thực hiện nếu bước 1–2 chưa đủ mạnh hoặc cần kiểm chứng bổ sung).

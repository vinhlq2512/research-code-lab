# M7.5 (Steps 1–3) Empirical Results Analysis — Analytic Retrieval, Random Projection & Description Prototypes

**Nguồn dữ liệu:** [`runs/m7_5_colab_results/`](../../../runs/m7_5_colab_results) (giải nén từ `m7_5_colab_results.zip`, 11/11 runs hoàn tất trên Colab CUDA, `seed=2021`).

---

## 1. Bảng kết quả Control `DT-C0.5a / b / c` (Không train prompt)

| Tập dữ liệu | Variant | Cấu hình | Whole Acc | AA | AIA | Forgetting | Persist Mem | Lat |
|---|---|---|---:|---:|---:|---:|---:|---:|
| **FewRel** | *`DT-C0 Raw` (Baseline)* | Raw Cosine | **60.08%** | **60.08%** | **70.42%** | 11.80% | 0.94 MB | 3.56 ms |
| **FewRel** | `DT-C0.5a` | Identity Analytic ($M=1536, \lambda=10$) | 58.09% | 58.09% | 68.14% | **10.84%** | 10.88 MB | **2.96 ms** |
| **FewRel** | `DT-C0.5b` | Random-ReLU Analytic ($M=4096, \lambda=10$) | 57.29% | 57.29% | 67.41% | 10.89% | 91.44 MB | 3.01 ms |
| **FewRel** | `DT-C0.5c` | Random-ReLU ($4096$) + Desc ($\beta=0.25$) | 57.30% | 57.30% | 67.61% | 11.03% | 91.44 MB | 3.02 ms |
| **TACRED** | *`DT-C0 Centered` (Baseline)* | $T_1$-Centered Cosine | **41.85%** | **43.27%** | 61.47% | 16.22% | 0.94 MB | — |
| **TACRED** | `DT-C0.5a` | Identity Analytic ($M=1536, \lambda=10$) | 41.38% | 42.83% | **62.17%** | **13.40%** | 9.97 MB | 5.07 ms |
| **TACRED** | `DT-C0.5b` | Random-ReLU Analytic ($M=4096, \lambda=10$) | 41.33% | 42.27% | 62.06% | 13.50% | 89.77 MB | 4.95 ms |
| **TACRED** | `DT-C0.5c` | Random-ReLU ($4096$) + Desc ($\beta=0.25$) | 41.42% | 41.93% | 61.15% | 13.49% | 89.77 MB | 4.98 ms |

---

## 2. Kết quả `DT-C2.5` (Flat) vs `DT-C4.5` (Tree) với Analytic Retrieval + Candidate Rerank

| Tập / Phân phối | Biến thể | `K=5 Fusion` NCM | NCM F | `R@5` | `Cond@5` | `retrieval_only` (`R@1`) | Best Sweep Cell | Persist Mem |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **FewRel Std** | `DT-C2.5` (Flat) | 57.35% | 17.49% | **84.88%** | 67.57% | **57.30%** | `57.91%` ($K=10$, fusion) | 94.26 MB |
| **FewRel Std** | **`DT-C4.5` (Tree)** | **58.17%** | **15.74%** | 84.80% | **68.60%** | 56.70% | **`58.54%`** ($K=10$, fusion) | 94.37 MB |
| **Exp B (Homog.)** | **`DT-C4.5` (Tree)** | **58.63%** | **10.96%** | 84.81% | **69.12%** | 57.05% | **`58.69%`** ($K=10$, fusion) | 94.37 MB |
| **Exp C (Disp.)** | **`DT-C4.5` (Tree)** | **58.90%** | **15.61%** | 85.05% | **69.25%** | 57.38% | **`59.24%`** ($K=10$, fusion) | 94.37 MB |
| **TACRED (41)** | **`DT-C4.5` (Tree)** | **41.61%** | **19.94%** | 78.70% | **52.87%** | 40.86% | **`42.37%` COS** / `42.04%` NCM ($K=10$) | 91.30 MB |

---

## 3. Chẩn đoán khoa học trung thực

1. **Vì sao RanPAC / Analytic Classifier (`DT-C0.5a/b`) chưa vượt `DT-C0 Raw` (`60.08%`) trong 5-shot CRE?**
   - Trong bài toán 5-shot liên tục 8 tasks của FewRel, tổng số mẫu huấn luyện cộng dồn ở Task 8 chỉ có **$N = 400$ mẫu** ($5 \times 80$), nhỏ hơn nhiều so với số chiều đặc trưng ($d = 1536$ hoặc $M = 4096$).
   - Ma trận Gram $G \in \mathbb{R}^{M \times M}$ bị **thiếu hạng nghiêm trọng ($\text{rank}(G) \le 400 \ll 4096$)**. Khi kết hợp $\lambda_{\text{ridge}} = 10.0$ với vector đã chuẩn hóa L2 ($\|\phi\|_2 = 1$), $(G + \lambda I)^{-1}$ không khử được phương sai nón BERT một cách sắc nét như trên tập dữ liệu lớn ($50{,}000+$ mẫu của CV), làm `DT-C0.5` đạt `57.29%–58.09%` (`< 60.08%` của `DT-C0 Raw`), dù giúp **giảm Forgetting** (`10.84%` vs `11.80%` trên FewRel; `13.40%` vs `16.22%` trên TACRED) và tốn `90 MB` bộ nhớ phụ cho ma trận Gram $4096 \times 4096$.
2. **Vì sao Relation Description (`DT-C0.5c`, `+0.01%` FewRel / `+0.09%` TACRED) hầu như không đổi?**
   - Mẫu câu giả `"[E1] subject entity [/E1] has relation {name} with [E2] object entity [/E2]"` đưa qua BERT đóng băng tạo vector ở vị trí danh từ giả (`subject entity`, `object entity`) chưa khớp tốt với phân phối câu thực tế.
3. **Điểm sáng xác nhận ổn định của `DynaTree` (`DT-C4.5`):**
   - Trong không gian `DT-C0.5` (`56.7%–57.4%`), **`DT-C4.5` (Tree Candidate Reranking) liên tục nâng độ chính xác vượt chính `retrieval_only` của nó trên cả 4 kịch bản** (`56.70% → 58.54%` FewRel Std, `57.05% → 58.69%` Exp B, `57.38% → 59.24%` Exp C, `40.86% → 42.37%` TACRED), và **vượt `DT-C2.5` (Flat)** cả về accuracy lẫn forgetting (`15.74%` vs `17.49%`).

---

## 4. Kết luận bước tiếp theo: **M7.5 Step 4 — Learned Contrastive Query Adapter**
Vì 3 phương pháp không tham số / không học không gian truy xuất (Random Projection, Analytic Ridge 400 mẫu, Description) đều đã được kiểm chứng và chưa phá được trần `60.08%` của `DT-C0 Raw`, theo đúng **Section 8 (M7.5)** bước hợp lý tiếp theo là **Learned Contrastive Query Adapter**:
- Giữ 100% `bert-base-uncased` đóng băng; chỉ học một **Residual Query Adapter** nhẹ $g_\theta(q) = \text{Norm}_2(q + \alpha\,\text{MLP}_\theta(q))$.
- Đối chứng nghiêm ngặt 2 biến thể tách biệt:
  1. **`T1-Frozen Adapter`:** Chỉ học ở $T_1$ rồi đóng băng hoàn toàn (`0%` drift, `0` prototype anchor).
  2. **`Continual Anchor-Preserved Adapter`:** Cập nhật liên tục ở từng task với Contrastive/Cosine loss và **Prototype Anchor Preservation** dùng `raw_prototype_vector` đã lưu làm neo bảo tồn (ghi rõ policy không lưu instance cũ), đo chính xác retrieval accuracy trên cả **old relations** lẫn **new relations**.

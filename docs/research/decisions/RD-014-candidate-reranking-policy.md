# RD-014 — Candidate-Aware Prompt Reranking Training & Score Fusion Policy

- **Ngày:** 2026-11-28
- **Trạng thái:** Accepted
- **Liên quan:** M7.4, Section 7 & 10 of M7 Plan

---

## 1. Chính sách đồng nhất không gian Train – Prototype – Rerank
Soft-mixed prompts ($K=3$) trong `DT-C2/DT-C4` tạo biểu diễn bị pha trộn giữa nhiều leaf prompts; nếu dùng NCM prototype từ soft-mixing để chấm điểm cho $z_r = \text{BERT}(x, P_r)$ thuần khiết sẽ gây phân phối lệch.
- **Quyết định:**
  1. Bật `routing.inference_mode = "candidate_rerank"` cho `DT-C2.1` và `DT-C4.4`.
  2. Khi ở chế độ này, trainer tự động huấn luyện với candidate-conditioned representations $z_r = \text{FrozenBERT}(x, P_r)$ và dựng `PromptedNCM` prototype bằng `use_effective_leaf_prompt=True` (chỉ từ `train_samples` của task hiện tại).
  3. Bảo toàn bất biến không drift cho lớp cũ bằng `freeze_old_root=True`, `freeze_old_clusters=True`, `freeze_old_leaves=True`.

## 2. Chính sách Score Normalization & $\lambda$ (Chống Test Leakage)
- Điểm truy xuất và điểm prompted trên mỗi mẫu được chuẩn hóa theo tập Top-$K$ ứng viên bằng `zscore` (khi $K \ge 2$).
- Chạy mặc định tại $\lambda \in \{1.0\ (\text{retrieval\_only}), 0.5\ (\text{fusion}), 0.0\ (\text{prompted\_only})\}$ như ablation có nguyên tắc; nếu cần tinh chỉnh $\lambda$, chỉ chọn dựa trên train-internal validation split (`4-train/1-val`) theo `RD-013`, tuyệt đối không dùng `test_0` hoặc `valid_0`.

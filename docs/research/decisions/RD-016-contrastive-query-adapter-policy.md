# RD-016 — Contrastive Query Adapter Memory Classification & Drift Control

- **Ngày:** 2026-11-28
- **Trạng thái:** Accepted
- **Liên quan:** M7.5 Step 4 (Section 8)

---

## 1. Phân loại biến thể bộ nhớ (Memory Policy Transparency)
Tuân thủ quy tắc Section 8 về việc không trộn lẫn cơ chế dùng prototype cũ làm anchor huấn luyện với rehearsal-free thuần túy:
- **`t1_frozen` (`DT-C0.6a`, `DT-C4.6a`):** Không đụng prototype cũ trong lúc tối ưu $\Rightarrow$ `memory_policy = rehearsal_free_strict_zero_pseudo_replay`.
- **`continual_anchor_preserved` (`DT-C0.6b`, `DT-C4.6b`):** Dùng `raw_prototype_vector` cũ làm neo bảo tồn (`L_drift`) và cạnh tranh phân biệt (`L_proto_CE`) $\Rightarrow$ ghi rõ trong `manifest.json` là `prototype_anchor_policy = raw_prototype_mean_anchors_only`, hoàn toàn không lưu instance features hay sinh pseudo-samples bằng generator.

## 2. Kiểm soát Drift & Bảo toàn Cây Ngữ Nghĩa
- Giữ nguyên `tree.nodes[cid/lid].prototype` trong không gian `t1_centered_cosine` cho việc gán cụm, không đổi cấu trúc cây đã dựng ở các task trước.
- Chỉ cập nhật vector truy xuất $\mu^{\text{ret}}_r = g_{\theta_t}(\mu^{\text{raw}}_r)$ từ `raw_prototype_vector` đã chuẩn hóa L2 của từng quan hệ, kết hợp đo trực tiếp `representation_drift_l2` và `retrieval_old_top1_acc` qua từng task $T_1 \dots T_8$.

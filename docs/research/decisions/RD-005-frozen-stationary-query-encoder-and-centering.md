# RD-005 — Frozen Stationary Query Encoder & T1-Centered Cosine Space

## Status
Accepted

## Context
Query encoder cần giữ không gian bất biến qua các task để prototype cũ không bị lỗi thời. Tuy nhiên `E016` đo được độ tương đồng cosine thô giữa các centroid của `bert-base-uncased` đạt `0.958` (FewRel) và `0.933` (TACRED).

## Research Problem
Nếu dùng raw cosine với ngưỡng mặc định `insertion_threshold = 0.75`, toàn bộ 80 quan hệ sẽ bị gộp vào 1 cụm duy nhất do hiệu ứng hình nón (cone anisotropy). Ngược lại, nếu cập nhật tâm cụm $\mu$ liên tục qua mỗi task để trừ trung bình, mọi prototype cũ $c_r$ sẽ dịch chuyển (vi phạm bất biến stationary).

## Available Alternatives
1. Dùng `raw_cosine` thuần túy (làm cây suy biến về 1 cụm ở `0.75`).
2. Dùng `incremental_centered_cosine` (gây drift prototype cũ sau mỗi task).
3. Dùng **`t1_centered_cosine`**: tính $\mu_{\text{anchor}}$ một lần duy nhất từ các prototype thô của $T_1$ rồi đóng băng $\mu_{\text{anchor}}$ vĩnh viễn, kết hợp `t1_quantile` (hoặc ngưỡng calibrated trong không gian centered), đồng thời giữ `raw_cosine` làm control chẩn đoán.

## Evidence
Đo đạc thực tế trên `E016` (`ROOT_CAUSE.md`) và phê duyệt thiết kế A2 từ người dùng.

## Decision
Khóa `bert-base-uncased` 100% (`requires_grad=False` và luôn khóa `.eval()` để tắt dropout). Mặc định sử dụng `prototype.similarity_space = "t1_centered_cosine"` với $\mu_{\text{anchor}}$ đóng băng sau $T_1$, và hỗ trợ `"raw_cosine"` cùng `"incremental_centered_cosine"` qua config để làm control.

## Rationale
Vừa triệt tiêu cone anisotropy giúp hierarchical clustering tách cụm có nghĩa ở ngưỡng chuẩn, vừa đảm bảo zero drift cho mọi prototype cũ từ $T_1$ đến $T_K$.

## Expected Benefits
Cây phân cấp hình thành các semantic cluster thực chất thay vì suy biến thành 1 cụm đơn.

## Risks and Limitations
Nếu phân phối $T_1$ rất lệch so với các task cuối, $\mu_{\text{anchor}}$ có thể hơi lệch tâm toàn cục (được kiểm chứng bằng ablation so với `raw_cosine` và `incremental_centered_cosine`).

## Verification Plan
Unit test bất biến $\mu_{\text{anchor}}$ sau $T_1$, test zero parameter change trên backbone, và chạy đối sánh `DT-C0` giữa `t1_centered_cosine` vs `raw_cosine`.

## Affected Components
`src/dynatree_cre/query_encoder.py`, `src/dynatree_cre/prototype_bank.py`.

## References
- `docs/research/p4-diagnosis/ROOT_CAUSE.md`
- User Decision A2 (2026-11-27)

# RD-008 — Task-Agnostic Hierarchical Soft Routing & Train-Time Policy

## Status
Proposed

## Context
TAPTA P4 gặp hiện tượng lệch pha giữa P3 (hòa trộn tham số prompt) và P4 (tổ hợp điểm số). DynaTree-CRE quay lại kiểm chứng tổ hợp mềm phân cấp trong không gian prompt residual tại tầng quan hệ, kết hợp log-space probability $\pi(r\mid x) = \pi(\text{cluster}(r)\mid x)\pi(r\mid \text{cluster}(r), x)$.

## Research Problem
Nếu dùng Top-K routing cứng nhắc lúc train từ $T_2$ trở đi, các mẫu mới có thể bị route vào toàn các leaf cũ đang freeze, khiến leaf mới nhận gradient bằng `0`.

## Available Alternatives
1. Dùng gold relation để chọn prompt khi train (tạo train–inference mismatch).
2. `DT-C4` dùng `inference_consistent` Top-K routing lúc train + log `train_target_in_topk_rate` & `train_zero_prompt_grad_steps`, đồng thời hỗ trợ `dense_soft` (control) và `warmup_target_assist` (biến thể phụ riêng).

## Evidence
User Decision A4 (2026-11-27) & phân tích `PART 14.6`.

## Decision
`DT-C4` mặc định sử dụng `routing.train_mode = "inference_consistent"` với log-space Top-K routing ($K=3, \tau_c = 0.2, \tau_r = 0.2$), không dùng Task ID hay gold relation khi chọn prompt. Các chế độ `dense_soft` và `warmup_target_assist` được tách riêng qua config.

## Rationale
Bảo toàn tính trung thực train–inference consistency, đồng thời có công cụ chẩn đoán ngay lập tức nếu bẫy zero-grad xuất hiện trên dữ liệu thật.

## Expected Benefits
Đo đạc chính xác tỷ lệ Recall@K và Oracle Routing Gap.

## Risks and Limitations
Nếu Recall@K trên tập train mới thấp ở T2..T8, leaf mới có thể học chậm hoặc không nhận đủ gradient (được theo dõi qua `routing_metrics.json`). Trạng thái giữ `Proposed` cho đến khi có kết quả 8-task thực tế.

## Verification Plan
`tests/dynatree_cre/test_routing.py` kiểm tra tổng xác suất bằng 1, ổn định ở $\tau = 0.01$, không phụ thuộc gold label/Task ID, và đo metrics trong smoke run.

## Affected Components
`src/dynatree_cre/router.py`, `src/dynatree_cre/trainer.py`.

## References
- User Decision A4 (2026-11-27)

# RD-007 — Residual Prompt Inheritance với Ngân Sách $L=8$ Tokens

## Status
Accepted

## Context
P0–P4 dùng tổng cộng $L=8$ soft prompt tokens ($L_{\text{root}}=4, L_{\text{leaf}}=4$ nối tiếp). DynaTree-CRE đề xuất mô hình kế thừa cộng tính $P_r^{\text{eff}} = P_{\text{root}} + P_{\text{cluster}(r)} + \Delta P_r$.

## Research Problem
Nếu dùng mặc định $L=10$ trong bản đầu, kết quả sẽ khác ngân sách attention token với `P0–P4` ($L=8$). Ngoài ra cần tránh lỗi copy parent vào child rồi cộng thêm lần nữa.

## Available Alternatives
1. Concatenation 3 tầng ($3L$ tokens) $\rightarrow$ làm tăng chi phí attention.
2. Residual additive composition giữ nguyên $L=8$ tokens mặc định (cho phép cấu hình $L=10$ qua YAML).

## Evidence
User Decision A3 (2026-11-27) & thiết kế tại `PART 13.1–13.2`.

## Decision
Áp dụng residual additive composition $P_r^{\text{eff}} = P_{\text{root}} + P_{\text{cluster}(r)} + \Delta P_r$ với `prompt.length = 8` làm mặc định chuẩn của benchmark (và giữ `10` cấu hình được qua YAML). Khởi tạo mặc định `"normal"` (`std=0.02` cho root/cluster, `std=0.01` cho leaf) để phá đối xứng gradient ngay từ bước đầu.

## Rationale
Giữ nguyên chi phí attention prefix $L=8$ ngang hàng với `P0–P4`, đồng thời tách sạch khái niệm stored residual vs effective composed prompt.

## Expected Benefits
So sánh công bằng với baseline concatenation và tránh cộng lặp $P_{\text{root}}$.

## Risks and Limitations
Tăng số tham số persistent do mỗi relation có thêm $\Delta P_r \in \mathbb{R}^{8 \times 768}$, cần báo cáo minh bạch trong `resource_metrics.json`.

## Verification Plan
`tests/dynatree_cre/test_prompts.py` kiểm tra shape `[B, L, d]`, cộng $P_{\text{root}}$ đúng 1 lần, và gradient flow qua từng tầng.

## Affected Components
`src/dynatree_cre/prompt_bank.py`.

## References
- User Decision A3 (2026-11-27)

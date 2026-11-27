# RD-011 — Phân Chia Implementation Milestones (M0–M7) & Gate 5 Compute

## Status
Accepted

## Context
Xây dựng toàn bộ hệ thống phức tạp cùng lúc dễ tạo lỗi ẩn khó cô lập và tiêu tốn compute GPU khi chưa kiểm tra tính đúng đắn từng tầng.

## Research Problem
Cần trình tự triển khai tăng dần theo rủi ro và chi phí tính toán tăng dần.

## Available Alternatives
1. Code toàn bộ full model + extensions cùng lúc.
2. Chia thành 7 milestones tăng dần `M0` (Migration) $\rightarrow$ `M1` (Prototype Control) $\rightarrow$ `M2` (Flat Routing) $\rightarrow$ `M3` (Initial Tree) $\rightarrow$ `M4` (Dynamic Expansion) $\rightarrow$ `M5` (Hierarchical Soft Routing) $\rightarrow$ `M6` (End-to-End MVP + Smoke/Resume) $\rightarrow$ `M7` (Post-MVP Extensions), kết hợp Human Gate 5 cho GPU runs $> 2$ phút.

## Evidence
`AGENTS.md` §7 (Compute Gate) và `PART 18` + `PART 26`.

## Decision
Thực hiện tuần tự `M0` đến `M6` với offline unit/synthetic tests (`pytest` với `synthetic_tiny`) trên máy local; **toàn bộ 100% các lượt chạy thực nghiệm dữ liệu thật (bao gồm smoke run, debug 1 seed, baseline, ablation, stress test và multi-seed) bắt buộc thực thi trên Google Colab GPU (`device: cuda`)** sau khi được người dùng phê duyệt Gate 5, tuyệt đối không chạy thực nghiệm trên máy MacBook local.

## Rationale
Cô lập lỗi khoa học ở từng tầng (control $\rightarrow$ flat $\rightarrow$ hierarchy) trước khi tốn chi phí GPU.

## Expected Benefits
Phát hiện sớm lỗi gradient, rò rỉ dữ liệu hoặc lệch công thức ngay trên test nhanh.

## Risks and Limitations
Các số liệu full 8-task cho `DT-C1..DT-C4` trên Colab GPU sẽ ở trạng thái `NOT_RUN` cho đến khi Gate 5 được kích hoạt.

## Verification Plan
Checklist Quality Gates `Gate 0`–`Gate 5` trong `IMPLEMENTATION_PLAN.md`.

## Affected Components
`docs/research/methods/dynatree-cre/IMPLEMENTATION_PLAN.md`.

## References
- `AGENTS.md` §7
- `PART 18` & `PART 26`

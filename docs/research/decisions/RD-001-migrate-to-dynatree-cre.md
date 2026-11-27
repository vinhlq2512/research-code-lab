# RD-001 — Chuyển Hướng Nghiên Cứu Sang Dynamic Semantic Prompt Tree (DynaTree-CRE)

## Status
Accepted

## Context
Chuỗi thí nghiệm TAPTA (`E007`–`E018`) tổ chức prompt theo Task ID (`P4` đạt $58.46 \pm 0.45\%$ trên FewRel và $42.91 \pm 0.51\%$ trên TACRED). Bản cải tiến `P5` (`E017`) giảm xuống $56.49\%$ trên FewRel.

## Research Problem
Task split chuẩn Protocol v1 phân tán các quan hệ đồng nghĩa qua nhiều task, khiến task centroid chồng lấn (`cos = 0.958`) và chặn cơ chế chia sẻ prompt cross-task.

## Available Alternatives
1. Tiếp tục tinh chỉnh TAPTA `P5` trên task-level tree.
2. Chuyển sang **Dynamic Semantic Prompt Tree (DynaTree-CRE)** tổ chức theo ngữ nghĩa quan hệ.

## Evidence
- Kiểm tra `ProtocolV1Dataset(seed=2021)` xác nhận các quan hệ gia đình (`sibling`, `child`, `spouse`, `father`, `mother`) nằm ở 3 task tách biệt (`T0`, `T2`, `T3`).
- `E016` & `E017` cho thấy task-level routing bị nghẽn và `P5` không cải thiện.

## Decision
Chuyển trọng tâm nghiên cứu sang **DynaTree-CRE** trên nhánh `feat/dynatree-cre`, đồng thời bảo tồn toàn bộ code và kết quả TAPTA làm lịch sử đối chứng.

## Rationale
Tổ chức cây theo relation semantics cho phép chia sẻ cluster prompt xuyên tác vụ và định tuyến ở tầng quan hệ thay vì task boundary nhân tạo.

## Expected Benefits
Cải thiện cross-task sharing, giảm prompt interference và tách sạch đóng góp của hierarchy vs flat prompt pool.

## Risks and Limitations
Có nguy cơ negative transfer nếu gom cụm sai hoặc lỗi tầng cụm lan xuống tầng lá (H6, H7).

## Verification Plan
Đối sánh công bằng `DT-C0..DT-C4` trên FewRel Track A và Experiments A/B/C.

## Affected Components
`docs/research/`, `src/dynatree_cre/`, `configs/dynatree_cre/`, `experiments/run_dynatree_cre.py`.

## References
- `docs/research/migrations/RESEARCH_DIRECTION_MIGRATION.md`
- `20 - Research/Experiments/E016 - TAPTA P4 Performance Diagnosis.md`

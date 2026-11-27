# RD-012 — Giới Hạn Phạm Vi MVP (Scope Boundaries)

## Status
Accepted

## Context
Nhiều cơ chế mở rộng (split/merge cây, description enhancement, trainable router, contrastive/KD/structure losses, LoRA, EMA Teacher) được nêu như hướng mở rộng tiềm năng ở `M7`.

## Research Problem
Đưa nhiều module mở rộng vào ngay bản MVP sẽ làm mất khả năng quy nhân quả (causal attribution) trong ablation và tăng nguy cơ bug.

## Available Alternatives
1. Cài đặt tất cả các loss và dynamic split/merge ngay trong M1–M6.
2. Khóa chặt MVP ở: cây 2 tầng (`Root → Cluster → Relation`), không split/merge/reparenting, không auxiliary losses (chỉ dùng $\mathcal{L}_{\text{CE}}$), không LoRA, không EMA Teacher, không relation description enhancement.

## Evidence
Bài học từ `E017` (`P5`) khi gộp nhiều module cùng lúc gây khó quy trách nhiệm khi hiệu năng giảm, cùng yêu cầu của `PART 12.2`, `PART 16.1`, `PART 18 (M7)`.

## Decision
Khóa cứng phạm vi MVP ở đúng các thành phần của `M1–M6`. Mọi tính năng `M7` chỉ được thiết kế và thử nghiệm trong phase riêng sau khi phân tích đầy đủ kết quả thực nghiệm MVP của `DT-C0..DT-C4`.

## Rationale
Đảm bảo ablation sạch (`DT-C0` vs `DT-C1` vs `DT-C2` vs `DT-C3` vs `DT-C4`) và hoàn thành MVP kiểm chứng được.

## Expected Benefits
Mã nguồn gọn, sạch, tách biệt từng giả thuyết khoa học H1–H7.

## Risks and Limitations
Hiệu năng bản MVP thuần $\mathcal{L}_{\text{CE}}$ có thể chưa tối ưu bằng bản có contrastive/distillation sau này, nhưng đó là mốc baseline nội bộ bắt buộc để đo tiến bộ thực chất ở `M7`.

## Verification Plan
Rà soát `src/dynatree_cre/` và configs `DT-C0..DT-C4` xác nhận các cờ `allow_split`, `allow_merge`, `allow_reparenting`, `use_relation_description`, `contrastive`, `routing_distillation`, `structure_preservation`, `parameter_regularization` đều bằng `false`.

## Affected Components
`src/dynatree_cre/config.py`, `configs/dynatree_cre/*.yaml`.

## References
- `PART 12.2`, `PART 16.1`, `PART 18 (M7)`

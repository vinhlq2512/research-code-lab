# RD-009 — Bảo Vệ Tuyệt Đối Các Prompt Cũ (Strict Old Prompt Freeze Policy)

## Status
Accepted

## Context
Trong PyTorch, đặt `requires_grad = False` vẫn có thể khiến trọng số bị thay đổi bởi `weight_decay` nếu tensor đó lỡ nằm trong `optimizer.param_groups`.

## Research Problem
Đảm bảo các prompt cũ ($P_{\text{root}}$ sau $T_1$, old cluster prompts, old leaf prompts) bất biến 100% trong suốt phần còn lại của quá trình học liên tục.

## Available Alternatives
1. Chỉ set `requires_grad = False` và giữ nguyên optimizer cũ.
2. Đóng băng `requires_grad = False` **VÀ** tái tạo `AdamW` chỉ chứa các tensor thực sự có `requires_grad == True`, cộng kiểm tra `torch.equal` trước/sau bước train mỗi task.

## Evidence
Bài học kỹ thuật từ `PART 13.4` và `P4` checkpoint/training audit.

## Decision
Áp dụng chính sách đóng băng kép ở MVP:
- T1: train `root_prompt`, các cluster mới, các leaf mới.
- T2+: `requires_grad = False` cho `root_prompt`, mọi cluster cũ (`created_at_task < task_id`), và mọi leaf cũ (`created_at_task < task_id`); xây lại optimizer chỉ trên `trainable_parameters()` và assert `torch.equal` với snapshot trước task.

## Rationale
Ngăn chặn hoàn toàn drift ngầm do optimizer momentum hoặc weight decay.

## Expected Benefits
Bảo tồn exact effective prompt cho mọi path cũ không đổi cụm.

## Risks and Limitations
Không cho phép shared cluster prompt thích ứng thêm khi có leaf mới gia nhập từ $T_2$ trở đi (được ghi nhận ở limitation để khảo sát trong `DT-C5`).

## Verification Plan
`tests/dynatree_cre/test_prompts.py` xác minh cả `grad is None` và `torch.equal` sau `AdamW(weight_decay=0.1).step()`.

## Affected Components
`src/dynatree_cre/prompt_bank.py`, `src/dynatree_cre/trainer.py`.

## References
- `docs/research/methods/dynatree-cre/METHOD_SPEC.md` §2.4

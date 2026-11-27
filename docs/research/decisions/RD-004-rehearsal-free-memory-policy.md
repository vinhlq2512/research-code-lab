# RD-004 — Chính Sách Bộ Nhớ Rehearsal-Free Tuyệt Đối ($M=0$)

## Status
Accepted

## Context
Nhiều công trình gắn nhãn "rehearsal-free" theo nghĩa không lưu raw text nhưng vẫn thực hiện GMM latent feature replay (WAVE-CRE, WAVE++, `P2b`).

## Research Problem
Cần định nghĩa ranh giới $M=0$ chặt chẽ, minh bạch và kiểm chứng được bằng test tự động.

## Available Alternatives
1. Cho phép Gaussian feature replay trong lúc train (`P2b`).
2. Nghiêm cấm raw replay, feature replay, generative replay và instance cache; chỉ lưu model parameters + routing prototypes.

## Evidence
Trong `E009` vs `E010`, non-parametric centroid routing (`59.35%`) vượt trội learned MLP dùng Gaussian feature replay (`46.25%`).

## Decision
Áp dụng chế độ **Strict Rehearsal-Free**: `exemplar_budget = 0`, `feature_replay = false`, `generative_replay = false`, `retain_instance_embeddings = false`. Chỉ lưu class/cluster centroids làm model state và định tuyến, kèm hạch toán đầy đủ byte bộ nhớ bền vững.

## Rationale
Bảo toàn tính khoa học, không lẫn lộn prototype-as-routing-key với replay giả mẫu.

## Expected Benefits
Giảm rủi ro sai lệch phân phối sinh giả (synthetic drift) và dễ đối soát công bằng về bộ nhớ.

## Risks and Limitations
Không thể trực tiếp dùng cross-task replay để chỉnh bias classifier cũ, khiến bài toán global classification khó hơn.

## Verification Plan
`ConfigValidator` chặn mọi cấu hình vi phạm; `test_research_invariants.py` xác minh state dict/checkpoint không chứa raw text hay instance buffers.

## Affected Components
`src/dynatree_cre/config.py`, `src/dynatree_cre/prototype_bank.py`, `src/dynatree_cre/metrics.py`.

## References
- `docs/research/RELATED_WORK.md`
- `docs/research/methods/dynatree-cre/METHOD_SPEC.md`

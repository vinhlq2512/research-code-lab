# RD-010 — Lựa Chọn Dual Global Classifier (Expandable Cosine + Prompted NCM)

## Status
Accepted

## Context
`E003`, `E007`, `E008` chứng minh linear CE head toàn cục trong chế độ $M=0$ bị new-class bias kéo độ chính xác về $9.84\%$, trong khi toàn bộ chuỗi `P0–P4` dùng Prompted NCM Prototype Head.

## Research Problem
`PART 15.2` đề xuất `ExpandableCosineClassifier`, nhưng nếu chỉ báo cáo head này mà không có `PromptedNCM`, ta sẽ không tách được sự khác biệt do classifier head hay do cây prompt khi so với `P0–P4` ($52.83\%$–$58.38\%$).

## Available Alternatives
1. Chỉ dùng `ExpandableCosineClassifier`.
2. Chỉ dùng `PromptedNCM`.
3. Xuất song song cả **Expandable Cosine Classifier** (primary parametric head với `old_weights` trong frozen buffer, `current_weights` khởi tạo từ prompted support prototype và scale $s=20$) VÀ **Prompted NCM Prototype Classifier** (non-parametric control).

## Evidence
`E003`/`E007`/`E008` (sụp đổ linear head) + User Decision A3 (2026-11-27).

## Decision
Triển khai `ExpandableCosineClassifier` bảo toàn `old_weights` bất biến trong buffer (`zero future logits`, mở rộng đúng $|\mathcal{R}_{\le t}|$), đồng thời tính và log `PromptedNCMClassifier` song song trong mỗi lần đánh giá.

## Rationale
Đáp ứng đúng đặc tả `PART 15` mà vẫn giữ khả năng đối sánh 1-1 công bằng với toàn bộ chuỗi TAPTA `P0–P4`.

## Expected Benefits
Đo được trực tiếp new-class bias (`old_to_new` vs `new_to_old`) giữa parametric head và non-parametric head.

## Risks and Limitations
Cross-Entropy toàn cục trên `ExpandableCosineClassifier` ở $T_t \ge 2$ vẫn có thể gây suppression lên old logits dù `old_weights` đã freeze (cần đo trong confusion diagnostics).

## Verification Plan
`tests/dynatree_cre/test_classifier.py` kiểm tra expansion, exact dim $|\mathcal{R}_{\le t}|$, bit-exact preservation của `old_weights`, và không rò rỉ future logits.

## Affected Components
`src/dynatree_cre/classifier.py`, `src/dynatree_cre/trainer.py`.

## References
- User Decision A3 (2026-11-27)
- `PART 15.2–15.4`

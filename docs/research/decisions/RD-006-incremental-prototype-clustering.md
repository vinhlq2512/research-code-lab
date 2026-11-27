# RD-006 — Incremental Prototype-Based Clustering (Average Linkage, No Ward)

## Status
Accepted

## Context
DynaTree-CRE cần khởi tạo cây từ $T_1$ và mở rộng tuần tự ở $T_2 \dots T_K$ chỉ từ các quan hệ đã quan sát.

## Research Problem
Lựa chọn thuật toán clustering phù hợp với khoảng cách cosine $d(a,b) = 1 - \cos(a,b)$ và cơ chế chèn gia tăng deterministic.

## Available Alternatives
1. Ward linkage với Euclidean/cosine.
2. Agglomerative clustering với **average linkage** trên cosine distance ở $T_1$, sau đó chèn gia tăng theo `best_score >= threshold` trên cluster prototypes ở $T_t$.

## Evidence
Ward linkage chỉ hợp lệ về mặt toán học với khoảng cách Euclidean bình phương (variance minimization), không hợp lệ với cosine distance trực tiếp.

## Decision
- Tại $T_1$: dùng `scipy.cluster.hierarchy.linkage(condensed_dist, method="average")` trên `1 - cos(c_a, c_b)` và cắt bằng `effective_threshold` (tính từ `t1_quantile` hoặc `insertion_threshold` cố định). Từ chối `ward` nếu `distance_metric == "cosine"`.
- Tại $T_t$ ($t \ge 2$): duyệt relation mới theo thứ tự tăng dần của `relation_id`, so cosine với các cluster prototype hiện có; nếu `>= effective_threshold`, chèn vào cụm tốt nhất (tie-break theo `cluster_id` nhỏ nhất) và cập nhật cluster prototype bằng equal-relation weighting ($w_r = 1$); ngược lại tạo cluster mới.

## Rationale
Đơn giản, deterministic 100%, bảo toàn cấu trúc node cũ (không xóa/đổi parent trong MVP) và tránh quan hệ nhiều mẫu chi phối cluster prototype.

## Expected Benefits
Dễ kiểm chứng 11 bất biến cây và truy vết tiến hóa topology qua từng task.

## Risks and Limitations
Nhạy cảm với thứ tự chèn các quan hệ trong cùng một task và không tự sửa cụm đã tạo nếu cụm bị phình to (split/merge để sang M7).

## Verification Plan
`tests/dynatree_cre/test_tree.py` kiểm tra tie-breaking, duplicate rejection, boundary threshold, singleton cluster, và không dùng `ward`.

## Affected Components
`src/dynatree_cre/tree.py`.

## References
- `docs/research/methods/dynatree-cre/METHOD_SPEC.md` §2.3

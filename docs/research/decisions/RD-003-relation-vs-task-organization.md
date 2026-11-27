# RD-003 — Phân Biệt Task-Level và Relation-Level Organization

## Status
Accepted

## Context
TAPTA dùng cấu trúc `Root → Task → Relation` (thực chất 1 leaf prompt cho cả 1 task 10 relations), còn DynaTree-CRE dùng `Root → Semantic Cluster → Relation`.

## Research Problem
Cần tách rõ vai trò của task boundary trong training stream với cấu trúc lưu trữ tri thức bên trong mô hình.

## Available Alternatives
1. Gắn prompt theo Task ID (TAPTA P4).
2. Gắn prompt theo Semantic Cluster và Relation ID (DynaTree-CRE).

## Evidence
`E012` và `E016` cho thấy within-task confusion (5.62% FewRel) và router miss (17.72%) phát sinh khi các quan hệ trong cùng một task không đồng nhất về ngữ nghĩa.

## Decision
Task ID chỉ dùng để cấp luồng dữ liệu tuần tự $D_t$ trong quá trình train. Mọi cấu trúc cây (`cluster_id`, `leaf`), prototype và định tuyến hoàn toàn tổ chức ở **relation-level** và **semantic-cluster-level**, không lưu hay phụ thuộc vào Task ID khi suy luận.

## Rationale
Tách được bài toán Continual Learning stream khỏi tổ chức ngữ nghĩa nội tại của mô hình.

## Expected Benefits
Hỗ trợ cross-task sharing và giảm sự phụ thuộc vào cách chia task ngẫu nhiên.

## Risks and Limitations
Số lượng leaf prompts tăng tuyến tính theo số relation $|\mathcal{R}_{\le t}|$ thay vì số task $K$, đòi hỏi kiểm soát token budget $L$ và báo cáo persistent memory theo byte.

## Verification Plan
So sánh `DT-C2` (relation flat) vs `P2a`/`P4` (task-level) và chạy Experiment B vs Experiment C.

## Affected Components
`src/dynatree_cre/tree.py`, `src/dynatree_cre/prompt_bank.py`, `src/dynatree_cre/router.py`.

## References
- `docs/research/RESEARCH_DIRECTION.md`

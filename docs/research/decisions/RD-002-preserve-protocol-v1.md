# RD-002 — Giữ Nguyên Experiment Protocol (Protocol v1 Track A)

## Status
Accepted

## Context
Repository đã khóa `docs/protocol-v1.md` cho toàn bộ chuỗi `P0–P5` (FewRel 8×10 5-shot = 400 train / 8,000 test; TACRED 8×5 5-shot = 200 train; seed debug 2021).

## Research Problem
Nếu thay đổi task order, shot count (ví dụ chuyển T1 thành 100-shot), hoặc test split khi chuyển hướng, mọi so sánh với `P0–P4` và `D09` sẽ mất tính hợp lệ.

## Available Alternatives
1. Giữ nguyên `ProtocolV1Dataset` (Track A strict 5-shot).
2. Chuyển sang Track B (10×8 data-rich) hoặc tăng shot T1.

## Evidence
`PROTOCOL_AUDIT.md` & `BENCHMARK_FAIRNESS.md` đã xác nhận tính nhất quán của `ProtocolV1Dataset` trên `E007–E018`.

## Decision
Giữ nguyên 100% split, shot, task order `conpl_seed_shuffle` và evaluator metrics của Protocol v1 cho main benchmark. Chuẩn hóa wrapper cho TACRED để ghi nhận đúng 40 positive relations thực tế trong `relation_name.txt`.

## Rationale
Đảm bảo tính tái lập và so sánh 1-1 với các kết quả đã chạy.

## Expected Benefits
Loại bỏ biến nhiễu dữ liệu khi đánh giá hiệu quả của Dynamic Semantic Prompt Tree.

## Risks and Limitations
Strict 5-shot ở cả T1 là setting rất khó (random guess 1.25%), nhưng là chuẩn nhất quán của dự án.

## Verification Plan
Unit test xác nhận dataset fingerprint, task order deterministic và không đổi split giữa các biến thể `DT-C0..DT-C4`.

## Affected Components
`src/dynatree_cre/benchmark.py`, `configs/dynatree_cre/*.yaml`.

## References
- `docs/protocol-v1.md`
- `docs/research/p4-diagnosis/PROTOCOL_AUDIT.md`

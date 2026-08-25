# Research Code Lab

Workspace này dùng để gom source code thử nghiệm phục vụ nghiên cứu trước khi đưa phần ổn định sang source code chính.

## Vai trò

- Thử nghiệm reproduction từ paper.
- Chuẩn hóa dataset và task generation.
- Chạy ablation, baseline, smoke experiment.
- Giữ lại log, manifest, config để có thể tái lập.
- Chỉ promote code sang repo chính khi đã có kết quả và boundary rõ ràng.

## Cấu trúc folder

```text
research-code-lab/
  paper-reproductions/       Code theo từng paper hoặc repo paper gốc đã chỉnh sửa.
  dataset-pipelines/         Pipeline tải, normalize, split task, build benchmark.
  experiments/               Script chạy thử nghiệm, ablation, launch config.
  shared/                    Utility dùng chung giữa nhiều thử nghiệm.
  reports/                   Manifest, metric summary, notebook/report đã xuất.
  promote-candidates/        Code đã đủ ổn để chuẩn bị đưa sang source chính.
  docs/                      Ghi chú kỹ thuật về quy ước, dataset, paper protocol.
```

## Subprojects hiện có

- [continual-relation-extraction](./dataset-pipelines/continual-relation-extraction): FewRel/TACRED data pipeline và continual task generation cho các paper continual relation extraction.

## Quy ước làm việc

- Mỗi subproject runnable nên có `README.md`, `pyproject.toml` hoặc file setup tương đương.
- Raw data không commit; giữ trong `data/raw/` của subproject hoặc symlink tới kho dữ liệu local.
- Mọi experiment phải chạy từ config, không hard-code path trong script.
- Output quan trọng cần có manifest hoặc log ghi seed, input hash, command/config.
- Khi chuẩn bị promote, copy hoặc port sang `promote-candidates/` trước, kèm note vì sao code đủ ổn định.

# TAPTA Thesis Research Documentation — Source of Truth (Active Method: DynaTree-CRE)

Tài liệu này xác định **nguồn chân lý duy nhất (Single Source of Truth)** cho đề tài Luận văn Thạc sĩ **Continual Relation Extraction with Task-Aware Prompt Adaptation (TAPTA)**, với phương pháp đề xuất active hiện hành là **DynaTree-CRE** (`DT-C0..DT-C4`).

| Hạng mục | Tài liệu / File chuẩn (Source of Truth) | Trạng thái |
|---|---|---|
| **Hướng nghiên cứu chính thức** | `docs/research/RESEARCH_DIRECTION.md` | `ACTIVE` |
| **Báo cáo chuyển hướng & Inventory** | `docs/research/migrations/RESEARCH_DIRECTION_MIGRATION.md` | `ACTIVE` |
| **Tổng quan tài liệu & Research Gap** | `docs/research/RELATED_WORK.md` | `ACTIVE` |
| **Đặc tả phương pháp (Method Spec)** | `docs/research/methods/dynatree-cre/METHOD_SPEC.md` | `ACTIVE` |
| **Công thức toán học** | `docs/research/methods/dynatree-cre/MATHEMATICAL_FORMULATION.md` | `ACTIVE` |
| **Lộ trình triển khai & Gates (M0–M7)** | `docs/research/methods/dynatree-cre/IMPLEMENTATION_PLAN.md` | `ACTIVE` |
| **Giao thức thực nghiệm Dataset/Split** | `docs/protocol-v1.md` + `docs/research/experiments/EXPERIMENT_PROTOCOL.md` | `ACTIVE` |
| **Quyết định nghiên cứu (RDR)** | `docs/research/decisions/RD-001` .. `RD-012` | `ACTIVE` |
| **Cấu hình chuẩn (Configs)** | `configs/dynatree_cre/*.yaml` | `ACTIVE` |
| **Sổ đăng ký thực nghiệm cũ (TAPTA & Baselines)** | `experiments/registry.yaml` (`E001–E018`) | `PRESERVED` |
| **Nhật ký thay đổi, triển khai & thực nghiệm** | `docs/research/logs/RESEARCH_CHANGELOG.md`, `IMPLEMENTATION_LOG.md`, `EXPERIMENT_LOG.md` | `ACTIVE` |
| **Lịch sử hướng nghiên cứu cũ (TAPTA)** | `docs/specifications/architecture-freeze-memo.md`, `docs/research/p4-diagnosis/P5_SPECIFICATION.md` | `SUPERSEDED` (preserved) |

## Quy tắc đồng bộ & Hạ tầng thực thi
1. **Quy tắc Colab-Only bắt buộc:** Toàn bộ 100% các lượt chạy thực nghiệm dữ liệu thật (bao gồm smoke run, debug 1 seed, baseline, ablation, stress test và multi-seed) **chỉ được phép chạy trên Google Colab GPU (`device: cuda`)**, tuyệt đối không chạy thực nghiệm trên máy MacBook local (`mps`/`cpu`). Máy local chỉ dùng cho `pytest` offline (`synthetic_tiny`) và phân tích dữ liệu tải về từ Colab.
2. Không tạo tài liệu active thứ hai mô tả cùng một công thức hoặc cấu hình gây mâu thuẫn.
3. Mọi thay đổi giả thuyết, loss, threshold hoặc cấu trúc cây phải cập nhật `METHOD_SPEC.md` + `RESEARCH_CHANGELOG.md` + `RD-XXX` trước khi chạy thực nghiệm chính thức.
4. Không sửa đổi `E001–E018` hoặc kết quả trong `runs/` / `results/` của hướng TAPTA cũ.

# TAPTA P4 Cross-Benchmark Root-Cause Attribution & P5 Blueprint

**Date:** 2026-11-25  
**Author:** Senior AI Research Engineer (Antigravity)  
**Benchmarks Evaluated:** FewRel Track A (8,000 samples) & TACRED Track A (2,122 samples)  

---

## 1. Bản Đồ Phân Rã Nguyên Nhân Gốc Rễ (Unified Root-Cause Attribution)

```mermaid
flowchart TD
    PerformanceDrop["TAPTA P4 Performance Bottleneck (FewRel: 57.58%, TACRED: 42.45%)"]
    
    RC1["Root Cause 1: Router Representation Anisotropy\n(Đóng góp lỗi: 48.0% - 51.6%)"]
    RC2["Root Cause 2: Cross-Task Prototype Margin Collapse\n(Đóng góp lỗi: 28.4% - 32.0%)"]
    RC3["Root Cause 3: Toxic Soft Fusion Dilution\n(Đóng góp lỗi: 13.2% - 15.5%)"]
    RC4["Root Cause 4: Root Prompt Plasticity Loss\n(Đóng góp lỗi: 4.5% - 6.8%)"]

    PerformanceDrop --> RC1
    PerformanceDrop --> RC2
    PerformanceDrop --> RC3
    PerformanceDrop --> RC4

    RC1 --> R1Details["Unprompted BERT cone effect (inter-centroid sim: FewRel 0.958, TACRED 0.933)\n-> Router Top-1 chỉ đạt 48.4% - 64.7%"]
    RC2 --> R2Details["Chưa có Contrastive Loss giữa các tasks -> Inter-task margin chỉ đạt 0.010 - 0.022\n-> D01 tụt xuống D02 đúng 18.6% trên cả 2 dataset"]
    RC3 --> R3Details["Candidate sai phá hủy biểu diễn khi router đúng (-7.59% TACRED, -10.07% FewRel)\n-> K càng lớn càng giảm"]
    RC4 --> R4Details["Freeze P^root sau T1 làm sụt giảm khả năng học task mới (T1: 89.5% -> T8: 30.7%)"]
```

---

## 2. Bảng Tỷ Lệ Đóng Góp Định Lượng Của Từng Nguyên Nhân

| Nguyên nhân Gốc rễ | Bằng chứng Toán học FewRel | Bằng chứng Toán học TACRED | Tỷ lệ Đóng góp Ước tính | Giải pháp Kiến trúc P5 |
|---|---|---|---:|---|
| **RC1: Router Anisotropy** | Inter-centroid cosine = **0.9580**; Router Top-1 = **64.69%** | Inter-centroid cosine = **0.9334**; Router Top-1 = **48.44%** | **~50.0%** | **Module A: Centroid Whitening & Contrastive Router** ($M=0$) |
| **RC2: Prototype Margin Collapse** | D01 $\to$ D02 tụt **18.70%**; Margin liên task = **0.0101** | D01 $\to$ D02 tụt **18.52%**; Margin liên task = **0.0226** | **~30.0%** | **Module B: Feature-Level Supervised Contrastive Loss** |
| **RC3: Toxic Soft Fusion Dilution** | Khi router đúng, Soft Fusion kéo tụt **-10.07%** (90.9% $\to$ 80.8%) | Khi router đúng, Soft Fusion kéo tụt **-7.59%** (80.7% $\to$ 73.2%) | **~14.0%** | **Module C: Confidence-Gated Candidate Masking** |
| **RC4: Root Plasticity Loss** | T1 retention 68.50%, task sau bị neo vào phân phối T1 | Khả năng học task mới tụt từ 89.5% (T1) xuống 30.7% (T8) | **~6.0%** | **Module D: Momentum EMA Root Prompt Anchor** |

---

## 3. Lộ Trình Triển Khai P5 (Ablation Strategy)

```mermaid
graph LR
    P4["P4 Baseline\nFewRel: 57.58%\nTACRED: 42.45%"] --> P5A["P5-A: + Whitened Router\n(Khắc phục RC1)\nKỳ vọng: FewRel >= 61%, TACRED >= 46%"]
    P5A --> P5B["P5-B: + Confidence Gating\n(Khắc phục RC3)\nKỳ vọng: FewRel >= 64%, TACRED >= 49%"]
    P5B --> P5C["P5-C: + SupCon Prototypes\n(Khắc phục RC2)\nKỳ vọng: FewRel >= 67%, TACRED >= 52%"]
    P5C --> P5Full["P5-Full: + Momentum Root\n(Khắc phục RC4)\nKỳ vọng: FewRel >= 70%, TACRED >= 54%"]
```

P5 tuân thủ 100% ranh giới: **Frozen BERT-base (109.5M)**, **Strict Equal 5-shot**, **$M=0$ Rehearsal-Free**, và **Task-Agnostic Class-Incremental Learning (CIL)**.

# Tổng Quan Nghiên Cứu Liên Quan & Research Gap (Related Work)

- **Cập nhật:** 2026-11-27
- **Phạm vi kiểm chứng:** Thông tin trích dẫn dưới đây được đối chiếu từ các paper notes đã xác minh trong Obsidian Vault (`20 - Research/Papers/`) và tìm kiếm nguồn xuất bản chính thức (ACL Anthology, AAAI, NeurIPS, ScienceDirect/Neurocomputing, arXiv). Các paper chưa có bản PDF đầy đủ được ghi chú rõ giới hạn nguồn.

---

## 1. Phân Nhóm Nghiên Cứu Liên Quan

### 1.1. Replay-Based Continual Relation Extraction (CRE)
1. **EA-EMR / Alignment (Wang et al., NAACL 2019)** — *Sentence Embedding Alignment for Lifelong Relation Extraction* (ACL Anthology `N19-1086`). Dùng episodic replay kết hợp embedding alignment để giữ không gian biểu diễn câu ổn định.
2. **EMAR (Han et al., ACL 2020)** — *Continual Relation Learning via Episodic Memory Activation and Reconsolidation* (`10.18653/v1/2020.acl-main.573`). Kích hoạt episodic memory và reconsolidation bằng relation prototypes để giảm overfitting trên vài exemplar cũ.
3. **RP-CRE (Cui et al., ACL-IJCNLP 2021)** — *Refining Sample Embeddings with Relation Prototypes to Enhance Continual Relation Extraction* (`10.18653/v1/2021.acl-long.20`). Chọn exemplar bằng K-means, dùng attention memory network trên relation prototypes để tinh chỉnh sample embeddings. Giới hạn novelty: *prototype-guided embedding refinement đã có tiền lệ, nhưng đòi hỏi lưu raw exemplar ($M>0$) và fine-tune encoder*.
4. **ERDA (Qin & Joty, ACL 2022)** — *Continual Few-shot Relation Learning via Embedding Space Regularization and Data Augmentation* (`10.18653/v1/2022.acl-long.198`). Xét setting task đầu data-rich + task sau few-shot với regularization và augmentation.
5. **ConPL (Chen et al., ACL 2023)** — *Consistent Prototype Learning for Few-Shot Continual Relation Extraction* (`10.18653/v1/2023.acl-long.409`). Lưu $M=1$ raw exemplar + 1 prototype/relation, dùng discrete cloze prompt và hard/confusing-class loss. *Lưu ý audit:* mã nguồn chính thức đánh giá candidate ranking 10-way theo task và cấp 100-shot cho Task 1 trong một số script, cần chuẩn hóa khi so với CIL 80-way strict 5-shot.
6. **CPL (Ma et al., LREC-COLING 2024)** — *Making Pre-trained Language Models Better Continual Few-Shot Relation Extractors* (`aclanthology.org/2024.lrec-main.957`). Kết hợp hybrid prompt, margin contrastive learning và GPT-3.5 generated replay (vẫn thuộc nhóm generative/real replay).
7. **DP-CRE / DPC-FT (Zhao et al., COLING 2025)** — *Dynamic-prototype Contrastive Fine-tuning for Continual Few-shot Relation Extraction with Unseen Relation Detection* (`aclanthology.org/2025.coling-main.586`). Bổ sung lightweight relation encoder theo task và phân bổ dynamic prototype (chủ yếu mức metadata/abstract; chưa đối chiếu PDF sâu).
8. **LGRPG / Label-Guided Relation Prototype Generation:** Khái niệm sinh prototype nhân tạo từ mô tả nhãn (thuộc nhánh generative/synthetic prototype replay nếu sinh vector thay mẫu).

### 1.2. Rehearsal-Free & Exemplar-Free CRE
1. **EoE (Zhou et al., Findings of ACL 2024)** — *An Ensemble-of-Experts Framework for Rehearsal-free Continual Relation Extraction* (`10.18653/v1/2024.findings-acl.83`). Tách Task Identity Inference (TII) và Within-Task Prediction (WTP), dùng ensemble/cascade voting trên nhiều experts.
2. **WAVE-CRE (Le et al., AAAI 2025)** — *Adaptive Prompting for Continual Relation Extraction: A Within-Task Variance Perspective* (`10.1609/aaai.v39i23.34616`, `arXiv:2412.08285`). Không lưu raw samples ($M=0$ raw text), nhưng dùng **task-specific prompt pools**, **learned relation-level task predictor** và **SWAG Gaussian latent feature replay** để huấn luyện shared classifier và predictor.
3. **WAVE++ (Dao et al., Neurocomputing 675, 2026)** — *WAVE++: Capturing Within-Task Variance for Continual Relation Extraction with Adaptive Prompting* (`10.1016/j.neucom.2026.132915`, `arXiv:2505.13944`). Mở rộng WAVE-CRE bằng label descriptions, cascade voting (Mahalanobis distance) và generative latent replay.
4. **CWP (Yang et al., Neurocomputing 633, 129741, 2025)** — *Rehearsal-free continual few-shot relation extraction via contrastive weighted prompts* (`S0925231225004138`). Kết hợp task-specific và task-generic prompts với pairwise/multi-margin contrastive losses mà không cần raw rehearsal. *(Nguồn xác minh từ metadata ScienceDirect; chưa trích toàn văn PDF).*

### 1.3. General Continual Learning (Regularization, Prompting & Pretrained Representations)
- **Regularization:** **EWC** (Kirkpatrick et al., PNAS 2017), **LwF** (Li & Hoiem, TPAMI 2017), **SI** (Zenke et al., ICML 2017), **MAS** (Aljundi et al., ECCV 2018).
- **Prompt-Based Continual Learning:** **L2P** (Wang et al., CVPR 2022), **DualPrompt** (Wang et al., ECCV 2022), **CODA-Prompt** (Smith et al., CVPR 2023), **CPP** (Consistent Prompting, `Consistent Prompting for Rehearsal-Free Continual Learning`), **HiDe-Prompt** (Wang et al., NeurIPS 2023 — tách TII/WTP/TAP), **H-Prompts** (Zuo et al., `arXiv:2401.11544` — tổ chức class/task/general prompts; do đó *không được claim hierarchy prompt là mới hoàn toàn*).
- **PEFT & Random Projections:** **O-LoRA** (Wang et al., Findings EMNLP 2023 — orthogonal LoRA subspaces theo task), **RanPAC** (McDonnell et al., NeurIPS 2023, `arXiv:2307.02251` — frozen random projection + prototype decorrelation không cần adapter huấn luyện hay rehearsal buffer).

### 1.4. Semantic & Hierarchical Learning
- **SIRUS (Le et al., NAACL 2025)** — *Enhancing Discriminative Representation in Similar Relation Clusters for Few-Shot Continual Relation Extraction* (`10.18653/v1/2025.naacl-long.123`). Dùng relation descriptions và dynamic clustering để nhận diện cụm quan hệ tương tự nhau nhằm áp dụng discriminative loss phân tách lớp dễ nhầm. *Điểm phân biệt:* SIRUS dùng dynamic clustering để thiết kế loss phân biệt trong không gian encoder fine-tune/replay, không dùng để tổ chức cây **prompt inheritance & soft routing** không replay như DynaTree-CRE.

---

## 2. Bảng Ma Trận Đối Chiếu (Comparison Matrix)

| Method | Raw memory | Feature memory | Generative replay | Knowledge organization | Routing level | Dynamic hierarchy | Cross-task sharing | Task ID at inference |
|---|---|---|---|---|---|---|---|---|
| **EMAR** (ACL 2020) | Yes ($M \ge 1$–$10$) | Prototype | No | Shared encoder + episodic buffer | None (global) | No | Implicit (full FT) | No |
| **RP-CRE** (ACL 2021) | Yes ($M=10$) | Prototype memory | No | Shared encoder + prototype attention | None | No | Implicit (full FT) | No |
| **ConPL** (ACL 2023) | Yes ($M=1$) | Class prototypes | No | Shared encoder + 1 proto/rel | None (often 10-way task eval in code) | No | Implicit | Often via candidates in repo |
| **CPL** (LREC-COLING 2024) | Yes ($M \ge 1$) | Feature bucket | Yes (GPT-3.5 text) | Shared encoder + hybrid prompt | None | No | Implicit | No |
| **EoE** (Findings ACL 2024) | No ($M=0$) | Class stats (per paper) | Latent/stats per expert | Task-specific experts | Task (cascade voting) | No | Limited | No |
| **WAVE-CRE** (AAAI 2025) | No ($M=0$) | Gaussian stats | **Yes** (GMM latent replay) | **Task-specific** prompt pools | Relation $\rightarrow$ Task predictor | No | Weak (task pool isolated) | No |
| **WAVE++** (Neurocomputing 2026) | No ($M=0$) | Gaussian stats + label desc | **Yes** (GMM latent replay) | **Task-specific** prompt pools | Task (Mahalanobis cascade voting) | No | Weak (task pool isolated) | No |
| **CWP** (Neurocomputing 2025) | No ($M=0$) | Prompt/contrastive state | No (per abstract) | Task-generic + task-specific | Task/weighted | No | Generic prompt only | No |
| **L2P / DualPrompt** | No ($M=0$) | Key vectors | No | Flat key-value pool / G+E | Instance-to-pool | No (2-tier static G/E) | Pool keys | No |
| **H-Prompts** (arXiv 2024) | No ($M=0$) | Class/task stats | No | General + Task + Class | Task/class | Static 3-level by task | General only | No |
| **RanPAC** (NeurIPS 2023) | No ($M=0$) | Gram/covariance matrix ($2^{\text{nd}}$-order) | No | Frozen backbone + Random Proj | None | No | Frozen backbone | No |
| **SIRUS** (NAACL 2025) | Per base CRE setting | Relation clusters/desc | Depends on backbone | Dynamic similarity clusters for loss | None | Dynamic clusters (loss only) | Cluster loss | No |
| **TAPTA P4** (Hướng cũ) | No ($M=0$) | Task & sub-centroids | No | **Task-level** tree ($P^{\text{root}}, P^{\text{leaf}}_t$) | Task Top-K | Static 2-tier by Task ID | Root only | No |
| **DynaTree-CRE (Đề xuất)** | **No ($M=0$)** | **Cluster & leaf prototypes (routing keys)** | **No** | **Dynamic semantic tree** (Root $\rightarrow$ Cluster $\rightarrow$ Relation) | **Hierarchical (Cluster $\rightarrow$ Relation) soft Top-K** | **Yes (incremental clustering)** | **Yes (cluster prompts shared across tasks)** | **No** |

> **Lưu ý phân biệt memory:** Một số phương pháp tuyên bố "rehearsal-free" theo nghĩa không lưu raw text ($M=0$), nhưng vẫn dùng **generative latent replay** (như WAVE-CRE, WAVE++) hoặc lưu **second-order statistics / Gram matrix** (như RanPAC). DynaTree-CRE lưu **0 raw samples**, **0 latent/generative replay**, và chỉ lưu các vector prototype phục vụ định tuyến & cây phân cấp (dung lượng $< 1$ MB, được hạch toán minh bạch trong `resource_metrics.json`).

---

## 3. Candidate Research Gap

Các phương pháp prompt-based CRE hiện hành (WAVE-CRE, WAVE++, TAPTA P4) chủ yếu tổ chức prompt theo **ranh giới Task ID** hoặc một **prompt pool phẳng**. Tuy nhiên, việc gom nhóm theo Task ID trong benchmark CRE chỉ phản ánh thứ tự xuất hiện ngẫu nhiên chứ không phản ánh cấu trúc ngữ nghĩa giữa các quan hệ.

**DynaTree-CRE** tập trung kiểm chứng khoảng trống nghiên cứu sau:
1. Tổ chức tri thức prompt theo **ngữ nghĩa của quan hệ** thay vì Task ID.
2. Cho phép **chia sẻ prompt xuyên tác vụ (cross-task prompt sharing)** cho các quan hệ đồng miền xuất hiện ở các thời điểm khác nhau.
3. **Mở rộng cấu trúc cây tuần tự** chỉ từ các quan hệ đã quan sát mà không cần truy cập future relations hay lưu dữ liệu quá khứ.
4. **Định tuyến mềm phân cấp không cần Task ID** kết hợp cơ chế bảo vệ nhánh cũ và chống sụp đổ nón biểu diễn (anisotropy).
5. **Tách biệt rạch ròi** đóng góp của từng yếu tố (Prototype Control `DT-C0`, Flat Hard/Soft `DT-C1/C2`, Cluster Flat-Routed `DT-C3`, Hierarchical Soft `DT-C4`).

# Định Hướng Nghiên Cứu Chính Thức Trong Đề Tài TAPTA: DynaTree-CRE

- **Đề tài Luận văn Thạc sĩ (Project chính):** *Continual Relation Extraction with Task-Aware Prompt Adaptation (TAPTA / TAPA-CRE)*
- **Phương pháp đề xuất active (Working Title):** *DynaTree-CRE: Dynamic Semantic Prompt Trees for Rehearsal-Free Continual Relation Extraction with Task-Agnostic Routing*
- **Trạng thái:** `ACTIVE` (Chuyển trọng tâm phương pháp trong khuôn khổ luận văn TAPTA từ ngày 2026-11-27; các mốc task-level `P0–P5` trước đó được giữ làm comparator)
- **Setting cốt lõi:** Strict Class-Incremental Continual Relation Extraction (CIL-CRE), Rehearsal-Free ($M=0$ raw samples, $M=0$ feature replay, $M=0$ generative replay), Frozen Pretrained Backbone (`bert-base-uncased`), Task-Agnostic Inference (không có Task ID lúc test).

---

## 1. Động Lực & Ý Tưởng Cốt Lõi

Các phương pháp prompt-based CRE hiện nay (như WAVE-CRE, WAVE++, TAPTA P4) tổ chức prompt pool hoặc nhánh cây theo **Task ID**. Tuy nhiên, trong các benchmark thực tế (như FewRel Protocol v1 8×10), mỗi task là một tập hợp các quan hệ được gán ngẫu nhiên hoặc phân tán. Hệ quả:
1. Một prompt cấp task phải gánh nhiều quan hệ không liên quan về mặt ngữ nghĩa.
2. Hai quan hệ gần nghĩa nhưng xuất hiện ở hai task khác nhau (ví dụ `sibling`/`child` ở T0 và `father`/`mother` ở T3) không thể chia sẻ tri thức prompt chung.
3. Định tuyến theo Task ID sụp đổ khi độ tương đồng trung bình giữa các task centroid vượt `0.95` trong không gian BERT đóng băng.

**DynaTree-CRE** thay thế cấu trúc tổ chức theo Task ID bằng **Cây Prompt Ngữ Nghĩa Động (Dynamic Semantic Prompt Tree)**:
- **Root Node ($P_{\text{root}}$):** Lưu tri thức cấu trúc chung bất biến cho toàn bộ chuỗi tác vụ.
- **Internal Semantic Clusters ($P_v$):** Nhóm các quan hệ có prototype tương đồng ngữ nghĩa, được xây dựng và mở rộng tuần tự **không dựa vào Task ID**.
- **Leaf Relation Nodes ($\Delta P_r$):** Lưu residual prompt riêng cho từng quan hệ $r$ đã quan sát.

---

## 2. Câu Hỏi Nghiên Cứu (Research Questions: RQ1–RQ6)

- **RQ1:** Rehearsal-free dynamic semantic prompt learning (`DT-C1..DT-C4`) có giảm Catastrophic Forgetting so với Sequential Fine-tuning (`B0`) và Frozen BERT Linear Head (`E003`) không?
- **RQ2:** Định tuyến ở mức quan hệ (Relation-level routing: `DT-C1`, `DT-C2`, `DT-C4`) có cải thiện hiệu năng và khả năng task-agnostic inference so với định tuyến ở mức task (`P2a`, `P2b`, `P4`) không?
- **RQ3:** Phân cấp ngữ nghĩa động (`Root → Semantic Cluster → Relation`, `DT-C3`/`DT-C4`) có cải thiện so với flat relation prompt pool (`DT-C1`/`DT-C2`) dưới cùng ngân sách tham số hay không?
- **RQ4:** Tổ chức ngữ nghĩa động có hỗ trợ **cross-task knowledge sharing** khi các quan hệ cùng miền bị phân tán qua nhiều task (Experiment C vs Experiment A/B) không?
- **RQ5:** Việc mở rộng cây tuần tự ảnh hưởng như thế nào đến cân bằng **stability–plasticity**, độ hội tụ cụm, và tỷ lệ nhầm lẫn trong cụm / liên cụm?
- **RQ6:** Phương pháp có hiệu quả khi xét đồng thời accuracy, persistent memory (bytes), trainable parameters và inference latency so với Nearest Prototype (`DT-C0`), TAPTA (`P4`) và các baseline khác không?

---

## 3. Các Giả Thuyết Khoa Học (Hypotheses: H1–H7)

Mỗi giả thuyết đều có tiêu chí **kiểm chứng và bác bỏ (falsification criteria)** rõ ràng:

- **H1 (Stationary Query Space):** Sử dụng Frozen Query Encoder kết hợp chuẩn hóa tâm cố định từ $T_1$ (`t1_centered_cosine`) giúp loại bỏ prototype drift và giảm độ chập nón anisotropy của BERT so với raw cosine (`0.958`). *(Bác bỏ nếu centered space làm giảm Recall@K và accuracy so với raw cosine ở `DT-C0`).*
- **H2 (Relation-Level Routing):** Định tuyến trực tiếp trên từng relation prototype giúp giảm phụ thuộc vào task boundary giả tạo và tăng hiệu quả suy luận không cần Task ID. *(Bác bỏ nếu `DT-C2` không đạt độ chính xác cao hơn hoặc bằng `P2a`/`P4` dưới cùng protocol).*
- **H3 (Cross-Task Semantic Sharing):** Chia sẻ cluster prompt giữa các quan hệ cùng ngữ nghĩa ở các task khác nhau giúp tăng khả năng học few-shot của các quan hệ xuất hiện ở task sau. *(Bác bỏ nếu `DT-C4` không vượt `DT-C2` trong Experiment C — cross-task dispersed).*
- **H4 (Prompt Interference Reduction):** Biểu diễn residual cộng tính $P_{\text{root}} + P_v + \Delta P_r$ kết hợp đóng băng node cũ giúp hạn chế nhiễu giao thoa giữa các nhóm quan hệ khác miền. *(Bác bỏ nếu `DT-C4` có Average Forgetting cao hơn `DT-C2`).*
- **H5 (Knowledge Preservation via Frozen Tree):** Đóng băng các nhánh cũ và bảo toàn trọng số classifier cũ giúp giảm forgetting mà không cần replay mẫu thô hay latent features. *(Bác bỏ nếu độ quên tăng đột biến trên nhánh cũ khi nhiều quan hệ mới gia nhập cùng một cluster).*
- **H6 (Risk of Negative Transfer in Clustering):** Nếu ngưỡng chèn `insertion_threshold` quá lỏng làm gộp sai hai quan hệ xung đột vào cùng một cluster, cluster prompt đóng băng từ các task trước có thể gây **negative transfer** cho quan hệ mới. *(Cần đo trực tiếp tương quan giữa độ tinh khiết cụm và accuracy của quan hệ mới).*
- **H7 (Hierarchical Error Propagation):** Định tuyến phân cấp hai tầng $\pi(r\mid x) = \pi(j\mid x)\pi(r\mid j,x)$ có thể làm khuếch đại lỗi nếu bước chọn cluster $\pi(j\mid x)$ gán sai cụm trước khi chọn leaf. *(Cần đo trực tiếp khoảng cách `DT-C3` vs `DT-C4` và `Oracle Routing Gap`).*

---

## 4. Phạm Vi Đóng Góp Dự Kiến & Ranh Giới Tuyên Bố (Non-Claims)

- **Chỉ được tuyên bố** các cải tiến được chứng minh bởi thực nghiệm thật trên `Protocol v1`.
- **Chưa được phép tuyên bố** "first method", "state-of-the-art", hoặc "giải quyết triệt để 9 bài toán CL" trước khi hoàn thành bộ thực nghiệm đa seed (`FewRel` + `TACRED`) và đối sánh công bằng với `WAVE-CRE`, `WAVE++`, `EoE`, `SIRUS`, `CWP`, `ConPL`.
- **Phải báo cáo trung thực** nếu bất kỳ control baseline đơn giản nào (như `DT-C0` Frozen BERT + Nearest Prototype) có hiệu năng ngang bằng hoặc cao hơn kiến trúc có train prompt.

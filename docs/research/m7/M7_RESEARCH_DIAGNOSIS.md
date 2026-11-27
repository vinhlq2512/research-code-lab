# M7 Research Diagnosis — Tổng hợp Chẩn đoán & Điều chỉnh Kết luận

**Dự án luận văn:** TAPTA (`Continual Relation Extraction with Task-Aware Prompt Adaptation`)  
**Phương pháp đang phát triển:** `DynaTree-CRE` (`M0–M6` MVP đã hoàn thành về mặt kỹ thuật, **chưa** xác nhận giả thuyết nghiên cứu).

---

## 1. Các kết luận cũ cần điều chỉnh sau Gate 5 Audit

1. **Không tuyên bố `t1_centered_cosine` vượt trội toàn diện so với `raw_cosine`:**
   - `t1_centered_cosine` giúp phân cụm cân đối (`4` cụm, `0%` singleton vs `17` cụm, `64.7%` singleton ở `raw_cosine`).
   - Tuy nhiên, trên tác vụ retrieval/nearest-prototype, `raw_cosine` (`60.08%`) vượt `t1_centered_cosine` (`58.61%`, `+1.47%`). Khi $T_1$ đồng nhất miền (Exp B), `t1_centered_cosine` còn tạo ra tâm lệch miền gây mất cân đối cây (`3–43` lá).
2. **Không tuyên bố `DT-C4` (Hierarchical Soft Routing) tốt hơn Flat Routing (`DT-C1/C2`) hay `P4` ở trạng thái hiện tại:**
   - Trên FewRel `seed=2021`, `DT-C4` (`57.23%` NCM) thấp hơn `DT-C0 Raw` (`60.08%`), `DT-C1` (`58.66%`), `DT-C2` (`58.03%`) và `P4` (`57.57%`). Trên TACRED, `DT-C4` (`36.95%`) thấp hơn `P4` (`42.45%`).
3. **Không dùng Oracle Accuracy (`84%–89%`) làm kỳ vọng trực tiếp cho hệ trộn prompt:**
   - Oracle trong MVP dùng `gold_relation_ids` với $K=1$ trên `ExpandableCosineClassifier`, phản ánh headroom lý thuyết nếu chọn đúng đường dẫn prompt, không phải mức chắc chắn đạt được dưới routing gần đúng.
4. **Không tuyên bố bit-exact resume trên Colab khi mới kiểm tra metric JSON:**
   - Đã xác nhận metric equality và tree JSON equality trên Colab; tensor equality và prediction equality cần chạy kiểm tra trực tiếp trên file `.pt` lưu ở Google Drive.

---

## 2. Trạng thái Giả thuyết (Cập nhật M7.0–M7.1)

| Giả thuyết | Nội dung | Trạng thái sau Gate 5 |
|---|---|---|
| **H1 (Semantic Tree & Anisotropy)** | `t1_centered_cosine` chống sụp đổ cụm tốt hơn `raw_cosine` | **Xác nhận một phần** (đúng cho tree clustering, nhưng giảm `-1.47%` ở retrieval) |
| **H2 (Zero-Grad & Trainability)** | Inference-consistent routing không bị kẹt zero-grad | **Xác nhận** ở cấp task (`0` zero-grad steps), cần kiểm tra cấp từng leaf |
| **H3 (Soft Mixing vs Hard Routing)** | Soft Top-3 tốt hơn Hard Top-1 | **Bác bỏ trong MVP** (`DT-C1` `58.66%` > `DT-C2` `58.03%` dưới routing tĩnh) |
| **H4 (Hierarchical Routing)** | `hierarchical_soft` vượt flat routing | **Bác bỏ trong MVP** do cluster-size bias trong `log_softmax` |
| **H5 (Dual Head: NCM vs Cosine)** | `PromptedNCM` giảm new-class bias so với `ExpandableCosine` | **Xác nhận mạnh** (forgetting giảm từ `~30%` xuống `~12–16%`) |
| **H6 (Cross-Task Dispersed)** | `DynaTree-CRE` giữ ổn định dưới `cross_task_dispersed` | **Chưa chứng minh được lợi thế** so với `DT-C2` (`57.66%` vs `58.51%`) |
| **H7 (Tree + Hier vs Tree + Flat)** | `DT-C4` vượt `DT-C3` | **Bác bỏ trong MVP** (`57.23%` vs `57.56%`) |

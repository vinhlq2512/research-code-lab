# Kế hoạch tái lập ConPL: FewRel 10-way 5-shot, một seed

**Ngày lập:** 2026-10-08  
**Mục tiêu W9–W11:** một run BERT thật, seed 100, đủ 8 task trên FewRel; đối chiếu từng task với [ConPL (ACL 2023)](https://aclanthology.org/2023.acl-long.409/). Một seed là mốc kiểm chứng tối thiểu, không tái tạo trung bình 6 sequence của paper.

## 1. Khóa mã và protocol — 08–10/10

- [ ] Dùng checkout chính thức [XiudiChen/ConPL](https://github.com/XiudiChen/ConPL) tại commit `973911899e4b70c705214a2172f54941508ca6b1`; ghi `git diff` trước/sau run. Không dùng report `conpl_lite` làm kết quả gốc.
- [ ] Khóa `fewrel_5shot.py`, `config/config_fewrel_5and10.json`, `CFRLdata_10_100_10_5/*_0.*`, `rel_index.npy`, `rel_cluster_label_0.npy` bằng SHA-256. Input train có 1350 dòng do chứa task đầu data-rich; sampler ConPL phải rút còn **5 mẫu × 10 relation ở cả 8 task**. Assert 50 train samples/task và 80 relation không trùng.
- [ ] Dùng seed `100`, lưu task order thực tế do sampler in ra, số mẫu test/task và candidate labels tại mỗi task. Không thay bằng task order `seed_2021` của Track A hoặc split tự dựng trong repo.
- [ ] Khóa `bert-base-uncased` bằng revision/checksum của weights, config và tokenizer. Dùng cùng checkpoint cho mọi lần retry.

**Gate:** có manifest commit, input hashes, model hashes, task order và xác nhận 8 × 50 train samples trước khi train.

## 2. Dựng môi trường cách ly — 08–12/10

- [ ] Ưu tiên **Linux x86_64 + NVIDIA CUDA** với Python 3.9 trong môi trường riêng của ConPL. Bản `transformers/` đi kèm repo là `4.22.0.dev0`, yêu cầu `huggingface-hub>=0.8.1,<1.0` và `tokenizers>=0.11.1,<0.13`; PyPI có wheel `tokenizers==0.12.1` cho CPython 3.9/Linux x86_64. Pin bản hub/tokenizers cài được cùng PyTorch/CUDA tương thích, `numpy<2`, scikit-learn, `wordninja`, `six`, `tqdm`, rồi lưu lock/freeze. Không thay dependency của `.venv` dùng cho B0/CPL.
- [ ] Chạy import smoke và một forward pass BERT trên batch FewRel thật. Kiểm tra VRAM, dtype, loss hữu hạn và backward/update của một optimizer step.
- [ ] Trên Mac M3 hiện tại, `tokenizers==0.12.1` không có wheel macOS arm64 và build dừng vì thiếu Rust. Chỉ dùng Mac nếu cài được toolchain/dependency cách ly và run có khả năng hoàn thành; ghi rõ khác biệt CPU/MPS so với môi trường paper.

**Gate:** import, tải đúng BERT, forward và backward trên batch thật đều thành công; lưu Python/package/device manifest. Không fallback sang mock model.

## 3. Patch tối thiểu để chạy một seed — 11–13/10

- [ ] Giữ bản gốc sạch; tạo branch/patch riêng. Đổi vòng `for i in range(6)` thành **một lượt seed 100**. Gỡ dòng gán cứng `CUDA_VISIBLE_DEVICES="7"` và chọn GPU bằng biến môi trường của launcher. Không đổi loss, memory selection, hyperparameters hay test candidates.
- [ ] Kiểm tra thứ tự đặt seed: script khởi tạo encoder/model trước `set_seed(...)` trong vòng lặp. Run **mã gốc 1 seed** giữ thứ tự này và ghi RNG state/config để không nhận nhầm seed 100 là khởi tạo model deterministic. Nếu cần sửa seed trước model init, chạy **một run riêng có nhãn `seed-init-fix`**, không ghép vào run mã gốc.
- [ ] Thêm **output-only instrumentation** để lưu `accuracy_whole` sau T1–T8, `accuracy_average`, accuracy theo từng task đã thấy (ma trận tam giác 36 ô), task order, relation IDs, và checkpoint BERT thực sau T8. Việc ghi artifact không được thay đổi gradient/update.
- [ ] Lưu patch (`git diff`), lệnh chạy, stdout/stderr và thời điểm bắt đầu/kết thúc vào thư mục run riêng, ví dụ `results/fewrel/5shot/ConPL_original/seed_100/`.

**Gate:** diff của run mã gốc chỉ chứa lựa chọn thiết bị, một seed và xuất artifact; reviewer xác nhận không đổi thuật toán. Bất kỳ sửa thứ tự seed nào nằm trong diff/run riêng.

## 4. Smoke rồi chạy đủ 8 task — 13–20/10

- [ ] Smoke T1 với dữ liệu thật: 10 relation, 50 train samples, accuracy hữu hạn, prototype/memory hình thành và checkpoint có trọng số BERT. Smoke để riêng, không đưa vào bảng paper.
- [ ] Chạy đầy đủ T1→T8 từ đầu; giữ log thô và checkpoint. Nếu lỗi, sửa nguyên nhân và chạy lại trong thư mục mới; không ghép kết quả của nhiều lần retry.
- [ ] Kiểm tra `accuracy_whole` là accuracy trên hợp test của tất cả relation đã thấy; giữ `accuracy_average` ở cột riêng. Không đánh đồng hai chuỗi in ra ở cuối script. Kiểm tra 36 ô per-task và 8 giá trị cumulative accuracy.

**Gate:** run thoát mã 0, có 8/8 task, model/checkpoint thật, task order, raw logs và đầy đủ metrics. Không dùng số mô phỏng hoặc số paper để điền cột local.

## 5. Đối chiếu và kết luận — 20–25/10

- [ ] So trajectory một seed với hàng FewRel 10-way 5-shot trong Table 1 của paper: `95.72, 93.53, 91.31, 89.95, 88.93, 88.39, 87.43, 85.77` (%). Ghi `local - paper` theo từng task, với nhãn rõ **local 1 seed** và **paper mean 6 sequences**.
- [ ] Nếu lệch, kiểm tra lần lượt split/hash, task order, BERT revision, K-shot ở T1, checkpoint init, candidate-label rule, memory size, evaluation series, rồi dependency/device. Ghi phát hiện và patch cụ thể; không tự quy lỗi cho seed.
- [ ] Lưu `manifest.json`, `config.json`, `environment.txt`, `patch.diff`, `stdout.log`, `stderr.log`, `task_order.json`, `accuracy_by_task.csv`, `performance_matrix.csv`, `summary.json`, checkpoint và `mismatch.md`. Cập nhật tracker bằng trạng thái **đã chạy thật / chưa chạy** và link artifact.
- [ ] Nếu cần so sánh mean±std với paper, chạy tiếp 5 seed còn lại trong giai đoạn sau; không gọi run 1 seed là full paper reproduction.

**Tiêu chí hoàn thành tối thiểu:** 1 seed 100 chạy hết 8 task bằng BERT thật, artifacts có provenance, accuracy và mismatch log. Tracker hiện vẫn yêu cầu B0 thật → B1 → B2 trước khi dùng ConPL để chốt bảng so sánh chung; việc chuẩn bị/chạy ConPL không biến các gate đó thành đã hoàn thành.

## Trạng thái đầu vào

Preflight ngày 2026-10-08 và lỗi dependency đã ghi ở [nhật ký ConPL/CPL](conpl-cpl-fewrel-5shot-2026-10-08.md). Chưa có local accuracy ConPL.

Nguồn xác minh wheel: [PyPI `tokenizers` 0.12.1](https://pypi.org/project/tokenizers/0.12.1/).

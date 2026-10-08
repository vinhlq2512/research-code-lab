# Nhật ký tái lập ConPL/CPL trên FewRel 5-shot — 2026-10-08

## Mục tiêu và trạng thái

W9–W11 (2026-10-05 đến 2026-10-25): chạy **mỗi phương pháp 1 seed thật** với FewRel, lưu trajectory sau từng task và đối chiếu với paper theo đúng protocol. Tính đến ngày ghi log, **chưa có run huấn luyện BERT thành công**, nên không có accuracy tái lập và **chưa thể tính mismatch số học với paper**. Các `fewrel_*_lite_*` report và B0 mock trước đó không được tính là kết quả ConPL/CPL gốc.

## Mã, dữ liệu, môi trường

| Thành phần | Provenance |
|---|---|
| ConPL | `https://github.com/XiudiChen/ConPL`, commit `973911899e4b70c705214a2172f54941508ca6b1`, checkout `paper-reproductions/ConPL` |
| CPL | `https://github.com/mashengkun/CPL`, commit `6ce0b2a07d37a243804094f73a1743ab620aea04`, checkout `paper-reproductions/CPL` |
| Máy preflight | Apple M3 Pro, macOS, **không có CUDA/NVIDIA** |
| Python | 3.11.9; `.venv` có PyTorch 2.14.1, Transformers 5.18.0, NumPy 2.4.6; đã bổ sung `scikit-learn 1.9.1`, `six 1.17.0`, `openai 0.28.0`, `nltk 3.8.1`, `retry 0.9.2` cho preflight |
| CPL paper/code yêu cầu | Python 3.8, PyTorch 1.7.0, Transformers 4.10.0, NumPy 1.24.4, scikit-learn 0.24.2, OpenAI API và BERT weights trong `./bert-base-uncased` |
| BERT local | Có cache `bert-base-uncased` của Hugging Face, nhưng thư mục trọng số mà CPL code mặc định đọc trong checkout đang rỗng; chưa nối hoặc thay đổi đường dẫn gốc. |

Hai checkout có cùng `CFRLdata_10_100_10_5/train_0.txt` (SHA-256 `af862bd183852ee5dd8be0fe1b85db0b1c520697f7b0f68a6f2e0ef25eca4dfe`, 1350 dòng) và cùng `rel_cluster_label_0.npy` (SHA-256 `5107e26201ce9a46370ac25d10c8dc6c640cc9221c9f7a12ece49751415d40d1`). **Cùng raw split không có nghĩa cùng training protocol**: ConPL sampler lấy 5 mẫu/relation cả T1; CPL giữ 100 mẫu/relation ở T1 và 5 mẫu/relation ở T2–T8.

## Lần chạy thử bằng mã gốc

### ConPL

Từ `paper-reproductions/ConPL`:

```bash
../../.venv/bin/python fewrel_5shot.py
```

- Lần đầu dừng khi import `six`; đã cài `six` trong `.venv` và chạy lại.
- Lần hai dừng trước lúc nạp BERT: `ImportError: cannot import name 'HfFolder' from 'huggingface_hub'`. Checkout ConPL đóng gói Transformers `4.22.0.dev0`, yêu cầu `huggingface-hub>=0.8.1,<1.0` và `tokenizers>=0.11.1,<0.13`; môi trường hiện tại có hub/tokenizers của Transformers 5.x. Chưa sửa mã gốc hoặc âm thầm bỏ qua version guard.
- Thử môi trường cách ly Python 3.10.19 tại `paper-reproductions/ConPL/.venv`: `tokenizers==0.12.1` không có wheel dùng được cho macOS arm64; build từ source dừng ở `error: can't find Rust compiler`. Môi trường cách ly chưa có dependencies hoàn chỉnh, nên cũng chưa vào train. Đây là blocker của host/toolchain hiện tại, không phải kết quả mô hình.
- Script mặc định `random_seed=100`, chạy **6 sequence** (`for i in range(6)`, seed `100+10*i`). Để chạy 1 seed phải ghi nhận patch vòng lặp `range(6) -> range(1)`; không được gọi output đó là paper mean.

### CPL

Từ `paper-reproductions/CPL`:

```bash
../../.venv/bin/python train.py --task_name FewRel --num_k 5 --num_gen 2
```

- Nạp được ba tệp FewRel `.pkl`, in `Task_order: [7 3 0 5 4 1 6 2]`, seed `100`, rồi dừng trước train: `OSError: Error no file named model.safetensors, or pytorch_model.bin, found in directory ./bert-base-uncased.`
- `config.ini` mặc định `device = cuda:1`, `total_round = 6`, `gen = 1`, `num_gen = 2`; checkout không chứa BERT weights và máy này không có CUDA. Sau khi nối đúng weights vẫn cần môi trường tính toán tương thích.
- CPL gốc gọi `openai.ChatCompletion.create` để tạo replay bằng GPT-3.5. Không có API key/corpus generated được cung cấp cho run này; **chưa gọi dịch vụ sinh dữ liệu**. Tắt `gen` sẽ là **CPL without generation**, không phải reproduction gốc.

## Bảng baseline v1 — ranh giới so sánh

| Baseline | FewRel protocol gốc | Paper T8 accuracy | Local 1-seed T8 | Trạng thái |
|---|---|---:|---:|---|
| ConPL | 8 × 10 relations, **5-shot cả T1**, 1 exemplar + 1 prototype/relation | 85.77% (paper Table 1, trung bình các sequence) | — | Import blocker; chưa train |
| CPL | 8 × 10 relations, **T1 100-shot**, T2–T8 5-shot, 1 exemplar + GPT-generated replay | 64.50% (paper Table 1, trung bình các run) | — | Weights/device/API blocker; chưa train |

**Mismatch log:** chưa có local model result để lấy `local - paper`. Mọi chênh lệch xuất hiện sau này phải kèm seed, task order, số shot từng task, candidate-label rule, model/checkpoint, memory/generation, metric và raw trajectory. Hai con số paper ở trên thuộc **hai protocol khác nhau**; không lấy hiệu `85.77 - 64.50` làm so sánh phương pháp.

## Điều kiện để đóng W9–W11

1. Hoàn tất gate hiện hành B0 thật → B1 → B2, hoặc ghi rõ quyết định thay đổi gate trong tracker trước khi dùng kết quả ConPL/CPL để chọn baseline.
2. Tạo môi trường cách ly cho từng repo và lưu `python --version`, `pip freeze`, device, commit SHA, SHA-256 split. Không dùng mock predictor hoặc weights giả. ConPL cần hub/tokenizers tương thích với bản Transformers đi kèm; CPL cần BERT weights và thiết bị hợp lệ.
3. Chạy **seed 100 duy nhất** bằng mã gốc với thay đổi có log (`range(6) -> range(1)` ở ConPL; `total_round=1` ở CPL). Với CPL, cung cấp corpus generation gốc có provenance hoặc API key/cost được chủ sở hữu chấp thuận; không tự đổi `gen=0` rồi gắn nhãn CPL gốc.
4. Giữ stdout/stderr, config diff, checkpoint thực, task order, per-task cumulative accuracy, summary và công thức đối chiếu. Đánh dấu run thất bại là thất bại, không điền số paper vào cột local.

## Nguồn paper

- [ConPL, ACL 2023](https://aclanthology.org/2023.acl-long.409/), Table 1 và thiết lập NK-CRE.
- [CPL, LREC-COLING 2024](https://aclanthology.org/2024.lrec-main.957/), §5.1 và Table 1.

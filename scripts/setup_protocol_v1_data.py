#!/usr/bin/env python3
"""Setup and verify Protocol v1 datasets (FewRel 8x10 5-shot & TACRED 8x5 5-shot).

Copies verified ConPL/CFRL split files into data/protocol_v1/ and generates
provenance_hashes.json to guarantee scientific reproducibility and data integrity.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_CONPL_DATA = REPO_ROOT / "paper-reproductions" / "ConPL" / "data"
TARGET_DATA_DIR = REPO_ROOT / "data" / "protocol_v1"

EXPECTED_HASHES = {
    "fewrel/relation_name.txt": "e2bf1364fdfe927045457c4627eaec64bf7cf8a5d7bdd145491dd3b101a99fd0",
    "fewrel/rel_index.npy": "bf877a1b803646492899cf49abb456f7490ce38c92bdda46a341d0a6fcc563b8",
    "fewrel/rel_feature.npy": "edecff6648585f39217bc05657d17085df2dc4f7c2880a5ef1c5c6c8e33dcb1c",
    "fewrel/CFRLdata_10_100_10_5/rel_cluster_label_0.npy": "5107e26201ce9a46370ac25d10c8dc6c640cc9221c9f7a12ece49751415d40d1",
    "fewrel/CFRLdata_10_100_10_5/train_0.txt": "af862bd183852ee5dd8be0fe1b85db0b1c520697f7b0f68a6f2e0ef25eca4dfe",
    "fewrel/CFRLdata_10_100_10_5/valid_0.txt": "6d6f14ce646b986ba3a600c10826ac7e61a42be8e1885d250fdbf83e35db8727",
    "fewrel/CFRLdata_10_100_10_5/test_0.txt": "6d6f14ce646b986ba3a600c10826ac7e61a42be8e1885d250fdbf83e35db8727",
}


def sha256_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def copy_dataset(dataset_name: str, files_and_dirs: list[str]) -> dict[str, str]:
    src_base = SOURCE_CONPL_DATA / dataset_name
    tgt_base = TARGET_DATA_DIR / dataset_name
    tgt_base.mkdir(parents=True, exist_ok=True)

    hashes = {}
    for item in files_and_dirs:
        src_path = src_base / item
        tgt_path = tgt_base / item

        if not src_path.exists():
            print(f"⚠️ Cảnh báo: Không tìm thấy nguồn: {src_path}")
            continue

        if src_path.is_dir():
            shutil.copytree(src_path, tgt_path, dirs_exist_ok=True)
            for sub_file in tgt_path.glob("**/*"):
                if sub_file.is_file():
                    rel_k = f"{dataset_name}/{sub_file.relative_to(tgt_base)}"
                    hashes[rel_k] = sha256_file(sub_file)
        else:
            tgt_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, tgt_path)
            rel_k = f"{dataset_name}/{item}"
            hashes[rel_k] = sha256_file(tgt_path)

    return hashes


def main():
    print(f"=== THIẾT LẬP DỮ LIỆU PROTOCOL V1 ===")
    print(f"Nguồn: {SOURCE_CONPL_DATA}")
    print(f"Đích:  {TARGET_DATA_DIR}")

    if not SOURCE_CONPL_DATA.exists():
        sys.exit(f"Lỗi: Không tìm thấy thư mục dữ liệu nguồn tại {SOURCE_CONPL_DATA}")

    TARGET_DATA_DIR.mkdir(parents=True, exist_ok=True)

    fewrel_items = [
        "relation_name.txt",
        "rel_index.npy",
        "rel_feature.npy",
        "CFRLdata_10_100_10_5",
    ]
    tacred_items = [
        "relation_name.txt",
        "rel_index.npy",
        "rel_feature.npy",
        "CFRLdata_10_100_10_5",
    ]

    all_hashes = {}
    print("\n[1/3] Sao chép và hash FewRel 8x10 5-shot...")
    all_hashes.update(copy_dataset("fewrel", fewrel_items))

    print("\n[2/3] Sao chép và hash TACRED 8x5 5-shot...")
    all_hashes.update(copy_dataset("tacred", tacred_items))

    # Verify FewRel against known hashes
    mismatches = []
    for rel_path, exp_h in EXPECTED_HASHES.items():
        act_h = all_hashes.get(rel_path)
        if act_h != exp_h:
            mismatches.append((rel_path, exp_h, act_h))

    if mismatches:
        print("\n❌ LỖI TOÀN VẸN DỮ LIỆU:")
        for path, exp, act in mismatches:
            print(f"  - {path}: mong đợi {exp[:12]}..., thực tế {act[:12] if act else 'None'}")
        sys.exit(1)
    else:
        print("✓ Kiểm tra SHA-256 FewRel: 100% Khớp với đặc tả gốc ACL 2023 ConPL!")

    provenance_path = TARGET_DATA_DIR / "provenance_hashes.json"
    with open(provenance_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "protocol": "v1",
                "benchmarks": ["FewRel 8x10 5-shot", "TACRED 8x5 5-shot"],
                "source": "paper-reproductions/ConPL/data",
                "sha256": all_hashes,
            },
            f,
            indent=2,
        )

    print(f"\n[3/3] Đã lưu mã băm chứng thực tại: {provenance_path}")
    print("=== HOÀN TẤT THIẾT LẬP DỮ LIỆU PROTOCOL V1 ===")


if __name__ == "__main__":
    main()

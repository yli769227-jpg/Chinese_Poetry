"""
prepare.py
读取 corpus.txt，构建字符级 tokenizer，
编码为 uint16 数组（vocab < 65536，uint16 足够），
切分 train/val（最后 10% 做验证），
输出 train.bin / val.bin / meta.pkl 供 nanoGPT 直接加载。

用法（在项目根目录下，先跑过 extract_corpus.py）：
    python scripts/prepare.py
"""
import pickle
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INPUT_FILE = DATA_DIR / "corpus.txt"
TRAIN_BIN = DATA_DIR / "train.bin"
VAL_BIN = DATA_DIR / "val.bin"
META_FILE = DATA_DIR / "meta.pkl"


def main():
    print(f"读取 {INPUT_FILE} ...")
    text = INPUT_FILE.read_text(encoding="utf-8")
    print(f"字符总数: {len(text):,}")

    # 1. 构建字符级词表（sorted 保证可重现）
    chars = sorted(set(text))
    vocab_size = len(chars)
    print(f"词表大小: {vocab_size:,}")

    # 2. 创建 stoi (str->int) / itos (int->str) 映射
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    # 3. 编码整个 corpus
    def encode(s):
        return [stoi[c] for c in s]

    print("编码中（约 10 秒）...")
    ids = np.array(encode(text), dtype=np.uint16)  # vocab < 65536，uint16 足够
    print(f"token 总数: {len(ids):,}")

    # 4. 切分 train/val（90/10）
    n = len(ids)
    split_idx = int(n * 0.9)
    train_ids = ids[:split_idx]
    val_ids = ids[split_idx:]
    print(f"train: {len(train_ids):,}  val: {len(val_ids):,}")

    # 5. 写入 .bin 文件
    train_ids.tofile(TRAIN_BIN)
    val_ids.tofile(VAL_BIN)

    # 6. 保存元信息（vocab_size + 映射表，sample 时需要）
    meta = {
        "vocab_size": vocab_size,
        "itos": itos,
        "stoi": stoi,
    }
    with open(META_FILE, "wb") as f:
        pickle.dump(meta, f)

    print("\n输出文件:")
    print(f"  {TRAIN_BIN} ({TRAIN_BIN.stat().st_size / 1e6:.1f} MB)")
    print(f"  {VAL_BIN} ({VAL_BIN.stat().st_size / 1e6:.1f} MB)")
    print(f"  {META_FILE}")


if __name__ == "__main__":
    main()

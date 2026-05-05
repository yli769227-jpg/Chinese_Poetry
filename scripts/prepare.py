"""
prepare.py
读取 corpus.txt，构建字符级 tokenizer，
编码为 uint16 数组（vocab < 65536，uint16 足够），
切分 train/val（文档级随机切分，避免 distribution shift），
输出 train.bin / val.bin / meta.pkl 供 nanoGPT 直接加载。

切分策略：
    corpus 由若干"文档"组成（用 \\n\\n 分隔）。先按文档切分、
    用固定 seed 打乱后再 90/10 切分，保证 train 和 val 分布一致。

用法（在项目根目录下，先跑过 extract_corpus.py）：
    python scripts/prepare.py
"""
import pickle
import random
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INPUT_FILE = DATA_DIR / "corpus.txt"
TRAIN_BIN = DATA_DIR / "train.bin"
VAL_BIN = DATA_DIR / "val.bin"
META_FILE = DATA_DIR / "meta.pkl"

SEED = 42
TRAIN_RATIO = 0.9


def main():
    print(f"读取 {INPUT_FILE} ...")
    text = INPUT_FILE.read_text(encoding="utf-8")
    print(f"字符总数: {len(text):,}")

    # 1. 构建字符级词表（基于完整 corpus，sorted 保证可重现）
    chars = sorted(set(text))
    vocab_size = len(chars)
    print(f"词表大小: {vocab_size:,}")

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    # 2. 按文档切分（文档间用空行分隔），打乱顺序避免 distribution shift
    documents = [d for d in text.split("\n\n") if d.strip()]
    print(f"文档总数: {len(documents):,}")

    random.seed(SEED)
    random.shuffle(documents)
    print(f"已用 seed={SEED} 打乱文档顺序")

    # 3. 90/10 切分（文档级，保证 train/val 同分布）
    split_idx = int(len(documents) * TRAIN_RATIO)
    train_text = "\n\n".join(documents[:split_idx])
    val_text = "\n\n".join(documents[split_idx:])
    print(f"train 文档数: {split_idx:,}  val 文档数: {len(documents) - split_idx:,}")

    # 4. 分别编码为 uint16 数组
    print("编码中...")
    train_ids = np.array([stoi[c] for c in train_text], dtype=np.uint16)
    val_ids = np.array([stoi[c] for c in val_text], dtype=np.uint16)
    print(f"train tokens: {len(train_ids):,}  val tokens: {len(val_ids):,}")

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

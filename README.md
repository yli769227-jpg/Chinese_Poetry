# Chinese Poetry GPT — train a tiny LLM from scratch on classical Chinese poetry

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![Params](https://img.shields.io/badge/params-15M%20~%2032M-green.svg)](#训练成果速览)

> **A character-level GPT trained from scratch (no pretrained weights) on Tang/Song poetry, Chu Ci, Shijing & more — 15M params, ~5 min on a single GPU.** Built on [nanoGPT](https://github.com/karpathy/nanoGPT), core training code < 600 lines. The goal isn't SOTA — it's understanding the full pretraining pipeline: data → tokenizer → model → training → sampling.
>
> 一个**教学性质**的从头预训练项目：用字符级 GPT，在唐诗、宋词、楚辞、诗经、论语等古典语料上从零训练一个小语言模型，完整走通 LLM 预训练全流程。

**A 32M-param model, trained from scratch in ~17 minutes, wrote this:**

```
《登高》杜甫
风急天高雁北飞，江山无处不相违。
江城日落三更后，野寺钟鸣一夜归。
```

> ↑ The model *composed* this — not a verbatim training sample. More in [`samples/`](samples/).

## 项目特点

- **从零训练**：不使用任何预训练权重，完全从随机初始化开始
- **小巧可复现**：A 档 15M 参数，单张 24GB 显存 GPU 几分钟训完
- **风格鲜明**：选择领域窄但风格独特的语料 —— 古典诗词
- **代码极简**：基于 [nanoGPT](https://github.com/karpathy/nanoGPT)，核心训练代码 < 600 行
- **中文友好**：字符级 tokenizer，每个 token 直接对应一个汉字

## 训练成果速览

| 配置 | 参数量 | 训练步数 | best val_loss | 实际耗时 | 样本质量 |
|------|--------|---------|----|----|----|
| **A 档 v1 (small, 旧切分)** | 15.45M | 5,000 | 4.84 | ~5 分钟 (单卡) | 格律正确，主题单调 |
| **B 档 (medium, 旧切分)** | 31.61M | 15,000 | 4.59 | ~17 分钟 (双卡 DDP) | 主题多样，对仗工整 |
| **A 档 v2 (small, 新切分)** | 15.45M | 5,000 | **3.73** | ~5 分钟 (单卡) | 同 v1 |

> 📌 **关于切分修复**：早期版本 `prepare.py` 把 corpus 末尾 10% 当作验证集，
> 而 corpus 末尾恰好是楚辞 + 诗经（与训练分布差异大），导致 val_loss 数字偏高。
> 现已修复为**文档级随机 shuffle 后 90/10 切分**。A 档 v2 用新切分重训，
> val_loss 从 4.84 降到 3.73，train-val gap 从 1.34 降到 0.09 —— 证实早期"过拟合"
> 是 distribution shift 假象，不是真过拟合。详见 [`docs/training_log.md`](docs/training_log.md)。

模型权重（A/B 档）通过 [GitHub Releases](https://github.com/yli769227-jpg/Chinese_Poetry/releases) 提供下载。

## 数据来源

来自 [chinese-poetry/chinese-poetry](https://github.com/chinese-poetry/chinese-poetry)（CC0 / Public Domain）：

| 子集 | 内容 | 字符 |
|------|------|------|
| 全唐诗 | 唐诗 + 宋诗 ~5.5 万 + 25 万首 | 繁体 |
| 宋词 | 词 ~2 万首 | 简体 |
| 元曲 | 元曲选集 | 简体 |
| 楚辞 | 屈原等 | 简体 |
| 诗经 | 305 篇 | 简体 |
| 论语 | 全本 | 简体 |

经简繁统一 + 去空白处理后：
- **总文档数**：344,355
- **总字符数**：28,468,085 (~28.5M tokens)
- **独立字符数**：12,573（即 vocab_size）

## 硬件要求

- 至少 1 张 NVIDIA GPU，24GB 显存（A30 / RTX 3090 / 4090 / A6000 等）
- ~10GB 磁盘空间（含原始语料 + 训练数据 + checkpoint）
- Python 3.10+

## Quick Start

### 1. 环境准备

```bash
# 克隆本项目
git clone https://github.com/yli769227-jpg/Chinese_Poetry.git
cd Chinese_Poetry

# 创建虚拟环境（推荐用 uv，pip + venv 也行）
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.10
source .venv/bin/activate

# 装依赖
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv pip install -r requirements.txt

# 验证 GPU
python -c "import torch; print('CUDA:', torch.cuda.is_available(), 'GPUs:', torch.cuda.device_count())"
```

### 2. 克隆 nanoGPT 与诗词语料

```bash
# nanoGPT 训练框架（项目依赖）
git clone https://github.com/karpathy/nanoGPT.git

# 古诗词原始语料 (~455 MB)
mkdir -p raw && cd raw
git clone --depth 1 https://github.com/chinese-poetry/chinese-poetry.git
cd ..
```

### 3. 数据预处理

```bash
mkdir -p data
python scripts/extract_corpus.py     # JSON → corpus.txt（约 1 分钟）
python scripts/prepare.py             # corpus.txt → train.bin/val.bin/meta.pkl
```

### 4. 链接数据到 nanoGPT

```bash
mkdir -p nanoGPT/data/poetry
ln -sf $(pwd)/data/train.bin nanoGPT/data/poetry/train.bin
ln -sf $(pwd)/data/val.bin   nanoGPT/data/poetry/val.bin
ln -sf $(pwd)/data/meta.pkl  nanoGPT/data/poetry/meta.pkl
cp configs/poetry_small.py  nanoGPT/config/
cp configs/poetry_medium.py nanoGPT/config/
```

### 5. 训练

```bash
cd nanoGPT

# A 档：单卡 ~5-30 分钟
python train.py config/poetry_small.py

# B 档：双卡 DDP ~2-3 小时
torchrun --standalone --nproc_per_node=2 train.py config/poetry_medium.py
```

### 6. 推理生成

```bash
# 自由生成
python sample.py --out_dir=/path/to/out/poetry_small \
    --start='《' --num_samples=3 --max_new_tokens=300 \
    --temperature=0.8 --top_k=200

# 给一个 prompt 让模型续写
echo -n '《登高》杜甫
风急天高' > /tmp/prompt.txt
python sample.py --out_dir=/path/to/out/poetry_small \
    --start=FILE:/tmp/prompt.txt --num_samples=3 \
    --max_new_tokens=200 --temperature=0.5 --top_k=50
```

### 7. (可选) 直接用预训练 checkpoint

```bash
# A 档（~178 MB，验证流程用）
mkdir -p out/poetry_small
wget -O out/poetry_small/ckpt.pt \
    https://github.com/yli769227-jpg/Chinese_Poetry/releases/download/v0.1-small/ckpt.pt

# B 档（~380 MB，质量更好，推荐）
mkdir -p out/poetry_medium
wget -O out/poetry_medium/ckpt.pt \
    https://github.com/yli769227-jpg/Chinese_Poetry/releases/download/v0.2-medium/ckpt.pt
```

## 模型架构（A 档）

```
Token Embedding (12573 → 384)
    +
Position Embedding (256 → 384)
    ↓
6 × TransformerBlock {
    LayerNorm
    Multi-Head Self-Attention (6 heads, dim 384)
    LayerNorm
    Feed-Forward (4× expansion, GELU)
    Residual connections + dropout
}
    ↓
LayerNorm
    ↓
LM Head (384 → 12573)
    ↓
softmax over vocab → next char probability
```

详细参数见 [`configs/poetry_small.py`](configs/poetry_small.py)。

## 生成样例

### A 档（15M）—— 格律对，主题单调

```
《偈颂一百零四首其九五》释绍昙
达磨莫相识，知津是阿谁。
全提不知处，一点混融丝。
```

### B 档（32M）—— 对仗工整，画面感强

```
《金陵杂兴二百首其一八》苏泂
霜落风悲古渡头，两边黄叶一边秋。
相思吟罢清江畔，回望渔人已白头。

《登高》杜甫
风急天高雁北飞，江山无处不相违。
江城日落三更后，野寺钟鸣一夜归。
```

完整对比见 [`samples/`](samples/) 目录。

### 关键观察

1. **作者风格学习**：模型抓到了"宋太宗 ≈ 道教/金丹"、"陆游 ≈ 漂泊/秋雨/暮年"等指纹
2. **系列编号**：从训练数据学到了"其一、其二...其一○○" 这种连续编号模式
3. **格律掌握**：五言/七言、绝句/律诗、词牌格式都基本正确
4. **幻觉**：编造不存在的诗作和题目（这就是所有 LLM 的本质）
5. **OOV 局限**：字符级 tokenizer 遇到训练集外字符会卡住（已在 sample.py 中加保护）

## 完整训练流程详解

详见 [`docs/training_log.md`](docs/training_log.md) 和 [`docs/architecture.md`](docs/architecture.md)，包含：

- 数据清洗的设计选择（为什么删空格 / 为什么繁简统一）
- Tokenizer 选型（字符级 vs BPE 的取舍）
- 模型超参的物理意义
- DDP 分布式训练原理
- 学习率 warmup + cosine decay
- 采样策略（temperature / top_k）的影响

## Roadmap

- [x] A 档：15M 参数，5000 步训练
- [x] B 档：32M 参数，15000 步 + DDP 双卡
- [x] **修复 train/val 切分**：文档级随机 shuffle 后再 90/10（解决 distribution shift）
- [ ] B 档重训以拿到干净的 val_loss 数据
- [ ] C 档：100M 参数（试试硬件极限）
- [ ] 加显式分隔 token（`<|title|>`、`<|author|>`、`<|content|>`）让边界更清晰
- [ ] BPE / SentencePiece tokenizer 对比实验
- [ ] 显式格律标注（平仄 / 韵脚）的输入增强

## 致谢

- [Andrej Karpathy / nanoGPT](https://github.com/karpathy/nanoGPT) — 极简的 GPT 训练框架
- [chinese-poetry/chinese-poetry](https://github.com/chinese-poetry/chinese-poetry) — 高质量中文古诗词数据
- [zhconv](https://github.com/gumblex/zhconv) — 简繁转换

## License

MIT — 见 [LICENSE](LICENSE)

数据语料的版权归原始项目所有，本项目不重新分发原始数据。

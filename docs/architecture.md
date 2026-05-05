# 架构与原理

## 整体流程

```
原始 JSON 语料 (chinese-poetry repo)
        │
        ▼ extract_corpus.py
   corpus.txt (UTF-8 纯文本)
        │
        ▼ prepare.py
   train.bin / val.bin (uint16 数组)
   meta.pkl (vocab + stoi + itos)
        │
        ▼ nanoGPT/train.py
   ckpt.pt (模型权重 + 优化器状态)
        │
        ▼ nanoGPT/sample.py
   生成的诗词文本
```

## 字符级 Tokenizer

### 为什么选字符级而不是 BPE

| 维度 | 字符级 | BPE / SentencePiece |
|------|--------|--------------------|
| 实现复杂度 | 极简（一行 `set(text)`） | 需训练分词器 |
| 词表大小 | 12,573 | 通常 30k-50k |
| 序列长度 | 长（每个汉字 1 token） | 短（多字组合成 1 token） |
| 教学价值 | ⭐⭐⭐ 每个 token = 1 汉字，可读性满分 | ⭐⭐ 中间层不直观 |
| 古文贴合度 | ⭐⭐⭐ 古文里"字"本身就是有强语义的最小单位 | ⭐ BPE 切词在古文上未必合理 |
| OOV 鲁棒性 | ❌ 训练集外字符会崩 | ✅ 字节级 fallback |

对**古典诗词 + 教学**这两个目标，字符级是完美选择。

### 词表构建

```python
chars = sorted(set(corpus_text))     # 12,573 个独立字符
stoi  = {ch: i for i, ch in enumerate(chars)}  # 字符 → ID
itos  = {i: ch for i, ch in enumerate(chars)}  # ID → 字符
```

### Encode / Decode

```python
encode = lambda s: [stoi[c] for c in s if c in stoi]   # 静默跳过 OOV
decode = lambda l: "".join([itos[i] for i in l])
```

## 模型架构（GPT 风格 decoder-only Transformer）

### A 档 (15M)

```
                    Input tokens [B, T]
                          │
                          ▼
   ┌──────────────────────────────────────────┐
   │ Token Embedding   (12573 × 384)          │
   │ Position Embedding (256 × 384)           │
   │ + Dropout                                │
   └──────────────────────────────────────────┘
                          │  [B, T, 384]
                          ▼
   ┌──────────────────────────────────────────┐
   │ TransformerBlock × 6:                    │
   │   ┌───────────────────────────────────┐  │
   │   │ LayerNorm                          │  │
   │   │ Multi-Head Self-Attention          │  │
   │   │   • 6 heads, head_dim = 64         │  │
   │   │   • causal mask（只看前文）        │  │
   │   │   • Flash Attention（PyTorch 自带）│  │
   │   │ + Residual + Dropout               │  │
   │   ├───────────────────────────────────┤  │
   │   │ LayerNorm                          │  │
   │   │ Feed-Forward                       │  │
   │   │   • Linear: 384 → 1536 (4× expand) │  │
   │   │   • GELU activation                │  │
   │   │   • Linear: 1536 → 384             │  │
   │   │ + Residual + Dropout               │  │
   │   └───────────────────────────────────┘  │
   └──────────────────────────────────────────┘
                          │  [B, T, 384]
                          ▼
   ┌──────────────────────────────────────────┐
   │ Final LayerNorm                           │
   │ LM Head: Linear (384 → 12573)             │
   │   ↑ 通常与 Token Embedding 权重共享       │
   └──────────────────────────────────────────┘
                          │  [B, T, 12573] logits
                          ▼
                  cross-entropy loss
                  (训练) or
                  softmax + sample (推理)
```

### 关键超参的物理意义

| 超参 | 名称 | 物理意义 |
|------|------|----------|
| `n_layer` | 层数 | 模型"思考深度"。每层在前一层基础上做更复杂的特征组合 |
| `n_embd` | 隐藏维度 | 每个 token 的表示向量大小。决定每个位置能携带多少信息 |
| `n_head` | 注意力头数 | 每层把 attention 拆成多个并行的"关注模式" |
| `block_size` | 上下文长度 | 模型一次能看多少字符的前文 |
| `batch_size` | 批大小 | 每次梯度更新基于多少样本 |
| `dropout` | 丢弃率 | 训练时随机置零，防过拟合 |
| `learning_rate` | 学习率 | 每次梯度更新的步长 |

## 训练（next-token prediction）

### 目标函数

给定一段文本 `x_1, x_2, ..., x_T`，模型最大化：

```
L = Σ_t log P(x_t | x_1, ..., x_{t-1})
```

具体实现：cross-entropy loss between 模型预测的 logits 和真实下一个 token。

### 数据采样

每个 batch：
1. 从 train.bin 随机选 `batch_size` 个起点
2. 每个起点取 `block_size+1` 个连续 token
3. `x = tokens[:-1]`, `y = tokens[1:]`（每个位置预测下一个）

### 学习率调度

```
lr(iter) = {
    iter / warmup_iters * learning_rate           if iter < warmup_iters
    cosine_decay 从 learning_rate → min_lr        if warmup_iters ≤ iter < lr_decay_iters
    min_lr                                          otherwise
}
```

- **warmup**：开头几百步 lr 从 0 线性升到峰值，避免初始大梯度炸权重
- **cosine decay**：之后用余弦曲线缓慢衰减，比线性衰减更平滑

### Mixed precision (BF16)

A30 / A100 / H100 / RTX 30+ 系列原生支持 BF16：
- 计算用 BF16（速度 2-4x 提升）
- 梯度累积、优化器状态保持 FP32（精度）
- 不需要 loss scaling（不像 FP16）

### DDP（B 档双卡训练）

```
GPU 0 ← batch[0:48]
GPU 1 ← batch[48:96]
   │              │
   ▼              ▼
forward         forward
   │              │
backward      backward
   │              │
   └──── all_reduce ──── 求平均梯度，所有卡同步
   │              │
   ▼              ▼
optim.step    optim.step    （结果一致）
```

每张卡：自己的 forward/backward，再用 NCCL all-reduce 同步梯度。
有效 batch_size = `per_gpu_batch × n_gpus`（即 48 × 2 = 96）。

启动命令：`torchrun --standalone --nproc_per_node=2 train.py config/poetry_medium.py`

## 推理采样

### Greedy（贪心）→ 重复严重，弃用

每次取概率最大的 token：
```
next_token = argmax(P(x_t | context))
```
确定性输出，但容易卡在循环（"风急风急风急..."）。

### Temperature + Top-K（实际使用）

```python
logits = model(context)[..., -1, :]        # [vocab_size]
logits = logits / temperature              # 放大/缩小差异
top_k_values = top_k_logits(logits, k)
probs = softmax(top_k_values)
next_token = multinomial(probs)            # 按概率采样
```

| 参数 | 调高 | 调低 |
|------|------|------|
| `temperature` | 更随机、更有创造力，也更容易胡言乱语 | 更保守、更接近最大概率路径 |
| `top_k` | 词汇更丰富 | 更"安全"，但可能重复 |

**经验值**：
- 想准确续写已知诗：`temperature=0.2, top_k=10`
- 想看创造性生成：`temperature=0.8, top_k=200`
- 想看"放飞自我"：`temperature=1.5, top_k=2000`

## 参考资源

- [Andrej Karpathy: GPT from scratch (YouTube)](https://www.youtube.com/watch?v=kCc8FmEb1nY) — nanoGPT 同款讲解，本项目基础
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Transformer 原始论文
- [GPT-2 paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — 大致同款架构
- [Chinchilla paper](https://arxiv.org/abs/2203.15556) — 模型大小 vs 数据量的最优配比研究

# 训练日志

## 硬件环境

| 项 | 配置 |
|---|---|
| GPU | 2 × NVIDIA A30 (24 GB HBM2 each) |
| GPU 算力 | ~165 TFLOPS BF16 / 卡 |
| CPU | Intel Xeon Platinum 8336C, 56 cores |
| 内存 | 460 GB |
| 磁盘 | 500 GB ext4 (`/data`) |
| OS | Ubuntu 22.04 (Linux 5.15) |
| CUDA / Driver | 12.4 / 550.144.03 |
| PyTorch | 2.6.0 + cu124 |

## 数据准备阶段

### 数据来源
- [chinese-poetry/chinese-poetry](https://github.com/chinese-poetry/chinese-poetry)（455 MB raw JSON）

### 子集选择
| 子集 | 字段名 | 字符 | 是否纳入 |
|------|--------|------|----------|
| 全唐诗 (poet.tang.*) | `paragraphs` | 繁体 | ✅ |
| 全唐诗 (poet.song.*) | `paragraphs` | 繁体 | ✅ |
| 宋词 (ci.song.*) | `paragraphs` | 简体 | ✅ |
| 元曲 (yuanqu.json) | `paragraphs` | 简体 | ✅ |
| 论语 (lunyu.json) | `paragraphs` | 简体 | ✅ |
| 楚辞 (chuci.json) | **`content`** | 简体 | ✅ |
| 诗经 (shijing.json) | **`content`** | 简体 | ✅ |
| 御定全唐詩 / 水墨唐诗 | — | — | ❌（与全唐诗重复） |

### 关键设计决策

**1. 简繁统一（zhconv → zh-cn）**
- 不处理：vocab 几乎翻倍（同字"国/國"两个 token），模型容量浪费在简繁对应上
- 处理后：vocab 12,573；不处理估计 ~20,000+

**2. 删除所有空白字符**
- 古文 corpus 里的空格大多是排版残留，无语义价值
- 副作用：模型不会处理含空格的 prompt（已在 sample.py 加 `if c in stoi` 保护）

**3. 文档分隔策略**
- 每个文档格式：`《标题》作者\n第一行\n第二行\n...`
- 文档之间用空行（`\n\n`）分隔
- 模型从这种结构隐式学到"标题→作者→正文"的生成模式
- ⚠️ 没有显式 `<|endoftext|>` token，导致 prompt 边界有时被误解析（见 samples/02 观察）

### 数据统计
```
唐诗:   58 个文件   ~50,000+ 首
宋诗:  255 个文件   ~250,000+ 首
宋词:   23 个文件   ~20,000+ 首
元曲:    1 个文件
论语:    1 个文件
楚辞:    1 个文件   65 篇章节
诗经:    1 个文件   305 篇

总文档数:    344,355
总字符数:     28,468,085
独立字符数:        12,573  (← vocab_size)
corpus.txt:    ~85 MB
```

## A 档训练（15M 参数）

### 配置回顾
```python
n_layer = 6
n_head = 6
n_embd = 384
block_size = 256
batch_size = 64
learning_rate = 1e-3
max_iters = 5000
warmup_iters = 200
dtype = "bfloat16"
```

### 模型参数估算
- Transformer blocks: 12 × n_layer × n_embd² ≈ 12 × 6 × 384² ≈ 10.6M
- Token embedding: vocab × n_embd = 12,573 × 384 ≈ 4.83M
- Position embedding: 256 × 384 ≈ 0.10M
- 其他 (LayerNorm, output projection 共享): 略
- **实测：15.45M**

### 训练曲线（关键节点）

| Iter | Train Loss | 备注 |
|------|-----------|------|
| 0 | ~9.4 | 接近 ln(12573) = 9.44，等于完全随机的理论 loss |
| 200 | ~6 | 学会高频字符 |
| 1000 | ~4.5 | 学会基本格律 |
| 2500 | ~3.8 | 风格成型 |
| 5000 | ~3.5 | 训练结束 |

最终 perplexity ≈ exp(3.5) ≈ **33** —— 模型对下一个字的"不确定性"从 12,573 降到 ~33。

### MFU (Model FLOPs Utilization)
- 实测 ~9.7%
- 相对偏低，原因：模型小、Python 数据加载/通信开销占比高
- 对学习目的影响不大；B 档模型大、能拉到更高 MFU

### 训练耗时
- 5000 iters × ~53 ms/iter ≈ 265 秒 ≈ **~5 分钟**
- 比预期 20-30 分钟快很多（A30 在 BF16 下表现不错）

## 踩过的坑

### 坑 1：磁盘挂载
- `/dev/vdb` 是空盘，最初格式化了两次导致 fstab 里 UUID 与磁盘真实 UUID 不一致
- `nofail` 选项把挂载失败错误吞掉，调试时一度看不到错误
- **教训**：调试期先去掉 `nofail`、用 `findmnt --verify` 体检

### 坑 2：sample.py 遇到空格 KeyError
- prompt 含空格但训练 corpus 把空格全删了，stoi 里没有 ' ' 这个 key
- 修复：encode lambda 加 `if c in stoi` 过滤未知字符
- **教训**：字符级 tokenizer 必须有 OOV 兜底；这是 BPE/SentencePiece 字节级 fallback 的优势

### 坑 3：prompt 与训练数据格式不一致
- 我们用 `《标题》作者\n正文` 隐式格式，没有特殊分隔 token
- 模型有时把 prompt 里的"风急天高"误解析为作者名延续
- **教训**：真实 LLM 都用 `<|im_start|>` / `<|im_end|>` 这类显式分隔，不是没原因的

## B 档训练（计划）

| 项 | A 档 | B 档 |
|---|---|---|
| 参数 | 15M | ~30M |
| n_layer | 6 | 8 |
| n_embd | 384 | 512 |
| max_iters | 5,000 | 15,000 |
| batch (effective) | 64 | 96 (DDP × 2) |
| 训练方式 | 单卡 | 双卡 DDP |
| 预计 val_loss | 3.5 | ≤ 2.8 |
| 预计耗时 | 5 分钟 | 2-3 小时 |

启动命令：
```bash
cd nanoGPT
torchrun --standalone --nproc_per_node=2 train.py config/poetry_medium.py
```

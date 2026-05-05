"""
config/poetry_small.py
A 档配置：~15M 参数，单卡 A30，约 5-30 分钟训完。

用法：
    cp configs/poetry_small.py nanoGPT/config/
    cd nanoGPT && python train.py config/poetry_small.py
"""

# ===== 输出 =====
out_dir = "out/poetry_small"
eval_interval = 250            # 每 250 步评估一次
eval_iters = 100               # 评估时跑 100 个 batch 求平均
log_interval = 10              # 每 10 步打印一次 loss
always_save_checkpoint = False # 只在 val_loss 创新低时保存

wandb_log = False              # 不用 wandb（如需启用，改 True 并配 wandb_project）

# ===== 数据 =====
dataset = "poetry"             # 对应 nanoGPT/data/poetry/ 目录
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256               # 上下文长度：诗词远小于 256 字符，足够

# ===== 模型 =====
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.1                  # 数据相对小，给点 dropout 防过拟合
bias = False

# ===== 优化器 =====
learning_rate = 1e-3           # 小模型可以用大点的 lr
max_iters = 5000               # 总训练步数
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 200

# ===== 系统 =====
device = "cuda"
compile = False                # PyTorch 2.x compile，先关掉避免兼容问题
dtype = "bfloat16"             # A30 / A100 / H100 / RTX 30+ 系列支持

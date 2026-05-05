"""
config/poetry_medium.py
B 档：~30M 参数，双卡 DDP，约 2-3 小时训完。

对比 A 档：层数 6→8、宽度 384→512、训练步数 5000→15000、加 DDP 双卡数据并行。

用法：
    cp configs/poetry_medium.py nanoGPT/config/
    cd nanoGPT && torchrun --standalone --nproc_per_node=2 train.py config/poetry_medium.py
"""

# ===== 输出 =====
out_dir = "out/poetry_medium"
eval_interval = 500
eval_iters = 100
log_interval = 20
always_save_checkpoint = False
wandb_log = False

# ===== 数据 =====
dataset = "poetry"
gradient_accumulation_steps = 1
batch_size = 48                # 每张 GPU；2 卡 DDP 实际 batch=96
block_size = 256

# ===== 模型（~30M 参数） =====
n_layer = 8
n_head = 8
n_embd = 512
dropout = 0.1
bias = False

# ===== 优化器 =====
learning_rate = 6e-4           # 模型变大，lr 适当降一点
max_iters = 15000              # ~9 epochs
lr_decay_iters = 15000
min_lr = 6e-5
beta2 = 0.95
warmup_iters = 500
weight_decay = 0.1
grad_clip = 1.0

# ===== 系统 =====
device = "cuda"
compile = False
dtype = "bfloat16"

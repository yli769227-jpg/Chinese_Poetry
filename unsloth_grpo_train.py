"""
unsloth_grpo_train.py
=====================

基于 **Unsloth + TRL GRPOTrainer** 的强化学习训练入口，
与现有的 nanoGPT SFT/from-scratch 训练流程**并行存在**。

设计要点
--------
1. Base model：默认 **Qwen2.5-0.5B-Instruct**（最小可用且支持中文）。
   注：现有 nanoGPT 项目是从零训练的字符级 GPT，无现成 HF 权重，
   因此这里用 Qwen 小尺寸作为 GRPO 的起点；不是替代 SFT，是并行尝试。
2. Reward = `rewards.rhyme_pingze.combined_reward` (0.6 rhyme + 0.4 pingze)。
3. 数据来源：复用现有 `data/corpus.txt`（由 scripts/extract_corpus.py 生成），
   按 `\n\n` 切分为单首诗；prompt 取标题行 `《X》作者`，completion 由模型生成。
4. **--dry-run** 模式：不加载模型权重，只验证：
   - rewards 模块 import 成功
   - dataset 第一条加载成功
   - 第一首诗能算出 reward
   用于在无 GPU 的环境(如 macOS 本地)做冒烟测试。
5. 关键路径日志：连接、状态变化、错误、请求/响应，prefix `[grpo]`。

跑法
----
    # 冒烟（无 GPU 也能跑）
    python unsloth_grpo_train.py --dry-run

    # 正式训练（需 CUDA + unsloth + trl）
    python unsloth_grpo_train.py \\
        --base-model unsloth/Qwen2.5-0.5B-Instruct \\
        --corpus data/corpus.txt \\
        --output-dir out/grpo_qwen05 \\
        --max-steps 200
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# -----------------------------------------------------------------------------
# 日志
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [grpo] %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("unsloth_grpo_train")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))  # 保证 rewards.* 能被 import

DEFAULT_CORPUS = PROJECT_ROOT / "data" / "corpus.txt"
DEFAULT_BASE_MODEL = "unsloth/Qwen2.5-0.5B-Instruct"
DEFAULT_OUT_DIR = PROJECT_ROOT / "out" / "grpo_qwen05"


# -----------------------------------------------------------------------------
# Dataset 加载
# -----------------------------------------------------------------------------
def load_poems(corpus_path: Path, max_n: int | None = None) -> List[Dict[str, str]]:
    """读 corpus.txt → list of {"prompt": <《标题》作者>, "title": ..., "raw": <全文>}。

    Prompt = 头部行（《标题》作者）+ 换行；
    Completion = 模型补出的正文。
    若文档缺标题（论语等），fallback 用前 6 字符作 prompt。
    """
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"corpus not found: {corpus_path}\n"
            "请先跑 scripts/extract_corpus.py 生成 data/corpus.txt"
        )

    text = corpus_path.read_text(encoding="utf-8")
    raw_docs = [d.strip() for d in text.split("\n\n") if d.strip()]
    logger.info("loaded %d raw documents from %s", len(raw_docs), corpus_path)

    poems: List[Dict[str, str]] = []
    for doc in raw_docs:
        lines = doc.split("\n")
        head = lines[0]
        if "》" in head:
            prompt = head + "\n"
        else:
            prompt = doc[:6] + "\n"
        poems.append({"prompt": prompt, "title": head, "raw": doc})
        if max_n is not None and len(poems) >= max_n:
            break

    logger.info("prepared %d prompt/poem pairs", len(poems))
    return poems


# -----------------------------------------------------------------------------
# Reward 适配 TRL GRPOTrainer 接口
# -----------------------------------------------------------------------------
def build_reward_fn():
    """返回一个 TRL GRPOTrainer 期望的 reward function。

    TRL 0.11+ GRPOTrainer 的 reward function 签名:
        fn(prompts: List[str], completions: List[str], **kwargs) -> List[float]
    我们把 prompt + completion 拼回成一首"完整诗"再喂 combined_reward。
    """
    from rewards.rhyme_pingze import combined_reward

    _step_counter = {"n": 0}

    def reward_fn(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
        t0 = time.time()
        rewards: List[float] = []
        for p, c in zip(prompts, completions):
            full_poem = (p or "") + (c or "")
            try:
                r = combined_reward(full_poem)
            except Exception as e:
                logger.warning("[grpo] reward_fn failed: %s", e)
                r = 0.0
            rewards.append(r)

        _step_counter["n"] += 1
        mean_r = sum(rewards) / max(1, len(rewards))
        logger.info(
            "step=%d batch=%d mean_reward=%.3f took=%.2fs",
            _step_counter["n"], len(rewards), mean_r, time.time() - t0,
        )
        return rewards

    return reward_fn


# -----------------------------------------------------------------------------
# Dry-run：不加载模型，只跑数据 + reward
# -----------------------------------------------------------------------------
def run_dry_run(args) -> int:
    logger.info("=== DRY-RUN MODE ===")

    # 1) import rewards
    try:
        from rewards.rhyme_pingze import combined_reward, pingze_reward, rhyme_reward
        logger.info("rewards module imported OK")
    except Exception as e:
        logger.error("FAILED to import rewards: %s", e)
        return 1

    # 2) 试着 import 重型库（不必成功），仅记录
    for libname in ("unsloth", "trl", "torch", "transformers", "datasets", "pypinyin"):
        try:
            __import__(libname)
            logger.info("optional dep present: %s", libname)
        except Exception as e:
            logger.warning("optional dep missing: %s (%s)", libname, e)

    # 3) load dataset 第一条
    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        logger.warning("corpus %s not found — synthesizing a tiny fake sample", corpus_path)
        sample_poem = {
            "prompt": "《静夜思》李白\n",
            "title": "《静夜思》李白",
            "raw": "《静夜思》李白\n床前明月光，疑是地上霜。\n举头望明月，低头思故乡。",
        }
    else:
        try:
            poems = load_poems(corpus_path, max_n=1)
            if not poems:
                logger.error("corpus loaded but empty")
                return 1
            sample_poem = poems[0]
            logger.info("first poem prompt: %r", sample_poem["prompt"])
            logger.info("first poem raw:    %r", sample_poem["raw"][:60])
        except Exception as e:
            logger.error("FAILED to load dataset: %s", e)
            return 1

    # 4) compute rewards on the first poem (its full text)
    try:
        r = rhyme_reward(sample_poem["raw"])
        p = pingze_reward(sample_poem["raw"])
        c = combined_reward(sample_poem["raw"])
        logger.info("first poem reward: rhyme=%.3f pingze=%.3f combined=%.3f", r, p, c)
    except Exception as e:
        logger.error("FAILED to compute reward: %s", e)
        return 1

    # 5) reward_fn 接口
    try:
        fn = build_reward_fn()
        batch = fn(
            prompts=[sample_poem["prompt"]],
            completions=[sample_poem["raw"][len(sample_poem["prompt"]):]],
        )
        logger.info("reward_fn batch output: %s", batch)
    except Exception as e:
        logger.error("FAILED reward_fn smoke: %s", e)
        return 1

    logger.info("=== DRY-RUN OK ===")
    return 0


# -----------------------------------------------------------------------------
# 真训练入口
# -----------------------------------------------------------------------------
def run_train(args) -> int:
    logger.info("=== TRAIN MODE base_model=%s ===", args.base_model)

    # 重型依赖 import 放在这里，dry-run 时根本不会执行
    try:
        from unsloth import FastLanguageModel  # type: ignore
    except Exception as e:
        logger.error(
            "unsloth import failed: %s\n"
            "Unsloth 需要 CUDA + Linux/WSL，macOS 暂不支持。"
            "如需在 mac 上跑，可改用 trl GRPOTrainer + 普通 HF model 加载。",
            e,
        )
        return 2

    try:
        import torch  # noqa: F401
        from datasets import Dataset
        from trl import GRPOConfig, GRPOTrainer
    except Exception as e:
        logger.error("trl/datasets/torch import failed: %s", e)
        return 2

    corpus_path = Path(args.corpus)
    poems = load_poems(corpus_path, max_n=args.max_train_samples)
    if len(poems) == 0:
        logger.error("no poems loaded, abort")
        return 1

    # GRPOTrainer 期望 column "prompt"
    ds = Dataset.from_list([{"prompt": pm["prompt"]} for pm in poems])
    logger.info("dataset size=%d  first prompt=%r", len(ds), ds[0]["prompt"])

    # 模型 + tokenizer
    logger.info("loading base model: %s", args.base_model)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=16,
    )
    logger.info("model + LoRA ready")

    reward_fn = build_reward_fn()

    cfg = GRPOConfig(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        logging_steps=1,
        save_steps=max(50, args.max_steps // 4),
        bf16=True,
        report_to="none",
        max_prompt_length=64,
        max_completion_length=args.max_seq_length - 64,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_fn],
        args=cfg,
        train_dataset=ds,
    )
    logger.info("starting GRPO training, max_steps=%d", args.max_steps)
    trainer.train()
    logger.info("training done, saving to %s", args.output_dir)
    trainer.save_model(str(args.output_dir))
    return 0


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def build_argparser():
    ap = argparse.ArgumentParser(
        description="GRPO trainer for Chinese poetry (rhyme + pingze reward)."
    )
    ap.add_argument("--dry-run", action="store_true",
                    help="不加载模型，只跑 import + dataset + reward 冒烟")
    ap.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    ap.add_argument("--corpus", default=str(DEFAULT_CORPUS),
                    help="path to data/corpus.txt produced by scripts/extract_corpus.py")
    ap.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--max-train-samples", type=int, default=2000)
    ap.add_argument("--max-seq-length", type=int, default=256)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--num-generations", type=int, default=4)
    ap.add_argument("--learning-rate", type=float, default=5e-6)
    return ap


def main():
    args = build_argparser().parse_args()
    args.output_dir = Path(args.output_dir)
    if args.dry_run:
        rc = run_dry_run(args)
    else:
        rc = run_train(args)
    logger.info("exit code: %d", rc)
    sys.exit(rc)


if __name__ == "__main__":
    main()

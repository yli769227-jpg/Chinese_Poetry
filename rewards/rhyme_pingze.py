"""
rewards/rhyme_pingze.py

Reward functions used by GRPO training (`unsloth_grpo_train.py`).

设计目标：
    给定模型生成的一首诗（字符串，可能含《标题》作者头部和正文若干句），
    输出 0~1 区间的浮点数 reward，用于强化学习信号。

简化策略（不是完整的《平水韵》/《广韵》体系，**够用就好**）：
    1. 押韵 (rhyme_reward)：
       - 切句：以「，。!?；！？，。」等为句末标点切句，取每个非空句的"末字"。
       - 用 pypinyin 取末字的 *韵母 (final)*（去掉声调），统计众数比例。
       - 占比即 reward。一句或零句保守返回 0.0。
       - 习惯：古诗常偶数句押韵，所以只看偶数句末字（第 2、4、6 ... 句）。
         若总句数 < 2 退化为看所有句。

    2. 平仄 (pingze_reward)：
       - 用 pypinyin tone 取每字声调：1、2 声 → 平 (P)；3、4 声 → 仄 (Z)；轻声 → 跳过。
       - 五/七言一句内应有"平仄交替"倾向，**逐字两两不同**的占比即一句 score。
       - 全诗 score = 各句 score 的均值。

    3. combined_reward = 0.6 * rhyme + 0.4 * pingze。

参考：
    pypinyin 0.50+：lazy_pinyin(style=Style.FINALS) / Style.TONE3
    https://pypinyin.readthedocs.io/

日志：
    所有 reward 函数都用 prefix `[grpo]` 打 logger.debug，避免在大批量 rollout 时刷屏；
    `unsloth_grpo_train.py` 里在 step 边界上打 INFO 级日志。
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from typing import List

logger = logging.getLogger(__name__)

# 句末标点（中英文都覆盖）
_SENT_END = "，。！？；,.!?;"
_SPLIT_RE = re.compile(f"[{re.escape(_SENT_END)}]")

# 仅保留汉字的正则（CJK 基本区 + 扩展 A）
_HAN_RE = re.compile(r"[㐀-鿿]")


# 简化的"韵类"归并表：把现代普通话韵母粗合并到几个传统韵类。
# 不是严格《平水韵》/《广韵》，仅用于 reward signal。
# key: 单字最终归到的韵类标签；value: 该类下若干现代韵母变体。
_RHYME_CLASS_MAP = {
    "a":     {"a", "ia", "ua"},
    "o":     {"o", "uo", "e"},
    "ie":    {"ie", "ue", "ve", "üe"},
    "i":     {"i", "er", "ir"},
    "u":     {"u"},
    "v":     {"v", "ü"},
    "ai":    {"ai", "uai"},
    "ei":    {"ei", "ui", "uei"},
    "ao":    {"ao", "iao"},
    "ou":    {"ou", "iu", "iou"},
    "an":    {"an", "ian", "uan", "van", "üan"},
    "en":    {"en", "in", "un", "vn", "ün", "uen"},
    "ang":   {"ang", "iang", "uang"},
    "eng":   {"eng", "ing", "ong", "iong", "ueng"},
}


def _to_rhyme_class(final: str) -> str:
    """把一个 (去声调) 拼音 final 映射到上面的粗韵类标签。

    若都不匹配，原样返回（用作 fallback 标签）。
    """
    if not final:
        return ""
    for label, members in _RHYME_CLASS_MAP.items():
        if final in members:
            return label
    return final


def _strip_to_body(poem: str) -> str:
    """去掉 《...》 标题头和「作者」前缀，保留正文。

    保险起见，如果检测不到 》，则原样返回。
    """
    if "》" in poem:
        # 取 》 之后第一个换行后的内容（作者行通常和标题在同一行）
        after = poem.split("》", 1)[1]
        # 跳过作者行：第一个 \n 之后是正文
        if "\n" in after:
            return after.split("\n", 1)[1]
        return after
    return poem


def _split_sentences(poem: str) -> List[str]:
    """切句，只保留含汉字的句子。"""
    body = _strip_to_body(poem)
    raw = _SPLIT_RE.split(body)
    sents = []
    for s in raw:
        s = s.strip()
        # 只留下含汉字的句子
        han_only = "".join(_HAN_RE.findall(s))
        if han_only:
            sents.append(han_only)
    return sents


# -----------------------------------------------------------------------------
# 押韵
# -----------------------------------------------------------------------------
def rhyme_reward(poem: str) -> float:
    """韵脚一致度 0~1。看偶数句末字的韵母 (final) 众数占比。"""
    from pypinyin import lazy_pinyin, Style

    sents = _split_sentences(poem)
    if len(sents) < 2:
        logger.debug("[grpo] rhyme_reward: <2 sentences, return 0.0")
        return 0.0

    # 偶数句末字（1-indexed 偶数 → 0-indexed 奇数）
    rhyme_sents = [s for i, s in enumerate(sents) if i % 2 == 1]
    if len(rhyme_sents) < 2:
        # 退化：看所有句
        rhyme_sents = sents

    last_chars = [s[-1] for s in rhyme_sents if s]
    if not last_chars:
        return 0.0

    finals = []
    for ch in last_chars:
        try:
            fin = lazy_pinyin(ch, style=Style.FINALS_TONE3, strict=False)
            if fin and fin[0]:
                # 去声调数字
                f = re.sub(r"\d+$", "", fin[0])
                # 归并到粗韵类
                finals.append(_to_rhyme_class(f))
        except Exception as e:
            logger.debug("[grpo] rhyme_reward: pypinyin failed on %r: %s", ch, e)
            continue

    if not finals:
        return 0.0

    counter = Counter(finals)
    most_common_n = counter.most_common(1)[0][1]
    score = most_common_n / len(finals)
    logger.debug(
        "[grpo] rhyme_reward: finals=%s mode_count=%d total=%d -> %.3f",
        finals, most_common_n, len(finals), score,
    )
    return float(score)


# -----------------------------------------------------------------------------
# 平仄
# -----------------------------------------------------------------------------
def _tone_to_pingze(ch: str) -> str | None:
    """单字 → 'P' (平) / 'Z' (仄) / None (跳过)。"""
    from pypinyin import lazy_pinyin, Style

    try:
        pin = lazy_pinyin(ch, style=Style.TONE3, strict=False)
        if not pin or not pin[0]:
            return None
        m = re.search(r"(\d)$", pin[0])
        if not m:
            return None  # 轻声或无声调
        tone = int(m.group(1))
        if tone in (1, 2):
            return "P"
        if tone in (3, 4):
            return "Z"
    except Exception as e:
        logger.debug("[grpo] _tone_to_pingze: failed on %r: %s", ch, e)
    return None


def pingze_reward(poem: str) -> float:
    """平仄交替度 0~1。每句内逐字两两不同的占比，多句取均值。"""
    sents = _split_sentences(poem)
    if not sents:
        logger.debug("[grpo] pingze_reward: no sentence, return 0.0")
        return 0.0

    sent_scores = []
    for s in sents:
        tones = [_tone_to_pingze(c) for c in s]
        tones = [t for t in tones if t is not None]
        if len(tones) < 2:
            continue
        # 相邻不同的对数 / 总对数
        diffs = sum(1 for a, b in zip(tones, tones[1:]) if a != b)
        total_pairs = len(tones) - 1
        sent_scores.append(diffs / total_pairs)

    if not sent_scores:
        return 0.0
    score = sum(sent_scores) / len(sent_scores)
    logger.debug(
        "[grpo] pingze_reward: per_sent=%s -> %.3f",
        [round(x, 2) for x in sent_scores], score,
    )
    return float(score)


# -----------------------------------------------------------------------------
# 组合
# -----------------------------------------------------------------------------
def combined_reward(poem: str) -> float:
    """combined = 0.6 * rhyme + 0.4 * pingze."""
    r = rhyme_reward(poem)
    p = pingze_reward(poem)
    score = 0.6 * r + 0.4 * p
    logger.debug("[grpo] combined_reward: rhyme=%.3f pingze=%.3f -> %.3f", r, p, score)
    return float(score)


# -----------------------------------------------------------------------------
# 当作脚本运行：快速 sanity check
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(message)s")
    demo = "《静夜思》李白\n床前明月光，疑是地上霜。\n举头望明月，低头思故乡。"
    print(f"demo poem:\n{demo}")
    print(f"rhyme    = {rhyme_reward(demo):.3f}")
    print(f"pingze   = {pingze_reward(demo):.3f}")
    print(f"combined = {combined_reward(demo):.3f}")

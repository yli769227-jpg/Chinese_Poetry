"""
tests/test_rewards.py

单测 rewards.rhyme_pingze 中的三个 reward 函数。

核心断言：
    5 首真诗(李白/白居易/王维等) 的均分 > 5 首乱写文字的均分。
    （宽松断言：差距 >= 0.1，避免某次小概率反例。）

跑法：
    cd Chinese_Poetry && python -m pytest tests/test_rewards.py -v
    或：python tests/test_rewards.py
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

# 允许直接 `python tests/test_rewards.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rewards.rhyme_pingze import (  # noqa: E402
    combined_reward,
    pingze_reward,
    rhyme_reward,
)

REAL_POEMS = [
    # 1. 李白《静夜思》
    "《静夜思》李白\n床前明月光，疑是地上霜。\n举头望明月，低头思故乡。",
    # 2. 白居易《赋得古原草送别》(节选 前四句)
    "《赋得古原草送别》白居易\n离离原上草，一岁一枯荣。\n野火烧不尽，春风吹又生。",
    # 3. 王维《相思》
    "《相思》王维\n红豆生南国，春来发几枝。\n愿君多采撷，此物最相思。",
    # 4. 李白《赠汪伦》
    "《赠汪伦》李白\n李白乘舟将欲行，忽闻岸上踏歌声。\n桃花潭水深千尺，不及汪伦送我情。",
    # 5. 杜甫《绝句》
    "《绝句》杜甫\n两个黄鹂鸣翠柳，一行白鹭上青天。\n窗含西岭千秋雪，门泊东吴万里船。",
]

# 5 首"乱写"——结构像诗(有《标题》作者，有逗号句号)，但内容随机：
# 末字不押韵、字也尽量平仄不交替
GARBAGE_POEMS = [
    "《乱码一》无名\n冰箱电视空调机，桌椅板凳塑料桶。\n手机电脑充电线，地铁公交三轮车。",
    "《乱码二》随机\n苹果香蕉橙子梨，铅笔橡皮订书机。\n键盘鼠标显示器，沙发地毯吊灯泡。",
    "《乱码三》瞎写\n打印机器复印纸，咖啡奶茶豆浆机。\n电梯楼梯消防门，垃圾桶里塑料袋。",
    "《乱码四》随便\n洗衣机器烘干机，热水瓶里塑料杯。\n窗帘床单被罩枕，毛巾牙刷剃须刀。",
    "《乱码五》凑数\n葡萄柚汁西瓜瓤，巧克力派饼干筒。\n摩托轿车自行车，安全帽子护膝套。",
]


def _avg(scores):
    return statistics.fmean(scores)


def test_rhyme_real_beats_garbage():
    real_scores = [rhyme_reward(p) for p in REAL_POEMS]
    garbage_scores = [rhyme_reward(p) for p in GARBAGE_POEMS]
    real_avg = _avg(real_scores)
    garbage_avg = _avg(garbage_scores)
    print(f"\n[rhyme] real_avg={real_avg:.3f}  garbage_avg={garbage_avg:.3f}")
    print(f"[rhyme] real    per-poem: {[round(x,3) for x in real_scores]}")
    print(f"[rhyme] garbage per-poem: {[round(x,3) for x in garbage_scores]}")
    assert real_avg > garbage_avg, (
        f"rhyme: real ({real_avg:.3f}) should beat garbage ({garbage_avg:.3f})"
    )


def test_pingze_real_beats_garbage():
    real_scores = [pingze_reward(p) for p in REAL_POEMS]
    garbage_scores = [pingze_reward(p) for p in GARBAGE_POEMS]
    real_avg = _avg(real_scores)
    garbage_avg = _avg(garbage_scores)
    print(f"\n[pingze] real_avg={real_avg:.3f}  garbage_avg={garbage_avg:.3f}")
    print(f"[pingze] real    per-poem: {[round(x,3) for x in real_scores]}")
    print(f"[pingze] garbage per-poem: {[round(x,3) for x in garbage_scores]}")
    # 双字音步建模后真诗应显著高于乱写（实测 ~0.73 vs ~0.38）
    assert real_avg > garbage_avg + 0.1, (
        f"pingze: real ({real_avg:.3f}) should clearly beat "
        f"garbage ({garbage_avg:.3f}) with margin >= 0.1"
    )


def test_pingze_directional_ordering():
    """方向性断言：标准律句 > 逐字交替句 > 全平/全仄句。

    旧实现（逐字交替占比）下逐字交替得满分 1.0、标准律句只得 0.5，
    信号反向；双字音步建模修复后顺序必须是律句最高、单调句最低。
    """
    # 标准七言律句：平平仄仄平平仄，仄仄平平仄仄平（杜甫《登高》）
    lvju = "无边落木萧萧下，不尽长江滚滚来。"
    # 逐字平仄交替（平仄平仄平仄平）——出律，旧实现却给满分
    alternating = "天地天地天地天，地天地天地天地。"
    # 全平 / 全仄 —— 声调单调，应最低
    all_ping = "悠悠悠悠悠悠悠，茫茫茫茫茫茫茫。"
    all_ze = "落落落落落落落，月月月月月月月。"

    s_lvju = pingze_reward(lvju)
    s_alt = pingze_reward(alternating)
    s_ping = pingze_reward(all_ping)
    s_ze = pingze_reward(all_ze)
    print(
        f"\n[pingze-dir] 律句={s_lvju:.3f} 逐字交替={s_alt:.3f} "
        f"全平={s_ping:.3f} 全仄={s_ze:.3f}"
    )
    assert s_lvju > s_alt, f"律句 ({s_lvju:.3f}) 应高于逐字交替 ({s_alt:.3f})"
    assert s_alt > s_ping, f"逐字交替 ({s_alt:.3f}) 应高于全平 ({s_ping:.3f})"
    assert s_alt > s_ze, f"逐字交替 ({s_alt:.3f}) 应高于全仄 ({s_ze:.3f})"


def test_combined_real_beats_garbage():
    real_scores = [combined_reward(p) for p in REAL_POEMS]
    garbage_scores = [combined_reward(p) for p in GARBAGE_POEMS]
    real_avg = _avg(real_scores)
    garbage_avg = _avg(garbage_scores)
    print(f"\n[combined] real_avg={real_avg:.3f}  garbage_avg={garbage_avg:.3f}")
    assert real_avg > garbage_avg + 0.05, (
        f"combined: real ({real_avg:.3f}) should clearly beat "
        f"garbage ({garbage_avg:.3f}) with margin >= 0.05"
    )


def test_score_bounds():
    """所有 reward 必须落在 [0, 1]。"""
    for p in REAL_POEMS + GARBAGE_POEMS:
        for fn in (rhyme_reward, pingze_reward, combined_reward):
            v = fn(p)
            assert 0.0 <= v <= 1.0, f"{fn.__name__}({p[:10]}...) -> {v} out of [0,1]"


if __name__ == "__main__":
    # 无 pytest 时可直接跑
    print("=" * 60)
    print("test_rewards.py — manual run")
    print("=" * 60)
    test_score_bounds()
    print("OK score_bounds")
    test_rhyme_real_beats_garbage()
    print("OK rhyme")
    test_pingze_real_beats_garbage()
    print("OK pingze")
    test_pingze_directional_ordering()
    print("OK pingze directional ordering")
    test_combined_real_beats_garbage()
    print("OK combined")
    print("\nAll tests passed.")

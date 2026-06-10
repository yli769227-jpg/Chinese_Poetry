"""
tests/test_grpo_dataset.py

针对 unsloth_grpo_train.load_poems 的数据切分单测。

核心断言（对应 dry-run completion 切片错位 fix）：
    无论有无标题，prompt 与 completion 拼回必须能无损还原原文，
    fallback（无《标题》）路径不得吞掉正文第 7 个字符。
"""
from __future__ import annotations

import sys
from pathlib import Path

# 允许直接 `python tests/test_grpo_dataset.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from unsloth_grpo_train import load_poems  # noqa: E402


def _write_corpus(tmp_path: Path, docs):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("\n\n".join(docs), encoding="utf-8")
    return corpus


def test_titled_poem_prompt_completion_lossless(tmp_path):
    doc = "《静夜思》李白\n床前明月光，疑是地上霜。"
    corpus = _write_corpus(tmp_path, [doc])
    poems = load_poems(corpus)
    assert len(poems) == 1
    pm = poems[0]
    # prompt 含人造换行；去掉它再拼 completion 应还原 raw
    assert pm["prompt"].rstrip("\n") + pm["completion"] == pm["raw"] == doc


def test_untitled_fallback_does_not_drop_char(tmp_path):
    # 无《标题》——走 doc[:6] fallback。doc 第 7 个字符是「也」，绝不能被吞。
    doc = "学而时习之也不亦说乎"  # 长度 > 6
    corpus = _write_corpus(tmp_path, [doc])
    poems = load_poems(corpus)
    pm = poems[0]
    assert "》" not in pm["prompt"]
    # prompt 前缀 = 前 6 字 + 换行；completion 必须从第 7 字「之」起，不丢字符
    assert pm["prompt"] == doc[:6] + "\n"
    assert pm["completion"] == doc[6:]
    assert pm["completion"][0] == doc[6]  # 第 7 个字符未被吞
    assert pm["prompt"].rstrip("\n") + pm["completion"] == doc


def test_short_untitled_doc_no_index_error(tmp_path):
    # doc 比 6 字符还短时不应越界，completion 为空
    doc = "之乎也"
    corpus = _write_corpus(tmp_path, [doc])
    pm = load_poems(corpus)[0]
    assert pm["prompt"] == doc + "\n"
    assert pm["completion"] == ""
    assert pm["prompt"].rstrip("\n") + pm["completion"] == doc


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        tp = Path(d)
        test_titled_poem_prompt_completion_lossless(tp)
        test_untitled_fallback_does_not_drop_char(tp)
        test_short_untitled_doc_no_index_error(tp)
    print("All grpo dataset tests passed.")

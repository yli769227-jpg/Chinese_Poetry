"""
extract_corpus.py
读取 chinese-poetry 仓库下的各子集 JSON 文件，
合并为一个统一格式的 corpus.txt，全部转简体。

输出格式：每首诗/每段是一个"文档"，文档之间空行分隔。
模型会通过这种结构学会"标题 → 作者 → 正文"的生成模式。

用法（在项目根目录下）：
    mkdir -p data
    python scripts/extract_corpus.py
"""
import json
import re
from pathlib import Path

from tqdm import tqdm
from zhconv import convert

# ---- 路径配置（按需调整）----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "raw" / "chinese-poetry"
OUT_FILE = PROJECT_ROOT / "data" / "corpus.txt"
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)


def to_simplified(s: str) -> str:
    """繁体 -> 简体"""
    return convert(s, "zh-cn") if s else ""


def clean(s: str) -> str:
    """去除奇怪空白字符 + 简繁统一"""
    return re.sub(r"\s+", "", to_simplified(s))


def format_poem(title, author, lines, prefix="《", suffix="》"):
    """统一格式：《标题》作者\n正文行..."""
    parts = []
    head = ""
    if title:
        head += f"{prefix}{clean(title)}{suffix}"
    if author:
        head += clean(author)
    if head:
        parts.append(head)
    for line in lines:
        line = clean(line)
        if line:
            parts.append(line)
    return "\n".join(parts)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_paragraph_subset(dir_glob, label):
    """处理 paragraphs 字段的子集（唐诗 / 宋诗 / 宋词 / 元曲 / 论语）"""
    files = sorted(RAW_DIR.glob(dir_glob))
    docs = []
    for fp in tqdm(files, desc=label):
        try:
            data = load_json(fp)
        except Exception as e:
            print(f"  跳过 {fp}: {e}")
            continue
        for entry in data:
            title = entry.get("title") or entry.get("rhythmic") or entry.get("chapter") or ""
            author = entry.get("author", "")
            lines = entry.get("paragraphs", [])
            doc = format_poem(title, author, lines)
            if doc:
                docs.append(doc)
    return docs


def extract_content_subset(json_path, label):
    """处理 content 字段的子集（楚辞 / 诗经）"""
    if not json_path.exists():
        return []
    data = load_json(json_path)
    docs = []
    for entry in tqdm(data, desc=label):
        title = entry.get("title", "")
        author = entry.get("author", "")
        section = entry.get("section", "")
        chapter = entry.get("chapter", "")
        full_title = title
        if chapter or section:
            full_title = (
                f"{title}（{chapter}·{section}）"
                if chapter and section
                else f"{title}（{chapter or section}）"
            )
        lines = entry.get("content", [])
        doc = format_poem(full_title, author, lines)
        if doc:
            docs.append(doc)
    return docs


def main():
    all_docs = []

    # 唐诗 + 宋诗（注意 全唐诗/ 目录里同时有 poet.tang.* 和 poet.song.*）
    all_docs += extract_paragraph_subset("全唐诗/poet.tang.*.json", "唐诗")
    all_docs += extract_paragraph_subset("全唐诗/poet.song.*.json", "宋诗")

    # 宋词
    all_docs += extract_paragraph_subset("宋词/ci.song.*.json", "宋词")

    # 元曲
    all_docs += extract_paragraph_subset("元曲/yuanqu.json", "元曲")

    # 论语
    all_docs += extract_paragraph_subset("论语/lunyu.json", "论语")

    # 楚辞 / 诗经
    all_docs += extract_content_subset(RAW_DIR / "楚辞/chuci.json", "楚辞")
    all_docs += extract_content_subset(RAW_DIR / "诗经/shijing.json", "诗经")

    # 写出
    print(f"\n总文档数: {len(all_docs):,}")
    text = "\n\n".join(all_docs)
    print(f"总字符数: {len(text):,}")
    print(f"独立字符数: {len(set(text)):,}")
    OUT_FILE.write_text(text, encoding="utf-8")
    print(f"\n已写入: {OUT_FILE}")
    print(f"文件大小: {OUT_FILE.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()

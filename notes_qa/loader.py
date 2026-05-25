import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Document:
    """一个笔记文档。"""
    content: str
    file_path: str
    title: str


_FRONT_MATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)
_HEADING_RE = re.compile(r"^#+\s+(.+)", re.MULTILINE)


def _strip_front_matter(text: str) -> str:
    """去除 Markdown 的 YAML front matter。"""
    return _FRONT_MATTER_RE.sub("", text, count=1).lstrip()


def _extract_title(text: str, file_path: str) -> str:
    """从 Markdown 内容中提取第一个标题，找不到则用文件名。"""
    m = _HEADING_RE.search(text)
    if m:
        return m.group(1).strip()
    return Path(file_path).stem


def load_documents(notes_dir: str | Path) -> list[Document]:
    """递归扫描目录，加载所有 Markdown 文件为 Document 列表。"""
    notes_dir = Path(notes_dir)
    if not notes_dir.is_dir():
        raise NotADirectoryError(f"笔记目录不存在: {notes_dir}")

    documents = []
    for md_file in sorted(notes_dir.rglob("*.md")):
        raw = md_file.read_text(encoding="utf-8")
        content = _strip_front_matter(raw)
        if not content.strip():
            continue
        doc = Document(
            content=content,
            file_path=str(md_file),
            title=_extract_title(content, str(md_file)),
        )
        documents.append(doc)

    return documents

import re
from dataclasses import dataclass

from notes_qa.loader import Document


@dataclass
class Chunk:
    """一个文本块。"""
    content: str
    file_path: str
    title: str
    chunk_index: int


# 分隔符优先级：heading > 双换行 > 单换行 > 句号
_SEPARATORS = [
    re.compile(r"\n(?=#{1,6}\s)"),   # heading 行
    re.compile(r"\n\s*\n"),          # 空行（段落分隔）
    re.compile(r"\n"),               # 单换行
    re.compile(r"[。！？.!?]"),      # 句号等
]


def _split_recursive(text: str, separators: list[re.Pattern], chunk_size: int) -> list[str]:
    """按分隔符优先级递归拆分文本。"""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    if not separators:
        # 没有更多分隔符，强制按字符数切分
        parts = []
        for i in range(0, len(text), chunk_size):
            parts.append(text[i:i + chunk_size])
        return parts

    sep = separators[0]
    rest_seps = separators[1:]

    segments = sep.split(text)
    chunks = []
    current = ""

    for seg in segments:
        candidate = (current + seg) if current else seg
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current.strip():
                chunks.append(current.strip())
            # 当前段本身就超长，递归用更细的分隔符拆分
            if len(seg) > chunk_size:
                chunks.extend(_split_recursive(seg, rest_seps, chunk_size))
                current = ""
            else:
                current = seg

    if current.strip():
        chunks.append(current.strip())

    return chunks


def _add_overlap(chunks: list[str], overlap: int) -> list[str]:
    """给相邻块添加重叠：把前一块的末尾拼到后一块开头。"""
    if overlap <= 0 or len(chunks) <= 1:
        return chunks
    result = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_tail = chunks[i - 1][-overlap:]
        result.append(prev_tail + chunks[i])
    return result


def chunk_document(doc: Document, chunk_size: int = 500, chunk_overlap: int = 50) -> list[Chunk]:
    """将一个 Document 拆分为 Chunk 列表。"""
    raw_chunks = _split_recursive(doc.content, _SEPARATORS, chunk_size)
    raw_chunks = _add_overlap(raw_chunks, chunk_overlap)

    return [
        Chunk(
            content=c,
            file_path=doc.file_path,
            title=doc.title,
            chunk_index=i,
        )
        for i, c in enumerate(raw_chunks)
    ]


def chunk_documents(documents: list[Document], chunk_size: int = 500, chunk_overlap: int = 50) -> list[Chunk]:
    """批量将 Document 列表拆分为 Chunk 列表。"""
    chunks = []
    for doc in documents:
        chunks.extend(chunk_document(doc, chunk_size, chunk_overlap))
    return chunks

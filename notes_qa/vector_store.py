import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from notes_qa.chunker import Chunk


class VectorStore:
    """基于 numpy 的向量存储与检索，支持持久化。"""

    def __init__(self, store_dir: str | Path):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self._vectors: np.ndarray | None = None
        self._metas: list[dict] = []

    @property
    def _vectors_path(self) -> Path:
        return self.store_dir / "vectors.npy"

    @property
    def _metas_path(self) -> Path:
        return self.store_dir / "metas.json"

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """添加 chunks 及其对应的向量。"""
        if len(chunks) != len(embeddings):
            raise ValueError(f"chunks 数量 ({len(chunks)}) 与 embeddings 数量 ({len(embeddings)}) 不匹配")

        new_metas = [
            {
                "content": c.content,
                "file_path": c.file_path,
                "title": c.title,
                "chunk_index": c.chunk_index,
            }
            for c in chunks
        ]

        if self._vectors is None or len(self._vectors) == 0:
            self._vectors = embeddings.astype(np.float32)
            self._metas = new_metas
        else:
            self._vectors = np.vstack([self._vectors, embeddings.astype(np.float32)])
            self._metas.extend(new_metas)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[dict]:
        """余弦相似度检索，返回 top_k 个结果。每个结果包含 content, file_path, title, score。"""
        if self._vectors is None or len(self._vectors) == 0:
            return []

        # 向量已归一化，余弦相似度 = 点积
        scores = self._vectors @ query_embedding
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            meta = self._metas[idx].copy()
            meta["score"] = float(scores[idx])
            results.append(meta)
        return results

    def save(self) -> None:
        """持久化向量和元数据到磁盘。"""
        if self._vectors is not None:
            np.save(str(self._vectors_path), self._vectors)
        with open(self._metas_path, "w", encoding="utf-8") as f:
            json.dump(self._metas, f, ensure_ascii=False, indent=2)

    def load(self) -> bool:
        """从磁盘加载向量和元数据，返回是否成功。"""
        if not self._vectors_path.exists() or not self._metas_path.exists():
            return False
        self._vectors = np.load(str(self._vectors_path))
        with open(self._metas_path, encoding="utf-8") as f:
            self._metas = json.load(f)
        return True

    def clear(self) -> None:
        """清空内存中的数据。"""
        self._vectors = None
        self._metas = []

    @property
    def count(self) -> int:
        return len(self._metas)

import tempfile
from pathlib import Path

import numpy as np

from notes_qa.chunker import Chunk
from notes_qa.vector_store import VectorStore


def _make_chunks(n: int) -> list[Chunk]:
    return [
        Chunk(content=f"chunk {i}", file_path=f"f{i}.md", title=f"T{i}", chunk_index=i)
        for i in range(n)
    ]


def test_add_and_search():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = VectorStore(tmpdir)
        chunks = _make_chunks(5)
        vecs = np.random.randn(5, 32).astype(np.float32)
        # 归一化
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

        store.add(chunks, vecs)
        assert store.count == 5

        query = vecs[0]  # 与第一个完全一致
        results = store.search(query, top_k=3)
        assert len(results) == 3
        assert results[0]["file_path"] == "f0.md"
        assert results[0]["score"] > 0.99


def test_save_and_load():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = VectorStore(tmpdir)
        chunks = _make_chunks(3)
        vecs = np.random.randn(3, 16).astype(np.float32)
        store.add(chunks, vecs)
        store.save()

        store2 = VectorStore(tmpdir)
        assert store2.load() is True
        assert store2.count == 3

        results = store2.search(vecs[0], top_k=1)
        assert len(results) == 1


def test_empty_store():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = VectorStore(tmpdir)
        results = store.search(np.zeros(16), top_k=5)
        assert results == []


def test_incremental_add():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = VectorStore(tmpdir)
        chunks1 = _make_chunks(2)
        vecs1 = np.random.randn(2, 8).astype(np.float32)
        store.add(chunks1, vecs1)
        assert store.count == 2

        chunks2 = _make_chunks(3)  # offset indices
        vecs2 = np.random.randn(3, 8).astype(np.float32)
        store.add(chunks2, vecs2)
        assert store.count == 5

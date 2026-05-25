import tempfile

from notes_qa.chunker import Chunk
from notes_qa.keyword_index import KeywordIndex


def _make_chunks() -> list[Chunk]:
    return [
        Chunk(content="Python 是一种编程语言，广泛用于人工智能开发", file_path="python.md", title="Python", chunk_index=0),
        Chunk(content="机器学习是人工智能的一个分支，Python 是常用工具", file_path="ml.md", title="机器学习", chunk_index=0),
        Chunk(content="今天天气很好，适合出去散步", file_path="weather.md", title="天气", chunk_index=0),
    ]


def test_build_and_search():
    with tempfile.TemporaryDirectory() as tmpdir:
        idx = KeywordIndex(tmpdir)
        chunks = _make_chunks()
        idx.build(chunks)
        assert idx.count == 3

        results = idx.search("Python 编程", top_k=2)
        assert len(results) == 2
        # Python 相关文档应排在前面
        titles = [r["title"] for r in results]
        assert "Python" in titles or "机器学习" in titles

        idx.close()


def test_search_no_match():
    with tempfile.TemporaryDirectory() as tmpdir:
        idx = KeywordIndex(tmpdir)
        idx.build(_make_chunks())

        results = idx.search("量子力学", top_k=5)
        assert len(results) == 0
        idx.close()


def test_search_returns_scores():
    with tempfile.TemporaryDirectory() as tmpdir:
        idx = KeywordIndex(tmpdir)
        idx.build(_make_chunks())

        results = idx.search("Python", top_k=3)
        for r in results:
            assert "score" in r
            assert r["score"] > 0
        idx.close()


def test_rebuild():
    with tempfile.TemporaryDirectory() as tmpdir:
        idx = KeywordIndex(tmpdir)
        idx.build(_make_chunks())
        assert idx.count == 3

        # 重新构建
        new_chunks = [Chunk(content="全新内容", file_path="new.md", title="新", chunk_index=0)]
        idx.build(new_chunks)
        assert idx.count == 1
        idx.close()

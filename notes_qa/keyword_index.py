import json
import math
import sqlite3
from pathlib import Path

import jieba

from notes_qa.chunker import Chunk


class KeywordIndex:
    """基于 jieba 分词 + SQLite 倒排索引的全文检索，支持 BM25 排序。"""

    def __init__(self, index_dir: str | Path, k1: float = 1.5, b: float = 0.75):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.index_dir / "keyword_index.db"
        self.k1 = k1
        self.b = b
        self._conn: sqlite3.Connection | None = None

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._init_tables()
        return self._conn

    def _init_tables(self) -> None:
        conn = self._conn
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                file_path TEXT NOT NULL,
                title TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                doc_length INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS postings (
                token TEXT NOT NULL,
                chunk_id INTEGER NOT NULL,
                term_freq INTEGER NOT NULL,
                PRIMARY KEY (token, chunk_id),
                FOREIGN KEY (chunk_id) REFERENCES chunks(id)
            );

            CREATE TABLE IF NOT EXISTS stats (
                key TEXT PRIMARY KEY,
                value REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_postings_token ON postings(token);
        """)
        conn.commit()

    def _tokenize(self, text: str) -> list[str]:
        """jieba 分词，过滤空白 token。"""
        return [t for t in jieba.cut(text) if t.strip()]

    def build(self, chunks: list[Chunk]) -> None:
        """从零构建倒排索引。"""
        conn = self._get_conn()
        # 清空旧数据
        conn.executescript("DELETE FROM postings; DELETE FROM chunks; DELETE FROM stats;")

        total_doc_length = 0
        for chunk in chunks:
            tokens = self._tokenize(chunk.content)
            doc_length = len(tokens)
            total_doc_length += doc_length

            cursor = conn.execute(
                "INSERT INTO chunks (content, file_path, title, chunk_index, doc_length) VALUES (?, ?, ?, ?, ?)",
                (chunk.content, chunk.file_path, chunk.title, chunk.chunk_index, doc_length),
            )
            chunk_id = cursor.lastrowid

            # 统计词频
            tf_map: dict[str, int] = {}
            for t in tokens:
                tf_map[t] = tf_map.get(t, 0) + 1

            conn.executemany(
                "INSERT OR REPLACE INTO postings (token, chunk_id, term_freq) VALUES (?, ?, ?)",
                [(token, chunk_id, freq) for token, freq in tf_map.items()],
            )

        num_docs = len(chunks)
        avg_dl = total_doc_length / num_docs if num_docs > 0 else 0
        conn.execute("INSERT INTO stats (key, value) VALUES ('num_docs', ?)", (num_docs,))
        conn.execute("INSERT INTO stats (key, value) VALUES ('avg_dl', ?)", (avg_dl,))
        conn.commit()

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """BM25 检索，返回 top_k 个结果。"""
        conn = self._get_conn()
        tokens = self._tokenize(query)
        if not tokens:
            return []

        # 读取统计信息
        row = conn.execute("SELECT value FROM stats WHERE key = 'num_docs'").fetchone()
        num_docs = row[0] if row else 0
        row = conn.execute("SELECT value FROM stats WHERE key = 'avg_dl'").fetchone()
        avg_dl = row[0] if row else 0

        if num_docs == 0:
            return []

        # 计算每个 chunk 的 BM25 得分
        scores: dict[int, float] = {}
        for token in set(tokens):
            # 包含该 token 的文档数
            df_row = conn.execute(
                "SELECT COUNT(DISTINCT chunk_id) FROM postings WHERE token = ?", (token,)
            ).fetchone()
            df = df_row[0] if df_row else 0
            if df == 0:
                continue

            # IDF
            idf = math.log((num_docs - df + 0.5) / (df + 0.5) + 1)

            # 包含该 token 的所有文档及其词频
            rows = conn.execute(
                "SELECT chunk_id, term_freq FROM postings WHERE token = ?", (token,)
            ).fetchall()

            for chunk_id, tf in rows:
                if chunk_id not in scores:
                    # 获取文档长度
                    dl_row = conn.execute(
                        "SELECT doc_length FROM chunks WHERE id = ?", (chunk_id,)
                    ).fetchone()
                    dl = dl_row[0] if dl_row else avg_dl
                    scores[chunk_id] = 0.0

                # BM25 公式
                tf_norm = (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * (dl / avg_dl)))
                scores[chunk_id] += idf * tf_norm

        # 排序取 top_k
        sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)[:top_k]

        results = []
        for chunk_id in sorted_ids:
            row = conn.execute(
                "SELECT content, file_path, title FROM chunks WHERE id = ?", (chunk_id,)
            ).fetchone()
            if row:
                results.append({
                    "content": row[0],
                    "file_path": row[1],
                    "title": row[2],
                    "score": scores[chunk_id],
                })
        return results

    @property
    def count(self) -> int:
        conn = self._get_conn()
        row = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        return row[0] if row else 0

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

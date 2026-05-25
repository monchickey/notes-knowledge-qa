from notes_qa.embedder import Embedder
from notes_qa.keyword_index import KeywordIndex
from notes_qa.vector_store import VectorStore


class Retriever:
    """混合检索器：融合向量检索和关键词检索结果（RRF）。"""

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        keyword_index: KeywordIndex,
        vector_top_k: int = 10,
        keyword_top_k: int = 10,
        final_top_k: int = 5,
        vector_weight: float = 0.6,
        rrf_k: int = 60,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.keyword_index = keyword_index
        self.vector_top_k = vector_top_k
        self.keyword_top_k = keyword_top_k
        self.final_top_k = final_top_k
        self.vector_weight = vector_weight
        self.keyword_weight = 1.0 - vector_weight
        self.rrf_k = rrf_k

    def _rrf_score(self, rank: int) -> float:
        """RRF 得分公式：1 / (k + rank)。"""
        return 1.0 / (self.rrf_k + rank)

    def retrieve(self, query: str) -> list[dict]:
        """混合检索，返回融合排序后的 top_k 结果。"""
        # 向量检索
        query_emb = self.embedder.encode_query(query)
        vector_results = self.vector_store.search(query_emb, top_k=self.vector_top_k)

        # 关键词检索
        keyword_results = self.keyword_index.search(query, top_k=self.keyword_top_k)

        # RRF 融合：用 file_path + chunk_index 作为唯一标识
        rrf_scores: dict[str, float] = {}
        result_map: dict[str, dict] = {}

        for rank, r in enumerate(vector_results):
            key = f"{r['file_path']}:{r.get('chunk_index', 0)}"
            rrf_scores[key] = rrf_scores.get(key, 0) + self.vector_weight * self._rrf_score(rank)
            result_map[key] = r

        for rank, r in enumerate(keyword_results):
            key = f"{r['file_path']}:{r.get('chunk_index', 0)}"
            rrf_scores[key] = rrf_scores.get(key, 0) + self.keyword_weight * self._rrf_score(rank)
            if key not in result_map:
                result_map[key] = r

        # 按 RRF 分数排序
        sorted_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)

        results = []
        for key in sorted_keys[: self.final_top_k]:
            item = result_map[key].copy()
            item["rrf_score"] = rrf_scores[key]
            results.append(item)
        return results

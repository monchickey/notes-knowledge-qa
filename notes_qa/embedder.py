import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    """基于 sentence-transformers 的本地向量嵌入服务。"""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5", batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model: SentenceTransformer | None = None

    def _load_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        """批量将文本编码为向量，返回 shape=(len(texts), dim) 的 numpy 数组。"""
        model = self._load_model()
        embeddings = model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.array(embeddings, dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """编码单条查询文本，返回 shape=(dim,) 的 numpy 数组。"""
        return self.encode([query])[0]

from pathlib import Path

import numpy as np

from notes_qa.embedder import Embedder

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "bge-small-zh-v1.5"


def test_encode_returns_correct_shape():
    embedder = Embedder(model_name=str(MODEL_DIR))
    texts = ["你好世界", "人工智能"]
    embeddings = embedder.encode(texts)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == 2
    assert embeddings.shape[1] > 0


def test_encode_query_returns_1d():
    embedder = Embedder(model_name=str(MODEL_DIR))
    vec = embedder.encode_query("测试查询")
    assert isinstance(vec, np.ndarray)
    assert vec.ndim == 1


def test_similar_texts_have_high_score():
    embedder = Embedder(model_name=str(MODEL_DIR))
    v1 = embedder.encode_query("机器学习")
    v2 = embedder.encode_query("深度学习")
    v3 = embedder.encode_query("今天天气不错")
    sim_related = float(np.dot(v1, v2))
    sim_unrelated = float(np.dot(v1, v3))
    assert sim_related > sim_unrelated

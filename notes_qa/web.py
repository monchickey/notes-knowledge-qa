import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from notes_qa.config import load_config

_STATIC_DIR = Path(__file__).parent / "static"

# 全局组件，在 lifespan 中初始化
_qa_engine = None
_keyword_index = None
_config = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _qa_engine, _keyword_index, _config
    from notes_qa.cli import _build_qa_components

    _config = load_config(app.state.config_path)
    _, vector_store, keyword_index, _, qa_engine = _build_qa_components(_config)
    if not vector_store.load():
        raise RuntimeError("向量索引不存在，请先运行 `qa index` 构建索引。")
    _qa_engine = qa_engine
    _keyword_index = keyword_index
    yield
    if _keyword_index:
        _keyword_index.close()


def create_app(config_path: str | None = None) -> FastAPI:
    """创建 FastAPI 应用实例。"""
    app = FastAPI(title="笔记知识问答", lifespan=lifespan)
    app.state.config_path = config_path

    @app.get("/", response_class=HTMLResponse)
    async def index():
        html_path = _STATIC_DIR / "index.html"
        return html_path.read_text(encoding="utf-8")

    class QueryRequest(BaseModel):
        question: str

    @app.post("/api/query")
    async def query(req: QueryRequest):
        question = req.question.strip()
        if not question:
            return StreamingResponse(
                iter(['event: error\ndata: {"message": "问题不能为空"}\n\n']),
                media_type="text/event-stream",
            )

        def event_stream():
            # 检索参考来源
            sources = _qa_engine.get_sources(question)
            sources_data = [
                {"file_path": s["file_path"], "content": s["content"][:200]}
                for s in sources
            ]
            yield f"event: sources\ndata: {json.dumps(sources_data, ensure_ascii=False)}\n\n"

            # 流式输出 LLM 回答
            for token in _qa_engine.query(question, stream=True):
                yield f"event: token\ndata: {json.dumps({'text': token}, ensure_ascii=False)}\n\n"

            yield "event: done\ndata: {}\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    class SearchRequest(BaseModel):
        question: str
        top_k: int = 5

    @app.post("/api/search")
    async def search(req: SearchRequest):
        from notes_qa.retriever import Retriever

        question = req.question.strip()
        if not question:
            return {"error": "问题不能为空"}

        embedder = _qa_engine.retriever.embedder
        vector_store = _qa_engine.retriever.vector_store
        retriever = Retriever(
            embedder=embedder,
            vector_store=vector_store,
            keyword_index=_keyword_index,
            vector_top_k=_config["retriever"]["vector_top_k"],
            keyword_top_k=_config["retriever"]["keyword_top_k"],
            final_top_k=req.top_k,
            vector_weight=_config["retriever"]["vector_weight"],
        )
        results = retriever.retrieve(question)
        return {
            "results": [
                {
                    "file_path": r["file_path"],
                    "title": r.get("title", ""),
                    "content": r["content"],
                    "score": r.get("rrf_score", 0),
                }
                for r in results
            ]
        }

    return app

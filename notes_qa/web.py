import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from notes_qa.config import load_config
from notes_qa.history import ChatHistory

_STATIC_DIR = Path(__file__).parent / "static"

# 全局组件，在 lifespan 中初始化
_qa_engine = None
_keyword_index = None
_config = None
_chat_history = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _qa_engine, _keyword_index, _config, _chat_history
    from notes_qa.cli import _build_qa_components

    _config = load_config(app.state.config_path)
    _, vector_store, keyword_index, _, qa_engine = _build_qa_components(_config)
    if not vector_store.load():
        raise RuntimeError("向量索引不存在，请先运行 `qa index` 构建索引。")
    _qa_engine = qa_engine
    _keyword_index = keyword_index
    _chat_history = ChatHistory()
    yield
    if _keyword_index:
        _keyword_index.close()
    if _chat_history:
        _chat_history.close()


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
        conversation_id: int | None = None

    @app.post("/api/query")
    async def query(req: QueryRequest):
        question = req.question.strip()
        if not question:
            return StreamingResponse(
                iter(['event: error\ndata: {"message": "问题不能为空"}\n\n']),
                media_type="text/event-stream",
            )

        # 如果没有指定对话 ID，创建新对话
        conversation_id = req.conversation_id
        if conversation_id is None:
            # 使用问题的前 20 个字符作为标题
            title = question[:20] + ("..." if len(question) > 20 else "")
            conversation_id = _chat_history.create_conversation(title)

        # 保存用户消息
        _chat_history.add_message(conversation_id, "user", question, mode="chat")

        def event_stream():
            # 检索参考来源
            sources = _qa_engine.get_sources(question)
            sources_data = [
                {"file_path": s["file_path"], "content": s["content"][:200]}
                for s in sources
            ]
            yield f"event: sources\ndata: {json.dumps(sources_data, ensure_ascii=False)}\n\n"

            # 流式输出 LLM 回答
            full_answer = ""
            for token in _qa_engine.query(question, stream=True):
                full_answer += token
                yield f"event: token\ndata: {json.dumps({'text': token}, ensure_ascii=False)}\n\n"

            # 保存助手回答
            _chat_history.add_message(conversation_id, "assistant", full_answer,
                                      sources=sources_data, mode="chat")

            # 返回对话 ID 和完成事件
            yield f"event: done\ndata: {json.dumps({'conversation_id': conversation_id}, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    class SearchRequest(BaseModel):
        question: str
        top_k: int = 5
        conversation_id: int | None = None

    @app.post("/api/search")
    async def search(req: SearchRequest):
        from notes_qa.retriever import Retriever

        question = req.question.strip()
        if not question:
            return {"error": "问题不能为空"}

        # 如果没有指定对话 ID，创建新对话
        conversation_id = req.conversation_id
        if conversation_id is None:
            title = f"检索: {question[:15]}" + ("..." if len(question) > 15 else "")
            conversation_id = _chat_history.create_conversation(title)

        # 保存用户检索请求
        _chat_history.add_message(conversation_id, "user", question, mode="search")

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
        results_data = [
            {
                "file_path": r["file_path"],
                "title": r.get("title", ""),
                "content": r["content"],
                "score": r.get("rrf_score", 0),
            }
            for r in results
        ]

        # 保存检索结果
        result_summary = f"找到 {len(results_data)} 条相关结果"
        _chat_history.add_message(conversation_id, "assistant", result_summary,
                                  sources=results_data, mode="search")

        return {
            "conversation_id": conversation_id,
            "results": results_data
        }

    # 历史记录 API 端点
    @app.get("/api/conversations")
    async def get_conversations(limit: int = 50, offset: int = 0):
        """获取对话列表"""
        conversations = _chat_history.get_conversations(limit=limit, offset=offset)
        return {"conversations": conversations}

    @app.get("/api/conversations/{conversation_id}")
    async def get_conversation(conversation_id: int):
        """获取单个对话及其消息"""
        conversation = _chat_history.get_conversation(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="对话不存在")

        messages = _chat_history.get_conversation_messages(conversation_id)
        return {
            "conversation": conversation,
            "messages": messages
        }

    @app.delete("/api/conversations/{conversation_id}")
    async def delete_conversation(conversation_id: int):
        """删除对话"""
        success = _chat_history.delete_conversation(conversation_id)
        if not success:
            raise HTTPException(status_code=404, detail="对话不存在")
        return {"message": "删除成功"}

    class UpdateTitleRequest(BaseModel):
        title: str

    @app.put("/api/conversations/{conversation_id}/title")
    async def update_conversation_title(conversation_id: int, req: UpdateTitleRequest):
        """更新对话标题"""
        success = _chat_history.update_conversation_title(conversation_id, req.title)
        if not success:
            raise HTTPException(status_code=404, detail="对话不存在")
        return {"message": "更新成功"}

    @app.get("/api/search-messages")
    async def search_messages(keyword: str, limit: int = 20):
        """搜索消息内容"""
        messages = _chat_history.search_messages(keyword, limit=limit)
        return {"messages": messages}

    return app

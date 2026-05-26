from collections.abc import Iterator

from openai import OpenAI

from notes_qa.retriever import Retriever


_SYSTEM_PROMPT = """你是一个笔记知识问答助手。根据用户的问题和提供的笔记内容片段，给出准确、有帮助的回答。

要求：
1. 基于提供的笔记内容回答，如果笔记中没有相关信息，请如实说明
2. 回答要简洁清晰
3. 在回答末尾标注参考来源（文件路径）

如果被问到“你是谁？”、“你是什么模型”之类的问题，请回答 “我是个人笔记问答助手。”
"""


def _build_context(results: list[dict]) -> str:
    """将检索结果拼接为上下文文本。"""
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(
            f"---\n来源 {i}: {r['file_path']}\n{r['content']}\n"
        )
    return "\n".join(parts)


class QAEngine:
    """问答引擎：检索 + LLM 生成。"""

    def __init__(self, retriever: Retriever, llm_config: dict):
        self.retriever = retriever
        self.llm_config = llm_config
        self.client = OpenAI(
            api_key=llm_config.get("api_key", ""),
            base_url=llm_config.get("api_base", "https://api.deepseek.com/v1"),
        )

    def query(self, question: str, stream: bool = False) -> str | Iterator[str]:
        """根据问题检索并生成回答。"""
        results = self.retriever.retrieve(question)
        context = _build_context(results)

        user_message = f"笔记内容：\n{context}\n\n问题：{question}"

        kwargs = {
            "model": self.llm_config.get("model", "deepseek-chat"),
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            "temperature": self.llm_config.get("temperature", 0.3),
            "max_completion_tokens": self.llm_config.get("max_tokens", 2048),
            "stream": stream,
        }

        response = self.client.chat.completions.create(**kwargs)

        if stream:
            return self._stream_response(response)

        answer = response.choices[0].message.content
        return answer

    def _stream_response(self, response):
        """流式输出回答。"""
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def get_sources(self, question: str) -> list[dict]:
        """仅获取检索结果，不调用 LLM。"""
        return self.retriever.retrieve(question)

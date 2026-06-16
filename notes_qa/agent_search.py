"""纯 Agent 检索引擎，不依赖索引，通过工具调用让 LLM 自主检索笔记"""

import json
from pathlib import Path
from openai import OpenAI

# System Prompt
SYSTEM_PROMPT = """你是一个笔记知识检索助手。用户会问你问题，你需要从笔记文件中找到答案。

工作流程：
1. 首先使用 list_files 了解笔记目录结构
2. 根据问题判断可能相关的文件（基于文件名、目录结构）
3. 使用 read_file 读取相关文件内容
4. 如果需要，使用 search_keyword 搜索特定关键词
5. 基于找到的内容回答问题，使用 finish 工具返回答案

注意事项：
- 优先根据文件名和目录结构判断相关性，避免盲目读取
- 每次只读取最可能相关的文件，不要一次读取太多
- 如果第一次没找到，尝试其他文件或关键词
- 回答时必须标注参考来源（文件路径）
- 当读取文件数达到限制时，基于已有内容回答"""

# 工具定义
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "列出笔记目录下的文件和子目录",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "要列出的子目录路径，留空表示根目录"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取单个 Markdown 文件的内容",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "要读取的文件路径（相对于笔记目录）"
                    }
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_keyword",
            "description": "在笔记文件中搜索关键词，返回匹配的行和上下文",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "要搜索的关键词"
                    },
                    "file_path": {
                        "type": "string",
                        "description": "指定在某个文件中搜索，留空表示搜索所有文件"
                    }
                },
                "required": ["keyword"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "检索完成，返回最终答案",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "基于笔记内容的回答"
                    },
                    "sources": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "参考来源的文件路径列表"
                    }
                },
                "required": ["answer", "sources"]
            }
        }
    }
]


class AgentSearchEngine:
    """纯 Agent 检索引擎，不依赖索引，通过工具调用让 LLM 自主检索。

    支持两种 API 模式（通过 llm.api_type 配置）：
    - chat_completions（默认）：使用 Chat Completions API
    - responses：使用 Responses API
    """

    def __init__(self, notes_dir: str, llm_config: dict,
                 max_rounds: int = 10, max_files: int = 20, verbose: bool = True):
        self.notes_dir = Path(notes_dir).resolve()
        self.max_rounds = max_rounds
        self.max_files = max_files
        self.verbose = verbose
        self.files_read = []  # 已读取的文件列表

        # 初始化 LLM 客户端
        api_key = llm_config.get("api_key") or "sk-placeholder"
        api_base = llm_config.get("api_base", "https://api.deepseek.com/v1")
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.model = llm_config.get("model", "deepseek-chat")
        self.temperature = llm_config.get("temperature", 0.3)
        self.api_type = llm_config.get("api_type", "chat_completions")

    def _log(self, message: str, level: str = "info"):
        """输出日志"""
        if self.verbose:
            icons = {"info": "ℹ️", "tool": "🔧", "result": "📥", "answer": "✅", "warning": "⚠️", "error": "❌"}
            icon = icons.get(level, "ℹ️")
            print(f"{icon} {message}")

    def _list_files(self, path: str = "") -> str:
        """列出目录下的文件"""
        target_dir = self.notes_dir / path if path else self.notes_dir

        if not target_dir.exists():
            return f"错误：目录 {path or '.'} 不存在"

        if not target_dir.is_dir():
            return f"错误：{path or '.'} 不是目录"

        items = []
        for item in sorted(target_dir.iterdir()):
            if item.name.startswith('.'):
                continue
            rel_path = item.relative_to(self.notes_dir)
            if item.is_dir():
                items.append(f"📁 {rel_path}/")
            elif item.suffix == '.md':
                items.append(f"📄 {rel_path}")

        if not items:
            return "目录为空"

        return "\n".join(items)

    def _read_file(self, file_path: str) -> str:
        """读取文件内容"""
        # 检查文件数限制
        if len(self.files_read) >= self.max_files:
            return f"警告：已达到最大文件读取限制（{self.max_files}个），无法继续读取。已读取：{', '.join(self.files_read)}"

        target_file = self.notes_dir / file_path

        if not target_file.exists():
            return f"错误：文件 {file_path} 不存在"

        if not target_file.is_file():
            return f"错误：{file_path} 不是文件"

        try:
            content = target_file.read_text(encoding='utf-8')

            # 去除 YAML front matter
            if content.startswith('---'):
                parts = content.split('---', 2)
                if len(parts) >= 3:
                    content = parts[2].strip()

            # 记录已读取
            if file_path not in self.files_read:
                self.files_read.append(file_path)

            return content if content else "文件内容为空"
        except Exception as e:
            return f"读取文件失败：{e}"

    def _search_keyword(self, keyword: str, file_path: str = "") -> str:
        """搜索关键词"""
        results = []
        files_to_search = []

        if file_path:
            # 在指定文件中搜索
            target = self.notes_dir / file_path
            if target.exists() and target.suffix == '.md':
                files_to_search = [target]
            else:
                return f"错误：文件 {file_path} 不存在或不是 Markdown 文件"
        else:
            # 搜索所有文件
            files_to_search = list(self.notes_dir.rglob("*.md"))

        for f in files_to_search[:50]:  # 限制最多搜索 50 个文件
            try:
                content = f.read_text(encoding='utf-8')
                lines = content.split('\n')

                for i, line in enumerate(lines):
                    if keyword.lower() in line.lower():
                        # 获取上下文（前后各 1 行）
                        start = max(0, i - 1)
                        end = min(len(lines), i + 2)
                        context = "\n".join(lines[start:end])

                        rel_path = f.relative_to(self.notes_dir)
                        results.append(f"--- {rel_path} (行 {i+1}) ---\n{context}")

                        if len(results) >= 10:  # 最多返回 10 个匹配
                            break
            except Exception:
                continue

            if len(results) >= 10:
                break

        if not results:
            return f"未找到包含 '{keyword}' 的内容"

        return "\n\n".join(results)

    def _execute_tool(self, tool_name: str, arguments: dict) -> str:
        """执行工具调用"""
        if tool_name == "list_files":
            return self._list_files(arguments.get("path", ""))
        elif tool_name == "read_file":
            return self._read_file(arguments["file_path"])
        elif tool_name == "search_keyword":
            return self._search_keyword(
                arguments["keyword"],
                arguments.get("file_path", "")
            )
        elif tool_name == "finish":
            return "DONE"  # 特殊标记，表示完成
        else:
            return f"错误：未知工具 {tool_name}"

    # ---- Responses API 支持 ----

    @staticmethod
    def _convert_tools_for_responses(tools: list) -> list:
        """将 Chat Completions 格式的 tools 转换为 Responses API 格式。"""
        return [
            {
                "type": "function",
                "name": t["function"]["name"],
                "description": t["function"]["description"],
                "parameters": t["function"]["parameters"],
            }
            for t in tools
        ]

    @staticmethod
    def _messages_to_responses_input(messages: list) -> list:
        """将 Chat Completions 格式的 messages 转换为 Responses API 的 input 格式。"""
        input_items = []
        for msg in messages:
            role = msg.get("role")
            if role == "system":
                input_items.append({"role": "developer", "content": msg["content"]})
            elif role == "user":
                input_items.append({"role": "user", "content": msg["content"]})
            elif role == "assistant":
                tool_calls = msg.get("tool_calls")
                if tool_calls:
                    for tc in tool_calls:
                        func = tc["function"]
                        input_items.append({
                            "type": "function_call",
                            "call_id": tc["id"],
                            "name": func["name"],
                            "arguments": func["arguments"],
                        })
                else:
                    input_items.append({"role": "assistant", "content": msg.get("content", "") or ""})
            elif role == "tool":
                input_items.append({
                    "type": "function_call_output",
                    "call_id": msg["tool_call_id"],
                    "output": msg["content"],
                })
        return input_items

    def _call_llm(self, messages: list) -> tuple:
        """调用 LLM，返回标准化的 (tool_calls, content)。

        tool_calls: list of dict {id, name, arguments}，无工具调用时为 None
        content: str，无内容时为 ""
        """
        if self.api_type == "responses":
            return self._call_llm_responses(messages)
        return self._call_llm_completions(messages)

    def _call_llm_completions(self, messages: list) -> tuple:
        """使用 Chat Completions API 调用 LLM。"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=TOOLS,
            temperature=self.temperature,
        )
        message = response.choices[0].message
        if not message.tool_calls:
            return None, message.content or ""
        tool_calls = [
            {"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments}
            for tc in message.tool_calls
        ]
        return tool_calls, message.content or ""

    def _call_llm_responses(self, messages: list) -> tuple:
        """使用 Responses API 调用 LLM。"""
        resp_tools = self._convert_tools_for_responses(TOOLS)
        input_items = self._messages_to_responses_input(messages)
        response = self.client.responses.create(
            model=self.model,
            input=input_items,
            tools=resp_tools,
            temperature=self.temperature,
        )
        tool_calls = []
        content = None
        for item in response.output:
            if item.type == "function_call":
                tool_calls.append({
                    "id": item.call_id,
                    "name": item.name,
                    "arguments": item.arguments,
                })
            elif item.type == "message":
                content = item.content[0].text if item.content else ""
        if not tool_calls:
            return None, content or ""
        return tool_calls, content or ""

    @staticmethod
    def _append_tool_messages(messages: list, tool_calls: list, results: list[str]):
        """将工具调用和执行结果追加到消息列表（Completions 格式）。"""
        messages.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": tc["arguments"]},
                }
                for tc in tool_calls
            ],
        })
        for tc, result in zip(tool_calls, results):
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})

    def search(self, question: str) -> dict:
        """
        执行 Agent 检索

        返回：
            {
                "answer": "回答内容",
                "sources": ["来源文件路径"],
                "rounds": 实际轮数,
                "files_read": ["已读取文件列表"]
            }
        """
        self.files_read = []  # 重置已读取文件列表

        self._log(f"问题：{question}", "info")
        self._log(f"开始 Agent 检索（最多 {self.max_rounds} 轮，最多读取 {self.max_files} 个文件）", "info")
        print("-" * 60)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ]

        for round_num in range(1, self.max_rounds + 1):
            self._log(f"第 {round_num} 轮", "info")

            # 调用 LLM
            try:
                tool_calls, content = self._call_llm(messages)
            except Exception as e:
                self._log(f"LLM 调用失败：{e}", "error")
                return {
                    "answer": f"LLM 调用失败：{e}",
                    "sources": [],
                    "rounds": round_num,
                    "files_read": self.files_read
                }

            # 如果没有工具调用，说明 LLM 直接返回了回答
            if not tool_calls:
                self._log("LLM 直接返回了回答（未使用工具）", "answer")
                print("-" * 60)
                return {
                    "answer": content or "无法生成回答",
                    "sources": [],
                    "rounds": round_num,
                    "files_read": self.files_read
                }

            # 处理工具调用
            executed_results = []
            finish_result = None

            for tc in tool_calls:
                tool_name = tc["name"]
                try:
                    arguments = json.loads(tc["arguments"])
                except json.JSONDecodeError:
                    arguments = {}

                self._log(f"调用工具：{tool_name}", "tool")
                self._log(f"参数：{json.dumps(arguments, ensure_ascii=False)}", "tool")

                # 检查是否是 finish 工具
                if tool_name == "finish":
                    answer = arguments.get("answer", "无法生成回答")
                    sources = arguments.get("sources", [])
                    self._log(f"检索完成，共 {round_num} 轮，读取 {len(self.files_read)} 个文件", "answer")
                    print("-" * 60)
                    finish_result = {
                        "answer": answer,
                        "sources": sources,
                        "rounds": round_num,
                        "files_read": self.files_read
                    }
                    break

                # 执行工具
                result = self._execute_tool(tool_name, arguments)

                # 截断过长的结果
                if len(result) > 3000:
                    result = result[:3000] + "\n... (内容已截断)"

                self._log(f"结果：{result[:200]}{'...' if len(result) > 200 else ''}", "result")
                executed_results.append(result)

            if finish_result is not None:
                return finish_result

            # 将工具调用和结果追加到消息
            self._append_tool_messages(messages, tool_calls, executed_results)

        # 达到最大轮数
        self._log(f"达到最大轮数（{self.max_rounds}），强制结束", "warning")
        print("-" * 60)

        # 尝试让 LLM 基于已有内容生成回答
        messages.append({
            "role": "user",
            "content": "请基于你已经读取的笔记内容，使用 finish 工具返回答案。"
        })

        try:
            tool_calls, content = self._call_llm(messages)

            if tool_calls:
                for tc in tool_calls:
                    if tc["name"] == "finish":
                        try:
                            arguments = json.loads(tc["arguments"])
                        except json.JSONDecodeError:
                            arguments = {}
                        return {
                            "answer": arguments.get("answer", "无法生成回答"),
                            "sources": arguments.get("sources", []),
                            "rounds": self.max_rounds,
                            "files_read": self.files_read
                        }

            return {
                "answer": content or "达到最大轮数，无法生成回答",
                "sources": [],
                "rounds": self.max_rounds,
                "files_read": self.files_read
            }
        except Exception as e:
            return {
                "answer": f"达到最大轮数，且最终 LLM 调用失败：{e}",
                "sources": [],
                "rounds": self.max_rounds,
                "files_read": self.files_read
            }

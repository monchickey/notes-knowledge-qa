# 基于 AI 的笔记知识检索问答系统

一个轻量级的个人笔记 RAG 系统，利用大模型对本地 Markdown 笔记进行检索问答。

## 特点

- **轻量部署**：无需 Qdrant、Milvus 等向量数据库，也无需 Elasticsearch，单机即可运行
- **混合检索**：向量语义检索（numpy）+ 全文关键词检索（jieba + SQLite），RRF 融合排序
- **中文友好**：jieba 分词 + BGE 中文 embedding 模型，原生支持中文笔记
- **本地嵌入**：向量嵌入使用本地 BGE 模型（sentence-transformers），CPU 即可运行，无需 API 调用
- **兼容性强**：问答生成支持所有 OpenAI 兼容格式的 API（DeepSeek、通义千问等）
- **Web 界面**：内置 FastAPI Web 服务，支持浏览器端对话问答和检索，流式输出
- **历史记录**：Web 端对话历史自动保存到 SQLite，支持查看、搜索、删除历史记录

## 技术栈

- Python 3.10+
- sentence-transformers + `BAAI/bge-small-zh-v1.5` - 本地向量嵌入
- numpy - 向量运算与相似度检索
- jieba - 中文分词
- SQLite - 倒排索引存储
- OpenAI 兼容 API - 问答生成（DeepSeek、通义千问等）
- FastAPI + Uvicorn - Web 服务

## 快速开始

```bash
# 创建虚拟环境并安装依赖
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 编辑配置
cp config.yaml config.local.yaml
# 填入你的 API key、API 地址、笔记目录路径

# 构建索引
qa index

# 单次问答
qa query "什么是 RAG？"

# 交互式对话
qa chat

# 启动 Web 服务
qa web
```

## CLI 命令

| 命令 | 说明 |
|------|------|
| `qa index` | 扫描笔记目录，构建向量索引和全文索引 |
| `qa query "问题"` | 单次问答 |
| `qa chat` | 交互式对话模式 |
| `qa search "关键词"` | 仅检索，不调用大模型，返回原始结果 |
| `qa agent "问题"` | 纯 Agent 检索，无需预构建索引 |
| `qa web` | 启动 Web 服务（默认 `127.0.0.1:8000`） |
| `qa config-show` | 查看当前配置 |

## 项目结构

```
notes-knowledge-qa/
├── pyproject.toml          # 项目配置与依赖
├── config.yaml             # 运行时配置模板
├── design.md               # 详细设计文档
├── notes_qa/
│   ├── config.py           # 配置加载
│   ├── loader.py           # 笔记文件加载
│   ├── chunker.py          # 文本分块
│   ├── embedder.py         # 向量嵌入
│   ├── vector_store.py     # 向量存储与检索
│   ├── keyword_index.py    # 全文检索
│   ├── retriever.py        # 混合检索器
│   ├── qa.py               # 问答生成
│   ├── agent_search.py     # 纯 Agent 检索引擎
│   ├── history.py          # 对话历史管理 (SQLite)
│   ├── web.py              # Web 服务 (FastAPI)
│   ├── cli.py              # CLI 入口
│   └── static/
│       └── index.html      # Web 前端页面
└── tests/
```

## 开发进度

- [x] Phase 1: 项目基础结构（配置、依赖）
- [x] Phase 2: 笔记加载与分块
- [x] Phase 3: 向量嵌入与存储
- [x] Phase 4: 全文检索
- [x] Phase 5: 混合检索与问答生成
- [x] Phase 6: CLI 交互
- [x] Phase 7: 测试与优化
- [x] Phase 8: Web 界面
- [x] Phase 9: 对话历史记录
- [x] Phase 10: 纯 Agent 检索模式

## License

MIT

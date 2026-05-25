# 设计文档 - 个人笔记知识检索问答系统

## 概述

构建一个轻量级的个人笔记 RAG 系统，面向小规模本地 Markdown 笔记，采用向量检索 + 全文检索混合方案，单机即可部署运行。

## 技术选型

| 类别 | 选择 | 说明 |
|------|------|------|
| 语言 | Python 3.10+ | RAG 生态成熟 |
| 笔记格式 | 本地 Markdown | 递归扫描目录 |
| LLM | 兼容 OpenAI 格式的 API | 支持 DeepSeek / 通义千问等 |
| 向量嵌入 | sentence-transformers + BGE 本地模型 | 本地推理，无需 API，CPU 即可运行 |
| 向量检索 | numpy | 轻量，零额外依赖 |
| 全文检索 | jieba + SQLite 倒排索引 | BM25 排序 |
| 混合检索 | RRF (Reciprocal Rank Fusion) | 融合向量与关键词结果 |
| CLI 框架 | click + rich | 命令行交互与美化 |

## 项目结构

```
notes-knowledge-qa/
├── pyproject.toml          # 项目配置与依赖
├── config.yaml             # 运行时配置（API key、笔记路径等）
├── design.md               # 设计文档
├── README.md               # 项目说明
├── notes_qa/
│   ├── __init__.py
│   ├── config.py           # 配置加载
│   ├── loader.py           # 笔记文件加载
│   ├── chunker.py          # 文本分块
│   ├── embedder.py         # 向量嵌入
│   ├── vector_store.py     # 向量存储与检索
│   ├── keyword_index.py    # 全文检索（倒排索引）
│   ├── retriever.py        # 混合检索器
│   ├── qa.py               # 问答生成
│   └── cli.py              # CLI 入口
└── tests/
```

## 核心流程

```
Markdown 文件 -> 加载 -> 分块 -> 向量嵌入 + 全文索引 -> 持久化存储
                                                      ↓
用户提问 -> 混合检索（向量 + 关键词） -> RRF 融合 -> LLM 生成回答
```

---

## 开发计划

### Phase 1: 项目基础结构

搭建项目骨架、依赖管理、配置系统。

- 初始化 `pyproject.toml`，声明依赖：`openai`, `numpy`, `jieba`, `pyyaml`, `rich`, `click`, `sentence-transformers`
- 创建 `config.yaml` 模板，包含 API 地址、密钥、笔记目录、分块参数等
- 实现 `config.py`：加载 YAML 配置，提供全局配置访问

**产出：** `pyproject.toml`, `config.yaml`, `notes_qa/config.py`

### Phase 2: 笔记加载与分块

读取 Markdown 文件，按语义分块。

- **`loader.py`**：递归扫描 `.md` 文件，去除 YAML front matter，输出 `Document`（content + metadata: file_path, title）
- **`chunker.py`**：递归分块策略（heading > 段落 > 句子），chunk_size=500，overlap=50，输出 `Chunk` 数据类

**产出：** `notes_qa/loader.py`, `notes_qa/chunker.py`

### Phase 3: 向量嵌入与存储

将文本块转为向量并持久化。

- **`embedder.py`**：使用 sentence-transformers 加载本地 BGE 模型（默认 `BAAI/bge-small-zh-v1.5`），支持批量编码，CPU 即可运行
- **`vector_store.py`**：numpy 实现余弦相似度检索，`.npy` + JSON 持久化，支持增量添加和全量重建

**产出：** `notes_qa/embedder.py`, `notes_qa/vector_store.py`

### Phase 4: 全文检索

基于关键词的中文全文检索。

- **`keyword_index.py`**：jieba 分词 + SQLite 倒排索引，BM25 排序算法，支持增量更新

**产出：** `notes_qa/keyword_index.py`

### Phase 5: 混合检索与问答生成

融合检索结果，调用 LLM 生成回答。

- **`retriever.py`**：同时调用向量/全文检索，RRF 融合排序，可配置权重
- **`qa.py`**：构造 prompt（检索上下文 + 用户问题），调用 chat API，标注来源，支持流式输出

**产出：** `notes_qa/retriever.py`, `notes_qa/qa.py`

### Phase 6: CLI 交互

命令行入口，完成端到端体验。

- `qa index`：扫描笔记目录，构建索引
- `qa query "问题"`：单次问答
- `qa chat`：交互式对话模式
- `qa config show`：查看当前配置

**产出：** `notes_qa/cli.py`

### Phase 7: 测试与优化

- 单元测试：loader、chunker、vector_store、keyword_index
- 端到端测试：用示例笔记验证完整流程
- 优化：索引构建进度条、检索缓存、错误处理与友好提示

**产出：** `tests/` 目录

---

## 验证方式

1. 准备一个包含几篇 Markdown 笔记的测试目录
2. 运行 `qa index` 构建索引，确认无报错
3. 运行 `qa query "某个问题"` 验证检索结果和回答质量
4. 运行 `qa chat` 进入交互模式，测试多轮对话
5. 运行单元测试确认各模块基本逻辑正确

## 后续扩展

- Web 界面（FastAPI + 前端）
- 增量索引更新（监听文件变化）
- 支持更多文件格式（PDF、docx 等）
- 多笔记库管理

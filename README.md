# 基于 AI 的笔记知识检索问答系统

一个轻量级的个人笔记 RAG 系统，利用大模型对本地 Markdown 笔记进行检索问答。

## 特点

- **轻量部署**：无需 Qdrant、Milvus 等向量数据库，也无需 Elasticsearch，单机即可运行
- **混合检索**：向量语义检索（numpy）+ 全文关键词检索（jieba + SQLite），RRF 融合排序
- **中文友好**：jieba 分词 + BGE 中文 embedding 模型，原生支持中文笔记
- **本地嵌入**：向量嵌入使用本地 BGE 模型（sentence-transformers），CPU 即可运行，无需 API 调用
- **兼容性强**：问答生成支持所有 OpenAI 兼容格式的 API（DeepSeek、通义千问等）

## 技术栈

- Python 3.10+
- sentence-transformers + `BAAI/bge-small-zh-v1.5` - 本地向量嵌入
- numpy - 向量运算与相似度检索
- jieba - 中文分词
- SQLite - 倒排索引存储
- OpenAI 兼容 API - 问答生成（DeepSeek、通义千问等）

## 快速开始

```bash
# 安装依赖
pip install -e .

# 编辑配置
cp config.yaml config.local.yaml
# 填入你的 API key、API 地址、笔记目录路径

# 构建索引
qa index

# 单次问答
qa query "什么是 RAG？"

# 交互式对话
qa chat
```

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
│   └── cli.py              # CLI 入口
└── tests/
```

## 开发进度

- [ ] Phase 1: 项目基础结构（配置、依赖）
- [ ] Phase 2: 笔记加载与分块
- [ ] Phase 3: 向量嵌入与存储
- [ ] Phase 4: 全文检索
- [ ] Phase 5: 混合检索与问答生成
- [ ] Phase 6: CLI 交互
- [ ] Phase 7: 测试与优化

## License

MIT

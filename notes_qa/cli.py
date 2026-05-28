import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from notes_qa.config import load_config
from notes_qa.loader import load_documents

console = Console()


def _build_index_components(cfg: dict):
    """构建索引相关组件（不涉及 LLM）。"""
    from notes_qa.embedder import Embedder
    from notes_qa.keyword_index import KeywordIndex
    from notes_qa.vector_store import VectorStore

    embedder = Embedder(
        model_name=cfg["embedding"]["model"],
        batch_size=cfg["embedding"]["batch_size"],
    )
    vector_store = VectorStore(cfg["index_dir"])
    keyword_index = KeywordIndex(
        cfg["index_dir"],
        k1=cfg["keyword_index"]["k1"],
        b=cfg["keyword_index"]["b"],
    )
    return embedder, vector_store, keyword_index


def _build_qa_components(cfg: dict):
    """构建完整问答组件（含 LLM）。"""
    from notes_qa.qa import QAEngine
    from notes_qa.retriever import Retriever

    embedder, vector_store, keyword_index = _build_index_components(cfg)
    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
        keyword_index=keyword_index,
        vector_top_k=cfg["retriever"]["vector_top_k"],
        keyword_top_k=cfg["retriever"]["keyword_top_k"],
        final_top_k=cfg["retriever"]["final_top_k"],
        vector_weight=cfg["retriever"]["vector_weight"],
    )
    qa_engine = QAEngine(retriever=retriever, llm_config=cfg["llm"])
    return embedder, vector_store, keyword_index, retriever, qa_engine


@click.group()
@click.option("--config", "-c", "config_path", default=None, help="配置文件路径")
@click.pass_context
def main(ctx, config_path):
    """笔记知识检索问答系统"""
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config(config_path)


@main.command()
@click.pass_context
def index(ctx):
    """扫描笔记目录，构建向量索引和全文索引。"""
    cfg = ctx.obj["config"]
    notes_dir = cfg["notes_dir"]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # 加载笔记
        task = progress.add_task("加载笔记文件...", total=None)
        documents = load_documents(notes_dir)
        progress.update(task, description=f"已加载 {len(documents)} 篇笔记")
        progress.stop_task(task)

        if not documents:
            console.print("[yellow]未找到任何 Markdown 文件，请检查 notes_dir 配置。[/yellow]")
            return

        # 分块
        from notes_qa.chunker import chunk_documents
        task = progress.add_task("文本分块...", total=None)
        chunks = chunk_documents(
            documents,
            chunk_size=cfg["chunking"]["chunk_size"],
            chunk_overlap=cfg["chunking"]["chunk_overlap"],
        )
        progress.update(task, description=f"已生成 {len(chunks)} 个文本块")
        progress.stop_task(task)

        # 构建向量索引
        task = progress.add_task("构建向量索引（首次加载模型可能较慢）...", total=None)
        embedder, vector_store, keyword_index = _build_index_components(cfg)
        embeddings = embedder.encode([c.content for c in chunks])
        vector_store.clear()
        vector_store.add(chunks, embeddings)
        vector_store.save()
        progress.update(task, description=f"向量索引完成：{vector_store.count} 条")
        progress.stop_task(task)

        # 构建全文索引
        task = progress.add_task("构建全文索引...", total=None)
        keyword_index.build(chunks)
        progress.update(task, description=f"全文索引完成：{keyword_index.count} 条")
        progress.stop_task(task)
        keyword_index.close()

    console.print(f"\n[green]索引构建完成！[/green]")
    console.print(f"  笔记数: {len(documents)}  |  文本块: {len(chunks)}  |  存储目录: {cfg['index_dir']}")


@main.command()
@click.argument("question")
@click.pass_context
def query(ctx, question):
    """单次问答。"""
    cfg = ctx.obj["config"]
    _, vector_store, keyword_index, _, qa_engine = _build_qa_components(cfg)

    # 加载已有索引
    if not vector_store.load():
        console.print("[red]向量索引不存在，请先运行 `qa index` 构建索引。[/red]")
        return

    console.print(f"\n[bold]问题：[/bold]{question}\n")

    # 流式输出
    answer_parts = []
    with console.status("正在思考..."):
        sources = qa_engine.get_sources(question)

    console.print("[bold]回答：[/bold]")
    for chunk_text in qa_engine.query(question, stream=True):
        console.print(chunk_text, end="")
        answer_parts.append(chunk_text)
    console.print()

    # 显示参考来源
    if sources:
        console.print("\n[bold]参考来源：[/bold]")
        table = Table(show_header=True, header_style="bold")
        table.add_column("文件", style="cyan")
        table.add_column("内容摘要")
        for s in sources:
            preview = s["content"][:80].replace("\n", " ")
            if len(s["content"]) > 80:
                preview += "..."
            table.add_row(s["file_path"], preview)
        console.print(table)

    keyword_index.close()


@main.command()
@click.pass_context
def chat(ctx):
    """交互式对话模式。"""
    cfg = ctx.obj["config"]
    _, vector_store, keyword_index, _, qa_engine = _build_qa_components(cfg)

    # 加载已有索引
    if not vector_store.load():
        console.print("[red]向量索引不存在，请先运行 `qa index` 构建索引。[/red]")
        return

    console.print(Panel("笔记知识问答 - 交互模式\n输入问题进行查询，输入 quit/exit 退出", title="notes-qa"))
    console.print()

    while True:
        try:
            question = console.input("[bold green]你>[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n再见！")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            console.print("再见！")
            break

        console.print()
        with console.status("正在思考..."):
            sources = qa_engine.get_sources(question)

        console.print("[bold cyan]助手>[/bold cyan] ", end="")
        for chunk_text in qa_engine.query(question, stream=True):
            console.print(chunk_text, end="")
        console.print("\n")

        if sources:
            with console.capture() as cap:
                console.print("[dim]参考来源:[/dim]")
                for s in sources:
                    console.print(f"[dim]  - {s['file_path']}[/dim]")
            console.print(cap.get())

    keyword_index.close()


@main.command()
@click.argument("question")
@click.option("--top-k", "-k", default=None, type=int, help="返回结果数量")
@click.pass_context
def search(ctx, question, top_k):
    """仅检索，不调用大模型，返回原始 TopK 结果。"""
    from notes_qa.retriever import Retriever

    cfg = ctx.obj["config"]
    embedder, vector_store, keyword_index = _build_index_components(cfg)

    if not vector_store.load():
        console.print("[red]向量索引不存在，请先运行 `qa index` 构建索引。[/red]")
        return

    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
        keyword_index=keyword_index,
        vector_top_k=cfg["retriever"]["vector_top_k"],
        keyword_top_k=cfg["retriever"]["keyword_top_k"],
        final_top_k=top_k or cfg["retriever"]["final_top_k"],
        vector_weight=cfg["retriever"]["vector_weight"],
    )

    with console.status("检索中..."):
        results = retriever.retrieve(question)

    keyword_index.close()

    if not results:
        console.print("[yellow]未找到相关结果。[/yellow]")
        return

    console.print(f"\n[bold]查询：[/bold]{question}")
    console.print(f"[bold]返回 {len(results)} 条结果：[/bold]\n")

    for i, r in enumerate(results, 1):
        score = r.get("rrf_score", 0)
        console.print(Panel(
            r["content"],
            title=f"[cyan]#{i} {r['file_path']}[/cyan]  (score: {score:.4f})",
            title_align="left",
        ))


@main.command()
@click.option("--host", default="127.0.0.1", help="监听地址")
@click.option("--port", "-p", default=8000, type=int, help="监听端口")
@click.pass_context
def web(ctx, host, port):
    """启动 Web 服务，通过浏览器进行问答。"""
    import uvicorn

    from notes_qa.web import create_app

    config_path = None
    # 传递配置路径给 web 应用
    from pathlib import Path

    base = Path(__file__).parent.parent
    local = base / "config.local.yaml"
    config_path = str(local if local.exists() else base / "config.yaml")

    app = create_app(config_path)
    console.print(f"[green]启动 Web 服务: http://{host}:{port}[/green]")
    uvicorn.run(app, host=host, port=port, log_level="info")


@main.command()
@click.pass_context
def config_show(ctx):
    """显示当前配置。"""
    cfg = ctx.obj["config"]
    table = Table(title="当前配置", show_header=True, header_style="bold")
    table.add_column("配置项", style="cyan")
    table.add_column("值")

    table.add_row("笔记目录", cfg.get("notes_dir", ""))
    table.add_row("Embedding 模型", cfg.get("embedding", {}).get("model", ""))
    table.add_row("Chunk Size", str(cfg.get("chunking", {}).get("chunk_size", "")))
    table.add_row("LLM 模型", cfg.get("llm", {}).get("model", ""))
    table.add_row("LLM API", cfg.get("llm", {}).get("api_base", ""))
    table.add_row("索引目录", cfg.get("index_dir", ""))

    console.print(table)

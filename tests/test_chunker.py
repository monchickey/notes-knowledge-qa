from notes_qa.loader import Document
from notes_qa.chunker import chunk_document, chunk_documents


def test_chunk_short_document():
    doc = Document(content="这是一段很短的文本。", file_path="test.md", title="test")
    chunks = chunk_document(doc, chunk_size=100, chunk_overlap=10)
    assert len(chunks) == 1
    assert chunks[0].content == "这是一段很短的文本。"
    assert chunks[0].chunk_index == 0


def test_chunk_long_document():
    content = "段落一。\n\n" * 50
    doc = Document(content=content, file_path="test.md", title="test")
    chunks = chunk_document(doc, chunk_size=100, chunk_overlap=10)
    assert len(chunks) > 1
    # 每个 chunk 不超过 chunk_size + overlap
    for c in chunks:
        assert len(c.content) <= 120


def test_chunk_preserves_metadata():
    doc = Document(content="内容A\n\n内容B\n\n内容C", file_path="path/to/note.md", title="My Note")
    chunks = chunk_document(doc, chunk_size=20, chunk_overlap=0)
    for c in chunks:
        assert c.file_path == "path/to/note.md"
        assert c.title == "My Note"


def test_chunk_documents_batch():
    docs = [
        Document(content="文档一内容。", file_path="a.md", title="A"),
        Document(content="文档二内容。", file_path="b.md", title="B"),
    ]
    chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=0)
    assert len(chunks) == 2
    assert chunks[0].file_path == "a.md"
    assert chunks[1].file_path == "b.md"

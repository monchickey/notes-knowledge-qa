import tempfile
from pathlib import Path

from notes_qa.loader import load_documents, _strip_front_matter, _extract_title


def test_strip_front_matter():
    text = "---\ntitle: test\ntags: [a, b]\n---\n\nHello world"
    result = _strip_front_matter(text)
    assert result == "Hello world"


def test_strip_no_front_matter():
    text = "# Title\n\nContent"
    result = _strip_front_matter(text)
    assert result == text


def test_extract_title_from_heading():
    text = "# My Title\n\nSome content"
    assert _extract_title(text, "test.md") == "My Title"


def test_extract_title_from_filename():
    text = "No heading here"
    assert _extract_title(text, "/path/to/my-note.md") == "my-note"


def test_load_documents():
    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建测试文件
        (Path(tmpdir) / "note1.md").write_text("# 笔记一\n\n这是第一篇笔记。")
        (Path(tmpdir) / "note2.md").write_text("# 笔记二\n\n这是第二篇笔记。")
        (Path(tmpdir) / "empty.md").write_text("")
        (Path(tmpdir) / "sub").mkdir()
        (Path(tmpdir) / "sub" / "note3.md").write_text("# 子目录笔记\n\n嵌套内容。")

        docs = load_documents(tmpdir)
        assert len(docs) == 3  # empty.md 应被跳过
        titles = [d.title for d in docs]
        assert "笔记一" in titles
        assert "笔记二" in titles
        assert "子目录笔记" in titles


def test_load_nonexistent_dir():
    import pytest
    with pytest.raises(NotADirectoryError):
        load_documents("/nonexistent/path")

"""历史记录模块测试"""

import pytest
import tempfile
from pathlib import Path

from notes_qa.history import ChatHistory


@pytest.fixture
def history():
    """创建临时数据库的历史记录实例"""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    history = ChatHistory(db_path)
    yield history
    # 清理
    Path(db_path).unlink(missing_ok=True)


def test_create_conversation(history):
    """测试创建对话"""
    conv_id = history.create_conversation("测试对话")
    assert conv_id is not None
    assert conv_id > 0


def test_add_message(history):
    """测试添加消息"""
    conv_id = history.create_conversation("测试对话")
    msg_id = history.add_message(conv_id, "user", "你好")
    assert msg_id is not None
    assert msg_id > 0


def test_get_conversations(history):
    """测试获取对话列表"""
    # 创建多个对话
    history.create_conversation("对话1")
    history.create_conversation("对话2")
    history.create_conversation("对话3")

    conversations = history.get_conversations()
    assert len(conversations) == 3
    # 验证返回了所有对话
    titles = {c['title'] for c in conversations}
    assert titles == {"对话1", "对话2", "对话3"}


def test_get_conversation_messages(history):
    """测试获取对话消息"""
    conv_id = history.create_conversation("测试对话")
    history.add_message(conv_id, "user", "问题1")
    history.add_message(conv_id, "assistant", "回答1")
    history.add_message(conv_id, "user", "问题2")

    messages = history.get_conversation_messages(conv_id)
    assert len(messages) == 3
    assert messages[0]['role'] == 'user'
    assert messages[0]['content'] == '问题1'
    assert messages[1]['role'] == 'assistant'
    assert messages[1]['content'] == '回答1'


def test_delete_conversation(history):
    """测试删除对话"""
    conv_id = history.create_conversation("测试对话")
    history.add_message(conv_id, "user", "问题")

    success = history.delete_conversation(conv_id)
    assert success is True

    # 验证对话已删除
    conversation = history.get_conversation(conv_id)
    assert conversation is None


def test_update_conversation_title(history):
    """测试更新对话标题"""
    conv_id = history.create_conversation("原标题")

    success = history.update_conversation_title(conv_id, "新标题")
    assert success is True

    conversation = history.get_conversation(conv_id)
    assert conversation['title'] == "新标题"


def test_search_messages(history):
    """测试搜索消息"""
    conv_id = history.create_conversation("测试对话")
    history.add_message(conv_id, "user", "什么是 Python？")
    history.add_message(conv_id, "assistant", "Python 是一种编程语言")
    history.add_message(conv_id, "user", "什么是 Java？")

    results = history.search_messages("Python")
    assert len(results) == 2  # 用户问题和助手回答都包含 Python


def test_get_message_count(history):
    """测试获取消息数量"""
    conv_id = history.create_conversation("测试对话")
    history.add_message(conv_id, "user", "问题1")
    history.add_message(conv_id, "assistant", "回答1")

    count = history.get_message_count(conv_id)
    assert count == 2


def test_sources_storage(history):
    """测试参考来源存储"""
    conv_id = history.create_conversation("测试对话")
    sources = [
        {"file_path": "test.md", "content": "测试内容"},
    ]
    history.add_message(conv_id, "assistant", "回答", sources=sources)

    messages = history.get_conversation_messages(conv_id)
    assert messages[0]['sources'] is not None
    assert len(messages[0]['sources']) == 1
    assert messages[0]['sources'][0]['file_path'] == "test.md"


def test_mode_storage(history):
    """测试模式存储"""
    conv_id = history.create_conversation("测试对话")
    history.add_message(conv_id, "user", "搜索词", mode="search")

    messages = history.get_conversation_messages(conv_id)
    assert messages[0]['mode'] == "search"

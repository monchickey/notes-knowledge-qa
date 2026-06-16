import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional


class ChatHistory:
    """Web 端对话历史记录管理"""

    def __init__(self, db_path: str | Path = "chat_history.db"):
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self):
        """初始化数据库表"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    sources TEXT,
                    mode TEXT DEFAULT 'chat',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_conversation_id
                ON messages(conversation_id)
            """)
            conn.commit()

    def create_conversation(self, title: str) -> int:
        """创建新对话，返回对话 ID"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "INSERT INTO conversations (title) VALUES (?)",
                (title,)
            )
            conn.commit()
            return cursor.lastrowid

    def add_message(self, conversation_id: int, role: str, content: str,
                    sources: Optional[List[Dict]] = None, mode: str = "chat") -> int:
        """添加消息到对话"""
        sources_json = json.dumps(sources, ensure_ascii=False) if sources else None
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "INSERT INTO messages (conversation_id, role, content, sources, mode) VALUES (?, ?, ?, ?, ?)",
                (conversation_id, role, content, sources_json, mode)
            )
            # 更新对话的更新时间
            conn.execute(
                "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (conversation_id,)
            )
            conn.commit()
            return cursor.lastrowid

    def get_conversations(self, limit: int = 50, offset: int = 0) -> List[Dict]:
        """获取对话列表"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """SELECT id, title, created_at, updated_at
                   FROM conversations
                   ORDER BY updated_at DESC
                   LIMIT ? OFFSET ?""",
                (limit, offset)
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_conversation_messages(self, conversation_id: int) -> List[Dict]:
        """获取对话的所有消息"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """SELECT id, role, content, sources, mode, created_at
                   FROM messages
                   WHERE conversation_id = ?
                   ORDER BY created_at ASC""",
                (conversation_id,)
            )
            messages = []
            for row in cursor.fetchall():
                msg = dict(row)
                if msg['sources']:
                    msg['sources'] = json.loads(msg['sources'])
                messages.append(msg)
            return messages

    def get_conversation(self, conversation_id: int) -> Optional[Dict]:
        """获取单个对话信息"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?",
                (conversation_id,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def delete_conversation(self, conversation_id: int) -> bool:
        """删除对话及其所有消息"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM conversations WHERE id = ?",
                (conversation_id,)
            )
            conn.commit()
            return cursor.rowcount > 0

    def update_conversation_title(self, conversation_id: int, title: str) -> bool:
        """更新对话标题"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "UPDATE conversations SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (title, conversation_id)
            )
            conn.commit()
            return cursor.rowcount > 0

    def search_messages(self, keyword: str, limit: int = 20) -> List[Dict]:
        """搜索消息内容"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """SELECT m.id, m.conversation_id, m.role, m.content, m.sources, m.mode, m.created_at,
                          c.title as conversation_title
                   FROM messages m
                   JOIN conversations c ON m.conversation_id = c.id
                   WHERE m.content LIKE ?
                   ORDER BY m.created_at DESC
                   LIMIT ?""",
                (f"%{keyword}%", limit)
            )
            messages = []
            for row in cursor.fetchall():
                msg = dict(row)
                if msg['sources']:
                    msg['sources'] = json.loads(msg['sources'])
                messages.append(msg)
            return messages

    def get_message_count(self, conversation_id: int) -> int:
        """获取对话中的消息数量"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE conversation_id = ?",
                (conversation_id,)
            )
            return cursor.fetchone()[0]

    def close(self):
        """关闭数据库连接（如果需要）"""
        pass

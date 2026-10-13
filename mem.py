import sqlite3

DB = "memory.db"


def init_memory():
    conn = sqlite3.connect(DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def remember(content):
    conn = sqlite3.connect(DB)
    conn.execute(
        "INSERT INTO memories (content) VALUES (?)",
        (content,)
    )
    conn.commit()
    conn.close()


def recall():
    conn = sqlite3.connect(DB)
    rows = conn.execute("""
        SELECT content
        FROM memories
        ORDER BY created_at DESC
    """).fetchall()
    conn.close()
    return [row[0] for row in rows]
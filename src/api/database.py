import sqlite3
import os
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "feedback.db")

def init_db():
    """Initialise la base de données SQLite."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            session_id TEXT,
            niveau_predit INTEGER NOT NULL,
            niveau_reel INTEGER,
            commentaire TEXT,
            utile INTEGER NOT NULL DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()

def save_feedback(
    niveau_predit: int,
    utile: bool,
    niveau_reel: int | None = None,
    commentaire: str | None = None,
    session_id: str | None = None,
) -> int:
    """Sauvegarde un feedback en base."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO feedback (timestamp, session_id, niveau_predit, niveau_reel, commentaire, utile)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        datetime.now(timezone.utc).isoformat(),
        session_id,
        niveau_predit,
        niveau_reel,
        commentaire,
        1 if utile else 0,
    ))
    conn.commit()
    feedback_id = cursor.lastrowid
    conn.close()
    return feedback_id

def get_feedbacks(limit: int = 50) -> list:
    """Récupère les derniers feedbacks."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, timestamp, niveau_predit, niveau_reel, commentaire, utile
        FROM feedback
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [
        {
            "id": r[0],
            "timestamp": r[1],
            "niveau_predit": r[2],
            "niveau_reel": r[3],
            "commentaire": r[4],
            "utile": bool(r[5]),
        }
        for r in rows
    ]
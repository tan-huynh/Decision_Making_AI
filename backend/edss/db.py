import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "decision_learning.db"

def get_db_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS decision_history (
            id TEXT PRIMARY KEY,
            problem_text TEXT,
            problem_type TEXT,
            solver_used TEXT,
            decision_outcome TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def log_decision(problem_text: str, problem_type: str, solver_used: str, decision_outcome: str) -> str:
    conn = get_db_connection()
    cursor = conn.cursor()
    record_id = str(uuid.uuid4())
    cursor.execute('''
        INSERT INTO decision_history (id, problem_text, problem_type, solver_used, decision_outcome)
        VALUES (?, ?, ?, ?, ?)
    ''', (record_id, problem_text, problem_type, solver_used, decision_outcome))
    conn.commit()
    conn.close()
    return record_id

# Auto-initialize on import
init_db()

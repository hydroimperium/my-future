import sqlite3
import json
from pathlib import Path
from datetime import datetime
from calculator import UserHabitsInput

# База всегда будет создаваться строго внутри папки backend
DB_PATH = Path(__file__).parent / "predictions.db"


def init_db():
    """Создает таблицу в SQLite при запуске сервера, если её ещё нет."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                age INTEGER,
                gender TEXT,
                bmi REAL,
                predicted_expectancy REAL,
                potential_expectancy REAL,
                biological_age REAL,
                ten_year_savings_rub INTEGER,
                raw_input TEXT,
                full_result TEXT,
                ai_explanation TEXT
            )
        """)
        conn.commit()


def save_prediction(user_input: UserHabitsInput, calc_result: dict, ai_text: str) -> int:
    """Сохраняет анкету пользователя и результат расчёта в БД. Возвращает ID записи."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO predictions (
                created_at, age, gender, bmi,
                predicted_expectancy, potential_expectancy,
                biological_age, ten_year_savings_rub,
                raw_input, full_result, ai_explanation
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(timespec="seconds"),
            user_input.age,
            user_input.gender,
            calc_result["bmi"],
            calc_result["predicted_expectancy"],
            calc_result["potential_expectancy"],
            calc_result["biological_age"],
            calc_result["ten_year_savings_rub"],
            json.dumps(user_input.model_dump(), ensure_ascii=False),
            json.dumps(calc_result, ensure_ascii=False),
            ai_text
        ))
        conn.commit()
        return cursor.lastrowid


def get_recent_predictions(limit: int = 10) -> list[dict]:
    """Возвращает последние сохранённые расчёты (для проверки и личного кабинета)."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, created_at, age, gender, bmi,
                   predicted_expectancy, potential_expectancy,
                   biological_age, ten_year_savings_rub, ai_explanation
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    
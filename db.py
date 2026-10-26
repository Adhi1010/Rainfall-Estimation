import sqlite3
import os
from werkzeug.security import generate_password_hash, check_password_hash

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FOLDER = os.path.join(BASE_DIR, "database")
DB_PATH = os.path.join(DB_FOLDER, "rainfall.db")


def get_db_connection():
    os.makedirs(DB_FOLDER, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def create_tables():
    conn = get_db_connection()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            place_name TEXT NOT NULL,
            temperature REAL,
            humidity REAL,
            pressure REAL,
            wind_speed REAL,
            rainfall_mm REAL NOT NULL,
            predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    """)

    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_pred_user ON predictions(user_id)"
    )

    conn.commit()
    conn.close()


# ---------------- USERS ----------------
def create_user(name, email, password):
    """Returns True if created, False if the email already exists."""
    conn = get_db_connection()
    try:
        conn.execute(
            "INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
            (name.strip(), email.strip().lower(), generate_password_hash(password)),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def verify_user(email, password):
    """Returns the user row if the login is correct, else None."""
    conn = get_db_connection()
    user = conn.execute(
        "SELECT * FROM users WHERE email = ?", (email.strip().lower(),)
    ).fetchone()
    conn.close()
    if user and check_password_hash(user["password"], password):
        return user
    return None


# ---------------- PREDICTIONS ----------------
def save_prediction(user_id, place_name, temperature, humidity,
                    pressure, wind_speed, rainfall_mm):
    conn = get_db_connection()
    conn.execute("""
        INSERT INTO predictions
        (user_id, place_name, temperature, humidity, pressure, wind_speed, rainfall_mm)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (user_id, place_name, temperature, humidity,
          pressure, wind_speed, rainfall_mm))
    conn.commit()
    conn.close()


def get_history(user_id):
    conn = get_db_connection()
    rows = conn.execute(
        "SELECT * FROM predictions WHERE user_id = ? ORDER BY predicted_at DESC",
        (user_id,),
    ).fetchall()
    conn.close()
    return rows


if __name__ == "__main__":
    create_tables()
    print("Database created successfully!")
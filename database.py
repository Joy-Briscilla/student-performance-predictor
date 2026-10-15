"""SQLite storage: users, student submissions, staff-verified data, unit marks, predictions."""
import json
import sqlite3
from datetime import datetime

from werkzeug.security import check_password_hash, generate_password_hash

from common import BASE_DIR

DB_PATH = BASE_DIR / "data" / "app.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('student','staff')),
    class_name TEXT NOT NULL,
    mentor TEXT DEFAULT '',
    password_hash TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS submissions (          -- student-entered data (many per student)
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT NOT NULL, created_at TEXT NOT NULL, data TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS verified (             -- staff-verified data (one current row)
    student_id TEXT PRIMARY KEY, g1 REAL, g2 REAL, absences INTEGER, failures INTEGER,
    updated_at TEXT, updated_by TEXT
);
CREATE TABLE IF NOT EXISTS unit_marks (           -- staff-entered unit marks out of 20
    student_id TEXT NOT NULL, subject TEXT NOT NULL, unit INTEGER NOT NULL, mark REAL,
    PRIMARY KEY (student_id, subject, unit)
);
CREATE TABLE IF NOT EXISTS predictions (          -- history used for the progress chart
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT NOT NULL, created_at TEXT NOT NULL, pass_prob REAL, grade REAL,
    risk TEXT, mode TEXT, source TEXT
);
"""


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    with conn() as c:
        c.executescript(SCHEMA)


# ---------------------------------------------------------------- users
def create_user(user_id, name, role, class_name, password, mentor=""):
    try:
        with conn() as c:
            c.execute("INSERT INTO users VALUES (?,?,?,?,?,?)",
                      (user_id.strip().upper(), name.strip(), role, class_name.strip().upper(),
                       mentor.strip(), generate_password_hash(password)))
        return True
    except sqlite3.IntegrityError:
        return False


def check_login(user_id, password):
    with conn() as c:
        u = c.execute("SELECT * FROM users WHERE user_id=?", (user_id.strip().upper(),)).fetchone()
    return u if u and check_password_hash(u["password_hash"], password) else None


def get_user(user_id):
    with conn() as c:
        return c.execute("SELECT * FROM users WHERE user_id=?", (user_id,)).fetchone()


def class_students(class_name):
    with conn() as c:
        return c.execute("SELECT * FROM users WHERE role='student' AND class_name=? ORDER BY name",
                         (class_name,)).fetchall()


# ---------------------------------------------------------- submissions
def add_submission(student_id, data: dict, when=None):
    with conn() as c:
        c.execute("INSERT INTO submissions (student_id, created_at, data) VALUES (?,?,?)",
                  (student_id, when or now(), json.dumps(data)))


def latest_submission(student_id):
    with conn() as c:
        r = c.execute("SELECT * FROM submissions WHERE student_id=? ORDER BY id DESC LIMIT 1",
                      (student_id,)).fetchone()
    return (json.loads(r["data"]), r["created_at"]) if r else (None, None)


# ------------------------------------------------------------- verified
def get_verified(student_id):
    with conn() as c:
        r = c.execute("SELECT * FROM verified WHERE student_id=?", (student_id,)).fetchone()
    return dict(r) if r else None


def save_verified(student_id, g1, g2, absences, failures, staff_id, when=None):
    with conn() as c:
        c.execute("""INSERT INTO verified VALUES (?,?,?,?,?,?,?)
                     ON CONFLICT(student_id) DO UPDATE SET g1=excluded.g1, g2=excluded.g2,
                     absences=excluded.absences, failures=excluded.failures,
                     updated_at=excluded.updated_at, updated_by=excluded.updated_by""",
                  (student_id, g1, g2, absences, failures, when or now(), staff_id))


def get_unit_marks(student_id):
    """Returns {subject: {unit: mark}}"""
    out = {}
    with conn() as c:
        for r in c.execute("SELECT * FROM unit_marks WHERE student_id=?", (student_id,)):
            out.setdefault(r["subject"], {})[r["unit"]] = r["mark"]
    return out


def save_unit_marks(student_id, marks: dict):
    """marks = {(subject, unit): value or None}"""
    with conn() as c:
        for (subject, unit), mark in marks.items():
            if mark is None:
                c.execute("DELETE FROM unit_marks WHERE student_id=? AND subject=? AND unit=?",
                          (student_id, subject, unit))
            else:
                c.execute("""INSERT INTO unit_marks VALUES (?,?,?,?)
                             ON CONFLICT(student_id, subject, unit) DO UPDATE SET mark=excluded.mark""",
                          (student_id, subject, unit, mark))


# ----------------------------------------------------------- predictions
def add_prediction(student_id, result: dict, source, when=None):
    with conn() as c:
        c.execute("""INSERT INTO predictions (student_id, created_at, pass_prob, grade, risk, mode, source)
                     VALUES (?,?,?,?,?,?,?)""",
                  (student_id, when or now(), result["pass_prob"], result["grade"],
                   result["risk"], result["mode"], source))


def prediction_history(student_id):
    with conn() as c:
        return [dict(r) for r in c.execute(
            "SELECT * FROM predictions WHERE student_id=? ORDER BY id", (student_id,))]


def latest_prediction(student_id):
    h = prediction_history(student_id)
    return h[-1] if h else None

"""Prediction engine: turns student-entered + staff-verified data into a prediction and advice."""
from functools import lru_cache

import joblib
import pandas as pd

import database as db
from common import MODEL_DIR, clean_data, load_data, risk_level, suggestions

PASS_MARK_20 = 10          # unit / subject pass level (out of 20)
SUBJECTS = {"Mathematics": 5, "Physics": 5, "Programming": 5}   # subject -> number of units


@lru_cache(maxsize=1)
def _models():
    return {s: (joblib.load(MODEL_DIR / f"pass_fail_{s}.joblib"), joblib.load(MODEL_DIR / f"score_{s}.joblib"))
            for s in ("full", "early")}


@lru_cache(maxsize=1)
def _defaults():
    """Typical value (median / mode) for every column the student does not enter."""
    df, _ = clean_data(load_data())
    d = {}
    for c in df.columns:
        d[c] = df[c].median() if df[c].dtype != object else df[c].mode()[0]
    return d


def hours_to_studytime(hours):
    return 1 if hours < 2 else 2 if hours < 5 else 3 if hours < 10 else 4


def units_covered(ratings: dict):
    """ratings = {subject: [r1..rN]} where 0 = not studied, 1-5 = how well. Returns {subject: count}."""
    return {s: sum(1 for r in ratings.get(s, []) if r and r > 0) for s in SUBJECTS}


def predict(habits: dict | None, verified: dict | None):
    """Return dict(pass_prob, grade, risk, mode). Uses FULL mode only when G1 and G2 are verified."""
    row = dict(_defaults())
    habits = habits or {}
    verified = verified or {}
    if habits:
        row["studytime"] = hours_to_studytime(habits.get("study_hours", 5))
        for k in ("freetime", "goout", "health", "internet", "famsup"):
            if k in habits:
                row[k] = habits[k]
    for k in ("absences", "failures"):
        if verified.get(k) is not None:
            row[k] = verified[k]
    full = verified.get("g1") is not None and verified.get("g2") is not None
    if full:
        row["G1"], row["G2"] = verified["g1"], verified["g2"]
    clf, reg = _models()["full" if full else "early"]
    X = pd.DataFrame([row])[clf["num"] + clf["cat"]]
    p = float(clf["model"].predict_proba(X)[0, 1])
    grade = float(max(0, min(20, reg["model"].predict(X)[0])))
    return dict(pass_prob=p, grade=grade, risk=risk_level(p), mode="full" if full else "early", row=row)


def analyse_units(unit_marks: dict, ratings: dict | None):
    """Per-subject averages, weak units (sorted worst first), marks needed to pass, units still to cover."""
    subjects, weak, to_cover = [], [], []
    covered = units_covered(ratings or {})
    for s, n in SUBJECTS.items():
        marks = unit_marks.get(s, {})
        avg = sum(marks.values()) / len(marks) if marks else None
        needed = max(0.0, PASS_MARK_20 - avg) if avg is not None else None
        subjects.append(dict(name=s, avg=avg, needed=needed, covered=covered[s], total=n))
        for u, m in marks.items():
            if m < PASS_MARK_20:
                weak.append(dict(subject=s, unit=u, mark=m))
        if covered[s] < n and ratings:
            to_cover.append((s, n - covered[s]))
    weak.sort(key=lambda w: w["mark"])
    return subjects, weak, to_cover


def build_advice(result, habits, verified, unit_marks):
    """Rule-based advice on top of the ML prediction. Returns (tips, subjects, weak_units)."""
    ratings = (habits or {}).get("ratings")
    subjects, weak, to_cover = analyse_units(unit_marks, ratings)
    tips = []
    for w in weak[:4]:
        tips.append(f"Revise {w['subject']} Unit {w['unit']} first (scored {w['mark']:g}/20).")
    for s in subjects:
        if s["needed"]:
            tips.append(f"{s['name']}: improve your average by about {s['needed']:.1f} marks to reach pass level ({PASS_MARK_20}/20).")
    for name, n in to_cover:
        tips.append(f"{name}: complete {n} more unit{'s' if n > 1 else ''}.")
    base = dict(result["row"])
    base["schoolsup"] = base.get("schoolsup", "no")
    base["studytime"] = result["row"]["studytime"]
    for t in suggestions(base):
        if not t.startswith("Latest internal mark") or result["mode"] == "full":
            if not t.startswith("No major risk"):
                tips.append(t)
    if not tips:
        tips.append("No major risk factors found. Keep up the current routine.")
    return tips, subjects, weak


def run_and_store(student_id, source, when=None):
    """Recompute and save a prediction from the student's latest data."""
    habits, _ = db.latest_submission(student_id)
    verified = db.get_verified(student_id)
    result = predict(habits, verified)
    db.add_prediction(student_id, result, source, when)
    return result


def progress_svg(history, width=520, height=200):
    """Small inline SVG line chart of pass probability over time (no JS library needed)."""
    if not history:
        return ""
    pad_l, pad_r, pad_t, pad_b = 40, 16, 16, 34
    w, h = width - pad_l - pad_r, height - pad_t - pad_b
    n = len(history)
    xs = [pad_l + (w * i / (n - 1) if n > 1 else w / 2) for i in range(n)]
    ys = [pad_t + h * (1 - hh["pass_prob"]) for hh in history]
    out = [f'<svg viewBox="0 0 {width} {height}" class="chart" role="img" aria-label="Pass probability over time">']
    for pct in (0, 25, 50, 75, 100):
        y = pad_t + h * (1 - pct / 100)
        out.append(f'<line x1="{pad_l}" x2="{width - pad_r}" y1="{y:.1f}" y2="{y:.1f}" stroke="#e3e8f0"/>'
                   f'<text x="{pad_l - 6}" y="{y + 4:.1f}" text-anchor="end" font-size="11" fill="#6b7788">{pct}%</text>')
    out.append(f'<line x1="{pad_l}" x2="{width - pad_r}" y1="{pad_t + h / 2:.1f}" y2="{pad_t + h / 2:.1f}" stroke="#e0524d" stroke-dasharray="4 4"/>')
    if n > 1:
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
        out.append(f'<polyline points="{pts}" fill="none" stroke="#2f6fdb" stroke-width="2.5"/>')
    for x, y, hh in zip(xs, ys, history):
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="#2f6fdb"/>'
                   f'<text x="{x:.1f}" y="{y - 9:.1f}" text-anchor="middle" font-size="11" font-weight="600" fill="#1d2b3f">{hh["pass_prob"]:.0%}</text>'
                   f'<text x="{x:.1f}" y="{height - 12}" text-anchor="middle" font-size="10" fill="#6b7788">{hh["created_at"][5:10]}</text>')
    out.append("</svg>")
    return "".join(out)

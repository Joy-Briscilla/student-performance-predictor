"""Shared helpers: data loading, cleaning, feature lists, risk levels, advice."""
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "student_data.csv"
MODEL_DIR = BASE_DIR / "models"
FIG_DIR = BASE_DIR / "figures"
REPORT_DIR = BASE_DIR / "reports"

PASS_MARK = 10  # G3 >= 10 (out of 20) means "pass"

NUMERIC_COLS = [
    "age", "Medu", "Fedu", "traveltime", "studytime", "failures", "famrel",
    "freetime", "goout", "Dalc", "Walc", "health", "absences", "G1", "G2", "G3",
]
CATEGORICAL_COLS = [
    "school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob", "reason",
    "guardian", "schoolsup", "famsup", "paid", "activities", "nursery",
    "higher", "internet", "romantic",
]
GRADE_COLS = ["G1", "G2"]
TARGET = "G3"


def load_data(path=DATA_PATH) -> pd.DataFrame:
    """Load the CSV (auto-detects ',' or ';' so the real UCI file works too)."""
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip() for c in df.columns]
    return df


def clean_data(df: pd.DataFrame):
    """Remove duplicates, fix types and fill missing values. Returns (df, report)."""
    report = {"rows_before": int(len(df))}
    df = df.drop_duplicates().copy()
    report["duplicates_removed"] = report["rows_before"] - int(len(df))

    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype("object")

    report["missing_before"] = {c: int(v) for c, v in df.isna().sum().items() if v > 0}

    for col in NUMERIC_COLS:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    for col in CATEGORICAL_COLS:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    # keep marks inside the valid 0-20 range
    for col in ["G1", "G2", "G3"]:
        df[col] = df[col].clip(0, 20)

    report["rows_after"] = int(len(df))
    return df, report


def feature_lists(scenario: str):
    """'full' uses every column; 'early' hides G1/G2 (prediction before any test)."""
    drop = {TARGET} | (set(GRADE_COLS) if scenario == "early" else set())
    num = [c for c in NUMERIC_COLS if c not in drop]
    cat = list(CATEGORICAL_COLS)
    return num, cat


def risk_level(p_pass: float) -> str:
    if p_pass >= 0.75:
        return "Low"
    if p_pass >= 0.50:
        return "Medium"
    return "High"


def suggestions(row: dict) -> list:
    """Simple rule-based advice shown next to a prediction."""
    tips = []
    if row.get("absences", 0) >= 10:
        tips.append("High absences: set an attendance target and follow up with the student's mentor.")
    if row.get("studytime", 2) <= 1:
        tips.append("Very low weekly study time: plan a fixed daily study slot (aim for 5+ hours a week).")
    if row.get("failures", 0) >= 1:
        tips.append("Past failures: arrange remedial or bridge classes on the weak subjects.")
    if row.get("schoolsup", "no") == "no" and row.get("failures", 0) >= 1:
        tips.append("Enrol the student in extra educational support.")
    if row.get("goout", 3) >= 4:
        tips.append("Heavy social outings: discuss time management and balance.")
    if row.get("Walc", 1) >= 3 or row.get("Dalc", 1) >= 3:
        tips.append("Alcohol use is high: refer to a counsellor.")
    if row.get("higher", "yes") == "no":
        tips.append("Low interest in higher education: hold a career-guidance session to build motivation.")
    if row.get("internet", "yes") == "no":
        tips.append("No internet at home: give access to the library or computer lab.")
    if row.get("health", 3) <= 2:
        tips.append("Poor health reported: check in on wellbeing and allow flexible deadlines.")
    if "G2" in row and row.get("G2", 10) < 10:
        tips.append("Latest internal mark is below pass level: schedule revision and a re-test.")
    if not tips:
        tips.append("No major risk factors found. Keep up the current routine.")
    return tips

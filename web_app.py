"""Student Performance Predictor - website (Flask).   Run:  python web_app.py   then open http://127.0.0.1:5000"""
import csv
import io
import os
import secrets
from functools import wraps

from flask import (Flask, Response, abort, flash, redirect, render_template, request, session, url_for)

import database as db
import engine
from engine import SUBJECTS

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change-this-secret-in-production")
STAFF_CODE = os.environ.get("STAFF_CODE", "STAFF2026")   # needed to register as staff
db.init_db()

RISK_ORDER = {"High": 0, "Medium": 1, "Low": 2}


# ------------------------------------------------------------ security helpers
@app.before_request
def csrf_protect():
    if "csrf" not in session:
        session["csrf"] = secrets.token_hex(16)
    if request.method == "POST" and request.form.get("csrf") != session.get("csrf"):
        abort(400, "Invalid form token. Please go back and try again.")


@app.context_processor
def inject():
    return dict(csrf=session.get("csrf", ""), me=current_user(), subjects=SUBJECTS)


def current_user():
    uid = session.get("uid")
    return db.get_user(uid) if uid else None


def login_required(role):
    def deco(fn):
        @wraps(fn)
        def wrapper(*a, **kw):
            u = current_user()
            if not u:
                return redirect(url_for("login"))
            if u["role"] != role:
                abort(403)
            return fn(u, *a, **kw)
        return wrapper
    return deco


def num(field, label, lo, hi, integer=True, required=True):
    """Read and validate a numeric form field."""
    raw = (request.form.get(field) or "").strip()
    if raw == "":
        if required:
            raise ValueError(f"{label} is required.")
        return None
    try:
        v = int(raw) if integer else float(raw)
    except ValueError:
        raise ValueError(f"{label} must be a number.")
    if not lo <= v <= hi:
        raise ValueError(f"{label} must be between {lo} and {hi}.")
    return v


# ------------------------------------------------------------------ auth pages
@app.route("/")
def home():
    u = current_user()
    if not u:
        return redirect(url_for("login"))
    return redirect(url_for("student_home" if u["role"] == "student" else "staff_home"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        u = db.check_login(request.form.get("user_id", ""), request.form.get("password", ""))
        if u:
            session.clear()
            session["uid"] = u["user_id"]
            return redirect(url_for("home"))
        flash("Wrong ID or password.", "error")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        f = request.form
        role = f.get("role", "student")
        uid, name, cls, pw = f.get("user_id", "").strip(), f.get("name", "").strip(), f.get("class_name", "").strip(), f.get("password", "")
        if not (uid and name and cls):
            flash("Please fill in ID, name and class.", "error")
        elif len(pw) < 6:
            flash("Password must be at least 6 characters.", "error")
        elif role == "staff" and f.get("staff_code") != STAFF_CODE:
            flash("Wrong staff registration code.", "error")
        elif not db.create_user(uid, name, role, cls, pw, f.get("mentor", "")):
            flash("That ID is already registered.", "error")
        else:
            flash("Account created. Please log in.", "ok")
            return redirect(url_for("login"))
    return render_template("register.html")


# ---------------------------------------------------------------- student side
def student_context(student_id):
    habits, habits_date = db.latest_submission(student_id)
    verified = db.get_verified(student_id)
    units = db.get_unit_marks(student_id)
    history = db.prediction_history(student_id)
    result = engine.predict(habits, verified)
    tips, subs, weak = engine.build_advice(result, habits, verified, units)
    return dict(habits=habits, habits_date=habits_date, verified=verified, units=units,
                history=history, latest=history[-1] if history else None, result=result,
                tips=tips, subs=subs, weak=weak, chart=engine.progress_svg(history))


@app.route("/student")
@login_required("student")
def student_home(u):
    return render_template("student_dashboard.html", s=u, **student_context(u["user_id"]))


@app.route("/student/submit", methods=["GET", "POST"])
@login_required("student")
def student_submit(u):
    prev, _ = db.latest_submission(u["user_id"])
    if request.method == "POST":
        try:
            data = dict(
                study_hours=num("study_hours", "Weekly study hours", 0, 80),
                revision_hours=num("revision_hours", "Weekly revision hours", 0, 40),
                freetime=num("freetime", "Free time", 1, 5),
                goout=num("goout", "Going out", 1, 5),
                health=num("health", "Health", 1, 5),
                internet=request.form.get("internet", "yes"),
                famsup=request.form.get("famsup", "yes"),
                ratings={s: [num(f"r_{s}_{i}", f"{s} unit {i}", 0, 5) for i in range(1, n + 1)]
                         for s, n in SUBJECTS.items()},
            )
            if data["internet"] not in ("yes", "no") or data["famsup"] not in ("yes", "no"):
                raise ValueError("Invalid choice.")
        except ValueError as e:
            flash(str(e), "error")
            return render_template("student_form.html", s=u, prev=prev)
        db.add_submission(u["user_id"], data)
        r = engine.run_and_store(u["user_id"], "student")
        flash(f"Saved. Your pass chance is now {r['pass_prob']:.0%}.", "ok")
        return redirect(url_for("student_home"))
    return render_template("student_form.html", s=u, prev=prev)


# ------------------------------------------------------------------ staff side
def class_rows(staff):
    rows = []
    for st in db.class_students(staff["class_name"]):
        sid = st["user_id"]
        pred = db.latest_prediction(sid)
        units = db.get_unit_marks(sid)
        avgs = {s: (sum(units[s].values()) / len(units[s]) if units.get(s) else None) for s in SUBJECTS}
        rows.append(dict(id=sid, name=st["name"], pred=pred, verified=db.get_verified(sid), avgs=avgs))
    return rows


@app.route("/staff")
@login_required("staff")
def staff_home(u):
    rows = class_rows(u)
    q = request.args.get("q", "").strip().lower()
    risk = request.args.get("risk", "")
    subj = request.args.get("subject", "")
    view = [r for r in rows
            if (not q or q in r["name"].lower() or q in r["id"].lower())
            and (not risk or (r["pred"] and r["pred"]["risk"] == risk))
            and (not subj or (r["avgs"].get(subj) is not None and r["avgs"][subj] < engine.PASS_MARK_20))]
    view.sort(key=lambda r: (RISK_ORDER.get(r["pred"]["risk"], 3) if r["pred"] else 3,
                             r["pred"]["pass_prob"] if r["pred"] else 1))
    predicted = [r for r in rows if r["pred"]]
    n = len(predicted)
    passing = sum(1 for r in predicted if r["pred"]["pass_prob"] >= 0.5)
    risk_counts = {k: sum(1 for r in predicted if r["pred"]["risk"] == k) for k in ("High", "Medium", "Low")}
    subj_avg = {}
    for s in SUBJECTS:
        vals = [r["avgs"][s] for r in rows if r["avgs"][s] is not None]
        subj_avg[s] = sum(vals) / len(vals) if vals else None
    return render_template("staff_dashboard.html", s=u, rows=view, total=len(rows), n=n, passing=passing,
                           risk_counts=risk_counts, subj_avg=subj_avg, q=q, risk=risk, subj=subj)


def own_student(staff, sid):
    st = db.get_user(sid)
    if not st or st["role"] != "student" or st["class_name"] != staff["class_name"]:
        abort(404)   # staff cannot see other classes
    return st


@app.route("/staff/student/<sid>", methods=["GET", "POST"])
@login_required("staff")
def staff_student(u, sid):
    st = own_student(u, sid)
    if request.method == "POST":
        try:
            g1 = num("g1", "G1", 0, 20, integer=False, required=False)
            g2 = num("g2", "G2", 0, 20, integer=False, required=False)
            if (g1 is None) != (g2 is None):
                raise ValueError("Enter both G1 and G2, or leave both empty.")
            absences = num("absences", "Absences", 0, 93)
            failures = num("failures", "Past failures", 0, 3)
            marks = {(s, i): num(f"um_{s}_{i}", f"{s} unit {i} mark", 0, 20, integer=False, required=False)
                     for s, n in SUBJECTS.items() for i in range(1, n + 1)}
        except ValueError as e:
            flash(str(e), "error")
        else:
            db.save_verified(sid, g1, g2, absences, failures, u["user_id"])
            db.save_unit_marks(sid, marks)
            r = engine.run_and_store(sid, "staff")
            flash(f"Saved. {st['name']}'s pass chance is now {r['pass_prob']:.0%} ({r['risk']} risk).", "ok")
            return redirect(url_for("staff_student", sid=sid))
    return render_template("staff_student.html", s=u, st=st, **student_context(sid))


@app.route("/staff/export.csv")
@login_required("staff")
def export_csv(u):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["student_id", "name", "class", "pass_probability_%", "risk", "predicted_grade", "mode",
                "G1", "G2", "absences", "failures", *[f"{s}_avg_of_20" for s in SUBJECTS], "last_updated"])
    for r in class_rows(u):
        p, v = r["pred"], r["verified"] or {}
        w.writerow([r["id"], r["name"], u["class_name"],
                    round(p["pass_prob"] * 100, 1) if p else "", p["risk"] if p else "",
                    round(p["grade"], 1) if p else "", p["mode"] if p else "",
                    v.get("g1", ""), v.get("g2", ""), v.get("absences", ""), v.get("failures", ""),
                    *[round(r["avgs"][s], 1) if r["avgs"][s] is not None else "" for s in SUBJECTS],
                    p["created_at"] if p else ""])
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename={u['class_name']}_report.csv"})


if __name__ == "__main__":
    app.run(debug=False)

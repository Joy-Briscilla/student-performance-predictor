"""Create demo accounts and sample data so the dashboards are not empty.
Run once:  python seed_demo.py      (it resets data/app.db)

Demo logins:  staff  T001 / staff123  (class ECE-A)     student  S001 ... S020 / student123
              staff  T002 / staff123  (class ECE-B)     student  S101 ... S106 / student123
"""
import random
from datetime import datetime, timedelta

import numpy as np

import database as db
import engine
from engine import SUBJECTS

random.seed(7)
rng = np.random.default_rng(7)

NAMES = ["Aarav", "Divya", "Karthik", "Meena", "Rahul", "Priya", "Suresh", "Anitha", "Vignesh", "Harini",
         "Naveen", "Keerthi", "Arun", "Swetha", "Dinesh", "Lakshmi", "Manoj", "Nandhini", "Prakash", "Revathi",
         "Sanjay", "Tamil", "Uma", "Vimal", "Yazhini", "Zaid"]

if db.DB_PATH.exists():
    db.DB_PATH.unlink()
db.init_db()

db.create_user("T001", "Mrs. Lakshmi Devi", "staff", "ECE-A", "staff123")
db.create_user("T002", "Mr. Ramesh Kumar", "staff", "ECE-B", "staff123")

students = [(f"S{i:03d}", "ECE-A", "Mrs. Lakshmi Devi") for i in range(1, 21)] + \
           [(f"S{100 + i}", "ECE-B", "Mr. Ramesh Kumar") for i in range(1, 7)]
t_old = (datetime.now() - timedelta(days=14)).strftime("%Y-%m-%d %H:%M")
t_new = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d %H:%M")

for k, (sid, cls, mentor) in enumerate(students):
    db.create_user(sid, f"{NAMES[k % len(NAMES)]} {chr(65 + k % 26)}.", "student", cls, "student123", mentor)
    ability = rng.normal(0, 1)                       # hidden quality of the student
    hours = int(np.clip(round(6 + 3 * ability + rng.normal(0, 2)), 0, 30))
    absences = int(np.clip(round(6 - 3 * ability + rng.normal(0, 3)), 0, 40))
    failures = int(np.clip(round(-ability * 0.8 + rng.normal(0, 0.6)), 0, 3))

    def habits(h, covered):
        return dict(study_hours=h, revision_hours=int(h / 3),
                    freetime=int(rng.integers(2, 5)), goout=int(np.clip(3 - ability + rng.integers(-1, 2), 1, 5)),
                    health=int(rng.integers(2, 6)), internet=random.choice(["yes", "yes", "no"]),
                    famsup=random.choice(["yes", "no"]),
                    ratings={s: [int(np.clip(round(3 + ability + rng.normal(0, 1)), 1, 5)) if u <= covered else 0
                                 for u in range(1, n + 1)] for s, n in SUBJECTS.items()})

    # two weeks ago: student filled the form, staff had only entered attendance + failures (early mode)
    db.add_submission(sid, habits(max(0, hours - 2), 2), t_old)
    db.save_verified(sid, None, None, absences, failures, "T001", t_old)
    engine.run_and_store(sid, "student", t_old)

    # two days ago: student updated the form, staff added marks (full mode)
    covered = 3 + int(ability > 0)
    db.add_submission(sid, habits(hours, covered), t_new)
    unit_marks = {(s, u): (float(np.clip(round(11.5 + 3 * ability + rng.normal(0, 2.5)), 0, 20)) if u <= covered else None)
                  for s, n in SUBJECTS.items() for u in range(1, n + 1)}
    vals = [m for m in unit_marks.values() if m is not None]
    g1 = float(np.clip(round(np.mean(vals) + rng.normal(0, 1.5)), 0, 20))
    g2 = float(np.clip(round(np.mean(vals) + rng.normal(0, 1.5)), 0, 20))
    db.save_unit_marks(sid, unit_marks)
    db.save_verified(sid, g1, g2, absences, failures, "T001", t_new)
    engine.run_and_store(sid, "staff", t_new)

print("Demo data created.")
print("Staff   : T001 / staff123 (class ECE-A),  T002 / staff123 (class ECE-B)")
print("Students: S001 ... S020 / student123 (ECE-A),  S101 ... S106 / student123 (ECE-B)")

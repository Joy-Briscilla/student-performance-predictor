"""
Generate a demo dataset that follows the SAME column schema as the public
UCI "Student Performance" dataset (student-mat.csv / student-por.csv).

Why this exists: so the project runs end-to-end out of the box.
To use the REAL dataset, download student-mat.csv from UCI / Kaggle and put it in
data/student_data.csv (the loader auto-detects the ';' separator). Nothing else
needs to change.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 1000


def choice(options, p=None):
    return rng.choice(options, size=N, p=p)


def clip_round(x, lo, hi):
    return np.clip(np.round(x), lo, hi).astype(int)


# --- background / demographic features ---------------------------------
school = choice(["GP", "MS"], p=[0.75, 0.25])
sex = choice(["F", "M"], p=[0.53, 0.47])
age = clip_round(rng.normal(16.7, 1.2, N), 15, 22)
address = choice(["U", "R"], p=[0.77, 0.23])
famsize = choice(["GT3", "LE3"], p=[0.7, 0.3])
Pstatus = choice(["T", "A"], p=[0.88, 0.12])
Medu = clip_round(rng.normal(2.7, 1.1, N), 0, 4)
Fedu = clip_round(Medu * 0.6 + rng.normal(1.1, 1.0, N), 0, 4)
Mjob = choice(["teacher", "health", "services", "at_home", "other"], p=[0.09, 0.09, 0.26, 0.15, 0.41])
Fjob = choice(["teacher", "health", "services", "at_home", "other"], p=[0.07, 0.05, 0.28, 0.05, 0.55])
reason = choice(["home", "reputation", "course", "other"], p=[0.28, 0.27, 0.37, 0.08])
guardian = choice(["mother", "father", "other"], p=[0.69, 0.23, 0.08])
traveltime = clip_round(rng.choice([1, 2, 3, 4], size=N, p=[0.65, 0.27, 0.06, 0.02]), 1, 4)

# --- behaviour features --------------------------------------------------
studytime = clip_round(rng.choice([1, 2, 3, 4], size=N, p=[0.27, 0.5, 0.16, 0.07]), 1, 4)
failures = clip_round(rng.choice([0, 1, 2, 3], size=N, p=[0.80, 0.12, 0.05, 0.03]), 0, 3)
schoolsup = choice(["yes", "no"], p=[0.13, 0.87])
famsup = choice(["yes", "no"], p=[0.61, 0.39])
paid = choice(["yes", "no"], p=[0.46, 0.54])
activities = choice(["yes", "no"], p=[0.5, 0.5])
nursery = choice(["yes", "no"], p=[0.8, 0.2])
higher = choice(["yes", "no"], p=[0.94, 0.06])
internet = choice(["yes", "no"], p=[0.83, 0.17])
romantic = choice(["yes", "no"], p=[0.33, 0.67])
famrel = clip_round(rng.normal(3.9, 0.9, N), 1, 5)
freetime = clip_round(rng.normal(3.2, 1.0, N), 1, 5)
goout = clip_round(rng.normal(3.1, 1.1, N), 1, 5)
Dalc = clip_round(rng.choice([1, 2, 3, 4, 5], size=N, p=[0.70, 0.19, 0.07, 0.03, 0.01]), 1, 5)
Walc = clip_round(Dalc + rng.choice([0, 1, 2], size=N, p=[0.45, 0.35, 0.20]), 1, 5)
health = clip_round(rng.normal(3.5, 1.4, N), 1, 5)
absences = np.clip(np.round(rng.gamma(1.4, 4.2, N)), 0, 45).astype(int)

# --- hidden "ability" drives the marks ----------------------------------
ability = (
    0.55 * (studytime - 2.2)
    - 1.1 * failures
    - 0.07 * absences
    + 0.25 * (Medu + Fedu - 4) / 2
    + 0.5 * (higher == "yes")
    + 0.25 * (internet == "yes")
    + 0.3 * (schoolsup == "no")
    + 0.25 * (paid == "yes")
    - 0.18 * (goout - 3)
    - 0.12 * (Walc - 1.5)
    - 0.25 * (romantic == "yes")
    + 0.15 * (famsup == "yes")
    + rng.normal(0, 1.0, N)
)

base = 10.6 + 2.2 * ability
G1 = clip_round(base + rng.normal(0, 1.7, N), 0, 20)
G2 = clip_round(0.65 * G1 + 0.35 * base + rng.normal(0, 1.5, N), 0, 20)
G3 = clip_round(0.15 * G1 + 0.70 * G2 + 0.15 * base + rng.normal(0, 1.3, N), 0, 20)

# a few drop-outs / no-shows score 0 in the final exam (also seen in the real data)
dropout = (rng.random(N) < 0.035) & (G2 < 11)
G3 = np.where(dropout, 0, G3)

df = pd.DataFrame(dict(
    school=school, sex=sex, age=age, address=address, famsize=famsize, Pstatus=Pstatus,
    Medu=Medu, Fedu=Fedu, Mjob=Mjob, Fjob=Fjob, reason=reason, guardian=guardian,
    traveltime=traveltime, studytime=studytime, failures=failures, schoolsup=schoolsup,
    famsup=famsup, paid=paid, activities=activities, nursery=nursery, higher=higher,
    internet=internet, romantic=romantic, famrel=famrel, freetime=freetime, goout=goout,
    Dalc=Dalc, Walc=Walc, health=health, absences=absences, G1=G1, G2=G2, G3=G3,
))

# sprinkle a little missing data so the cleaning step is meaningful
for col in ["absences", "studytime", "famsup"]:
    idx = rng.choice(N, size=8, replace=False)
    df[col] = df[col].astype(object)
    df.loc[idx, col] = np.nan

df.to_csv("data/student_data.csv", index=False)
print("Saved data/student_data.csv", df.shape)
print("Pass rate (G3>=10):", round((G3 >= 10).mean(), 3))

"""Command-line predictor.  Example:  python predict.py
Edit the `student` dictionary below (or import predict_student in your own code)."""
import joblib
import pandas as pd

from common import MODEL_DIR, risk_level, suggestions


def predict_student(student: dict, early: bool = False) -> dict:
    scenario = "early" if early else "full"
    clf = joblib.load(MODEL_DIR / f"pass_fail_{scenario}.joblib")
    reg = joblib.load(MODEL_DIR / f"score_{scenario}.joblib")
    row = pd.DataFrame([student])[clf["num"] + clf["cat"]]
    p_pass = float(clf["model"].predict_proba(row)[0, 1])
    score = float(max(0, min(20, reg["model"].predict(row)[0])))
    return dict(
        prediction="Pass" if p_pass >= 0.5 else "Fail",
        pass_probability=round(p_pass, 3),
        predicted_final_grade=round(score, 1),
        risk_level=risk_level(p_pass),
        suggestions=suggestions({k: v for k, v in student.items() if not (early and k in ('G1', 'G2'))}),
    )


if __name__ == "__main__":
    student = dict(
        school="GP", sex="F", age=17, address="U", famsize="GT3", Pstatus="T", Medu=3, Fedu=2,
        Mjob="services", Fjob="other", reason="course", guardian="mother", traveltime=1,
        studytime=1, failures=1, schoolsup="no", famsup="yes", paid="no", activities="no",
        nursery="yes", higher="yes", internet="yes", romantic="no", famrel=4, freetime=3,
        goout=4, Dalc=1, Walc=3, health=3, absences=12, G1=9, G2=8,
    )
    for early in (False, True):
        print("EARLY (no G1/G2)" if early else "FULL (with G1/G2)")
        for k, v in predict_student(student, early).items():
            print(f"  {k}: {v}")

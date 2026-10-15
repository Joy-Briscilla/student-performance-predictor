"""Streamlit web app.  Run:  streamlit run app.py"""
import joblib
import pandas as pd
import streamlit as st

from common import FIG_DIR, MODEL_DIR, REPORT_DIR, risk_level, suggestions

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="wide")


@st.cache_resource
def load_models():
    return {
        s: (joblib.load(MODEL_DIR / f"pass_fail_{s}.joblib"), joblib.load(MODEL_DIR / f"score_{s}.joblib"))
        for s in ("full", "early")
    }


models = load_models()
st.title("🎓 Student Performance Predictor")
st.caption("Predicts pass/fail and the final grade (0-20) so at-risk students can be helped early.")

tab_predict, tab_insights = st.tabs(["Predict", "Model insights"])

with tab_predict:
    mode = st.radio(
        "Prediction mode",
        ["Early prediction (no internal marks yet)", "Full prediction (with G1 and G2 marks)"],
        horizontal=True,
    )
    early = mode.startswith("Early")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Background")
        school = st.selectbox("School", ["GP", "MS"])
        sex = st.selectbox("Sex", ["F", "M"])
        age = st.slider("Age", 15, 22, 17)
        address = st.selectbox("Home area", ["U", "R"], format_func=lambda x: "Urban" if x == "U" else "Rural")
        famsize = st.selectbox("Family size", ["GT3", "LE3"], format_func=lambda x: "> 3 members" if x == "GT3" else "<= 3 members")
        Pstatus = st.selectbox("Parents living", ["T", "A"], format_func=lambda x: "Together" if x == "T" else "Apart")
        Medu = st.slider("Mother's education (0-4)", 0, 4, 2)
        Fedu = st.slider("Father's education (0-4)", 0, 4, 2)
        Mjob = st.selectbox("Mother's job", ["teacher", "health", "services", "at_home", "other"])
        Fjob = st.selectbox("Father's job", ["teacher", "health", "services", "at_home", "other"])
        reason = st.selectbox("Reason for choosing school", ["home", "reputation", "course", "other"])
        guardian = st.selectbox("Guardian", ["mother", "father", "other"])
    with c2:
        st.subheader("Study habits")
        traveltime = st.slider("Travel time (1 = <15 min ... 4 = >1 h)", 1, 4, 1)
        studytime = st.slider("Weekly study time (1 = <2 h ... 4 = >10 h)", 1, 4, 2)
        failures = st.slider("Past class failures", 0, 3, 0)
        absences = st.slider("Absences", 0, 45, 4)
        schoolsup = st.selectbox("Extra school support", ["no", "yes"])
        famsup = st.selectbox("Family educational support", ["yes", "no"])
        paid = st.selectbox("Paid extra classes", ["no", "yes"])
        activities = st.selectbox("Extra-curricular activities", ["no", "yes"])
        higher = st.selectbox("Wants higher education", ["yes", "no"])
        internet = st.selectbox("Internet at home", ["yes", "no"])
    with c3:
        st.subheader("Lifestyle")
        nursery = st.selectbox("Attended nursery", ["yes", "no"])
        romantic = st.selectbox("In a relationship", ["no", "yes"])
        famrel = st.slider("Family relationship quality (1-5)", 1, 5, 4)
        freetime = st.slider("Free time after school (1-5)", 1, 5, 3)
        goout = st.slider("Going out with friends (1-5)", 1, 5, 3)
        Dalc = st.slider("Weekday alcohol (1-5)", 1, 5, 1)
        Walc = st.slider("Weekend alcohol (1-5)", 1, 5, 1)
        health = st.slider("Health status (1-5)", 1, 5, 4)
        G1 = G2 = None
        if not early:
            st.subheader("Internal marks (0-20)")
            G1 = st.slider("G1 (first period)", 0, 20, 10)
            G2 = st.slider("G2 (second period)", 0, 20, 10)

    if st.button("Predict", type="primary"):
        student = dict(
            school=school, sex=sex, age=age, address=address, famsize=famsize, Pstatus=Pstatus,
            Medu=Medu, Fedu=Fedu, Mjob=Mjob, Fjob=Fjob, reason=reason, guardian=guardian,
            traveltime=traveltime, studytime=studytime, failures=failures, schoolsup=schoolsup,
            famsup=famsup, paid=paid, activities=activities, nursery=nursery, higher=higher,
            internet=internet, romantic=romantic, famrel=famrel, freetime=freetime, goout=goout,
            Dalc=Dalc, Walc=Walc, health=health, absences=absences,
        )
        if not early:
            student.update(G1=G1, G2=G2)

        clf, reg = models["early" if early else "full"]
        row = pd.DataFrame([student])[clf["num"] + clf["cat"]]
        p_pass = float(clf["model"].predict_proba(row)[0, 1])
        score = float(max(0, min(20, reg["model"].predict(row)[0])))
        level = risk_level(p_pass)

        m1, m2, m3 = st.columns(3)
        m1.metric("Prediction", "PASS" if p_pass >= 0.5 else "FAIL")
        m2.metric("Pass probability", f"{p_pass:.0%}")
        m3.metric("Predicted final grade", f"{score:.1f} / 20")
        {"Low": st.success, "Medium": st.warning, "High": st.error}[level](f"Risk level: {level}")
        st.progress(min(max(p_pass, 0.0), 1.0))

        st.subheader("Suggested actions")
        for tip in suggestions(student):
            st.write("• " + tip)

with tab_insights:
    st.subheader("What drives student results?")
    left, right = st.columns(2)
    for img, col in [("importance_full.png", left), ("importance_early.png", right),
                     ("03_correlation_heatmap.png", left), ("roc_early.png", right)]:
        if (FIG_DIR / img).exists():
            col.image(str(FIG_DIR / img))
    for s in ("full", "early"):
        f = REPORT_DIR / f"classification_comparison_{s}.csv"
        if f.exists():
            st.markdown(f"**Model comparison ({s} features)**")
            st.dataframe(pd.read_csv(f).round(3), use_container_width=True)

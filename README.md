# Student Performance Predictor

## NEW: the full website (student portal + staff dashboard)
```bash
pip install -r requirements.txt
python train_model.py        # once: trains and saves the ML models
python seed_demo.py          # optional: creates demo accounts and sample students
python web_app.py            # open http://127.0.0.1:5000
```
Demo logins (after `seed_demo.py`): staff **T001 / staff123** (class ECE-A), students **S001 ... S020 / student123**.
New accounts: use the Register page. Staff registration needs the code `STAFF2026` (change it with the `STAFF_CODE` environment variable; also set `SECRET_KEY`).

| Part | What it does |
|---|---|
| Student portal | Enters study hours, lifestyle and unit-wise "how well studied" (0-5). Can update again any time. Sees pass %, grade, risk, progress chart, weak units and advice. Cannot edit marks. |
| Staff dashboard | Sees only their own class. Class pass/fail ratio, risk distribution, average marks per subject, searchable and filterable table (risk, weak subject), at-risk students first, CSV export. |
| Staff student page | Staff enter or verify G1, G2, absences, past failures and unit-wise marks (0-20). Saving re-runs the prediction. |
| ML model | Random Forest / best model from `train_model.py`. FULL mode when G1 and G2 are verified, otherwise EARLY mode. |
| Advice | Rule-based on top of the ML result: weak units, marks needed to pass, units still to cover, attendance, study time, past failures. |

Files: `web_app.py` (routes), `database.py` (SQLite), `engine.py` (prediction + advice + chart), `templates/`, `static/style.css`, `seed_demo.py`.
Note: the ML model learns from the training dataset (overall marks and habits). Unit-wise marks are used by the advice rules, not by the model.
Before real use: train on the real UCI dataset (see below), do NOT run `seed_demo.py` (it is fake demo data and it resets the database), and run behind HTTPS with a strong `SECRET_KEY`.

---

## Original ML project (also includes the older single-page Streamlit app)

A machine-learning project that predicts whether a student will **pass or fail** (classification) and their **final grade out of 20** (regression), so teachers can spot at-risk students early and act before the exam.

## Project structure
```
student_performance_predictor/
├── data/student_data.csv        # dataset (demo data in the UCI schema, see note below)
├── generate_dataset.py          # creates the demo dataset
├── common.py                    # loading, cleaning, risk levels, advice rules
├── train_model.py               # full ML pipeline: EDA, training, tuning, evaluation, saving
├── predict.py                   # command-line prediction
├── app.py                       # Streamlit web app
├── models/                      # trained models (.joblib)
├── figures/                     # charts used in the report
├── reports/                     # metrics.json and comparison tables (.csv)
└── requirements.txt
```

## How to run
```bash
pip install -r requirements.txt
python train_model.py          # trains and saves models, charts and metrics
python predict.py              # quick command-line test
streamlit run app.py           # opens the web app in the browser
```
Google Colab: upload the folder, run `!pip install streamlit`, then `!python train_model.py`.

## IMPORTANT: using the real dataset
The included `data/student_data.csv` is a **synthetic demo dataset** that follows the same 33 columns as the public UCI *Student Performance* dataset, so the project runs out of the box.
For your final submission, replace it with the real data:
1. Download **student-mat.csv** (UCI Machine Learning Repository, "Student Performance", or the Kaggle copy).
2. Save it as `data/student_data.csv` (the `;` separator is detected automatically).
3. Run `python train_model.py` again and use the new numbers in your report.

## Method in short
1. **Target:** Pass = final grade G3 >= 10. Regression target = G3.
2. **Cleaning:** remove duplicates, fix types, fill missing values (median / mode), clip marks to 0-20.
3. **Preprocessing:** StandardScaler for numbers, One-Hot encoding for categories (inside a scikit-learn Pipeline, so there is no data leakage).
4. **Models compared:** Logistic Regression, Decision Tree, Random Forest, SVM, Gradient Boosting (classification); Linear Regression, Random Forest, Gradient Boosting (regression).
5. **Evaluation:** 80/20 stratified split, 5-fold cross-validation, accuracy, precision, recall, F1, ROC-AUC, confusion matrix; R2, RMSE and MAE for regression.
6. **Tuning:** GridSearchCV on the two best model families.
7. **Explainability:** permutation feature importance.
8. **Two modes:**
   - **Full** uses all features including G1 and G2 (most accurate).
   - **Early** hides G1 and G2, so a student can be flagged before any internal test (more useful in practice).
9. **Risk level:** P(pass) >= 75% Low, 50-75% Medium, < 50% High, plus rule-based advice.

## Future scope
Attendance capture with RFID/IoT, LMS integration, SHAP explanations, a teacher dashboard with login, and retraining each semester.

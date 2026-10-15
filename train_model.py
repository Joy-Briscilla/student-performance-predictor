"""
Student Performance Predictor - training pipeline.

Run:  python train_model.py
Produces: trained models (models/), charts (figures/), metrics (reports/).
"""
import json
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,
                             f1_score, mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from common import (FIG_DIR, MODEL_DIR, PASS_MARK, REPORT_DIR, TARGET, clean_data,
                    feature_lists, load_data)

warnings.filterwarnings("ignore")
SEED = 42
sns.set_theme(style="whitegrid")


def make_preprocessor(num_cols, cat_cols):
    return ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])


# ------------------------------------------------------------------ EDA
def run_eda(df):
    FIG_DIR.mkdir(exist_ok=True)
    pass_flag = np.where(df[TARGET] >= PASS_MARK, "Pass", "Fail")

    plt.figure(figsize=(7, 4))
    sns.histplot(df[TARGET], bins=21, kde=True, color="#2f6fdb")
    plt.axvline(PASS_MARK, color="red", ls="--", label=f"Pass mark = {PASS_MARK}")
    plt.title("Distribution of final grade (G3)")
    plt.xlabel("G3 (0-20)")
    plt.legend()
    plt.tight_layout(); plt.savefig(FIG_DIR / "01_grade_distribution.png", dpi=150); plt.close()

    plt.figure(figsize=(4.5, 4))
    sns.countplot(x=pass_flag, order=["Pass", "Fail"], palette=["#3aa76d", "#e0524d"])
    plt.title("Pass vs Fail count")
    plt.tight_layout(); plt.savefig(FIG_DIR / "02_pass_fail_count.png", dpi=150); plt.close()

    num_df = df.select_dtypes("number")
    plt.figure(figsize=(10, 8))
    sns.heatmap(num_df.corr(), cmap="coolwarm", center=0, annot=False)
    plt.title("Correlation heatmap")
    plt.tight_layout(); plt.savefig(FIG_DIR / "03_correlation_heatmap.png", dpi=150); plt.close()

    plt.figure(figsize=(7, 4))
    sns.scatterplot(data=df, x="absences", y=TARGET, alpha=0.5, color="#2f6fdb")
    sns.regplot(data=df, x="absences", y=TARGET, scatter=False, color="red")
    plt.title("Absences vs final grade")
    plt.tight_layout(); plt.savefig(FIG_DIR / "04_absences_vs_grade.png", dpi=150); plt.close()

    plt.figure(figsize=(7, 4))
    sns.boxplot(data=df, x="studytime", y=TARGET, palette="Blues")
    plt.title("Study time (1 = <2h ... 4 = >10h per week) vs final grade")
    plt.tight_layout(); plt.savefig(FIG_DIR / "05_studytime_vs_grade.png", dpi=150); plt.close()

    plt.figure(figsize=(7, 4))
    sns.boxplot(data=df, x="failures", y=TARGET, palette="Reds")
    plt.title("Past failures vs final grade")
    plt.tight_layout(); plt.savefig(FIG_DIR / "06_failures_vs_grade.png", dpi=150); plt.close()


# ------------------------------------------------------------ classification
def classifier_zoo(pre):
    return {
        "Logistic Regression": Pipeline([("pre", pre), ("m", LogisticRegression(max_iter=2000, random_state=SEED))]),
        "Decision Tree": Pipeline([("pre", pre), ("m", DecisionTreeClassifier(max_depth=5, random_state=SEED))]),
        "Random Forest": Pipeline([("pre", pre), ("m", RandomForestClassifier(n_estimators=300, random_state=SEED))]),
        "SVM (RBF)": Pipeline([("pre", pre), ("m", SVC(probability=True, random_state=SEED))]),
        "Gradient Boosting": Pipeline([("pre", pre), ("m", GradientBoostingClassifier(random_state=SEED))]),
    }


TUNE_GRIDS = {
    "Logistic Regression": {"m__C": [0.05, 0.1, 0.5, 1, 5]},
    "Decision Tree": {"m__max_depth": [3, 4, 5, 7], "m__min_samples_leaf": [2, 5, 10]},
    "Random Forest": {"m__n_estimators": [200, 400], "m__max_depth": [None, 6, 10], "m__min_samples_leaf": [1, 3]},
    "SVM (RBF)": {"m__C": [0.5, 1, 5], "m__gamma": ["scale", 0.01]},
    "Gradient Boosting": {"m__n_estimators": [100, 200], "m__learning_rate": [0.05, 0.1], "m__max_depth": [2, 3]},
}


def run_classification(df, scenario):
    num, cat = feature_lists(scenario)
    X = df[num + cat]
    y = (df[TARGET] >= PASS_MARK).astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    pre = make_preprocessor(num, cat)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    rows, fitted = [], {}
    for name, pipe in classifier_zoo(pre).items():
        cv_acc = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="accuracy").mean()
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        proba = pipe.predict_proba(X_te)[:, 1]
        rows.append(dict(
            Model=name, CV_Accuracy=cv_acc,
            Accuracy=accuracy_score(y_te, pred), Precision=precision_score(y_te, pred),
            Recall=recall_score(y_te, pred), F1=f1_score(y_te, pred), ROC_AUC=roc_auc_score(y_te, proba)))
        fitted[name] = pipe
    results = pd.DataFrame(rows).sort_values("CV_Accuracy", ascending=False).reset_index(drop=True)

    # tune the best two model families, keep the better one
    best_pipe, best_name, best_cv = None, None, -1
    for name in results["Model"].head(2):
        gs = GridSearchCV(classifier_zoo(pre)[name], TUNE_GRIDS[name], cv=cv, scoring="f1", n_jobs=-1)
        gs.fit(X_tr, y_tr)
        if gs.best_score_ > best_cv:
            best_pipe, best_name, best_cv, best_params = gs.best_estimator_, name, gs.best_score_, gs.best_params_

    pred = best_pipe.predict(X_te)
    proba = best_pipe.predict_proba(X_te)[:, 1]
    final = dict(
        model=best_name, best_params={k: (v if v is not None else "None") for k, v in best_params.items()},
        accuracy=accuracy_score(y_te, pred), precision=precision_score(y_te, pred),
        recall=recall_score(y_te, pred), f1=f1_score(y_te, pred), roc_auc=roc_auc_score(y_te, proba),
        tuned_cv_f1=best_cv, test_size=int(len(y_te)), train_size=int(len(y_tr)))

    # figures
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ConfusionMatrixDisplay.from_predictions(y_te, pred, display_labels=["Fail", "Pass"], cmap="Blues", ax=ax)
    ax.set_title(f"Confusion matrix ({scenario}): {best_name}")
    plt.tight_layout(); plt.savefig(FIG_DIR / f"cm_{scenario}.png", dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for name, p in fitted.items():
        RocCurveDisplay.from_estimator(p, X_te, y_te, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_title(f"ROC curves ({scenario} features)")
    plt.tight_layout(); plt.savefig(FIG_DIR / f"roc_{scenario}.png", dpi=150); plt.close()

    imp = permutation_importance(best_pipe, X_te, y_te, n_repeats=15, random_state=SEED, scoring="f1")
    imp_df = (pd.DataFrame({"feature": X.columns, "importance": imp.importances_mean})
              .sort_values("importance", ascending=False))
    top = imp_df.head(12).iloc[::-1]
    plt.figure(figsize=(7, 5))
    plt.barh(top["feature"], top["importance"], color="#2f6fdb")
    plt.title(f"Top factors affecting pass/fail ({scenario})")
    plt.xlabel("Drop in F1 when the feature is shuffled")
    plt.tight_layout(); plt.savefig(FIG_DIR / f"importance_{scenario}.png", dpi=150); plt.close()

    # final model re-fit on all data for deployment
    best_pipe.fit(X, y)
    joblib.dump(dict(model=best_pipe, num=num, cat=cat, scenario=scenario), MODEL_DIR / f"pass_fail_{scenario}.joblib")
    return results, final, imp_df


# ---------------------------------------------------------------- regression
def run_regression(df, scenario):
    num, cat = feature_lists(scenario)
    X, y = df[num + cat], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)
    pre = make_preprocessor(num, cat)
    zoo = {
        "Linear Regression": Pipeline([("pre", pre), ("m", LinearRegression())]),
        "Random Forest": Pipeline([("pre", pre), ("m", RandomForestRegressor(n_estimators=300, random_state=SEED))]),
        "Gradient Boosting": Pipeline([("pre", pre), ("m", GradientBoostingRegressor(random_state=SEED))]),
    }
    rows, best, best_r2 = [], None, -9
    for name, pipe in zoo.items():
        pipe.fit(X_tr, y_tr)
        p = np.clip(pipe.predict(X_te), 0, 20)
        r2 = r2_score(y_te, p)
        rows.append(dict(Model=name, R2=r2, RMSE=mean_squared_error(y_te, p) ** 0.5, MAE=mean_absolute_error(y_te, p)))
        if r2 > best_r2:
            best, best_name, best_r2, best_pred = pipe, name, r2, p
    res = pd.DataFrame(rows).sort_values("R2", ascending=False).reset_index(drop=True)

    plt.figure(figsize=(5, 5))
    plt.scatter(y_te, best_pred, alpha=0.5, color="#2f6fdb")
    plt.plot([0, 20], [0, 20], "r--")
    plt.xlabel("Actual G3"); plt.ylabel("Predicted G3")
    plt.title(f"Actual vs predicted ({scenario}): {best_name}")
    plt.tight_layout(); plt.savefig(FIG_DIR / f"reg_{scenario}.png", dpi=150); plt.close()

    best.fit(X, y)
    joblib.dump(dict(model=best, num=num, cat=cat, scenario=scenario), MODEL_DIR / f"score_{scenario}.joblib")
    return res, dict(model=best_name, r2=res.iloc[0]["R2"], rmse=res.iloc[0]["RMSE"], mae=res.iloc[0]["MAE"])


# ---------------------------------------------------------------------- main
def main():
    MODEL_DIR.mkdir(exist_ok=True); REPORT_DIR.mkdir(exist_ok=True)
    df, clean_report = clean_data(load_data())
    print("Cleaning report:", clean_report)
    run_eda(df)

    summary = {"cleaning": clean_report, "pass_rate": float((df[TARGET] >= PASS_MARK).mean())}
    for scenario in ["full", "early"]:
        print(f"\n=== Scenario: {scenario} ===")
        cls_table, cls_final, imp = run_classification(df, scenario)
        reg_table, reg_final = run_regression(df, scenario)
        print(cls_table.round(3).to_string(index=False))
        print("Tuned best classifier:", cls_final)
        print(reg_table.round(3).to_string(index=False))
        cls_table.to_csv(REPORT_DIR / f"classification_comparison_{scenario}.csv", index=False)
        reg_table.to_csv(REPORT_DIR / f"regression_comparison_{scenario}.csv", index=False)
        imp.to_csv(REPORT_DIR / f"feature_importance_{scenario}.csv", index=False)
        summary[scenario] = dict(classification=cls_final, regression=reg_final)

    with open(REPORT_DIR / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print("\nDone. Models in models/, charts in figures/, metrics in reports/.")


if __name__ == "__main__":
    main()

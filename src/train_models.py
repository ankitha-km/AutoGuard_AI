"""
AutoGuard AI - Model Training & Evaluation
Trains Random Forest, Decision Tree, and XGBoost classifiers,
evaluates them, selects the best model, and saves artifacts.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from data_preprocessing import load_raw_data, preprocess_pipeline, FEATURE_COLUMNS, TARGET_COLUMN

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_PATH = os.path.join(BASE_DIR, "data", "vehicle_sensor_data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)


def evaluate_model(name, model, X_test, y_test, scaled=False, scaler=None):
    X_eval = X_test
    y_pred = model.predict(X_eval)
    y_proba = model.predict_proba(X_eval)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1_Score": f1_score(y_test, y_pred),
        "ROC_AUC": roc_auc_score(y_test, y_proba),
    }

    cm = confusion_matrix(y_test, y_pred)

    print(f"\n===== {name} =====")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print("Confusion Matrix:\n", cm)
    print(classification_report(y_test, y_pred, target_names=["Healthy", "Failure"]))

    return metrics, cm, y_pred


def plot_confusion_matrix(cm, name, ax=None):
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Healthy", "Failure"], yticklabels=["Healthy", "Failure"])
    ax.set_title(f"{name} - Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    if standalone:
        plt.tight_layout()
        plt.savefig(os.path.join(ASSETS_DIR, f"confusion_matrix_{name.replace(' ', '_').lower()}.png"), dpi=120)
        plt.close()


def main():
    print("Loading and preprocessing data...")
    raw = load_raw_data(DATA_PATH)
    df = preprocess_pipeline(raw)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features (helps Decision Tree/RF less, but keeps pipeline consistent;
    # XGBoost and tree models are scale-invariant, scaling mainly helps reproducibility)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}
    models = {}
    confusion_matrices = {}
    predictions = {}

    # ---------------- Decision Tree ----------------
    dt = DecisionTreeClassifier(max_depth=8, min_samples_leaf=10, random_state=42, class_weight="balanced")
    dt.fit(X_train, y_train)
    metrics, cm, y_pred = evaluate_model("Decision Tree", dt, X_test, y_test)
    results["Decision Tree"] = metrics
    models["Decision Tree"] = dt
    confusion_matrices["Decision Tree"] = cm
    predictions["Decision Tree"] = y_pred

    # ---------------- Random Forest ----------------
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_leaf=5,
        random_state=42, class_weight="balanced", n_jobs=-1
    )
    rf.fit(X_train, y_train)
    metrics, cm, y_pred = evaluate_model("Random Forest", rf, X_test, y_test)
    results["Random Forest"] = metrics
    models["Random Forest"] = rf
    confusion_matrices["Random Forest"] = cm
    predictions["Random Forest"] = y_pred

    # ---------------- XGBoost ----------------
    if XGBOOST_AVAILABLE:
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        xgb = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.9,
            scale_pos_weight=scale_pos_weight,
            random_state=42, eval_metric="logloss"
        )
        xgb.fit(X_train, y_train)
        metrics, cm, y_pred = evaluate_model("XGBoost", xgb, X_test, y_test)
        results["XGBoost"] = metrics
        models["XGBoost"] = xgb
        confusion_matrices["XGBoost"] = cm
        predictions["XGBoost"] = y_pred
    else:
        print("\nXGBoost not available - skipping.")

    # ---------------- Comparison & Best Model Selection ----------------
    results_df = pd.DataFrame(results).T
    print("\n========== MODEL COMPARISON ==========")
    print(results_df.round(4))

    # Select best model based on F1 Score (balances precision/recall - important for failure detection)
    best_model_name = results_df["F1_Score"].idxmax()
    best_model = models[best_model_name]
    print(f"\n>>> BEST MODEL SELECTED: {best_model_name} (F1 = {results_df.loc[best_model_name, 'F1_Score']:.4f})")

    # Save all confusion matrices in one figure
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
    if n_models == 1:
        axes = [axes]
    for ax, (name, cm) in zip(axes, confusion_matrices.items()):
        plot_confusion_matrix(cm, name, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, "all_confusion_matrices.png"), dpi=120)
    plt.close()

    # Save model comparison bar chart
    plt.figure(figsize=(9, 5))
    results_df[["Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC"]].plot(kind="bar")
    plt.title("Model Performance Comparison")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=0)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, "model_comparison.png"), dpi=120)
    plt.close()

    # Feature importance for best model
    if hasattr(best_model, "feature_importances_"):
        importances = pd.Series(best_model.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)
        plt.figure(figsize=(8, 6))
        sns.barplot(x=importances.values, y=importances.index, palette="viridis")
        plt.title(f"Feature Importance - {best_model_name}")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(ASSETS_DIR, "feature_importance.png"), dpi=120)
        plt.close()
        importances.to_csv(os.path.join(MODELS_DIR, "feature_importance.csv"))

    # ---------------- Save Artifacts ----------------
    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))

    # Save all models for comparison/demo purposes
    for name, model in models.items():
        fname = name.replace(" ", "_").lower() + ".pkl"
        joblib.dump(model, os.path.join(MODELS_DIR, fname))

    # Save metadata
    metadata = {
        "best_model": best_model_name,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "metrics": results,
        "xgboost_available": XGBOOST_AVAILABLE,
    }
    with open(os.path.join(MODELS_DIR, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, default=float)

    print("\nAll artifacts saved to:", MODELS_DIR)
    print("Saved files:", os.listdir(MODELS_DIR))


if __name__ == "__main__":
    main()

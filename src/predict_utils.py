"""
AutoGuard AI - Prediction Utilities
Computes Vehicle Health Score, Failure Risk, RUL Estimate,
Maintenance Recommendations, and Explainability outputs.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from data_preprocessing import preprocess_pipeline, FEATURE_COLUMNS

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
MODELS_DIR = os.path.join(BASE_DIR, "models")


def load_artifacts():
    model = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    with open(os.path.join(MODELS_DIR, "model_metadata.json")) as f:
        metadata = json.load(f)
    return model, scaler, metadata


def compute_health_score(row: pd.Series) -> float:
    """
    Vehicle Health Score (0-100, higher = healthier).
    Derived as the inverse of the Composite Risk Score (0-100 scale).
    """
    risk = row["Composite_Risk_Score"]
    health = 100 - min(risk, 100)
    return round(max(health, 0), 1)


def estimate_rul(row: pd.Series, failure_probability: float) -> dict:
    """
    Estimate Remaining Useful Life in days/km using a heuristic decay model
    based on risk factors and failure probability.

    This is a simplified engineering heuristic suitable for a PoC:
    RUL decreases non-linearly as failure probability increases,
    and is adjusted by wear indicators (mileage, age, oil quality).
    """
    max_rul_days = 365  # Healthy vehicle baseline

    # Base decay from failure probability (exponential decay)
    prob_factor = np.exp(-4 * failure_probability)  # 1.0 at p=0, ~0.018 at p=1

    # Wear adjustment - higher wear shortens RUL further
    wear_penalty = 1 - min(row["Wear_Index"] / 5, 0.6)

    # Oil quality adjustment
    oil_factor = 0.5 + 0.5 * (row["Oil_Quality_Index"] / 100)

    rul_days = max_rul_days * prob_factor * wear_penalty * oil_factor
    rul_days = max(rul_days, 1)

    # Convert to estimated km (assuming average 40 km/day usage)
    rul_km = rul_days * 40

    return {
        "rul_days": round(rul_days, 1),
        "rul_km": round(rul_km, 0),
    }


def get_maintenance_recommendation(row: pd.Series, failure_probability: float, rul: dict) -> dict:
    """Generate human-readable maintenance recommendations based on sensor states."""
    issues = []
    actions = []

    if row["Overheating_Flag"] == 1 or row["Engine_Temperature"] > 100:
        issues.append("Engine overheating detected")
        actions.append("Inspect cooling system, coolant levels, and radiator fan immediately")

    if row["High_Vibration_Flag"] == 1 or row["Vibration"] > 4.5:
        issues.append("Abnormal vibration levels")
        actions.append("Check engine mounts, wheel balance/alignment, and drivetrain components")

    if row["Battery_Weak_Flag"] == 1 or row["Battery_Voltage"] < 11.8:
        issues.append("Low battery voltage")
        actions.append("Test battery health and alternator output; consider battery replacement")

    if row["Oil_Degraded"] == 1 or row["Oil_Quality_Index"] < 50:
        issues.append("Degraded engine oil quality")
        actions.append("Schedule an oil change and filter replacement")

    if row["Engine_Load"] > 85:
        issues.append("Sustained high engine load")
        actions.append("Review driving patterns / load distribution to reduce engine strain")

    if row["Wear_Index"] > 1.5:
        issues.append("High cumulative wear (mileage/age)")
        actions.append("Schedule a comprehensive preventive maintenance inspection")

    # Determine urgency
    if failure_probability >= 0.7 or rul["rul_days"] < 15:
        urgency = "CRITICAL"
        urgency_msg = "Immediate workshop visit recommended within 1-3 days."
    elif failure_probability >= 0.4 or rul["rul_days"] < 60:
        urgency = "HIGH"
        urgency_msg = "Schedule maintenance within 1-2 weeks."
    elif failure_probability >= 0.2:
        urgency = "MODERATE"
        urgency_msg = "Plan a routine inspection within the next month."
    else:
        urgency = "LOW"
        urgency_msg = "No immediate action needed. Continue regular maintenance schedule."

    if not issues:
        issues.append("All monitored parameters within normal range")
        actions.append("Continue routine maintenance schedule")

    return {
        "urgency": urgency,
        "urgency_message": urgency_msg,
        "issues": issues,
        "actions": actions,
    }


def get_feature_explanations(model, scaler, X_row: pd.DataFrame, feature_columns=FEATURE_COLUMNS, top_n=5):
    """
    Explainability: returns top contributing features for this prediction
    using the model's feature importances combined with how far each
    feature value deviates from the 'healthy' baseline (z-score style).
    """
    importances = pd.Series(model.feature_importances_, index=feature_columns)

    # Standardize the row using the saved scaler to get deviation magnitude
    X_scaled = scaler.transform(X_row[feature_columns])
    deviations = pd.Series(np.abs(X_scaled[0]), index=feature_columns)

    # Combine importance and deviation -> contribution score
    contribution = (importances * deviations).sort_values(ascending=False)

    top_features = contribution.head(top_n)

    explanations = []
    readable_names = {
        "Engine_Temperature": "Engine Temperature",
        "Vibration": "Vibration Level",
        "Battery_Voltage": "Battery Voltage",
        "Engine_Load": "Engine Load",
        "RPM": "Engine RPM",
        "Oil_Quality_Index": "Oil Quality Index",
        "Mileage_km": "Mileage",
        "Vehicle_Age_Years": "Vehicle Age",
        "Temp_Deviation": "Temperature Deviation from Optimal",
        "Vibration_Severity": "Vibration Severity Ratio",
        "Voltage_Drop": "Battery Voltage Drop",
        "Load_Temp_Interaction": "Load-Temperature Stress",
        "Wear_Index": "Overall Wear Index",
        "Oil_Degraded": "Oil Degradation Flag",
        "High_Vibration_Flag": "High Vibration Flag",
        "Battery_Weak_Flag": "Weak Battery Flag",
        "Overheating_Flag": "Overheating Flag",
    }

    for feat, score in top_features.items():
        explanations.append({
            "feature": readable_names.get(feat, feat),
            "value": round(float(X_row[feat].values[0]), 2),
            "contribution_score": round(float(score), 4),
        })

    return explanations


def predict_single(input_dict: dict, model=None, scaler=None, metadata=None):
    """
    Full prediction pipeline for a single vehicle reading.
    input_dict should contain raw sensor values:
        Engine_Temperature, Vibration, Battery_Voltage, Engine_Load,
        RPM, Oil_Quality_Index, Mileage_km, Vehicle_Age_Years
    """
    if model is None or scaler is None or metadata is None:
        model, scaler, metadata = load_artifacts()

    raw_df = pd.DataFrame([input_dict])

    # Ensure required columns exist with sensible defaults
    defaults = {
        "RPM": 2200, "Oil_Quality_Index": 75, "Mileage_km": 40000, "Vehicle_Age_Years": 3
    }
    for col, default_val in defaults.items():
        if col not in raw_df.columns:
            raw_df[col] = default_val

    processed = preprocess_pipeline(raw_df)
    X = processed[FEATURE_COLUMNS]

    failure_proba = model.predict_proba(X)[0, 1]
    failure_pred = int(failure_proba >= 0.5)

    row = processed.iloc[0]
    health_score = compute_health_score(row)
    rul = estimate_rul(row, failure_proba)
    recommendation = get_maintenance_recommendation(row, failure_proba, rul)
    explanations = get_feature_explanations(model, scaler, X)

    return {
        "failure_probability": round(float(failure_proba), 4),
        "failure_prediction": failure_pred,
        "health_score": health_score,
        "rul": rul,
        "recommendation": recommendation,
        "explanations": explanations,
        "processed_row": row,
    }


def predict_batch(df: pd.DataFrame, model=None, scaler=None, metadata=None):
    """Run predictions on a batch dataframe of raw sensor readings."""
    if model is None or scaler is None or metadata is None:
        model, scaler, metadata = load_artifacts()

    defaults = {
        "RPM": 2200, "Oil_Quality_Index": 75, "Mileage_km": 40000, "Vehicle_Age_Years": 3
    }
    df = df.copy()
    for col, default_val in defaults.items():
        if col not in df.columns:
            df[col] = default_val

    processed = preprocess_pipeline(df)
    X = processed[FEATURE_COLUMNS]

    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= 0.5).astype(int)

    results = processed.copy()
    results["Failure_Probability"] = np.round(proba, 4)
    results["Failure_Prediction"] = preds
    results["Health_Score"] = results.apply(compute_health_score, axis=1)

    rul_list = [estimate_rul(results.iloc[i], proba[i]) for i in range(len(results))]
    results["RUL_Days"] = [r["rul_days"] for r in rul_list]
    results["RUL_Km"] = [r["rul_km"] for r in rul_list]

    return results


if __name__ == "__main__":
    # Quick smoke test
    sample_healthy = {
        "Engine_Temperature": 85, "Vibration": 1.8, "Battery_Voltage": 13.2,
        "Engine_Load": 45, "RPM": 2100, "Oil_Quality_Index": 85,
        "Mileage_km": 25000, "Vehicle_Age_Years": 2
    }
    sample_risky = {
        "Engine_Temperature": 108, "Vibration": 6.5, "Battery_Voltage": 11.2,
        "Engine_Load": 90, "RPM": 3200, "Oil_Quality_Index": 35,
        "Mileage_km": 180000, "Vehicle_Age_Years": 9
    }

    for name, sample in [("Healthy Vehicle", sample_healthy), ("At-Risk Vehicle", sample_risky)]:
        print(f"\n===== {name} =====")
        result = predict_single(sample)
        print(f"Failure Probability: {result['failure_probability']:.2%}")
        print(f"Health Score: {result['health_score']}/100")
        print(f"RUL: {result['rul']['rul_days']} days (~{result['rul']['rul_km']} km)")
        print(f"Urgency: {result['recommendation']['urgency']} - {result['recommendation']['urgency_message']}")
        print("Issues:", result['recommendation']['issues'])
        print("Top contributing factors:")
        for exp in result['explanations']:
            print(f"  - {exp['feature']}: value={exp['value']}, contribution={exp['contribution_score']}")

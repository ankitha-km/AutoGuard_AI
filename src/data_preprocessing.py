"""
AutoGuard AI - Data Preprocessing & Feature Engineering
Handles cleaning, missing values, outlier treatment, and feature creation.
"""

import pandas as pd
import numpy as np


NUMERIC_COLS = [
    "Engine_Temperature",
    "Vibration",
    "Battery_Voltage",
    "Engine_Load",
    "RPM",
    "Oil_Quality_Index",
    "Mileage_km",
    "Vehicle_Age_Years",
]


def load_raw_data(path):
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean dataset: handle missing values and cap outliers."""
    df = df.copy()

    # 1. Handle missing values via median imputation (robust to outliers)
    for col in NUMERIC_COLS:
        if col in df.columns and df[col].isna().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    # 2. Cap extreme outliers using IQR method (sensor glitches)
    for col in ["Engine_Temperature", "Vibration", "Battery_Voltage"]:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 3 * iqr
        upper = q3 + 3 * iqr
        df[col] = df[col].clip(lower=lower, upper=upper)

    # 3. Drop duplicates
    df = df.drop_duplicates()

    # 4. Reset index
    df = df.reset_index(drop=True)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create domain-informed engineered features."""
    df = df.copy()

    # Temperature deviation from ideal operating point (90C)
    df["Temp_Deviation"] = (df["Engine_Temperature"] - 90).abs()

    # Vibration severity ratio (relative to safe threshold of 3.0 mm/s)
    df["Vibration_Severity"] = df["Vibration"] / 3.0

    # Voltage drop from nominal 13.5V
    df["Voltage_Drop"] = (13.5 - df["Battery_Voltage"]).clip(lower=0)

    # Load-to-temperature interaction (high load + high temp = stress)
    df["Load_Temp_Interaction"] = df["Engine_Load"] * df["Engine_Temperature"] / 100

    # Wear index: combines mileage and age
    df["Wear_Index"] = (df["Mileage_km"] / 100000) + (df["Vehicle_Age_Years"] / 10)

    # Oil degradation flag
    df["Oil_Degraded"] = (df["Oil_Quality_Index"] < 50).astype(int)

    # High vibration flag
    df["High_Vibration_Flag"] = (df["Vibration"] > 4.5).astype(int)

    # Battery weak flag
    df["Battery_Weak_Flag"] = (df["Battery_Voltage"] < 11.8).astype(int)

    # Overheating flag
    df["Overheating_Flag"] = (df["Engine_Temperature"] > 100).astype(int)

    # Composite Risk Score (0-100, used for health score display)
    # Uses FIXED normalization constants (not dataset-relative) so the score
    # is consistent for both batch processing and single-row predictions.
    TEMP_DEV_NORM = 30.0      # >30C deviation from optimal = max risk contribution
    VOLTAGE_DROP_NORM = 3.0   # >3V drop from nominal = max risk contribution
    WEAR_INDEX_NORM = 3.0     # wear index of 3 = max risk contribution

    base_risk = (
        0.25 * (df["Temp_Deviation"].clip(upper=TEMP_DEV_NORM) / TEMP_DEV_NORM)
        + 0.25 * (df["Vibration_Severity"].clip(upper=3) / 3)
        + 0.20 * (df["Voltage_Drop"].clip(upper=VOLTAGE_DROP_NORM) / VOLTAGE_DROP_NORM)
        + 0.15 * (df["Wear_Index"].clip(upper=WEAR_INDEX_NORM) / WEAR_INDEX_NORM)
        + 0.15 * (1 - df["Oil_Quality_Index"] / 100)
    ) * 100

    # Critical-flag penalty: each active critical flag adds extra risk points
    flag_penalty = (
        df["Overheating_Flag"] * 12
        + df["High_Vibration_Flag"] * 10
        + df["Battery_Weak_Flag"] * 10
        + df["Oil_Degraded"] * 6
    )

    df["Composite_Risk_Score"] = (base_risk + flag_penalty).clip(upper=100)

    return df


FEATURE_COLUMNS = [
    "Engine_Temperature",
    "Vibration",
    "Battery_Voltage",
    "Engine_Load",
    "RPM",
    "Oil_Quality_Index",
    "Mileage_km",
    "Vehicle_Age_Years",
    "Temp_Deviation",
    "Vibration_Severity",
    "Voltage_Drop",
    "Load_Temp_Interaction",
    "Wear_Index",
    "Oil_Degraded",
    "High_Vibration_Flag",
    "Battery_Weak_Flag",
    "Overheating_Flag",
]

TARGET_COLUMN = "Failure_Status"


def preprocess_pipeline(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Full pipeline: clean -> engineer features."""
    df = clean_data(raw_df)
    df = engineer_features(df)
    return df


if __name__ == "__main__":
    df = load_raw_data("../data/vehicle_sensor_data.csv")
    print("Raw shape:", df.shape)
    print("Missing values:\n", df.isna().sum())

    df_clean = preprocess_pipeline(df)
    print("\nProcessed shape:", df_clean.shape)
    print("Missing values after cleaning:", df_clean.isna().sum().sum())
    print("\nNew columns:", [c for c in df_clean.columns if c not in df.columns])
    df_clean.to_csv("../data/vehicle_data_processed.csv", index=False)
    print("\nSaved processed dataset.")

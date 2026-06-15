"""
AutoGuard AI - Exploratory Data Analysis & Visualization
Generates charts saved to /assets for use in README and presentation slides.
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

from data_preprocessing import load_raw_data, preprocess_pipeline

sns.set_theme(style="whitegrid", palette="viridis")

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)


def main():
    raw = load_raw_data(os.path.join(os.path.dirname(__file__), "..", "data", "vehicle_sensor_data.csv"))
    df = preprocess_pipeline(raw)

    # 1. Failure distribution
    plt.figure(figsize=(5, 4))
    sns.countplot(x="Failure_Status", data=df, palette=["#2ecc71", "#e74c3c"])
    plt.title("Failure Status Distribution")
    plt.xlabel("Failure Status (0 = Healthy, 1 = Failure)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, "failure_distribution.png"), dpi=120)
    plt.close()

    # 2. Correlation heatmap
    plt.figure(figsize=(10, 8))
    num_cols = ["Engine_Temperature", "Vibration", "Battery_Voltage", "Engine_Load",
                 "RPM", "Oil_Quality_Index", "Mileage_km", "Vehicle_Age_Years",
                 "Composite_Risk_Score", "Failure_Status"]
    corr = df[num_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, "correlation_heatmap.png"), dpi=120)
    plt.close()

    # 3. Distributions by failure status
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    sensors = ["Engine_Temperature", "Vibration", "Battery_Voltage", "Engine_Load"]
    for ax, col in zip(axes.flatten(), sensors):
        sns.histplot(data=df, x=col, hue="Failure_Status", kde=True, ax=ax,
                      palette=["#2ecc71", "#e74c3c"], element="step")
        ax.set_title(f"{col} Distribution by Failure Status")
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, "sensor_distributions.png"), dpi=120)
    plt.close()

    # 4. Boxplots of key sensors vs failure
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, col in zip(axes, ["Engine_Temperature", "Vibration", "Battery_Voltage"]):
        sns.boxplot(x="Failure_Status", y=col, data=df, ax=ax, palette=["#2ecc71", "#e74c3c"])
        ax.set_title(f"{col} vs Failure Status")
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, "sensor_boxplots.png"), dpi=120)
    plt.close()

    # 5. Composite risk score distribution
    plt.figure(figsize=(7, 5))
    sns.histplot(data=df, x="Composite_Risk_Score", hue="Failure_Status", kde=True,
                  palette=["#2ecc71", "#e74c3c"], element="step")
    plt.title("Composite Risk Score Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, "risk_score_distribution.png"), dpi=120)
    plt.close()

    print("All EDA charts saved to:", ASSETS_DIR)
    print(df.describe().T)


if __name__ == "__main__":
    main()

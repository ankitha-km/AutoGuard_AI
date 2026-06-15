"""
AutoGuard AI - Synthetic Vehicle Sensor Dataset Generator
Generates realistic vehicle sensor data with engineered failure patterns
for predictive maintenance modeling.
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

N_SAMPLES = 6000


def generate_dataset(n_samples=N_SAMPLES):
    data = {}

    # Vehicle identifiers
    data["Vehicle_ID"] = [f"V{1000 + i}" for i in range(n_samples)]

    # Engine Temperature (Celsius) - normal range 70-105, failures push higher
    base_temp = np.random.normal(85, 8, n_samples)

    # Vibration (mm/s RMS) - normal 1-4, failures push higher
    base_vibration = np.random.gamma(2, 1.2, n_samples)

    # Battery Voltage (V) - normal 12.0-14.5, failures push lower
    base_voltage = np.random.normal(13.0, 0.6, n_samples)

    # Engine Load (%) - 0-100
    base_load = np.random.uniform(10, 100, n_samples)

    # RPM
    base_rpm = np.random.normal(2200, 500, n_samples)
    base_rpm = np.clip(base_rpm, 600, 6000)

    # Oil Quality Index (0-100, lower = worse)
    oil_quality = np.random.normal(75, 15, n_samples)
    oil_quality = np.clip(oil_quality, 5, 100)

    # Mileage (km)
    mileage = np.random.exponential(45000, n_samples)
    mileage = np.clip(mileage, 500, 300000)

    # Vehicle Age (years)
    vehicle_age = np.clip(np.random.exponential(4, n_samples), 0.1, 20)

    # Compute a latent "stress score" driving failure probability
    temp_stress = (base_temp - 85) / 15
    vib_stress = (base_vibration - 2.5) / 2.5
    voltage_stress = (13.0 - base_voltage) / 1.5
    load_stress = (base_load - 50) / 50
    oil_stress = (75 - oil_quality) / 50
    age_stress = vehicle_age / 12
    mileage_stress = mileage / 150000

    stress_score = (
        0.22 * temp_stress
        + 0.20 * vib_stress
        + 0.18 * voltage_stress
        + 0.10 * load_stress
        + 0.12 * oil_stress
        + 0.10 * age_stress
        + 0.08 * mileage_stress
    )

    # Add noise
    stress_score += np.random.normal(0, 0.25, n_samples)

    # Convert to failure probability via logistic function
    failure_prob = 1 / (1 + np.exp(-3.2 * (stress_score - 0.35)))
    failure_status = (np.random.rand(n_samples) < failure_prob).astype(int)

    # Make sensor values for failing units more extreme (creates learnable patterns)
    temp = base_temp + failure_status * np.random.normal(12, 4, n_samples)
    vibration = base_vibration + failure_status * np.random.normal(2.5, 1.0, n_samples)
    vibration = np.clip(vibration, 0.1, None)
    voltage = base_voltage - failure_status * np.random.normal(1.5, 0.5, n_samples)
    voltage = np.clip(voltage, 8.0, 15.0)
    load = base_load

    data["Engine_Temperature"] = np.round(temp, 2)
    data["Vibration"] = np.round(vibration, 3)
    data["Battery_Voltage"] = np.round(voltage, 2)
    data["Engine_Load"] = np.round(load, 1)
    data["RPM"] = np.round(base_rpm, 0)
    data["Oil_Quality_Index"] = np.round(oil_quality, 1)
    data["Mileage_km"] = np.round(mileage, 0)
    data["Vehicle_Age_Years"] = np.round(vehicle_age, 2)
    data["Failure_Status"] = failure_status

    df = pd.DataFrame(data)

    # Inject some missing values to simulate real-world sensor dropouts
    for col in ["Engine_Temperature", "Vibration", "Battery_Voltage", "Oil_Quality_Index"]:
        missing_idx = np.random.choice(df.index, size=int(0.02 * n_samples), replace=False)
        df.loc[missing_idx, col] = np.nan

    # Inject a few outliers / sensor glitches
    glitch_idx = np.random.choice(df.index, size=int(0.005 * n_samples), replace=False)
    df.loc[glitch_idx, "Vibration"] = df.loc[glitch_idx, "Vibration"] * 15

    glitch_idx2 = np.random.choice(df.index, size=int(0.005 * n_samples), replace=False)
    df.loc[glitch_idx2, "Engine_Temperature"] = df.loc[glitch_idx2, "Engine_Temperature"] + 60

    return df


if __name__ == "__main__":
    df = generate_dataset()
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "vehicle_sensor_data.csv")
    df.to_csv(out_path, index=False)
    print(f"Dataset generated: {out_path}")
    print(f"Shape: {df.shape}")
    print(f"Failure rate: {df['Failure_Status'].mean():.2%}")
    print(df.head())

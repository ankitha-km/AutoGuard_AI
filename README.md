# 🚗 AutoGuard AI: Vehicle Health & Predictive Maintenance

An end-to-end machine learning system that predicts vehicle failure risk, estimates Remaining Useful Life (RUL), generates maintenance recommendations, and explains *why* a vehicle is at risk — all wrapped in a modern Streamlit dashboard. Built as a hackathon-ready Proof of Concept (PoC).

---

## ✨ Key Features

- **Synthetic + realistic vehicle sensor dataset** (temperature, vibration, battery voltage, engine load, RPM, oil quality, mileage, age)
- **Full data pipeline**: cleaning, missing-value imputation, outlier capping, feature engineering
- **3 trained models**: Decision Tree, Random Forest, XGBoost — compared on Accuracy, Precision, Recall, F1, ROC-AUC
- **Best model auto-selected** (by F1 score) and saved with `joblib`
- **Interactive Streamlit dashboard** with:
  - 🩺 Vehicle Health Score (0–100 gauge)
  - ⚠️ Failure Risk Prediction (probability gauge)
  - ⏳ Remaining Useful Life (RUL) estimate in days & km
  - 🛠️ Maintenance Recommendations with urgency levels
  - 📊 Charts: distributions, correlations, feature importance
- **3 input modes**: CSV upload (batch), manual sensor entry, preset sample vehicles
- **Explainable AI**: top contributing features per prediction (importance × deviation from healthy baseline)

---

## 📁 Project Structure

```
autoguard-ai/
├── app.py                          # Streamlit dashboard (main entry point)
├── run_pipeline.py                 # Runs full ML pipeline end-to-end
├── requirements.txt
├── README.md
├── data/
│   └── vehicle_sensor_data.csv     # Generated sample dataset (6,000 rows)
├── models/
│   ├── best_model.pkl              # Best model (Random Forest)
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── scaler.pkl                  # StandardScaler used in preprocessing
│   ├── feature_importance.csv
│   └── model_metadata.json         # Metrics, feature list, best model name
├── src/
│   ├── generate_dataset.py         # Synthetic dataset generator
│   ├── data_preprocessing.py       # Cleaning + feature engineering
│   ├── eda_visualization.py        # EDA charts -> assets/
│   ├── train_models.py             # Trains DT, RF, XGBoost; evaluates; saves best
│   └── predict_utils.py            # Health score, RUL, recommendations, XAI
└── assets/
    ├── failure_distribution.png
    ├── correlation_heatmap.png
    ├── sensor_distributions.png
    ├── sensor_boxplots.png
    ├── risk_score_distribution.png
    ├── model_comparison.png
    ├── all_confusion_matrices.png
    └── feature_importance.png
```

---

## 📊 Dataset

`data/vehicle_sensor_data.csv` contains **6,000 synthetic vehicle readings** with realistic correlations between sensor degradation and failure:

| Column | Description |
|---|---|
| `Vehicle_ID` | Unique vehicle identifier |
| `Engine_Temperature` | Engine temp in °C (normal ~70–105°C) |
| `Vibration` | Vibration RMS in mm/s (normal ~1–4) |
| `Battery_Voltage` | Battery voltage in V (normal ~12–14.5V) |
| `Engine_Load` | Engine load % (0–100) |
| `RPM` | Engine RPM |
| `Oil_Quality_Index` | Oil quality score (0–100, higher = better) |
| `Mileage_km` | Total mileage |
| `Vehicle_Age_Years` | Vehicle age |
| `Failure_Status` | Target: 0 = Healthy, 1 = Failure |

The dataset includes **~2% injected missing values** and **sensor glitch outliers** to simulate real-world data quality issues, plus a **32% failure rate**.

> 💡 To use a real Kaggle dataset (e.g. "Machine Predictive Maintenance Classification" or "Vehicle Sensor Data"), download it into `data/` and rename/adjust columns in `src/data_preprocessing.py` to match `FEATURE_COLUMNS`.

---

## 🧠 Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| Decision Tree | 94.17% | 90.93% | 90.93% | 90.93% | 97.52% |
| **Random Forest ✅ (Best)** | **95.75%** | **93.96%** | **92.75%** | **93.35%** | **99.02%** |
| XGBoost | 95.42% | 94.13% | 91.45% | 92.77% | 99.00% |

**Random Forest** was automatically selected as the best model based on F1 Score, balancing precision (avoiding false alarms) and recall (catching real failures).

### Top Predictive Features
1. Battery Voltage / Voltage Drop
2. Engine Temperature / Temperature Deviation
3. Vibration / Vibration Severity
4. Oil Quality Index
5. Wear Index (mileage + age)

---

## 🚀 Step-by-Step: Run Locally

### 1. Clone / extract the project
```bash
cd autoguard-ai
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the full ML pipeline (dataset → preprocessing → EDA → training)
```bash
python run_pipeline.py
```
This will:
- Generate `data/vehicle_sensor_data.csv`
- Clean data & engineer features
- Save EDA charts to `assets/`
- Train Decision Tree, Random Forest, XGBoost
- Print evaluation metrics & confusion matrices
- Save the best model to `models/best_model.pkl`

> ⏱️ Takes about 10–20 seconds on a typical laptop.

### 5. Launch the Streamlit dashboard
```bash
streamlit run app.py
```
Open the URL shown in the terminal (typically `http://localhost:8501`).

### 6. Using the dashboard
- **📁 Upload CSV**: Upload a CSV with sensor columns (or download the provided template) for batch predictions on a fleet
- **✍️ Manual Entry**: Use sliders to enter live sensor readings for a single vehicle
- **🎲 Try Sample Vehicle**: Instantly demo with 4 presets (Healthy → Critical)

---

## 🔬 How It Works

### Feature Engineering
- `Temp_Deviation` — absolute deviation from ideal 90°C
- `Vibration_Severity` — ratio to 3.0 mm/s safe threshold
- `Voltage_Drop` — drop below nominal 13.5V
- `Load_Temp_Interaction` — combined engine stress
- `Wear_Index` — normalized mileage + age
- Binary flags: `Oil_Degraded`, `High_Vibration_Flag`, `Battery_Weak_Flag`, `Overheating_Flag`
- `Composite_Risk_Score` — weighted 0–100 risk index used to derive the Health Score

### Vehicle Health Score
`Health Score = 100 - Composite_Risk_Score` (clamped 0–100). Combines temperature deviation, vibration severity, voltage drop, wear index, oil quality, and penalty points for critical flags.

### Remaining Useful Life (RUL) Estimate
A heuristic decay model:
```
RUL_days = 365 × exp(-4 × failure_probability) × wear_penalty × oil_factor
```
This gives intuitively shorter RUL for vehicles with high failure probability, heavy wear, and poor oil quality.

### Maintenance Recommendations
Rule-based engine maps detected flags (overheating, high vibration, weak battery, degraded oil, high load, high wear) to specific actions, with urgency levels: **LOW → MODERATE → HIGH → CRITICAL**.

### Explainable AI
For each prediction, the dashboard shows the top 5 features ranked by `feature_importance × |standardized deviation|` — i.e., features that are both globally important AND unusually high/low for this specific vehicle.

---

## 🖥️ Dashboard Screenshots / Mockup Descriptions (for Presentation Slides)

Since this is a PoC, here are descriptions of what each dashboard view looks like — use these as a guide for screenshots or to recreate mockup slides:

### Slide 1: Header & Sidebar
- Dark theme with a bold blue gradient title "🚗 AutoGuard AI"
- Sidebar shows active model (Random Forest), live accuracy/F1/ROC-AUC, and a mini comparison table of all 3 models
- Three input mode tabs: Upload CSV, Manual Entry, Sample Vehicle

### Slide 2: Prediction Results — Healthy Vehicle
- Two circular gauges side-by-side: **Health Score = 85/100** (green zone) and **Failure Risk = 1%** (green zone)
- Metric cards show **RUL ≈ 254 days (~10,170 km)** and status **"✅ Healthy"**
- Green "🟢 LOW RISK" badge with message "No immediate action needed"
- Right panel shows a horizontal bar chart of top contributing features (small bars, all green-ish)

### Slide 3: Prediction Results — Critical Vehicle
- Health Score gauge in **red zone (~10/100)**, Failure Risk gauge in **red zone (~99%)**
- RUL drops to **~2-5 days (~100-200 km)**, status **"⚠️ Failure Likely"**
- Red "🔴 CRITICAL RISK" badge: "Immediate workshop visit recommended within 1-3 days"
- Left panel lists detected issues (overheating, high vibration, weak battery, degraded oil) each in red issue-boxes, with matching green action-boxes below ("Inspect cooling system...", "Test battery health...")
- Right panel bar chart highlights Battery Voltage, Engine Temperature, and Vibration as top contributors (large orange/red bars)

### Slide 4: Batch CSV Upload View
- Top metrics row: Total Vehicles, At-Risk Count, Avg Health Score, Avg RUL
- Data table with all vehicles, sortable, color-coded predictions
- Two charts: histogram of Health Score distribution (fleet-wide), and a scatter plot of Temperature vs Vibration colored by failure probability (risk heatmap style)
- "Inspect Individual Vehicle" dropdown lets the presenter drill into any row and show the full XAI breakdown

### Slide 5: Model Comparison & EDA (from `assets/`)
- Bar chart comparing Accuracy/Precision/Recall/F1/ROC-AUC across Decision Tree, Random Forest, XGBoost
- Confusion matrices for all 3 models side by side
- Feature importance chart (Random Forest) showing Battery Voltage, Temp Deviation, Vibration Severity at top
- Correlation heatmap and sensor distribution histograms split by failure status

---

## 🧩 Tech Stack
- **Python 3.9+**
- **scikit-learn** — Decision Tree, Random Forest, preprocessing
- **XGBoost** — gradient boosting classifier
- **pandas / numpy** — data wrangling
- **matplotlib / seaborn** — EDA visualizations
- **Streamlit** — interactive dashboard
- **Plotly** — gauges, interactive charts
- **joblib** — model persistence

---

## 🏆 Hackathon Pitch Summary

> *"AutoGuard AI turns raw vehicle sensor streams into actionable insights in real time. Fleet operators upload a CSV or enter readings manually and instantly get a Health Score, Failure Risk %, Remaining Useful Life estimate, prioritized maintenance actions, and a transparent explanation of WHY the AI flagged a vehicle — reducing downtime, preventing breakdowns, and cutting maintenance costs through proactive servicing."*

### Potential Extensions
- Connect to real-time OBD-II / IoT sensor streams
- Time-series RUL modeling (LSTM/Survival analysis) using historical degradation curves
- Fleet-wide alerting via email/SMS for CRITICAL vehicles
- Integration with maintenance scheduling systems (ERP/CMMS)

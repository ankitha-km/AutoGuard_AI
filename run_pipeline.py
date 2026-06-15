"""
AutoGuard AI - Full Pipeline Runner
Runs the entire pipeline end-to-end: dataset generation -> preprocessing ->
EDA visualization -> model training -> evaluation -> artifact saving.

Usage:
    python run_pipeline.py
"""

import subprocess
import sys
import os

STEPS = [
    ("Generating synthetic vehicle sensor dataset...", "src/generate_dataset.py"),
    ("Running data preprocessing & feature engineering...", "src/data_preprocessing.py"),
    ("Generating EDA visualizations...", "src/eda_visualization.py"),
    ("Training & evaluating ML models...", "src/train_models.py"),
]


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for message, script in STEPS:
        print("\n" + "=" * 60)
        print(message)
        print("=" * 60)
        script_path = os.path.join(base_dir, script)
        script_dir = os.path.dirname(script_path)
        result = subprocess.run([sys.executable, os.path.basename(script_path)], cwd=script_dir)
        if result.returncode != 0:
            print(f"\nStep failed: {script}")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("Pipeline complete! Run the dashboard with:")
    print("    streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()

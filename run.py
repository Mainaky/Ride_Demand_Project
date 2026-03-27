"""
run.py — Master script for Ride Demand Prediction System
=========================================================
Runs the full pipeline in order:
  Step 1: ETL  → processes all 14 CSVs → processed.csv
  Step 2: ML   → stats, PCA, clustering, XGBoost → model.pkl
  Step 3: EDA  → plots → data/processed/plot_*.png

Then prints instructions for starting the API and dashboard.

Usage:
  python run.py
"""

import sys
import os

print("=" * 60)
print("  RIDE DEMAND PREDICTION SYSTEM — FULL PIPELINE")
print("=" * 60)

# ── Step 1: ETL ─────────────────────────────────────────────────
print("\n[STEP 1] Running ETL pipeline...")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "etl"))
from etl import run_etl

try:
    RAW_DIR       = os.path.join(os.path.dirname(__file__), "data", "raw")
    PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "data", "processed")
    run_etl(RAW_DIR, PROCESSED_DIR)
except Exception as e:
    print(f"  ETL failed: {e}")
    sys.exit(1)

# ── Step 2: ML Training ─────────────────────────────────────────
print("\n[STEP 2] Running ML training...")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "model"))
from train import run_training

try:
    run_training()
except Exception as e:
    print(f"  Training failed: {e}")
    sys.exit(1)

# ── Step 3: EDA plots ────────────────────────────────────────────
print("\n[STEP 3] Running EDA analysis...")
try:
    exec(open(os.path.join(os.path.dirname(__file__), "notebooks", "eda_analysis.py")).read())
except Exception as e:
    print(f"  EDA skipped (non-critical): {e}")

# ── Done ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  PIPELINE COMPLETE ✓")
print("=" * 60)
print("""
Next steps:

  Start the API:
    cd api
    uvicorn app:app --reload
    → Open http://localhost:8000/docs

  Start the dashboard:
    cd dashboard
    streamlit run app.py
    → Opens in browser automatically

  Upload to Snowflake (optional):
    python snowflake/connection.py
    → Fill in your credentials in .env first

  Push to GitHub:
    git init
    git add .
    git commit -m "feat: complete ride demand prediction system"
    git remote add origin https://github.com/YOUR_USERNAME/ride-demand-project.git
    git push -u origin main
""")

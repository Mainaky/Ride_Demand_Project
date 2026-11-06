# ZonePulse: Ride Demand Prediction & Surge Pricing System
### Uber/Ola · NYC · Data Engineering + Machine Learning + Full-Stack Cloud Deployment

> Production-grade demand forecasting and fleet intelligence system. Processes 4.7M+ raw NYC ride-hailing records across heterogeneous schemas, loads data into Snowflake cloud data warehouse, predicts zone-hour ride demand using XGBoost + PCA + KMeans, and serves real-time predictions via an authenticated FastAPI REST microservice and an interactive Streamlit dashboard.

---

## 🏛️ System Architecture

```text
               ┌────────────────────────────────────────────────────────┐
               │              CLIENT / WEB BROWSER                      │
               │  Streamlit UI (Port 8501 / Render Web Service)         │
               └──────────────────────────┬─────────────────────────────┘
                                          │
                  HTTPS Requests / Bearer JWT Authentication
                                          │
                                          ▼
               ┌────────────────────────────────────────────────────────┐
               │              BACKEND REST MICROSERVICE                 │
               │  FastAPI (Port 8000 / Render Web Service)              │
               ├──────────────────────────┬─────────────────────────────┤
               │  • /auth/signup          │  • /predict (Regression)    │
               │  • /auth/login           │  • /classify (Surge Tier)   │
               │  • /auth/me              │  • /predict/batch (x1000)   │
               │  • /stats & /model-info  │  • /docs (OpenAPI Swagger)  │
               └─────────────┬────────────────────────────┬─────────────┘
                             │                            │
                             ▼                            ▼
               ┌──────────────────────────┐  ┌──────────────────────────┐
               │    SQLITE AUTH STORE     │  │   SNOWFLAKE CLOUD DWH    │
               │   api/users.db (PBKDF2)  │  │   UBER_PROJECT.RIDE_DATA │
               │   Salted Passwords & JWT │  │   write_pandas Ingestion │
               └──────────────────────────┘  └──────────────────────────┘
```

---

## 🌟 Key Features

1. **Enterprise User Authentication:**
   - **User Sign-Up:** Secure registration with email validation and PBKDF2-HMAC-SHA256 salted password hashing.
   - **User Login:** Issues tamper-evident signed JWT bearer tokens with 7-day expiration.
   - **1-Click Recruiter Demo:** Instant guest review access without manual form entry.
2. **Dual Machine Learning Engine:**
   - **Regression Track (XGBoost):** Explains **79.3% of demand variance** ($R^2 = 0.7930$, $\text{MAE} = 7.97$ rides) to forecast exact ride counts for driver capacity sizing.
   - **Classification Track (XGBoost):** Categorizes demand into 4 operational tiers (**Low**, **Medium**, **High**, **Very High**) with **67.8% accuracy** to trigger automated surge multipliers.
3. **Unsupervised Intelligence:**
   - **PCA (Principal Component Analysis):** Projects 5-dimensional feature space onto 2 principal components for interactive 2D cluster inspection.
   - **K-Means Clustering ($k=4$):** Segments zones into behavioral groups and feeds cluster labels back into XGBoost as augmented features.
4. **Snowflake Cloud Data Warehouse:**
   - Staged bulk loading using `write_pandas` and analytical SQL queries identifying evening peak commute windows (17:00–21:00) and spatial corridors.
5. **Production Deployment Ready:**
   - Multi-service `render.yaml` Blueprint for 1-click cloud deployment on Render.

---

## 🚀 Local Quickstart

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.11).

### 2. Install Dependencies
```bash
python -m venv .venv

# Windows PowerShell:
.venv\Scripts\Activate.ps1

# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Start Backend API (Terminal 1)
```powershell
cd api
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```
* Interactive Swagger Docs: `http://127.0.0.1:8000/docs`
* API Root Health: `http://127.0.0.1:8000/`

### 4. Start Frontend Dashboard (Terminal 2)
```powershell
cd dashboard
streamlit run app.py
```
* Dashboard will automatically open in your browser at `http://localhost:8501`.

---

## ☁️ Deploying to Render

This repository includes a `render.yaml` Blueprint for 1-click cloud deployment.

### Option A: Render Blueprint (Recommended)
1. Push this repository to GitHub: `https://github.com/Mainaky/Ride_Demand_Project`.
2. Open your [Render Dashboard](https://dashboard.render.com).
3. Click **New +** $\rightarrow$ **Blueprint**.
4. Select `Ride_Demand_Project`.
5. Render will automatically provision both:
   - `zonepulse-api` (FastAPI backend)
   - `zonepulse-dashboard` (Streamlit UI)
6. Once deployed, open the public URL of `zonepulse-dashboard` to view the live app!

---

## 📈 Model Performance Benchmark

| Model | Task | Metric 1 | Metric 2 | R² / F1 |
| :--- | :--- | :--- | :--- | :--- |
| **XGBoost Regressor (Best)** | Continuous Demand Count | MAE: 7.97 rides | RMSE: 21.24 | **R²: 0.7930 (79.3%)** |
| **Random Forest Regressor** | Continuous Demand Count | MAE: 6.65 rides | RMSE: 21.70 | **R²: 0.7839 (78.4%)** |
| **XGBoost Classifier (Best)** | Surge Tier Classification | Accuracy: 67.8% | Precision: 62.4% | **Weighted F1: 0.6235** |
| **Random Forest Classifier** | Surge Tier Classification | Accuracy: 55.1% | Precision: 46.9% | **Weighted F1: 0.4814** |

---

## 📂 Repository Structure

```text
Ride_Demand_Project/
├── api/
│   ├── app.py             ← FastAPI REST service (Regression, Classification, Batch)
│   ├── auth.py            ← User authentication & SQLite token manager
│   └── users.db           ← Auto-generated persistent user database
├── dashboard/
│   └── app.py             ← 4-tab Streamlit dashboard with login/signup gate
├── data/
│   └── processed/
│       └── processed.csv  ← 311,156 model-ready aggregated zone-hour records (8.3 MB)
├── etl/
│   └── etl.py             ← 4-schema ETL ingestion, NYC bounding box, 1.1km grid
├── model/
│   ├── train.py           ← Dual-track ML training pipeline (PCA, KMeans, XGBoost, RF)
│   ├── model.pkl          ← Trained XGBoost Regressor
│   ├── classifier.pkl     ← Trained XGBoost Classifier
│   ├── kmeans.pkl         ← Trained KMeans clustering model (k=4)
│   ├── pca.pkl            ← Trained 2D PCA projector
│   ├── scaler.pkl         ← Feature standardizer
│   ├── label_encoder.pkl  ← Zone encoder
│   ├── metrics.json       ← Evaluation metrics & feature importances
│   └── stats.json         ← Dataset descriptive statistics
├── notebooks/
│   └── eda_analysis.py    ← 7 exploratory data analysis visualization scripts
├── snowflake/
│   └── connection.py      ← Snowflake cloud data warehouse client & SQL analytics
├── render.yaml            ← 1-click Render cloud deployment blueprint
├── requirements.txt       ← Python dependencies
├── .gitignore             ← Clean ignore rules (excludes >600MB raw data)
└── README.md
```

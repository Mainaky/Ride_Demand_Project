# ZonePulse: Ride Demand Prediction & Surge Pricing System
### Global mobility planning · NYC-trained demand intelligence · Full-Stack Cloud Deployment

> ZonePulse is a full-stack Node.js ride planning app with a responsive JavaScript frontend, Express API, and PostgreSQL account and ride storage. The Python ETL and model training pipeline remains available for research; the live Node app uses transparent global planning estimates.

---

## 🏛️ System Architecture

```text
               ┌────────────────────────────────────────────────────────┐
               │              CLIENT / WEB BROWSER                      │
               │  Responsive JavaScript UI (Node.js web service)        │
               └──────────────────────────┬─────────────────────────────┘
                                          │
                  HTTPS Requests / Bearer JWT Authentication
                                          │
                                          ▼
               ┌────────────────────────────────────────────────────────┐
               │           EXPRESS API + STATIC FRONTEND                │
               │  Single Node.js web service (one public origin)        │
               ├──────────────────────────┬─────────────────────────────┤
               │  • /api/auth/signup      │  • /api/ride/quote          │
               │  • /api/auth/login       │  • /api/ride/book           │
               │  • /api/auth/me          │  • /api/ride/history        │
               │  • /api/weather          │  • /api/geocode             │
               └─────────────┬────────────────────────────┬─────────────┘
                             │                            │
                             ▼                            ▼
               ┌──────────────────────────┐  ┌──────────────────────────┐
               │   RENDER POSTGRESQL     │  │   SNOWFLAKE CLOUD DWH    │
               │ Passengers · Drivers    │  │   UBER_PROJECT.RIDE_DATA │
               │ Rides · bcrypt hashes   │  │   write_pandas Ingestion │
               └──────────────────────────┘  └──────────────────────────┘
```

---

## 🌟 Key Features

1. **Passenger accounts and authentication:**
   - Passenger names, email addresses, optional contact numbers, password hashes, and account timestamps are stored in PostgreSQL. Existing PBKDF2 password hashes are verified and upgraded to bcrypt after a successful sign-in.
   - Login issues signed bearer tokens with a 7-day expiration.
   - Retried account creation safely resumes the same account after a delayed response.
2. **PostgreSQL ride records:**
   - Passenger profiles, driver and vehicle profiles, trip routes and coordinates, fare and demand estimates, timestamps, and booking status are stored in related tables.
   - Each passenger can view only their own trip history.
   - Driver availability is simulated for this portfolio application and is not a live dispatch network.
3. **Global location planning:**
   - Worldwide addresses and landmarks are resolved with the [Photon geocoding API](https://github.com/komoot/photon/blob/master/docs/api-v1.md). Its public demo service may throttle heavy use and has no availability guarantee.
   - Current pickup weather and local time are fetched globally from [Open-Meteo](https://open-meteo.com/) using the selected coordinates; no API key is required.
   - The Node app displays a transparent time-of-week planning baseline worldwide; its Python ML assets remain in the repository for model research.
   - Route distances and estimated fares are shown as USD-equivalent planning estimates, not local market rates or live ride quotes.
4. **Dual Machine Learning Engine:**
   - **Regression Track (XGBoost):** Explains **79.3% of demand variance** ($R^2 = 0.7930$, $\text{MAE} = 7.97$ rides) to forecast exact ride counts for driver capacity sizing.
   - **Classification Track (XGBoost):** Categorizes demand into 4 operational tiers (**Low**, **Medium**, **High**, **Very High**) with **67.8% accuracy** to trigger automated surge multipliers.
5. **Unsupervised Intelligence:**
   - **PCA (Principal Component Analysis):** Projects 5-dimensional feature space onto 2 principal components for interactive 2D cluster inspection.
   - **K-Means Clustering ($k=4$):** Segments zones into behavioral groups and feeds cluster labels back into XGBoost as augmented features.
6. **Snowflake Cloud Data Warehouse:**
   - Staged bulk loading using `write_pandas` and analytical SQL queries identifying evening peak commute windows (17:00–21:00) and spatial corridors.
7. **Production Deployment Ready:**
   - Multi-service `render.yaml` Blueprint for 1-click cloud deployment on Render.

---

## Passenger, driver, and ride storage

Render's Blueprint connects the Node app to PostgreSQL through the internal database URL. The schema contains:

- `passengers`: display name, sign-in username, email, optional phone, password hash, role, and creation time.
- `drivers`: simulated driver reference, rating, vehicle, color, plate, latest location estimate, ETA, and status.
- `rides`: passenger and driver references plus route coordinates, fare, ride tier, demand and availability estimates, booking status, and timestamp.

Only password hashes are stored. New passwords use bcrypt; existing PBKDF2 hashes are checked on sign-in and upgraded automatically. A contact number is optional; when supplied for a booking, it is saved to the passenger profile and trip record.

Sign in with either the account username (the name entered at signup) or account email, plus the original password. Accounts persist only in the database currently configured for the API. If a deployment switches from an earlier SQLite store or PostgreSQL database to a newly provisioned database, old accounts do not move automatically; they must be migrated from the original store while it is still available. Password hashes cannot recover a forgotten password.

**Current Render selection:** Free PostgreSQL is temporary. Render expires it 30 days after creation; after a further 14-day recovery window, its data is deleted. Upgrade the database before expiry or export the records you need to keep.

---

## 🚀 Local Quickstart

## Weather endpoint

`GET /api/weather?lat=<latitude>&lon=<longitude>` returns current conditions and the local time zone for any valid global coordinates. The endpoint uses Open-Meteo's current weather feed and requires a signed-in session, like the ride-planning endpoints.

### 1. Prerequisites
- Node.js 20 or newer
- PostgreSQL

### 2. Configure and install
```bash
npm install
cp .env.example .env
# Set DATABASE_URL and JWT_SECRET_KEY in .env
```
On Windows PowerShell, use `Copy-Item .env.example .env` in place of `cp`.

### 3. Start the full-stack app
```bash
npm start
```
Open `http://localhost:3000`. The Node server serves the JavaScript UI and its API from one origin, so signup and login do not depend on a second cold-starting web service.

---

## ☁️ Deploying to Render

This repository includes a `render.yaml` Blueprint for 1-click cloud deployment.

### Option A: Render Blueprint (Recommended)
1. Push this repository to GitHub: `https://github.com/Mainaky/Ride_Demand_Project`.
2. Open your [Render Dashboard](https://dashboard.render.com).
3. Click **New +** $\rightarrow$ **Blueprint**.
4. Select `Ride_Demand_Project`.
5. The Blueprint includes `zonepulse-api` for legacy Python model endpoints and `Ride_Demand_Project` as the Node.js full-stack app.
6. The Node service connects to `zonepulse-db`. Open the existing site URL to sign in or create an account; account and trip tables stay in the same database.

The app's frontend and backend share one Node.js Render web service and one database connection. Free Render services can spin down when idle, and free PostgreSQL expires after 30 days. The public Photon geocoder is intended for reasonable interactive search volume. Forecasts are global planning baselines, not live local demand.

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
├── server.js             ← Express backend and static asset server
├── server/
│   └── domain.js         ← Global estimate and weather helpers
├── public/
│   ├── app.js            ← Responsive JavaScript frontend
│   ├── styles.css        ← UI design system
│   └── index.html        ← App shell
├── package.json          ← Node.js runtime dependencies and scripts
├── package-lock.json     ← Reproducible Node dependency lock
├── api/
│   ├── app.py             ← Legacy FastAPI model endpoints
│   └── ride_service.py    ← Python model helpers
├── dashboard/
│   └── app.py             ← Previous Streamlit interface
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

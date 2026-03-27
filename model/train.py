"""
ML Training Pipeline - Ride Demand Prediction System
=====================================================
Steps:
  1. Load processed.csv
  2. Statistical analysis (mean, variance, skewness, correlation)
  3. Encode categorical features
  4. PCA – dimensionality reduction + explained variance
  5. KMeans clustering – demand segment labels
  6. Train XGBoost + Random Forest
  7. Evaluate (MAE, RMSE, R²)
  8. Save model.pkl + encoders
  9. Save evaluation metrics + feature importance
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODEL_DIR     = os.path.dirname(__file__)
os.makedirs(MODEL_DIR, exist_ok=True)


def load_data(processed_dir: str = PROCESSED_DIR) -> pd.DataFrame:
    path = os.path.join(processed_dir, "processed.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"processed.csv not found at {path}. Run etl/etl.py first."
        )
    df = pd.read_csv(path)
    print(f"  Loaded processed.csv → {len(df):,} rows, {df.shape[1]} columns")
    return df


def statistical_analysis(df: pd.DataFrame) -> dict:
    """Compute and print key statistics."""
    print("\n  ── Statistical Analysis ──────────────────────")
    stats = {
        "demand_mean":     round(df["demand"].mean(),   2),
        "demand_median":   round(df["demand"].median(), 2),
        "demand_std":      round(df["demand"].std(),    2),
        "demand_variance": round(df["demand"].var(),    2),
        "demand_skewness": round(df["demand"].skew(),   4),
        "demand_kurtosis": round(df["demand"].kurtosis(), 4),
        "demand_min":      int(df["demand"].min()),
        "demand_max":      int(df["demand"].max()),
        "total_records":   len(df),
        "unique_locations":int(df["location"].nunique()),
    }
    for k, v in stats.items():
        print(f"    {k:30s}: {v}")

    print("\n  Correlation with demand:")
    numeric_df = df[["hour", "day_of_week", "month", "is_weekend", "hour_bin", "demand"]]
    corr = numeric_df.corr()["demand"].drop("demand").round(4)
    for feat, val in corr.items():
        print(f"    {feat:20s}: {val:+.4f}")

    stats["correlations"] = corr.to_dict()
    return stats


def encode_features(df: pd.DataFrame):
    """Encode location strings to integer codes."""
    le = LabelEncoder()
    df = df.copy()
    df["location_code"] = le.fit_transform(df["location"])
    return df, le


def run_pca(df: pd.DataFrame, features: list) -> tuple:
    """Scale + PCA on numeric features."""
    print("\n  ── PCA ───────────────────────────────────────")
    X = df[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
    ev = pca.explained_variance_ratio_
    print(f"    PC1 explained variance: {ev[0]:.4f} ({ev[0]*100:.1f}%)")
    print(f"    PC2 explained variance: {ev[1]:.4f} ({ev[1]*100:.1f}%)")
    print(f"    Total retained:         {sum(ev)*100:.1f}%")

    df = df.copy()
    df["pca_1"] = components[:, 0]
    df["pca_2"] = components[:, 1]
    return df, scaler, pca


def run_clustering(df: pd.DataFrame, n_clusters: int = 4) -> tuple:
    """KMeans clustering on location_code + hour."""
    print(f"\n  ── KMeans Clustering (k={n_clusters}) ───────────────")
    X_clust = df[["location_code", "hour"]].values
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df = df.copy()
    df["cluster"] = kmeans.fit_predict(X_clust)

    for c in range(n_clusters):
        mask = df["cluster"] == c
        avg_demand = df.loc[mask, "demand"].mean()
        print(f"    Cluster {c}: {mask.sum():>6,} rows | avg demand = {avg_demand:.2f}")

    return df, kmeans


def train_and_evaluate(df: pd.DataFrame, model_dir: str = MODEL_DIR):
    """Train XGBoost + RF, evaluate, save best model."""
    print("\n  ── Model Training ────────────────────────────")

    FEATURES = ["location_code", "hour", "day_of_week", "month",
                "is_weekend", "hour_bin", "cluster"]
    TARGET   = "demand"

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"    Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    models = {
        "XGBoost": XGBRegressor(
            n_estimators=200,
            learning_rate=0.08,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ),
    }

    results = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        pred = m.predict(X_test)
        mae  = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        r2   = r2_score(y_test, pred)
        results[name] = {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4)}
        print(f"    {name:15s}  MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")

    # Best model = lowest MAE
    best_name = min(results, key=lambda n: results[n]["MAE"])
    best_model = models[best_name]
    print(f"\n    Best model: {best_name}")

    # Feature importance (XGBoost only)
    feat_imp = {}
    if best_name == "XGBoost":
        fi = best_model.feature_importances_
        feat_imp = dict(zip(FEATURES, [round(float(v), 4) for v in fi]))
        print("    Feature importance:")
        for feat, imp in sorted(feat_imp.items(), key=lambda x: -x[1]):
            bar = "█" * int(imp * 40)
            print(f"      {feat:20s} {imp:.4f}  {bar}")

    # Save model
    model_path = os.path.join(model_dir, "model.pkl")
    joblib.dump(best_model, model_path)
    print(f"\n    [SAVED] model.pkl  ({best_name})")

    # Save metrics
    metrics = {
        "best_model": best_name,
        "features": FEATURES,
        "results": results,
        "feature_importance": feat_imp,
    }
    metrics_path = os.path.join(model_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"    [SAVED] metrics.json")

    return best_model, results, FEATURES


def run_training():
    print("=" * 60)
    print("  RIDE DEMAND PREDICTION — ML TRAINING PIPELINE")
    print("=" * 60)

    df = load_data()

    stats = statistical_analysis(df)
    stats_path = os.path.join(MODEL_DIR, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    df, le = encode_features(df)

    pca_features = ["location_code", "hour", "day_of_week", "month", "is_weekend"]
    df, scaler, pca = run_pca(df, pca_features)

    df, kmeans = run_clustering(df)

    model, results, features = train_and_evaluate(df)

    # Save encoders/transformers
    joblib.dump(le,     os.path.join(MODEL_DIR, "label_encoder.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(pca,    os.path.join(MODEL_DIR, "pca.pkl"))
    joblib.dump(kmeans, os.path.join(MODEL_DIR, "kmeans.pkl"))

    # Save cluster centres for API use
    cluster_info = {
        str(i): {
            "centre_location_code": round(float(kmeans.cluster_centers_[i][0]), 2),
            "centre_hour":          round(float(kmeans.cluster_centers_[i][1]), 2),
        }
        for i in range(4)
    }
    with open(os.path.join(MODEL_DIR, "cluster_info.json"), "w") as f:
        json.dump(cluster_info, f, indent=2)

    print("\n  TRAINING COMPLETE ✓")
    print("=" * 60)
    return model


if __name__ == "__main__":
    run_training()

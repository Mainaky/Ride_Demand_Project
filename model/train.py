"""
ML Training Pipeline - Ride Demand Prediction System
=====================================================
REGRESSION  → predict exact demand count
CLASSIFICATION → predict demand category (Low / Medium / High / Very High)

Steps:
  1. Load processed.csv
  2. Statistical analysis
  3. Encode + PCA + KMeans
  4. REGRESSION  : XGBoost + RandomForest → MAE, RMSE, R²
  5. CLASSIFICATION: create demand_class label → XGBoost + RF + LogisticReg
                     → Accuracy, Precision, Recall, F1, Confusion Matrix
  6. Save all models + metrics
"""

import os, json, warnings
import numpy as np
import pandas as pd
import joblib
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from xgboost import XGBRegressor, XGBClassifier

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODEL_DIR     = os.path.dirname(__file__)
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURES = ["location_code", "hour", "day_of_week", "month",
            "is_weekend", "hour_bin", "cluster"]

CLASS_LABELS = ["Low", "Medium", "High", "Very High"]


def load_data():
    path = os.path.join(PROCESSED_DIR, "processed.csv")
    if not os.path.exists(path):
        raise FileNotFoundError("processed.csv not found. Run etl/etl.py first.")
    df = pd.read_csv(path)
    print(f"  Loaded processed.csv → {len(df):,} rows, {df.shape[1]} columns")
    return df


def statistical_analysis(df):
    print("\n  ── Statistical Analysis ──────────────────────")
    stats = {
        "demand_mean":      round(df["demand"].mean(),      2),
        "demand_median":    round(df["demand"].median(),    2),
        "demand_std":       round(df["demand"].std(),       2),
        "demand_variance":  round(df["demand"].var(),       2),
        "demand_skewness":  round(df["demand"].skew(),      4),
        "demand_kurtosis":  round(df["demand"].kurtosis(),  4),
        "demand_min":       int(df["demand"].min()),
        "demand_max":       int(df["demand"].max()),
        "total_records":    len(df),
        "unique_locations": int(df["location"].nunique()),
    }
    for k, v in stats.items():
        print(f"    {k:30s}: {v}")
    numeric_df = df[["hour","day_of_week","month","is_weekend","hour_bin","demand"]]
    corr = numeric_df.corr()["demand"].drop("demand").round(4)
    print("\n  Correlation with demand:")
    for feat, val in corr.items():
        print(f"    {feat:20s}: {val:+.4f}")
    stats["correlations"] = corr.to_dict()
    return stats


def encode_features(df):
    le = LabelEncoder()
    df = df.copy()
    df["location_code"] = le.fit_transform(df["location"])
    return df, le


def run_pca(df):
    print("\n  ── PCA ───────────────────────────────────────")
    feats = ["location_code","hour","day_of_week","month","is_weekend"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feats].values)
    pca = PCA(n_components=2)
    comps = pca.fit_transform(X_scaled)
    ev = pca.explained_variance_ratio_
    print(f"    PC1: {ev[0]*100:.1f}%  PC2: {ev[1]*100:.1f}%  Total: {sum(ev)*100:.1f}%")
    df = df.copy()
    df["pca_1"] = comps[:, 0]
    df["pca_2"] = comps[:, 1]
    return df, scaler, pca


def run_clustering(df):
    print("\n  ── KMeans Clustering (k=4) ───────────────────")
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df = df.copy()
    df["cluster"] = kmeans.fit_predict(df[["location_code","hour"]].values)
    for c in range(4):
        mask = df["cluster"] == c
        print(f"    Cluster {c}: {mask.sum():>6,} rows | avg demand = {df.loc[mask,'demand'].mean():.2f}")
    return df, kmeans


# ─────────────────────────────────────────────────────────────────
# CLASSIFICATION LABEL CREATION
# ─────────────────────────────────────────────────────────────────

def create_demand_class(df):
    """
    Convert continuous demand into 4 classes based on percentiles.

    Class 0 — Low       : demand <= 25th percentile
    Class 1 — Medium    : demand <= 50th percentile
    Class 2 — High      : demand <= 75th percentile
    Class 3 — Very High : demand >  75th percentile
    """
    p25 = df["demand"].quantile(0.25)
    p50 = df["demand"].quantile(0.50)
    p75 = df["demand"].quantile(0.75)

    def label(d):
        if d <= p25: return 0
        elif d <= p50: return 1
        elif d <= p75: return 2
        else: return 3

    df = df.copy()
    df["demand_class"] = df["demand"].apply(label)

    thresholds = {
        "Low":       f"demand ≤ {p25:.0f}",
        "Medium":    f"demand ≤ {p50:.0f}",
        "High":      f"demand ≤ {p75:.0f}",
        "Very High": f"demand >  {p75:.0f}",
    }
    print("\n  ── Demand Class Distribution ─────────────────")
    for cls, thresh in thresholds.items():
        count = (df["demand_class"] == list(thresholds.keys()).index(cls)).sum()
        print(f"    {cls:12s} ({thresh}): {count:>7,} rows")

    return df, {"p25": round(p25,2), "p50": round(p50,2), "p75": round(p75,2)}


# ─────────────────────────────────────────────────────────────────
# REGRESSION TRAINING
# ─────────────────────────────────────────────────────────────────

def train_regression(df):
    print("\n  ══════════════════════════════════════════════")
    print("  ── REGRESSION MODELS ─────────────────────────")
    print("  ══════════════════════════════════════════════")

    X = df[FEATURES]
    y = df["demand"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"    Train: {len(X_train):,}  Test: {len(X_test):,}")

    reg_models = {
        "XGBoost_Regressor": XGBRegressor(
            n_estimators=200, learning_rate=0.08, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0
        ),
        "RandomForest_Regressor": RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        ),
    }

    reg_results = {}
    best_reg_model = None
    best_reg_name  = None
    best_r2 = -999

    for name, m in reg_models.items():
        m.fit(X_train, y_train)
        pred = m.predict(X_test)
        mae  = round(float(mean_absolute_error(y_test, pred)),  4)
        rmse = round(float(np.sqrt(mean_squared_error(y_test, pred))), 4)
        r2   = round(float(r2_score(y_test, pred)), 4)
        reg_results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2}
        print(f"    {name:30s}  MAE={mae}  RMSE={rmse}  R²={r2}")
        if r2 > best_r2:
            best_r2 = r2
            best_reg_name  = name
            best_reg_model = m

    print(f"\n    Best Regression Model: {best_reg_name} (R²={best_r2})")

    # Feature importance
    feat_imp = {}
    if hasattr(best_reg_model, "feature_importances_"):
        fi = best_reg_model.feature_importances_
        feat_imp = dict(zip(FEATURES, [round(float(v), 4) for v in fi]))
        print("    Feature importance:")
        for feat, imp in sorted(feat_imp.items(), key=lambda x: -x[1]):
            bar = "█" * int(imp * 40)
            print(f"      {feat:20s} {imp:.4f}  {bar}")

    joblib.dump(best_reg_model, os.path.join(MODEL_DIR, "model.pkl"))
    print(f"\n    [SAVED] model.pkl  ({best_reg_name})")

    return reg_results, best_reg_name, feat_imp


# ─────────────────────────────────────────────────────────────────
# CLASSIFICATION TRAINING
# ─────────────────────────────────────────────────────────────────

def train_classification(df):
    print("\n  ══════════════════════════════════════════════")
    print("  ── CLASSIFICATION MODELS ─────────────────────")
    print("  ══════════════════════════════════════════════")

    X = df[FEATURES]
    y = df["demand_class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"    Train: {len(X_train):,}  Test: {len(X_test):,}")
    print(f"    Classes: {CLASS_LABELS}")

    cls_models = {
        "XGBoost_Classifier": XGBClassifier(
            n_estimators=200, learning_rate=0.08, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0,
            use_label_encoder=False, eval_metric="mlogloss"
        ),
        "RandomForest_Classifier": RandomForestClassifier(
            n_estimators=100, max_depth=10,
            random_state=42, n_jobs=-1
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=1000, random_state=42, multi_class="multinomial"
        ),
    }

    cls_results = {}
    best_cls_model = None
    best_cls_name  = None
    best_f1 = -1

    for name, m in cls_models.items():
        m.fit(X_train, y_train)
        pred = m.predict(X_test)

        acc  = round(float(accuracy_score(y_test, pred)),  4)
        prec = round(float(precision_score(y_test, pred, average="weighted", zero_division=0)), 4)
        rec  = round(float(recall_score(y_test, pred, average="weighted",    zero_division=0)), 4)
        f1   = round(float(f1_score(y_test, pred, average="weighted",        zero_division=0)), 4)
        cm   = confusion_matrix(y_test, pred).tolist()

        cls_results[name] = {
            "Accuracy":  acc,
            "Precision": prec,
            "Recall":    rec,
            "F1_Score":  f1,
            "confusion_matrix": cm,
        }
        print(f"\n    {name}")
        print(f"      Accuracy={acc}  Precision={prec}  Recall={rec}  F1={f1}")
        print(f"      {classification_report(y_test, pred, target_names=CLASS_LABELS, zero_division=0)}")

        if f1 > best_f1:
            best_f1 = f1
            best_cls_name  = name
            best_cls_model = m

    print(f"\n    Best Classification Model: {best_cls_name} (F1={best_f1})")

    # Feature importance for classifier
    cls_feat_imp = {}
    if hasattr(best_cls_model, "feature_importances_"):
        fi = best_cls_model.feature_importances_
        cls_feat_imp = dict(zip(FEATURES, [round(float(v), 4) for v in fi]))
        print("    Feature importance:")
        for feat, imp in sorted(cls_feat_imp.items(), key=lambda x: -x[1]):
            bar = "█" * int(imp * 40)
            print(f"      {feat:20s} {imp:.4f}  {bar}")

    joblib.dump(best_cls_model, os.path.join(MODEL_DIR, "classifier.pkl"))
    print(f"\n    [SAVED] classifier.pkl  ({best_cls_name})")

    return cls_results, best_cls_name, cls_feat_imp


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def run_training():
    print("=" * 60)
    print("  RIDE DEMAND — REGRESSION + CLASSIFICATION PIPELINE")
    print("=" * 60)

    df = load_data()

    stats = statistical_analysis(df)
    with open(os.path.join(MODEL_DIR, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    df, le     = encode_features(df)
    df, scaler, pca = run_pca(df)
    df, kmeans = run_clustering(df)
    df, thresholds = create_demand_class(df)

    # ── Regression ──────────────────────────────────────────────
    reg_results, best_reg, reg_feat_imp = train_regression(df)

    # ── Classification ──────────────────────────────────────────
    cls_results, best_cls, cls_feat_imp = train_classification(df)

    # ── Save all artefacts ───────────────────────────────────────
    joblib.dump(le,     os.path.join(MODEL_DIR, "label_encoder.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(pca,    os.path.join(MODEL_DIR, "pca.pkl"))
    joblib.dump(kmeans, os.path.join(MODEL_DIR, "kmeans.pkl"))

    metrics = {
        "best_model":              best_reg,
        "features":                FEATURES,
        "class_labels":            CLASS_LABELS,
        "demand_thresholds":       thresholds,
        "results":                 reg_results,
        "feature_importance":      reg_feat_imp,
        "best_classifier":         best_cls,
        "classification_results":  cls_results,
        "cls_feature_importance":  cls_feat_imp,
    }
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    cluster_info = {
        str(i): {
            "centre_location_code": round(float(kmeans.cluster_centers_[i][0]), 2),
            "centre_hour":          round(float(kmeans.cluster_centers_[i][1]), 2),
        }
        for i in range(4)
    }
    with open(os.path.join(MODEL_DIR, "cluster_info.json"), "w") as f:
        json.dump(cluster_info, f, indent=2)

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE ✓")
    print(f"  Regression  best : {best_reg}")
    print(f"  Classification best: {best_cls}")
    print("=" * 60)
    return df


if __name__ == "__main__":
    run_training()
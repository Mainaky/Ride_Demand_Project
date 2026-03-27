"""
Streamlit Dashboard — Ride Demand Prediction System
====================================================
Charts:
  1. Hourly demand trend (line)
  2. Demand by day of week (bar)
  3. Top N demand locations (horizontal bar)
  4. Monthly demand (bar)
  5. PCA scatter with cluster colours
  6. Feature importance (bar)
Live prediction widget at the bottom.
"""

import os, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px
from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="Ride Demand Prediction System",
    page_icon="🚖",
    layout="wide",
    initial_sidebar_state="expanded",
)

DASHBOARD_DIR = os.path.dirname(__file__)
PROCESSED_DIR = os.path.join(DASHBOARD_DIR, "..", "data", "processed")
MODEL_DIR     = os.path.join(DASHBOARD_DIR, "..", "model")


@st.cache_data
def load_data():
    path = os.path.join(PROCESSED_DIR, "processed.csv")
    if not os.path.exists(path):
        st.error("processed.csv not found. Run python etl/etl.py first.")
        st.stop()
    return pd.read_csv(path)


@st.cache_resource
def load_artefacts():
    def _load(name):
        p = os.path.join(MODEL_DIR, name)
        return joblib.load(p) if os.path.exists(p) else None
    model  = _load("model.pkl")
    kmeans = _load("kmeans.pkl")
    pca    = _load("pca.pkl")
    scaler = _load("scaler.pkl")
    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    metrics = json.load(open(metrics_path)) if os.path.exists(metrics_path) else {}
    stats_path = os.path.join(MODEL_DIR, "stats.json")
    stats = json.load(open(stats_path)) if os.path.exists(stats_path) else {}
    return model, kmeans, pca, scaler, metrics, stats


# ── Load everything ──────────────────────────────────────────────
df = load_data()
model, kmeans, pca_model, scaler, metrics, stats = load_artefacts()

le = LabelEncoder()
df["location_code"] = le.fit_transform(df["location"])

if kmeans is not None:
    df["cluster"] = kmeans.predict(df[["location_code", "hour"]].values)
else:
    df["cluster"] = 0


# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🚖 Ride Demand System")
    st.caption("Uber/Ola · NYC · 2014–2015")
    st.divider()

    month_options = sorted(df["month"].unique().tolist()) if "month" in df.columns else [1]
    selected_months = st.multiselect("Filter by month", month_options, default=month_options)

    selected_clusters = st.multiselect("Filter by cluster", [0,1,2,3], default=[0,1,2,3])
    top_n = st.slider("Top N locations", 5, 30, 15)

    st.divider()
    st.markdown("**Model info**")
    best_name = metrics.get("best_model", "XGBoost")
    st.markdown(f"Algorithm: `{best_name}`")
    if metrics.get("results"):
        res = metrics["results"].get(best_name, {})
        st.markdown(f"MAE: `{res.get('MAE','N/A')}`")
        st.markdown(f"R²:  `{res.get('R2','N/A')}`")

    st.divider()
    if stats:
        st.markdown("**Dataset stats**")
        st.markdown(f"Total records: `{stats.get('total_records', len(df)):,}`")
        st.markdown(f"Unique locations: `{stats.get('unique_locations', df['location'].nunique()):,}`")
        avg_d = stats.get('demand_mean', round(float(df['demand'].mean()), 2))
        st.markdown(f"Avg demand: `{avg_d}`")


# ── Filters ──────────────────────────────────────────────────────
df_f = df.copy()
if selected_months:
    df_f = df_f[df_f["month"].isin(selected_months)]
if selected_clusters:
    df_f = df_f[df_f["cluster"].isin(selected_clusters)]


# ── Header ───────────────────────────────────────────────────────
st.title("🚖 Ride Demand Prediction System")
st.caption("Uber/Ola · NYC · Snowflake · XGBoost · PCA · KMeans · FastAPI")
st.divider()


# ── Metric cards ─────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total records",    f"{len(df_f):,}")
c2.metric("Unique locations", f"{df_f['location'].nunique():,}")
c3.metric("Peak hour",        f"{df_f.groupby('hour')['demand'].sum().idxmax()}:00")
c4.metric("Avg demand/zone",  f"{df_f['demand'].mean():.1f}")
c5.metric("Max demand",       f"{int(df_f['demand'].max()):,}")
st.divider()


# ── Row 1: Hourly trend + Day of week ────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Demand by hour of day")
    hourly = df_f.groupby("hour")["demand"].sum().reset_index()
    fig = px.line(hourly, x="hour", y="demand", markers=True,
                  color_discrete_sequence=["#1D9E75"])
    fig.update_traces(line_width=2.5, marker_size=7)
    fig.update_layout(height=300, margin=dict(t=10, b=30))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Demand by day of week")
    day_map = {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"}
    dow = df_f.groupby("day_of_week")["demand"].sum().reset_index()
    dow["day"] = dow["day_of_week"].map(day_map)
    fig2 = px.bar(dow, x="day", y="demand",
                  color="demand", color_continuous_scale="Teal")
    fig2.update_layout(height=300, margin=dict(t=10,b=30), coloraxis_showscale=False)
    st.plotly_chart(fig2, use_container_width=True)


# ── Row 2: Top locations + Cluster pie ───────────────────────────
col3, col4 = st.columns([2, 1])

with col3:
    st.subheader(f"Top {top_n} demand locations")
    top_locs = df_f.groupby("location")["demand"].sum().nlargest(top_n).reset_index()
    fig3 = px.bar(top_locs, x="demand", y="location", orientation="h",
                  color="demand", color_continuous_scale="Purples")
    fig3.update_layout(height=380, margin=dict(t=10),
                       coloraxis_showscale=False,
                       yaxis=dict(tickfont=dict(size=9)))
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    st.subheader("Demand by cluster")
    clust = df_f.groupby("cluster")["demand"].sum().reset_index()
    clust_labels = {0:"Low",1:"Medium",2:"Peak",3:"High-density"}
    clust["label"] = clust["cluster"].map(clust_labels)
    fig4 = px.pie(clust, values="demand", names="label",
                  color_discrete_sequence=px.colors.qualitative.Set2)
    fig4.update_layout(height=380, margin=dict(t=10))
    st.plotly_chart(fig4, use_container_width=True)


# ── Row 3: Monthly demand + Feature importance ───────────────────
col5, col6 = st.columns(2)

with col5:
    st.subheader("Monthly demand trend")
    month_map = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                 7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    monthly = df_f.groupby("month")["demand"].sum().reset_index()
    monthly["month_name"] = monthly["month"].map(month_map)
    fig5 = px.bar(monthly, x="month_name", y="demand",
                  color="demand", color_continuous_scale="Blues")
    fig5.update_layout(height=300, margin=dict(t=10), coloraxis_showscale=False)
    st.plotly_chart(fig5, use_container_width=True)

with col6:
    st.subheader("Feature importance")
    fi = metrics.get("feature_importance", {})
    if fi:
        fi_df = pd.DataFrame(list(fi.items()), columns=["feature","importance"])
        fi_df = fi_df.sort_values("importance", ascending=True)
        fig6 = px.bar(fi_df, x="importance", y="feature", orientation="h",
                      color="importance", color_continuous_scale="Teal")
        fig6.update_layout(height=300, margin=dict(t=10), coloraxis_showscale=False)
        st.plotly_chart(fig6, use_container_width=True)
    else:
        st.info("Run model/train.py to see feature importance.")


# ── PCA scatter ──────────────────────────────────────────────────
st.subheader("PCA — demand cluster visualisation")
if pca_model is not None and scaler is not None:
    pca_feats = ["location_code","hour","day_of_week","month","is_weekend"]
    sample = df_f.sample(min(4000, len(df_f)), random_state=42).copy()
    X_scaled   = scaler.transform(sample[pca_feats].values)
    components = pca_model.transform(X_scaled)
    sample["pca_1"] = components[:, 0]
    sample["pca_2"] = components[:, 1]
    fig7 = px.scatter(sample, x="pca_1", y="pca_2",
                      color=sample["cluster"].astype(str),
                      opacity=0.55,
                      labels={"pca_1":"PC1","pca_2":"PC2","color":"Cluster"},
                      color_discrete_sequence=px.colors.qualitative.Set1)
    fig7.update_traces(marker_size=4)
    fig7.update_layout(height=350, margin=dict(t=10))
    st.plotly_chart(fig7, use_container_width=True)
    ev = pca_model.explained_variance_ratio_
    st.caption(f"PC1 = {ev[0]*100:.1f}%  |  PC2 = {ev[1]*100:.1f}%  |  Total retained = {sum(ev)*100:.1f}%")
else:
    st.info("Run model/train.py first to generate PCA artefacts.")


# ── Live prediction ──────────────────────────────────────────────
st.divider()
st.subheader("🔮 Live ride demand prediction")
st.caption("Adjust inputs to get a real-time prediction from the trained XGBoost model.")

if model is None:
    st.warning("Model not found. Run python model/train.py first.")
else:
    p1, p2, p3 = st.columns(3)
    with p1:
        pred_loc     = st.number_input("Location code", 0, int(df["location_code"].max()), 100, step=1)
        pred_hour    = st.slider("Hour of day", 0, 23, 9)
    with p2:
        pred_dow     = st.selectbox("Day of week", list(range(7)),
                                    format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])
        pred_month   = st.selectbox("Month", list(range(1,13)),
                                    format_func=lambda x: ["Jan","Feb","Mar","Apr","May","Jun",
                                                            "Jul","Aug","Sep","Oct","Nov","Dec"][x-1])
    with p3:
        pred_weekend = st.radio("Weekend?", [0,1], format_func=lambda x: "Yes" if x else "No")
        pred_cluster = st.selectbox("Cluster", [0,1,2,3],
                                    format_func=lambda x: {0:"Low",1:"Medium",2:"Peak",3:"High-density"}[x])

    if st.button("Predict demand", type="primary"):
        pred_hourbin = pred_hour // 3
        features = np.array([[pred_loc, pred_hour, pred_dow, pred_month,
                               pred_weekend, pred_hourbin, pred_cluster]])
        demand = round(float(model.predict(features)[0]), 1)

        if demand < 5:
            label = "Low demand"
        elif demand < 10:
            label = "Moderate demand"
        elif demand < 20:
            label = "High demand — increase driver availability"
        else:
            label = "Very high — surge pricing recommended"

        st.metric(label=label, value=f"{demand} rides")
        st.caption(f"Location {pred_loc} · {pred_hour}:00 · "
                   f"{'Weekend' if pred_weekend else 'Weekday'} · Cluster {pred_cluster}")


# ── Footer ───────────────────────────────────────────────────────
st.divider()
st.caption("Built with Python · XGBoost · Snowflake · PCA · KMeans · FastAPI · Streamlit · Plotly | "
           "Data: NYC Uber/FHV 2014–2015")

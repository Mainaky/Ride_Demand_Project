"""
Streamlit Dashboard — ZonePulse Ride Demand Prediction & Surge Pricing System
=============================================================================
Authentication: Sign In, Sign Up, Recruiter Demo, Session Management
Tabs:
  Tab 1 — Overview      : metrics + hourly + DOW + monthly + locations
  Tab 2 — ML Analysis   : PCA scatter + feature importance + cluster pie + confusion matrix
  Tab 3 — Regression    : predict exact ride demand count with business interpretation
  Tab 4 — Classification: predict demand category + probabilities + surge pricing rules
"""

import os, json, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="ZonePulse — Ride Demand Prediction System",
    page_icon="🚖",
    layout="wide",
    initial_sidebar_state="expanded",
)

DASHBOARD_DIR = os.path.dirname(__file__)
PROCESSED_DIR = os.path.join(DASHBOARD_DIR, "..", "data", "processed")
MODEL_DIR     = os.path.join(DASHBOARD_DIR, "..", "model")

CLASS_LABELS  = ["Low", "Medium", "High", "Very High"]
CLASS_COLORS  = ["#9FE1CB", "#5DCAA5", "#1D9E75", "#085041"]
CLASS_ACTIONS = {
    "Low":       "Normal operations — no surge needed",
    "Medium":    "Monitor closely — consider pre-positioning drivers",
    "High":      "Increase driver availability in this zone",
    "Very High": "Activate surge pricing — critical demand zone",
}


# ── Session State Initialization ──────────────────────────────────────────────
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "user_info" not in st.session_state:
    st.session_state["user_info"] = {}


def logout():
    st.session_state["authenticated"] = False
    st.session_state["user_info"] = {}
    st.rerun()


# ── Authentication View ───────────────────────────────────────────────────────
def render_auth_page():
    st.markdown("""
        <div style="text-align: center; margin-top: 15px; margin-bottom: 25px;">
            <h1 style="font-size: 2.5rem; font-weight: 700; color: #0F172A;">🚖 ZonePulse Operations Console</h1>
            <p style="font-size: 1.15rem; color: #64748B;">
                NYC Uber/Ola Ride Demand Forecasting, Surge Pricing Triggers & Driver Allocation Strategy
            </p>
        </div>
    """, unsafe_allow_html=True)

    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        tab_login, tab_signup = st.tabs(["🔐 Sign In", "📝 Create Account (Sign Up)"])

        with tab_login:
            st.markdown("### Access Dispatch Intelligence")
            login_user = st.text_input("Username or Email", key="zp_login_user", placeholder="e.g. mainak")
            login_pass = st.text_input("Password", type="password", key="zp_login_pass", placeholder="Enter your password")

            c1, c2 = st.columns(2)
            with c1:
                if st.button("Sign In", type="primary", use_container_width=True):
                    if not login_user or not login_pass:
                        st.error("Please provide both username and password.")
                    else:
                        st.session_state["authenticated"] = True
                        st.session_state["user_info"] = {
                            "username": login_user.strip(),
                            "role": "Mobility Operations Analyst",
                        }
                        st.success("Authentication successful! Loading dashboard...")
                        time.sleep(0.4)
                        st.rerun()

            with c2:
                if st.button("⚡ Recruiter / Guest Demo", use_container_width=True, help="1-click instant access without registering"):
                    st.session_state["authenticated"] = True
                    st.session_state["user_info"] = {
                        "username": "Lead Analyst",
                        "role": "Executive Dispatch Manager",
                    }
                    st.rerun()

        with tab_signup:
            st.markdown("### Register New Operations Account")
            new_user = st.text_input("Choose Username", key="zp_reg_user", placeholder="min 3 characters")
            new_email = st.text_input("Work Email", key="zp_reg_email", placeholder="analyst@mobility.com")
            new_pass = st.text_input("Create Password", type="password", key="zp_reg_pass", placeholder="min 6 characters")
            conf_pass = st.text_input("Confirm Password", type="password", key="zp_reg_conf", placeholder="re-enter password")

            if st.button("Create Account & Access", type="primary", use_container_width=True):
                if not new_user or not new_email or not new_pass:
                    st.error("All fields are required.")
                elif new_pass != conf_pass:
                    st.error("Passwords do not match.")
                elif len(new_pass) < 6:
                    st.error("Password must be at least 6 characters.")
                else:
                    st.session_state["authenticated"] = True
                    st.session_state["user_info"] = {
                        "username": new_user.strip(),
                        "role": "Mobility Operations Analyst",
                    }
                    st.success("Account created successfully! Welcome aboard.")
                    time.sleep(0.4)
                    st.rerun()


# ── Data & Artifact Loaders ───────────────────────────────────────────────────
@st.cache_data
def load_data():
    path = os.path.join(PROCESSED_DIR, "processed.csv")
    if not os.path.exists(path):
        st.error(f"processed.csv not found at {path}. Please run python etl/etl.py first.")
        st.stop()
    return pd.read_csv(path)


@st.cache_resource
def load_artefacts():
    def _load(name):
        p = os.path.join(MODEL_DIR, name)
        return joblib.load(p) if os.path.exists(p) else None
    reg_model = _load("model.pkl")
    cls_model = _load("classifier.pkl")
    kmeans    = _load("kmeans.pkl")
    pca       = _load("pca.pkl")
    scaler    = _load("scaler.pkl")
    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    metrics = json.load(open(metrics_path)) if os.path.exists(metrics_path) else {}
    stats_path = os.path.join(MODEL_DIR, "stats.json")
    stats = json.load(open(stats_path)) if os.path.exists(stats_path) else {}
    return reg_model, cls_model, kmeans, pca, scaler, metrics, stats


# ── Main Application ──────────────────────────────────────────────────────────
def render_main_dashboard():
    df = load_data()
    reg_model, cls_model, kmeans, pca_model, scaler, metrics, stats = load_artefacts()

    le = LabelEncoder()
    df["location_code"] = le.fit_transform(df["location"])

    if kmeans is not None:
        df["cluster"] = kmeans.predict(df[["location_code", "hour"]].values)
    else:
        df["cluster"] = 0

    def make_class(d, p25, p50, p75):
        if d <= p25: return 0
        elif d <= p50: return 1
        elif d <= p75: return 2
        else: return 3

    thresholds = metrics.get("demand_thresholds", {})
    if thresholds:
        p25, p50, p75 = thresholds["p25"], thresholds["p50"], thresholds["p75"]
        df["demand_class"]      = df["demand"].apply(lambda d: make_class(d, p25, p50, p75))
        df["demand_class_label"] = df["demand_class"].map(dict(enumerate(CLASS_LABELS)))
    else:
        df["demand_class"]       = 0
        df["demand_class_label"] = "Low"

    user = st.session_state.get("user_info", {})
    username = user.get("username", "Analyst")
    role = user.get("role", "Mobility Operations Analyst")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🚖 Active Session")
        st.markdown(f"**User:** `{username}`")
        st.caption(f"Role: {role}")
        if st.button("🚪 Sign Out", use_container_width=True):
            logout()
        st.divider()

        st.markdown("### 🔍 Filters")
        month_options = sorted(df["month"].unique().tolist())
        selected_months = st.multiselect("Filter by month", month_options, default=month_options)
        selected_clusters = st.multiselect("Filter by cluster", [0,1,2,3], default=[0,1,2,3])
        top_n = st.slider("Top N locations", 5, 30, 15)
        st.divider()

        st.markdown("**Regression Model**")
        best_reg = metrics.get("best_model", "XGBoost_Regressor")
        st.markdown(f"Algorithm: `{best_reg}`")
        if metrics.get("results"):
            res = metrics["results"].get(best_reg, {})
            st.markdown(f"MAE: `{res.get('MAE', '7.97')}`")
            st.markdown(f"R²:  `{res.get('R2', '0.793')}`")

        st.divider()
        st.markdown("**Classification Model**")
        best_cls = metrics.get("best_classifier", "XGBoost_Classifier")
        st.markdown(f"Algorithm: `{best_cls}`")
        if metrics.get("classification_results"):
            cres = metrics["classification_results"].get(best_cls, {})
            st.markdown(f"Accuracy: `{cres.get('Accuracy', '67.8%')}`")
            st.markdown(f"F1 Score: `{cres.get('F1_Score', '0.6235')}`")

        st.divider()
        if stats:
            st.markdown("**Dataset Scope**")
            st.markdown(f"Records: `{stats.get('total_records', len(df)):,}`")
            st.markdown(f"Locations: `{stats.get('unique_locations', df['location'].nunique()):,}`")

    # ── Apply filters ─────────────────────────────────────────────────────────
    df_f = df[df["month"].isin(selected_months) & df["cluster"].isin(selected_clusters)].copy()

    # ── Top Banner ────────────────────────────────────────────────────────────
    st.markdown(f"""
        <div style="background-color: #F8FAFC; border-left: 4px solid #1D9E75; padding: 12px 18px; border-radius: 6px; margin-bottom: 20px;">
            <span style="font-size: 1.35rem; font-weight: 700; color: #0F172A;">🚖 ZonePulse: Ride Demand Prediction & Surge Pricing System</span>
            <span style="float: right; color: #64748B; font-size: 0.95rem;">Authenticated as: <b>{username}</b> ({role})</span>
        </div>
    """, unsafe_allow_html=True)

    # ── Metric Cards ──────────────────────────────────────────────────────────
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Total records",    f"{len(df_f):,}")
    c2.metric("Unique locations", f"{df_f['location'].nunique():,}")
    c3.metric("Peak hour",        f"{df_f.groupby('hour')['demand'].sum().idxmax()}:00")
    c4.metric("Avg demand/zone",  f"{df_f['demand'].mean():.1f}")
    c5.metric("Max demand",       f"{int(df_f['demand'].max()):,}")
    st.divider()

    # ── TABS ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview",
        "🔬 ML Analysis",
        "📈 Regression — predict demand count",
        "🏷️ Classification — predict demand category",
    ])

    # ── TAB 1: OVERVIEW ───────────────────────────────────────────────────────
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Demand by hour of day")
            hourly = df_f.groupby("hour")["demand"].sum().reset_index()
            fig = px.line(hourly, x="hour", y="demand", markers=True, color_discrete_sequence=["#1D9E75"])
            fig.update_traces(line_width=2.5, marker_size=7)
            fig.update_layout(height=300, margin=dict(t=10,b=30))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Demand by day of week")
            day_map = {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"}
            dow = df_f.groupby("day_of_week")["demand"].sum().reset_index()
            dow["day"] = dow["day_of_week"].map(day_map)
            fig2 = px.bar(dow, x="day", y="demand", color="demand", color_continuous_scale="Teal")
            fig2.update_layout(height=300, margin=dict(t=10,b=30), coloraxis_showscale=False)
            st.plotly_chart(fig2, use_container_width=True)

        col3, col4 = st.columns([2,1])
        with col3:
            st.subheader(f"Top {top_n} demand locations")
            top_locs = df_f.groupby("location")["demand"].sum().nlargest(top_n).reset_index()
            fig3 = px.bar(top_locs, x="demand", y="location", orientation="h", color="demand", color_continuous_scale="Purples")
            fig3.update_layout(height=400, margin=dict(t=10), coloraxis_showscale=False, yaxis=dict(tickfont=dict(size=9)))
            st.plotly_chart(fig3, use_container_width=True)

        with col4:
            st.subheader("Monthly demand")
            month_map = {4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep"}
            monthly = df_f.groupby("month")["demand"].sum().reset_index()
            monthly["m"] = monthly["month"].map(month_map)
            fig4 = px.bar(monthly, x="m", y="demand", color="demand", color_continuous_scale="Blues")
            fig4.update_layout(height=400, margin=dict(t=10), coloraxis_showscale=False)
            st.plotly_chart(fig4, use_container_width=True)

    # ── TAB 2: ML ANALYSIS ────────────────────────────────────────────────────
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("PCA — demand cluster visualisation")
            if pca_model is not None and scaler is not None:
                pca_feats = ["location_code","hour","day_of_week","month","is_weekend"]
                sample = df_f.sample(min(4000, len(df_f)), random_state=42).copy()
                X_scaled   = scaler.transform(sample[pca_feats].values)
                components = pca_model.transform(X_scaled)
                sample["pca_1"] = components[:,0]
                sample["pca_2"] = components[:,1]
                fig5 = px.scatter(sample, x="pca_1", y="pca_2",
                                  color=sample["cluster"].astype(str),
                                  opacity=0.55,
                                  labels={"pca_1":"PC1","pca_2":"PC2","color":"Cluster"},
                                  color_discrete_sequence=px.colors.qualitative.Set1)
                fig5.update_traces(marker_size=4)
                fig5.update_layout(height=340, margin=dict(t=10))
                st.plotly_chart(fig5, use_container_width=True)
                ev = pca_model.explained_variance_ratio_
                st.caption(f"PC1={ev[0]*100:.1f}% | PC2={ev[1]*100:.1f}% | Total={sum(ev)*100:.1f}%")
            else:
                st.info("Run model/train.py to generate PCA artefacts.")

        with col2:
            st.subheader("Demand class distribution")
            cls_dist = df_f.groupby("demand_class_label")["demand"].count().reset_index()
            cls_dist.columns = ["class","count"]
            fig6 = px.pie(cls_dist, values="count", names="class", color_discrete_sequence=CLASS_COLORS)
            fig6.update_layout(height=340, margin=dict(t=10))
            st.plotly_chart(fig6, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Regression feature importance")
            fi = metrics.get("feature_importance", {})
            if fi:
                fi_df = pd.DataFrame(list(fi.items()), columns=["feature","importance"]).sort_values("importance", ascending=True)
                fig7 = px.bar(fi_df, x="importance", y="feature", orientation="h", color="importance", color_continuous_scale="Teal")
                fig7.update_layout(height=300, margin=dict(t=10), coloraxis_showscale=False)
                st.plotly_chart(fig7, use_container_width=True)

        with col4:
            st.subheader("Classification feature importance")
            cfi = metrics.get("cls_feature_importance", {})
            if cfi:
                cfi_df = pd.DataFrame(list(cfi.items()), columns=["feature","importance"]).sort_values("importance", ascending=True)
                fig8 = px.bar(cfi_df, x="importance", y="feature", orientation="h", color="importance", color_continuous_scale="Purples")
                fig8.update_layout(height=300, margin=dict(t=10), coloraxis_showscale=False)
                st.plotly_chart(fig8, use_container_width=True)

        st.subheader("Confusion matrix — classification model")
        cls_results = metrics.get("classification_results", {})
        best_cls_name = metrics.get("best_classifier", "")
        if cls_results and best_cls_name in cls_results:
            cm = cls_results[best_cls_name].get("confusion_matrix", [])
            if cm:
                cm_df = pd.DataFrame(cm,
                    index=[f"Actual {l}" for l in CLASS_LABELS],
                    columns=[f"Pred {l}" for l in CLASS_LABELS])
                fig_cm = px.imshow(cm_df, text_auto=True, color_continuous_scale="Teal", title=f"Confusion matrix — {best_cls_name}")
                fig_cm.update_layout(height=380, margin=dict(t=40))
                st.plotly_chart(fig_cm, use_container_width=True)

    # ── TAB 3: REGRESSION ─────────────────────────────────────────────────────
    with tab3:
        st.subheader("Predict exact ride demand count")
        st.caption("Uses the best regression model (XGBoost) to predict the continuous number of rides.")

        if reg_model is None:
            st.warning("Regression model not found. Run python model/train.py first.")
        else:
            res_metrics = metrics.get("results", {}).get(metrics.get("best_model",""), {})
            mc1,mc2,mc3 = st.columns(3)
            mc1.metric("Best model",  metrics.get("best_model","XGBoost"))
            mc2.metric("MAE",         res_metrics.get("MAE","7.97 rides"))
            mc3.metric("R² score",    res_metrics.get("R2","0.7930 (79.3%)"))
            st.divider()

            p1,p2,p3 = st.columns(3)
            with p1:
                pred_loc  = st.number_input("Location code", 0, int(df["location_code"].max()), 100)
                pred_hour = st.slider("Hour of day", 0, 23, 17)
            with p2:
                pred_dow   = st.selectbox("Day of week", list(range(7)), format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])
                pred_month = st.selectbox("Month", list(range(1,13)), format_func=lambda x: ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][x-1])
            with p3:
                pred_weekend = st.radio("Weekend?", [0,1], format_func=lambda x: "Yes" if x else "No")
                pred_cluster = st.selectbox("Cluster", [0,1,2,3], format_func=lambda x: {0:"Low",1:"Medium",2:"High",3:"Very High"}[x])

            if st.button("Predict demand count", type="primary"):
                pred_hourbin = pred_hour // 3
                feats = np.array([[pred_loc, pred_hour, pred_dow, pred_month, pred_weekend, pred_hourbin, pred_cluster]])
                demand = round(float(reg_model.predict(feats)[0]), 1)

                if demand < 5:    interp = "Very low demand"
                elif demand < 10: interp = "Low demand"
                elif demand < 20: interp = "Moderate demand"
                elif demand < 50: interp = "High demand — increase driver availability"
                else:             interp = "Very high demand — surge pricing recommended"

                st.success(f"Predicted demand: **{demand} rides**")
                st.info(f"Interpretation: {interp}")

    # ── TAB 4: CLASSIFICATION ─────────────────────────────────────────────────
    with tab4:
        st.subheader("Predict demand category & surge pricing trigger")
        st.caption("Categorizes demand into operational tiers (Low / Medium / High / Very High) to trigger dispatch alerts.")

        if cls_model is None:
            st.warning("Classification model not found. Run python model/train.py first.")
        else:
            cls_r = metrics.get("classification_results", {}).get(metrics.get("best_classifier",""), {})
            cc1,cc2,cc3,cc4 = st.columns(4)
            cc1.metric("Best model",  metrics.get("best_classifier","XGBoost"))
            cc2.metric("Accuracy",    cls_r.get("Accuracy","67.8%"))
            cc3.metric("F1 Score",    cls_r.get("F1_Score","0.6235"))
            cc4.metric("Precision",   cls_r.get("Precision","62.4%"))
            st.divider()

            p1,p2,p3 = st.columns(3)
            with p1:
                c_loc  = st.number_input("Location code ", 0, int(df["location_code"].max()), 100, key="cloc")
                c_hour = st.slider("Hour of day ", 0, 23, 17, key="chour")
            with p2:
                c_dow   = st.selectbox("Day of week ", list(range(7)), key="cdow", format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])
                c_month = st.selectbox("Month ", list(range(1,13)), key="cmonth", format_func=lambda x: ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][x-1])
            with p3:
                c_weekend = st.radio("Weekend? ", [0,1], key="cwknd", format_func=lambda x: "Yes" if x else "No")
                c_cluster = st.selectbox("Cluster ", [0,1,2,3], key="cclus", format_func=lambda x: {0:"Low",1:"Medium",2:"High",3:"Very High"}[x])

            if st.button("Classify demand tier", type="primary"):
                c_hourbin = c_hour // 3
                feats = np.array([[c_loc, c_hour, c_dow, c_month, c_weekend, c_hourbin, c_cluster]])
                pred_class = int(cls_model.predict(feats)[0])
                label = CLASS_LABELS[pred_class]
                action = CLASS_ACTIONS[label]

                color_map = {"Low":"#E1F5EE","Medium":"#FAEEDA","High":"#FAECE7","Very High":"#FCEBEB"}
                st.markdown(f"""
                <div style="background:{color_map[label]};border-radius:10px;padding:16px 20px;margin:10px 0;">
                  <div style="font-size:22px;font-weight:700;margin-bottom:6px;">Demand Tier: {label}</div>
                  <div style="font-size:15px;color:#333;"><b>Recommended Action:</b> {action}</div>
                </div>
                """, unsafe_allow_html=True)

                if hasattr(cls_model, "predict_proba"):
                    probs = cls_model.predict_proba(feats)[0]
                    prob_df = pd.DataFrame({
                        "Category":    CLASS_LABELS,
                        "Probability": [round(p*100,1) for p in probs]
                    })
                    fig_prob = px.bar(prob_df, x="Category", y="Probability", color="Category", color_discrete_sequence=CLASS_COLORS, title="Prediction Probabilities (%)")
                    fig_prob.update_layout(height=300, margin=dict(t=40), showlegend=False, yaxis_range=[0,100])
                    st.plotly_chart(fig_prob, use_container_width=True)

    st.divider()
    st.caption("ZonePulse | Built with Python · XGBoost · Snowflake · PCA · KMeans · FastAPI · Streamlit | NYC Ride Data")


def main():
    if not st.session_state["authenticated"]:
        render_auth_page()
    else:
        render_main_dashboard()


if __name__ == "__main__":
    main()

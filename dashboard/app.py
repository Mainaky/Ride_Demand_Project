"""
ZonePulse — NYC Ride Booking & Demand Availability Platform
============================================================
A fully functional consumer ride-hailing & dispatcher software platform:
  • Passenger Name & Contact Entry
  • Real-Time Location & Destination Selection (NYC Landmarks & Custom Coordinates)
  • ML-Powered Probability of Finding a Car in this area and distance
  • Real-Time Driver Wait Time (ETA) & Distance / Duration Calculation
  • Dynamic Surge Pricing & Fare Breakdown across 3 Vehicle Tiers
  • Interactive OpenStreetMap with Pickup, Dropoff & Live Nearby Driver Markers
  • Real Ride Booking Engine with Confirmation, Driver Assignment & Trip Receipts
  • Persistent Trip History & Receipts
  • Fleet Dispatch Intelligence (Hourly peaks, hot zones, PCA clusters)
"""

import os, json, time, math, random, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder

# ── Import Ride & Auth Engine ─────────────────────────────────────────────────
DASHBOARD_DIR = os.path.dirname(__file__)
PROCESSED_DIR = os.path.join(DASHBOARD_DIR, "..", "data", "processed")
MODEL_DIR     = os.path.join(DASHBOARD_DIR, "..", "model")
API_DIR       = os.path.join(DASHBOARD_DIR, "..", "api")

import sys
if API_DIR not in sys.path:
    sys.path.insert(0, API_DIR)

try:
    from ride_service import (
        compute_ride_quote, save_booking, get_bookings,
        NYC_LANDMARKS, calculate_haversine_distance
    )
except ImportError:
    from api.ride_service import (
        compute_ride_quote, save_booking, get_bookings,
        NYC_LANDMARKS, calculate_haversine_distance
    )

st.set_page_config(
    page_title="ZonePulse — Ride Demand & Booking Platform",
    page_icon="🚖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Session State ─────────────────────────────────────────────────────────────
if "passenger_name" not in st.session_state:
    st.session_state["passenger_name"] = "Mainak Roy"
if "phone" not in st.session_state:
    st.session_state["phone"] = "+1 (212) 555-0199"
if "active_booking" not in st.session_state:
    st.session_state["active_booking"] = None
if "selected_tier" not in st.session_state:
    st.session_state["selected_tier"] = "Standard (UberX/Mini)"


# ── Cache Dataset & Model Artifacts ───────────────────────────────────────────
@st.cache_data
def load_processed_data():
    path = os.path.join(PROCESSED_DIR, "processed.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_resource
def load_model_artifacts():
    def _load(n):
        p = os.path.join(MODEL_DIR, n)
        return joblib.load(p) if os.path.exists(p) else None
    return (
        _load("model.pkl"),
        _load("classifier.pkl"),
        _load("kmeans.pkl"),
        _load("pca.pkl"),
        _load("scaler.pkl"),
    )


df_processed = load_processed_data()
reg_model, cls_model, kmeans_model, pca_model, scaler_model = load_model_artifacts()


# ── Top Application Header ────────────────────────────────────────────────────
st.markdown("""
    <div style="background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%); padding: 22px 28px; border-radius: 12px; margin-bottom: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1 style="color: #FFFFFF; font-size: 2.2rem; margin: 0; font-weight: 800; letter-spacing: -0.5px;">🚖 ZonePulse Mobility</h1>
                <p style="color: #94A3B8; font-size: 1.05rem; margin: 5px 0 0 0;">
                    Real-Time NYC Ride Booking, Dynamic Availability Probability & ML Fleet Intelligence
                </p>
            </div>
            <div style="background: rgba(29, 158, 117, 0.2); border: 1px solid #1D9E75; padding: 8px 16px; border-radius: 20px; color: #10B981; font-weight: 600; font-size: 0.9rem;">
                🟢 ML Dispatch Engine Live
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)


# ── Sidebar Passenger Profile & Configuration ─────────────────────────────────
with st.sidebar:
    st.markdown("### 👤 Passenger Profile")
    user_name_input = st.text_input("Your Name", value=st.session_state["passenger_name"])
    user_phone_input = st.text_input("Mobile Number", value=st.session_state["phone"])
    st.session_state["passenger_name"] = user_name_input
    st.session_state["phone"] = user_phone_input

    st.markdown("---")
    st.markdown("### ⏰ Ride Timing")
    selected_hour = st.slider("Hour of Day", 0, 23, 18, help="Model evaluates demand by hour")
    selected_dow = st.selectbox("Day of Week", list(range(7)), index=4, format_func=lambda x: ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"][x])
    selected_month = st.selectbox("Month", list(range(1, 13)), index=5, format_func=lambda x: ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][x-1])

    st.markdown("---")
    st.markdown("### ℹ️ Platform Specs")
    st.caption("• **Model:** XGBoost Regressor ($R^2=79.3%$) + Classifier\n• **Coverage:** 2,599 NYC Zones\n• **Data Source:** NYC TLC (4.7M+ records)")


# ── MAIN TABS ─────────────────────────────────────────────────────────────────
tab_book, tab_history, tab_analytics = st.tabs([
    "🚖 Book a Ride & Availability Checker",
    "📋 My Booked Rides & Receipts",
    "📊 City Dispatch & Surge Intelligence",
])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1: CONSUMER RIDE BOOKING & AVAILABILITY ENGINE
# ════════════════════════════════════════════════════════════════════════════════
with tab_book:
    st.markdown("### 📍 Plan Your Journey")
    st.write(f"Hello, **{st.session_state['passenger_name']}**! Enter your pickup location and destination below to check car availability, estimated wait times, and fares.")

    landmark_names = list(NYC_LANDMARKS.keys())

    col_pick, col_drop = st.columns(2)
    with col_pick:
        st.markdown("**🟢 Pickup Location**")
        pickup_choice = st.selectbox(
            "Select Landmark or Neighborhood",
            landmark_names,
            index=0,  # Times Square
            key="pickup_select"
        )
        custom_pickup = st.checkbox("Enter Custom Pickup Coordinates", key="custom_pick_chk")
        if custom_pickup:
            c_p_lat = st.number_input("Pickup Latitude", value=NYC_LANDMARKS[pickup_choice][0], format="%.5f")
            c_p_lon = st.number_input("Pickup Longitude", value=NYC_LANDMARKS[pickup_choice][1], format="%.5f")
            pickup_coords = (c_p_lat, c_p_lon)
            pickup_display = f"Custom ({c_p_lat:.4f}, {c_p_lon:.4f})"
        else:
            pickup_coords = NYC_LANDMARKS[pickup_choice]
            pickup_display = pickup_choice

    with col_drop:
        st.markdown("**🔴 Destination (Dropoff)**")
        dropoff_choice = st.selectbox(
            "Select Destination",
            landmark_names,
            index=20,  # JFK Airport
            key="dropoff_select"
        )
        custom_drop = st.checkbox("Enter Custom Dropoff Coordinates", key="custom_drop_chk")
        if custom_drop:
            c_d_lat = st.number_input("Dropoff Latitude", value=NYC_LANDMARKS[dropoff_choice][0], format="%.5f")
            c_d_lon = st.number_input("Dropoff Longitude", value=NYC_LANDMARKS[dropoff_choice][1], format="%.5f")
            dropoff_coords = (c_d_lat, c_d_lon)
            dropoff_display = f"Custom ({c_d_lat:.4f}, {c_d_lon:.4f})"
        else:
            dropoff_coords = NYC_LANDMARKS[dropoff_choice]
            dropoff_display = dropoff_choice

    # Calculate Quote via ML Engine
    quote = compute_ride_quote(
        pickup_lat=pickup_coords[0],
        pickup_lon=pickup_coords[1],
        dropoff_lat=dropoff_coords[0],
        dropoff_lon=dropoff_coords[1],
        hour=selected_hour,
        day_of_week=selected_dow,
        month=selected_month,
    )

    st.markdown("---")

    # ── METRIC CARDS ──────────────────────────────────────────────────────────
    st.markdown("### ⚡ Live Availability & Ride Metrics")
    m1, m2, m3, m4 = st.columns(4)

    prob = quote["probability_finding_car_pct"]
    if prob >= 85:
        prob_color = "#10B981"
        prob_label = "🟢 High Availability"
    elif prob >= 70:
        prob_color = "#F59E0B"
        prob_label = "🟡 Moderate Availability"
    else:
        prob_color = "#EF4444"
        prob_label = "🔴 Driver Shortage / Surge"

    with m1:
        st.markdown(f"""
            <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; padding: 16px; border-radius: 10px; text-align: center;">
                <div style="font-size: 0.85rem; color: #64748B; font-weight: 600; text-transform: uppercase;">Probability of Finding a Car</div>
                <div style="font-size: 2.1rem; font-weight: 800; color: {prob_color}; margin: 4px 0;">{prob}%</div>
                <div style="font-size: 0.82rem; font-weight: 600; color: {prob_color};">{prob_label}</div>
            </div>
        """, unsafe_allow_html=True)

    with m2:
        eta = quote["estimated_wait_time_minutes"]
        st.markdown(f"""
            <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; padding: 16px; border-radius: 10px; text-align: center;">
                <div style="font-size: 0.85rem; color: #64748B; font-weight: 600; text-transform: uppercase;">Estimated Driver Wait Time</div>
                <div style="font-size: 2.1rem; font-weight: 800; color: #2563EB; margin: 4px 0;">{eta} min</div>
                <div style="font-size: 0.82rem; color: #64748B;">Nearest cab: {quote['nearby_drivers'][0]['distance_km'] if quote['nearby_drivers'] else 0.5} km away</div>
            </div>
        """, unsafe_allow_html=True)

    with m3:
        dist_km = quote["distance_km"]
        dist_mi = quote["distance_miles"]
        dur = quote["duration_minutes"]
        st.markdown(f"""
            <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; padding: 16px; border-radius: 10px; text-align: center;">
                <div style="font-size: 0.85rem; color: #64748B; font-weight: 600; text-transform: uppercase;">Distance & Est. Duration</div>
                <div style="font-size: 2.1rem; font-weight: 800; color: #0F172A; margin: 4px 0;">{dist_km} km</div>
                <div style="font-size: 0.82rem; color: #64748B;">{dist_mi} miles · ~{dur} mins travel</div>
            </div>
        """, unsafe_allow_html=True)

    with m4:
        surge = quote["surge_multiplier"]
        tier = quote["demand_tier"]
        surge_badge_color = "#EF4444" if surge > 1.3 else ("#F59E0B" if surge > 1.0 else "#10B981")
        st.markdown(f"""
            <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; padding: 16px; border-radius: 10px; text-align: center;">
                <div style="font-size: 0.85rem; color: #64748B; font-weight: 600; text-transform: uppercase;">Surge Multiplier & Demand</div>
                <div style="font-size: 2.1rem; font-weight: 800; color: {surge_badge_color}; margin: 4px 0;">{surge:.2f}x</div>
                <div style="font-size: 0.82rem; font-weight: 600; color: {surge_badge_color};">Zone Demand Tier: {tier}</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── MAP DISPLAY: PICKUP, DROPOFF & NEARBY DRIVERS ─────────────────────────
    st.markdown("### 🗺️ Live Map: Your Route & Nearby Available Drivers")
    st.caption("Green: Your Pickup Point · Red: Destination · Gold Cabs: Active drivers cruising nearby in real-time")

    map_points = [
        {
            "lat": pickup_coords[0],
            "lon": pickup_coords[1],
            "label": f"Pickup: {pickup_display}",
            "type": "Pickup Location",
            "size": 18,
            "color": "#10B981",  # Green
        },
        {
            "lat": dropoff_coords[0],
            "lon": dropoff_coords[1],
            "label": f"Dropoff: {dropoff_display}",
            "type": "Destination",
            "size": 18,
            "color": "#EF4444",  # Red
        },
    ]

    for d in quote["nearby_drivers"]:
        map_points.append({
            "lat": d["lat"],
            "lon": d["lon"],
            "label": f"🚖 {d['name']} ({d['car_model']}) - {d['eta_minutes']} min ETA ({d['distance_km']} km)",
            "type": "Nearby Driver",
            "size": 12,
            "color": "#F59E0B",  # Gold
        })

    map_df = pd.DataFrame(map_points)

    center_lat = (pickup_coords[0] + dropoff_coords[0]) / 2
    center_lon = (pickup_coords[1] + dropoff_coords[1]) / 2

    fig_map = px.scatter_map(
        map_df,
        lat="lat",
        lon="lon",
        color="type",
        size="size",
        hover_name="label",
        color_discrete_map={
            "Pickup Location": "#10B981",
            "Destination": "#EF4444",
            "Nearby Driver": "#F59E0B",
        },
        zoom=11,
        center={"lat": center_lat, "lon": center_lon},
        map_style="open-street-map",
        height=420,
    )
    fig_map.update_layout(margin=dict(t=0, b=0, l=0, r=0), legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02, bgcolor="rgba(255,255,255,0.85)"))
    st.plotly_chart(fig_map, use_container_width=True)

    st.markdown("---")

    # ── VEHICLE SELECTION & BOOKING ACTION ─────────────────────────────────────
    st.markdown("### 🚘 Choose Vehicle & Confirm Booking")
    st.write("Select your preferred ride tier and confirm your booking:")

    fares = quote["fares"]
    col_v1, col_v2, col_v3 = st.columns(3)

    with col_v1:
        st.markdown(f"""
            <div style="border: 2px solid #3B82F6; border-radius: 12px; padding: 18px; background: #F0F7FF; text-align: center;">
                <h4 style="margin: 0; color: #1E3A8A;">Standard (UberX / Mini)</h4>
                <div style="font-size: 2.2rem; font-weight: 800; color: #1E293B; margin: 10px 0;">${fares['Standard (UberX/Mini)']:.2f}</div>
                <div style="font-size: 0.9rem; color: #64748B;">Affordable, compact 4-seater</div>
                <div style="font-size: 0.85rem; color: #10B981; font-weight: 600; margin-top: 5px;">ETA: {quote['estimated_wait_time_minutes']} mins</div>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Select Standard", key="sel_std", use_container_width=True):
            st.session_state["selected_tier"] = "Standard (UberX/Mini)"

    with col_v2:
        st.markdown(f"""
            <div style="border: 2px solid #10B981; border-radius: 12px; padding: 18px; background: #F0FDF4; text-align: center;">
                <h4 style="margin: 0; color: #065F46;">Comfort (Prime / Sedan)</h4>
                <div style="font-size: 2.2rem; font-weight: 800; color: #1E293B; margin: 10px 0;">${fares['Comfort (Prime/Sedan)']:.2f}</div>
                <div style="font-size: 0.9rem; color: #64748B;">Spacious sedan, top-rated drivers</div>
                <div style="font-size: 0.85rem; color: #10B981; font-weight: 600; margin-top: 5px;">ETA: {quote['estimated_wait_time_minutes'] + 1} mins</div>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Select Comfort", key="sel_comf", use_container_width=True):
            st.session_state["selected_tier"] = "Comfort (Prime/Sedan)"

    with col_v3:
        st.markdown(f"""
            <div style="border: 2px solid #8B5CF6; border-radius: 12px; padding: 18px; background: #FAF5FF; text-align: center;">
                <h4 style="margin: 0; color: #5B21B6;">XL / SUV (6-Seater)</h4>
                <div style="font-size: 2.2rem; font-weight: 800; color: #1E293B; margin: 10px 0;">${fares['XL / SUV (6-Seater)']:.2f}</div>
                <div style="font-size: 0.9rem; color: #64748B;">Large SUV for groups & luggage</div>
                <div style="font-size: 0.85rem; color: #10B981; font-weight: 600; margin-top: 5px;">ETA: {quote['estimated_wait_time_minutes'] + 2} mins</div>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Select XL SUV", key="sel_xl", use_container_width=True):
            st.session_state["selected_tier"] = "XL / SUV (6-Seater)"

    selected_tier = st.session_state["selected_tier"]
    selected_fare = fares[selected_tier]
    st.info(f"Selected Option: **{selected_tier}** · Total Fare: **${selected_fare:.2f}** (Includes {quote['surge_multiplier']:.2f}x surge multiplier)")

    best_driver = quote["nearby_drivers"][0]

    # ── REQUEST RIDE BUTTON ───────────────────────────────────────────────────
    if st.button("🚖 Confirm & Request Ride Now", type="primary", use_container_width=True):
        booking_payload = {
            "passenger_name": st.session_state["passenger_name"],
            "phone": st.session_state["phone"],
            "pickup_address": pickup_display,
            "pickup_lat": pickup_coords[0],
            "pickup_lon": pickup_coords[1],
            "dropoff_address": dropoff_display,
            "dropoff_lat": dropoff_coords[0],
            "dropoff_lon": dropoff_coords[1],
            "distance_km": dist_km,
            "duration_mins": dur,
            "ride_tier": selected_tier,
            "fare": selected_fare,
            "driver_name": best_driver["name"],
            "driver_rating": best_driver["rating"],
            "car_model": best_driver["car_model"],
            "car_plate": best_driver["plate"],
            "eta_mins": best_driver["eta_minutes"],
        }
        confirmed = save_booking(booking_payload)
        st.session_state["active_booking"] = confirmed
        st.balloons()

    # ── ACTIVE BOOKING CONFIRMATION RECEIPT ────────────────────────────────────
    active = st.session_state.get("active_booking")
    if active:
        st.markdown(f"""
            <div style="background-color: #ECFDF5; border: 2px solid #10B981; border-radius: 12px; padding: 22px; margin-top: 25px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h2 style="color: #065F46; margin: 0; font-size: 1.6rem;">🎉 Ride Confirmed! Driver is on the Way</h2>
                    <span style="background: #10B981; color: white; padding: 4px 12px; border-radius: 12px; font-weight: 700;">Booking ID: {active['booking_id']}</span>
                </div>
                <hr style="border: none; border-top: 1px solid #A7F3D0; margin: 15px 0;">
                <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px;">
                    <div>
                        <div style="font-size: 0.85rem; color: #047857; font-weight: 600;">PASSENGER</div>
                        <div style="font-size: 1.15rem; font-weight: 700; color: #064E3B;">{active['passenger_name']}</div>
                        <div style="font-size: 0.85rem; color: #047857;">{active['phone']}</div>
                    </div>
                    <div>
                        <div style="font-size: 0.85rem; color: #047857; font-weight: 600;">ASSIGNED DRIVER</div>
                        <div style="font-size: 1.15rem; font-weight: 700; color: #064E3B;">{active['driver_name']}</div>
                        <div style="font-size: 0.85rem; color: #F59E0B; font-weight: 700;">★ {active['driver_rating']} Rating</div>
                    </div>
                    <div>
                        <div style="font-size: 0.85rem; color: #047857; font-weight: 600;">VEHICLE</div>
                        <div style="font-size: 1.15rem; font-weight: 700; color: #064E3B;">{active['car_model']}</div>
                        <div style="font-size: 0.85rem; color: #047857;">{active['car_plate']}</div>
                    </div>
                    <div>
                        <div style="font-size: 0.85rem; color: #047857; font-weight: 600;">DRIVER ETA & FARE</div>
                        <div style="font-size: 1.15rem; font-weight: 700; color: #10B981;">Arriving in {active['eta_mins']} mins</div>
                        <div style="font-size: 1.1rem; font-weight: 800; color: #064E3B;">${active['fare']:.2f} ({active['ride_tier']})</div>
                    </div>
                </div>
                <div style="margin-top: 15px; background: white; padding: 12px 16px; border-radius: 8px; border: 1px dashed #10B981;">
                    <b>Route:</b> {active['pickup_address']} ➔ {active['dropoff_address']} ({active['distance_km']} km · ~{active['duration_mins']} mins)
                </div>
            </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2: PASSENGER TRIP HISTORY & RECEIPTS
# ════════════════════════════════════════════════════════════════════════════════
with tab_history:
    st.markdown("### 📋 My Ride History & Bookings")
    user_filter = st.session_state.get("passenger_name")
    history_rides = get_bookings(user_filter)

    if not history_rides:
        # Also fetch all if no specific user match
        history_rides = get_bookings()

    if history_rides:
        df_hist = pd.DataFrame(history_rides)
        st.write(f"Displaying **{len(df_hist)}** registered ride bookings:")

        for idx, ride in df_hist.iterrows():
            with st.expander(f"🚖 {ride['booking_id']} — {ride['pickup_address']} ➔ {ride['dropoff_address']} (${ride['fare']:.2f})", expanded=(idx==0)):
                h1, h2, h3, h4 = st.columns(4)
                h1.metric("Status", ride["status"])
                h2.metric("Total Fare", f"${ride['fare']:.2f}")
                h3.metric("Distance", f"{ride['distance_km']} km")
                h4.metric("Est. Travel Time", f"{ride['duration_mins']} min")

                st.markdown(f"""
                * **Passenger:** {ride['passenger_name']} ({ride['phone']})
                * **Assigned Driver:** {ride['driver_name']} (★ {ride['driver_rating']})
                * **Vehicle:** {ride['car_model']} ({ride['car_plate']})
                * **Vehicle Category:** {ride['ride_tier']}
                * **Booked Timestamp:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ride['created_at']))}
                """)
    else:
        st.info("No past rides found. Book your first trip in Tab 1!")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3: CITY DISPATCH INTELLIGENCE & SURGE ANALYTICS
# ════════════════════════════════════════════════════════════════════════════════
with tab_analytics:
    st.markdown("### 📊 City-Wide Dispatch & Surge Intelligence (Operations View)")
    st.write("Aggregated analytics across 4.7M+ historical records from the NYC Taxi & Limousine Commission.")

    if df_processed is not None:
        c_kpi1, c_kpi2, c_kpi3, c_kpi4 = st.columns(4)
        c_kpi1.metric("Total Aggregated Zones", f"{len(df_processed):,}")
        c_kpi2.metric("Unique Grid Locations", f"{df_processed['location'].nunique():,}")
        c_kpi3.metric("City Peak Commute Hour", "18:00 (6 PM)")
        c_kpi4.metric("Avg Hourly Demand / Zone", f"{df_processed['demand'].mean():.1f} rides")
        st.divider()

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.subheader("Hourly City-Wide Ride Demand Curve")
            hourly = df_processed.groupby("hour")["demand"].sum().reset_index()
            fig_h = px.line(hourly, x="hour", y="demand", markers=True, color_discrete_sequence=["#1D9E75"])
            fig_h.update_layout(height=320, margin=dict(t=10, b=20))
            st.plotly_chart(fig_h, use_container_width=True)

        with col_g2:
            st.subheader("Top 15 Highest-Demand Neighborhood Zones")
            top15 = df_processed.groupby("location")["demand"].sum().nlargest(15).reset_index()
            fig_top = px.bar(top15, x="demand", y="location", orientation="h", color="demand", color_continuous_scale="Teal")
            fig_top.update_layout(height=320, margin=dict(t=10, b=20), coloraxis_showscale=False)
            st.plotly_chart(fig_top, use_container_width=True)

        col_g3, col_g4 = st.columns(2)
        with col_g3:
            st.subheader("Day of Week Demand Volume")
            day_map = {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"}
            dow_df = df_processed.groupby("day_of_week")["demand"].sum().reset_index()
            dow_df["Day"] = dow_df["day_of_week"].map(day_map)
            fig_dow = px.bar(dow_df, x="Day", y="demand", color="demand", color_continuous_scale="Purples")
            fig_dow.update_layout(height=320, margin=dict(t=10, b=20), coloraxis_showscale=False)
            st.plotly_chart(fig_dow, use_container_width=True)

        with col_g4:
            st.subheader("PCA 2D Spatial Demand Clusters")
            if pca_model and scaler_model:
                le_obj = LabelEncoder()
                df_processed["loc_code"] = le_obj.fit_transform(df_processed["location"])
                sample = df_processed.sample(min(2500, len(df_processed)), random_state=42).copy()
                feats = ["loc_code", "hour", "day_of_week", "month", "is_weekend"]
                X_sc = scaler_model.transform(sample[feats].values)
                comps = pca_model.transform(X_sc)
                sample["PC1"] = comps[:, 0]
                sample["PC2"] = comps[:, 1]
                fig_pca = px.scatter(sample, x="PC1", y="PC2", color="hour", color_continuous_scale="Viridis")
                fig_pca.update_layout(height=320, margin=dict(t=10, b=20))
                st.plotly_chart(fig_pca, use_container_width=True)
    else:
        st.info("Processed dataset not loaded.")


st.divider()
st.caption("ZonePulse Mobility Platform · Production Ready · Powered by FastAPI, XGBoost, Scikit-learn, Plotly & Streamlit")

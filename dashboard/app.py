"""ZonePulse web experience: signup first, then mobility operations."""
import html
import os
import requests
import streamlit as st

_api_address = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
API_BASE_URL = _api_address if "://" in _api_address else f"http://{_api_address}"
st.set_page_config(page_title="ZonePulse | NYC Mobility", page_icon="Z", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root { --ink:#102b3f; --muted:#6b7c8c; --teal:#087e78; --mint:#d9f3ee; --line:#e6edf0; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; color:var(--ink); }
.stApp { background:linear-gradient(180deg,#f4faf9 0%,#f7f9fb 44%,#f7f9fb 100%); }
h1,h2,h3 { font-family:'Manrope',sans-serif; letter-spacing:-.035em; color:var(--ink); }
.block-container { max-width:1180px; padding-top:2rem; padding-bottom:3rem; }
[data-testid="stSidebar"] { background:#fff; border-right:1px solid var(--line); }
div[data-testid="stMetric"] { background:white; border:1px solid var(--line); border-radius:16px; padding:18px 20px; box-shadow:0 7px 24px rgba(16,43,63,.035); }
div[data-testid="stForm"] { background:white; border:1px solid var(--line); padding:24px; border-radius:18px; }
.stButton > button[kind="primary"], .stFormSubmitButton > button { background:#087e78; border:0; border-radius:10px; color:white; font-weight:700; min-height:44px; }
.stButton > button { border-radius:10px; min-height:42px; }
input, textarea, [data-baseweb="select"] { border-radius:9px !important; }
.zp-hero { background:linear-gradient(118deg,#102b3f 0%,#174957 55%,#087e78 100%); color:white; padding:28px 32px; border-radius:20px; margin:0 0 25px; box-shadow:0 16px 38px rgba(16,43,63,.12); }
.zp-hero h1 { color:white; margin:4px 0 8px; font-size:2rem; }
.zp-hero p { color:#d7e8e8; margin:0; }
.zp-eyebrow { color:#a8dfd3; text-transform:uppercase; letter-spacing:.13em; font-size:.72rem; font-weight:700; }
.zp-auth { background:white; border:1px solid var(--line); border-radius:22px; padding:28px 30px; box-shadow:0 18px 50px rgba(16,43,63,.07); }
.zp-note { color:var(--muted); font-size:.9rem; }
.zp-chip { display:inline-block; border-radius:99px; padding:6px 11px; background:#e4f5f1; color:#087e78; font-size:.78rem; font-weight:700; }
</style>
""", unsafe_allow_html=True)

def api_call(path, method="GET", payload=None, token=None, timeout=25):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    response = requests.request(method, f"{API_BASE_URL}{path}", json=payload, headers=headers, timeout=timeout)
    try:
        body = response.json()
    except ValueError:
        body = {"detail": response.text or "The API returned an unreadable response."}
    if not response.ok:
        detail = body.get("detail", "Request failed.")
        if isinstance(detail, list):
            detail = "; ".join(item.get("msg", "Invalid input") for item in detail)
        raise RuntimeError(str(detail))
    return body

def show_auth():
    left, right = st.columns([1.04, .96], gap="large")
    with left:
        st.markdown('<div class="zp-hero"><div class="zp-eyebrow">NYC mobility intelligence</div><h1>Move through the city with a clearer picture.</h1><p>Plan a ride, understand demand, and keep your trip details together in one calm workspace.</p></div>', unsafe_allow_html=True)
        st.markdown("#### A useful view of the next ride")
        st.markdown("- Demand-aware fare estimates for NYC locations")
        st.markdown("- Pickup availability and estimated driver wait")
        st.markdown("- Private trip history tied to your account")
        st.caption("ZonePulse is a portfolio demonstration. Driver availability and trip quotes are simulated estimates.")
    with right:
        st.markdown('<div class="zp-auth"><span class="zp-chip">WELCOME TO ZONEPULSE</span><h2 style="margin-top:16px">Your mobility workspace</h2><p class="zp-note">Create an account to get started, or sign in to continue.</p></div>', unsafe_allow_html=True)
        create_tab, login_tab = st.tabs(["Create account", "Sign in"])
        with create_tab:
            with st.form("signup_form", clear_on_submit=False):
                name = st.text_input("Name", placeholder="e.g. Alex Morgan")
                email = st.text_input("Email address", placeholder="you@example.com")
                password = st.text_input("Password", type="password", help="Use at least 6 characters.")
                confirm = st.text_input("Confirm password", type="password")
                submitted = st.form_submit_button("Create my account", type="primary", use_container_width=True)
            if submitted:
                if password != confirm:
                    st.error("The passwords do not match.")
                elif len(name.strip()) < 3:
                    st.error("Please enter a name with at least 3 characters.")
                else:
                    try:
                        with st.spinner("Creating your secure account…"):
                            result = api_call("/auth/signup", "POST", {"username": name.strip(), "email": email.strip(), "password": password}, timeout=120)
                        st.session_state["zp_token"] = result["access_token"]
                        st.session_state["zp_username"] = result["username"]
                        st.session_state["zp_email"] = result["email"]
                        st.rerun()
                    except requests.RequestException:
                        st.error("The sign-up service is taking longer than expected. Please try again.")
                    except RuntimeError as exc:
                        st.error(str(exc))
        with login_tab:
            with st.form("login_form"):
                identifier = st.text_input("Username or email", placeholder="Your account name")
                login_password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Sign in", type="primary", use_container_width=True)
            if submitted:
                try:
                    with st.spinner("Signing you in…"):
                        result = api_call("/auth/login", "POST", {"username": identifier.strip(), "password": login_password}, timeout=120)
                    st.session_state["zp_token"] = result["access_token"]
                    st.session_state["zp_username"] = result["username"]
                    st.session_state["zp_email"] = result["email"]
                    st.rerun()
                except requests.RequestException:
                    st.error("The sign-in service is taking longer than expected. Please try again.")
                except RuntimeError as exc:
                    st.error(str(exc))
        st.caption("Your password is hashed by the API and is never stored in the dashboard session.")

if not st.session_state.get("zp_token"):
    show_auth()
    st.stop()

token = st.session_state["zp_token"]
username = st.session_state.get("zp_username", "Rider")
display_name = html.escape(username)

with st.sidebar:
    st.markdown("## ZonePulse")
    st.caption("NYC mobility intelligence")
    st.divider()
    page = st.radio("Workspace", ["Overview", "Book a ride", "My trips", "Demand lab"], label_visibility="collapsed")
    st.divider()
    st.markdown(f"**{username}**")
    st.caption(st.session_state.get("zp_email", "Signed-in account"))
    if st.button("Sign out", use_container_width=True):
        for key in ("zp_token", "zp_username", "zp_email", "ride_quote", "ride_route", "ride_tier"):
            st.session_state.pop(key, None)
        st.rerun()
    st.caption("Estimates are simulated for demonstration.")

st.markdown(f'<div class="zp-hero"><div class="zp-eyebrow">ZonePulse / {page}</div><h1>{page if page != "Overview" else "Good to have you here, " + display_name + "."}</h1><p>Ride planning and demand signals for New York City.</p></div>', unsafe_allow_html=True)

def render_overview():
    try:
        stats = api_call("/stats", token=token)
        peak = api_call("/peak-hours", token=token)
        info = api_call("/model-info", token=token)
    except (requests.RequestException, RuntimeError) as exc:
        st.error(f"Could not load the mobility overview: {exc}")
        return
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Historical records", f'{stats.get("total_records", 0):,}')
    m2.metric("NYC zones", f'{stats.get("unique_locations", 0):,}')
    m3.metric("Average hourly demand", f'{stats.get("demand_mean", 0):.1f}')
    reg = (info.get("regression") or {}).get("results") or {}
    best = reg.get("XGBoost_Regressor", {})
    m4.metric("Forecast R²", f'{best.get("R2", 0):.1%}' if best else "—")
    st.subheader("City rhythm")
    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        st.markdown("#### Strongest demand window")
        hours = peak.get("top_demand_hours", [])
        st.metric("Evening peak", f'{min(hours):02d}:00–{max(hours):02d}:00' if hours else "—")
        st.write(peak.get("insight", "Peak-hour information is unavailable."))
        st.markdown("#### Quieter hours")
        quiet = peak.get("low_demand_hours", [])
        st.write(", ".join(f"{h:02d}:00" for h in quiet) if quiet else "—")
    with c2:
        st.markdown("#### What the forecast model reports")
        st.write(f"Regression model: **{(info.get('regression') or {}).get('best_model', 'Unavailable')}**")
        st.write(f"Demand classification: **{(info.get('classification') or {}).get('best_model', 'Unavailable')}**")
        st.info("Ride availability, nearby drivers, and fare quotes are modeled estimates for this demonstration, not a live dispatch network.")

def render_booking():
    try:
        landmarks = api_call("/landmarks", token=token)
    except (requests.RequestException, RuntimeError) as exc:
        st.error(f"Could not load pickup locations: {exc}")
        return
    names = [item["name"] for item in landmarks]
    by_name = {item["name"]: item for item in landmarks}
    st.markdown("### Build a trip")
    with st.form("quote_form"):
        a, b = st.columns(2)
        with a:
            pickup = st.selectbox("Pickup", names, index=names.index("Times Square, Manhattan") if "Times Square, Manhattan" in names else 0)
        with b:
            dropoff = st.selectbox("Destination", names, index=names.index("JFK International Airport (Terminal 4)") if "JFK International Airport (Terminal 4)" in names else min(1, len(names)-1))
        c, d = st.columns(2)
        with c:
            hour = st.slider("Pickup hour", 0, 23, 18, format="%d:00")
        with d:
            day = st.selectbox("Day", list(range(7)), index=4, format_func=lambda i: ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"][i])
        make_quote = st.form_submit_button("Calculate ride estimate", type="primary")
    if make_quote:
        p, q = by_name[pickup], by_name[dropoff]
        payload = {"passenger_name": username, "pickup_address": pickup, "pickup_lat": p["lat"], "pickup_lon": p["lon"], "dropoff_address": dropoff, "dropoff_lat": q["lat"], "dropoff_lon": q["lon"], "hour": hour, "day_of_week": day}
        try:
            with st.spinner("Estimating demand and fare…"):
                quote = api_call("/ride/quote", "POST", payload, token=token, timeout=60)
            st.session_state["ride_quote"] = quote
            st.session_state["ride_route"] = payload
            st.session_state["ride_tier"] = "Standard (UberX/Mini)"
        except (requests.RequestException, RuntimeError) as exc:
            st.error(f"Could not calculate this estimate: {exc}")
    quote = st.session_state.get("ride_quote")
    route = st.session_state.get("ride_route")
    if not quote or not route:
        return
    st.divider()
    st.markdown(f'#### {route["pickup_address"]}  →  {route["dropoff_address"]}')
    x1, x2, x3, x4 = st.columns(4)
    x1.metric("Estimated fare", f'{quote["fares"][st.session_state["ride_tier"]]:.2f} USD')
    x2.metric("Distance", f'{quote["distance_km"]:.1f} km')
    x3.metric("Driver wait", f'{quote["estimated_wait_time_minutes"]} min')
    x4.metric("Find-a-car estimate", f'{quote["probability_finding_car_pct"]:.0f}%')
    st.write(f'Demand tier: **{quote["demand_tier"]}** · Estimated demand: **{quote["predicted_hourly_demand"]} rides/hour** · Surge factor: **{quote["surge_multiplier"]:.2f}×**')
    tier = st.radio("Ride option", list(quote["fares"]), horizontal=True, format_func=lambda label: f"{label} · {quote['fares'][label]:.2f} USD", key="ride_tier")
    phone = st.text_input("Contact number (optional)", value="")
    if st.button("Confirm demonstration booking", type="primary"):
        driver = quote.get("nearby_drivers", [{}])[0]
        booking = {"passenger_name": username, "phone": phone, "pickup_address": route["pickup_address"], "pickup_lat": route["pickup_lat"], "pickup_lon": route["pickup_lon"], "dropoff_address": route["dropoff_address"], "dropoff_lat": route["dropoff_lat"], "dropoff_lon": route["dropoff_lon"], "distance_km": quote["distance_km"], "duration_mins": quote["duration_minutes"], "ride_tier": tier, "fare": quote["fares"][tier], "driver_name": driver.get("name", "Assigned driver"), "driver_rating": driver.get("rating", 4.9), "car_model": driver.get("car_model", "Standard vehicle"), "car_plate": driver.get("plate", "—"), "eta_mins": quote["estimated_wait_time_minutes"]}
        try:
            result = api_call("/ride/book", "POST", booking, token=token, timeout=30)
            st.success(result.get("message", "Booking saved."))
            st.json(result.get("booking", {}))
        except (requests.RequestException, RuntimeError) as exc:
            st.error(f"Booking could not be saved: {exc}")

def render_trips():
    try:
        trips = api_call("/ride/history", token=token)
    except (requests.RequestException, RuntimeError) as exc:
        st.error(f"Could not load your trips: {exc}")
        return
    st.subheader("Your recent bookings")
    if not trips:
        st.info("No bookings yet. Your trip history will appear here.")
        return
    for trip in trips:
        title = f'{trip.get("pickup_address", "Pickup")} → {trip.get("dropoff_address", "Destination")} · {trip.get("fare", 0):.2f} USD'
        with st.expander(title):
            st.write(f'**Status:** {trip.get("status", "Confirmed")} · **Ride:** {trip.get("ride_tier", "Standard")}')
            st.write(f'**Distance:** {trip.get("distance_km", 0)} km · **Duration:** {trip.get("duration_mins", 0)} min')
            st.write(f'**Booking reference:** {trip.get("booking_id", "—")}')

def render_demand_lab():
    st.markdown("### Forecast demand by zone and time")
    with st.form("forecast_form"):
        a, b, c = st.columns(3)
        with a:
            location = st.number_input("Location code", min_value=0, max_value=10000, value=100, step=1)
        with b:
            hour = st.slider("Hour", 0, 23, 18)
        with c:
            day = st.selectbox("Day of week", list(range(7)), index=4, format_func=lambda i: ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"][i])
        month = st.selectbox("Month", list(range(1,13)), index=5)
        run = st.form_submit_button("Run demand forecast", type="primary")
    if run:
        params = {"location_code": int(location), "hour": hour, "day_of_week": day, "month": month, "is_weekend": int(day >= 5), "hour_bin": hour // 3, "cluster": 0}
        try:
            response = requests.get(f"{API_BASE_URL}/predict", params=params, headers={"Authorization": f"Bearer {token}"}, timeout=60)
            if not response.ok:
                raise RuntimeError(response.json().get("detail", "Forecast request failed."))
            prediction = response.json()
            classification = requests.get(f"{API_BASE_URL}/classify", params=params, headers={"Authorization": f"Bearer {token}"}, timeout=60)
            category = classification.json() if classification.ok else {}
            st.metric("Predicted hourly demand", f'{prediction.get("predicted_demand", 0):.1f}')
            st.write(f'**Demand tier:** {category.get("demand_category", "Unavailable")} · {category.get("recommended_action", "")}')
        except (requests.RequestException, RuntimeError, ValueError) as exc:
            st.error(f"Forecast unavailable: {exc}")

if page == "Overview":
    render_overview()
elif page == "Book a ride":
    render_booking()
elif page == "My trips":
    render_trips()
else:
    render_demand_lab()
st.divider()
st.caption("ZonePulse · NYC ride demand and booking demonstration · Estimates are simulated.")

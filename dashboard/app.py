"""ZonePulse — a concise mobility operations workspace."""
import html
import os
import re
import requests
import streamlit as st

raw_api_address = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
API_BASE_URL = raw_api_address if "://" in raw_api_address else f"http://{raw_api_address}"

st.set_page_config(
    page_title="ZonePulse | Mobility Operations",
    page_icon="Z",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root { --navy:#142b3c; --navy-2:#213b4e; --teal:#137c72; --teal-soft:#e8f4f1; --ink:#202c36; --muted:#71808b; --line:#e4e9ed; --paper:#f5f7f8; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; color:var(--ink); }
.stApp { background:var(--paper); }
.block-container { max-width:1260px; padding-top:1.8rem; padding-bottom:3rem; }
h1,h2,h3,h4 { font-family:'Manrope',sans-serif; color:var(--navy); letter-spacing:-.025em; }
h1 { font-size:2rem !important; }
[data-testid="stSidebar"] { background:#fff; border-right:1px solid var(--line); }
div[data-testid="stMetric"] { background:#fff; border:1px solid var(--line); border-radius:12px; padding:16px 18px; box-shadow:0 2px 10px rgba(20,43,60,.025); }
div[data-testid="stForm"] { background:#fff; border:1px solid var(--line); padding:22px; border-radius:14px; }
.stButton > button[kind="primary"], .stFormSubmitButton > button { background:var(--navy); border:1px solid var(--navy); border-radius:8px; color:#fff; font-weight:600; min-height:44px; }
.stButton > button[kind="primary"]:hover, .stFormSubmitButton > button:hover { background:var(--teal); border-color:var(--teal); color:#fff; }
.stButton > button { border-radius:8px; min-height:40px; }
input, textarea, [data-baseweb="select"] { border-radius:7px !important; }
[data-testid="stTabs"] button[role="tab"] { font-weight:600; }
.zp-topline { color:var(--teal); font-size:.73rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase; }
.zp-panel { color:#fff; background:linear-gradient(145deg,#142b3c,#1c4852); border-radius:18px; padding:38px; min-height:380px; box-shadow:0 12px 32px rgba(20,43,60,.10); }
.zp-panel h1 { color:#fff; font-size:2.25rem !important; line-height:1.16; margin:14px 0; }
.zp-panel p { color:#d4e0e3; font-size:1rem; line-height:1.7; }
.zp-panel .label { color:#a8d9cd; font-size:.72rem; font-weight:700; letter-spacing:.13em; text-transform:uppercase; }
.zp-auth-card { background:white; border:1px solid var(--line); border-radius:16px; padding:28px; box-shadow:0 8px 28px rgba(20,43,60,.045); }
.zp-caption { color:var(--muted); font-size:.9rem; }
.zp-section { border-bottom:1px solid var(--line); padding:0 0 12px; margin:26px 0 16px; }
.zp-status { display:inline-block; background:var(--teal-soft); color:var(--teal); border-radius:99px; padding:5px 10px; font-weight:700; font-size:.75rem; }
.zp-hero { background:#fff; border:1px solid var(--line); border-radius:14px; padding:22px 26px; margin-bottom:20px; }
.zp-hero p { color:var(--muted); margin:0; }
</style>
""", unsafe_allow_html=True)


def api_call(path, method="GET", payload=None, token=None, timeout=45):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    response = requests.request(
        method, f"{API_BASE_URL}{path}", json=payload, headers=headers, timeout=timeout
    )
    try:
        body = response.json()
    except ValueError:
        body = {"detail": response.text or "The service returned an unreadable response."}
    if not response.ok:
        detail = body.get("detail", "Request failed.")
        if isinstance(detail, list):
            detail = "; ".join(item.get("msg", "Invalid input") for item in detail)
        raise RuntimeError(str(detail))
    return body


def establish_session(result):
    st.session_state["zp_token"] = result["access_token"]
    st.session_state["zp_username"] = result["username"]
    st.session_state["zp_display_name"] = result.get("full_name") or result["username"]
    st.session_state["zp_email"] = result["email"]
    st.rerun()


def show_auth():
    left, right = st.columns([1.08, .92], gap="large")
    with left:
        st.markdown(
            '<div class="zp-panel"><div class="label">ZonePulse · New York City</div>'
            '<h1>Clearer decisions for every city ride.</h1>'
            '<p>A focused workspace for fare estimates, pickup planning, and demand signals. '
            'Your bookings and account details stay together in one place.</p>'
            '<br><span class="zp-status">Rider workspace</span> '
            '<span class="zp-status">Demand intelligence</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown("### Built around the trip")
        st.write("Demand-aware estimates across mapped New York locations.")
        st.write("Pickup timing, nearby driver estimates, and saved trip history.")
        st.caption("This portfolio demonstration uses simulated driver availability and ride estimates.")
    with right:
        st.markdown('<div class="zp-auth-card">', unsafe_allow_html=True)
        st.markdown('<div class="zp-topline">Welcome</div>', unsafe_allow_html=True)
        st.markdown("## Your mobility workspace")
        st.markdown('<p class="zp-caption">Create an account or sign in to continue.</p>', unsafe_allow_html=True)
        create_tab, sign_in_tab = st.tabs(["Create account", "Sign in"])
        with create_tab:
            with st.form("signup_form", clear_on_submit=False):
                name = st.text_input("Full name", placeholder="Alex Morgan")
                email = st.text_input("Email address", placeholder="you@example.com")
                password = st.text_input("Password", type="password", help="At least 6 characters.")
                confirm_password = st.text_input("Confirm password", type="password")
                submitted = st.form_submit_button("Create account", type="primary", use_container_width=True)
            if submitted:
                normalized_email = email.strip().lower()
                if len(name.strip()) < 3:
                    st.error("Enter a name with at least 3 characters.")
                elif not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", normalized_email):
                    st.error("Enter a valid email address.")
                elif len(password) < 6:
                    st.error("Use a password with at least 6 characters.")
                elif password != confirm_password:
                    st.error("The passwords do not match.")
                else:
                    try:
                        with st.spinner("Securely creating your account…"):
                            result = api_call(
                                "/auth/signup", "POST",
                                {"username": name.strip(), "email": normalized_email, "password": password},
                                timeout=120,
                            )
                        establish_session(result)
                    except requests.RequestException:
                        st.error("The account service is unavailable right now. Please retry; your details are not shown in this message.")
                    except RuntimeError as exc:
                        st.error(str(exc))
        with sign_in_tab:
            with st.form("login_form"):
                identifier = st.text_input("Username or email", placeholder="you@example.com")
                login_password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Sign in", type="primary", use_container_width=True)
            if submitted:
                try:
                    with st.spinner("Signing you in…"):
                        result = api_call(
                            "/auth/login", "POST",
                            {"username": identifier.strip(), "password": login_password},
                            timeout=120,
                        )
                    establish_session(result)
                except requests.RequestException:
                    st.error("The sign-in service is unavailable right now. Please retry.")
                except RuntimeError as exc:
                    st.error(str(exc))
        st.caption("Passwords are stored as salted hashes. They are never saved in the dashboard session.")
        st.markdown("</div>", unsafe_allow_html=True)


if not st.session_state.get("zp_token"):
    show_auth()
    st.stop()

token = st.session_state["zp_token"]
username = st.session_state.get("zp_username", "Rider")
display_name = st.session_state.get("zp_display_name", username)
safe_username = html.escape(display_name)

with st.sidebar:
    st.markdown("## ZonePulse")
    st.caption("NYC mobility operations")
    st.divider()
    page = st.radio(
        "Workspace",
        ["Overview", "Plan a ride", "Trip history", "Driver fleet", "Demand insights"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown(f"**{safe_username}**")
    st.caption(st.session_state.get("zp_email", "Signed-in account"))
    if st.button("Sign out", use_container_width=True):
        for key in list(st.session_state.keys()):
            if key.startswith("zp_") or key.startswith("ride_"):
                del st.session_state[key]
        st.rerun()
    st.caption("Driver availability and fare figures are estimates.")


def page_header(title, description):
    st.markdown(
        f'<div class="zp-hero"><div class="zp-topline">ZonePulse / {html.escape(page)}</div>'
        f'<h1>{title}</h1><p>{description}</p></div>',
        unsafe_allow_html=True,
    )


def render_overview():
    page_header(f"Welcome back, {safe_username}.", "Your trips, pickup activity, and city demand at a glance.")
    try:
        stats = api_call("/stats", token=token)
        peak = api_call("/peak-hours", token=token)
        info = api_call("/model-info", token=token)
        trips = api_call("/ride/history", token=token)
        fleet = api_call("/drivers", token=token)
    except (requests.RequestException, RuntimeError) as exc:
        st.error(f"Overview data could not be loaded: {exc}")
        return

    historical = stats.get("total_records", 0)
    zones = stats.get("unique_locations", 0)
    avg_demand = stats.get("demand_mean", 0)
    st.subheader("Operating snapshot")
    a, b, c, d = st.columns(4)
    a.metric("Historical records", f"{historical:,}")
    b.metric("Mapped zones", f"{zones:,}")
    c.metric("Average hourly demand", f"{avg_demand:.1f}")
    d.metric("Saved trips", f"{len(trips):,}")

    left, right = st.columns([1.1, .9], gap="large")
    with left:
        st.markdown('<div class="zp-section"><h3>City rhythm</h3></div>', unsafe_allow_html=True)
        hours = peak.get("top_demand_hours", [])
        low_hours = peak.get("low_demand_hours", [])
        st.metric("Peak window", f"{min(hours):02d}:00–{max(hours):02d}:00" if hours else "—")
        st.write(peak.get("insight", "Peak-hour guidance is unavailable."))
        st.caption("Quieter hours: " + (", ".join(f"{hour:02d}:00" for hour in low_hours) or "—"))
    with right:
        st.markdown('<div class="zp-section"><h3>Model and fleet</h3></div>', unsafe_allow_html=True)
        regression = (info.get("regression") or {}).get("best_model", "Unavailable")
        classifier = (info.get("classification") or {}).get("best_model", "Unavailable")
        st.write(f"**Demand forecast**  ·  {regression}")
        st.write(f"**Demand classification**  ·  {classifier}")
        st.write(f"**Driver profiles recorded**  ·  {len(fleet)}")
        st.caption("Driver profiles shown here are generated for this demonstration and are not a live dispatch fleet.")

    st.markdown('<div class="zp-section"><h3>Recent trips</h3></div>', unsafe_allow_html=True)
    if not trips:
        st.info("Your confirmed trips will appear here.")
    else:
        for trip in trips[:4]:
            with st.container(border=True):
                route, details = st.columns([1.4, 1])
                route.markdown(f"**{trip['pickup_address']}**  →  **{trip['dropoff_address']}**")
                route.caption(f"{trip.get('created_at', 'Date unavailable')} · {trip.get('booking_id', '')}")
                details.write(f"{trip.get('ride_tier', 'Ride')} · {trip.get('fare', 0):.2f} USD")
                details.caption(f"{trip.get('status', 'Confirmed')} · Driver {trip.get('driver_name') or 'pending'}")


def render_ride_planner():
    page_header("Plan a ride", "Compare a route estimate, then save the confirmed trip to your account.")
    try:
        landmarks = api_call("/landmarks")
    except (requests.RequestException, RuntimeError) as exc:
        st.error(f"Pickup locations could not be loaded: {exc}")
        return
    if not landmarks:
        st.warning("No mapped locations are available.")
        return

    names = [item["name"] for item in landmarks]
    by_name = {item["name"]: item for item in landmarks}
    with st.form("quote_form"):
        col_a, col_b = st.columns(2)
        with col_a:
            pickup = st.selectbox("Pickup location", names, index=names.index("Times Square, Manhattan") if "Times Square, Manhattan" in names else 0)
        with col_b:
            destination_index = names.index("JFK International Airport (Terminal 4)") if "JFK International Airport (Terminal 4)" in names else min(1, len(names) - 1)
            destination = st.selectbox("Destination", names, index=destination_index)
        col_c, col_d = st.columns(2)
        with col_c:
            hour = st.slider("Pickup hour", 0, 23, 18, format="%d:00")
        with col_d:
            day = st.selectbox("Travel day", list(range(7)), index=4, format_func=lambda idx: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][idx])
        quote_submitted = st.form_submit_button("Calculate estimate", type="primary")
    if quote_submitted:
        if pickup == destination:
            st.error("Choose different pickup and destination locations.")
        else:
            start, end = by_name[pickup], by_name[destination]
            payload = {
                "passenger_name": username,
                "pickup_address": pickup,
                "pickup_lat": start["lat"],
                "pickup_lon": start["lon"],
                "dropoff_address": destination,
                "dropoff_lat": end["lat"],
                "dropoff_lon": end["lon"],
                "hour": hour,
                "day_of_week": day,
            }
            try:
                with st.spinner("Preparing the route estimate…"):
                    quote = api_call("/ride/quote", "POST", payload, token=token, timeout=90)
                st.session_state["ride_quote"] = quote
                st.session_state["ride_route"] = payload
                st.session_state["ride_tier"] = "Standard (UberX/Mini)"
                st.session_state["ride_driver"] = quote.get("nearby_drivers", [{}])[0]
            except (requests.RequestException, RuntimeError) as exc:
                st.error(f"The estimate could not be prepared: {exc}")

    quote = st.session_state.get("ride_quote")
    route = st.session_state.get("ride_route")
    if not quote or not route:
        return

    st.markdown('<div class="zp-section"><h3>Estimate details</h3></div>', unsafe_allow_html=True)
    route_col, metric_col = st.columns([1.1, 1.4], gap="large")
    with route_col:
        st.markdown(f"**{route['pickup_address']}**  →  **{route['dropoff_address']}**")
        st.caption(f"Approx. {quote['distance_km']:.1f} km · {quote['duration_minutes']} min")
        st.markdown(f"<span class='zp-status'>{html.escape(quote['demand_tier'])} demand</span>", unsafe_allow_html=True)
        st.write(quote.get("availability_status", "Availability estimate"))
    with metric_col:
        m1, m2, m3 = st.columns(3)
        m1.metric("Pickup wait", f"{quote['estimated_wait_time_minutes']} min")
        m2.metric("Find-a-car estimate", f"{quote['probability_finding_car_pct']:.0f}%")
        m3.metric("Surge factor", f"{quote['surge_multiplier']:.2f}×")

    tiers = list(quote["fares"])
    selected_tier = st.radio(
        "Ride option",
        tiers,
        horizontal=True,
        key="ride_tier",
        format_func=lambda tier: f"{tier} · {quote['fares'][tier]:.2f} USD",
    )
    drivers = quote.get("nearby_drivers", [])
    selected_driver = st.selectbox(
        "Suggested driver",
        drivers,
        format_func=lambda item: f"{item['name']} · {item['car_model']} · {item['eta_minutes']} min away · ★ {item['rating']}",
        key="ride_driver",
    )
    with st.form("confirm_ride_form"):
        phone = st.text_input("Passenger contact number (optional)", placeholder="Add a number for this trip")
        confirm = st.form_submit_button("Confirm and save trip", type="primary")
    if confirm:
        booking = {
            **route,
            "passenger_name": username,
            "phone": phone.strip() or None,
            "distance_km": quote["distance_km"],
            "duration_mins": quote["duration_minutes"],
            "ride_tier": selected_tier,
            "fare": quote["fares"][selected_tier],
            "driver_id": selected_driver.get("driver_id"),
            "driver_name": selected_driver.get("name"),
            "driver_rating": selected_driver.get("rating"),
            "car_model": selected_driver.get("car_model"),
            "car_color": selected_driver.get("car_color"),
            "car_plate": selected_driver.get("plate"),
            "driver_lat": selected_driver.get("lat"),
            "driver_lon": selected_driver.get("lon"),
            "driver_distance_km": selected_driver.get("distance_km"),
            "eta_mins": selected_driver.get("eta_minutes", quote["estimated_wait_time_minutes"]),
            "predicted_hourly_demand": quote.get("predicted_hourly_demand"),
            "demand_tier": quote.get("demand_tier"),
            "surge_multiplier": quote.get("surge_multiplier"),
            "probability_finding_car_pct": quote.get("probability_finding_car_pct"),
        }
        try:
            result = api_call("/ride/book", "POST", booking, token=token, timeout=60)
            st.success(f"Trip saved · {result['booking']['booking_id']}")
            st.session_state.pop("ride_quote", None)
            st.session_state.pop("ride_route", None)
            st.session_state.pop("ride_driver", None)
            st.rerun()
        except (requests.RequestException, RuntimeError) as exc:
            st.error(f"Trip could not be saved: {exc}")


def render_trip_history():
    page_header("Trip history", "A private record of rides saved to your passenger account.")
    try:
        trips = api_call("/ride/history", token=token)
    except (requests.RequestException, RuntimeError) as exc:
        st.error(f"Trip history could not be loaded: {exc}")
        return
    if not trips:
        st.info("No trips have been confirmed yet. Start with Plan a ride.")
        return
    st.caption(f"{len(trips)} saved trip{'s' if len(trips) != 1 else ''}")
    for trip in trips:
        title = f"{trip.get('pickup_address', 'Pickup')}  →  {trip.get('dropoff_address', 'Destination')}  ·  {trip.get('fare', 0):.2f} USD"
        with st.expander(title):
            left, right = st.columns(2)
            left.write(f"**Booking reference**  ·  {trip.get('booking_id', '—')}")
            left.write(f"**Booked at**  ·  {trip.get('created_at', '—')}")
            left.write(f"**Ride option**  ·  {trip.get('ride_tier', '—')}")
            left.write(f"**Distance / time**  ·  {trip.get('distance_km', 0):.1f} km / {trip.get('duration_mins', 0)} min")
            right.write(f"**Driver**  ·  {trip.get('driver_name') or '—'}")
            right.write(f"**Vehicle**  ·  {trip.get('car_color') or ''} {trip.get('car_model') or '—'}")
            right.write(f"**Plate**  ·  {trip.get('car_plate') or '—'}")
            right.write(f"**Status**  ·  {trip.get('status', 'Confirmed')}")


def render_fleet():
    page_header("Driver fleet", "Recorded driver profiles and vehicle details associated with ride estimates.")
    st.info("This is a simulated portfolio fleet. It is not a live driver dispatch service.")
    try:
        rows = api_call("/drivers", token=token)
    except (requests.RequestException, RuntimeError) as exc:
        st.error(f"Driver records could not be loaded: {exc}")
        return
    if not rows:
        st.info("Driver profiles are recorded after you calculate a ride estimate.")
        return
    columns = ["driver_id", "name", "rating", "car_model", "car_color", "license_plate", "eta_minutes", "status"]
    st.dataframe(
        [{col: row.get(col) for col in columns} for row in rows],
        use_container_width=True,
        hide_index=True,
        column_config={
            "driver_id": "Driver ref",
            "name": "Driver",
            "rating": st.column_config.NumberColumn("Rating", format="%.2f ★"),
            "car_model": "Vehicle",
            "car_color": "Color",
            "license_plate": "Plate",
            "eta_minutes": st.column_config.NumberColumn("ETA", format="%d min"),
            "status": "Status",
        },
    )


def render_demand_insights():
    page_header("Demand insights", "Explore the demand model using a zone and pickup time.")
    with st.form("forecast_form"):
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            location = st.number_input("Location code", min_value=0, max_value=10000, value=100, step=1)
        with col_b:
            hour = st.slider("Hour of day", 0, 23, 18)
        with col_c:
            day = st.selectbox("Day of week", list(range(7)), index=4, format_func=lambda idx: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][idx])
        month = st.selectbox("Month", list(range(1, 13)), index=5)
        run = st.form_submit_button("Run forecast", type="primary")
    if run:
        params = {
            "location_code": int(location),
            "hour": hour,
            "day_of_week": day,
            "month": month,
            "is_weekend": int(day >= 5),
            "hour_bin": hour // 3,
            "cluster": 0,
        }
        try:
            response = requests.get(
                f"{API_BASE_URL}/predict",
                params=params,
                headers={"Authorization": f"Bearer {token}"},
                timeout=90,
            )
            if not response.ok:
                detail = response.json().get("detail", "Forecast request failed.")
                raise RuntimeError(str(detail))
            prediction = response.json()
            classification_response = requests.get(
                f"{API_BASE_URL}/classify",
                params=params,
                headers={"Authorization": f"Bearer {token}"},
                timeout=90,
            )
            classification = classification_response.json() if classification_response.ok else {}
            result_a, result_b = st.columns(2)
            result_a.metric("Predicted hourly demand", f"{prediction.get('predicted_demand', 0):.1f}")
            result_b.metric("Demand category", classification.get("demand_category", "Unavailable"))
            st.write(classification.get("recommended_action", prediction.get("interpretation", "")))
        except (requests.RequestException, RuntimeError, ValueError) as exc:
            st.error(f"The forecast could not be calculated: {exc}")


if page == "Overview":
    render_overview()
elif page == "Plan a ride":
    render_ride_planner()
elif page == "Trip history":
    render_trip_history()
elif page == "Driver fleet":
    render_fleet()
else:
    render_demand_insights()

st.divider()
st.caption("ZonePulse · NYC mobility intelligence · Driver availability and ride quotes are estimates.")

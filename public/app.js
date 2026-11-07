const app = document.querySelector("#app");
const toast = document.querySelector("#toast");
const icons = {
  grid: '<rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/>',
  pin: '<path d="M20 10c0 5-8 11-8 11S4 15 4 10a8 8 0 1 1 16 0Z"/><circle cx="12" cy="10" r="2.5"/>',
  route: '<circle cx="6" cy="19" r="3"/><path d="M9 19h3.5a3.5 3.5 0 0 0 0-7H11a3.5 3.5 0 0 1 0-7H15"/><circle cx="18" cy="5" r="3"/>',
  clock: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
  car: '<path d="m5 11 1.5-4.5A2.5 2.5 0 0 1 8.9 5h6.2a2.5 2.5 0 0 1 2.4 1.5L19 11"/><path d="M3 11h18v7H3z"/><path d="M6 18v2m12-2v2M6 14h.01M18 14h.01"/>',
  chart: '<path d="M4 19V5m0 14h17"/><path d="m7 15 4-4 3 2 6-7"/>',
  user: '<circle cx="12" cy="8" r="4"/><path d="M5 21a7 7 0 0 1 14 0"/>',
  logout: '<path d="M10 17l5-5-5-5m5 5H3"/><path d="M12 3h7a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-7"/>',
  cloud: '<path d="M20 16.2A4.5 4.5 0 0 0 18 7.5h-1.1A6 6 0 1 0 5 16.5"/><path d="M8 19h.01M12 19h.01M16 19h.01"/>',
  arrow: '<path d="M7 17 17 7M7 7h10v10"/>',
  search: '<circle cx="11" cy="11" r="7"/><path d="m20 20-4-4"/>',
  plus: '<path d="M12 5v14m-7-7h14"/>',
  refresh: '<path d="M20 7v5h-5M4 17v-5h5"/><path d="M5.6 9A7 7 0 0 1 18 6l2 2M4 16l2 2a7 7 0 0 0 12.4-3"/>',
  check: '<path d="m5 12 4 4L19 6"/>',
};
const icon = (name, size = 19) => `<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${icons[name] || icons.grid}</svg>`;
const state = { token: localStorage.getItem("zp_token"), user: null, page: "Overview", places: { pickup: [], dropoff: [] }, weather: null, quote: null, trips: [], drivers: [], busy: false };
const esc = (s = "") => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const money = (n) => new Intl.NumberFormat(undefined, { style: "currency", currency: "USD" }).format(Number(n || 0));
const fmtDate = (date) => date ? new Date(date).toLocaleString([], { dateStyle: "medium", timeStyle: "short" }) : "—";

function notify(message, kind = "") {
  toast.textContent = message; toast.className = `toast show ${kind}`;
  clearTimeout(notify.timer); notify.timer = setTimeout(() => toast.classList.remove("show"), 3800);
}

async function api(path, options = {}) {
  let response;
  try {
    response = await fetch(path.startsWith("/api/") ? path : `/api${path}`, {
      ...options,
      headers: { "Content-Type": "application/json", ...(state.token ? { Authorization: `Bearer ${state.token}` } : {}), ...(options.headers || {}) },
      body: options.body && typeof options.body !== "string" ? JSON.stringify(options.body) : options.body,
    });
  } catch {
    throw new Error("The app could not reach its server. Refresh the page and try again.");
  }
  const type = response.headers.get("content-type") || "";
  if (!type.includes("application/json")) {
    if (response.status === 502 || response.status === 503 || response.status === 504) throw new Error("The server is waking up or temporarily unavailable. Wait a moment, then retry.");
    throw new Error("The server returned an unexpected response. Refresh the page and try again.");
  }
  const payload = await response.json();
  if (!response.ok) {
    if (response.status === 401 && state.token && path !== "/api/auth/login") {
      clearSession(); renderAuth();
      throw new Error("Your session ended. Sign in again to continue.");
    }
    throw new Error(payload.error || "That request could not be completed.");
  }
  return payload;
}

function clearSession() {
  localStorage.removeItem("zp_token"); state.token = null; state.user = null; state.trips = []; state.drivers = [];
}

function renderAuth() {
  app.innerHTML = `<main class="auth-shell">
    <section class="auth-story">
      <a class="brand" href="#"><span class="brand-mark">Z</span><span>ZonePulse<small>GLOBAL MOBILITY</small></span></a>
      <div class="story-content"><div class="eyebrow light">THE MOBILITY OPERATIONS DESK</div><h1>Every ride,<br><em>in better view.</em></h1><p>Plan routes across the globe with live pickup weather, clear demand estimates, and a trip history that stays with your account.</p><div class="story-points"><span>${icon("pin")} Global place search</span><span>${icon("cloud")} Current local weather</span><span>${icon("route")} Saved trip records</span></div></div>
      <div class="story-foot">ZonePulse · Mobility planning workspace</div>
    </section>
    <section class="auth-main"><div class="auth-card"><div class="eyebrow">WELCOME TO ZONEPULSE</div><h2>Your mobility workspace</h2><p class="muted">Sign in to continue or create an account to get started.</p>
      <div class="auth-tabs"><button class="auth-tab active" data-auth-tab="login">Sign in</button><button class="auth-tab" data-auth-tab="signup">Create account</button></div>
      <form id="login-form" class="auth-form"><label>Email or account username<input name="identifier" type="text" autocomplete="username" placeholder="you@example.com" required /></label><label>Password<input name="password" type="password" autocomplete="current-password" placeholder="Your password" required /></label><button class="button primary full" type="submit">Sign in <span>→</span></button><div class="form-message" aria-live="polite"></div></form>
      <form id="signup-form" class="auth-form hidden"><div class="form-grid"><label>Full name<input name="full_name" type="text" autocomplete="name" placeholder="Alex Morgan" minlength="3" maxlength="120" required /></label><label>Email address<input name="email" type="email" autocomplete="email" placeholder="you@example.com" required /></label><label>Phone <span class="optional">Optional</span><input name="phone" type="tel" autocomplete="tel" placeholder="Country code + number" maxlength="40" /></label><label>Username <span class="optional">Optional</span><input name="username" type="text" autocomplete="username" placeholder="Uses your name by default" maxlength="50" /></label></div><label>Password<input name="password" type="password" autocomplete="new-password" minlength="6" maxlength="72" placeholder="At least 6 characters" required /></label><label>Confirm password<input name="confirm_password" type="password" autocomplete="new-password" minlength="6" placeholder="Type it again" required /></label><button class="button primary full" type="submit">Create account <span>→</span></button><div class="form-message" aria-live="polite"></div></form>
      <div class="auth-note">Your password is protected. Your account and saved trips are stored in PostgreSQL.</div>
    </div><div class="auth-caption">Planning estimates are for demonstration and are not live ride quotes.</div></section>
  </main>`;
  document.querySelectorAll("[data-auth-tab]").forEach((button) => button.addEventListener("click", () => {
    document.querySelectorAll("[data-auth-tab]").forEach((item) => item.classList.toggle("active", item === button));
    document.querySelector("#login-form").classList.toggle("hidden", button.dataset.authTab !== "login");
    document.querySelector("#signup-form").classList.toggle("hidden", button.dataset.authTab !== "signup");
  }));
  document.querySelector("#login-form").addEventListener("submit", login);
  document.querySelector("#signup-form").addEventListener("submit", signup);
}

async function login(event) {
  event.preventDefault(); const form = event.currentTarget; const button = form.querySelector("button[type=submit]");
  button.disabled = true; button.innerHTML = "Signing in…";
  try {
    const data = await api("/api/auth/login", { method: "POST", body: { identifier: form.identifier.value.trim(), password: form.password.value } });
    acceptSession(data); notify("Welcome back.", "success");
  } catch (error) { form.querySelector(".form-message").textContent = error.message; }
  finally { button.disabled = false; button.innerHTML = 'Sign in <span>→</span>'; }
}

async function signup(event) {
  event.preventDefault(); const form = event.currentTarget; const button = form.querySelector("button[type=submit]");
  if (form.password.value !== form.confirm_password.value) { form.querySelector(".form-message").textContent = "The passwords do not match. Check both fields and try again."; return; }
  button.disabled = true; button.innerHTML = "Creating your account…";
  try {
    const data = await api("/api/auth/signup", { method: "POST", body: { full_name: form.full_name.value.trim(), email: form.email.value.trim(), phone: form.phone.value.trim(), username: form.username.value.trim(), password: form.password.value } });
    acceptSession(data); notify("Your account is ready.", "success");
  } catch (error) { form.querySelector(".form-message").textContent = error.message; }
  finally { button.disabled = false; button.innerHTML = 'Create account <span>→</span>'; }
}

function acceptSession(data) {
  state.token = data.token; state.user = data.user; localStorage.setItem("zp_token", data.token); state.page = "Overview"; renderShell(); loadPage();
}

function navItem(name, glyph) { return `<button class="nav-item ${state.page === name ? "selected" : ""}" data-page="${esc(name)}">${icon(glyph)}<span>${esc(name)}</span>${state.page === name ? '<i></i>' : ""}</button>`; }

function renderShell() {
  if (!state.user) return renderAuth();
  app.innerHTML = `<div class="app-layout"><aside class="sidebar"><a class="brand" href="#"><span class="brand-mark">Z</span><span>ZonePulse<small>MOBILITY DESK</small></span></a><div class="workspace-label">WORKSPACE</div><nav>${navItem("Overview", "grid")}${navItem("Plan a ride", "route")}${navItem("Trip history", "clock")}${navItem("Driver fleet", "car")}${navItem("Demand insights", "chart")}${navItem("My account", "user")}</nav><div class="side-bottom"><div class="side-note"><span class="live-dot"></span>Global planning active<p>Weather and local time available for route locations.</p></div><div class="profile-mini"><span class="avatar">${esc((state.user.full_name || "R").slice(0,1).toUpperCase())}</span><span class="profile-name">${esc(state.user.full_name)}<small>${esc(state.user.email)}</small></span><button class="icon-button" id="signout" title="Sign out" aria-label="Sign out">${icon("logout")}</button></div></div></aside><main class="main-area"><header class="topbar"><div><div class="eyebrow">ZONEPULSE / ${esc(state.page.toUpperCase())}</div><div class="top-title">Mobility operations</div></div><div class="top-actions"><span class="server-pill"><span class="live-dot"></span>Connected</span><span class="date-chip">${new Date().toLocaleDateString([], { weekday: "short", month: "short", day: "numeric" })}</span></div></header><section class="page-content" id="page-content"><div class="loading-card">Loading your workspace…</div></section><footer>ZonePulse <span>·</span> Global mobility planning <span>·</span> Estimates are illustrative</footer></main></div>`;
  document.querySelectorAll("[data-page]").forEach((button) => button.addEventListener("click", () => { state.page = button.dataset.page; renderShell(); loadPage(); }));
  document.querySelector("#signout").addEventListener("click", () => { clearSession(); renderAuth(); notify("You are signed out."); });
}

const setPage = (markup) => { const target = document.querySelector("#page-content"); if (target) target.innerHTML = markup; };
function hero(kicker, title, copy, action = "") { return `<div class="page-heading"><div><div class="eyebrow">${kicker}</div><h1>${title}</h1><p>${copy}</p></div>${action}</div>`; }

async function loadOverview() {
  const [overview, trips] = await Promise.all([api("/api/overview"), api("/api/ride/history")]);
  state.user = overview.account; state.trips = trips;
  setPage(`${hero("YOUR ACCOUNT / OVERVIEW", `Welcome back, ${esc(state.user.full_name.split(" ")[0])}.`, "Your trip activity and mobility planning at a glance.", '<button class="button primary" data-go-plan>Plan a ride <span>→</span></button>')}<div class="metric-grid"><article class="metric-card"><span class="metric-label">SAVED TRIPS</span><span class="metric-icon mint">${icon("route")}</span><strong>${trips.length}</strong><small>Across your account</small></article><article class="metric-card"><span class="metric-label">DRIVER PROFILES</span><span class="metric-icon blue">${icon("car")}</span><strong>${overview.drivers}</strong><small>Simulated demonstration fleet</small></article><article class="metric-card"><span class="metric-label">LOCATION COVERAGE</span><span class="metric-icon gold">${icon("pin")}</span><strong>Global</strong><small>Worldwide place search</small></article><article class="metric-card"><span class="metric-label">ACCOUNT STATUS</span><span class="metric-icon mint">${icon("check")}</span><strong class="status-value">Active</strong><small>${esc(state.user.email)}</small></article></div><div class="overview-grid"><section class="surface-card"><div class="card-heading"><div><div class="eyebrow">RECENT ACTIVITY</div><h2>Recent trips</h2></div><button class="text-button" data-go-history>View all ${icon("arrow", 16)}</button></div>${trips.length ? trips.slice(0,4).map(tripRow).join("") : '<div class="empty-inline"><span class="metric-icon mint">'+icon("route")+'</span><strong>Your next trip starts here.</strong><p>Search any city, address, or landmark to build a ride estimate.</p><button class="button secondary" data-go-plan>Plan your first ride</button></div>'}</section><section class="surface-card insight-card"><div class="eyebrow">PLANNING SNAPSHOT</div><h2>Global by design.</h2><p>Find pickup and destination places worldwide, review current pickup weather, and compare estimated wait, availability, and fare options.</p><div class="insight-line"><span>${icon("cloud")}</span><div><strong>Weather aware</strong><small>Current local conditions from Open-Meteo</small></div></div><div class="insight-line"><span>${icon("pin")}</span><div><strong>Worldwide search</strong><small>Address and landmark discovery</small></div></div><button class="button secondary full" data-go-plan>Open ride planner <span>→</span></button></section></div>`);
  document.querySelector("[data-go-plan]")?.addEventListener("click", () => navigate("Plan a ride"));
  document.querySelector("[data-go-history]")?.addEventListener("click", () => navigate("Trip history"));
}

function tripRow(trip) { return `<div class="trip-row"><span class="trip-route-icon">${icon("route")}</span><div class="trip-route"><strong>${esc(trip.pickup_address)}</strong><span>${esc(trip.dropoff_address)}</span></div><div class="trip-meta"><strong>${money(trip.fare)}</strong><span>${fmtDate(trip.created_at)}</span></div><span class="trip-status">${esc(trip.status || "Confirmed")}</span></div>`; }

function navigate(page) { state.page = page; renderShell(); loadPage(); }

function renderPlanner() {
  state.quote = null;
  setPage(`${hero("ROUTE PLANNING", "Plan a ride.", "Search a pickup and destination anywhere. Review the local weather, then compare an illustrative trip estimate.")}<div class="planner-layout"><section class="surface-card planner-card"><div class="card-heading"><div><div class="eyebrow">TRIP DETAILS</div><h2>Choose your route</h2></div><span class="step-tag">01 / ROUTE</span></div><div class="place-field"><label for="pickup-search">Pickup location</label><div class="search-box">${icon("pin")}<input id="pickup-search" placeholder="Search a city, street, airport…" autocomplete="off"/><button class="search-action" data-search="pickup" aria-label="Search pickup">${icon("search")}</button></div><div class="suggestions" id="pickup-results"></div></div><div class="place-field"><label for="dropoff-search">Destination</label><div class="search-box">${icon("route")}<input id="dropoff-search" placeholder="Where are you going?" autocomplete="off"/><button class="search-action" data-search="dropoff" aria-label="Search destination">${icon("search")}</button></div><div class="suggestions" id="dropoff-results"></div></div><div id="selected-route" class="selected-route"><span class="route-line"></span><div class="route-point">Select a pickup and destination to see your route</div></div><div id="weather-panel"></div><div class="plan-controls"><label>Pickup date and time <span class="optional">Pickup local time</span><input id="pickup-time" type="datetime-local" /></label></div><button class="button primary full estimate-button" id="get-estimate">Get ride estimate <span>→</span></button><div class="form-message" id="planner-message"></div><p class="disclaimer">Estimated fares use USD-equivalent planning rates. Driver availability and route times are simulated.</p></section><aside class="surface-card planner-aside"><div class="eyebrow">A CLEARER PICKUP</div><h2>Know what to expect.</h2><p>Check the pickup area's current local weather and time before reviewing demand and fare estimates.</p><div class="aside-stat"><span>${icon("cloud")}</span><div><strong>Current conditions</strong><small>Live global weather API</small></div></div><div class="aside-stat"><span>${icon("clock")}</span><div><strong>Location local time</strong><small>Displayed in the pickup time zone</small></div></div><div class="aside-stat"><span>${icon("route")}</span><div><strong>Saved to your account</strong><small>Your trip history stays private</small></div></div><div class="estimate-placeholder"><span class="placeholder-icon">${icon("route", 28)}</span><strong>Your estimate will appear here</strong><small>Choose both locations to continue</small></div></aside></div><div id="quote-result"></div>`);
  const timeInput = document.querySelector("#pickup-time");
  timeInput.value = new Date(Date.now() - new Date().getTimezoneOffset() * 60_000).toISOString().slice(0,16);
  for (const kind of ["pickup", "dropoff"]) {
    const input = document.querySelector(`#${kind}-search`);
    input.addEventListener("keydown", (event) => {
      if (event.key !== "Enter") return;
      event.preventDefault();
      if (state.places[kind].length) selectPlace(kind, 0);
      else searchPlaces(kind);
    });
    document.querySelector(`[data-search="${kind}"]`).addEventListener("click", () => searchPlaces(kind));
    input.addEventListener("input", () => {
      state.places[kind] = []; state.quote = null;
      document.querySelector("#quote-result").innerHTML = "";
      if (kind === "pickup") { state.weather = null; renderWeather(); }
      updateRouteLabel();
      clearTimeout(state.searchTimers?.[kind]);
      const query = input.value.trim(); const results = document.querySelector(`#${kind}-results`);
      results.classList.remove("is-selected");
      if (query.length < 3) { results.innerHTML = query ? '<div class="suggestion-empty">Type at least three characters to search worldwide.</div>' : ""; return; }
      results.innerHTML = '<div class="suggestion-empty">Searching worldwide places…</div>';
      state.searchTimers ||= {}; state.searchTimers[kind] = setTimeout(() => searchPlaces(kind), 350);
    });
  }
  document.querySelector("#get-estimate").addEventListener("click", getEstimate);
  updateRouteLabel();
}

async function searchPlaces(kind) {
  clearTimeout(state.searchTimers?.[kind]);
  const query = document.querySelector(`#${kind}-search`).value.trim(); const results = document.querySelector(`#${kind}-results`);
  results.classList.remove("is-selected");
  if (query.length < 3) { results.innerHTML = '<div class="suggestion-empty">Enter at least three characters.</div>'; return; }
  state.searchSequence ||= {};
  const requestSequence = state.searchSequence[kind] = (state.searchSequence[kind] || 0) + 1;
  results.innerHTML = '<div class="suggestion-empty">Searching worldwide places…</div>';
  try {
    const data = await api(`/api/geocode?q=${encodeURIComponent(query)}`);
    if (requestSequence !== state.searchSequence[kind] || document.querySelector(`#${kind}-search`).value.trim() !== query) return;
    state.places[kind] = data;
    results.innerHTML = data.length ? `<div class="suggestion-heading">${data.length} matching places · choose one</div>${data.map((item, index) => `<button class="suggestion" type="button" data-place="${kind}" data-index="${index}"><span>${icon("pin",17)}</span><strong>${esc(item.label)}</strong>${icon("arrow",15)}</button>`).join("")}` : '<div class="suggestion-empty">No match found. Add a city or country name and try again.</div>';
    results.querySelectorAll("[data-place]").forEach((button) => button.addEventListener("click", () => selectPlace(kind, Number(button.dataset.index))));
  } catch (error) {
    if (requestSequence !== state.searchSequence[kind]) return;
    results.innerHTML = `<div class="suggestion-empty error-text">${esc(error.message)}</div>`;
  }
}

async function selectPlace(kind, index) {
  const place = state.places[kind][index]; if (!place) return;
  state.places[kind] = [place]; document.querySelector(`#${kind}-search`).value = place.label;
  const results = document.querySelector(`#${kind}-results`);
  results.classList.add("is-selected");
  results.innerHTML = `<div class="selected-place">${icon("check",16)}<span>${esc(place.label)}</span><button class="clear-place" data-clear="${kind}" title="Clear" aria-label="Clear ${kind} location">×</button></div>`;
  document.querySelector(`[data-clear="${kind}"]`).addEventListener("click", () => { state.places[kind] = []; document.querySelector(`#${kind}-search`).value = ""; results.classList.remove("is-selected"); results.innerHTML = ""; if (kind === "pickup") { state.weather = null; renderWeather(); } updateRouteLabel(); });
  updateRouteLabel();
  if (kind === "pickup") await loadWeather(place);
}

function updateRouteLabel() {
  const el = document.querySelector("#selected-route"); if (!el) return;
  const pickup = state.places.pickup[0]; const dropoff = state.places.dropoff[0];
  if (pickup && dropoff) el.innerHTML = `<span class="route-line"></span><div class="route-point"><strong>${esc(pickup.label)}</strong><span>Pickup</span></div><div class="route-point"><strong>${esc(dropoff.label)}</strong><span>Destination</span></div>`;
  else el.innerHTML = `<span class="route-line"></span><div class="route-point">${pickup ? `<strong>${esc(pickup.label)}</strong><span>Choose a destination to complete your route</span>` : "Select a pickup and destination to see your route"}</div>`;
}

async function loadWeather(place, quiet = false) {
  const panel = document.querySelector("#weather-panel"); if (!quiet && panel) panel.innerHTML = '<div class="weather-loading">Loading pickup weather…</div>';
  try {
    const weather = await api(`/api/weather?lat=${place.lat}&lon=${place.lon}`); state.weather = { ...weather, coords: `${place.lat.toFixed(3)},${place.lon.toFixed(3)}`, loadedAt: Date.now() };
    renderWeather();
    if (state.clockTimer) clearInterval(state.clockTimer);
    state.clockTimer = setInterval(renderLocationClock, 1000);
    if (state.weatherRefreshTimer) clearInterval(state.weatherRefreshTimer);
    state.weatherRefreshTimer = setInterval(() => {
      if (state.page === "Plan a ride" && state.places.pickup[0]?.label === place.label) loadWeather(place, true);
    }, 600_000);
  } catch (error) { state.weather = null; if (panel) panel.innerHTML = `<div class="weather-failure">${esc(error.message)}</div>`; }
}

function renderLocationClock() {
  const clock = document.querySelector("#local-clock"); if (!clock || !state.weather?.timezone) return;
  clock.textContent = new Intl.DateTimeFormat(undefined, { timeZone: state.weather.timezone, hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: true }).format(new Date());
}

function renderWeather() {
  const panel = document.querySelector("#weather-panel"); if (!panel) return;
  if (!state.weather) { panel.innerHTML = ""; return; }
  const w = state.weather; const units = w.units || {};
  panel.innerHTML = `<div class="weather-card"><div class="weather-title"><span class="weather-icon">${icon("cloud",21)}</span><div><div class="eyebrow">PICKUP CONDITIONS</div><strong>${esc(w.description)}</strong></div><button id="refresh-weather" class="icon-button" title="Refresh weather" aria-label="Refresh weather">${icon("refresh")}</button></div><div class="weather-values"><div><strong>${w.temperature}${esc(units.temperature_2m || "°C")}</strong><small>Temperature</small></div><div><strong>${w.feels_like}${esc(units.apparent_temperature || "°C")}</strong><small>Feels like</small></div><div><strong>${w.humidity}${esc(units.relative_humidity_2m || "%")}</strong><small>Humidity</small></div><div><strong>${w.wind_speed} ${esc(units.wind_speed_10m || "km/h")}</strong><small>Wind</small></div></div><div class="weather-time">${icon("clock",15)}<span>Pickup local time</span><strong id="local-clock">—</strong><small>${esc(w.timezone)}</small></div><div class="weather-updated">Conditions updated ${esc(w.time)} · Open-Meteo</div></div>`;
  document.querySelector("#refresh-weather").addEventListener("click", () => loadWeather(state.places.pickup[0])); renderLocationClock();
}

async function getEstimate() {
  const pickup = state.places.pickup[0], dropoff = state.places.dropoff[0]; const message = document.querySelector("#planner-message");
  if (!pickup || !dropoff) { message.textContent = "Choose a pickup and destination from the worldwide search results."; return; }
  if (pickup.lat === dropoff.lat && pickup.lon === dropoff.lon) { message.textContent = "Choose two different locations for this trip."; return; }
  const button = document.querySelector("#get-estimate"); button.disabled = true; button.innerHTML = "Preparing your estimate…"; message.textContent = "";
  try {
    const quote = await api("/api/ride/quote", { method: "POST", body: {
      pickup_address: pickup.label, pickup_lat: pickup.lat, pickup_lon: pickup.lon,
      dropoff_address: dropoff.label, dropoff_lat: dropoff.lat, dropoff_lon: dropoff.lon,
      pickup_local_time: document.querySelector("#pickup-time").value, weather: state.weather,
    } });
    state.quote = quote; renderQuote(quote);
  } catch (error) { message.textContent = error.message; }
  finally { button.disabled = false; button.innerHTML = 'Get ride estimate <span>→</span>'; }
}

function renderQuote(quote) {
  const out = document.querySelector("#quote-result");
  out.innerHTML = `<section class="quote-panel"><div class="card-heading"><div><div class="eyebrow">YOUR RIDE ESTIMATE</div><h2>Route and ride options</h2></div><span class="demand-pill ${quote.demand_tier.toLowerCase().replace(" ", "-")}">${esc(quote.demand_tier)} demand</span></div><div class="quote-route"><div class="quote-location"><span class="route-dot start"></span><div><small>PICKUP</small><strong>${esc(quote.pickup_address)}</strong></div></div><span class="quote-connector"></span><div class="quote-location"><span class="route-dot end"></span><div><small>DESTINATION</small><strong>${esc(quote.dropoff_address)}</strong></div></div><div class="route-distance">${quote.distance_km} km <span>·</span> ${quote.duration_minutes} min</div></div><div class="quote-metrics"><div><small>PICKUP WAIT</small><strong>${quote.estimated_wait_time_minutes} <small>min</small></strong></div><div><small>EST. AVAILABILITY</small><strong>${quote.probability_finding_car_pct}%</strong></div><div><small>FARE ADJUSTMENT</small><strong>${quote.surge_multiplier}×</strong></div><div><small>DEMAND FORECAST</small><strong>${quote.predicted_hourly_demand} <small>rides/hr</small></strong></div></div><p class="quote-note">${esc(quote.prediction_note)} ${quote.weather_note ? `Pickup conditions: ${esc(quote.weather_note)}.` : ""}</p><div class="ride-options">${Object.entries(quote.fares).map(([name, fare], i) => `<button class="ride-option ${i === 0 ? "chosen" : ""}" data-tier="${esc(name)}" data-fare="${fare}"><span class="car-badge">${icon("car",20)}</span><span><strong>${esc(name)}</strong><small>${i === 2 ? "Room for up to 6" : i === 1 ? "Extra comfort" : "Everyday ride"}</small></span><b>${money(fare)}</b><span class="option-check">${icon("check",15)}</span></button>`).join("")}</div><div class="book-controls"><label>Passenger contact number <span class="optional">Optional</span><input id="booking-phone" type="tel" maxlength="40" value="${esc(state.user.phone || "")}" placeholder="Country code + number" /></label><label>Suggested driver<select id="driver-select">${quote.nearby_drivers.map((driver, i) => `<option value="${i}">${esc(driver.name)} · ${esc(driver.car_model)} · ${driver.eta_minutes} min · ★ ${driver.rating}</option>`).join("")}</select></label></div><button class="button primary" id="confirm-booking">Confirm and save trip <span>→</span></button><div class="form-message" id="booking-message"></div></section>`;
  out.querySelectorAll(".ride-option").forEach((button) => button.addEventListener("click", () => { out.querySelectorAll(".ride-option").forEach((item) => item.classList.toggle("chosen", item === button)); }));
  document.querySelector("#confirm-booking").addEventListener("click", bookTrip);
}

async function bookTrip() {
  const button = document.querySelector("#confirm-booking"), message = document.querySelector("#booking-message"), selected = document.querySelector(".ride-option.chosen"), driver = state.quote.nearby_drivers[Number(document.querySelector("#driver-select").value)];
  button.disabled = true; button.textContent = "Saving your trip…";
  try {
    const body = { pickup_address: state.quote.pickup_address, pickup_lat: state.places.pickup[0].lat, pickup_lon: state.places.pickup[0].lon,
      dropoff_address: state.quote.dropoff_address, dropoff_lat: state.places.dropoff[0].lat, dropoff_lon: state.places.dropoff[0].lon,
      distance_km: state.quote.distance_km, duration_mins: state.quote.duration_minutes, ride_tier: selected.dataset.tier, fare: Number(selected.dataset.fare),
      phone: document.querySelector("#booking-phone").value.trim(), predicted_hourly_demand: state.quote.predicted_hourly_demand, demand_tier: state.quote.demand_tier,
      surge_multiplier: state.quote.surge_multiplier, probability_finding_car_pct: state.quote.probability_finding_car_pct, driver };
    const result = await api("/api/ride/book", { method: "POST", body });
    notify(`Trip saved · ${result.booking.booking_id}`, "success"); state.quote = null; state.places = { pickup: [], dropoff: [] }; state.weather = null; navigate("Trip history");
  } catch (error) { message.textContent = error.message; }
  finally { if (button.isConnected) { button.disabled = false; button.innerHTML = 'Confirm and save trip <span>→</span>'; } }
}

async function loadHistory() {
  const trips = await api("/api/ride/history"); state.trips = trips;
  setPage(`${hero("YOUR ACCOUNT / TRIP HISTORY", "Your saved trips.", "A private record of trips saved to your account.", '<button class="button primary" data-go-plan>Plan a ride <span>→</span></button>')}<section class="surface-card history-card"><div class="card-heading"><div><div class="eyebrow">BOOKING RECORDS</div><h2>${trips.length} ${trips.length === 1 ? "trip" : "trips"}</h2></div><span class="step-tag">MOST RECENT FIRST</span></div>${trips.length ? trips.map((trip) => `<article class="history-item"><div class="history-icon">${icon("route")}</div><div class="history-main"><div class="history-route"><strong>${esc(trip.pickup_address)}</strong><span>→</span><strong>${esc(trip.dropoff_address)}</strong></div><div class="history-sub">${fmtDate(trip.created_at)} · ${esc(trip.booking_id)} · ${Number(trip.distance_km || 0).toFixed(1)} km</div></div><div class="history-fare"><strong>${money(trip.fare)}</strong><span>${esc(trip.ride_tier || "Ride")}</span></div><span class="trip-status">${esc(trip.status)}</span></article>`).join("") : '<div class="empty-inline"><span class="metric-icon mint">'+icon("clock")+'</span><strong>No saved trips yet.</strong><p>When you confirm a trip, it will appear here.</p><button class="button secondary" data-go-plan>Plan a ride</button></div>'}</section>`);
  document.querySelector("[data-go-plan]")?.addEventListener("click", () => navigate("Plan a ride"));
}

async function loadFleet() {
  const drivers = await api("/api/drivers"); state.drivers = drivers;
  setPage(`${hero("DEMONSTRATION FLEET", "Driver profiles.", "Driver and vehicle details observed during your ride estimates. Availability is simulated.")}<section class="surface-card history-card"><div class="card-heading"><div><div class="eyebrow">SAVED PROFILES</div><h2>${drivers.length} drivers</h2></div></div>${drivers.length ? `<div class="table-wrap"><table><thead><tr><th>DRIVER</th><th>VEHICLE</th><th>PLATE</th><th>RATING</th><th>EST. ETA</th><th>STATUS</th></tr></thead><tbody>${drivers.map((d) => `<tr><td>${esc(d.name)}</td><td>${esc(`${d.car_color || ""} ${d.car_model || ""}`)}</td><td>${esc(d.license_plate || "—")}</td><td>★ ${Number(d.rating || 0).toFixed(2)}</td><td>${d.eta_minutes || "—"} min</td><td><span class="trip-status">${esc(d.status)}</span></td></tr>`).join("")}</tbody></table></div>` : '<div class="empty-inline"><span class="metric-icon blue">'+icon("car")+'</span><strong>No driver records yet.</strong><p>Suggested profiles are saved when you estimate and confirm a ride.</p><button class="button secondary" data-go-plan>Get a ride estimate</button></div>'}</section>`);
  document.querySelector("[data-go-plan]")?.addEventListener("click", () => navigate("Plan a ride"));
}

function renderInsights() {
  setPage(`${hero("MOBILITY INTELLIGENCE", "Demand insights.", "Understand what a planning estimate represents, and where it comes from.")}<div class="insight-stats"><div class="surface-card"><div class="eyebrow">SEARCH COVERAGE</div><strong>Worldwide</strong><p>Global geocoding for addresses, airports, landmarks, and cities.</p></div><div class="surface-card"><div class="eyebrow">WEATHER</div><strong>Current conditions</strong><p>Location-aware weather and local pickup time from Open-Meteo.</p></div><div class="surface-card"><div class="eyebrow">DEMAND MODEL</div><strong>Planning baseline</strong><p>Time-of-week demand estimates are directional; they are not live local predictions.</p></div></div><section class="notice-card"><span>${icon("chart",22)}</span><div><strong>About these estimates</strong><p>Demand, availability, pickup time, and USD-equivalent fares are planning estimates. Driver profiles are simulated and no live ride-hailing or traffic provider is connected.</p></div></section>`);
}

function renderProfile() {
  const user = state.user;
  setPage(`${hero("ACCOUNT SETTINGS", "Your account.", "Review your saved sign-in details and update your contact number.")}<section class="surface-card profile-card"><div class="card-heading"><div><div class="eyebrow">PERSONAL DETAILS</div><h2>Account profile</h2></div><span class="trip-status">${esc(user.role || "Active account")}</span></div><form id="profile-form" class="profile-form"><label>Full name<input name="full_name" value="${esc(user.full_name)}" minlength="3" maxlength="120" required /></label><label>Email address<input value="${esc(user.email)}" readonly aria-readonly="true" /></label><label>Account username<input value="${esc(user.username)}" readonly aria-readonly="true" /></label><label>Contact number <span class="optional">Optional</span><input name="phone" type="tel" autocomplete="tel" value="${esc(user.phone || "")}" maxlength="40" placeholder="Country code + number" /></label><button class="button primary" type="submit">Save account details <span>→</span></button><div class="form-message" aria-live="polite"></div></form><div class="profile-security"><span>${icon("check",17)}</span><div><strong>Password protected</strong><small>Your password is stored as a secure one-way hash and is never shown here.</small></div></div></section>`);
  document.querySelector("#profile-form").addEventListener("submit", async (event) => {
    event.preventDefault(); const form = event.currentTarget; const button = form.querySelector("button[type=submit]");
    button.disabled = true; button.textContent = "Saving…";
    try {
      const result = await api("/api/profile", { method: "PATCH", body: { full_name: form.full_name.value.trim(), phone: form.phone.value.trim() } });
      state.user = result.user; renderShell(); renderProfile(); notify("Account details saved.", "success");
    } catch (error) { form.querySelector(".form-message").textContent = error.message; }
    finally { if (button.isConnected) { button.disabled = false; button.innerHTML = 'Save account details <span>→</span>'; } }
  });
}

async function loadPage() {
  try {
    if (state.page === "Overview") return await loadOverview();
    if (state.page === "Plan a ride") return renderPlanner();
    if (state.page === "Trip history") return await loadHistory();
    if (state.page === "Driver fleet") return await loadFleet();
    if (state.page === "My account") return renderProfile();
    return renderInsights();
  } catch (error) {
    if (error.message.includes("session") || error.message.includes("Sign in")) return;
    setPage(`<div class="empty-state"><div class="empty-icon">${icon("refresh",24)}</div><h3>Couldn’t load this section</h3><p>${esc(error.message)}</p><button class="button secondary" id="retry-page">Try again</button></div>`);
    document.querySelector("#retry-page")?.addEventListener("click", loadPage);
  }
}

async function start() {
  if (!state.token) return renderAuth();
  try { const { user } = await api("/api/auth/me"); state.user = user; renderShell(); loadPage(); }
  catch (error) {
    if (!state.token) return renderAuth();
    app.innerHTML = `<main class="startup-error"><div class="brand-mark">Z</div><h1>We couldn’t reconnect.</h1><p>${esc(error.message)} Your saved sign-in stays on this device.</p><button class="button primary" id="retry-start">Try again</button><button class="link-button" id="switch-account">Sign in with another account</button></main>`;
    document.querySelector("#retry-start").addEventListener("click", start);
    document.querySelector("#switch-account").addEventListener("click", () => { clearSession(); renderAuth(); });
  }
}
start();

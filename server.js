 1 file changed, 39 insertions(+), 12 deletions(-)
35d0d72befd4f2224b0b0d244db61b964b199726
import "dotenv/config";
import express from "express";
import helmet from "helmet";
import rateLimit from "express-rate-limit";
import bcrypt from "bcryptjs";
import jwt from "jsonwebtoken";
import pg from "pg";
import crypto from "node:crypto";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { quoteRide, weatherDescription } from "./server/domain.js";

const { Pool } = pg;
const app = express();
const here = path.dirname(fileURLToPath(import.meta.url));
const port = Number(process.env.PORT || 3000);
const isProduction = process.env.NODE_ENV === "production";
const databaseUrl = process.env.DATABASE_URL;
const jwtSecret = process.env.JWT_SECRET_KEY || (!isProduction ? "local-zonepulse-development-secret-change-me" : "");

if (!databaseUrl) throw new Error("DATABASE_URL is required. Configure the persistent PostgreSQL database before starting ZonePulse.");
if (!jwtSecret) throw new Error("JWT_SECRET_KEY is required in production.");

const pool = new Pool({
  connectionString: databaseUrl,
  ssl: process.env.PGSSL === "true" ? { rejectUnauthorized: false } : undefined,
  max: 5,
  idleTimeoutMillis: 30_000,
  connectionTimeoutMillis: 8_000,
});

app.disable("x-powered-by");
app.set("trust proxy", 1);
app.use(helmet({ contentSecurityPolicy: false, crossOriginEmbedderPolicy: false }));
app.use(express.json({ limit: "64kb" }));
app.use("/api/auth", rateLimit({ windowMs: 15 * 60_000, limit: 30, standardHeaders: "draft-8", legacyHeaders: false }));

async function initializeDatabase() {
  await pool.query(`
    CREATE TABLE IF NOT EXISTS passengers (
      id SERIAL PRIMARY KEY,
      username VARCHAR(50) NOT NULL UNIQUE,
      full_name VARCHAR(120) NOT NULL,
      email VARCHAR(120) NOT NULL UNIQUE,
      password_hash TEXT NOT NULL,
      role VARCHAR(80) NOT NULL DEFAULT 'Mobility Operations Analyst',
      phone VARCHAR(40),
      created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    CREATE TABLE IF NOT EXISTS drivers (
      driver_id VARCHAR(40) PRIMARY KEY,
      name VARCHAR(120) NOT NULL,
      rating DOUBLE PRECISION,
      car_model VARCHAR(120),
      car_color VARCHAR(40),
      license_plate VARCHAR(40),
      phone VARCHAR(40),
      latitude DOUBLE PRECISION,
      longitude DOUBLE PRECISION,
      distance_km DOUBLE PRECISION,
      eta_minutes INTEGER,
      status VARCHAR(40) NOT NULL DEFAULT 'Available',
      created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
      updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    CREATE TABLE IF NOT EXISTS rides (
      id SERIAL PRIMARY KEY,
      booking_id VARCHAR(32) NOT NULL UNIQUE,
      passenger_id INTEGER NOT NULL REFERENCES passengers(id),
      driver_id VARCHAR(40) REFERENCES drivers(driver_id),
      passenger_name VARCHAR(120) NOT NULL,
      passenger_email VARCHAR(120) NOT NULL,
      passenger_phone VARCHAR(40),
      pickup_address VARCHAR(240) NOT NULL,
      pickup_lat DOUBLE PRECISION,
      pickup_lon DOUBLE PRECISION,
      dropoff_address VARCHAR(240) NOT NULL,
      dropoff_lat DOUBLE PRECISION,
      dropoff_lon DOUBLE PRECISION,
      distance_km DOUBLE PRECISION,
      duration_mins INTEGER,
      ride_tier VARCHAR(80),
      fare DOUBLE PRECISION,
      predicted_hourly_demand DOUBLE PRECISION,
      demand_tier VARCHAR(40),
      surge_multiplier DOUBLE PRECISION,
      probability_finding_car_pct DOUBLE PRECISION,
      driver_name VARCHAR(120),
      driver_rating DOUBLE PRECISION,
      car_model VARCHAR(120),
      car_color VARCHAR(40),
      car_plate VARCHAR(40),
      driver_lat DOUBLE PRECISION,
      driver_lon DOUBLE PRECISION,
      driver_distance_km DOUBLE PRECISION,
      eta_mins INTEGER,
      status VARCHAR(40) NOT NULL DEFAULT 'Confirmed',
      created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    CREATE INDEX IF NOT EXISTS ix_passengers_username ON passengers(username);
    CREATE INDEX IF NOT EXISTS ix_passengers_email ON passengers(email);
    CREATE INDEX IF NOT EXISTS ix_rides_passenger_id ON rides(passenger_id);
    CREATE INDEX IF NOT EXISTS ix_rides_created_at ON rides(created_at);
  `);
}

const safeUser = (row) => ({
  id: row.id,
  username: row.username,
  full_name: row.full_name,
  email: row.email,
  phone: row.phone || "",
  role: row.role,
  created_at: row.created_at,
});

function issueToken(user) {
  return jwt.sign({ sub: user.username, role: user.role }, jwtSecret, { expiresIn: "7d", issuer: "zonepulse" });
}

function authenticate(req, res, next) {
  const token = req.get("authorization")?.match(/^Bearer\s+(.+)$/i)?.[1];
  if (!token) return res.status(401).json({ error: "Sign in to continue." });
  try {
    req.identity = jwt.verify(token, jwtSecret, { issuer: "zonepulse" });
    next();
  } catch {
    return res.status(401).json({ error: "Your session expired. Sign in again." });
  }
}

async function currentPassenger(req) {
  const { rows } = await pool.query("SELECT * FROM passengers WHERE username = $1", [req.identity.sub]);
  return rows[0] || null;
}

function requireText(value, label, min = 1, max = 160) {
  if (typeof value !== "string" || value.trim().length < min || value.trim().length > max) {
    const error = new Error(`${label} must be between ${min} and ${max} characters.`);
    error.status = 400;
    throw error;
  }
  return value.trim();
}

function oldPbkdf2Verify(password, saved) {
  const match = typeof saved === "string" && saved.match(/^([a-f\d]{32})\$([a-f\d]{64})$/i);
  if (!match) return false;
  const candidate = crypto.pbkdf2Sync(password, match[1], 100_000, 32, "sha256").toString("hex");
  return crypto.timingSafeEqual(Buffer.from(candidate, "hex"), Buffer.from(match[2], "hex"));
}

async function passwordMatches(password, saved) {
  if (typeof saved === "string" && saved.startsWith("$2")) return bcrypt.compare(crypto.createHash("sha256").update(password, "utf8").digest("hex"), saved);
  return oldPbkdf2Verify(password, saved);
}

async function makePasswordHash(password) {
  const normalized = crypto.createHash("sha256").update(password, "utf8").digest("hex");
  return bcrypt.hash(normalized, 12);
}

app.get("/health", async (_req, res) => {
  try {
    await pool.query("SELECT 1");
    res.json({ status: "ok", service: "ZonePulse", database: "connected" });
  } catch {
    res.status(503).json({ status: "unavailable", database: "disconnected" });
  }
});

app.post("/api/auth/signup", async (req, res, next) => {
  try {
    const fullName = requireText(req.body.full_name ?? req.body.username, "Full name", 3, 120);
    const email = requireText(req.body.email, "Email", 5, 120).toLowerCase();
    const password = requireText(req.body.password, "Password", 6, 100);
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) return res.status(400).json({ error: "Enter a valid email address." });
    const phone = typeof req.body.phone === "string" ? req.body.phone.trim().slice(0, 40) : null;
    const username = (typeof req.body.username === "string" && req.body.username.trim()
      ? req.body.username.trim() : fullName).toLowerCase().slice(0, 50);
    const { rows: existingRows } = await pool.query(
      "SELECT * FROM passengers WHERE username = $1 OR email = $2 LIMIT 1", [username, email]);
    if (existingRows[0]) {
      const existing = existingRows[0];
      if (existing.username === username && existing.email === email && await passwordMatches(password, existing.password_hash)) {
        return res.json({ user: safeUser(existing), token: issueToken(existing), message: "Your account is ready. Welcome back." });
      }
      return res.status(409).json({ error: "That username or email is already registered. Sign in with the account you created." });
    }
    const passwordHash = await makePasswordHash(password);
    const { rows } = await pool.query(
      `INSERT INTO passengers (username, full_name, email, password_hash, phone)
       VALUES ($1, $2, $3, $4, $5) RETURNING *`, [username, fullName, email, passwordHash, phone || null]);
    res.status(201).json({ user: safeUser(rows[0]), token: issueToken(rows[0]), message: "Your account is ready." });
  } catch (error) {
    if (error.code === "23505") return res.status(409).json({ error: "That username or email is already registered. Sign in instead." });
    next(error);
  }
});

app.post("/api/auth/login", async (req, res, next) => {
  try {
    const identifier = requireText(req.body.identifier ?? req.body.username, "Email or username", 1, 120).toLowerCase();
    const password = requireText(req.body.password, "Password", 1, 100);
    const { rows } = await pool.query("SELECT * FROM passengers WHERE lower(username) = $1 OR lower(email) = $1 LIMIT 1", [identifier]);
    const passenger = rows[0];
    if (!passenger || !await passwordMatches(password, passenger.password_hash)) {
      return res.status(401).json({ error: "Email or password did not match an account. Check the saved email and password, then try again." });
    }
    if (!passenger.password_hash.startsWith("$2")) {
      const upgraded = await makePasswordHash(password);
      await pool.query("UPDATE passengers SET password_hash = $1 WHERE id = $2", [upgraded, passenger.id]);
      passenger.password_hash = upgraded;
    }
    res.json({ user: safeUser(passenger), token: issueToken(passenger), message: "Signed in successfully." });
  } catch (error) { next(error); }
});

app.get("/api/auth/me", authenticate, async (req, res, next) => {
  try {
    const passenger = await currentPassenger(req);
    if (!passenger) return res.status(401).json({ error: "This account is no longer available. Sign in again." });
    res.json({ user: safeUser(passenger) });
  } catch (error) { next(error); }
});

app.patch("/api/profile", authenticate, async (req, res, next) => {
  try {
    const passenger = await currentPassenger(req);
    if (!passenger) return res.status(401).json({ error: "Sign in to continue." });
    const fullName = requireText(req.body.full_name ?? passenger.full_name, "Full name", 3, 120);
    const phone = typeof req.body.phone === "string" ? req.body.phone.trim().slice(0, 40) : passenger.phone;
    const { rows } = await pool.query("UPDATE passengers SET full_name=$1, phone=$2 WHERE id=$3 RETURNING *", [fullName, phone || null, passenger.id]);
    res.json({ user: safeUser(rows[0]) });
  } catch (error) { next(error); }
});

app.get("/api/overview", authenticate, async (req, res, next) => {
  try {
    const passenger = await currentPassenger(req);
    if (!passenger) return res.status(401).json({ error: "Sign in to continue." });
    const [rides, drivers] = await Promise.all([
      pool.query("SELECT count(*)::int AS count FROM rides WHERE passenger_id=$1", [passenger.id]),
      pool.query("SELECT count(*)::int AS count FROM drivers"),
    ]);
    res.json({ trips: rides.rows[0].count, drivers: drivers.rows[0].count, account: safeUser(passenger) });
  } catch (error) { next(error); }
});

app.get("/api/geocode", authenticate, async (req, res, next) => {
  try {
    const query = requireText(req.query.q, "Place search", 3, 160);
    const openMeteoUrl = new URL("https://geocoding-api.open-meteo.com/v1/search");
    openMeteoUrl.search = new URLSearchParams({ name: query, count: "5", language: "en", format: "json" });
    try {
      const response = await fetch(openMeteoUrl, { signal: AbortSignal.timeout(8_000) });
      if (!response.ok) throw new Error(`Open-Meteo returned HTTP ${response.status}`);
      const body = await response.json();
      const places = (body.results || []).map((place) => {
        const label = [place.name, place.admin1, place.admin2, place.country]
          .filter(Boolean).filter((value, index, all) => all.indexOf(value) === index).join(", ");
        return { label, lat: Number(place.latitude), lon: Number(place.longitude), country_code: place.country_code || "" };
      }).filter((place) => Number.isFinite(place.lat) && Number.isFinite(place.lon));
      if (places.length) return res.json(places);
    } catch (error) {
      console.warn(`Open-Meteo geocoding failed (${error.cause?.code || error.name || "upstream error"}).`);
    }

    // Keep detailed address/landmark search available when the city geocoder has no match.
    try {
      const photonUrl = new URL("https://photon.komoot.io/api/");
      photonUrl.search = new URLSearchParams({ q: query, limit: "5", lang: "en" });
      const response = await fetch(photonUrl, {
        signal: AbortSignal.timeout(8_000),
        headers: { "User-Agent": "ZonePulse/2.0 (mobility planning)" },
      });
      if (!response.ok) throw new Error(`Photon returned HTTP ${response.status}`);
      const body = await response.json();
      const places = (body.features || []).flatMap((feature) => {
        const [lon, lat] = feature.geometry?.coordinates || [];
        if (!Number.isFinite(lat) || !Number.isFinite(lon)) return [];
        const p = feature.properties || {};
        const label = [p.name, p.street, p.city || p.county, p.state, p.country]
          .filter(Boolean).filter((value, index, all) => all.indexOf(value) === index).join(", ");
        return [{ label: label || query, lat: Number(lat), lon: Number(lon), country_code: p.countrycode || "" }];
      });
      return res.json(places);
    } catch (error) {
      console.error(`Location search providers failed (${error.cause?.code || error.name || "upstream error"}).`);
      return res.status(503).json({ error: "The global place-search services could not be reached. Please retry shortly." });
    }
  } catch (error) {
    if (error.name === "TimeoutError" || error.name === "AbortError") return res.status(503).json({ error: "Location search took too long. Try again in a moment." });
    next(error);
  }
});

app.get("/api/weather", authenticate, async (req, res, next) => {
  try {
    const lat = Number(req.query.lat); const lon = Number(req.query.lon);
    if (!Number.isFinite(lat) || lat < -90 || lat > 90 || !Number.isFinite(lon) || lon < -180 || lon > 180) {
      return res.status(400).json({ error: "Enter valid map coordinates to load weather." });
    }
    const query = new URLSearchParams({
      latitude: String(lat), longitude: String(lon), timezone: "auto",
      current: "temperature_2m,relative_humidity_2m,apparent_temperature,is_day,precipitation,weather_code,wind_speed_10m",
      temperature_unit: "celsius", wind_speed_unit: "kmh",
    });
    const response = await fetch(`https://api.open-meteo.com/v1/forecast?${query}`, { signal: AbortSignal.timeout(10_000) });
    if (!response.ok) return res.status(502).json({ error: "Current weather is temporarily unavailable. Please refresh shortly." });
    const data = await response.json(); const now = data.current;
    res.json({ time: now.time, timezone: data.timezone, timezone_abbreviation: data.timezone_abbreviation,
      temperature: now.temperature_2m, feels_like: now.apparent_temperature,
      humidity: now.relative_humidity_2m, precipitation: now.precipitation,
      wind_speed: now.wind_speed_10m, is_day: Boolean(now.is_day),
      weather_code: now.weather_code, description: weatherDescription(now.weather_code), units: data.current_units });
  } catch (error) {
    if (error.name === "TimeoutError" || error.name === "AbortError") return res.status(503).json({ error: "Current weather is taking too long to respond. Please refresh shortly." });
    next(error);
  }
});

app.post("/api/ride/quote", authenticate, async (req, res, next) => {
  try {
    const place = (value, prefix) => ({
      label: requireText(value[`${prefix}_address`], `${prefix} address`, 2, 240),
      lat: Number(value[`${prefix}_lat`]), lon: Number(value[`${prefix}_lon`]),
    });
    const origin = place(req.body, "pickup"); const destination = place(req.body, "dropoff");
    for (const coordinate of [origin.lat, destination.lat]) if (!Number.isFinite(coordinate) || coordinate < -90 || coordinate > 90) throw Object.assign(new Error("Enter valid pickup and destination map points."), { status: 400 });
    for (const coordinate of [origin.lon, destination.lon]) if (!Number.isFinite(coordinate) || coordinate < -180 || coordinate > 180) throw Object.assign(new Error("Enter valid pickup and destination map points."), { status: 400 });
    const localTime = String(req.body.pickup_local_time || "");
    const hour = Number(localTime.slice(11, 13));
    const date = localTime.slice(0, 10).split("-").map(Number);
    const dayOfWeek = date.length === 3 ? new Date(Date.UTC(date[0], date[1] - 1, date[2])).getUTCDay() : new Date().getUTCDay();
    const quote = quoteRide(origin, destination, Number.isInteger(hour) && hour >= 0 && hour <= 23 ? hour : 18, dayOfWeek);
    res.json({ ...quote, pickup_address: origin.label, dropoff_address: destination.label, weather_note: req.body.weather ? `${req.body.weather.description}, ${req.body.weather.temperature}°C` : null });
  } catch (error) { next(error); }
});

app.post("/api/ride/book", authenticate, async (req, res, next) => {
  let client;
  try {
    client = await pool.connect();
    const passenger = await currentPassenger(req);
    if (!passenger) return res.status(401).json({ error: "Sign in to continue." });
    const pickup = requireText(req.body.pickup_address, "Pickup address", 2, 240);
    const dropoff = requireText(req.body.dropoff_address, "Destination", 2, 240);
    const tier = requireText(req.body.ride_tier, "Ride option", 2, 80);
    const fare = Number(req.body.fare);
    if (!Number.isFinite(fare) || fare < 0) return res.status(400).json({ error: "Select a valid ride option before confirming." });
    const phone = typeof req.body.phone === "string" ? req.body.phone.trim().slice(0, 40) : passenger.phone;
    const driver = req.body.driver && typeof req.body.driver === "object" ? req.body.driver : {};
    const bookingId = `ZP-${crypto.randomBytes(5).toString("hex").toUpperCase()}`;
    const driverId = typeof driver.driver_id === "string" ? driver.driver_id.slice(0, 40) : null;
    await client.query("BEGIN");
    await client.query("UPDATE passengers SET phone=$1 WHERE id=$2", [phone || null, passenger.id]);
    if (driverId) {
      await client.query(`INSERT INTO drivers (driver_id,name,rating,car_model,car_color,license_plate,latitude,longitude,distance_km,eta_minutes,status,updated_at)
        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,'Available',now())
        ON CONFLICT (driver_id) DO UPDATE SET name=EXCLUDED.name,rating=EXCLUDED.rating,car_model=EXCLUDED.car_model,
        car_color=EXCLUDED.car_color,license_plate=EXCLUDED.license_plate,latitude=EXCLUDED.latitude,longitude=EXCLUDED.longitude,
        distance_km=EXCLUDED.distance_km,eta_minutes=EXCLUDED.eta_minutes,updated_at=now()`,
      [driverId,driver.name||"Assigned driver",driver.rating||null,driver.car_model||null,driver.car_color||null,driver.plate||null,driver.lat||null,driver.lon||null,driver.distance_km||null,driver.eta_minutes||null]);
    }
    const fields = [bookingId,passenger.id,driverId,passenger.username,passenger.email,phone||null,pickup,Number(req.body.pickup_lat),Number(req.body.pickup_lon),dropoff,Number(req.body.dropoff_lat),Number(req.body.dropoff_lon),Number(req.body.distance_km),Number(req.body.duration_mins),tier,fare,Number(req.body.predicted_hourly_demand),req.body.demand_tier||null,Number(req.body.surge_multiplier),Number(req.body.probability_finding_car_pct),driver.name||null,Number(driver.rating)||null,driver.car_model||null,driver.car_color||null,driver.plate||null,Number(driver.lat)||null,Number(driver.lon)||null,Number(driver.distance_km)||null,Number(driver.eta_minutes)||null];
    const { rows } = await client.query(`INSERT INTO rides (
      booking_id,passenger_id,driver_id,passenger_name,passenger_email,passenger_phone,pickup_address,pickup_lat,pickup_lon,
      dropoff_address,dropoff_lat,dropoff_lon,distance_km,duration_mins,ride_tier,fare,predicted_hourly_demand,demand_tier,
      surge_multiplier,probability_finding_car_pct,driver_name,driver_rating,car_model,car_color,car_plate,driver_lat,driver_lon,
      driver_distance_km,eta_mins,status) VALUES (${fields.map((_,i)=>`$${i+1}`).join(",")},'Driver En Route') RETURNING *`, fields);
    await client.query("COMMIT");
    res.status(201).json({ booking: rows[0], message: "Your trip has been saved." });
  } catch (error) {
    await client.query("ROLLBACK").catch(() => {});
    next(error);
  } finally { client?.release(); }
});

app.get("/api/ride/history", authenticate, async (req, res, next) => {
  try {
    const passenger = await currentPassenger(req);
    if (!passenger) return res.status(401).json({ error: "Sign in to continue." });
    const { rows } = await pool.query("SELECT * FROM rides WHERE passenger_id=$1 ORDER BY created_at DESC LIMIT 100", [passenger.id]);
    res.json(rows);
  } catch (error) { next(error); }
});

app.get("/api/drivers", authenticate, async (_req, res, next) => {
  try {
    const { rows } = await pool.query("SELECT * FROM drivers ORDER BY updated_at DESC, name LIMIT 100");
    res.json(rows);
  } catch (error) { next(error); }
});

app.get("/api/insights", authenticate, async (_req, res) => {
  res.json({ coverage: "Worldwide route search", model: "Global time-of-week baseline", note: "Ride demand, driver availability and fare figures are planning estimates. They are not live dispatch, traffic, or market quotes." });
});

app.use(express.static(path.join(here, "public"), {
  maxAge: isProduction ? "1h" : 0,
  setHeaders(res, filePath) {
    if (/\.(?:html|js|css)$/i.test(filePath)) res.setHeader("Cache-Control", "no-cache, max-age=0, must-revalidate");
  },
}));
app.get("*", (req, res, next) => req.path.startsWith("/api/") ? next() : res.sendFile(path.join(here, "public", "index.html")));

app.use((error, _req, res, _next) => {
  if (!res.headersSent) {
    const status = Number(error.status) || 500;
    if (status >= 500) console.error("Request failed:", error.message);
    res.status(status).json({ error: status >= 500 ? "The service is temporarily unavailable. Please try again in a moment." : error.message });
  }
});

initializeDatabase().then(() => {
  app.listen(port, "0.0.0.0", () => console.log(`ZonePulse Node app listening on ${port}`));
}).catch((error) => {
  console.error("ZonePulse could not connect to its configured database:", error.message);
  process.exit(1);
});


import crypto from "node:crypto";

const radians = (degrees) => degrees * Math.PI / 180;

function distanceKm(a, b) {
  const dLat = radians(b.lat - a.lat);
  const dLon = radians(b.lon - a.lon);
  const h = Math.sin(dLat / 2) ** 2 + Math.cos(radians(a.lat)) * Math.cos(radians(b.lat)) * Math.sin(dLon / 2) ** 2;
  return Math.round(6371 * 2 * Math.atan2(Math.sqrt(h), Math.sqrt(1 - h)) * 100) / 100;
}

export function weatherDescription(code) {
  if (code === 0) return "Clear sky";
  if ([1, 2, 3].includes(code)) return ["", "Mainly clear", "Partly cloudy", "Overcast"][code];
  if ([45, 48].includes(code)) return "Fog";
  if ([51, 53, 55, 56, 57].includes(code)) return "Drizzle";
  if ([61, 63, 65, 66, 67, 80, 81, 82].includes(code)) return "Rain";
  if ([71, 73, 75, 77, 85, 86].includes(code)) return "Snow";
  if ([95, 96, 99].includes(code)) return "Thunderstorm";
  return "Current conditions";
}

export function quoteRide(origin, destination, hour = 18, day = new Date().getDay()) {
  const distance = distanceKm(origin, destination);
  const speed = (hour >= 8 && hour <= 10) || (hour >= 17 && hour <= 20) ? 20 : (hour >= 23 || hour <= 5) ? 35 : 26;
  const duration = Math.max(6, Math.floor(distance / speed * 60) + 4);
  const weekend = day === 0 || day === 6;
  const rush = [7, 8, 9, 16, 17, 18, 19].includes(hour);
  const demand = Math.round(8 * (rush ? 1.45 : 1) * (weekend ? 1.15 : 1) * 10) / 10;
  const tier = demand < 5 ? "Low" : demand < 10 ? "Medium" : demand < 20 ? "High" : "Very High";
  const surge = tier === "Very High" ? 1.75 : tier === "High" ? 1.35 : tier === "Medium" ? 1.1 : 1;
  const baseFare = Math.max(8, (3.5 + distance * 1.85 + duration * 0.35) * surge);
  const roundedFare = Math.round(baseFare * 100) / 100;
  const availability = tier === "Low" ? 76 : tier === "Medium" ? 94 : tier === "High" ? 89 : 68;
  const seed = crypto.createHash("sha256").update(`${origin.lat.toFixed(3)}:${origin.lon.toFixed(3)}:${hour}`).digest();
  const names = ["Maya R.", "Daniel K.", "Aisha M.", "Carlos T.", "Elena S."];
  const cars = [["Toyota Camry Hybrid", "Black", "DEMO-7K82"], ["Honda Accord", "Silver", "DEMO-4M19"], ["Hyundai Sonata", "White", "DEMO-9B44"], ["Toyota RAV4", "Gray", "DEMO-3X71"], ["Tesla Model Y", "Blue", "DEMO-8Z50"]];
  const count = tier === "Low" ? 3 : tier === "Very High" ? 2 : 5;
  const drivers = Array.from({ length: count }, (_, i) => {
    const latDelta = ((seed[i * 3] / 255) - 0.5) * 0.011;
    const lonDelta = ((seed[i * 3 + 1] / 255) - 0.5) * 0.011;
    const lat = origin.lat + latDelta; const lon = origin.lon + lonDelta;
    const driverDistance = distanceKm(origin, { lat, lon });
    const [car, color, plate] = cars[i];
    return { driver_id: `DEMO-${seed.readUInt32BE(i)}`, name: names[i], rating: (4.86 + seed[i + 10] / 2550).toFixed(2) * 1,
      car_model: car, car_color: color, plate, lat, lon, distance_km: driverDistance,
      eta_minutes: Math.max(1, Math.floor(driverDistance / 22 * 60) + 1) };
  }).sort((a, b) => a.eta_minutes - b.eta_minutes);
  return {
    distance_km: distance,
    distance_miles: Math.round(distance * 0.621371 * 100) / 100,
    duration_minutes: duration,
    prediction_source: "Global planning baseline",
    prediction_note: "Directional time-of-week planning estimate. This is not locally trained or live demand data.",
    fare_currency: "USD equivalent",
    predicted_hourly_demand: demand,
    demand_tier: tier,
    surge_multiplier: surge,
    probability_finding_car_pct: availability,
    availability_status: `${tier} demand · simulated driver availability`,
    estimated_wait_time_minutes: drivers[0]?.eta_minutes || 5,
    tier_probabilities: { Low: tier === "Low" ? 70 : 10, Medium: tier === "Medium" ? 70 : 10, High: tier === "High" ? 70 : 10, "Very High": tier === "Very High" ? 70 : 10 },
    fares: { "Standard (UberX/Mini)": roundedFare, "Comfort (Prime/Sedan)": Math.round(roundedFare * 1.28 * 100) / 100, "XL / SUV (6-Seater)": Math.round(roundedFare * 1.65 * 100) / 100 },
    nearby_drivers: drivers,
  };
}

"""
ETL Pipeline - Ride Demand Prediction System
=============================================
Handles ALL file formats robustly:
  - uber-raw-data-*.csv     → auto-detects Date/Time, Lat, Lon columns
  - other-Lyft_*.csv        → time_of_trip, start_lat, start_lng
  - other-*address*.csv     → DATE, TIME, ADDRESS (no lat/lon)
  - Uber-Jan-Feb-FOIL.csv   → aggregate trips
  - other-FHV-services*.csv → aggregate summary
"""

import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

RAW_DIR       = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)


def _find_col(df, candidates):
    """Return first matching column name from candidates list, or None."""
    for c in candidates:
        if c in df.columns:
            return c
    # fuzzy fallback
    for keyword in candidates:
        kl = keyword.lower()
        for col in df.columns:
            if kl in col.lower():
                return col
    return None


def load_spatial_file(filepath: str, source_label: str) -> pd.DataFrame:
    """
    Load any spatial CSV (Uber raw or Lyft).
    Auto-detects datetime, lat, lon columns regardless of naming.
    """
    df = pd.read_csv(filepath, encoding="latin1", low_memory=False)
    # Clean column names — strip quotes and whitespace
    df.columns = [c.strip().strip('"').strip() for c in df.columns]

    # ── Find datetime column ─────────────────────────────────────
    dt_col = _find_col(df, [
        "Date/Time", "date/time", "datetime", "pickup_datetime",
        "Pickup_date", "Trip_Pickup_DateTime", "tpep_pickup_datetime",
        "time_of_trip", "Dispatching_base_num",
    ])
    # If still None, take first column that has '/' or '-' in first value
    if dt_col is None:
        for col in df.columns:
            try:
                sample = str(df[col].dropna().iloc[0])
                if "/" in sample or "-" in sample:
                    dt_col = col
                    break
            except Exception:
                continue

    if dt_col is None:
        print(f"  [WARN] Cannot find datetime in {source_label}. Columns: {list(df.columns)}")
        return pd.DataFrame(columns=["pickup_datetime","Lat","Lon","source"])

    # ── Find Lat column ──────────────────────────────────────────
    lat_col = _find_col(df, [
        "Lat", "lat", "LAT", "start_lat", "pickup_latitude",
        "Pickup_latitude", "Start_Lat",
    ])

    # ── Find Lon column ──────────────────────────────────────────
    lon_col = _find_col(df, [
        "Lon", "lon", "LON", "Lng", "lng", "start_lng",
        "pickup_longitude", "Pickup_longitude", "Start_Lng",
    ])

    if lat_col is None or lon_col is None:
        print(f"  [WARN] No Lat/Lon in {source_label}. Columns: {list(df.columns)}")
        return pd.DataFrame(columns=["pickup_datetime","Lat","Lon","source"])

    df = df.rename(columns={dt_col: "pickup_datetime", lat_col: "Lat", lon_col: "Lon"})
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    df["Lat"] = pd.to_numeric(df["Lat"], errors="coerce")
    df["Lon"] = pd.to_numeric(df["Lon"], errors="coerce")
    df = df.dropna(subset=["pickup_datetime", "Lat", "Lon"])
    # Filter to NYC bounding box
    df = df[(df["Lat"].between(40.4, 41.0)) & (df["Lon"].between(-74.3, -73.6))]
    df["source"] = source_label
    return df[["pickup_datetime", "Lat", "Lon", "source"]]


def load_address_only(filepath: str, source_label: str) -> pd.DataFrame:
    """Load files with DATE + TIME but no lat/lon."""
    df = pd.read_csv(filepath, encoding="latin1", low_memory=False)
    df.columns = [c.strip().strip('"').strip() for c in df.columns]

    date_col = _find_col(df, ["Date", "DATE", "date", "Date/Time"])
    time_col = _find_col(df, ["Time", "TIME", "time"])

    if date_col and time_col:
        df["pickup_datetime"] = pd.to_datetime(
            df[date_col].astype(str) + " " + df[time_col].astype(str),
            errors="coerce"
        )
    elif date_col:
        df["pickup_datetime"] = pd.to_datetime(df[date_col], errors="coerce")
    else:
        return pd.DataFrame(columns=["pickup_datetime", "source"])

    df = df.dropna(subset=["pickup_datetime"])
    df["source"] = source_label
    return df[["pickup_datetime", "source"]]


def load_foil(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, encoding="latin1")
    df.columns = [c.strip() for c in df.columns]
    date_col = _find_col(df, ["date", "Date", "DATE"])
    trips_col = _find_col(df, ["trips", "Trips", "TRIPS"])
    if date_col:
        df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    if trips_col:
        df["trips"] = pd.to_numeric(df[trips_col], errors="coerce")
    df["source"] = "Uber-FOIL"
    cols = [c for c in ["date", "trips", "source"] if c in df.columns]
    return df[cols].dropna(subset=["date"])


def load_fhv(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, encoding="latin1")
    df.columns = [c.strip() for c in df.columns]
    date_col  = _find_col(df, ["Pick Up Date", "date", "Date"])
    trips_col = _find_col(df, ["Number of Trips", "trips", "Trips"])
    if date_col:
        df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    if trips_col:
        df["trips"] = pd.to_numeric(
            df[trips_col].astype(str).str.replace(",","").str.strip(), errors="coerce"
        )
    df["source"] = "FHV"
    cols = [c for c in ["date", "trips", "source"] if c in df.columns]
    return df[cols].dropna(subset=["date"])


def run_etl(raw_dir: str = RAW_DIR, processed_dir: str = PROCESSED_DIR):
    print("=" * 60)
    print("  RIDE DEMAND PREDICTION — ETL PIPELINE")
    print("=" * 60)

    os.makedirs(processed_dir, exist_ok=True)

    # ── All spatial files (Uber + Lyft) ─────────────────────────
    spatial_files = [
        ("uber-raw-data-apr14.csv",      "Uber-Apr14"),
        ("uber-raw-data-may14.csv",      "Uber-May14"),
        ("uber-raw-data-jun14.csv",      "Uber-Jun14"),
        ("uber-raw-data-jul14.csv",      "Uber-Jul14"),
        ("uber-raw-data-aug14.csv",      "Uber-Aug14"),
        ("uber-raw-data-sep14.csv",      "Uber-Sep14"),
        ("uber-raw-data-janjune-15.csv", "Uber-JanJun15"),
        ("other-Lyft_B02510.csv",        "Lyft"),
    ]

    spatial_frames = []
    for fname, label in spatial_files:
        fp = os.path.join(raw_dir, fname)
        if os.path.exists(fp):
            try:
                df = load_spatial_file(fp, label)
                if len(df) > 0:
                    spatial_frames.append(df)
                    print(f"  [OK] {fname:42s} → {len(df):>10,} rows")
                else:
                    print(f"  [WARN] {fname} loaded but 0 valid spatial rows")
            except Exception as e:
                print(f"  [ERROR] {fname}: {e}")
        else:
            print(f"  [SKIP] {fname} not found")

    # ── Address-only files ───────────────────────────────────────
    address_files = [
        ("other-American_B01362.csv",   "American"),
        ("other-Carmel_B00256.csv",     "Carmel"),
        ("other-Dial7_B00887.csv",      "Dial7"),
        ("other-Diplo_B01196.csv",      "Diplo"),
        ("other-Federal_02216.csv",     "Federal"),
        ("other-Firstclass_B01536.csv", "Firstclass"),
        ("other-Highclass_B01717.csv",  "Highclass"),
        ("other-Prestige_B01338.csv",   "Prestige"),
        ("other-Skyline_B00111.csv",    "Skyline"),
    ]
    addr_frames = []
    for fname, label in address_files:
        fp = os.path.join(raw_dir, fname)
        if os.path.exists(fp):
            try:
                df = load_address_only(fp, label)
                if len(df) > 0:
                    addr_frames.append(df)
                    print(f"  [OK] {fname:42s} → {len(df):>10,} rows (no lat/lon)")
            except Exception as e:
                print(f"  [ERROR] {fname}: {e}")

    # ── Aggregate files ──────────────────────────────────────────
    foil_fp = os.path.join(raw_dir, "Uber-Jan-Feb-FOIL.csv")
    if os.path.exists(foil_fp):
        try:
            foil_df = load_foil(foil_fp)
            foil_df.to_csv(os.path.join(processed_dir, "foil_aggregate.csv"), index=False)
            print(f"  [OK] {'Uber-Jan-Feb-FOIL.csv':42s} → {len(foil_df):>10,} rows (aggregate)")
        except Exception as e:
            print(f"  [ERROR] FOIL: {e}")

    fhv_fp = os.path.join(raw_dir, "other-FHV-services_jan-aug-2015.csv")
    if os.path.exists(fhv_fp):
        try:
            fhv_df = load_fhv(fhv_fp)
            fhv_df.to_csv(os.path.join(processed_dir, "fhv_aggregate.csv"), index=False)
            print(f"  [OK] {'other-FHV-services_jan-aug-2015.csv':42s} → {len(fhv_df):>10,} rows (aggregate)")
        except Exception as e:
            print(f"  [ERROR] FHV: {e}")

    # ── Build ML dataset from spatial data ───────────────────────
    if not spatial_frames:
        raise RuntimeError("No spatial files loaded. Cannot build ML dataset.")

    df_all = pd.concat(spatial_frames, ignore_index=True)
    print(f"\n  Combined spatial records : {len(df_all):,}")

    # Feature engineering
    df_all["hour"]        = df_all["pickup_datetime"].dt.hour
    df_all["day_of_week"] = df_all["pickup_datetime"].dt.dayofweek
    df_all["month"]       = df_all["pickup_datetime"].dt.month
    df_all["is_weekend"]  = df_all["day_of_week"].isin([5, 6]).astype(int)
    df_all["hour_bin"]    = df_all["hour"] // 3

    # Location grid (~1.1km cells)
    df_all["location"] = (
        df_all["Lat"].round(2).astype(str) + "_" +
        df_all["Lon"].round(2).astype(str)
    )

    # Aggregate demand
    df_grouped = (
        df_all
        .groupby(["location", "hour", "day_of_week", "month", "is_weekend", "hour_bin"])
        .size()
        .reset_index(name="demand")
    )
    print(f"  Aggregated demand rows   : {len(df_grouped):,}")

    # Save outputs
    df_grouped.to_csv(os.path.join(processed_dir, "processed.csv"), index=False)
    print(f"  [SAVED] processed.csv → {len(df_grouped):,} rows")

    df_all[["pickup_datetime","Lat","Lon","hour","day_of_week",
            "month","is_weekend","hour_bin","location","source"]]\
        .to_csv(os.path.join(processed_dir, "spatial_raw.csv"), index=False)
    print(f"  [SAVED] spatial_raw.csv  → {len(df_all):,} rows")

    if addr_frames:
        df_addr = pd.concat(addr_frames, ignore_index=True)
        df_addr["hour"]        = df_addr["pickup_datetime"].dt.hour
        df_addr["day_of_week"] = df_addr["pickup_datetime"].dt.dayofweek
        df_addr["month"]       = df_addr["pickup_datetime"].dt.month
        agg = df_addr.groupby(["source","hour","day_of_week","month"])\
                     .size().reset_index(name="trips")
        agg.to_csv(os.path.join(processed_dir, "other_carriers_demand.csv"), index=False)
        print(f"  [SAVED] other_carriers_demand.csv")

    print("\n  ETL COMPLETE ✓")
    print("=" * 60)
    return df_grouped


if __name__ == "__main__":
    run_etl()

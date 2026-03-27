"""
Snowflake Integration — Ride Demand Prediction System
======================================================
Handles:
  1. Connecting to Snowflake
  2. Creating the database + schema + table
  3. Uploading processed.csv
  4. Running SQL analytics queries
  5. Fetching data back into Pandas for model training

Usage:
  python snowflake/connection.py

Or import in your scripts:
  from snowflake.connection import SnowflakeClient
"""

import os
import json
import pandas as pd
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas


class SnowflakeClient:
    """
    Thin wrapper around snowflake.connector for this project.
    Credentials loaded from environment variables or config.json.
    """

    def __init__(self, config: dict = None):
        """
        config keys: account, user, password, warehouse, database, schema
        If config is None, reads from environment variables:
          SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD,
          SNOWFLAKE_WAREHOUSE, SNOWFLAKE_DATABASE, SNOWFLAKE_SCHEMA
        """
        if config is None:
            config = self._config_from_env()
        self.config   = config
        self.conn     = None
        self.cursor   = None

    def _config_from_env(self) -> dict:
        """Read config from environment variables."""
        return {
            "account":   os.environ.get("SNOWFLAKE_ACCOUNT",   "your-account"),
            "user":      os.environ.get("SNOWFLAKE_USER",      "your-user"),
            "password":  os.environ.get("SNOWFLAKE_PASSWORD",  "your-password"),
            "warehouse": os.environ.get("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
            "database":  os.environ.get("SNOWFLAKE_DATABASE",  "UBER_PROJECT"),
            "schema":    os.environ.get("SNOWFLAKE_SCHEMA",    "PUBLIC"),
        }

    def connect(self):
        """Open connection and cursor."""
        self.conn   = snowflake.connector.connect(**self.config)
        self.cursor = self.conn.cursor()
        print(f"  [OK] Connected to Snowflake → {self.config['account']}")
        return self

    def close(self):
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        print("  [OK] Connection closed")

    def execute(self, sql: str, fetch: bool = False):
        """Execute a SQL statement; optionally return results."""
        self.cursor.execute(sql)
        if fetch:
            return self.cursor.fetchall()
        return None

    def query_to_df(self, sql: str) -> pd.DataFrame:
        """Execute a SELECT and return a DataFrame."""
        self.cursor.execute(sql)
        cols = [d[0] for d in self.cursor.description]
        rows = self.cursor.fetchall()
        return pd.DataFrame(rows, columns=cols)

    # ── Setup ────────────────────────────────────────────────────
    def setup_database(self, database: str = "UBER_PROJECT", schema: str = "PUBLIC"):
        """Create database + schema if not exist."""
        self.execute(f"CREATE DATABASE IF NOT EXISTS {database}")
        self.execute(f"USE DATABASE {database}")
        self.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
        self.execute(f"USE SCHEMA {schema}")
        print(f"  [OK] Database: {database}.{schema}")

    def create_ride_data_table(self):
        """Create RIDE_DATA table (drop if exists for clean load)."""
        self.execute("DROP TABLE IF EXISTS RIDE_DATA")
        self.execute("""
            CREATE TABLE RIDE_DATA (
                LOCATION     VARCHAR(50),
                HOUR         INTEGER,
                DAY_OF_WEEK  INTEGER,
                MONTH        INTEGER,
                IS_WEEKEND   INTEGER,
                HOUR_BIN     INTEGER,
                DEMAND       INTEGER
            )
        """)
        print("  [OK] Table RIDE_DATA created")

    def upload_processed_csv(self, csv_path: str):
        """Upload processed.csv to RIDE_DATA table using write_pandas."""
        df = pd.read_csv(csv_path)
        df.columns = [c.upper() for c in df.columns]
        success, nchunks, nrows, _ = write_pandas(
            self.conn, df, "RIDE_DATA",
            database=self.config["database"],
            schema=self.config["schema"],
        )
        if success:
            print(f"  [OK] Uploaded {nrows:,} rows to RIDE_DATA in {nchunks} chunks")
        else:
            print("  [FAIL] Upload failed")
        return nrows

    # ── Analytics queries ────────────────────────────────────────
    def peak_hours_query(self) -> pd.DataFrame:
        sql = """
            SELECT   HOUR,
                     SUM(DEMAND)  AS TOTAL_DEMAND,
                     AVG(DEMAND)  AS AVG_DEMAND,
                     COUNT(*)     AS LOCATION_HOURS
            FROM     RIDE_DATA
            GROUP BY HOUR
            ORDER BY TOTAL_DEMAND DESC
        """
        df = self.query_to_df(sql)
        print(f"  Peak hours query → {len(df)} rows")
        return df

    def top_locations_query(self, limit: int = 20) -> pd.DataFrame:
        sql = f"""
            SELECT   LOCATION,
                     SUM(DEMAND) AS TOTAL_DEMAND
            FROM     RIDE_DATA
            GROUP BY LOCATION
            ORDER BY TOTAL_DEMAND DESC
            LIMIT    {limit}
        """
        df = self.query_to_df(sql)
        print(f"  Top locations query → {len(df)} rows")
        return df

    def weekday_vs_weekend(self) -> pd.DataFrame:
        sql = """
            SELECT   IS_WEEKEND,
                     SUM(DEMAND)  AS TOTAL_DEMAND,
                     AVG(DEMAND)  AS AVG_DEMAND
            FROM     RIDE_DATA
            GROUP BY IS_WEEKEND
            ORDER BY IS_WEEKEND
        """
        return self.query_to_df(sql)

    def demand_by_month(self) -> pd.DataFrame:
        sql = """
            SELECT   MONTH,
                     SUM(DEMAND) AS TOTAL_DEMAND
            FROM     RIDE_DATA
            GROUP BY MONTH
            ORDER BY MONTH
        """
        return self.query_to_df(sql)

    def demand_heatmap(self) -> pd.DataFrame:
        """Hour × Day-of-week demand matrix."""
        sql = """
            SELECT   HOUR,
                     DAY_OF_WEEK,
                     SUM(DEMAND) AS TOTAL_DEMAND
            FROM     RIDE_DATA
            GROUP BY HOUR, DAY_OF_WEEK
            ORDER BY DAY_OF_WEEK, HOUR
        """
        return self.query_to_df(sql)


# ── CLI runner ───────────────────────────────────────────────────
def run_snowflake_setup():
    """
    Full setup + upload + analytics demo.
    Fill in your credentials below or set environment variables.
    """
    CONFIG = {
        "account":   os.environ.get("SNOWFLAKE_ACCOUNT",   "YOUR_ACCOUNT"),
        "user":      os.environ.get("SNOWFLAKE_USER",       "YOUR_USER"),
        "password":  os.environ.get("SNOWFLAKE_PASSWORD",   "YOUR_PASSWORD"),
        "warehouse": os.environ.get("SNOWFLAKE_WAREHOUSE",  "COMPUTE_WH"),
        "database":  "UBER_PROJECT",
        "schema":    "PUBLIC",
    }

    print("=" * 60)
    print("  SNOWFLAKE SETUP + UPLOAD")
    print("=" * 60)

    csv_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "processed", "processed.csv"
    )
    if not os.path.exists(csv_path):
        print("  ERROR: processed.csv not found. Run etl/etl.py first.")
        return

    client = SnowflakeClient(CONFIG)
    try:
        client.connect()
        client.setup_database()
        client.create_ride_data_table()
        client.upload_processed_csv(csv_path)

        print("\n  ── Analytics Queries ──────────────────────────")
        ph = client.peak_hours_query()
        print(f"\n  Peak hours (top 5):\n{ph.head()}\n")

        tl = client.top_locations_query(10)
        print(f"  Top 10 locations:\n{tl.head()}\n")

        ww = client.weekday_vs_weekend()
        print(f"  Weekday vs weekend:\n{ww}\n")

        print("  SNOWFLAKE SETUP COMPLETE ✓")
    except Exception as e:
        print(f"  ERROR: {e}")
        print("  Make sure your credentials are correct in CONFIG above.")
    finally:
        client.close()


if __name__ == "__main__":
    run_snowflake_setup()

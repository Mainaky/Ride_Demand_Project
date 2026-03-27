"""
EDA Analysis — Ride Demand Prediction System
Run: python notebooks/eda_analysis.py
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # no display needed — saves to PNG
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid", palette="muted")

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
OUT = PROCESSED_DIR

df = pd.read_csv(os.path.join(PROCESSED_DIR, "processed.csv"))
print(f"Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
print(df.dtypes)
print("\nDescriptive stats:")
print(df["demand"].describe())
print(f"Skewness : {df['demand'].skew():.4f}")
print(f"Kurtosis : {df['demand'].kurtosis():.4f}")
print(f"Unique locations: {df['location'].nunique():,}")

# 1. Demand distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
sns.histplot(df["demand"], bins=60, ax=axes[0], color="#1D9E75")
axes[0].set_title("Demand distribution")
axes[0].set_xlabel("Demand (rides/location/hour)")
sns.boxplot(x=df["demand"], ax=axes[1], color="#9FE1CB")
axes[1].set_title("Demand boxplot")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "plot_demand_distribution.png"), dpi=150)
plt.close()
print("Saved: plot_demand_distribution.png")

# 2. Hourly demand
hourly = df.groupby("hour")["demand"].sum().reset_index()
plt.figure(figsize=(12, 4))
sns.lineplot(data=hourly, x="hour", y="demand", marker="o",
             color="#1D9E75", linewidth=2.5)
plt.title("Total demand by hour of day")
plt.xlabel("Hour (0-23)")
plt.ylabel("Total demand")
plt.xticks(range(24))
plt.tight_layout()
plt.savefig(os.path.join(OUT, "plot_hourly_demand.png"), dpi=150)
plt.close()
print(f"Peak hour: {hourly.loc[hourly['demand'].idxmax(), 'hour']}:00")

# 3. Day of week
day_labels = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
dow = df.groupby("day_of_week")["demand"].sum().reset_index()
dow["day"] = dow["day_of_week"].map(dict(enumerate(day_labels)))
plt.figure(figsize=(10, 4))
sns.barplot(data=dow, x="day", y="demand", palette="viridis")
plt.title("Total demand by day of week")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "plot_dow_demand.png"), dpi=150)
plt.close()

# 4. Correlation matrix
numeric_cols = ["hour","day_of_week","month","is_weekend","hour_bin","demand"]
corr = df[numeric_cols].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            square=True, linewidths=0.5)
plt.title("Feature correlation matrix")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "plot_correlation.png"), dpi=150)
plt.close()
print("\nCorrelation with demand:")
print(corr["demand"].drop("demand").sort_values(ascending=False))

# 5. Weekday vs weekend
ww = df.groupby(["is_weekend","hour"])["demand"].mean().reset_index()
ww["type"] = ww["is_weekend"].map({0:"Weekday",1:"Weekend"})
plt.figure(figsize=(12, 4))
sns.lineplot(data=ww, x="hour", y="demand", hue="type",
             palette={"Weekday":"#1D9E75","Weekend":"#D85A30"}, linewidth=2)
plt.title("Average demand: weekday vs weekend by hour")
plt.xticks(range(24))
plt.tight_layout()
plt.savefig(os.path.join(OUT, "plot_weekday_weekend.png"), dpi=150)
plt.close()

# 6. Monthly demand
if "month" in df.columns:
    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                   7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    monthly = df.groupby("month")["demand"].sum().reset_index()
    monthly["month_name"] = monthly["month"].map(month_names)
    plt.figure(figsize=(10, 4))
    sns.barplot(data=monthly, x="month_name", y="demand", palette="Blues_d")
    plt.title("Total demand by month")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "plot_monthly.png"), dpi=150)
    plt.close()

# 7. Top 20 locations
top20 = df.groupby("location")["demand"].sum().nlargest(20).reset_index()
plt.figure(figsize=(12, 6))
sns.barplot(data=top20, x="demand", y="location", palette="Greens_d")
plt.title("Top 20 highest-demand locations")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "plot_top_locations.png"), dpi=150)
plt.close()

print("\nEDA COMPLETE ✓ — all plots saved to data/processed/")

"""
Traffic vs. snow plots, one figure per mountain.

Inputs (same folder as this script, or edit the paths below):
  - Traffic_2023-2025_combined.csv     (hourly counts, 2 directions per day)
  - ski_snow_depth_in_2023_2025.csv    (daily snow depth in inches, one column per resort)

Note: the snow file contains snow DEPTH (base), not snowfall. This script plots
traffic against depth and also against estimated daily snowfall, which is the
day-over-day INCREASE in depth (settling/melt is ignored, so it is approximate).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

TRAFFIC_CSV = "Traffic_2023-2025_combined.csv"
SNOW_CSV = "ski_snow_depth_in_2023_2025.csv"
SAVE_FIGS = True          # save one PNG per mountain
SHOW_FIGS = True

# ---------- 1. Load traffic and build total daily vehicle counts ----------
traffic = pd.read_csv(TRAFFIC_CSV)
hour_cols = [f"HOUR{i}" for i in range(24)]
traffic["date"] = pd.to_datetime(traffic["FormattedDate"], format="%m/%d/%Y")
traffic["daily_total"] = traffic[hour_cols].sum(axis=1)

# Both directions (P and S) are summed, so a day only counts if both are present
dir_counts = traffic.groupby(["RESORT", "date"])["COUNTDIR"].nunique()
complete_days = dir_counts[dir_counts == 2].index
daily_traffic = (
    traffic.groupby(["RESORT", "date"])["daily_total"].sum()
    .loc[complete_days]
    .reset_index()
)

# ---------- 2. Load snow depth and reshape to long format ----------
snow = pd.read_csv(SNOW_CSV, parse_dates=["date"])
snow_long = snow.melt(id_vars="date", var_name="col", value_name="snow_depth_in")
snow_long["RESORT"] = (
    snow_long["col"].str.replace("_snow_depth_in", "", regex=False).str.replace("_", " ")
)
snow_long = snow_long.drop(columns="col").sort_values(["RESORT", "date"])

# Estimated daily snowfall = positive change in depth
snow_long["snowfall_in"] = (
    snow_long.groupby("RESORT")["snow_depth_in"].diff().clip(lower=0)
)

# ---------- 3. Merge ----------
df = daily_traffic.merge(snow_long, on=["RESORT", "date"], how="inner")
df = df.dropna(subset=["snow_depth_in"])


# ---------- 4. Plot helper ----------
def scatter_with_fit(ax, x, y, xlabel, color):
    ax.scatter(x, y, s=12, alpha=0.4, color=color)
    mask = x.notna() & y.notna()
    if mask.sum() > 2 and x[mask].nunique() > 1:
        m, b = np.polyfit(x[mask], y[mask], 1)
        xs = np.linspace(x[mask].min(), x[mask].max(), 100)
        ax.plot(xs, m * xs + b, color="black", lw=1.5)
        r = np.corrcoef(x[mask], y[mask])[0, 1]
        ax.set_title(f"r = {r:.2f}  (n = {mask.sum()})", fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Daily vehicles (both directions)")
    ax.grid(alpha=0.3)


# ---------- 5. One figure per mountain ----------
colors = {"Loveland": "tab:blue", "Breckenridge": "tab:orange",
          "Vail": "tab:green", "Winter Park": "tab:red"}

for resort, g in df.groupby("RESORT"):
    g = g.sort_values("date")
    c = colors.get(resort, "tab:gray")

    fig = plt.figure(figsize=(15, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])
    ax_depth = fig.add_subplot(gs[0, 0])
    ax_fall = fig.add_subplot(gs[0, 1])
    ax_ts = fig.add_subplot(gs[1, :])

    scatter_with_fit(ax_depth, g["snow_depth_in"], g["daily_total"],
                     "Snow depth (in)", c)
    scatter_with_fit(ax_fall, g["snowfall_in"], g["daily_total"],
                     "Estimated daily snowfall (in)", c)

    # Time series: traffic (left axis) and snow depth (right axis)
    ax_ts.plot(g["date"], g["daily_total"], color=c, lw=0.8, label="Traffic")
    ax_ts.set_ylabel("Daily vehicles", color=c)
    ax_ts2 = ax_ts.twinx()
    ax_ts2.fill_between(g["date"], g["snow_depth_in"], color="lightsteelblue",
                        alpha=0.5, label="Snow depth")
    ax_ts2.set_ylabel("Snow depth (in)")
    ax_ts.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax_ts.grid(alpha=0.3)
    ax_ts.set_title("Traffic and snow depth over time", fontsize=10)

    fig.suptitle(f"{resort}: Traffic vs. Snow", fontsize=15, fontweight="bold")

    if SAVE_FIGS:
        fig.savefig(f"traffic_vs_snow_{resort.replace(' ', '_')}.png", dpi=150)

if SHOW_FIGS:
    plt.show()

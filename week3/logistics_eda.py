"""
Week 3: Exploratory data analysis and visualization of a hypothetical logistics dataset.
15,000 shipments from 3 distribution centres (DCs) over 12 months of 2025.
Relationships (mode speed, carrier quality, DC bottleneck, peak surcharge) are built into
the simulation on purpose, so the analysis can be checked against a known structure.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

sns.set_theme(style="whitegrid", context="notebook", palette="deep")
rng = np.random.default_rng(2025)
R = {}                      # results collected for the report

# ----------------------------------------------------------------- 1. SIMULATE
def simulate(n=15000):
    dates = pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D")
    dc = rng.choice(["DC-North", "DC-South", "DC-West"], n, p=[.35, .30, .35])
    region = rng.choice(["North", "South", "East", "West", "Central"], n, p=[.22, .24, .18, .20, .16])
    mode = rng.choice(["Road", "Rail", "Air"], n, p=[.62, .28, .10])
    carrier = rng.choice(["Carrier A", "Carrier B", "Carrier C", "Carrier D"], n, p=[.32, .28, .22, .18])
    dist_base = {"North": 650, "South": 900, "East": 1100, "West": 750, "Central": 500}
    distance = np.clip(pd.Series(region).map(dist_base).values * rng.lognormal(0, .30, n), 80, 3500)
    weight = np.clip(rng.lognormal(5.0, 1.0, n), 5, 8000)                     # kg, right-skewed
    volume = np.round(weight / rng.uniform(150, 350, n), 2)                    # m3
    df = pd.DataFrame(dict(ship_date=dates, dc=dc, region=region, mode=mode, carrier=carrier,
                           distance_km=distance.round(0), weight_kg=weight.round(1), volume_cbm=volume))
    df["month"] = df.ship_date.dt.month
    df["weekday"] = df.ship_date.dt.day_name()
    df["peak_season"] = df.month.isin([10, 11, 12]).astype(int)
    df["weather_event"] = (rng.random(n) < np.where(df.month.isin([6, 7, 8]), .12, .04)).astype(int)
    df["fuel_index"] = (100 + 6 * np.sin(2 * np.pi * (df.month - 2) / 12) + rng.normal(0, 1.5, n)).round(1)

    # dispatch (warehouse) delay in hours: DC-West is the built-in bottleneck, Mondays and peak are worse
    dc_eff = df.dc.map({"DC-North": 5, "DC-South": 6, "DC-West": 11}).values
    mon = np.where(df.weekday == "Monday", 1.35, 1.0)
    pk = np.where(df.peak_season == 1, 1.30, 1.0)
    df["dispatch_delay_h"] = np.clip(rng.gamma(3, dc_eff / 3, n) * mon * pk, 0.5, 96).round(1)

    # transit hours: distance / speed + noise; carrier C weaker, weather and peak add delay
    speed = df["mode"].map({"Road": 45, "Rail": 38, "Air": 450}).values          # km/h effective door to door
    car_pen = df.carrier.map({"Carrier A": 1.0, "Carrier B": 1.05, "Carrier C": 1.22, "Carrier D": 1.0}).values
    south_c = np.where((df.carrier == "Carrier C") & (df.region == "South"), 1.15, 1.0)
    transit = df.distance_km / speed * car_pen * south_c * rng.lognormal(0, .12, n)
    transit += df.weather_event * rng.uniform(6, 30, n) + df.peak_season * rng.uniform(0, 8, n)
    df["transit_h"] = transit.round(1)
    df["total_hours"] = (df.dispatch_delay_h + df.transit_h).round(1)
    df["delivery_days"] = np.ceil(df.total_hours / 24).astype(int)
    df["promised_days"] = np.ceil(df.distance_km / (speed * 24) * 1.25 + 0.6 + (df["mode"] == "Road") * 0.4).astype(int).clip(lower=1)
    df["on_time"] = (df.delivery_days <= df.promised_days).astype(int)

    # cost (USD): mode rate x distance x weight^0.85, carrier markup, fuel, peak surcharge
    rate = df["mode"].map({"Road": 0.0045, "Rail": 0.0030, "Air": 0.0300}).values
    mark = df.carrier.map({"Carrier A": 1.0, "Carrier B": .92, "Carrier C": .88, "Carrier D": 1.12}).values
    cost = (40 + rate * df.distance_km * df.weight_kg ** 0.85) * mark * (df.fuel_index / 100)
    cost *= np.where(df.peak_season == 1, 1.12, 1.0) * rng.lognormal(0, .08, n)
    df["cost_usd"] = cost.round(2)
    df["cost_per_kg"] = (df.cost_usd / df.weight_kg).round(3)
    df["damaged"] = (rng.random(n) < (0.012 + 0.010 * (df["mode"] == "Road") + 0.006 * (df.carrier == "Carrier C"))).astype(int)
    return df

df = simulate()
df.to_csv("data/shipments.csv", index=False)
R["n"] = len(df); R["cols"] = df.shape[1]

# ----------------------------------------------------------------- 2. EDA
num = ["distance_km", "weight_kg", "volume_cbm", "dispatch_delay_h", "transit_h", "total_hours", "delivery_days", "cost_usd", "cost_per_kg"]
desc = df[num].describe().T
desc["median"] = df[num].median(); desc["skew"] = df[num].skew(); desc["cv"] = desc["std"] / desc["mean"]
R["describe"] = desc[["mean", "median", "std", "min", "max", "skew"]].round(2).to_dict("index")
R["otif_overall"] = round(df.on_time.mean() * 100, 1)
R["damage_rate"] = round(df.damaged.mean() * 100, 2)
R["total_cost"] = round(df.cost_usd.sum())
R["mode_share"] = (df["mode"].value_counts(normalize=True) * 100).round(1).to_dict()
R["mode_cost_share"] = (df.groupby("mode").cost_usd.sum() / df.cost_usd.sum() * 100).round(1).to_dict()

grp = lambda c: df.groupby(c).agg(shipments=("on_time", "size"), on_time_pct=("on_time", lambda s: s.mean() * 100),
        dispatch_h=("dispatch_delay_h", "mean"), transit_h=("transit_h", "mean"), cost_per_kg=("cost_per_kg", "median"),
        damage_pct=("damaged", lambda s: s.mean() * 100)).round(2)
R["by_carrier"] = grp("carrier").to_dict("index")
R["by_dc"] = grp("dc").to_dict("index")
R["by_mode"] = grp("mode").to_dict("index")
R["by_region"] = grp("region").to_dict("index")
R["by_peak"] = grp("peak_season").to_dict("index")
R["by_weekday"] = df.groupby("weekday").dispatch_delay_h.mean().round(2).to_dict()
R["weather_effect_h"] = round(df[df.weather_event == 1].transit_h.mean() - df[df.weather_event == 0].transit_h.mean(), 1)
R["weather_share_pct"] = round(df.weather_event.mean() * 100, 1)
R["share_dispatch_of_total"] = round(df.dispatch_delay_h.sum() / df.total_hours.sum() * 100, 1)
road = df[df["mode"] == "Road"]
R["share_dispatch_road"] = round(road.dispatch_delay_h.sum() / road.total_hours.sum() * 100, 1)

corr = df[["distance_km", "weight_kg", "dispatch_delay_h", "transit_h", "total_hours", "cost_usd", "fuel_index", "on_time"]].corr(method="spearman")
R["corr_cost_weight"] = round(corr.loc["cost_usd", "weight_kg"], 2)
R["corr_cost_distance"] = round(corr.loc["cost_usd", "distance_km"], 2)
R["corr_ontime_dispatch"] = round(corr.loc["on_time", "dispatch_delay_h"], 2)
R["corr_ontime_transit"] = round(corr.loc["on_time", "transit_h"], 2)

# statistical tests
groups = [g.dispatch_delay_h.values for _, g in df.groupby("dc")]
H, p = stats.kruskal(*groups); R["kruskal_dc"] = {"H": round(H, 1), "p": float(p)}
ct = pd.crosstab(df.carrier, df.on_time); chi, p2, _, _ = stats.chi2_contingency(ct); R["chi2_carrier"] = {"chi2": round(chi, 1), "p": float(p2)}

# cost driver model: log-cost on standardized drivers
X = pd.get_dummies(df[["distance_km", "weight_kg", "fuel_index", "peak_season", "mode", "carrier"]], drop_first=True, dtype=float)
X["weight_kg"] = np.log(X["weight_kg"]); X["distance_km"] = np.log(X["distance_km"])
Xs = StandardScaler().fit_transform(X); y = np.log(df.cost_usd)
lr = LinearRegression().fit(Xs, y)
R["cost_model_r2"] = round(lr.score(Xs, y), 3)
coef = pd.Series(lr.coef_, index=X.columns).sort_values(key=abs, ascending=False)
R["cost_drivers"] = coef.round(3).to_dict()  # (mode dummies are relative to Air)

# monthly trend
mo = df.groupby("month").agg(volume=("weight_kg", "sum"), shipments=("on_time", "size"), cost_kg=("cost_per_kg", "median"),
                              otif=("on_time", lambda s: s.mean() * 100), dispatch=("dispatch_delay_h", "mean")).round(2)
R["monthly"] = mo.to_dict("index")

# ----------------------------------------------------------------- 3. VISUALS
def save(name): plt.tight_layout(); plt.savefig(f"figures/{name}.png", dpi=150); plt.close()

# F1 distributions: histogram + KDE of delivery time and cost (log) 
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
sns.histplot(df.total_hours, bins=50, kde=True, ax=ax[0], color="#2E75B6"); ax[0].axvline(df.total_hours.median(), c="r", ls="--", lw=1); ax[0].set_title("Total delivery time (hours)")
sns.histplot(df.weight_kg, bins=50, kde=True, ax=ax[1], color="#2E75B6"); ax[1].set_title("Shipment weight (kg), skewed")
sns.histplot(np.log10(df.cost_usd), bins=50, kde=True, ax=ax[2], color="#2E75B6"); ax[2].set_title("Cost per shipment (log10 USD)")
save("f1_distributions")

# F2 boxplot delivery time and cost per kg by mode
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
sns.boxplot(data=df, x="mode", y="total_hours", order=["Road", "Rail", "Air"], ax=ax[0], showfliers=False); ax[0].set_yscale("log"); ax[0].set_title("Delivery time by mode (log scale)")
sns.boxplot(data=df, x="mode", y="cost_per_kg", order=["Road", "Rail", "Air"], ax=ax[1], showfliers=False); ax[1].set_yscale("log"); ax[1].set_title("Cost per kg by mode (log scale)")
save("f2_mode_box")

# F3 correlation heatmap
plt.figure(figsize=(7.2, 5.6))
sns.heatmap(corr, annot=True, fmt=".2f", annot_kws={"size": 8}, cmap="RdBu_r", center=0, vmin=-1, vmax=1, square=True, cbar_kws={"shrink": .8})
plt.xticks(rotation=40, ha="right", fontsize=8); plt.yticks(fontsize=8)
plt.title("Spearman correlation matrix"); save("f3_corr")

# F4 cost vs weight scatter coloured by mode (log-log)
plt.figure(figsize=(7, 4.2))
sns.scatterplot(data=df.sample(4000, random_state=1), x="weight_kg", y="cost_usd", hue="mode", hue_order=["Road", "Rail", "Air"], alpha=.45, s=14)
plt.xscale("log"); plt.yscale("log"); plt.title("Cost vs weight by transport mode (log-log)"); save("f4_cost_scatter")

# F5 monthly trend, dual panel
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
ax[0].bar(mo.index, mo.shipments, color="#9DC3E6", label="Shipments"); ax[0].set_xlabel("Month"); ax[0].set_ylabel("Shipments"); ax[0].set_title("Volume and on-time rate by month")
a2 = ax[0].twinx(); a2.plot(mo.index, mo.otif, "-o", color="#C00000"); a2.set_ylabel("On-time %"); a2.grid(False)
ax[1].plot(mo.index, mo.cost_kg, "-o", color="#1F3A5F"); ax[1].set_title("Median cost per kg by month"); ax[1].set_xlabel("Month")
save("f5_monthly")

# F6 carrier x region on-time heatmap
pv = df.pivot_table(index="carrier", columns="region", values="on_time", aggfunc="mean") * 100
plt.figure(figsize=(7, 3.6)); sns.heatmap(pv, annot=True, fmt=".1f", cmap="RdYlGn", vmin=91, vmax=99, cbar_kws={"label": "On-time %"})
plt.yticks(rotation=0)
plt.title("On-time delivery % by carrier and region"); save("f6_carrier_region")

# F7 bottleneck: dispatch delay by DC x weekday
order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
pv2 = df.pivot_table(index="dc", columns="weekday", values="dispatch_delay_h", aggfunc="mean")[order]
plt.figure(figsize=(8.5, 3.4)); sns.heatmap(pv2, annot=True, fmt=".1f", cmap="YlOrRd", cbar_kws={"label": "Mean hours"})
plt.yticks(rotation=0); plt.xticks(rotation=0, fontsize=8)
plt.title("Warehouse dispatch delay (hours) by DC and weekday"); save("f7_dispatch_heat")

# F8 stacked delay components by DC and mode
comp = df.groupby("dc")[["dispatch_delay_h", "transit_h"]].mean()
comp2 = df[df["mode"] == "Road"].groupby("dc")[["dispatch_delay_h", "transit_h"]].mean()
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
comp.plot(kind="barh", stacked=True, ax=ax[0], color=["#C00000", "#9DC3E6"]); ax[0].set_title("Average hours by DC (all modes)"); ax[0].set_xlabel("Hours"); ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2, fontsize=8, frameon=False)
comp2.plot(kind="barh", stacked=True, ax=ax[1], color=["#C00000", "#9DC3E6"]); ax[1].set_title("Road shipments only"); ax[1].set_xlabel("Hours"); ax[1].legend().remove()
save("f8_components")

# F9 cost drivers (standardized coefficients)
coef.index = [i.replace("mode_Road", "mode_Road (vs Air)").replace("mode_Rail", "mode_Rail (vs Air)") for i in coef.index]
plt.figure(figsize=(7, 4)); coef.sort_values().plot.barh(color=["#C00000" if v < 0 else "#2E75B6" for v in coef.sort_values()])
plt.title(f"Drivers of log(cost): standardized coefficients (R² = {R['cost_model_r2']})"); plt.xlabel("Effect on log cost per 1 SD change"); save("f9_cost_drivers")

# F10 carrier scorecard: on-time vs cost per kg bubble
sc = df.groupby("carrier").agg(on_time=("on_time", lambda s: s.mean() * 100), cost=("cost_per_kg", "median"), n=("on_time", "size"), dmg=("damaged", lambda s: s.mean() * 100))
plt.figure(figsize=(6.4, 4))
plt.scatter(sc.cost, sc.on_time, s=sc.n / 8, alpha=.6, c=sc.dmg, cmap="OrRd", edgecolor="k")
for c, r in sc.iterrows(): plt.annotate(c, (r.cost, r.on_time), xytext=(0, 18), textcoords="offset points", ha="center", fontsize=8)
plt.margins(x=0.15, y=0.12)
plt.colorbar(label="Damage rate %"); plt.xlabel("Median cost per kg (USD)"); plt.ylabel("On-time %"); plt.title("Carrier scorecard (bubble = volume)"); save("f10_carrier_scorecard")


# ----------------------------------------------------------------- 4. WHAT-IF SCENARIOS
def otif_with(dispatch):
    days = np.ceil((dispatch + df.transit_h) / 24).astype(int)
    return float((days <= df.promised_days).mean() * 100)

target = df[df.dc == "DC-North"].dispatch_delay_h.mean()
west = df.dc == "DC-West"
d1 = df.dispatch_delay_h.copy(); d1[west] = d1[west] * target / df[west].dispatch_delay_h.mean()
mon_fix = df.weekday == "Monday"
d2 = df.dispatch_delay_h.copy(); d2[mon_fix] = d2[mon_fix] * df[~mon_fix].dispatch_delay_h.mean() / df[mon_fix].dispatch_delay_h.mean()
d3 = d1.copy(); d3[mon_fix] = d3[mon_fix] * df[~mon_fix].dispatch_delay_h.mean() / df[mon_fix].dispatch_delay_h.mean()
base = otif_with(df.dispatch_delay_h)
R["scenario"] = {"baseline": round(base, 1), "fix_dc_west": round(otif_with(d1), 1),
                 "fix_monday": round(otif_with(d2), 1), "both": round(otif_with(d3), 1)}
R["late_shipments_baseline"] = int((1 - base / 100) * len(df))
R["late_shipments_both"] = int((1 - R["scenario"]["both"] / 100) * len(df))
R["air_dispatch_share"] = round(df[df["mode"] == "Air"].dispatch_delay_h.sum() / df[df["mode"] == "Air"].total_hours.sum() * 100, 1)
R["carrierC_south"] = round(pv.loc["Carrier C", "South"], 1)
R["carrierC_other"] = round(df[(df.carrier == "Carrier C") & (df.region != "South")].on_time.mean() * 100, 1)
R["otif_by_month_min"] = [int(mo.otif.idxmin()), float(mo.otif.min())]
R["otif_by_month_max"] = [int(mo.otif.idxmax()), float(mo.otif.max())]
R["west_monday_dispatch"] = round(pv2.loc["DC-West", "Monday"], 1)
R["north_midweek_dispatch"] = round(pv2.loc["DC-North", ["Tuesday", "Wednesday", "Thursday"]].mean(), 1)
R["heavy_share_top10_weight"] = round(df.nlargest(int(len(df) * .1), "weight_kg").cost_usd.sum() / df.cost_usd.sum() * 100, 1)
R["air_cost_per_kg_x_road"] = round(R["by_mode"]["Air"]["cost_per_kg"] / R["by_mode"]["Road"]["cost_per_kg"], 1)
print("SCENARIO", R["scenario"], R["late_shipments_baseline"], R["late_shipments_both"])
print({k: R[k] for k in ["air_dispatch_share","carrierC_south","carrierC_other","otif_by_month_min","otif_by_month_max","west_monday_dispatch","north_midweek_dispatch","heavy_share_top10_weight","air_cost_per_kg_x_road"]})

json.dump(R, open("results.json", "w"), indent=1, default=str)
print(json.dumps({k: R[k] for k in R if k not in ("monthly", "describe")}, indent=1, default=str))
print(desc[["mean", "median", "std", "skew"]].round(2))

"""
Logistics analytics pipeline (Week 1 prototype)
Scenario: regional distribution network (1 DC -> 30 stores, 40 SKUs)
Runs end-to-end on synthetic data so the approach can be demonstrated
before real data (e.g., Olist, DataCo, M5) is plugged in.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

rng = np.random.default_rng(42)

# ---------------------------------------------------------------- 1. DATA
def make_demand(n_skus=40, days=730):
    dates = pd.date_range("2024-01-01", periods=days, freq="D")
    rows = []
    for sku in range(n_skus):
        base = rng.uniform(15, 120)
        trend = rng.uniform(-0.01, 0.03)
        amp = rng.uniform(0.05, 0.35)
        for i, d in enumerate(dates):
            season = 1 + amp * np.sin(2 * np.pi * d.dayofyear / 365)
            weekly = 1.25 if d.dayofweek >= 5 else 1.0
            promo = int(rng.random() < 0.06)
            mu = base * (1 + trend * i / 30) * season * weekly * (1.4 if promo else 1)
            rows.append((d, f"SKU{sku:02d}", max(0, rng.poisson(mu)), promo,
                         rng.integers(0, 2 if d.dayofweek < 5 else 3)))
    df = pd.DataFrame(rows, columns=["date", "sku", "units", "promo", "holiday_flag"])
    # inject dirty data to show cleaning
    idx = rng.choice(df.index, 200, replace=False)
    df.loc[idx[:100], "units"] = np.nan
    df.loc[idx[100:], "units"] = df.loc[idx[100:], "units"] * 10
    return df

# ---------------------------------------------------------------- 2. CLEAN
def clean(df):
    df = df.drop_duplicates(["date", "sku"]).sort_values(["sku", "date"]).copy()
    df["units"] = df.groupby("sku")["units"].transform(lambda s: s.interpolate(limit_direction="both"))
    q1 = df.groupby("sku")["units"].transform(lambda s: s.quantile(.25))
    q3 = df.groupby("sku")["units"].transform(lambda s: s.quantile(.75))
    cap = q3 + 3 * (q3 - q1)
    df["units"] = np.minimum(df["units"], cap)          # winsorise extreme spikes
    return df

# ---------------------------------------------------------------- 3. FEATURES
def add_features(df):
    g = df.groupby("sku")["units"]
    df["lag_1"] = g.shift(1)
    df["lag_7"] = g.shift(7)
    df["lag_28"] = g.shift(28)
    df["roll_mean_7"] = g.shift(1).rolling(7).mean().reset_index(level=0, drop=True)
    df["roll_mean_28"] = g.shift(1).rolling(28).mean().reset_index(level=0, drop=True)
    df["dow"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["sku_code"] = df["sku"].str[3:].astype(int)
    return df.dropna()

FEATS = ["lag_1", "lag_7", "lag_28", "roll_mean_7", "roll_mean_28",
         "dow", "month", "promo", "sku_code"]

# ---------------------------------------------------------------- 4. FORECAST
def mape(y, yhat):
    y = np.asarray(y); yhat = np.asarray(yhat)
    return np.mean(np.abs(y - yhat) / np.maximum(y, 1)) * 100

def forecast(df):
    cutoff = df["date"].max() - pd.Timedelta(days=60)      # time-based split
    train, test = df[df.date <= cutoff], df[df.date > cutoff]
    model = GradientBoostingRegressor(n_estimators=250, max_depth=4,
                                      learning_rate=0.05, random_state=42)
    model.fit(train[FEATS], train["units"])
    pred = model.predict(test[FEATS])
    naive = test["lag_7"].values                             # seasonal-naive baseline
    print(f"Gradient boosting  MAE={mean_absolute_error(test.units, pred):.2f}  "
          f"MAPE={mape(test.units, pred):.1f}%")
    print(f"Seasonal naive     MAE={mean_absolute_error(test.units, naive):.2f}  "
          f"MAPE={mape(test.units, naive):.1f}%")
    test = test.assign(pred=pred)
    return model, test

# ---------------------------------------------------------------- 5. SEGMENTATION
def segment_skus(df):
    s = df.groupby("sku")["units"].agg(["mean", "std"]).reset_index()
    s["cv"] = s["std"] / s["mean"]                           # demand variability
    s["volume"] = s["mean"] * 365
    X = StandardScaler().fit_transform(np.log1p(s[["volume"]]).join(s[["cv"]]))
    s["cluster"] = KMeans(n_clusters=3, n_init=10, random_state=42).fit_predict(X)
    return s

# ---------------------------------------------------------------- 6. INVENTORY
def reorder_policy(df, lead_time=3, service_z=1.65):
    s = df.groupby("sku")["units"].agg(["mean", "std"])
    s["safety_stock"] = service_z * s["std"] * np.sqrt(lead_time)
    s["reorder_point"] = s["mean"] * lead_time + s["safety_stock"]
    return s.round(1)

# ---------------------------------------------------------------- 7. ROUTING
def dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))

def savings_vrp(depot, stops, demand, capacity):
    """Clarke-Wright savings heuristic (pure python)."""
    n = len(stops)
    routes = {i: [i] for i in range(n)}
    load = {i: demand[i] for i in range(n)}
    sav = sorted(((dist(depot, stops[i]) + dist(depot, stops[j]) - dist(stops[i], stops[j]), i, j)
                  for i in range(n) for j in range(i + 1, n)), reverse=True)
    where = {i: i for i in range(n)}
    for s, i, j in sav:
        ri, rj = where[i], where[j]
        if ri == rj or load[ri] + load[rj] > capacity:
            continue
        a, b = routes[ri], routes[rj]
        if a[-1] == i and b[0] == j:   merged = a + b
        elif a[0] == i and b[-1] == j: merged = b + a
        elif a[0] == i and b[0] == j:  merged = a[::-1] + b
        elif a[-1] == i and b[-1] == j: merged = a + b[::-1]
        else: continue
        routes[ri], load[ri] = merged, load[ri] + load[rj]
        del routes[rj], load[rj]
        for k in merged: where[k] = ri
    return list(routes.values())

def route_cost(depot, stops, r):
    pts = [depot] + [stops[i] for i in r] + [depot]
    return sum(dist(pts[k], pts[k + 1]) for k in range(len(pts) - 1))

def routing_demo():
    depot = (50, 50)
    stops = [tuple(rng.uniform(0, 100, 2)) for _ in range(30)]
    demand = rng.integers(5, 25, 30)
    cap = 100
    # baseline: one trip per store
    base = sum(2 * dist(depot, s) for s in stops)
    routes = savings_vrp(depot, stops, demand, cap)
    opt = sum(route_cost(depot, stops, r) for r in routes)
    print(f"Routing: {len(routes)} routes, distance {base:.0f} -> {opt:.0f} "
          f"({(1 - opt / base) * 100:.1f}% saving)")
    return depot, stops, routes, base, opt

# ---------------------------------------------------------------- MAIN
if __name__ == "__main__":
    raw = make_demand()
    print("Missing values before cleaning:", int(raw.units.isna().sum()))
    df = add_features(clean(raw))
    print("Missing values after cleaning :", int(df.units.isna().sum()))

    model, test = forecast(df)
    seg = segment_skus(df)
    print(seg.groupby("cluster").agg(skus=("sku", "count"), volume=("volume", "mean"), cv=("cv", "mean")).round(2))
    pol = reorder_policy(df)
    print(pol.head())
    depot, stops, routes, base, opt = routing_demo()

    # figures
    one = test[test.sku == "SKU05"]
    plt.figure(figsize=(8, 3.6))
    plt.plot(one.date, one.units, label="Actual", lw=1.4)
    plt.plot(one.date, one.pred, label="Forecast", lw=1.4)
    plt.xticks(rotation=30, ha="right")
    plt.title("Demand forecast vs actual (SKU05, 60-day hold-out)")
    plt.legend(); plt.tight_layout(); plt.savefig("figures/forecast.png", dpi=150); plt.close()

    plt.figure(figsize=(5, 4))
    for c in sorted(seg.cluster.unique()):
        m = seg[seg.cluster == c]
        plt.scatter(m.volume, m.cv, label=f"Cluster {c}", s=45)
    plt.xlabel("Annual volume (units)"); plt.ylabel("Demand variability (CV)")
    plt.title("SKU segmentation (K-Means)"); plt.legend(); plt.tight_layout()
    plt.savefig("figures/clusters.png", dpi=150); plt.close()

    plt.figure(figsize=(5, 4.6))
    cols = plt.cm.tab10.colors
    for k, r in enumerate(routes):
        pts = [depot] + [stops[i] for i in r] + [depot]
        plt.plot(*zip(*pts), "-o", ms=3, color=cols[k % 10], lw=1.2)
    plt.scatter(*depot, marker="s", s=90, color="black", zorder=5)
    plt.title(f"Savings-heuristic routes ({(1-opt/base)*100:.0f}% shorter)")
    plt.tight_layout(); plt.savefig("figures/routes.png", dpi=150); plt.close()

    imp = pd.Series(model.feature_importances_, index=FEATS).sort_values()
    plt.figure(figsize=(5, 3.6)); imp.plot.barh(); plt.title("Feature importance")
    plt.tight_layout(); plt.savefig("figures/importance.png", dpi=150); plt.close()

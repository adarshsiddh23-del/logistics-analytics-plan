"""
Week 2: Data collection, cleaning and preprocessing pipeline for logistics data.
Reference dataset: DataCo Smart Supply Chain (Kaggle / Mendeley Data).
Because the real file is not bundled, a synthetic dataset with a similar schema
and deliberately injected quality problems is generated, so every step can be
demonstrated and verified. To use real data, replace load_raw() with pd.read_csv().
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

rng = np.random.default_rng(7)
METRICS = {}

# ------------------------------------------------------------------ 1. COLLECT
def make_raw(n=20000):
    modes = ["Standard Class", "Second Class", "First Class", "Same Day"]
    regions = ["South Asia", "Western Europe", "Central America", "North America", "Southeast Asia"]
    cats = ["Electronics", "Apparel", "Sporting Goods", "Furniture", "Groceries", "Toys"]
    order_date = pd.to_datetime("2023-01-01") + pd.to_timedelta(rng.integers(0, 730, n), unit="D")
    mode = rng.choice(modes, n, p=[.55, .20, .15, .10])
    sched = pd.Series(mode).map({"Standard Class": 4, "Second Class": 2, "First Class": 1, "Same Day": 0}).values
    real = np.clip(sched + rng.poisson(1.2, n) - rng.integers(0, 2, n), 0, 12)
    qty = rng.integers(1, 6, n)
    price = np.round(rng.lognormal(3.6, 0.7, n), 2)
    sales = np.round(qty * price * (1 - rng.choice([0, .05, .1, .2], n)), 2)
    df = pd.DataFrame({
        "order_id": np.arange(100000, 100000 + n),
        "order_date": order_date,
        "shipping_mode": mode,
        "region": rng.choice(regions, n),
        "category": rng.choice(cats, n),
        "quantity": qty,
        "unit_price": price,
        "sales": sales,
        "distance_km": np.round(rng.gamma(4, 180, n), 1),
        "weight_kg": np.round(rng.gamma(2.2, 3.5, n) * qty ** 0.5, 2),
        "days_shipping_scheduled": sched,
        "days_shipping_real": real,
    })
    df["delivery_date"] = df["order_date"] + pd.to_timedelta(df["days_shipping_real"], unit="D")

    # ---- inject realistic data quality problems
    df["order_date"] = df["order_date"].dt.strftime("%Y-%m-%d")
    df["delivery_date"] = df["delivery_date"].dt.strftime("%Y-%m-%d")
    m = rng.random(n) < 0.08                                    # mixed date format
    df.loc[m, "order_date"] = pd.to_datetime(df.loc[m, "order_date"]).dt.strftime("%d/%m/%Y")
    df["sales"] = df["sales"].astype(str)
    s = rng.random(n) < 0.15
    df.loc[s, "sales"] = "$" + df.loc[s, "sales"]               # currency symbol -> text
    c = rng.random(n) < 0.20                                    # inconsistent labels
    df.loc[c, "shipping_mode"] = df.loc[c, "shipping_mode"].str.upper() + " "
    c2 = rng.random(n) < 0.05
    df.loc[c2, "region"] = df.loc[c2, "region"].str.lower()
    for col, p in [("weight_kg", .06), ("distance_km", .04), ("region", .02), ("category", .015), ("delivery_date", .01)]:
        df.loc[rng.random(n) < p, col] = np.nan                 # missing values
    o = rng.choice(n, 90, replace=False)                        # extreme outliers
    df.loc[o[:45], "weight_kg"] = df.loc[o[:45], "weight_kg"].astype(float) * 40
    df.loc[o[45:], "distance_km"] = df.loc[o[45:], "distance_km"].astype(float) * 25
    neg = rng.choice(n, 40, replace=False); df.loc[neg, "quantity"] = -df.loc[neg, "quantity"]
    bad = rng.choice(n, 60, replace=False)                      # delivery before order
    df.loc[bad, "delivery_date"] = "2020-01-01"
    dup = df.sample(400, random_state=1)                        # duplicate rows
    return pd.concat([df, dup], ignore_index=True).sample(frac=1, random_state=3).reset_index(drop=True)

# ------------------------------------------------------------------ 2. AUDIT
def audit(df, label):
    rep = pd.DataFrame({"dtype": df.dtypes.astype(str),
                        "missing": df.isna().sum(),
                        "missing_pct": (df.isna().mean() * 100).round(2),
                        "unique": df.nunique()})
    print(f"\n=== Audit: {label} === rows={len(df)}  duplicates(order_id)={df.duplicated('order_id').sum()}")
    print(rep)
    return rep

# ------------------------------------------------------------------ 3. CLEAN
def clean_types(df):
    df = df.copy()
    for col in ["shipping_mode", "region", "category"]:
        df[col] = df[col].astype("string").str.strip().str.title()
    df["sales"] = pd.to_numeric(df["sales"].astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")
    # explicit formats: guessing day/month order silently corrupts dates like 03/04/2023
    iso = pd.to_datetime(df["order_date"], format="%Y-%m-%d", errors="coerce")
    dmy = pd.to_datetime(df["order_date"], format="%d/%m/%Y", errors="coerce")
    df["order_date"] = iso.fillna(dmy)
    df["delivery_date"] = pd.to_datetime(df["delivery_date"], errors="coerce")
    for col in ["distance_km", "weight_kg"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def remove_invalid(df):
    n0 = len(df)
    df = df.drop_duplicates()
    df = df.drop_duplicates("order_id", keep="first")
    METRICS["duplicates_removed"] = n0 - len(df)
    bad_qty = (df["quantity"] <= 0)
    METRICS["negative_quantity"] = int(bad_qty.sum())
    df.loc[bad_qty, "quantity"] = df.loc[bad_qty, "quantity"].abs()          # sign error -> correct
    bad_date = df["delivery_date"] < df["order_date"]
    METRICS["delivery_before_order"] = int(bad_date.sum())
    df.loc[bad_date, "delivery_date"] = pd.NaT                               # impossible -> treat as missing
    return df

def impute(df):
    df = df.copy()
    num_cols = ["weight_kg", "distance_km"]
    for col in num_cols + ["category", "region"]:
        df[f"{col}_was_missing"] = df[col].isna().astype(int)               # keep the signal
    METRICS["missing_before_impute"] = {c: int(df[c].isna().sum()) for c in num_cols + ["category", "region", "delivery_date"]}
    for col in num_cols:
        df[col] = df[col].fillna(df.groupby(["category", "region"])[col].transform("median"))
        df[col] = df[col].fillna(df[col].median())
    for col in ["category", "region"]:
        df[col] = df[col].fillna(df[col].mode()[0])
    # delivery_date can be rebuilt from order date + actual shipping days
    rebuilt = df["order_date"] + pd.to_timedelta(df["days_shipping_real"], unit="D")
    df["delivery_date"] = df["delivery_date"].fillna(rebuilt)
    df = df.dropna(subset=["order_date", "sales"])
    return df

def iqr_bounds(s, k=3.0):
    q1, q3 = s.quantile([.25, .75])
    return q1 - k * (q3 - q1), q3 + k * (q3 - q1)

def treat_outliers(df, cols=("weight_kg", "distance_km", "sales")):
    df = df.copy(); out = {}
    for col in cols:
        lo, hi = iqr_bounds(df[col])
        mask = (df[col] < lo) | (df[col] > hi)
        out[col] = {"flagged": int(mask.sum()), "upper_bound": round(float(hi), 2)}
        df[f"{col}_outlier"] = mask.astype(int)
        df[col] = df[col].clip(lower=max(lo, 0), upper=hi)                  # winsorise, keep the row
    METRICS["outliers"] = out
    return df

# ------------------------------------------------------------------ 4. FEATURES
def add_features(df):
    df = df.copy()
    df["lead_time_days"] = (df["delivery_date"] - df["order_date"]).dt.days
    df["delay_days"] = df["days_shipping_real"] - df["days_shipping_scheduled"]
    df["is_late"] = (df["delay_days"] > 0).astype(int)
    df["order_month"] = df["order_date"].dt.month
    df["order_dow"] = df["order_date"].dt.dayofweek
    df["cost_proxy_per_kg_km"] = df["sales"] / (df["weight_kg"] * df["distance_km"]).clip(lower=1)
    return df

# ------------------------------------------------------------------ 5. SCALE / ENCODE
def scale_and_encode(df):
    skewed = ["weight_kg", "distance_km", "sales"]
    METRICS["skew_before"] = {c: round(float(df[c].skew()), 2) for c in skewed}
    for c in skewed:
        df[f"{c}_log"] = np.log1p(df[c])
    METRICS["skew_after_log"] = {c: round(float(df[f"{c}_log"].skew()), 2) for c in skewed}
    df = pd.get_dummies(df, columns=["shipping_mode", "region", "category"], drop_first=True, dtype=int)
    num = [f"{c}_log" for c in skewed] + ["quantity", "days_shipping_scheduled"]
    train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["is_late"])
    train, test = train.copy(), test.copy()
    std = StandardScaler().fit(train[num])                                  # fit on TRAIN only
    train[[f"{c}_z" for c in num]] = std.transform(train[num])
    test[[f"{c}_z" for c in num]] = std.transform(test[num])
    mm = MinMaxScaler().fit(train[num])
    train[[f"{c}_mm" for c in num]] = mm.transform(train[num])
    test[[f"{c}_mm" for c in num]] = mm.transform(test[num])
    METRICS["z_train_mean"] = round(float(train[[f"{c}_z" for c in num]].mean().abs().max()), 4)
    METRICS["mm_range"] = [round(float(train[[f"{c}_mm" for c in num]].min().min()), 2),
                           round(float(train[[f"{c}_mm" for c in num]].max().max()), 2)]
    return df, train, test

# ------------------------------------------------------------------ 6. VALIDATE
def validate(df):
    checks = {
        "no duplicate order_id": df["order_id"].is_unique,
        "no missing in key columns": df[["order_date", "sales", "weight_kg", "distance_km", "region", "category"]].notna().all().all(),
        "quantity > 0": bool((df["quantity"] > 0).all()),
        "delivery >= order": bool((df["delivery_date"] >= df["order_date"]).all()),
        "sales > 0": bool((df["sales"] > 0).all()),
    }
    for k, v in checks.items(): print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    METRICS["validation"] = {k: bool(v) for k, v in checks.items()}
    assert all(checks.values()), "Validation failed"

# ------------------------------------------------------------------ MAIN
if __name__ == "__main__":
    raw = make_raw()
    raw.to_csv("data/raw/logistics_orders_raw.csv", index=False)
    METRICS["raw_rows"], METRICS["raw_cols"] = len(raw), raw.shape[1]
    rep_before = audit(raw, "RAW")
    METRICS["missing_pct_raw"] = rep_before["missing_pct"][rep_before["missing_pct"] > 0].to_dict()
    METRICS["label_variants_shipping_before"] = int(raw["shipping_mode"].nunique())
    METRICS["label_variants_region_before"] = int(raw["region"].nunique())

    typed = clean_types(raw)
    METRICS["label_variants_shipping_after"] = int(typed["shipping_mode"].nunique())
    METRICS["label_variants_region_after"] = int(typed["region"].nunique())
    METRICS["unparseable_dates"] = int(typed["order_date"].isna().sum())
    pre = typed[["weight_kg", "distance_km", "sales"]].copy()
    valid = remove_invalid(typed)
    filled = impute(valid)
    treated = treat_outliers(filled)
    feat = add_features(treated)
    METRICS["rows_final"] = len(feat)
    print("\n=== VALIDATION ==="); validate(feat)
    full, train, test = scale_and_encode(feat)
    METRICS["final_cols"] = full.shape[1]
    METRICS["train_rows"], METRICS["test_rows"] = len(train), len(test)
    METRICS["describe_before"] = pre.describe().loc[["mean", "std", "max"]].round(1).to_dict()
    METRICS["describe_after"] = feat[["weight_kg", "distance_km", "sales"]].describe().loc[["mean", "std", "max"]].round(1).to_dict()
    METRICS["late_rate_pct"] = round(float(feat["is_late"].mean() * 100), 1)
    audit(feat, "CLEANED")
    full.to_csv("data/processed/logistics_orders_clean.csv", index=False)
    json.dump(METRICS, open("metrics.json", "w"), indent=2, default=str)

    # ---- figures
    miss = rep_before["missing_pct"][rep_before["missing_pct"] > 0].sort_values()
    plt.figure(figsize=(6, 3.2)); miss.plot.barh(color="#2E75B6")
    plt.xlabel("% missing"); plt.title("Missing values in raw data"); plt.tight_layout()
    plt.savefig("figures/missing.png", dpi=150); plt.close()

    fig, ax = plt.subplots(1, 3, figsize=(9, 3.4))
    for a, c in zip(ax, ["weight_kg", "distance_km", "sales"]):
        a.boxplot([pre[c].dropna(), feat[c]], tick_labels=["Before", "After"], showfliers=True,
                  flierprops=dict(markersize=2, alpha=.4))
        a.set_title(c)
    plt.suptitle("Outlier treatment (IQR x 3 winsorizing)"); plt.tight_layout()
    plt.savefig("figures/outliers.png", dpi=150); plt.close()

    fig, ax = plt.subplots(1, 3, figsize=(9, 3))
    ax[0].hist(feat["weight_kg"], bins=40, color="#2E75B6"); ax[0].set_title("weight_kg (cleaned)")
    ax[1].hist(full["weight_kg_log"], bins=40, color="#2E75B6"); ax[1].set_title("log1p(weight_kg)")
    ax[2].hist(train["weight_kg_log_z"], bins=40, color="#2E75B6"); ax[2].set_title("standardized (z-score)")
    plt.tight_layout(); plt.savefig("figures/scaling.png", dpi=150); plt.close()
    print("\nMETRICS:", json.dumps(METRICS, indent=1, default=str))

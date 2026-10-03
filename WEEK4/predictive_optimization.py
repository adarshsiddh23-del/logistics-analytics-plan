"""
Week 4: Predictive modelling and optimisation for a logistics network.
Prediction target : total delivery time in hours (dispatch delay + transit), per shipment.
Optimisation      : (A) data-driven delivery promise, (B) cost-minimising transport mode choice (MILP),
                    (C) warehouse capacity what-if.
Data are simulated (24 months, 25,000 shipments). The simulator keeps the outcome each shipment WOULD have had
under every mode ("potential outcomes"), so optimisation policies can be scored against ground truth,
which is impossible with real historical data.
"""
import json, time, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from scipy import sparse
from scipy.optimize import milp, LinearConstraint, Bounds
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score, brier_score_loss
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid"); rng = np.random.default_rng(404); R = {}; MODES = ["Road", "Rail", "Air"]
t0 = time.time()

# =============================================================== 1. SIMULATE
def simulate(n=25000):
    date = pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 731, n), unit="D")
    dc = rng.choice(["DC-North", "DC-South", "DC-West"], n, p=[.35, .30, .35])
    region = rng.choice(["North", "South", "East", "West", "Central"], n, p=[.22, .24, .18, .20, .16])
    carrier = rng.choice(["Carrier A", "Carrier B", "Carrier C", "Carrier D"], n, p=[.32, .28, .22, .18])
    urgent = (rng.random(n) < .18).astype(int)
    base = pd.Series(region).map({"North": 650, "South": 900, "East": 1100, "West": 750, "Central": 500}).values
    dist = np.clip(base * rng.lognormal(0, .30, n), 80, 3500).round(0)
    weight = np.clip(rng.lognormal(5.0, 1.0, n), 5, 8000).round(1)
    df = pd.DataFrame(dict(date=date, dc=dc, region=region, carrier=carrier, urgent=urgent, distance_km=dist, weight_kg=weight))
    df["month"] = df.date.dt.month; df["weekday"] = df.date.dt.day_name(); df["peak"] = df.month.isin([10, 11, 12]).astype(int)
    event = rng.random(n) < np.where(df.month.isin([6, 7, 8]), .12, .04)
    df["weather_event"] = event.astype(int)
    df["weather_forecast"] = ((event & (rng.random(n) < .7)) | (~event & (rng.random(n) < .03))).astype(int)
    df["fuel_index"] = (100 + 6 * np.sin(2 * np.pi * (df.month - 2) / 12) + rng.normal(0, 1.5, n)).round(1)

    # daily workload per DC (orders to process); West has lower capacity -> structural bottleneck
    cap = {"DC-North": 40, "DC-South": 34, "DC-West": 36}; lam0 = {"DC-North": 35, "DC-South": 30, "DC-West": 38}
    days = pd.date_range("2024-01-01", periods=731)
    tab = pd.DataFrame([(d, c) for d in days for c in cap], columns=["date", "dc"])
    wk = tab.date.dt.dayofweek.map({0: 1.30, 5: .70, 6: .70}).fillna(1.0); pk = np.where(tab.date.dt.month >= 10, 1.25, 1.0)
    tab["dc_load"] = rng.poisson(tab.dc.map(lam0).values * wk.values * pk)
    df = df.merge(tab, on=["date", "dc"], how="left")
    ratio = df.dc_load / df.dc.map(cap)
    df["dispatch_h"] = rng.gamma(4, (2 + 4 * ratio ** 2) / 4).clip(0.5, 96).round(1)

    # potential outcomes under each mode, with shared random noise
    speed = {"Road": 45, "Rail": 38, "Air": 450}; fixed = {"Road": 0, "Rail": 6, "Air": 4}
    rate = {"Road": .0045, "Rail": .0030, "Air": .0300}
    cp = df.carrier.map({"Carrier A": 1.0, "Carrier B": 1.05, "Carrier C": 1.22, "Carrier D": 1.0}).values
    cp = cp * np.where((df.carrier == "Carrier C") & (df.region == "South"), 1.15, 1.0)
    mk = df.carrier.map({"Carrier A": 1.0, "Carrier B": .92, "Carrier C": .88, "Carrier D": 1.12}).values
    e_t = rng.lognormal(0, .12, n); e_c = rng.lognormal(0, .08, n)
    w_u = rng.uniform(0, 1, n); p_u = rng.uniform(0, 1, n)
    T, C = {}, {}
    for m in MODES:
        wx = (6 + 24 * w_u) if m != "Air" else (2 + 8 * w_u)
        px = 8 * p_u if m != "Air" else 3 * p_u
        T[m] = df.dispatch_h + (df.distance_km / speed[m] + fixed[m]) * cp * e_t + df.weather_event * wx + df.peak * px
        C[m] = (40 + rate[m] * df.distance_km * df.weight_kg ** .85) * mk * df.fuel_index / 100 * np.where(df.peak == 1, 1.12, 1.0) * e_c
    # company's current (rule-based) delivery promise, in hours
    df["promise_h"] = ((df.distance_km / 45 * 1.55 + 18) * np.where(df.urgent == 1, .55, 1.0)).round(1)
    # historical mode choice: urgent -> mostly air, otherwise road/rail
    u = rng.random(n)
    mode = np.where(df.urgent == 1, np.where(u < .55, "Air", np.where(u < .90, "Road", "Rail")),
                    np.where(u < .02, "Air", np.where(u < .72, "Road", "Rail")))
    df["mode"] = mode
    po_t = pd.DataFrame({m: T[m] for m in MODES}); po_c = pd.DataFrame({m: C[m] for m in MODES})
    idx = np.array([MODES.index(m) for m in mode])
    df["total_h"] = po_t.values[np.arange(n), idx].round(1)
    df["cost_usd"] = po_c.values[np.arange(n), idx].round(2)
    df["on_time"] = (df.total_h <= df.promise_h).astype(int)
    order = df.date.argsort(kind="stable").values
    return df.iloc[order].reset_index(drop=True), po_t.iloc[order].reset_index(drop=True), po_c.iloc[order].reset_index(drop=True)

df, PO_T, PO_C = simulate()
df.to_csv("data/shipments_w4.csv", index=False)
R["n"] = len(df); R["otif_hist"] = round(df.on_time.mean() * 100, 1); R["otif_hist_by_mode"] = (df.groupby("mode").on_time.mean() * 100).round(1).to_dict(); R["otif_hist_urgent"] = (df.groupby("urgent").on_time.mean() * 100).round(1).to_dict()
R["mode_share"] = (df["mode"].value_counts(normalize=True) * 100).round(1).to_dict()
R["total_h_mean"], R["total_h_median"] = round(df.total_h.mean(), 1), round(df.total_h.median(), 1)
R["total_h_by_mode"] = df.groupby("mode").total_h.agg(["mean", "std"]).round(1).to_dict("index")

# =============================================================== 2. FEATURES AND SPLIT
CAT = ["dc", "region", "mode", "carrier", "weekday"]
NUM = ["distance_km", "weight_kg", "urgent", "month", "peak", "weather_forecast", "fuel_index", "dc_load"]
FEATS = CAT + NUM            # leakage guard: dispatch_h, total_h, weather_event are NOT used
TARGET = "total_h"
cut = pd.Timestamp("2025-06-30")
train, test = df[df.date <= cut].copy(), df[df.date > cut].copy()
Xtr, ytr, Xte, yte = train[FEATS], train[TARGET], test[FEATS], test[TARGET]
R["split"] = {"train": len(train), "test": len(test), "train_end": str(cut.date())}

def prep(scale=False):
    num = StandardScaler() if scale else "passthrough"
    return ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT), ("num", num, NUM)])

def metrics(y, p):
    return {"MAE": mean_absolute_error(y, p), "RMSE": mean_squared_error(y, p) ** .5, "R2": r2_score(y, p),
            "within6h": float(np.mean(np.abs(y - p) <= 6) * 100)}

tscv = TimeSeriesSplit(n_splits=4)

# =============================================================== 3. MODELS
class MeanByMode(DummyRegressor):          # baseline: average hours of the shipment's mode
    def fit(self, X, y): self.m_ = pd.Series(y.values, index=X["mode"].values).groupby(level=0).mean(); return self
    def predict(self, X): return X["mode"].map(self.m_).fillna(self.m_.mean()).values

models = {"Baseline (mean by mode)": MeanByMode(),
          "Linear regression": Pipeline([("p", prep(True)), ("m", LinearRegression())])}
# decision tree: tune depth with time-series CV
gs = GridSearchCV(Pipeline([("p", prep()), ("m", DecisionTreeRegressor(min_samples_leaf=20, random_state=0))]),
                  {"m__max_depth": [4, 6, 8, 10, 14]}, cv=tscv, scoring="neg_mean_absolute_error").fit(Xtr, ytr)
models["Decision tree"] = gs.best_estimator_; R["tree_best_depth"] = int(gs.best_params_["m__max_depth"])
models["Random forest"] = Pipeline([("p", prep()), ("m", RandomForestRegressor(n_estimators=120, min_samples_leaf=5, max_features=.5, random_state=0, n_jobs=-1))])
# gradient boosting: randomised search with time-series CV
default_hgb = Pipeline([("p", prep()), ("m", HistGradientBoostingRegressor(random_state=0))])
rs = RandomizedSearchCV(Pipeline([("p", prep()), ("m", HistGradientBoostingRegressor(random_state=0))]),
    {"m__learning_rate": [.03, .05, .08, .12], "m__max_depth": [3, 4, 6, None], "m__max_leaf_nodes": [15, 31, 63],
     "m__min_samples_leaf": [10, 20, 40], "m__l2_regularization": [0, .1, 1.0], "m__max_iter": [200, 350, 500]},
    n_iter=14, cv=tscv, scoring="neg_mean_absolute_error", random_state=42).fit(Xtr, ytr)
best_hgb = rs.best_estimator_; models["Gradient boosting (tuned)"] = best_hgb
R["hgb_best_params"] = {k.replace("m__", ""): (None if v is None else (float(v) if isinstance(v, float) else int(v))) for k, v in rs.best_params_.items()}

res, cvres = {}, {}
for name, mdl in models.items():
    if name != "Baseline (mean by mode)": mdl.fit(Xtr, ytr)
    else: mdl.fit(Xtr, ytr)
    res[name] = metrics(yte.values, mdl.predict(Xte))
    if name not in ("Baseline (mean by mode)",):
        cvres[name] = (-cross_val_score(mdl, Xtr, ytr, cv=tscv, scoring="neg_mean_absolute_error")).tolist()
default_hgb.fit(Xtr, ytr); res["Gradient boosting (default)"] = metrics(yte.values, default_hgb.predict(Xte))
R["results"] = {k: {m: round(v, 3) for m, v in d.items()} for k, d in res.items()}
R["cv_mae"] = {k: [round(x, 2) for x in v] for k, v in cvres.items()}
R["cv_mae_mean"] = {k: round(float(np.mean(v)), 2) for k, v in cvres.items()}
best_name = min((k for k in res if "Gradient boosting (tuned)" == k or k in models), key=lambda k: res[k]["MAE"])
best = models[best_name]; R["best_model"] = best_name
pred = best.predict(Xte); test["pred"] = pred; test["resid"] = test.total_h - test.pred
R["resid_mean"], R["resid_std"] = round(test.resid.mean(), 2), round(test.resid.std(), 2)
R["mae_by_mode"] = test.groupby("mode").resid.apply(lambda s: s.abs().mean()).round(2).to_dict()
R["mae_by_dc"] = test.groupby("dc").resid.apply(lambda s: s.abs().mean()).round(2).to_dict()
R["mae_by_month"] = test.groupby("month").resid.apply(lambda s: s.abs().mean()).round(2).to_dict()
R["bias_by_month"] = test.groupby("month").resid.mean().round(2).to_dict()
print("models done", round(time.time() - t0), "s", best_name)

# permutation importance on the test set (raw columns -> groups one-hot back together)
pi = permutation_importance(best, Xte, yte, n_repeats=4, random_state=0, scoring="neg_mean_absolute_error")
imp = pd.Series(pi.importances_mean, index=FEATS).sort_values(ascending=False); R["perm_importance"] = imp.round(2).to_dict()

# =============================================================== 4. PREDICTION INTERVALS (quantile GB)
qp = {k: v for k, v in best_hgb.named_steps["m"].get_params().items() if k in ("learning_rate", "max_depth", "max_leaf_nodes", "min_samples_leaf", "l2_regularization", "max_iter")}
QS = [.1, .3, .5, .7, .8, .9, .95]; qmods = {}
def fitq(q): return Pipeline([("p", prep()), ("m", HistGradientBoostingRegressor(loss="quantile", quantile=q, random_state=0, **qp))]).fit(Xtr, ytr)
for q in QS: qmods[q] = fitq(q)
def qpred(X):
    P = np.column_stack([qmods[q].predict(X) for q in QS]); return np.maximum.accumulate(P, axis=1)   # no crossing
Q = qpred(Xte); qi = {q: i for i, q in enumerate(QS)}
R["interval80_coverage"] = round(float(np.mean((yte.values >= Q[:, qi[.1]]) & (yte.values <= Q[:, qi[.9]])) * 100), 1)
R["p90_coverage"] = round(float(np.mean(yte.values <= Q[:, qi[.9]]) * 100), 1)
R["p10_coverage"] = round(float(np.mean(yte.values <= Q[:, qi[.1]]) * 100), 1)

# =============================================================== 5. OPTIMISATION A: data-driven delivery promise
base_quote = test.promise_h.values
base_otif = float(np.mean(yte.values <= base_quote) * 100); base_avg = float(base_quote.mean())
R["A_baseline"] = {"otif": round(base_otif, 1), "avg_quote_h": round(base_avg, 1)}
curve_model = [(float(np.mean(yte.values <= Q[:, qi[q]]) * 100), float(Q[:, qi[q]].mean()), q) for q in QS if q >= .5]
ks = np.linspace(.7, 3.0, 70)
curve_base = [(float(np.mean(yte.values <= base_quote * k) * 100), float((base_quote * k).mean()), float(k)) for k in ks]
def avg_for_otif(curve, target): 
    c = sorted(curve, key=lambda t: t[0]); xs = [t[0] for t in c]; ys = [t[1] for t in c]; return float(np.interp(target, xs, ys))
tgt = curve_model[-2][0]          # on-time level reached by P90 quote
R["A_p90"] = {"otif": round(tgt, 1), "avg_quote_h": round(curve_model[-2][1], 1)}
R["A_base_same_otif_avg_h"] = round(avg_for_otif(curve_base, tgt), 1)
R["A_quote_saving_pct"] = round((1 - curve_model[-2][1] / avg_for_otif(curve_base, tgt)) * 100, 1)
same_len = [t for t in curve_base if abs(t[1] - curve_model[-2][1]) < 0.3]
R["A_base_same_len_otif"] = round(float(np.mean([t[0] for t in same_len])), 1) if same_len else None
R["A_curve_model"] = [(round(a, 1), round(b, 1), c) for a, b, c in curve_model]

# =============================================================== 6. OPTIMISATION B: cost-minimising mode choice (MILP)
# Step 1: models that predict, for any shipment and ANY mode, the cost and the probability of meeting the promise.
def swap(X, m):
    Z = X.copy(); Z["mode"] = m; return Z
cost_model = Pipeline([("p", prep()), ("m", HistGradientBoostingRegressor(random_state=0, **qp))]).fit(Xtr, np.log(train.cost_usd))
clf = Pipeline([("p", prep()), ("m", HistGradientBoostingClassifier(random_state=0, learning_rate=qp["learning_rate"], max_depth=qp["max_depth"],
        max_leaf_nodes=qp["max_leaf_nodes"], min_samples_leaf=qp["min_samples_leaf"], l2_regularization=qp["l2_regularization"], max_iter=qp["max_iter"]))]).fit(Xtr, train.on_time)
pc = np.column_stack([np.exp(cost_model.predict(swap(Xte, m))) for m in MODES])
pp = np.column_stack([clf.predict_proba(swap(Xte, m))[:, 1] for m in MODES])
R["cost_model_mape"] = round(float(np.mean(np.abs(np.exp(cost_model.predict(Xte)) - test.cost_usd) / test.cost_usd) * 100), 1)
R["cost_model_r2_log"] = round(r2_score(np.log(test.cost_usd), cost_model.predict(Xte)), 3)
ph = clf.predict_proba(Xte)[:, 1]
R["clf_auc"] = round(roc_auc_score(test.on_time, ph), 3); R["clf_brier"] = round(brier_score_loss(test.on_time, ph), 3)
R["clf_pred_otif"] = round(float(ph.mean() * 100), 1); R["clf_real_otif"] = round(float(test.on_time.mean() * 100), 1)

prom = test.promise_h.values; N = len(test)
T_true, C_true = PO_T.loc[test.index, MODES].values, PO_C.loc[test.index, MODES].values
okT = (T_true <= prom[:, None]).astype(float)               # ground truth: would this mode have met the promise?
A_eq = sparse.kron(sparse.eye(N), np.ones((1, 3))).tocsr()
A_air = sparse.csr_matrix((np.ones(N), (np.zeros(N, int), np.arange(N) * 3 + 2)), shape=(1, 3 * N))
cap_air = int(.15 * N)

def solve(cost, prob, target_otif):
    """min total cost  s.t. each shipment gets one mode, expected on-time >= target, air shipments <= cap."""
    cons = [LinearConstraint(A_eq, 1, 1), LinearConstraint(A_air, -np.inf, cap_air),
            LinearConstraint(sparse.csr_matrix(prob.flatten()[None, :]), target_otif * N, np.inf)]
    sol = milp(cost.flatten(), constraints=cons, integrality=np.ones(3 * N), bounds=Bounds(0, 1))
    return sol.x.reshape(N, 3).argmax(1) if sol.success else None

def score(ch, label):
    t = T_true[np.arange(N), ch]; cst = C_true[np.arange(N), ch]
    return {"policy": label, "total_cost": round(float(cst.sum())), "avg_cost": round(float(cst.mean()), 1),
            "on_time_pct": round(float(np.mean(t <= prom) * 100), 1), "late": int(np.sum(t > prom)),
            "air_pct": round(float(np.mean(ch == 2) * 100), 1), "rail_pct": round(float(np.mean(ch == 1) * 100), 1),
            "road_pct": round(float(np.mean(ch == 0) * 100), 1), "avg_hours": round(float(t.mean()), 1)}
hist = np.array([MODES.index(m) for m in test["mode"]])
hist_pred_otif = float(pp[np.arange(N), hist].mean())        # model's own estimate of historical on-time rate
hist_real = float(okT[np.arange(N), hist].mean())

pol = [score(hist, "Historical choices"), score(np.zeros(N, int), "All road")]
ch_same = solve(pc, pp, hist_pred_otif); pol.append(score(ch_same, "MILP, same on-time level as today"))
ch90 = solve(pc, pp, .90); pol.append(score(ch90, "MILP, 90% on-time target"))
ch95 = solve(pc, pp, .95); pol.append(score(ch95, "MILP, 95% on-time target")) if ch95 is not None else None
ch_or = solve(C_true, okT, hist_real); pol.append(score(ch_or, "Oracle, same on-time level (perfect information)"))
R["B_policies"] = pol; R["B_solver"] = {"n_vars": 3 * N, "air_cap": cap_air, "hist_pred_otif": round(hist_pred_otif * 100, 1)}
R["B_saving_same_pct"] = round((1 - pol[2]["total_cost"] / pol[0]["total_cost"]) * 100, 1)
R["B_oracle_saving_pct"] = round((1 - pol[-1]["total_cost"] / pol[0]["total_cost"]) * 100, 1)
R["B_switches_same"] = {f"{a}->{b}": int(((hist == i) & (ch_same == j)).sum()) for i, a in enumerate(MODES) for j, b in enumerate(MODES) if i != j}

# frontier: sweep the on-time target
front, front_or = [], []
for tg in np.arange(.78, .981, .02):
    ch = solve(pc, pp, tg)
    if ch is not None: sc = score(ch, ""); front.append((tg, sc["on_time_pct"], sc["total_cost"] / 1e6))
    ch = solve(C_true, okT, tg)
    if ch is not None: sc = score(ch, ""); front_or.append((tg, sc["on_time_pct"], sc["total_cost"] / 1e6))
R["B_frontier"] = [(round(a * 100), b, round(c, 3)) for a, b, c in front]

# =============================================================== 7. OPTIMISATION C: DC-West capacity what-if (model-based)
w = test[test.dc == "DC-West"]; Xw = w[FEATS].copy(); base_pred = best.predict(Xw)
what = {}
for red in (.10, .15, .20):
    Z = Xw.copy(); Z["dc_load"] = (Z["dc_load"] * (1 - red)).round()
    what[f"{int(red*100)}%"] = round(float(base_pred.mean() - best.predict(Z).mean()), 2)
R["C_west_hours_saved"] = what; R["C_west_mean_pred"] = round(float(base_pred.mean()), 1); R["C_west_n"] = len(w)
mon = w.weekday == "Monday"; R["C_west_monday_share"] = round(float(mon.mean() * 100), 1)

# =============================================================== 8. FIGURES
def save(n, w=None): plt.tight_layout(); plt.savefig(f"figures/{n}.png", dpi=150); plt.close()
fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.6))
sns.histplot(data=df, x="total_h", hue="mode", bins=60, element="step", stat="count", common_norm=False, ax=ax[0]); ax[0].set_xlim(0, 120); ax[0].set_title("Target: total delivery time (hours)")
tm = df.groupby(df.date.dt.to_period("M")).total_h.mean(); ax[1].plot(tm.index.astype(str), tm.values, "-o", ms=3, color="#1F3A5F")
ax[1].axvline(list(tm.index.astype(str)).index("2025-06") + .5, color="r", ls="--"); ax[1].text(list(tm.index.astype(str)).index("2025-06") + .8, tm.max() * .995, "test period", color="r", fontsize=8)
ax[1].set_xticks(range(0, len(tm), 3)); ax[1].set_xticklabels(tm.index.astype(str)[::3], rotation=45, ha="right", fontsize=7); ax[1].set_title("Mean delivery time by month and time-based split")
save("p1_target_split")

names = [k for k in res]; order = sorted(names, key=lambda k: res[k]["MAE"], reverse=True)
fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8))
for a, met in zip(ax, ["MAE", "RMSE"]):
    vals = [res[k][met] for k in order]; bars = a.barh(order, vals, color=["#2E75B6" if k == best_name else "#9DC3E6" for k in order])
    for b, v in zip(bars, vals): a.text(v + .1, b.get_y() + b.get_height() / 2, f"{v:.2f}", va="center", fontsize=8)
    a.set_title(f"{met} on test set (hours, lower is better)"); a.set_xlim(0, max(vals) * 1.15)
ax[1].set_yticklabels([])
save("p2_model_comparison")

fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
ax[0].scatter(test.total_h, test.pred, s=4, alpha=.3, c="#2E75B6"); lim = [0, np.percentile(test.total_h, 99.5)]; ax[0].plot(lim, lim, "r--", lw=1); ax[0].set_xlim(lim); ax[0].set_ylim(lim)
ax[0].set_xlabel("Actual hours"); ax[0].set_ylabel("Predicted hours"); ax[0].set_title("Predicted vs actual")
sns.histplot(test.resid, bins=60, kde=True, ax=ax[1], color="#2E75B6"); ax[1].axvline(0, c="r", ls="--"); ax[1].set_title("Residual distribution (actual - predicted)")
sns.boxplot(data=test, x="mode", y="resid", order=MODES, ax=ax[2], showfliers=False); ax[2].axhline(0, c="r", ls="--"); ax[2].set_title("Residuals by transport mode")
save("p3_pred_vs_actual")

fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.6))
imp.sort_values().plot.barh(ax=ax[0], color="#2E75B6"); ax[0].set_title("Permutation importance (MAE increase, hours)")
cvdf = pd.DataFrame({k: v for k, v in cvres.items()}, index=[f"Fold {i+1}" for i in range(4)]); cvdf.plot(kind="bar", ax=ax[1], width=.8); ax[1].set_ylabel("MAE (hours)"); ax[1].set_title("Time-series cross-validation MAE by fold")
ax[1].tick_params(axis="x", rotation=0); ax[1].legend(fontsize=7)
save("p4_importance_cv")

s = test.sample(160, random_state=3).copy(); s["lo"] = Q[test.index.get_indexer(s.index), qi[.1]]; s["hi"] = Q[test.index.get_indexer(s.index), qi[.9]]; s = s.sort_values("pred")
plt.figure(figsize=(8.5, 3.8)); x = np.arange(len(s)); plt.fill_between(x, s.lo, s.hi, alpha=.25, color="#2E75B6", label="80% prediction interval (P10-P90)")
plt.plot(x, s.pred, c="#1F3A5F", lw=1.2, label="Point prediction"); plt.scatter(x, s.total_h, s=9, c="#C00000", label="Actual", zorder=3)
plt.xlabel("160 random test shipments, sorted by prediction"); plt.ylabel("Hours"); plt.legend(fontsize=8); plt.title(f"Prediction intervals (80% interval covers {R['interval80_coverage']}% of test shipments)")
save("p5_intervals")

plt.figure(figsize=(7, 4.2))
cb = [c for c in curve_base if c[1] <= 62]; plt.plot([c[1] for c in cb], [c[0] for c in cb], "-", c="#7F7F7F", label="Rule-based promise x flat buffer")
plt.plot([c[1] for c in curve_model], [c[0] for c in curve_model], "-o", c="#2E75B6", label="Model-based promise (quantiles 0.5 to 0.95)")
plt.scatter([base_avg], [base_otif], c="#C00000", s=60, zorder=5, label="Current rule (no buffer)")
for a, b, q in curve_model: plt.annotate(f"P{int(q*100)}", (b, a), xytext=(-24, 4), textcoords="offset points", fontsize=8)
plt.xlabel("Average promised delivery time (hours)"); plt.ylabel("Shipments delivered within promise (%)"); plt.ylim(45, 101); plt.xlim(26, 62); plt.legend(fontsize=8, loc="lower right")
plt.title("Promise policy: reliability versus length of promise"); save("p6_promise_curve")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot([f[1] for f in front], [f[2] for f in front], "-o", c="#2E75B6", ms=4, label="Model-based MILP (realised result)")
ax[0].plot([f[1] for f in front_or], [f[2] for f in front_or], "--", c="#1F3A5F", label="Oracle (perfect information)")
ax[0].scatter([pol[0]["on_time_pct"]], [pol[0]["total_cost"] / 1e6], c="#C00000", s=70, zorder=5, label="Historical choices")
ax[0].scatter([pol[1]["on_time_pct"]], [pol[1]["total_cost"] / 1e6], c="#7F7F7F", s=55, marker="s", zorder=5, label="All road")
ax[0].set_xlabel("On-time delivery achieved (%)"); ax[0].set_ylabel("Total transport cost (million USD)"); ax[0].set_title("Cost versus service frontier (test period)"); ax[0].legend(fontsize=8)
mix = pd.DataFrame([[p["road_pct"], p["rail_pct"], p["air_pct"]] for p in [pol[0], pol[2], pol[3]]], columns=MODES, index=["Historical", "MILP same\non-time", "MILP 90%\ntarget"])
mix.plot(kind="bar", stacked=True, ax=ax[1], color=["#9DC3E6", "#70AD47", "#C00000"]); ax[1].set_title("Mode mix (% of shipments)"); ax[1].tick_params(axis="x", rotation=0); ax[1].legend(fontsize=8)
save("p7_mode_optimisation")

mm = test.groupby("month").resid.apply(lambda s_: s_.abs().mean()); plt.figure(figsize=(6.5, 3.4)); plt.bar(mm.index, mm.values, color=["#C00000" if m >= 10 else "#2E75B6" for m in mm.index])
plt.xlabel("Month of 2025 (test period)"); plt.ylabel("MAE (hours)"); plt.title("Forecast error by month (red = peak season)"); save("p8_error_by_month")

json.dump(R, open("results_w4.json", "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in R.items() if k not in ("A_curve_model",)}, indent=1, default=str)); print("total", round(time.time() - t0), "s")

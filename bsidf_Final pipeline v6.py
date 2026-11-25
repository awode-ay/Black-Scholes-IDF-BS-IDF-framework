"""
BS-IDF pipeline: stochastic non-stationary IDF curves for the Ogun River Basin (HydroBASINS level 6, HYBAS_ID 1060960280).
GEV baseline x geometric-Brownian-motion climate multiplier with uncertain drift (Black-Scholes-type exceedance equation).

Inputs (data/):  chirps_ams_ogun_1981_2024.csv   CHIRPS v2.0 basin annual maxima (Earth Engine export, gee/)
                 cmip6_pr_ams_v6.csv              NEX-GDDP-CMIP6 v2.0 basin annual maxima, 33 models (src/extract_cmip6_v6.py)
                 cmip6_tas_v6.csv                 basin temperature for the Clausius-Clapeyron reference
                 gee_ams_products.csv, model_indices_v6.csv, cmip6_arf.csv, cmip6_spatial_change.nc (optional)
                 ogun_hybas6_1060960280.geojson and Natural Earth layers (maps)
Outputs:         outputs/run/  tables T*.csv, figures *.png, results.json
Run:             python src/bsidf_pipeline.py            (full run, about 30 min)
Quick test:      B_BOOT=100 N_MC=200 N_SIMVOL=10 N_SOBOL=200 N_DLL=500 python src/bsidf_pipeline.py
"""
import os, sys, json, warnings
import numpy as np, pandas as pd
from scipy import stats
from scipy.stats import genextreme, norm, chi2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bsidf_core import (gev_fit_lmom, gev_cdf, gev_q, design_value, predictive_cdf,
                           closed_form_design_value, v5_equation6, ns_gev_fit)
from bsidf_params import model_series, fit_params, SLICES, T0
from rwvol import rw_sigma
from lmoments3 import lmom_ratios

SRC = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(SRC)
HERE = os.environ.get("BSIDF_DATA", os.path.join(ROOT, "data"))          # input data folder
OUT = os.environ.get("BSIDF_OUT", os.path.join(ROOT, "outputs", "run")); os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(2026)
TR = [2, 5, 10, 25, 50, 100]
DUR = [1, 3, 6, 12, 24]
BETA = 1 / 3
BETA_LO, BETA_HI = 0.10, 0.40   # IMD 1/3 central; West African gauge scaling ~0.08-0.20 (Sane et al. 2018)
WA_BETA = 0.15
HORIZONS = [2030, 2050, 2070, 2100]
SCEN = ["ssp245", "ssp585"]
SCEN_LAB = {"ssp245": "SSP2-4.5", "ssp585": "SSP5-8.5"}
B_BOOT = int(os.environ.get("B_BOOT", 2000))
N_MC = int(os.environ.get("N_MC", 4000))
R = {}  # results collected for json

# ------------------------------------------------------------------ figure style
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "axes.facecolor": "white",
                     "legend.frameon": False, "lines.linewidth": 1.6, "savefig.dpi": 300})


def savefig(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight"); plt.close(fig)


def tab(df, name):
    df.to_csv(os.path.join(OUT, name), index=True, float_format="%.6g")


# ================================================================== 1. DATA
obs = pd.read_csv(os.path.join(HERE, "chirps_ams_ogun_1981_2024.csv")).sort_values("year")
assert len(obs) == 44 and (obs.n_pixels == obs.n_pixels.iloc[0]).all()
x_all = obs.AMS_24hr.values; yrs = obs.year.values
x_cal = obs[obs.year <= 2005].AMS_24hr.values; x_val = obs[obs.year >= 2006].AMS_24hr.values
cm_file = "cmip6_pr_ams_v6.csv" if os.path.exists(os.path.join(HERE, "cmip6_pr_ams_v6.csv")) else "cmip6_ams_ogun.csv"
cm = pd.read_csv(os.path.join(HERE, cm_file))
MODELS = sorted(cm.model.unique())
R["data"] = dict(cmip6_file=cm_file, n_models=len(MODELS), models=MODELS, n_obs=len(obs))
print(f"CHIRPS n={len(obs)} | CMIP6 file={cm_file} models={len(MODELS)}")

# ================================================================== 2. STATIONARY GEV
def mann_kendall(x):
    n = len(x); s = sum(np.sign(x[j] - x[i]) for i in range(n - 1) for j in range(i + 1, n))
    var = n * (n - 1) * (2 * n + 5) / 18; z = (s - np.sign(s)) / np.sqrt(var) if s != 0 else 0
    slopes = [(x[j] - x[i]) / (j - i) for i in range(n - 1) for j in range(i + 1, n)]
    return dict(S=int(s), z=z, p=2 * (1 - norm.cdf(abs(z))), sen=np.median(slopes))


def ad_stat(x, cdf):
    x = np.sort(x); n = len(x); F = np.clip(cdf(x), 1e-12, 1 - 1e-12); i = np.arange(1, n + 1)
    return -n - np.mean((2 * i - 1) * (np.log(F) + np.log(1 - F[::-1])))


def gev_block(x, label):
    g = gev_fit_lmom(x); n = len(x)
    boot = []
    for _ in range(B_BOOT):
        xb = genextreme.rvs(c=g[2], loc=g[0], scale=g[1], size=n, random_state=RNG)
        try:
            gb = gev_fit_lmom(xb); boot.append([*gb, *[gev_q(1 - 1 / T, *gb) for T in TR]])
        except Exception:
            pass
    boot = np.array(boot)
    # GoF with parametric-bootstrap p-values (parameters re-estimated in every replicate)
    ks0 = stats.kstest(x, lambda v: gev_cdf(v, *g)).statistic; ad0 = ad_stat(x, lambda v: gev_cdf(v, *g))
    ksb, adb = [], []
    for _ in range(500):
        xb = genextreme.rvs(c=g[2], loc=g[0], scale=g[1], size=n, random_state=RNG); gb = gev_fit_lmom(xb)
        ksb.append(stats.kstest(xb, lambda v: gev_cdf(v, *gb)).statistic); adb.append(ad_stat(xb, lambda v: gev_cdf(v, *gb)))
    t3 = lmom_ratios(x, nmom=3)[2]
    res = dict(label=label, n=n, mu=g[0], sigma=g[1], k=g[2], xi_coles=-g[2], lskew=t3,
               ks=ks0, ks_p=np.mean(np.array(ksb) >= ks0), ad=ad0, ad_p=np.mean(np.array(adb) >= ad0))
    for j, nm in enumerate(["mu", "sigma", "k"]):
        res[f"{nm}_ci"] = np.percentile(boot[:, j], [5, 95]).tolist()
    for j, T in enumerate(TR):
        res[f"Q{T}"] = gev_q(1 - 1 / T, *g); res[f"Q{T}_ci"] = np.percentile(boot[:, 3 + j], [5, 95]).tolist()
    return g, res, boot


G_FULL, gev_full, boot_full = gev_block(x_all, "1981-2024 (design baseline)")
G_CAL, gev_cal, boot_cal = gev_block(x_cal, "1981-2005 (out-of-sample calibration)")
R["gev"] = [gev_full, gev_cal]
mk = mann_kendall(x_all); R["mann_kendall"] = mk


def mk_hamed_rao(x, alpha=0.05):
    """Modified Mann-Kendall (Hamed and Rao 1998): variance inflated by significant rank autocorrelations
    of the Sen-detrended series."""
    n = len(x); m = mann_kendall(x); r = stats.rankdata(x - m["sen"] * np.arange(n))
    r = r - r.mean(); acf = [np.sum(r[:-k] * r[k:]) / np.sum(r * r) for k in range(1, n - 1)]
    lim = norm.ppf(1 - alpha / 2) / np.sqrt(n)
    corr = 1 + 2 / (n * (n - 1) * (n - 2)) * sum((n - k) * (n - k - 1) * (n - k - 2) * a
                                                 for k, a in enumerate(acf, 1) if abs(a) > lim)
    var = n * (n - 1) * (2 * n + 5) / 18 * max(corr, 1e-6); z = (m["S"] - np.sign(m["S"])) / np.sqrt(var)
    return dict(z=z, p=2 * (1 - norm.cdf(abs(z))), var_factor=corr, rho1=acf[0])


R["mann_kendall_hr"] = mk_hamed_rao(x_all)
ols = stats.linregress(yrs, x_all); R["ols_trend"] = dict(slope=ols.slope, p=ols.pvalue)
nsg = ns_gev_fit(x_all, yrs - T0)
lr = 2 * (nsg["ns"]["ll"] - nsg["stat"]["ll"]); R["ns_gev_obs"] = dict(**nsg["ns"], LR=lr, p=1 - chi2.cdf(lr, 1))
# xi-sensitivity of Q100 (fix k on a grid, re-fit mu, sigma from L1, L2)
l1, l2 = lmom_ratios(x_all, nmom=2)[:2]
from scipy.special import gamma as Gm
xi_sens = []
for k in np.linspace(*gev_full["k_ci"], 7):
    sg = l2 * k / ((1 - 2 ** (-k)) * Gm(1 + k)); mu = l1 - sg * (1 - Gm(1 + k)) / k
    xi_sens.append(dict(k=k, xi=-k, Q100=gev_q(0.99, mu, sg, k), Q10=gev_q(0.9, mu, sg, k)))
R["xi_sensitivity"] = xi_sens
t1 = pd.DataFrame([{k: v for k, v in d.items() if not isinstance(v, list)} for d in [gev_full, gev_cal]]).set_index("label")
tab(t1, "T1_gev_parameters.csv")
print(f"GEV full: mu={G_FULL[0]:.2f} sig={G_FULL[1]:.2f} xi={-G_FULL[2]:.3f}  Q100={gev_full['Q100']:.1f} {np.round(gev_full['Q100_ci'],1)}")
print(f"MK p={mk['p']:.3f} sen={mk['sen']:.3f} | NS-GEV b={R['ns_gev_obs']['b']:.5f} LR p={R['ns_gev_obs']['p']:.3f}")

# ================================================================== 3. CMIP6 EVALUATION (1981-2014)
ob = obs[obs.year <= 2014].AMS_24hr.values
hist = cm[cm.scenario == "historical"].pivot_table(index="year", columns="model", values="pr_mm").loc[1981:2014]
t3o = lmom_ratios(ob, nmom=3)[2]
ev = []
for m in MODELS:
    xm = hist[m].values
    ev.append(dict(model=m, mean_bias_pct=100 * (xm.mean() / ob.mean() - 1), cv_ratio=(xm.std() / xm.mean()) / (ob.std() / ob.mean()),
                   lskew_diff=lmom_ratios(xm, nmom=3)[2] - t3o, q90_bias_pct=100 * (np.quantile(xm, .9) / np.quantile(ob, .9) - 1),
                   ks_shape=stats.ks_2samp(xm / xm.mean(), ob / ob.mean()).statistic))
ev = pd.DataFrame(ev).set_index("model")
idx_file = os.path.join(HERE, "model_indices_v6.csv"); obs_idx = os.path.join(HERE, "gee_chirps_indices.csv")
if os.path.exists(idx_file) and os.path.exists(obs_idx):
    mi = pd.read_csv(idx_file).set_index("model"); oi = pd.read_csv(obs_idx)
    for col in ["R10mm", "R20mm", "PRCPTOT"]:
        if col in mi and col in oi:
            ev[f"{col}_bias_pct"] = 100 * (mi[col] / oi[col].mean() - 1)
    pp = os.path.join(HERE, "gee_chirps_p95p99.csv")
    if os.path.exists(pp):
        p = pd.read_csv(pp).iloc[0]
        for q in ["p95", "p99"]:
            oc = [c for c in p.index if q in c.lower()]
            if oc and q.upper() in mi:
                ev[f"{q.upper()}_bias_pct"] = 100 * (mi[q.upper()] / p[oc[0]] - 1)
# composite score: shape/tail metrics only (mean bias is removed by the relative change factor)
score_cols = [c for c in ev.columns if c != "mean_bias_pct"]
err = pd.DataFrame({c: (ev[c] - (1 if c == "cv_ratio" else 0)).abs() for c in score_cols})
ev["score"] = (err / err.median()).mean(axis=1)
ev["rank"] = ev.score.rank().astype(int)
PERF = list(ev.sort_values("score").index[: max(3, int(np.ceil(len(MODELS) / 2)))])
ev["perf_subset"] = ev.index.isin(PERF)
tab(ev.sort_values("rank"), "T2_model_evaluation.csv")
R["model_eval"] = dict(score_cols=score_cols, perf_subset=PERF)

# ================================================================== 4. MULTIPLIER-LAW PARAMETERS
obs_mean = ob.mean()


def rw_by_model(s):
    return pd.Series({m: rw_sigma(s[m].dropna().values)[0] for m in s.columns})


def params_for(scen, models=None, bc="ratio"):
    s = model_series(cm, scen, bc=bc, obs_mean=obs_mean)
    if models is not None:
        s = s[models]
    fp = fit_params(s)
    return s, fp


P = {}
for sc in SCEN:
    s, fp = params_for(sc)
    rws = rw_by_model(s)
    # bootstrap over models: joint (mu_c, tau, s)
    bt = []
    cols = np.array(s.columns)
    for b in range(min(B_BOOT, 1000)):
        pick = RNG.choice(cols, len(cols), replace=True)
        ss = s[pick].copy(); ss.columns = [f"{c}__{i}" for i, c in enumerate(pick)]
        f = fit_params(ss); bt.append((f["mu_c"], f["tau"], np.median(rws[pick].values), f["alpha"]))
    bt = pd.DataFrame(bt, columns=["mu_c", "tau", "s", "alpha"])
    _, fpd = params_for(sc, bc="delta"); _, fpp = params_for(sc, models=PERF)
    P[sc] = dict(mu_c=fp["mu_c"], tau=fp["tau"], s=float(np.median(rws)), s_models=rws.to_dict(),
                 alpha=fp["alpha"], t=fp["t"].tolist(), V=fp["V"].tolist(), V_raw=fp["V_raw"].tolist(),
                 noise=fp["noise"].tolist(), Ybar=fp["Ybar"].tolist(),
                 lnY=fp["lnY"].tolist(), rss=dict(gbm=fp["rss_gbm"], drift=fp["rss_drift"], both=fp["rss_both"]),
                 boot=bt, mu_c_delta=fpd["mu_c"], tau_delta=fpd["tau"], mu_c_perf=fpp["mu_c"], tau_perf=fpp["tau"])
    print(f"{sc}: mu_c={fp['mu_c']:.5f} [{bt.mu_c.quantile(.05):.5f},{bt.mu_c.quantile(.95):.5f}] "
          f"tau={fp['tau']:.5f} s={P[sc]['s']:.4f} alpha={fp['alpha']:.2f} | delta-BC mu_c={fpd['mu_c']:.5f} perf mu_c={fpp['mu_c']:.5f}")

# observational volatility: local-level estimate + simulation-based 90% upper bound
s_obs, irr_obs, _ = rw_sigma(x_all)
obs_cv = x_all.std(ddof=1) / x_all.mean()
grid = [0.0, 0.01, 0.02, 0.03, 0.045, 0.06, 0.08, 0.10, 0.12, round(obs_cv, 3)]
sim_rows = []
for sg in grid:
    est = []
    for _ in range(int(os.environ.get("N_SIMVOL", 200))):
        lvl = np.exp(np.cumsum(sg * RNG.standard_normal(44)))
        est.append(rw_sigma(lvl * genextreme.rvs(c=G_FULL[2], loc=G_FULL[0], scale=G_FULL[1], size=44, random_state=RNG))[0])
    est = np.array(est)
    sim_rows.append(dict(s_true=sg, med=np.median(est), p5=np.quantile(est, .05), p95=np.quantile(est, .95),
                         p_le_obs=np.mean(est <= s_obs + 0.005)))
simv = pd.DataFrame(sim_rows)
# one-sided 90% upper confidence bound: largest s still giving P(s_hat <= observed) >= 0.10
pl = simv.p_le_obs.values
s_obs_u90 = float(np.interp(0.10, pl[::-1], simv.s_true.values[::-1])) if pl.min() < 0.10 else float(grid[-1])
p_v5 = float(simv.p_le_obs.iloc[-1])
R["volatility_obs"] = dict(s_hat=s_obs, s_upper90=s_obs_u90, irregular_sd=irr_obs, obs_cv=obs_cv,
                           p_shat_le_obs_if_s_eq_cv=p_v5, sim=simv.to_dict("records"))
tab(simv.set_index("s_true"), "T3b_volatility_detection_simulation.csv")
print(f"OBS local-level: s={s_obs:.4f} (90% upper {s_obs_u90:.3f}); irregular sd={irr_obs:.4f} vs CV={obs_cv:.4f}; P(s_hat<=obs | s=CV)={p_v5:.3f}")

t3 = []
for sc in SCEN:
    p = P[sc]; b = p["boot"]
    t3.append(dict(scenario=SCEN_LAB[sc], mu_c=p["mu_c"], mu_c_lo=b.mu_c.quantile(.05), mu_c_hi=b.mu_c.quantile(.95),
                   s_phys=p["s"], s_lo=b.s.quantile(.05), s_hi=b.s.quantile(.95),
                   tau_struct=p["tau"], tau_lo=b.tau.quantile(.05), tau_hi=b.tau.quantile(.95),
                   alpha=p["alpha"], E_Y_2050=np.exp(p["mu_c"] * (2050 - T0)), E_Y_2100=np.exp(p["mu_c"] * (2100 - T0)),
                   mu_c_deltaBC=p["mu_c_delta"], mu_c_perf=p["mu_c_perf"]))
t3 = pd.DataFrame(t3).set_index("scenario"); tab(t3, "T3_multiplier_parameters.csv")
R["params"] = {sc: {k: v for k, v in P[sc].items() if k != "boot"} for sc in SCEN}

# ================================================================== 5. DESIGN VALUES
def scale_c(D, beta=BETA):
    return (D / 24) ** beta


def dv(Tr, D, h, sc, gev=G_FULL, mu_c=None, s=None, beta=BETA):
    p = P[sc]; t = h - T0
    return design_value(Tr, t, gev, p["mu_c"] if mu_c is None else mu_c, p["s"] if s is None else s, 0.0, scale_c(D, beta))


def mc_design(Tr, D, h, sc, sources=("gev", "drift", "vol", "bc", "beta"), n=N_MC):
    """Monte Carlo of the design value: aleatory volatility s is inside the predictive
    quantile; epistemic sources sampled: GEV parameters (parametric bootstrap, xi unclipped),
    structural drift (mu_c,i ~ N(mu_c*, tau*^2) with (mu_c*,tau*) bootstrapped over models),
    volatility s (bootstrap over models), bias-correction choice (ratio vs delta), beta~U(.10,.40)."""
    p = P[sc]; b = p["boot"]; t = h - T0; out = np.empty(n)
    for i in range(n):
        g = tuple(boot_full[RNG.integers(len(boot_full)), :3]) if "gev" in sources else G_FULL
        j = RNG.integers(len(b))
        if "bc" in sources and RNG.random() < 0.5:
            mu0, tau0 = p["mu_c_delta"], p["tau_delta"]
        else:
            mu0, tau0 = (b.mu_c.iloc[j], b.tau.iloc[j]) if "drift" in sources else (p["mu_c"], p["tau"])
        mu = RNG.normal(mu0, tau0) if "drift" in sources else mu0
        s = b.s.iloc[j] if "vol" in sources else p["s"]
        beta = RNG.uniform(BETA_LO, BETA_HI) if ("beta" in sources and D != 24) else BETA
        out[i] = design_value(Tr, t, g, mu, s, 0.0, scale_c(D, beta))
    return out


rows = []
for sc in SCEN:
    for h in HORIZONS:
        for Tr in TR:
            rows.append(dict(scenario=SCEN_LAB[sc], horizon=h, Tr=Tr,
                             **{f"D{D}h": dv(Tr, D, h, sc) for D in DUR}))
nsidf = pd.DataFrame(rows)
stat_row = pd.DataFrame([dict(scenario="Stationary", horizon=2002, Tr=Tr,
                              **{f"D{D}h": gev_q(1 - 1 / Tr, *G_FULL) * scale_c(D) for D in DUR}) for Tr in TR])
tab(pd.concat([stat_row, nsidf]).set_index(["scenario", "horizon", "Tr"]), "T4a_design_depths_all_durations.csv")

# headline table: D=24h with 90% MC intervals and C(t)
head = []
for sc in SCEN:
    for h in HORIZONS:
        for Tr in [2, 10, 25, 100]:
            c = dv(Tr, 24, h, sc); mc = mc_design(Tr, 24, h, sc, n=N_MC // 2)
            q0 = gev_q(1 - 1 / Tr, *G_FULL)
            head.append(dict(scenario=SCEN_LAB[sc], horizon=h, Tr=Tr, stationary=q0, design=c,
                             lo90=np.quantile(mc, .05), hi90=np.quantile(mc, .95), Ct=c / q0,
                             closed_form=closed_form_design_value(Tr, h - T0, G_FULL, P[sc]["mu_c"], P[sc]["s"])))
head = pd.DataFrame(head); tab(head.set_index(["scenario", "horizon", "Tr"]), "T4_design_24h_with_intervals.csv")
R["closed_form_max_abs_err_pct"] = float((100 * (head.closed_form / head.design - 1)).abs().max())
print(head[head.Tr == 100].round(2).to_string(index=False))

# ================================================================== 6. BENCHMARKS
def cf_slice_gev(sc):
    """CMIP6 time-slice GEV change factors (quantile delta mapping on annual maxima)."""
    s = model_series(cm, sc); res = []
    for m in s.columns:
        gb = gev_fit_lmom(s[m].loc[1981:2024].values)
        for c, (a, b) in SLICES.items():
            gs = gev_fit_lmom(s[m].loc[a:b].values)
            for Tr in [10, 100]:
                res.append(dict(model=m, slice=c, Tr=Tr, CF=gev_q(1 - 1 / Tr, *gs) / gev_q(1 - 1 / Tr, *gb)))
    return pd.DataFrame(res)


tas_ok = os.path.exists(os.path.join(HERE, "cmip6_tas_v6.csv"))
tas = pd.read_csv(os.path.join(HERE, "cmip6_tas_v6.csv")) if tas_ok else None
bench = []
for sc in SCEN:
    cf = cf_slice_gev(sc)
    if tas_ok:
        tb = tas[tas.scenario == "historical"].groupby("model").tas_C.mean()
        dT = {c: (tas[(tas.scenario == sc) & tas.year.between(a, b)].groupby("model").tas_C.mean() - tb)
              for c, (a, b) in SLICES.items() if c in (2050, 2085)}
    for c, (a, b) in SLICES.items():
        t = (a + b) / 2 - T0
        for Tr in [10, 100]:
            q0 = gev_q(1 - 1 / Tr, *G_FULL); sub = cf[(cf.slice == c) & (cf.Tr == Tr)].CF
            row = dict(scenario=SCEN_LAB[sc], slice=f"{a}-{b}", centre=(a + b) / 2, Tr=Tr, stationary=q0,
                       bsidf=design_value(Tr, t, G_FULL, P[sc]["mu_c"], P[sc]["s"]),
                       cmip6_slice_median=q0 * sub.median(), cmip6_slice_p10=q0 * sub.quantile(.1),
                       cmip6_slice_p90=q0 * sub.quantile(.9),
                       ns_gev_obs=gev_q(1 - 1 / Tr, nsg["ns"]["mu0"] * (1 + nsg["ns"]["b"] * t),
                                        nsg["ns"]["sig0"] * (1 + nsg["ns"]["b"] * t), nsg["ns"]["k"]))
            if tas_ok and c in dT:
                row.update(dT_median=dT[c].median(), cc_median=q0 * 1.07 ** dT[c].median(),
                           cc_p10=q0 * 1.07 ** dT[c].quantile(.1), cc_p90=q0 * 1.07 ** dT[c].quantile(.9))
            bench.append(row)
bench = pd.DataFrame(bench); bench["bsidf_vs_slice_pct"] = 100 * (bench.bsidf / bench.cmip6_slice_median - 1)
tab(bench.set_index(["scenario", "slice", "Tr"]), "T5_benchmarks.csv")
print(bench[["scenario", "slice", "Tr", "bsidf", "cmip6_slice_median", "cmip6_slice_p10", "cmip6_slice_p90", "ns_gev_obs"]].round(1).to_string(index=False))

# ================================================================== 7. OUT-OF-SAMPLE TEST (1981-2005 -> 2006-2024)
T0c = 1993.0


def crps_num(cdf, xo, lo=0, hi=400, n=3000):
    g = np.linspace(lo, hi, n); F = cdf(g); return np.trapezoid((F - (g >= xo)) ** 2, g)


oos = []
nsc = ns_gev_fit(x_cal, obs[obs.year <= 2005].year.values - T0c)
for y, xo in zip(obs[obs.year >= 2006].year, x_val):
    t = y - T0c
    mods = {"Stationary GEV": lambda v: gev_cdf(v, *G_CAL),
            "NS-GEV (obs trend)": lambda v: gev_cdf(v, nsc["ns"]["mu0"] * (1 + nsc["ns"]["b"] * t), nsc["ns"]["sig0"] * (1 + nsc["ns"]["b"] * t), nsc["ns"]["k"]),
            "BS-IDF SSP2-4.5": lambda v: predictive_cdf(v, t, G_CAL, P["ssp245"]["mu_c"], P["ssp245"]["s"]),
            "BS-IDF SSP5-8.5": lambda v: predictive_cdf(v, t, G_CAL, P["ssp585"]["mu_c"], P["ssp585"]["s"])}
    for nm, F in mods.items():
        e = 1e-3; dens = (F(xo + e) - F(xo - e)) / (2 * e)
        oos.append(dict(year=y, model=nm, obs=xo, PIT=float(np.atleast_1d(F(xo))[0]),
                        logscore=-np.log(max(float(np.atleast_1d(dens)[0]), 1e-12)), CRPS=crps_num(lambda g: np.atleast_1d(F(g)), xo)))
oos = pd.DataFrame(oos)
oos_sum = oos.groupby("model").agg(mean_logscore=("logscore", "mean"), mean_CRPS=("CRPS", "mean"),
                                    PIT_mean=("PIT", "mean"))
oos_sum["PIT_KS_p"] = oos.groupby("model").PIT.apply(lambda u: stats.kstest(u, "uniform").pvalue)
base = oos[oos.model == "Stationary GEV"].set_index("year")
oos_sum["CRPS_diff_vs_stat_p"] = [np.nan if m == "Stationary GEV" else
                                  stats.wilcoxon(oos[oos.model == m].set_index("year").CRPS - base.CRPS).pvalue
                                  for m in oos_sum.index]
tab(oos_sum, "T8_out_of_sample_scores.csv"); R["oos"] = oos_sum.reset_index().to_dict("records")
print(oos_sum.round(4).to_string())

# signal-to-noise / time of emergence (relative change of the 30-yr mean AMS)
noise30 = obs_cv * np.sqrt(1 / 30 + 1 / 44)
toe = {sc: dict(ToE_SN1=T0 + noise30 / P[sc]["mu_c"], ToE_SN2=T0 + 2 * noise30 / P[sc]["mu_c"]) for sc in SCEN}
n_ = 44; se_slope_rel = obs_cv * np.sqrt(12 / (n_ * (n_**2 - 1)))
R["snr"] = dict(noise_30yr_mean=noise30, toe=toe, min_detectable_rel_trend_44yr=2 * se_slope_rel,
                obs_rel_trend=ols.slope / x_all.mean(), n_val=19,
                min_detectable_rel_trend_19yr=2 * obs_cv * np.sqrt(12 / (19 * (19**2 - 1))))

# ================================================================== 8. SUB-DAILY EXPONENT SENSITIVITY
bs = []
for beta in [0.10, 0.15, 0.20, 0.25, 0.30, 1 / 3, 0.35, 0.40]:
    for D in [1, 3]:
        for Tr in [10, 100]:
            bs.append(dict(beta=round(beta, 3), D=D, Tr=Tr, stationary=gev_q(1 - 1 / Tr, *G_FULL) * scale_c(D, beta),
                           ssp585_2050=dv(Tr, D, 2050, "ssp585", beta=beta)))
bs = pd.DataFrame(bs); ref = bs[bs.beta == 0.333].set_index(["D", "Tr"]).stationary
bs["pct_vs_beta_1_3"] = [100 * (r.stationary / ref.loc[(r.D, r.Tr)] - 1) for r in bs.itertuples()]
tab(bs.set_index(["beta", "D", "Tr"]), "T6_subdaily_beta_sensitivity.csv")

# ================================================================== 9. UNCERTAINTY DECOMPOSITION
dec = []
for h in [2050, 2100]:
    for D in [24, 1]:
        full = mc_design(100, D, h, "ssp585", n=N_MC // 2); vt = full.var()
        row = dict(horizon=h, D=D, central=dv(100, D, h, "ssp585"), lo90=np.quantile(full, .05), hi90=np.quantile(full, .95))
        parts = {}
        for src in ["gev", "drift", "vol", "bc", "beta"]:
            if src == "beta" and D == 24:
                parts[src] = 0.0; continue
            parts[src] = mc_design(100, D, h, "ssp585", sources=(src,), n=N_MC // 4).var()
        tot = sum(parts.values())
        row.update({f"share_{k}": 100 * v / tot for k, v in parts.items()}); row["var_total"] = vt
        dec.append(row)
dec = pd.DataFrame(dec); tab(dec.set_index(["horizon", "D"]), "T7_uncertainty_decomposition.csv")
print(dec.round(1).to_string(index=False))

# ================================================================== 9b. OBSERVATIONAL PRODUCT SENSITIVITY
prod_f0 = os.path.join(HERE, "gee_ams_products.csv")
if os.path.exists(prod_f0):
    prd = pd.read_csv(prod_f0).dropna(subset=["AMS_mm"])
    prd = prd[prd.n_days >= 365]
    C50 = dv(100, 24, 2050, "ssp585") / gev_q(0.99, *G_FULL)
    ps = []
    for nm, g in prd.groupby("product"):
        g = g.sort_values("year"); x = g.AMS_mm.values
        if len(x) < 20:
            continue
        gg = gev_fit_lmom(x); m_ = mann_kendall(x)
        bq = []
        for _ in range(500):
            xb = genextreme.rvs(c=gg[2], loc=gg[0], scale=gg[1], size=len(x), random_state=RNG)
            try:
                bq.append(gev_q(0.99, *gev_fit_lmom(xb)))
            except Exception:
                pass
        j = obs.set_index("year").AMS_24hr.reindex(g.year).values
        ps.append(dict(product=nm.replace("_", " "), period=f"{g.year.min()}–{g.year.max()}", n=len(x), mean=x.mean(),
                       cv=x.std(ddof=1) / x.mean(), xi=-gg[2], Q10=gev_q(0.9, *gg), Q100=gev_q(0.99, *gg),
                       Q100_lo=np.percentile(bq, 5), Q100_hi=np.percentile(bq, 95), mk_p=m_["p"], sen=m_["sen"],
                       sen_pct=100 * m_["sen"] / x.mean(), r_v2=np.corrcoef(x, j)[0, 1],
                       Q100_2050_ssp585=gev_q(0.99, *gg) * C50))
    PS = pd.DataFrame(ps).set_index("product"); tab(PS, "T2b_product_sensitivity.csv")
    R["product_sensitivity"] = PS.reset_index().to_dict("records")
    print(PS.round(3).to_string())

# ================================================================== 9c. VARIANCE-BASED (SOBOL) SENSITIVITY
GROUPS = {"gev": [0], "drift": [1, 2], "vol": [3], "bc": [4], "beta": [5]}


def _f_sobol(u, Tr, D, h, sc):
    p = P[sc]; b = p["boot"]; t = h - T0
    g = tuple(boot_full[min(int(u[0] * len(boot_full)), len(boot_full) - 1), :3])
    j = min(int(u[1] * len(b)), len(b) - 1); z = norm.ppf(np.clip(u[2], 1e-6, 1 - 1e-6))
    if u[4] < 0.5:
        mu = p["mu_c_delta"] + p["tau_delta"] * z
    else:
        mu = b.mu_c.iloc[j] + b.tau.iloc[j] * z
    s = b.s.iloc[min(int(u[3] * len(b)), len(b) - 1)]
    beta = BETA_LO + (BETA_HI - BETA_LO) * u[5] if D != 24 else BETA
    return design_value(Tr, t, g, mu, s, 0.0, scale_c(D, beta))


def sobol(Tr, D, h, sc, N=int(os.environ.get("N_SOBOL", 8000))):
    """Saltelli (2010) pick-freeze design with Saltelli first-order and Jansen total-effect estimators,
    grouped inputs, independent sampling of every source (no shared bootstrap index)."""
    k = 6; A = RNG.random((N, k)); Bm = RNG.random((N, k))
    fA = np.array([_f_sobol(u, Tr, D, h, sc) for u in A]); fB = np.array([_f_sobol(u, Tr, D, h, sc) for u in Bm])
    V = np.var(np.r_[fA, fB]); out = {}
    bidx = [RNG.integers(0, N, N) for _ in range(300)]
    for gname, cols in GROUPS.items():
        if gname == "beta" and D == 24:
            out[gname] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0); continue
        ABi = A.copy(); ABi[:, cols] = Bm[:, cols]
        fABi = np.array([_f_sobol(u, Tr, D, h, sc) for u in ABi])
        est = lambda i: (np.mean(fB[i] * (fABi[i] - fA[i])) / np.var(np.r_[fA[i], fB[i]]),
                         0.5 * np.mean((fA[i] - fABi[i]) ** 2) / np.var(np.r_[fA[i], fB[i]]))
        bs_ = np.array([est(i) for i in bidx]); s1, st = est(np.arange(N))
        out[gname] = (float(s1), float(st), *np.quantile(bs_[:, 0], [.05, .95]), *np.quantile(bs_[:, 1], [.05, .95]))
    return out, V


sob = []
for h in [2050, 2100]:
    for D in [24, 1]:
        o, V = sobol(100, D, h, "ssp585")
        sob.append(dict(horizon=h, D=D, var=V, **{f"S1_{k}": v[0] for k, v in o.items()},
                        **{f"ST_{k}": v[1] for k, v in o.items()}, sum_S1=sum(v[0] for v in o.values()),
                        **{f"S1lo_{k}": v[2] for k, v in o.items()}, **{f"S1hi_{k}": v[3] for k, v in o.items()},
                        **{f"STlo_{k}": v[4] for k, v in o.items()}, **{f"SThi_{k}": v[5] for k, v in o.items()}))
sob = pd.DataFrame(sob); tab(sob.set_index(["horizon", "D"]), "T9_sobol_indices.csv"); R["sobol"] = sob.to_dict("records")
print(sob.round(3).to_string(index=False))

# ================================================================== 9d. PROFILE LIKELIHOOD FOR THE OBSERVED VOLATILITY
import statsmodels.api as sm
_ucm = sm.tsa.UnobservedComponents(np.log(x_all), level="local linear deterministic trend")
_full = _ucm.fit(disp=False, method="lbfgs", maxiter=1000); ll_max = _full.llf
prof = []
for sg in np.r_[0.0, np.linspace(0.005, 0.15, 30)]:
    try:
        r_ = _ucm.fit_constrained({"sigma2.level": float(sg ** 2)}, disp=False)
        prof.append(dict(s=sg, ll=r_.llf))
    except Exception:
        pass
prof = pd.DataFrame(prof); prof["dev"] = 2 * (ll_max - prof.ll)
# boundary case (s=0 on the edge): 50:50 chi2_0/chi2_1 mixture -> one-sided 90% critical value 1.642, 95% 2.706
crit90, crit95 = chi2.ppf(0.80, 1), chi2.ppf(0.90, 1)
up = prof[prof.s > 0]
s_pl90 = float(np.interp(crit90, up.dev.values, up.s.values)) if up.dev.max() > crit90 else np.nan
s_pl95 = float(np.interp(crit95, up.dev.values, up.s.values)) if up.dev.max() > crit95 else np.nan
R["volatility_profile"] = dict(s_hat=float(np.sqrt(max(dict(zip(_ucm.param_names, _full.params))["sigma2.level"], 0))),
                               s_upper90=s_pl90, s_upper95=s_pl95, dev_at_cv=float(np.interp(obs_cv, prof.s, prof.dev)),
                               profile=prof.to_dict("records"))
print(f"profile likelihood: s upper90={s_pl90:.4f} upper95={s_pl95:.4f}")

# ================================================================== 9e. DESIGN LIFE LEVEL (Rootzen and Katz 2013)
def dll(T, start, life, sc, D=24, M=int(os.environ.get("N_DLL", 6000)), drift_unc=False, seed=7):
    """Level u with P(at least one exceedance in [start, start+life-1]) = 1-(1-1/T)^life, averaging over
    simulated GBM climate paths (path dependence kept), optional structural drift uncertainty."""
    rng = np.random.default_rng(seed); p = P[sc]; yrs_ = np.arange(start, start + life)
    tt = np.r_[0.0, yrs_ - T0]; dt = np.diff(tt)
    mu = rng.normal(p["mu_c"], p["tau"], M) if drift_unc else np.full(M, p["mu_c"])
    inc = (mu[:, None] - 0.5 * p["s"] ** 2) * dt[None, :] + p["s"] * np.sqrt(dt)[None, :] * rng.standard_normal((M, len(dt)))
    Y = np.exp(np.cumsum(inc, axis=1)); c = scale_c(D); mu0, sg0, k0 = G_FULL
    target = (1 - 1 / T) ** life
    def nonexc(u):
        return np.mean(np.exp(np.sum(np.log(np.clip(gev_cdf(u / (c * Y), mu0, sg0, k0), 1e-300, 1)), axis=1)))
    from scipy.optimize import brentq
    return brentq(lambda u: nonexc(u) - target, 1, 1000, xtol=1e-4)


dl = []
for sc in SCEN:
    for start, life in [(2025, 30), (2025, 50), (2050, 50)]:
        for T in [25, 100]:
            dl.append(dict(scenario=SCEN_LAB[sc], design_life=f"{start}-{start + life - 1}", T=T,
                           risk_pct=100 * (1 - (1 - 1 / T) ** life), stationary=gev_q(1 - 1 / T, *G_FULL),
                           DLL=dll(T, start, life, sc), DLL_drift_unc=dll(T, start, life, sc, drift_unc=True),
                           annual_at_end=dv(T, 24, start + life - 1, sc), annual_at_mid=dv(T, 24, start + life // 2, sc)))
dl = pd.DataFrame(dl); tab(dl.set_index(["scenario", "design_life", "T"]), "T10_design_life_levels.csv")
R["dll"] = dl.to_dict("records"); print(dl.round(1).to_string(index=False))

# ================================================================== 9f. TESTS OF THE CONSTANT-CV / FIXED-SHAPE ASSUMPTION
from scipy.optimize import minimize


def ns_gev_general(x, t, free_scale=False, free_shape=False):
    """GEV MLE: mu=mu0(1+b t); sig=sig0(1+c t) (c=b unless free_scale); k=k0+k1 t (k1=0 unless free_shape)."""
    def nll(th):
        mu0, ls, k0, b = th[:4]; i = 4
        c = th[i] if free_scale else b; i += free_scale
        k1 = th[i] if free_shape else 0.0
        fm, fs = 1 + b * t, 1 + c * t
        if np.any(fm <= 0) or np.any(fs <= 0):
            return 1e10
        ll = genextreme.logpdf(x, c=k0 + k1 * t, loc=mu0 * fm, scale=np.exp(ls) * fs)
        return -np.sum(ll) if np.all(np.isfinite(ll)) else 1e10
    base = ns_gev_fit(x, t)["ns"]; best = None
    for j in range(6):
        th0 = [base["mu0"], np.log(base["sig0"]), base["k"], base["b"]] + [base["b"] + 0.002 * (j - 2)] * free_scale + [0.001 * (j - 2)] * free_shape
        r_ = minimize(nll, th0, method="Nelder-Mead", options=dict(maxiter=60000, xatol=1e-9, fatol=1e-11))
        if best is None or r_.fun < best.fun:
            best = r_
    return -best.fun, best.x


tt_obs = yrs - T0
ll_cv = R["ns_gev_obs"]["ll"]
ll_sc, th_sc = ns_gev_general(x_all, tt_obs, free_scale=True)
ll_sh, th_sh = ns_gev_general(x_all, tt_obs, free_shape=True)
cvt = dict(ll_constCV=ll_cv, ll_free_scale=ll_sc, LR_scale=2 * (ll_sc - ll_cv), p_scale=1 - chi2.cdf(max(2 * (ll_sc - ll_cv), 0), 1),
           b_loc=th_sc[3], c_scale=th_sc[4], ll_free_shape=ll_sh, LR_shape=2 * (ll_sh - ll_cv),
           p_shape=1 - chi2.cdf(max(2 * (ll_sh - ll_cv), 0), 1), k1=th_sh[4])
# CMIP6: change of L-CV and L-skewness of annual maxima between baseline and late century, per model
lrows = []
for sc in SCEN:
    s_ = model_series(cm, sc)
    for m in s_.columns:
        b0 = lmom_ratios(s_[m].loc[1981:2024].dropna().values, nmom=3); 
        for c_, (a, bb) in [(2050, (2036, 2065)), (2085, (2071, 2100))]:
            l = lmom_ratios(s_[m].loc[a:bb].dropna().values, nmom=3)
            lrows.append(dict(scenario=SCEN_LAB[sc], slice=c_, model=m, lcv_ratio=(l[1] / l[0]) / (b0[1] / b0[0]), dlskew=l[2] - b0[2]))
lr_ = pd.DataFrame(lrows)
# sampling reference: same statistic between two halves of a stationary GEV of equal lengths
ref_ratio = []
for _ in range(2000):
    a_ = genextreme.rvs(c=G_FULL[2], loc=G_FULL[0], scale=G_FULL[1], size=44, random_state=RNG)
    b_ = genextreme.rvs(c=G_FULL[2], loc=G_FULL[0], scale=G_FULL[1], size=30, random_state=RNG)
    la_, lb_ = lmom_ratios(a_, nmom=2), lmom_ratios(b_, nmom=2); ref_ratio.append((lb_[1] / lb_[0]) / (la_[1] / la_[0]))
ref_ratio = np.array(ref_ratio)
lsum = lr_.groupby(["scenario", "slice"]).agg(n=("model", "size"), lcv_ratio_median=("lcv_ratio", "median"),
                                               lcv_up_frac=("lcv_ratio", lambda v: np.mean(v > 1)),
                                               dlskew_median=("dlskew", "median"))
lsum["wilcoxon_p_lcv"] = lr_.groupby(["scenario", "slice"]).lcv_ratio.apply(lambda v: stats.wilcoxon(np.log(v)).pvalue)
lsum["wilcoxon_p_lskew"] = lr_.groupby(["scenario", "slice"]).dlskew.apply(lambda v: stats.wilcoxon(v).pvalue)
lsum["single_model_noise_p90"] = np.quantile(ref_ratio, 0.9)
tab(lsum, "T11_constant_cv_tests.csv"); cvt["cmip6"] = lsum.reset_index().to_dict("records")
cf85 = cf_slice_gev("ssp585"); cfw = cf85.pivot_table(index=["model", "slice"], columns="Tr", values="CF").reset_index()
cvt["cf100_over_cf10_median_2085"] = float((cfw[cfw.slice == 2085][100] / cfw[cfw.slice == 2085][10]).median())
cvt["cf100_over_cf10_frac_gt1_2085"] = float(((cfw[cfw.slice == 2085][100] / cfw[cfw.slice == 2085][10]) > 1).mean())
R["constant_cv"] = cvt; print(lsum.round(3).to_string()); print({k: v for k, v in cvt.items() if k != "cmip6"})

# ================================================================== 9g. HAWKINS-SUTTON PARTITION OF THE 30-YR MEAN CHANGE
hs = []
for j, c_ in enumerate(SLICES):
    I = np.mean([P[sc]["noise"][j] for sc in SCEN]); M_ = np.mean([P[sc]["V"][j] for sc in SCEN])
    Sv = np.var([np.log(P[sc]["Ybar"][j]) for sc in SCEN])
    tot = I + M_ + Sv
    hs.append(dict(slice=c_, internal=100 * I / tot, model=100 * M_ / tot, scenario=100 * Sv / tot,
                   model_raw_pct=100 * np.mean([P[sc]["V_raw"][j] for sc in SCEN]) / tot))
hs = pd.DataFrame(hs).set_index("slice"); tab(hs, "T12_hawkins_sutton_partition.csv"); R["hawkins_sutton"] = hs.reset_index().to_dict("records")
print(hs.round(1).to_string())

# ================================================================== 9h. AREAL REDUCTION (indicative)
arf_f = os.path.join(HERE, "cmip6_arf.csv")
if os.path.exists(arf_f):
    af = pd.read_csv(arf_f)
    R["arf_cmip6"] = dict(n=len(af), median=float(af.ARF_mean.median()), p10=float(af.ARF_mean.quantile(.1)),
                          p90=float(af.ARF_mean.quantile(.9)), q100_median=float(af.ARF_Q100.median()))
obsarf_f = os.path.join(HERE, "gee_chirps_basinmean_ams.csv")
if os.path.exists(obsarf_f):
    oa = pd.read_csv(obsarf_f).merge(obs[["year", "AMS_24hr"]], on="year")
    colm = [c for c in oa.columns if c.lower().startswith("ams_basinmean") or c.lower() == "basin_mean_max"][0]
    R["arf_chirps"] = dict(n=len(oa), ARF_mean=float(oa[colm].mean() / oa.AMS_24hr.mean()),
                           ARF_Q100=float(gev_q(.99, *gev_fit_lmom(oa[colm].values)) / gev_q(.99, *G_FULL)))

# ================================================================== 10. FIGURES
# Fig 1 location map (north arrow, degree-minute graticule, scale bar)
from maputils import dm_axes, north_arrow, scale_bar
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
try:
    import geopandas as gpd
    basin = gpd.read_file(os.path.join(HERE, "ogun_hybas6_1060960280.geojson"))
    ctry = gpd.read_file(os.path.join(HERE, "ne_10m_admin_0_countries.geojson"))
    st = gpd.read_file(os.path.join(HERE, "ne_10m_admin_1_states_provinces.geojson"))
    riv = gpd.read_file(os.path.join(HERE, "ne_10m_rivers_lake_centerlines.geojson"))
    pl = gpd.read_file(os.path.join(HERE, "ne_10m_populated_places_simple.geojson"))
    fig, ax = plt.subplots(1, 2, figsize=(10, 5.2), gridspec_kw=dict(width_ratios=[1.15, 1]))
    nga = ctry[ctry.ADMIN == "Nigeria"]
    ctry.cx[-20:20, -2:20].plot(ax=ax[0], color="#f0efec", edgecolor="#b8b7b1", lw=0.4)
    nga.plot(ax=ax[0], color="#e4e3df", edgecolor=INK2, lw=0.9)
    basin.plot(ax=ax[0], color=C[0], edgecolor=C[0])
    ax[0].add_patch(plt.Rectangle((2.3, 6.2), 2.3, 3.1, fill=False, edgecolor=C[7], lw=1.1))
    ax[0].set_xlim(-18, 16); ax[0].set_ylim(2, 20)
    for nm_, xy in {"NIGERIA": (7.3, 10.3), "BENIN": (1.3, 10.6), "GHANA": (-2.2, 8.0), "NIGER": (7.0, 16.5),
                    "MALI": (-3.5, 16.5), "Gulf of Guinea": (-3.0, 3.2)}.items():
        ax[0].text(*xy, nm_, fontsize=7 if nm_ != "Gulf of Guinea" else 7.5, color=INK2, style="italic" if "Gulf" in nm_ else "normal")
    dm_axes(ax[0], step_deg=5); north_arrow(ax[0], x=0.93, y=0.82); scale_bar(ax[0], 1000, x0=0.05, y0=0.06)
    ax[0].set_title("(a) Location in West Africa", loc="left", fontsize=9)
    stv = st[st.admin.isin(["Nigeria"])].cx[1.5:5.5, 5.8:9.8]
    ctry[ctry.ADMIN == "Benin"].plot(ax=ax[1], color="#f0efec", edgecolor="#b8b7b1", lw=0.5)
    stv.plot(ax=ax[1], color="white", edgecolor="#b8b7b1", lw=0.6)
    basin.plot(ax=ax[1], color="#cde2fb", edgecolor=C[0], lw=1.3)
    riv.clip(basin).plot(ax=ax[1], color="#256abf", lw=1.0)
    for r_ in stv.itertuples():
        rp = r_.geometry.representative_point()
        if 2.35 < rp.x < 4.55 and 6.25 < rp.y < 9.25:
            ax[1].text(rp.x, rp.y, r_.name.upper(), fontsize=6.5, color="#8a8984", ha="center", style="italic")
    ax[1].text(2.45, 7.25, "BENIN", fontsize=7, color="#8a8984", style="italic")
    cities = pl[pl.name.isin(["Lagos", "Abeokuta", "Ibadan", "Oyo", "Ogbomosho", "Saki", "Iseyin"])].drop_duplicates("name")
    cities.plot(ax=ax[1], color=INK, markersize=12, zorder=5)
    for r_ in cities.itertuples():
        ax[1].annotate(r_.name, (r_.geometry.x, r_.geometry.y), xytext=(3, 3), textcoords="offset points", fontsize=7, color=INK)
    ax[1].set_xlim(2.3, 4.6); ax[1].set_ylim(6.2, 9.3)
    dm_axes(ax[1], step_deg=0.5); north_arrow(ax[1], x=0.9, y=0.84); scale_bar(ax[1], 50, x0=0.66, y0=0.10)
    ax[1].legend(handles=[Patch(facecolor="#cde2fb", edgecolor=C[0], label="Ogun River Basin (22,165 km²)"),
                          Line2D([], [], color="#256abf", lw=1.0, label="Main river"),
                          Line2D([], [], color="#b8b7b1", lw=0.8, label="State boundary"),
                          Line2D([], [], marker="o", ls="", color=INK, ms=4, label="Town")],
                 loc="upper left", fontsize=6.5, frameon=True, facecolor="white", edgecolor="#d8d7d2")
    ax[1].set_title("(b) Ogun River Basin", loc="left", fontsize=9)
    savefig(fig, "Fig1_location_map.png")
except Exception as e:
    print("map skipped:", e)

# Spatial maps of CMIP6 relative change (0.25 deg, ensemble median, stippled where >= 80% of models agree on sign)
try:
    import xarray as xr, geopandas as gpd
    from shapely.geometry import box as _box
    sp_f = os.path.join(HERE, "cmip6_spatial_change.nc")
    if os.path.exists(sp_f):
        dsx = xr.open_dataset(sp_f); basin = gpd.read_file(os.path.join(HERE, "ogun_hybas6_1060960280.geojson"))
        lon = dsx.lon.values; lon = np.where(lon > 180, lon - 360, lon); lat = dsx.lat.values
        from extract_cmip6_ams import weights as _w
        BM = _w(lat, lon, basin.geometry.iloc[0]) > 0          # cells intersecting the basin
        fig, ax = plt.subplots(2, 2, figsize=(6.4, 8.4), sharex=True, sharey=True, gridspec_kw=dict(wspace=0.08, hspace=0.16))
        meds = [dsx[v].values[BM] for v in dsx.data_vars if v.endswith("median")]
        vmin_ = min(float(np.nanmin(m)) for m in meds) * 100; vmax = max(float(np.nanmax(m)) for m in meds) * 100
        from matplotlib.colors import TwoSlopeNorm, Normalize
        # magnitude -> sequential scale when every basin cell changes in the same direction; diverging otherwise
        if vmin_ >= 0:
            norm_, cmap_ = Normalize(vmin=0, vmax=np.ceil(vmax / 5) * 5), "YlGnBu"
        else:
            norm_, cmap_ = TwoSlopeNorm(vmin=min(vmin_, -5), vcenter=0, vmax=vmax), "BrBG"
        for i, sc in enumerate(SCEN):
            for j, w in enumerate(["2036-2065", "2071-2100"]):
                a = ax[i, j]; med = dsx[f"{sc}_{w}_median"].values * 100; ag = dsx[f"{sc}_{w}_agree"].values
                im = a.pcolormesh(lon, lat, med, cmap=cmap_, norm=norm_, shading="nearest")
                ys, xs = np.where(BM & ((ag >= 0.8) | (ag <= 0.2)))
                a.plot(lon[xs], lat[ys], ".", color="#0b0b0b", ms=2.2)
                basin.boundary.plot(ax=a, color="#0b0b0b", lw=1.1)
                a.set_xlim(2.4, 4.45); a.set_ylim(6.3, 9.15); dm_axes(a, step_deg=0.5, fontsize=6.5)
                a.set_title(f"({'abcd'[2 * i + j]}) {SCEN_LAB[sc]}, {w}", loc="left", fontsize=8.5)
                north_arrow(a, x=0.88, y=0.84, size=0.07); scale_bar(a, 50, x0=0.06, y0=0.05, fontsize=6)
        cb = fig.colorbar(im, ax=ax, shrink=0.8, orientation="horizontal", pad=0.05, aspect=35)
        cb.set_label(f"Ensemble-median change in 30-year mean annual maximum daily rainfall (%), relative to 1981–2024\n"
                     f"Dots: ≥80% of the {dsx.attrs.get('ssp585_2071-2100_n', len(MODELS))} models agree on the sign of change", fontsize=7)
        savefig(fig, "Map_cmip6_change.png")
        R["spatial_change"] = {k: dict(min=float(np.nanmin(dsx[k].values[BM])), max=float(np.nanmax(dsx[k].values[BM])),
                                       mean=float(np.nanmean(dsx[k].values[BM])), n_cells=int(BM.sum())) for k in dsx.data_vars}
except Exception as e:
    print("spatial change map skipped:", e)

# Pixel-level CHIRPS maps: mean annual rainfall, mean AMS, stationary and 2050 100-year depth
try:
    ams_tif, map_tif = os.path.join(HERE, "ogun_chirps_ams_stack.tif"), os.path.join(HERE, "ogun_chirps_map.tif")
    if os.path.exists(ams_tif) and os.path.exists(map_tif):
        import rasterio, geopandas as gpd
        from rasterio.features import geometry_mask
        basin = gpd.read_file(os.path.join(HERE, "ogun_hybas6_1060960280.geojson"))
        with rasterio.open(ams_tif) as r1:
            stack = r1.read().astype(float); tr = r1.transform; H, W = r1.height, r1.width
        with rasterio.open(map_tif) as r2:
            MAPr = r2.read(1).astype(float)
        inside = ~geometry_mask(basin.geometry, (H, W), tr, all_touched=True)
        q100 = np.full((H, W), np.nan); xi_px = np.full((H, W), np.nan)
        for i_ in range(H):
            for j_ in range(W):
                if inside[i_, j_]:
                    xs_ = stack[:, i_, j_]
                    if np.all(np.isfinite(xs_)) and xs_.std() > 0:
                        gp = gev_fit_lmom(xs_); q100[i_, j_] = gev_q(0.99, *gp); xi_px[i_, j_] = -gp[2]
        C50_585 = dv(100, 24, 2050, "ssp585") / gev_q(0.99, *G_FULL); C50_245 = dv(100, 24, 2050, "ssp245") / gev_q(0.99, *G_FULL)
        meanams = np.where(inside, stack.mean(0), np.nan); MAPm = np.where(inside, MAPr, np.nan)
        xs_c = tr.c + tr.a * (np.arange(W) + 0.5); ys_c = tr.f + tr.e * (np.arange(H) + 0.5)
        panels = [("(a) Mean annual rainfall (mm/yr)", MAPm, "YlGnBu"),
                  ("(b) Mean annual maximum daily rainfall (mm)", meanams, "YlGnBu"),
                  ("(c) Stationary 100-yr, 24-h depth (mm)", q100, "YlOrRd"),
                  ("(d) 100-yr, 24-h depth in 2050, SSP5-8.5 (mm)", q100 * C50_585, "YlOrRd")]
        fig, ax = plt.subplots(2, 2, figsize=(8.4, 9.2), sharex=True, sharey=True)
        lims = (np.nanmin(q100), np.nanmax(q100 * C50_585))
        for k, (tt, Z, cm_) in enumerate(panels):
            a = ax.flat[k]
            kw = dict(vmin=lims[0], vmax=lims[1]) if k >= 2 else {}
            im = a.pcolormesh(xs_c, ys_c, Z, cmap=cm_, shading="nearest", **kw)
            basin.boundary.plot(ax=a, color="#0b0b0b", lw=1.0)
            a.set_xlim(2.4, 4.45); a.set_ylim(6.3, 9.15); dm_axes(a, step_deg=0.5, fontsize=6.5)
            a.set_title(tt, loc="left", fontsize=8.2); north_arrow(a, x=0.88, y=0.84, size=0.07); scale_bar(a, 50, x0=0.06, y0=0.05, fontsize=6)
            cb = fig.colorbar(im, ax=a, shrink=0.8, pad=0.02); cb.ax.tick_params(labelsize=6.5)
        savefig(fig, "Map_rainfall_climatology.png")
        lat_px = np.repeat(ys_c[:, None], W, 1)
        R["pixel_maps"] = dict(MAP_min=float(np.nanmin(MAPm)), MAP_max=float(np.nanmax(MAPm)), MAP_mean=float(np.nanmean(MAPm)),
                               AMS_min=float(np.nanmin(meanams)), AMS_max=float(np.nanmax(meanams)),
                               Q100_min=float(np.nanmin(q100)), Q100_max=float(np.nanmax(q100)), Q100_median=float(np.nanmedian(q100)),
                               Q100_2050_585_min=float(np.nanmin(q100 * C50_585)), Q100_2050_585_max=float(np.nanmax(q100 * C50_585)),
                               xi_median=float(np.nanmedian(xi_px)), n_px=int(np.isfinite(q100).sum()),
                               r_lat_MAP=float(np.corrcoef(lat_px[np.isfinite(MAPm)], MAPm[np.isfinite(MAPm)])[0, 1]),
                               r_lat_Q100=float(np.corrcoef(lat_px[np.isfinite(q100)], q100[np.isfinite(q100)])[0, 1]))
except Exception as e:
    print("pixel maps skipped:", e)

# Fig 2 observed AMS (+ products) and CMIP6 relative change
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].plot(yrs, x_all, "-o", color=C[0], ms=3.5, lw=1.2, label="CHIRPS v2.0 (basin AMS)")
prod_f = os.path.join(HERE, "gee_ams_products.csv")
if os.path.exists(prod_f):
    pr = pd.read_csv(prod_f)
    for i, (nm, g) in enumerate(pr[pr["product"] != "CHIRPS_v2"].groupby("product")):
        g = g.dropna(subset=["AMS_mm"])
        ax[0].plot(g.year, g.AMS_mm, lw=1, color=C[1 + i], label=nm.replace("_", " "))
ax[0].plot(yrs, ols.intercept + ols.slope * yrs, "--", color=INK2, lw=1, label=f"OLS {ols.slope:+.3f} mm/yr (p={ols.pvalue:.2f})")
ax[0].set_ylabel("Annual maximum daily rainfall (mm)"); ax[0].set_title("(a) Observed basin AMS, 1981–2024", loc="left")
ax[0].set_ylim(top=x_all.max() * 1.16); ax[0].legend(fontsize=7, loc="upper left", ncol=2)
for i, sc in enumerate(SCEN):
    s = model_series(cm, sc); rel = s / s.loc[1981:2024].mean()
    rr = rel.rolling(30, center=True, min_periods=20).mean()
    ax[1].fill_between(rr.index, rr.quantile(.1, axis=1), rr.quantile(.9, axis=1), color=C[i * 1 + (0 if i == 0 else 0)] if False else [C[0], C[1]][i], alpha=.15, lw=0)
    ax[1].plot(rr.index, rr.median(axis=1), color=[C[0], C[1]][i], label=f"{SCEN_LAB[sc]} model median (30-yr mean)")
    tt = np.arange(1981, 2101); ax[1].plot(tt, np.exp(P[sc]["mu_c"] * (tt - T0)), ":", color=[C[0], C[1]][i], lw=1.2,
                                            label=f"E[Y_t]=exp(μc t), μc={P[sc]['mu_c']:.4f}")
ax[1].axhline(1, color=INK2, lw=.8); ax[1].set_ylabel("Relative change in AMS (ratio to 1981–2024)")
ax[1].set_title(f"(b) CMIP6 ({len(MODELS)} models), 10–90% range shaded", loc="left"); ax[1].legend(fontsize=7, loc="upper left")
savefig(fig, "Fig2_observed_and_projected_AMS.png")

# Fig 3 model evaluation heat-map
cols = [c for c in ev.columns if c not in ("score", "rank", "perf_subset")]
evs = ev.sort_values("score"); Z = (evs[cols] - evs[cols].mean()) / evs[cols].std()
fig, ax = plt.subplots(figsize=(1.1 * len(cols) + 2.5, 0.22 * len(evs) + 1.2))
im = ax.imshow(Z.values, cmap="RdBu_r", vmin=-2.5, vmax=2.5, aspect="auto")
ax.set_xticks(range(len(cols))); NICE = dict(mean_bias_pct="Mean bias (%)", cv_ratio="CV ratio", lskew_diff="L-skewness difference", q90_bias_pct="P90 bias (%)", ks_shape="KS distance", score="Composite score")
ax.set_xticklabels([NICE.get(c, c.replace("_", " ")) for c in cols], rotation=35, ha="right", fontsize=7)
ax.set_yticks(range(len(evs))); ax.set_yticklabels([f"{m}{' *' if evs.perf_subset[m] else ''}" for m in evs.index], fontsize=7)
for i in range(len(evs)):
    for j, c in enumerate(cols):
        ax.text(j, i, f"{evs[c].iloc[i]:.2f}" if abs(evs[c].iloc[i]) < 10 else f"{evs[c].iloc[i]:.0f}", ha="center", va="center", fontsize=5.5, color=INK)
ax.grid(False); plt.colorbar(im, ax=ax, shrink=.6, label="standardised value (z)")
ax.set_title("CMIP6 evaluation vs CHIRPS, 1981–2014 (* = performance subset, ranked by composite tail score)", loc="left", fontsize=8)
savefig(fig, "Fig4_model_evaluation.png")

# Fig 4 stationary GEV: return-level plot with 90% CI + xi sensitivity
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
Trg = np.logspace(np.log10(1.05), 2.3, 80); ql = [gev_q(1 - 1 / T, *G_FULL) for T in Trg]
bq = np.array([[gev_q(1 - 1 / T, *bb[:3]) for T in Trg] for bb in boot_full[:600]])
ax[0].fill_between(Trg, np.percentile(bq, 5, 0), np.percentile(bq, 95, 0), color=C[0], alpha=.15, lw=0, label="90% parametric-bootstrap CI")
ax[0].plot(Trg, ql, color=C[0], label=f"GEV (L-moments), ξ={-G_FULL[2]:+.3f}")
xs = np.sort(x_all); pp = (np.arange(1, 45) - 0.44) / (44 + 0.12)
ax[0].plot(1 / (1 - pp), xs, "o", color=INK, ms=3, label="CHIRPS AMS (Gringorten)")
ax[0].set_xscale("log"); ax[0].set_xlabel("Return period (years)"); ax[0].set_ylabel("24-h depth (mm)")
ax[0].set_title("(a) Stationary return levels, 1981–2024", loc="left"); ax[0].legend(fontsize=7)
xsd = pd.DataFrame(xi_sens); ax[1].plot(xsd.xi, xsd.Q100, "-o", color=C[0], ms=3, label="Q100"); ax[1].plot(xsd.xi, xsd.Q10, "-o", color=C[2], ms=3, label="Q10")
ax[1].axvline(-G_FULL[2], color=INK2, lw=.8, ls=":"); ax[1].set_xlabel("GEV shape ξ (Coles convention), across its 90% CI")
ax[1].set_ylabel("24-h depth (mm)"); ax[1].set_title("(b) Sensitivity of return levels to ξ", loc="left"); ax[1].legend(fontsize=7)
savefig(fig, "Fig3_stationary_GEV.png")

# Fig 5 volatility diagnostics
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].fill_between(simv.s_true, simv.p5, simv.p95, color=C[0], alpha=.15, lw=0, label="5–95% of ŝ in simulations (n=44)")
ax[0].plot(simv.s_true, simv.med, "-o", color=C[0], ms=3, label="median ŝ")
ax[0].plot([0, simv.s_true.max()], [0, simv.s_true.max()], color=INK2, lw=.7, ls="--")
ax[0].axhline(s_obs, color=C[1], lw=1.4, label=f"CHIRPS ŝ = {s_obs:.3f}")
ax[0].axvline(s_obs_u90, color=INK2, ls=":", lw=1, label=f"90% upper bound: s ≤ {s_obs_u90:.3f}")
ax[0].axvline(obs_cv, color=C[7], ls="--", lw=1, label=f"hypothesis s = CV = {obs_cv:.3f} (P(ŝ ≤ observed) = {p_v5:.3f})")
ax[0].set_xlabel("true random-walk volatility s"); ax[0].set_ylabel("estimated ŝ"); ax[0].legend(fontsize=6.5, loc="upper left")
ax[0].set_title(f"(a) CHIRPS: random-walk s ≈ 0; noise sd {irr_obs:.3f} ≈ CV {obs_cv:.3f}", loc="left", fontsize=8.5)
for i, sc in enumerate(SCEN):
    p = P[sc]; t = np.array(p["t"]); tt = np.linspace(0, 100, 50)
    ax[1].plot(t, p["V_raw"], "o", color=[C[0], C[1]][i], ms=4, mfc="white", label=f"{SCEN_LAB[sc]} raw inter-model var")
    ax[1].plot(t, p["V"], "o", color=[C[0], C[1]][i], ms=4, label=f"{SCEN_LAB[sc]} minus internal noise")
    ax[1].plot(tt, p["s"] ** 2 * tt + p["tau"] ** 2 * tt ** 2, color=[C[0], C[1]][i], lw=1.2)
    ax[1].plot(tt, 0.141 ** 2 * tt, "--", color=C[7], lw=1) if i == 0 else None
ax[1].set_ylim(0, max(0.06, 1.2 * max(max(P[s]["V_raw"]) for s in SCEN)))
ax[1].set_xlabel("t (years from 2002.5)"); ax[1].set_ylabel("Var[ln Y_t] across models")
ax[1].set_title("(b) Structural spread: s²t + τ²t² fit (dashed: s = CV)", loc="left", fontsize=8.5); ax[1].legend(fontsize=6.5)
savefig(fig, "Fig5_volatility_diagnostics.png")

# Fig 6 NS-IDF curves (intensity, mm/h) for SSP5-8.5 and SSP2-4.5, Tr=100 and 10
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
Dg = np.array([1, 2, 3, 6, 12, 24])
for i, sc in enumerate(SCEN):
    a = ax[i]
    for j, Tr in enumerate([10, 100]):
        a.plot(Dg, [gev_q(1 - 1 / Tr, *G_FULL) * scale_c(D) / D for D in Dg], color=INK2, lw=1, ls=["-", "--"][j],
               label=f"Stationary, T = {Tr} yr")
        for k, h in enumerate([2050, 2100]):
            a.plot(Dg, [dv(Tr, D, h, sc) / D for D in Dg], color=[C[0], C[1]][k], ls=["-", "--"][j], lw=1.3,
                   label=f"{h}, T = {Tr} yr")
    a.set_xscale("log"); a.set_yscale("log"); a.set_xticks(Dg); a.set_xticklabels(Dg)
    from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter as _FF
    a.yaxis.set_major_locator(FixedLocator([2, 3, 5, 10, 20, 30, 50])); a.yaxis.set_minor_locator(NullLocator()); a.yaxis.set_major_formatter(_FF(lambda v, p: f"{v:g}"))
    a.set_xlabel("Duration (h)"); a.set_title(f"({'ab'[i]}) {SCEN_LAB[sc]}", loc="left")
ax[0].set_ylabel("Intensity (mm/h)"); ax[1].legend(fontsize=7)
savefig(fig, "Fig6_nonstationary_IDF.png")

# Fig 7 C(t) trajectories with intervals and benchmarks (Tr=100, D=24)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True)
hh = np.arange(2025, 2101, 5)
for i, sc in enumerate(SCEN):
    a = ax[i]; q0 = gev_q(0.99, *G_FULL)
    cen = [dv(100, 24, h, sc) for h in hh]
    lo, hi = zip(*[np.quantile(mc_design(100, 24, h, sc, n=600), [.05, .95]) for h in hh])
    a.fill_between(hh, lo, hi, color=C[0], alpha=.15, lw=0, label="BS-IDF 90% interval")
    a.plot(hh, cen, color=C[0], label="BS-IDF central value")
    bb = bench[(bench.scenario == SCEN_LAB[sc]) & (bench.Tr == 100)]
    a.errorbar(bb.centre, bb.cmip6_slice_median, yerr=[bb.cmip6_slice_median - bb.cmip6_slice_p10, bb.cmip6_slice_p90 - bb.cmip6_slice_median],
               fmt="s", color=C[2], ms=5, capsize=3, lw=1, label="CMIP6 time-slice GEV (median, 10–90%)")
    if "cc_median" in bb:
        b2 = bb.dropna(subset=["cc_median"]); a.plot(b2.centre, b2.cc_median, "D", color=C[3], ms=5, label="Clausius–Clapeyron 7%/K reference")
    a.plot(hh, [gev_q(0.99, nsg["ns"]["mu0"] * (1 + nsg["ns"]["b"] * (h - T0)), nsg["ns"]["sig0"] * (1 + nsg["ns"]["b"] * (h - T0)), nsg["ns"]["k"]) for h in hh],
           ":", color=C[6], lw=1.3, label="NS-GEV, observed trend (n.s.)")
    a.axhline(q0, color=INK2, lw=.8, ls="--", label=f"Stationary Q100 = {q0:.1f} mm")
    a.set_title(f"({'ab'[i]}) {SCEN_LAB[sc]}: 100-yr, 24-h design depth", loc="left"); a.set_xlabel("Year")
ax[0].set_ylabel("Depth (mm)")
h_, l_ = ax[1].get_legend_handles_labels()
fig.legend(h_, l_, loc="lower center", ncol=3, fontsize=7.5, bbox_to_anchor=(0.5, -0.06))
fig.subplots_adjust(bottom=0.2)
savefig(fig, "Fig8_design_trajectories_benchmarks.png")

# Fig 8 out-of-sample
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
for i, (nm, g) in enumerate(oos.groupby("model")):
    u = np.sort(g.PIT.values); ax[0].step(u, np.arange(1, len(u) + 1) / len(u), where="post", color=C[i], label=nm)
ax[0].plot([0, 1], [0, 1], color=INK2, lw=.8, ls="--"); ax[0].set_xlabel("PIT"); ax[0].set_ylabel("Empirical CDF")
ax[0].set_title("(a) PIT of 2006–2024 maxima (calibrated 1981–2005)", loc="left", fontsize=8.5); ax[0].legend(fontsize=7)
ss = oos_sum.reset_index()
ax[1].barh(ss.model, ss.mean_CRPS, color=[C[i] for i in range(len(ss))], height=.55)
for i, v in enumerate(ss.mean_CRPS):
    ax[1].text(v, i, f" {v:.2f}", va="center", fontsize=7)
ax[1].set_xlabel("Mean CRPS (mm), lower is better"); ax[1].set_title("(b) Probabilistic skill, 2006–2024", loc="left", fontsize=8.5)
fig.subplots_adjust(wspace=0.55)
savefig(fig, "Fig9_out_of_sample.png")

# Fig 9 uncertainty decomposition + beta sensitivity
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
labs = {"gev": "GEV parameters", "drift": "Structural drift (μc, τ)", "vol": "Volatility s", "bc": "Bias-correction choice", "beta": "Sub-daily exponent β"}
yl = [f"{r.horizon}, D = {r.D} h" for r in dec.itertuples()]; left = np.zeros(len(dec))
for j, k in enumerate(labs):
    v = dec[f"share_{k}"].values; ax[0].barh(yl, v, left=left, color=C[j], height=.55, label=labs[k], edgecolor="white", lw=1); left += v
ax[0].set_xlabel("Share of design-value variance (%)"); ax[0].set_title("(a) Uncertainty decomposition, T = 100 yr, SSP5-8.5", loc="left", fontsize=8.5)
ax[0].legend(fontsize=6.5, ncol=2, loc="upper center", bbox_to_anchor=(.5, -.22)); ax[0].grid(axis="y", visible=False)
for j, (D, Tr) in enumerate([(1, 100), (3, 100), (1, 10), (3, 10)]):
    g = bs[(bs.D == D) & (bs.Tr == Tr)]; ax[1].plot(g.beta, g.ssp585_2050, "-o", ms=3, color=C[j], label=f"D = {D} h, T = {Tr} yr (2050, SSP5-8.5)")
ax[1].axvspan(0.08, 0.20, color="#d8d7d2", alpha=0.5, lw=0, label="West African gauge scaling (Senegal, Niger)")
ax[1].axvline(1 / 3, color=INK2, ls=":", lw=1); ax[1].set_xlabel("Depth-duration exponent β (IMD value 1/3 dotted)"); ax[1].set_ylabel("Depth (mm)")
ax[1].set_title("(b) Sub-daily sensitivity to β", loc="left", fontsize=8.5); ax[1].legend(fontsize=6.5, ncol=2, loc="upper center", bbox_to_anchor=(.5, -.22))
savefig(fig, "Fig7_uncertainty_and_beta.png")


# Fig 10 variance-based sensitivity and Hawkins-Sutton partition
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
keys = list(labs); yl = [f"{r.horizon}, D = {r.D} h" for r in sob.itertuples()]; yy = np.arange(len(sob))
hgt = 0.8 / len(keys)
for j, k in enumerate(keys):
    ax[0].barh(yy + (j - 2) * hgt, sob[f"ST_{k}"], height=hgt * 0.92, color=C[j], label=labs[k])
    ax[0].scatter(np.clip(sob[f"S1_{k}"], 0, None), yy + (j - 2) * hgt, marker="|", s=40, color=INK, zorder=5)
ax[0].set_yticks(yy); ax[0].set_yticklabels(yl); ax[0].invert_yaxis(); ax[0].set_xlim(0, 1)
ax[0].set_xlabel("Sobol index (bar: total effect; tick: first order)")
ax[0].set_title("(a) Variance-based sensitivity, T = 100 yr, SSP5-8.5", loc="left", fontsize=8.5)
ax[0].legend(fontsize=6.5, ncol=2, loc="upper center", bbox_to_anchor=(.5, -.22)); ax[0].grid(axis="y", visible=False)
hsx = [f"{SLICES[c][0]}–{SLICES[c][1]}" for c in hs.index]; bot = np.zeros(len(hs))
for j, (k, lb) in enumerate([("internal", "Internal variability"), ("model", "Model (noise-corrected)"), ("scenario", "Scenario")]):
    ax[1].bar(hsx, hs[k], bottom=bot, color=[C[6], C[0], C[1]][j], width=.6, label=lb, edgecolor="white", lw=1)
    for xi_, (b_, v_) in enumerate(zip(bot, hs[k])):
        if v_ > 7:
            ax[1].text(xi_, b_ + v_ / 2, f"{v_:.0f}%", ha="center", va="center", fontsize=7, color="white")
    bot += hs[k].values
ax[1].set_ylabel("Share of variance of ln(30-yr mean AMS change) (%)"); ax[1].set_ylim(0, 100)
ax[1].set_title("(b) Partition of projection uncertainty", loc="left", fontsize=8.5)
ax[1].legend(fontsize=7, loc="upper center", bbox_to_anchor=(.5, -.14), ncol=3); ax[1].grid(axis="x", visible=False)
savefig(fig, "Fig10_sobol_partition.png")

# ================================================================== 11. SAVE
def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, pd.DataFrame):
        return o.to_dict("records")
    return o


json.dump(clean(R), open(os.path.join(OUT, "results.json"), "w"), indent=1, default=str)
print("done ->", OUT)

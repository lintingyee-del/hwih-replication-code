# -*- coding: utf-8 -*-
"""6B step 22 — referee-anticipation robustness battery.

E1  Exposure mean-reversion: split-half exposure (2014-15 vs 2016-17) swap on
    the two headline specs; within-province permutation inference (999).
E2  Charge substitution: mafia-organization docket response; market backstop
    and market caseload with same-cell mafia-docket controls / drops.
E3  Ex-ante cohort selection: origination-cohort documentation and rate
    re-estimated on matched litigation-lag windows (age <= 2, <= 1 years).
E4  Heterogeneity-robust estimation: manual Callaway-Sant'Anna (2021) with
    not-yet-treated controls on the prefecture-month relational-minus-placebo
    civil gap and on the enforcement-crime caseload, dose contrast = top vs
    bottom exposure tercile; province block bootstrap (999).

Outputs: output/referee_robustness.csv, output/cs_dynamics.csv,
         output/tables/numbers_ref.tex, output/tables/tab_meanrev.tex,
         output/tables/tab_cs.tex
"""

# Replication-package paths
from pathlib import Path as _ReplicationPath
import os as _ReplicationOS
_REP_PROJECT = _ReplicationPath(__file__).resolve().parents[1]
_REP_PACKAGE = _REP_PROJECT.parent
_REP_RESTRICTED = _REP_PACKAGE / 'restricted_data'
_REP_JUDGMENTS = _ReplicationPath(_ReplicationOS.environ.get(
    'HWIH_JUDGMENT_ARCHIVE', _REP_RESTRICTED / 'judgment_archive'))
_REP_CASE_ARCHIVE = _ReplicationPath(_ReplicationOS.environ.get(
    'HWIH_CASE_LEVEL_ARCHIVE', _REP_RESTRICTED / 'case_level_archive.parquet'))
_REP_MORTALITY = _ReplicationPath(_ReplicationOS.environ.get(
    'HWIH_CDC_SOURCE_ROOT', _REP_RESTRICTED / 'mortality_volumes'))
_REP_REGISTRY = _ReplicationPath(_ReplicationOS.environ.get(
    'HWIH_REGISTRY_ROOT', _REP_RESTRICTED / 'firm_registry'))
_REP_BAIDU = _ReplicationPath(_ReplicationOS.environ.get(
    'HWIH_BAIDU_ROOT', _REP_RESTRICTED / 'baidu'))
_REP_INTERVIEWS = _ReplicationPath(_ReplicationOS.environ.get(
    'HWIH_INTERVIEWS_ROOT', _REP_RESTRICTED / 'interviews'))
import duckdb, os
import numpy as np
import pandas as pd
import pyfixest as pf
from scipy import stats as sps

DATA = str(_REP_PROJECT / "data")
OUTD = str(_REP_PROJECT / "output")
EXT = str(_REP_PROJECT / "data" / "derived" / "extract_2014_2020")
rows = []

from _wild import wild_score_p, wild_p  # corrected shared WCR bootstrap


def run(tag, fml, df, coef, weights=None):
    m = pf.feols(fml, data=df, vcov={"CRV1": "prov_id"}, weights=weights)
    try: wp = wild_score_p(fml, df, coef, weights)
    except Exception: wp = np.nan
    rows.append(dict(tag=tag, coef=coef, est=m.coef()[coef], se=m.se()[coef],
                     p=m.pvalue()[coef], wild_p=wp, n=int(m._N)))
    print(f"{tag:44s} {m.coef()[coef]: .5f} ({m.se()[coef]:.5f}) "
          f"p={m.pvalue()[coef]:.4f} wild={wp:.3f} N={m._N}", flush=True)
    return m

# ============================================================================
# Part 0 — one duckdb pass over the raw criminal extracts:
#   split-half exposure components + mafia/extortion prefecture-month counts
# ============================================================================
print("== duckdb pass over raw criminal extracts ==", flush=True)
con = duckdb.connect()
con.sql("SET threads TO 10; SET memory_limit='20GB'")
con.sql(f"CREATE OR REPLACE TABLE xwalk AS SELECT * FROM '{DATA}/court_xwalk.parquet'")
con.sql(f"""
CREATE OR REPLACE TABLE crim0 AS
SELECT c.crime, c.detention_debt, c.d_backstop_collection,
       x.prefecture_code, x.province, TRY_CAST(c.judgment_date AS DATE) AS jdate
FROM read_parquet('{EXT}/crim_*.parquet') c
LEFT JOIN xwalk x ON c.court = x.court_name
WHERE x.prefecture_code IS NOT NULL
""")

VIOL = ("'非法拘禁','寻衅滋事','聚众斗殴','敲诈勒索','强迫交易',"
        "'组织、领导、参加黑社会性质组织'")
halves = {}
for tag, lo, hi in [("h1", "2014-01-01", "2015-12-31"),
                    ("h2", "2016-01-01", "2017-12-31")]:
    halves[tag] = con.sql(f"""
        SELECT prefecture_code, COUNT(*) AS n_pre,
          AVG((crime IN ({VIOL}))::INT) AS violent_share,
          AVG(d_backstop_collection) AS backstop_collect_rate
        FROM crim0 WHERE jdate BETWEEN DATE '{lo}' AND DATE '{hi}'
        GROUP BY 1 HAVING COUNT(*) >= 150
    """).df()

mafia = con.sql("""
    SELECT prefecture_code, strftime(date_trunc('month', jdate), '%Y-%m') AS month,
      SUM((crime = '组织、领导、参加黑社会性质组织')::INT) AS n_mafia,
      SUM((crime = '敲诈勒索')::INT) AS n_extort
    FROM crim0
    WHERE jdate BETWEEN DATE '2014-01-01' AND DATE '2020-12-31'
    GROUP BY 1, 2
""").df()
con.close()

def zcomp(d):
    z = lambda s: (s - s.mean()) / s.std()
    return (z(d["violent_share"]) + z(d["backstop_collect_rate"])) / 2

hh = halves["h1"][["prefecture_code"]].merge(halves["h2"][["prefecture_code"]])
for tag in ("h1", "h2"):
    d = halves[tag][halves[tag]["prefecture_code"].isin(hh["prefecture_code"])].copy()
    d[f"H_{tag}"] = zcomp(d)
    halves[tag] = d[["prefecture_code", f"H_{tag}"]]
half = halves["h1"].merge(halves["h2"], on="prefecture_code")
ex2 = pd.read_parquet(f"{DATA}/exposure_v2.parquet")[["prefecture_code", "exposure_v2_z"]]
half = half.merge(ex2, on="prefecture_code")
corr_hh = half["H_h1"].corr(half["H_h2"])
corr_h1f = half["H_h1"].corr(half["exposure_v2_z"])
corr_h2f = half["H_h2"].corr(half["exposure_v2_z"])
print(f"split-half corr: h1-h2 {corr_hh:.3f}  h1-full {corr_h1f:.3f} "
      f"h2-full {corr_h2f:.3f}  N={len(half)}", flush=True)

# ============================================================================
# Part 1 — headline spec replications + split-half swaps
# ============================================================================
print("== E1a: split-half exposure swaps ==", flush=True)
WINDOW = ("2017-01", "2019-03"); POST0 = "2018-09"
CLEAN_START = pd.Timestamp("2017-01-01")
CLEAN_END = pd.Timestamp("2019-04-01")

# ---- clean-window stacked civil flow (replicates 14_stacked S1_civil_flow_dose)
c = pd.read_parquet(f"{DATA}/civil_panel.parquet")
c = c[c["cause_family"] == "relational"].copy()
c["judgment_date"] = pd.to_datetime(c["jmonth"], errors="coerce")
c = c[(c["judgment_date"] >= CLEAN_START) &
      (c["judgment_date"] < CLEAN_END)].copy()
c["month"] = c["judgment_date"].dt.strftime("%Y-%m")
sched = pd.read_parquet(f"{DATA}/panel_month.parquet")[
    ["province", "inspection_round"]].drop_duplicates()
c = c.merge(sched, on="province", how="left")
c["treat"] = (c["inspection_round"] == 1).astype(int)
c["postc"] = (c["month"] >= POST0).astype(int)
c["prov_id"] = pd.factorize(c["province"])[0]
c["pref_cause"] = c["prefecture_code"] + "_" + c["cause"]
c["month_fe"] = c["month"]
c["asinh_n"] = np.arcsinh(c["n_cases"])
c["pt"] = c["postc"] * c["treat"]

def stacked_civil(tag, d, Hcol):
    d = d.dropna(subset=[Hcol]).copy()
    d["pth"] = d["pt"] * d[Hcol]
    d["ph"] = d["postc"] * d[Hcol]
    return run(tag, "asinh_n ~ pth + ph + pt | pref_cause + month_fe", d, "pth")

c["H_full"] = c["exposure_v2_z"]
stacked_civil("E1_civ_stacked_baseline", c, "H_full")
ch = c.merge(half[["prefecture_code", "H_h1", "H_h2"]], on="prefecture_code")
stacked_civil("E1_civ_stacked_common", ch, "H_full")
stacked_civil("E1_civ_stacked_H1415", ch, "H_h1")
stacked_civil("E1_civ_stacked_H1617", ch, "H_h2")

# ---- full-sample enforcement caseload (replicates 09 K2_enforcement_asinhN)
kp = pd.read_parquet(f"{DATA}/crim_panel_v2.parquet")
kp = kp[kp["n_cases"] > 0].copy()
kp["month"] = kp["jmonth"].astype(str)
kp["prov_id"] = pd.factorize(kp["province"])[0]
kp["pref"] = kp["prefecture_code"]
kp["prov_month"] = kp["province"] + "_" + kp["month"]
en = kp[kp["family"] == "enforcementcrime"].copy()
en["asinh_n"] = np.arcsinh(en["n_cases"])

def enforce_spec(tag, d, Hcol):
    d = d.dropna(subset=[Hcol]).copy()
    d["px"] = d["post"] * d[Hcol]
    return run(tag, "asinh_n ~ px | pref + prov_month", d, "px")

en["H_full"] = en["exposure_v2_z"]
enforce_spec("E1_enf_baseline", en, "H_full")
enh = en.merge(half[["prefecture_code", "H_h1", "H_h2"]], on="prefecture_code")
enforce_spec("E1_enf_common", enh, "H_full")
enforce_spec("E1_enf_H1415", enh, "H_h1")
enforce_spec("E1_enf_H1617", enh, "H_h2")

# ============================================================================
# Part 2 — E2: charge substitution
# ============================================================================
print("== E2: charge substitution ==", flush=True)
sched2 = pd.read_parquet(f"{DATA}/crim_panel_v2.parquet")[
    ["province", "insp_month"]].drop_duplicates()
prefs = pd.read_parquet(f"{DATA}/crim_panel_v2.parquet")[
    ["prefecture_code", "province", "exposure_v2_z"]].drop_duplicates()
months = pd.period_range("2014-01", "2020-12", freq="M").astype(str)
grid = prefs.merge(pd.DataFrame({"month": months}), how="cross")
grid = grid.merge(mafia.rename(columns={"month": "month"}),
                  on=["prefecture_code", "month"], how="left")
grid[["n_mafia", "n_extort"]] = grid[["n_mafia", "n_extort"]].fillna(0)
grid = grid.merge(sched2, on="province")
grid["insp"] = grid["insp_month"].astype(str).str[:7]
grid["post"] = (grid["month"] >= grid["insp"]).astype(int)
grid["px"] = grid["post"] * grid["exposure_v2_z"]
grid["prov_id"] = pd.factorize(grid["province"])[0]
grid["pref"] = grid["prefecture_code"]
grid["prov_month"] = grid["province"] + "_" + grid["month"]
grid["asinh_mafia"] = np.arcsinh(grid["n_mafia"])
run("E2_mafia_docket_response", "asinh_mafia ~ px | pref + prov_month",
    grid, "px")

mafia_m = grid[["prefecture_code", "month", "n_mafia", "asinh_mafia"]]

# v2 market backstop battery
mk = kp[kp["family"] == "market"].copy()
mk["H"] = mk["exposure_v2_z"]; mk["px"] = mk["post"] * mk["H"]
mk["month"] = mk["month"].str[:7]  # align with mafia_m YYYY-MM key
mk = mk.merge(mafia_m, on=["prefecture_code", "month"], how="left")
mk[["n_mafia", "asinh_mafia"]] = mk[["n_mafia", "asinh_mafia"]].fillna(0)
run("E2_v2_backstop_baseline", "y_backstop ~ px + x_doclen | pref + prov_month",
    mk, "px", weights="n_cases")
run("E2_v2_backstop_mafiactl",
    "y_backstop ~ px + x_doclen + asinh_mafia | pref + prov_month",
    mk, "px", weights="n_cases")
run("E2_v2_backstop_dropmafia",
    "y_backstop ~ px + x_doclen | pref + prov_month",
    mk[mk["n_mafia"] == 0], "px", weights="n_cases")
mk["asinh_n"] = np.arcsinh(mk["n_cases"])
run("E2_v2_marketN_baseline", "asinh_n ~ px | pref + prov_month", mk, "px")
run("E2_v2_marketN_mafiactl", "asinh_n ~ px + asinh_mafia | pref + prov_month",
    mk, "px")

# v1 market backstop battery (first-generation measure, panel_month)
p1 = pd.read_parquet(f"{DATA}/panel_month.parquet")
p1 = p1[(p1["analysis_group"] == "market") & (p1["n_fact"] > 0)].copy()
p1["month"] = p1["judgment_month"].astype(str).str[:7]
p1["prov_id"] = pd.factorize(p1["province"])[0]
p1["pref"] = p1["prefecture_code"]
p1["prov_month"] = p1["province"] + "_" + p1["month"]
p1["px"] = p1["post_judgment"] * p1["exposure_z"]
p1 = p1.merge(mafia_m, on=["prefecture_code", "month"], how="left")
p1[["n_mafia", "asinh_mafia"]] = p1[["n_mafia", "asinh_mafia"]].fillna(0)
run("E2_v1_backstop_baseline",
    "y_backstop ~ px + x_factshare + x_spanshare | pref + prov_month", p1, "px",
    weights="n_fact")
run("E2_v1_backstop_mafiactl",
    "y_backstop ~ px + x_factshare + x_spanshare + asinh_mafia | pref + prov_month",
    p1, "px", weights="n_fact")
run("E2_v1_backstop_dropmafia",
    "y_backstop ~ px + x_factshare + x_spanshare | pref + prov_month",
    p1[p1["n_mafia"] == 0], "px", weights="n_fact")

# ============================================================================
# Part 3 — E3: origination cohorts on matched litigation-lag windows
# ============================================================================
print("== E3: duration-matched cohorts ==", flush=True)
cc = pd.read_parquet(f"{DATA}/civil_case.parquet",
                     columns=["cause", "prefecture_code", "province", "jmonth",
                              "post", "insp_month", "evid_iou", "evid_transfer",
                              "rel_txn", "monthly_rate_pct", "orig_year",
                              "doc_len"])
ld = cc[cc["cause"] == "民间借贷纠纷"].merge(ex2, on="prefecture_code")
ld["prov_id"] = pd.factorize(ld["province"])[0]
ld["H"] = ld["exposure_v2_z"]
ld["month"] = ld["jmonth"].astype(str).str[:7]
ld["pref"] = ld["prefecture_code"]
ld["logdoclen"] = np.log(ld["doc_len"].clip(lower=1))
oc = ld[(ld["orig_year"] >= 2012) & (ld["orig_year"] <= 2020)].copy()
oc["insp_year"] = pd.to_datetime(oc["insp_month"]).dt.year
oc["post_cohort"] = (oc["orig_year"] >= oc["insp_year"]).astype(int)
oc["pcx"] = oc["post_cohort"] * oc["H"]
oc["oy"] = oc["orig_year"].astype(str)
oc["age"] = oc["month"].str[:4].astype(int) - oc["orig_year"]
oc = oc[oc["age"] >= 0]
age_pre = oc.loc[oc["post_cohort"] == 0, "age"].mean()
age_post = oc.loc[oc["post_cohort"] == 1, "age"].mean()
print(f"mean litigation lag (years): pre-cohort {age_pre:.2f}, "
      f"post-cohort {age_post:.2f}", flush=True)

run("E3_iou_baseline", "evid_iou ~ pcx + logdoclen | pref + oy + month", oc, "pcx")
run("E3_iou_age2", "evid_iou ~ pcx + logdoclen | pref + oy + month",
    oc[oc["age"] <= 2], "pcx")
run("E3_iou_age1", "evid_iou ~ pcx + logdoclen | pref + oy + month",
    oc[oc["age"] <= 1], "pcx")
run("E3_transfer_age2", "evid_transfer ~ pcx + logdoclen | pref + oy + month",
    oc[oc["age"] <= 2], "pcx")
lr = oc[(oc["monthly_rate_pct"] > 0) & (oc["monthly_rate_pct"] <= 10)
        & (oc["month"] <= "2020-07")].copy()
run("E3_rate_cohort_age2", "monthly_rate_pct ~ pcx + logdoclen | pref + oy + month",
    lr[lr["age"] <= 2], "pcx")

# ============================================================================
# Part 4 — E4: manual Callaway-Sant'Anna, not-yet-treated controls
# ============================================================================
print("== E4: Callaway-Sant'Anna dose contrast ==", flush=True)

def midx(s):  # 'YYYY-MM' -> integer month index
    return s.str[:4].astype(int) * 12 + s.str[5:7].astype(int)

cv = pd.read_parquet(f"{DATA}/civil_panel.parquet")
cv["month"] = cv["jmonth"].astype(str).str[:7]
rel_n = cv[cv["cause_family"] == "relational"].groupby(
    ["prefecture_code", "month"])["n_cases"].sum().rename("n_rel")
plc_n = cv[cv["cause_family"] == "placebo"].groupby(
    ["prefecture_code", "month"])["n_cases"].sum().rename("n_plc")
gap = grid[["prefecture_code", "province", "exposure_v2_z", "month",
            "insp"]].copy()
gap = gap.merge(rel_n, on=["prefecture_code", "month"], how="left") \
         .merge(plc_n, on=["prefecture_code", "month"], how="left")
gap[["n_rel", "n_plc"]] = gap[["n_rel", "n_plc"]].fillna(0)
gap["y"] = np.arcsinh(gap["n_rel"]) - np.arcsinh(gap["n_plc"])

enf_n = kp[kp["family"] == "enforcementcrime"].assign(
    month=lambda d: d["month"].str[:7]).groupby(
    ["prefecture_code", "month"])["n_cases"].sum().rename("n_enf")
gap = gap.merge(enf_n, on=["prefecture_code", "month"], how="left")
gap["n_enf"] = gap["n_enf"].fillna(0)
gap["y_enf"] = np.arcsinh(gap["n_enf"])

gap["t"] = midx(gap["month"])
gap["g"] = midx(gap["insp"])
gap["terc"] = pd.qcut(gap["exposure_v2_z"], 3, labels=[0, 1, 2]).astype(int)

PRE, POST = 12, 8
EV = list(range(-PRE, -1)) + list(range(0, POST + 1))

def cs_att(d, ycol, provs=None):
    """CS(2021) group-time ATTs, not-yet-treated controls, by exposure tercile.
    Returns {e: high-low dose contrast}, weighting groups by treated-cell count."""
    if provs is not None:  # bootstrap draw: province multiset
        parts = [d[d["province"] == pv].assign(bs=i) for i, pv in enumerate(provs)]
        d = pd.concat(parts, ignore_index=True)
    piv = d.pivot_table(index=["prefecture_code", "g", "terc"] +
                        (["bs"] if provs is not None else []),
                        columns="t", values=ycol)
    out = {}
    for e in EV:
        num = {2: 0.0, 0: 0.0}; den = {2: 0.0, 0: 0.0}
        idx_g = piv.index.get_level_values("g")
        idx_T = piv.index.get_level_values("terc")
        for g in np.unique(idx_g):
            t = g + e
            if t not in piv.columns or (g - 1) not in piv.columns: continue
            nyt = idx_g > max(t, g)          # strictly not yet treated
            dy = piv[t] - piv[g - 1]
            for T in (2, 0):
                tr = dy[(idx_g == g) & (idx_T == T)].dropna()
                ct = dy[nyt & (idx_T == T)].dropna()
                if len(tr) < 3 or len(ct) < 3: continue
                num[T] += len(tr) * (tr.mean() - ct.mean()); den[T] += len(tr)
        if den[2] > 0 and den[0] > 0:
            out[e] = num[2] / den[2] - num[0] / den[0]
    return out

def cs_summary(label, ycol, reps=999, seed=42):
    point = cs_att(gap, ycol)
    posts = [e for e in point if e >= 0]
    pres = [e for e in point if e < -1]
    overall = np.mean([point[e] for e in posts])
    provs_all = gap["province"].unique()
    rng = np.random.default_rng(seed)
    boot = {e: [] for e in point}; boot_o = []
    for r in range(reps):
        draw = rng.choice(provs_all, size=len(provs_all), replace=True)
        b = cs_att(gap, ycol, provs=draw)
        ok = [e for e in posts if e in b]
        if ok: boot_o.append(np.mean([b[e] for e in ok]))
        for e in point:
            if e in b: boot[e].append(b[e])
    se_o = np.std(boot_o, ddof=1)
    p_o = 2 * (1 - sps.norm.cdf(abs(overall / se_o)))
    # joint pre-test on leads (bootstrap covariance)
    pre_p = np.nan
    if pres:
        nb = min(len(boot[e]) for e in pres)
        B = np.array([[boot[e][j] for e in pres] for j in range(nb)])
        bvec = np.array([point[e] for e in pres])
        V = np.cov(B.T)
        try:
            pre_p = float(1 - sps.chi2.cdf(bvec @ np.linalg.solve(V, bvec),
                                           len(pres)))
        except Exception: pass
    rows.append(dict(tag=f"E4_CS_{label}_overall", coef="highlow",
                     est=overall, se=se_o, p=p_o, wild_p=np.nan,
                     n=int(gap["prefecture_code"].nunique())))
    print(f"CS {label}: overall(e0..{POST}) {overall:.4f} (boot SE {se_o:.4f}, "
          f"p {p_o:.3f}); joint lead p {pre_p:.3f}", flush=True)
    dyn = [dict(outcome=label, e=e, est=point[e],
                se=np.std(boot[e], ddof=1) if len(boot[e]) > 2 else np.nan)
           for e in sorted(point)]
    return overall, se_o, p_o, pre_p, dyn

cs_civ = cs_summary("civilgap", "y")
cs_enf = cs_summary("enforce", "y_enf")
pd.DataFrame(cs_civ[4] + cs_enf[4]).to_csv(f"{OUTD}/cs_dynamics.csv", index=False)

# ============================================================================
# Part 5 — E1b: within-province permutation inference (999) on headline specs
# ============================================================================
print("== E1b: permutation inference (999 draws each) ==", flush=True)

def permute_within_prov(prefH, rng):
    return prefH.groupby("province")["H"].transform(
        lambda s: rng.permutation(s.values))

def perm_test(tag, d, fit, reps=999, seed=42):
    b_obs = fit()
    prefH = d[["pref", "province", "H"]].drop_duplicates("pref").reset_index(drop=True)
    rng = np.random.default_rng(seed)
    hits = 0
    for r in range(reps):
        prefH["Hp"] = permute_within_prov(prefH.rename(columns={"H": "H"}), rng)
        hmap = dict(zip(prefH["pref"], prefH["Hp"]))
        if abs(fit(hmap)) >= abs(b_obs): hits += 1
        if (r + 1) % 200 == 0: print(f"  {tag} perm {r+1}/{reps}", flush=True)
    p = (1 + hits) / (1 + reps)
    rows.append(dict(tag=tag, coef="perm", est=b_obs, se=np.nan, p=p,
                     wild_p=np.nan, n=reps))
    print(f"{tag}: obs {b_obs:.4f}, permutation p = {p:.3f}", flush=True)
    return p

cd = c.dropna(subset=["exposure_v2_z"]).copy()
cd["pref"] = cd["prefecture_code"]
cd["H"] = cd["exposure_v2_z"]
def fit_civ(hmap=None):
    d = cd
    H = d["pref"].map(hmap) if hmap else d["exposure_v2_z"]
    d = d.assign(pth=d["pt"] * H, ph=d["postc"] * H)
    m = pf.feols("asinh_n ~ pth + ph + pt | pref_cause + month_fe", data=d)
    return m.coef()["pth"]
perm_p_civ = perm_test("E1_perm_civ_stacked", cd, fit_civ)

ed = en.dropna(subset=["exposure_v2_z"]).copy()
ed["H"] = ed["exposure_v2_z"]
def fit_enf(hmap=None):
    d = ed
    H = d["pref"].map(hmap) if hmap else d["exposure_v2_z"]
    d = d.assign(px=d["post"] * H)
    m = pf.feols("asinh_n ~ px | pref + prov_month", data=d)
    return m.coef()["px"]
perm_p_enf = perm_test("E1_perm_enf", ed, fit_enf)

# ============================================================================
# Part 6 — export: csv, numbers macros, tables
# ============================================================================
res = pd.DataFrame(rows)
res.to_csv(f"{OUTD}/referee_robustness.csv", index=False)

R = {r["tag"]: r for r in rows}
def g3(tag, f="est"): return f"{R[tag][f]:.3f}"
def gN(tag): return f"{R[tag]['n']:,}"
def star(tag):
    return ""

macros = {
    "RefHalfCorr": f"{corr_hh:.2f}",
    "RefCivHalfA": g3("E1_civ_stacked_H1415"),
    "RefCivHalfAWildP": g3("E1_civ_stacked_H1415", "wild_p"),
    "RefCivHalfB": g3("E1_civ_stacked_H1617"),
    "RefCivHalfBWildP": g3("E1_civ_stacked_H1617", "wild_p"),
    "RefEnfHalfA": g3("E1_enf_H1415"), "RefEnfHalfB": g3("E1_enf_H1617"),
    "RefEnfHalfAWildP": g3("E1_enf_H1415", "wild_p"),
    "RefEnfHalfBWildP": g3("E1_enf_H1617", "wild_p"),
    "RefPermCiv": g3("E1_perm_civ_stacked", "p"),
    "RefPermEnf": g3("E1_perm_enf", "p"),
    "RefMafiaResp": g3("E2_mafia_docket_response"),
    "RefMafiaRespWildP": g3("E2_mafia_docket_response", "wild_p"),
    "RefVOneBackstopMafia": g3("E2_v1_backstop_mafiactl"),
    "RefVOneBackstopMafiaWildP": g3("E2_v1_backstop_mafiactl", "wild_p"),
    "RefVOneBackstopDrop": g3("E2_v1_backstop_dropmafia"),
    "RefVTwoBackstopMafia": g3("E2_v2_backstop_mafiactl"),
    "RefMarketNMafia": g3("E2_v2_marketN_mafiactl"),
    "RefIouAgeTwo": g3("E3_iou_age2"), "RefIouAgeTwoP": g3("E3_iou_age2", "p"),
    "RefIouAgeOne": g3("E3_iou_age1"), "RefIouAgeOneP": g3("E3_iou_age1", "p"),
    "RefTransferAgeTwo": g3("E3_transfer_age2"),
    "RefTransferAgeTwoP": g3("E3_transfer_age2", "p"),
    "RefRateAgeTwo": g3("E3_rate_cohort_age2"),
    "RefRateAgeTwoP": g3("E3_rate_cohort_age2", "p"),
    "RefAgePre": f"{age_pre:.1f}", "RefAgePost": f"{age_post:.1f}",
    "RefCSCiv": f"{cs_civ[0]:.3f}", "RefCSCivSE": f"{cs_civ[1]:.3f}",
    "RefCSCivP": f"{cs_civ[2]:.3f}", "RefCSCivPreP": f"{cs_civ[3]:.3f}",
    "RefCSEnf": f"{cs_enf[0]:.3f}", "RefCSEnfSE": f"{cs_enf[1]:.3f}",
    "RefCSEnfP": f"{cs_enf[2]:.3f}", "RefCSEnfPreP": f"{cs_enf[3]:.3f}",
}
with open(f"{OUTD}/tables/numbers_ref.tex", "w", encoding="utf-8") as fh:
    for k, v in macros.items():
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

def trow(label, tag):
    r = R[tag]
    wp = "--" if np.isnan(r["wild_p"]) else f"{r['wild_p']:.3f}"
    return (f"{label} & {r['est']:.4f}{star(tag)} & ({r['se']:.4f}) & {wp} "
            f"& {r['n']:,} \\\\\n")

with open(f"{OUTD}/tables/tab_meanrev.tex", "w", encoding="utf-8") as fh:
    fh.write("\\begin{tabular}{lcccc}\n\\toprule\n"
             "Specification & Coefficient & (SE) & Wild $p$ & $N$ \\\\ \\midrule\n")
    fh.write("\\multicolumn{5}{l}{\\emph{Panel A. Split-half exposure: "
             "clean-window civil flow}} \\\\[2pt]\n")
    fh.write(trow("Full index, common sample", "E1_civ_stacked_common"))
    fh.write(trow("2014--15 half index", "E1_civ_stacked_H1415"))
    fh.write(trow("2016--17 half index", "E1_civ_stacked_H1617"))
    fh.write("\\midrule\n\\multicolumn{5}{l}{\\emph{Panel B. Split-half "
             "exposure: enforcement caseload}} \\\\[2pt]\n")
    fh.write(trow("Full index, common sample", "E1_enf_common"))
    fh.write(trow("2014--15 half index", "E1_enf_H1415"))
    fh.write(trow("2016--17 half index", "E1_enf_H1617"))
    fh.write("\\midrule\n\\multicolumn{5}{l}{\\emph{Panel C. Charge "
             "substitution: mafia-docket margin}} \\\\[2pt]\n")
    fh.write(trow("Mafia-organization caseload (asinh)", "E2_mafia_docket_response"))
    fh.write(trow("v1 market backstop, baseline", "E2_v1_backstop_baseline"))
    fh.write(trow("\\quad + same-cell mafia control", "E2_v1_backstop_mafiactl"))
    fh.write(trow("\\quad drop mafia-case cells", "E2_v1_backstop_dropmafia"))
    fh.write(trow("Market caseload + mafia control", "E2_v2_marketN_mafiactl"))
    fh.write("\\midrule\n\\multicolumn{5}{l}{\\emph{Panel D. Origination "
             "cohorts, matched litigation lag}} \\\\[2pt]\n")
    fh.write(trow("IOU share, lag $\\le$ 2 years", "E3_iou_age2"))
    fh.write(trow("IOU share, lag $\\le$ 1 year", "E3_iou_age1"))
    fh.write(trow("Transfer records, lag $\\le$ 2 years", "E3_transfer_age2"))
    fh.write(trow("Monthly rate, lag $\\le$ 2 years", "E3_rate_cohort_age2"))
    fh.write("\\midrule\n")
    fh.write(f"Permutation $p$ (civil flow; 999 within-province draws) & "
             f"\\multicolumn{{4}}{{c}}{{{perm_p_civ:.3f}}} \\\\\n")
    fh.write(f"Permutation $p$ (enforcement caseload) & "
             f"\\multicolumn{{4}}{{c}}{{{perm_p_enf:.3f}}} \\\\\n")
    fh.write(f"Split-half exposure correlation & "
             f"\\multicolumn{{4}}{{c}}{{{corr_hh:.2f}}} \\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")

dyn = pd.read_csv(f"{OUTD}/cs_dynamics.csv")
with open(f"{OUTD}/tables/tab_cs.tex", "w", encoding="utf-8") as fh:
    fh.write("\\begin{tabular}{lcccc}\n\\toprule\n"
             " & \\multicolumn{2}{c}{Civil relational $-$ placebo gap} & "
             "\\multicolumn{2}{c}{Enforcement caseload} \\\\\n"
             "Event time (months) & Estimate & (boot SE) & Estimate & (boot SE) "
             "\\\\ \\midrule\n")
    dc = dyn[dyn["outcome"] == "civilgap"].set_index("e")
    de = dyn[dyn["outcome"] == "enforce"].set_index("e")
    for e in sorted(set(dc.index) | set(de.index)):
        if e < 0 and e % 3 != 0: continue
        cells = []
        for dd in (dc, de):
            if e in dd.index:
                cells += [f"{dd.loc[e,'est']:.4f}", f"({dd.loc[e,'se']:.4f})"]
            else:
                cells += ["--", ""]
        fh.write(f"$e = {e}$ & " + " & ".join(cells) + " \\\\\n")
    fh.write("\\midrule\n")
    fh.write(f"Overall ATT ($e\\in[0,{POST}]$) & {cs_civ[0]:.4f} & "
             f"({cs_civ[1]:.4f}) & {cs_enf[0]:.4f} & ({cs_enf[1]:.4f}) \\\\\n")
    fh.write(f"Joint lead test $p$ & \\multicolumn{{2}}{{c}}{{{cs_civ[3]:.3f}}} "
             f"& \\multicolumn{{2}}{{c}}{{{cs_enf[3]:.3f}}} \\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")

print("referee robustness battery complete", flush=True)

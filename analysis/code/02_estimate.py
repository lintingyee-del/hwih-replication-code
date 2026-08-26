# -*- coding: utf-8 -*-
"""HWIH replication pipeline: step 02 — estimation.

Designs:
  A. Triple-diff (market vs traffic placebo): post x exposure x market
  B. Within-market dose DiD: post x exposure (4 mechanism outcomes + log count)
  C. Event studies (dynamic post x exposure within market), pretrend joint test
  D. Mechanism joint co-movement test (backstop DOWN and rel-failure UP), stacked
  E. Robustness: theft placebo, drop 2020 (COVID), exposure components,
     high-quality offense-date subsample flag via x_spanshare control
Inference: province cluster + wild cluster bootstrap (9,999 reps, Rademacher).
Output: output/results.csv (long format), output/eventstudy.csv
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
import pandas as pd, numpy as np, pyfixest as pf, json, os

DATA = str(_REP_PROJECT / "data")
OUTD = str(_REP_PROJECT / "output")
os.makedirs(OUTD, exist_ok=True)

p = pd.read_parquet(f"{DATA}/panel_month.parquet")
p["month"] = p["judgment_month"].astype("str")
p["pref_group"] = p["prefecture_code"] + "_" + p["analysis_group"]
p["prov_month"] = p["province"] + "_" + p["month"]
p["group_month"] = p["analysis_group"] + "_" + p["month"]
p["market"] = (p["analysis_group"] == "market").astype(int)
p["post"] = p["post_judgment"].astype(int)
p["px"] = p["post"] * p["exposure_z"]
p["pxm"] = p["px"] * p["market"]
p["pm"] = p["post"] * p["market"]
p["logn"] = np.log(p["n_cases"])
p = p[p["n_fact"] > 0].copy()
p["prov_id"] = pd.factorize(p["province"])[0]

OUTCOMES = ["y_backstop", "y_relational", "y_rel_failure", "y_formalization"]
rows = []

from _wild import wild_score_p, wild_p  # corrected shared WCR bootstrap


def run(tag, fml, df, coef, weights=None, wild=True):
    m = pf.feols(fml, data=df, vcov={"CRV1": "prov_id"}, weights=weights)
    est = m.coef()[coef]; se = m.se()[coef]; pv = m.pvalue()[coef]
    wp = np.nan
    if wild:
        try:
            wp = wild_score_p(fml, df, coef, weights)
        except Exception:
            wp = np.nan
    rows.append(dict(tag=tag, coef=coef, est=est, se=se, p=pv, wild_p=wp,
                     n=int(m._N), fml=fml))
    print(f"{tag:42s} {coef:6s} {est: .5f} ({se:.5f}) p={pv:.4f} wild={wp:.4f} N={m._N}")
    return m

# ---------------- A. triple-diff: market vs traffic placebo -------------------
tp = p[p["analysis_group"].isin(["market", "placebo"])].copy()
for y in OUTCOMES:
    run(f"A_triplediff_{y}",
        f"{y} ~ pxm + px + pm + x_factshare + x_spanshare | pref_group + prov_month + group_month",
        tp, "pxm", weights="n_fact")

# ---------------- B. within-market dose DiD ----------------------------------
mk = p[p["analysis_group"] == "market"].copy()
mk["pref"] = mk["prefecture_code"]
for y in OUTCOMES:
    run(f"B_market_dose_{y}",
        f"{y} ~ px + x_factshare + x_spanshare | pref + prov_month",
        mk, "px", weights="n_fact")
run("B_market_dose_logn", "logn ~ px + x_factshare + x_spanshare | pref + prov_month",
    mk, "px")
# violence margin: 故意伤害 cells, backstop usage
vi = p[p["analysis_group"] == "violence"].copy(); vi["pref"] = vi["prefecture_code"]
run("B_violence_dose_y_backstop",
    "y_backstop ~ px + x_factshare + x_spanshare | pref + prov_month",
    vi, "px", weights="n_fact")

# ---------------- C. event studies (within market) ---------------------------
BINS = [(-24,-19),(-18,-13),(-12,-7),(-6,-1),(0,5),(6,11),(12,17),(18,28)]
REF = (-6,-1)
es_rows = []
for y in ["y_backstop", "y_rel_failure", "y_formalization"]:
    dfm = mk[(mk["event_time"] >= -24) & (mk["event_time"] <= 28)].copy()
    terms = []
    for lo, hi in BINS:
        if (lo, hi) == REF: continue
        nm = f"b_{lo}_{hi}".replace("-", "m")
        dfm[nm] = ((dfm["event_time"] >= lo) & (dfm["event_time"] <= hi)).astype(int) * dfm["exposure_z"]
        terms.append(nm)
    m = pf.feols(f"{y} ~ {' + '.join(terms)} + x_factshare + x_spanshare | pref + prov_month",
                 data=dfm, vcov={"CRV1": "prov_id"}, weights="n_fact")
    # joint pretrend test on lead bins (R matrix over coefficient vector)
    leads = [t for t in terms if t.startswith("b_m") and int(t.split("_")[1][1:]) >= 7]
    names = list(m.coef().index)
    R = np.zeros((len(leads), len(names)))
    for i, t in enumerate(leads):
        R[i, names.index(t)] = 1.0
    try:
        wald = m.wald_test(R=R)
        pre_p = float(wald.get("pvalue", np.nan)) if hasattr(wald, "get") else float(wald[1])
    except Exception:
        b = m.coef()[leads].values
        V = m._vcov[np.ix_([names.index(t) for t in leads], [names.index(t) for t in leads])]
        stat = float(b @ np.linalg.solve(V, b))
        from scipy import stats as sps
        pre_p = float(1 - sps.chi2.cdf(stat, len(leads)))
    for t in terms:
        lo, hi = t[2:].replace("m", "-").rsplit("_", 1)
        es_rows.append(dict(outcome=y, bin_lo=int(lo), bin_hi=int(hi),
                            est=m.coef()[t], se=m.se()[t], pretrend_p=pre_p))
    print(f"ES {y}: pretrend joint p = {pre_p}")

# ---------------- D. joint co-movement (stacked SUR-style) --------------------
st = []
for y, lab in [("y_backstop", 0), ("y_rel_failure", 1)]:
    d = mk[["prefecture_code","province","prov_id","month","prov_month","px","x_factshare",
            "x_spanshare","n_fact", y]].rename(columns={y: "y"}).copy()
    d["eq"] = lab
    st.append(d)
st = pd.concat(st)
st["px_backstop"] = st["px"] * (st["eq"] == 0)
st["px_relfail"] = st["px"] * (st["eq"] == 1)
st["pref_eq"] = st["prefecture_code"] + "_" + st["eq"].astype(str)
st["provmonth_eq"] = st["prov_month"] + "_" + st["eq"].astype(str)
ms = pf.feols("y ~ px_backstop + px_relfail + x_factshare + x_spanshare | pref_eq + provmonth_eq",
              data=st, vcov={"CRV1": "prov_id"}, weights="n_fact")
names_s = list(ms.coef().index)
idx = [names_s.index("px_backstop"), names_s.index("px_relfail")]
b2 = ms.coef().values[idx]
V2 = ms._vcov[np.ix_(idx, idx)]
from scipy import stats as sps
joint_p = float(1 - sps.chi2.cdf(float(b2 @ np.linalg.solve(V2, b2)), 2))
rows.append(dict(tag="D_joint_stacked", coef="px_backstop", est=ms.coef()["px_backstop"],
                 se=ms.se()["px_backstop"], p=ms.pvalue()["px_backstop"], wild_p=np.nan,
                 n=int(ms._N), fml="stacked"))
rows.append(dict(tag="D_joint_stacked", coef="px_relfail", est=ms.coef()["px_relfail"],
                 se=ms.se()["px_relfail"], p=ms.pvalue()["px_relfail"], wild_p=np.nan,
                 n=int(ms._N), fml=f"joint_wald_p={joint_p}"))
print(f"D joint: backstop={ms.coef()['px_backstop']:.5f}, relfail={ms.coef()['px_relfail']:.5f}, joint p={joint_p}")

# ---------------- E. robustness ----------------------------------------------
tt = p[p["analysis_group"].isin(["market", "theft"])].copy()
for y in ["y_backstop", "y_rel_failure"]:
    run(f"E_theftplacebo_{y}",
        f"{y} ~ pxm + px + pm + x_factshare + x_spanshare | pref_group + prov_month + group_month",
        tt, "pxm", weights="n_fact")
mk19 = mk[mk["month"] < "2020-01"].copy()
for y in ["y_backstop", "y_rel_failure"]:
    run(f"E_pre2020_{y}", f"{y} ~ px + x_factshare + x_spanshare | pref + prov_month",
        mk19, "px", weights="n_fact")
for comp in ["direct_share_z", "coercive_rate_z"]:
    mk["pxc"] = mk["post"] * mk[comp]
    run(f"E_expcomp_{comp}", "y_rel_failure ~ pxc + x_factshare + x_spanshare | pref + prov_month",
        mk, "pxc", weights="n_fact")

pd.DataFrame(rows).to_csv(f"{OUTD}/results.csv", index=False)
pd.DataFrame(es_rows).to_csv(f"{OUTD}/eventstudy.csv", index=False)
print("saved results.csv / eventstudy.csv")

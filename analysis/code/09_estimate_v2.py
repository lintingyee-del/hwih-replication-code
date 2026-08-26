# -*- coding: utf-8 -*-
"""6B step 09 — v2 estimation: civil judicialization tests + criminal re-run.

Civil (model predictions):
  C1 litigation flow: asinh(cases), Post x H x RelationalCause triple-diff
  C2 origination cohort: ex-ante documentation among newly signed loans
  C3 prices: recorded monthly interest, lending cases, window <= 2020-07
  C5 composition: relational share among litigated lending disputes
  C6 mediation margin; plus civil backstop-collection residue (falls)
Criminal v2 (new dictionaries, new exposure):
  backstop/detention-debt dose responses; telecom-fraud split diagnostics
Inference: province CRV1 + wild-score cluster bootstrap (9,999).
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
import pandas as pd, numpy as np, pyfixest as pf, os

DATA = str(_REP_PROJECT / "data")
OUTD = str(_REP_PROJECT / "output")
rows = []

from _wild import wild_score_p, wild_p  # corrected shared WCR bootstrap


def run(tag, fml, df, coef, weights=None):
    m = pf.feols(fml, data=df, vcov={"CRV1": "prov_id"}, weights=weights)
    try: wp = wild_score_p(fml, df, coef, weights, "prov_id")
    except Exception: wp = np.nan
    rows.append(dict(tag=tag, coef=coef, est=m.coef()[coef], se=m.se()[coef],
                     p=m.pvalue()[coef], wild_p=wp, n=int(m._N)))
    print(f"{tag:40s} {m.coef()[coef]: .5f} ({m.se()[coef]:.5f}) "
          f"p={m.pvalue()[coef]:.4f} wild={wp:.3f} N={m._N}")
    return m

# ================= CIVIL =====================================================
cp = pd.read_parquet(f"{DATA}/civil_panel.parquet")
cp = cp[cp["cause_family"].isin(["relational", "placebo"])].copy()
cp["month"] = cp["jmonth"].astype(str)
cp["prov_id"] = pd.factorize(cp["province"])[0]
cp["rel"] = (cp["cause_family"] == "relational").astype(int)
cp["H"] = cp["exposure_v2_z"]
cp["px"] = cp["post"] * cp["H"]
cp["pxr"] = cp["px"] * cp["rel"]
cp["pr"] = cp["post"] * cp["rel"]
cp["asinh_n"] = np.arcsinh(cp["n_cases"])
cp["pref_cause"] = cp["prefecture_code"] + "_" + cp["cause"]
cp["prov_month"] = cp["province"] + "_" + cp["month"]
cp["cause_month"] = cp["cause"] + "_" + cp["month"]

# C1: litigation flow
run("C1_flow_asinh", "asinh_n ~ pxr + px + pr | pref_cause + prov_month + cause_month",
    cp, "pxr")
# C6: mediation margin + backstop residue in civil records
run("C6_mediated", "y_mediated ~ pxr + px + pr | pref_cause + prov_month + cause_month",
    cp, "pxr", weights="n_cases")
run("C6_backstop_collect",
    "y_backstop_collect ~ pxr + px + pr | pref_cause + prov_month + cause_month",
    cp, "pxr", weights="n_cases")

# C1 event study (relational cells only, exposure-interacted bins)
rel = cp[cp["rel"] == 1].copy()
rel["pref"] = rel["prefecture_code"]
BINS = [(-24,-19),(-18,-13),(-12,-7),(-6,-1),(0,5),(6,11),(12,17)]
terms = []
for lo, hi in BINS:
    if (lo, hi) == (-6, -1): continue
    nm = f"b_{lo}_{hi}".replace("-", "m")
    rel[nm] = ((rel["event_time"] >= lo) & (rel["event_time"] <= hi)).astype(int) * rel["H"]
    terms.append(nm)
mes = pf.feols(f"asinh_n ~ {' + '.join(terms)} | pref_cause + prov_month + cause_month",
               data=rel, vcov={"CRV1": "prov_id"})
es_rows = []
leads = [t for t in terms if t.startswith("b_m") and int(t.split("_")[1][1:]) >= 7]
names = list(mes.coef().index)
b = mes.coef()[leads].values
V = mes._vcov[np.ix_([names.index(t) for t in leads], [names.index(t) for t in leads])]
from scipy import stats as sps
pre_p = float(1 - sps.chi2.cdf(float(b @ np.linalg.solve(V, b)), len(leads)))
for t in terms:
    lo, hi = t[2:].replace("m", "-").rsplit("_", 1)
    es_rows.append(dict(outcome="civil_asinh_n", bin_lo=int(lo), bin_hi=int(hi),
                        est=mes.coef()[t], se=mes.se()[t], pretrend_p=pre_p))
print(f"C1 event study pretrend joint p = {pre_p:.3f}")

# ---- case-level lending tests ------------------------------------------------
cc = pd.read_parquet(f"{DATA}/civil_case.parquet",
                     columns=["case_no","cause","prefecture_code","province","jmonth",
                              "post","insp_month","evid_iou","evid_transfer","evid_any",
                              "rel_txn","monthly_rate_pct","amount_yuan","orig_year",
                              "doc_len","doc_type"])
ex = pd.read_parquet(f"{DATA}/exposure_v2.parquet")[["prefecture_code","exposure_v2_z"]]
ld = cc[cc["cause"] == "民间借贷纠纷"].merge(ex, on="prefecture_code")
ld["prov_id"] = pd.factorize(ld["province"])[0]
ld["H"] = ld["exposure_v2_z"]
ld["px"] = ld["post"] * ld["H"]
ld["month"] = ld["jmonth"].astype(str)
ld["prov_month"] = ld["province"] + "_" + ld["month"]
ld["pref"] = ld["prefecture_code"]
ld["logdoclen"] = np.log(ld["doc_len"].clip(lower=1))

# C5: composition — relational share among litigated lending disputes
run("C5_rel_share", "rel_txn ~ px + logdoclen | pref + prov_month", ld, "px")

# C3: prices — recorded monthly rate, window <= 2020-07 (pre 4xLPR cap)
lr = ld[(ld["monthly_rate_pct"] > 0) & (ld["monthly_rate_pct"] <= 10)
        & (ld["month"] <= "2020-07")].copy()
run("C3_rate", "monthly_rate_pct ~ px + logdoclen | pref + prov_month", lr, "px")

# C2: origination cohort — ex-ante documentation of newly signed loans
oc = ld[(ld["orig_year"] >= 2012) & (ld["orig_year"] <= 2020)].copy()
oc["insp_year"] = pd.to_datetime(oc["insp_month"]).dt.year
oc["post_cohort"] = (oc["orig_year"] >= oc["insp_year"]).astype(int)
oc["pcx"] = oc["post_cohort"] * oc["H"]
oc["oy"] = oc["orig_year"].astype(str)
for y, tag in [("evid_iou", "C2_orig_iou"), ("evid_transfer", "C2_orig_transfer")]:
    run(tag, f"{y} ~ pcx + logdoclen | pref + oy + month", oc, "pcx")

# ================= CRIMINAL v2 ===============================================
kp = pd.read_parquet(f"{DATA}/crim_panel_v2.parquet")
kp = kp[kp["n_cases"] > 0].copy()
kp["month"] = kp["jmonth"].astype(str)
kp["prov_id"] = pd.factorize(kp["province"])[0]
kp["H"] = kp["exposure_v2_z"]
kp["px"] = kp["post"] * kp["H"]
kp["pref"] = kp["prefecture_code"]
kp["prov_month"] = kp["province"] + "_" + kp["month"]
mk = kp[kp["family"] == "market"].copy()
run("K2_market_backstop", "y_backstop ~ px + x_doclen | pref + prov_month",
    mk, "px", weights="n_cases")
run("K2_market_relfail", "y_rel_fail ~ px + x_doclen | pref + prov_month",
    mk, "px", weights="n_cases")
run("K2_market_formalization", "y_formalization ~ px + x_doclen | pref + prov_month",
    mk, "px", weights="n_cases")
en = kp[kp["family"] == "enforcementcrime"].copy()
run("K2_enforcement_detentiondebt", "y_detention_debt ~ px + x_doclen | pref + prov_month",
    en, "px", weights="n_cases")
run("K2_enforcement_asinhN", "asinh_n ~ px | pref + prov_month",
    en.assign(asinh_n=np.arcsinh(en["n_cases"])), "px")
for fam, tag in [("violence", "K2_violence_backstop"), ("theft", "K2_theft_backstop")]:
    fk = kp[kp["family"] == fam].copy()
    run(tag, "y_backstop ~ px + x_doclen | pref + prov_month", fk, "px", weights="n_cases")

pd.DataFrame(rows).to_csv(f"{OUTD}/results_v2.csv", index=False)
pd.DataFrame(es_rows).to_csv(f"{OUTD}/eventstudy_v2.csv", index=False)
print("saved results_v2.csv / eventstudy_v2.csv")

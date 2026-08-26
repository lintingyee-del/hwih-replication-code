# -*- coding: utf-8 -*-
"""6B step 14 — stacked/clean-control DiD.

Sub-experiment: Round-1 provinces (inspected Jul-Sep 2018) vs not-yet-treated
Round-2/3 provinces (inspected Apr/Jun 2019). Clean window 2017-01..2019-03:
controls are strictly untreated throughout. Coefficient: Post x Treat x H
(dose-in-treated vs dose-in-untreated), plus binary Post x Treat.
Outcomes: criminal market backstop (6A flags, v1 measure) and civil relational
flow (asinh cases). Round-2-vs-Round-3 window is 2 months — noted, not used.
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
import pandas as pd, numpy as np, pyfixest as pf

DATA = str(_REP_PROJECT / "data")
OUTD = str(_REP_PROJECT / "output")
rows = []

from _wild import wild_score_p, wild_p  # corrected shared WCR bootstrap


def run(tag, fml, df, coef, weights=None):
    m = pf.feols(fml, data=df, vcov={"CRV1": "prov_id"}, weights=weights)
    try: wp = wild_score_p(fml, df, coef, weights)
    except Exception: wp = np.nan
    rows.append(dict(tag=tag, coef=coef, est=m.coef()[coef], se=m.se()[coef],
                     p=m.pvalue()[coef], wild_p=wp, n=int(m._N)))
    print(f"{tag:38s} {coef:4s} {m.coef()[coef]: .5f} ({m.se()[coef]:.5f}) "
          f"p={m.pvalue()[coef]:.4f} wild={wp:.3f} N={m._N}")

CLEAN_START = pd.Timestamp("2017-01-01")
CLEAN_END = pd.Timestamp("2019-04-01")
POST0 = "2018-09"

# ---- criminal market backstop (v1 measure, 6A flags) -------------------------
p = pd.read_parquet(f"{DATA}/panel_month.parquet")
p = p[(p["analysis_group"] == "market") & (p["n_fact"] > 0)].copy()
p["judgment_date"] = pd.to_datetime(p["judgment_month"], errors="coerce")
p = p[(p["judgment_date"] >= CLEAN_START) &
      (p["judgment_date"] < CLEAN_END)].copy()
p["month"] = p["judgment_date"].dt.strftime("%Y-%m")
p["treat"] = (p["inspection_round"] == 1).astype(int)
p["postc"] = (p["month"] >= POST0).astype(int)
p["prov_id"] = pd.factorize(p["province"])[0]
p["pref"] = p["prefecture_code"]
p["prov_month"] = p["province"] + "_" + p["month"]
p["pt"] = p["postc"] * p["treat"]
p["pth"] = p["pt"] * p["exposure_z"]
p["ph"] = p["postc"] * p["exposure_z"]
mth = p["month"].astype("category")
p["month_fe"] = p["month"]
run("S1_crim_backstop_binary",
    "y_backstop ~ pt + x_factshare | pref + month_fe", p, "pt", weights="n_fact")
run("S1_crim_backstop_dose",
    "y_backstop ~ pth + ph + pt + x_factshare | pref + month_fe", p, "pth",
    weights="n_fact")

# ---- civil relational flow ---------------------------------------------------
c = pd.read_parquet(f"{DATA}/civil_panel.parquet")
c = c[c["cause_family"] == "relational"].copy()
c["judgment_date"] = pd.to_datetime(c["jmonth"], errors="coerce")
c = c[(c["judgment_date"] >= CLEAN_START) &
      (c["judgment_date"] < CLEAN_END)].copy()
c["month"] = c["judgment_date"].dt.strftime("%Y-%m")
sched = pd.read_parquet(f"{DATA}/panel_month.parquet")[["province","inspection_round"]] \
    .drop_duplicates()
c = c.merge(sched, on="province", how="left")
c["treat"] = (c["inspection_round"] == 1).astype(int)
c["postc"] = (c["month"] >= POST0).astype(int)
c["prov_id"] = pd.factorize(c["province"])[0]
c["pref_cause"] = c["prefecture_code"] + "_" + c["cause"]
c["month_fe"] = c["month"]
c["asinh_n"] = np.arcsinh(c["n_cases"])
c["pt"] = c["postc"] * c["treat"]
c["pth"] = c["pt"] * c["exposure_v2_z"]
c["ph"] = c["postc"] * c["exposure_v2_z"]
run("S1_civil_flow_binary",
    "asinh_n ~ pt | pref_cause + month_fe", c, "pt")
run("S1_civil_flow_dose",
    "asinh_n ~ pth + ph + pt | pref_cause + month_fe", c, "pth")

old = pd.read_csv(f"{OUTD}/results_v2.csv")
new = pd.concat([old[~old["tag"].str.startswith("S1_")], pd.DataFrame(rows)])
new.to_csv(f"{OUTD}/results_v2.csv", index=False)
print("stacked results appended")

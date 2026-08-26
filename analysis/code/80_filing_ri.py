# -*- coding: utf-8 -*-
"""6B step 80 — wave-timing randomization inference for the filing-clock
clean-window specification (the one RI cell missing from the headline table).

Replicates step 62's filing-dated cell build (relational causes, extracted
filing month, 0<=duration<=270 days, window 2017-01..2019-03), then reassigns
first-wave status across the 31 provinces (count held at ten) 999 times and
refits asinh_n ~ pth + ph + pt | pref_cause + month.
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
import os, sys, io
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd
import pyfixest as pf

DATA = str(_REP_PROJECT / "data")
WINDOW = ("2017-01", "2019-03"); POST0 = "2018-09"

cc = pd.read_parquet(f"{DATA}/civil_case.parquet",
                     columns=["case_no", "cause", "cause_family",
                              "prefecture_code", "province", "jmonth"])
rel = cc[cc["cause_family"] == "relational"].copy()
fil = pd.read_parquet(f"{DATA}/civil_filing.parquet").rename(
    columns={"案号": "case_no"})
rel = rel.merge(fil[["case_no", "filing_ymd"]], on="case_no", how="left")
rel["fd"] = pd.to_datetime(rel["filing_ymd"], errors="coerce")
rel["jd"] = pd.to_datetime(rel["jmonth"], errors="coerce")
rel["dur"] = (rel["jd"] - rel["fd"]).dt.days
ok = rel["fd"].notna() & rel["dur"].between(0, 270)
rel = rel[ok].copy()
rel["fm"] = rel["fd"].dt.strftime("%Y-%m")
rel = rel[(rel["fm"] >= WINDOW[0]) & (rel["fm"] <= WINDOW[1])]

sched = pd.read_parquet(f"{DATA}/panel_month.parquet")[
    ["province", "inspection_round"]].drop_duplicates()
ex = pd.read_parquet(f"{DATA}/exposure_v2.parquet")[
    ["prefecture_code", "exposure_v2_z"]]
g = (rel.groupby(["prefecture_code", "province", "cause", "fm"]).size()
     .rename("n").reset_index().rename(columns={"fm": "month"})
     .merge(sched, on="province").merge(ex, on="prefecture_code")
     .dropna(subset=["exposure_v2_z", "inspection_round"]))
g["H"] = g["exposure_v2_z"]
g["postc"] = (g["month"] >= POST0).astype(int)
g["pref_cause"] = g["prefecture_code"] + "_" + g["cause"]
g["asinh_n"] = np.arcsinh(g["n"])
treat_obs = dict(g[["province", "inspection_round"]].drop_duplicates().assign(
    t=lambda d: (d["inspection_round"] == 1).astype(int))[["province", "t"]].values)
provs = sorted(treat_obs)
n_treat = sum(treat_obs[p] for p in provs)
FML = "asinh_n ~ pth + ph + pt | pref_cause + month"


def fit(tmap):
    d = g.assign(tr=g["province"].map(tmap).astype(float))
    d["pt"] = d["postc"] * d["tr"]
    d["pth"] = d["pt"] * d["H"]
    d["ph"] = d["postc"] * d["H"]
    return float(pf.feols(FML, data=d, vcov="iid").coef()["pth"])


b_obs = fit(treat_obs)
print(f"observed filing-clock pth = {b_obs:+.4f} (cells {len(g):,}; "
      f"expect ~+0.1385)", flush=True)
rng = np.random.default_rng(42)
hits = 0
for r in range(999):
    lab = np.zeros(len(provs), int)
    lab[rng.choice(len(provs), n_treat, replace=False)] = 1
    if abs(fit(dict(zip(provs, lab)))) >= abs(b_obs): hits += 1
    if (r + 1) % 100 == 0: print(f"   {r+1}/999", flush=True)
print(f"filing-clock wave-timing RI p = {(1+hits)/1000:.3f}", flush=True)

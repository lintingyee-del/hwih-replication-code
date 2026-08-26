# -*- coding: utf-8 -*-
"""6B step 33 — Cut 1: does the clean-window judicialization rise load on
individual/acquaintance lending or on organizational/professional lending?
Merge the org-plaintiff flag (step 32) onto clean-window lending cases, aggregate to
prefecture-month cells SEPARATELY for individual-plaintiff and org-plaintiff cases,
and run the clean-window dose (Post x Treat x H) on each.

Prediction: backstop-removal mechanism -> rise concentrated in INDIVIDUAL
(acquaintance) cells; manufactured-book confound -> rise in ORG (professional) cells.
Opposite sides => decisive, not 'bound + upper'.
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
WIN = ("2017-01", "2019-03"); POST0 = "2018-09"

from _wild import wild_score_p, wild_p  # corrected shared WCR bootstrap


# ---- clean-window lending cases + org flag ----------------------------------
cc = pd.read_parquet(f"{DATA}/civil_case.parquet",
                     columns=["case_no", "cause", "prefecture_code", "province", "jmonth"])
lend = cc[cc["cause"] == "民间借贷纠纷"].copy()
lend["month"] = lend["jmonth"].astype(str).str[:7]
lend = lend[(lend["month"] >= WIN[0]) & (lend["month"] <= WIN[1])]
org = pd.read_parquet(f"{DATA}/civil_party_orgflag.parquet").rename(columns={"案号": "case_no"})
lend = lend.merge(org[["case_no", "first_org"]], on="case_no", how="left")
match = lend["first_org"].notna().mean()
print(f"[merge] clean-window lending {len(lend)}, org-flag match rate {match:.3f}", flush=True)
lend = lend[lend["first_org"].notna()].copy()
lend["org_pl"] = lend["first_org"].astype(bool)
print(f"[split] org-plaintiff {lend['org_pl'].mean():.3f} of matched lending", flush=True)

sched = pd.read_parquet(f"{DATA}/panel_month.parquet")[["province", "inspection_round"]].drop_duplicates()
ex = pd.read_parquet(f"{DATA}/exposure_v2.parquet")[["prefecture_code", "exposure_v2_z"]]

def dose(df, label):
    g = (df.groupby(["prefecture_code", "province", "month"]).size()
         .rename("n").reset_index()
         .merge(sched, on="province", how="left").merge(ex, on="prefecture_code", how="inner")
         .dropna(subset=["exposure_v2_z", "inspection_round"]))
    g["H"] = g["exposure_v2_z"]; g["pref"] = g["prefecture_code"]
    g["treat"] = (g["inspection_round"] == 1).astype(int)
    g["postc"] = (g["month"] >= POST0).astype(int)
    g["prov_id"] = pd.factorize(g["province"])[0]
    g["pt"] = g["postc"] * g["treat"]; g["pth"] = g["pt"] * g["H"]; g["ph"] = g["postc"] * g["H"]
    g["y"] = np.arcsinh(g["n"])
    m = pf.feols("y ~ pth + ph + pt | pref + month", data=g, vcov={"CRV1": "prov_id"})
    wp = wild_score_p("y ~ pth + ph + pt | pref + month", g, "pth")
    b, se, p = m.coef()["pth"], m.se()["pth"], m.pvalue()["pth"]
    print(f"[{label:11s}] N_cells={m._N:5d}  pth={b:+.4f} ({se:.4f})  CRV1 p={p:.3f}  wild p={wp:.3f}", flush=True)
    return dict(group=label, b=b, se=se, p=p, wild_p=wp, n_cells=int(m._N),
                n_cases=int(df.shape[0]))

print("\n===== Cut 1: individual/acquaintance vs organizational/professional plaintiff =====", flush=True)
res_ind = dose(lend[~lend["org_pl"]], "individual")
res_org = dose(lend[lend["org_pl"]], "organizational")
res_all = dose(lend, "all lending")
pd.DataFrame([res_ind, res_org, res_all]).to_csv(f"{DATA}/../output/cut1_orgvsindiv.csv", index=False)
print("\n[verdict] mechanism predicts the rise on the INDIVIDUAL side; "
      "confound (professional-book dump) predicts it on the ORGANIZATIONAL side.", flush=True)
print(f"  individual pth = {res_ind['b']:+.4f} (wild {res_ind['wild_p']:.3f}); "
      f"organizational pth = {res_org['b']:+.4f} (wild {res_org['wild_p']:.3f})", flush=True)
print("[done] wrote output/cut1_orgvsindiv.csv", flush=True)

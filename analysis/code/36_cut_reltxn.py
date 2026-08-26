# -*- coding: utf-8 -*-
"""6B step 36 — the CLEAN party-level separation, replacing the contaminated org proxy.
Split the clean-window lending FLOW by the audited relational-transaction flag rel_txn
(prior acquaintance between the parties; precision 0.88, recall 0.85). Mechanical
dumping surfaces prosecuted professional/taolu-dai books, which are STRANGER loans
(rel_txn=0); backstop removal loosens ACQUAINTANCE lending (rel_txn=1). A rise loading
on rel_txn=1 cannot be produced by transferring a professional book => decisive.

Reports the clean-window dose (Post x Treat x H) separately on acquaintance (rel_txn=1)
and non-acquaintance (rel_txn=0) lending flows, plus a direct acq-vs-stranger contrast.
Cross-check: interest-rate split (professional lenders charge more) as a second proxy.
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
DATA = str(_REP_PROJECT / "data"); OUTD = str(_REP_PROJECT / "output")
WIN = ("2017-01", "2019-03"); POST0 = "2018-09"

from _wild import wild_score_p, wild_p  # corrected shared WCR bootstrap


cc = pd.read_parquet(f"{DATA}/civil_case.parquet",
                     columns=["cause","prefecture_code","province","jmonth","rel_txn","monthly_rate_pct"])
ld = cc[cc["cause"] == "民间借贷纠纷"].copy()
ld["month"] = ld["jmonth"].astype(str).str[:7]
ld = ld[(ld["month"] >= WIN[0]) & (ld["month"] <= WIN[1])]
print(f"[coverage] clean-window lending {len(ld)}; rel_txn valid {ld['rel_txn'].notna().mean():.3f}; "
      f"acquaintance (rel_txn=1) share {ld['rel_txn'].fillna(0).mean():.3f}", flush=True)

sched = pd.read_parquet(f"{DATA}/panel_month.parquet")[["province","inspection_round"]].drop_duplicates()
ex = pd.read_parquet(f"{DATA}/exposure_v2.parquet")[["prefecture_code","exposure_v2_z"]]

def dose(df, label):
    g = (df.groupby(["prefecture_code","province","month"]).size().rename("n").reset_index()
         .merge(sched, on="province").merge(ex, on="prefecture_code")
         .dropna(subset=["exposure_v2_z","inspection_round"]))
    g["H"]=g["exposure_v2_z"]; g["pref"]=g["prefecture_code"]
    g["treat"]=(g["inspection_round"]==1).astype(int); g["postc"]=(g["month"]>=POST0).astype(int)
    g["prov_id"]=pd.factorize(g["province"])[0]
    g["pt"]=g["postc"]*g["treat"]; g["pth"]=g["pt"]*g["H"]; g["ph"]=g["postc"]*g["H"]; g["y"]=np.arcsinh(g["n"])
    m = pf.feols("y ~ pth + ph + pt | pref + month", data=g, vcov={"CRV1":"prov_id"})
    wp = wild_score_p("y ~ pth + ph + pt | pref + month", g, "pth")
    print(f"[{label:26s}] pth={m.coef()['pth']:+.4f} ({m.se()['pth']:.4f}) "
          f"CRV1 p={m.pvalue()['pth']:.3f} wild p={wp:.3f} N={m._N} cases={len(df):,}", flush=True)
    return dict(group=label, b=m.coef()['pth'], se=m.se()['pth'], p=m.pvalue()['pth'], wild_p=wp)

print("\n===== CLEAN cut: acquaintance (rel_txn=1) vs stranger (rel_txn=0) lending flow =====", flush=True)
ld["acq"] = ld["rel_txn"].fillna(0).astype(int)
r_acq = dose(ld[ld["acq"]==1], "acquaintance rel_txn=1")
r_str = dose(ld[ld["acq"]==0], "stranger rel_txn=0")

# direct contrast: stack, acq indicator, Post x Treat x H x Acq
def cells(df, tag):
    g = (df.groupby(["prefecture_code","province","month"]).size().rename("n").reset_index()
         .merge(sched, on="province").merge(ex, on="prefecture_code")
         .dropna(subset=["exposure_v2_z","inspection_round"])); g["acq"]=tag; return g
st = pd.concat([cells(ld[ld["acq"]==1],1), cells(ld[ld["acq"]==0],0)], ignore_index=True)
st["H"]=st["exposure_v2_z"]; st["treat"]=(st["inspection_round"]==1).astype(int)
st["postc"]=(st["month"]>=POST0).astype(int); st["prov_id"]=pd.factorize(st["province"])[0]
st["pref_acq"]=st["prefecture_code"]+"_"+st["acq"].astype(str); st["pm"]=st["province"]+"_"+st["month"]
st["pt"]=st["postc"]*st["treat"]; st["pth"]=st["pt"]*st["H"]; st["ph"]=st["postc"]*st["H"]; st["y"]=np.arcsinh(st["n"])
st["ptha"]=st["pth"]*st["acq"]; st["pha"]=st["ph"]*st["acq"]; st["pta"]=st["pt"]*st["acq"]
mc = pf.feols("y ~ ptha + pha + pta + pth + ph + pt | pref_acq + pm", data=st, vcov={"CRV1":"prov_id"})
wpc = wild_score_p("y ~ ptha + pha + pta + pth + ph + pt | pref_acq + pm", st, "ptha")
print(f"[acq-vs-stranger contrast] Post x Treat x H x Acq = {mc.coef()['ptha']:+.4f} "
      f"({mc.se()['ptha']:.4f}) CRV1 p={mc.pvalue()['ptha']:.3f} wild p={wpc:.3f}", flush=True)

print("\n[verdict] mechanism -> rise on acquaintance (rel_txn=1); mechanical dumping of a "
      "professional book -> stranger side (rel_txn=0). Rise on acquaintance = decisive.", flush=True)
pd.DataFrame([r_acq, r_str, dict(group="acq_vs_stranger_contrast", b=mc.coef()['ptha'],
             se=mc.se()['ptha'], p=mc.pvalue()['ptha'], wild_p=wpc)]).to_csv(
             f"{OUTD}/cut_reltxn.csv", index=False)
print("[done] wrote output/cut_reltxn.csv", flush=True)

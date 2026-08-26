# -*- coding: utf-8 -*-
"""6B step 37 — is the acquaintance (rel_txn=1) rise contaminated by professional
lenders lending to acquaintances? The exact repeat-filer test needs cross-case
plaintiff-name matching, which per-document anonymization defeats. Proxy the same
question: professional lending carries two signatures a genuine one-shot acquaintance
loan lacks -- an organizational plaintiff, and a HIGH interest rate (business lending
near the usury cap; genuine 亲友 loans are often zero/low interest). Subtract the
professional-signature cases from the acquaintance bucket and see whether the +0.31
clean-window rise survives among genuinely one-shot acquaintance lending.

Buckets (all within rel_txn=1, clean window):
  (a) all acquaintance
  (b) acquaintance & INDIVIDUAL plaintiff (first_org=0)
  (c) acquaintance & individual & LOW interest (<=1.5%/mo or missing)  <- cleanest one-shot
Survival of +0.31 into (c) => clean, 'close'; collapse => contaminated, 'bound'.
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
                     columns=["case_no","cause","prefecture_code","province","jmonth","rel_txn","monthly_rate_pct"])
ld = cc[cc["cause"]=="民间借贷纠纷"].copy()
ld["month"]=ld["jmonth"].astype(str).str[:7]
ld = ld[(ld["month"]>=WIN[0])&(ld["month"]<=WIN[1])]
org = pd.read_parquet(f"{DATA}/civil_party_orgflag.parquet").rename(columns={"案号":"case_no"})
ld = ld.merge(org[["case_no","first_org"]], on="case_no", how="left")
acq = ld[ld["rel_txn"].fillna(0).astype(int)==1].copy()
acq["org"] = acq["first_org"].fillna(False).astype(bool)
r = acq["monthly_rate_pct"]
print(f"[acquaintance cases] N={len(acq):,}", flush=True)
print(f"  org-plaintiff share among acquaintance: {acq['org'].mean():.3f}", flush=True)
print(f"  rate present: {(r>0).mean():.3f}; among present, median {r[r>0].median():.2f}%/mo, "
      f"share >1.5%/mo {(r>1.5).mean():.3f}", flush=True)

sched = pd.read_parquet(f"{DATA}/panel_month.parquet")[["province","inspection_round"]].drop_duplicates()
ex = pd.read_parquet(f"{DATA}/exposure_v2.parquet")[["prefecture_code","exposure_v2_z"]]
def dose(df, label):
    g = (df.groupby(["prefecture_code","province","month"]).size().rename("n").reset_index()
         .merge(sched,on="province").merge(ex,on="prefecture_code")
         .dropna(subset=["exposure_v2_z","inspection_round"]))
    g["H"]=g["exposure_v2_z"]; g["pref"]=g["prefecture_code"]; g["treat"]=(g["inspection_round"]==1).astype(int)
    g["postc"]=(g["month"]>=POST0).astype(int); g["prov_id"]=pd.factorize(g["province"])[0]
    g["pt"]=g["postc"]*g["treat"]; g["pth"]=g["pt"]*g["H"]; g["ph"]=g["postc"]*g["H"]; g["y"]=np.arcsinh(g["n"])
    m=pf.feols("y ~ pth + ph + pt | pref + month",data=g,vcov={"CRV1":"prov_id"})
    wp=wild_score_p("y ~ pth + ph + pt | pref + month",g,"pth")
    print(f"[{label:38s}] pth={m.coef()['pth']:+.4f} ({m.se()['pth']:.4f}) "
          f"CRV1 p={m.pvalue()['pth']:.3f} wild p={wp:.3f} cases={len(df):,}", flush=True)
    return dict(bucket=label,b=m.coef()['pth'],se=m.se()['pth'],p=m.pvalue()['pth'],wild_p=wp,n=len(df))

print("\n===== does +0.31 survive stripping professional signatures? =====", flush=True)
lowint = (acq["monthly_rate_pct"].isna()) | (acq["monthly_rate_pct"]<=1.5)
res = [dose(acq, "(a) all acquaintance"),
       dose(acq[~acq["org"]], "(b) acq & individual plaintiff"),
       dose(acq[(~acq["org"]) & lowint], "(c) acq & individual & low-interest")]
print("\n[verdict] if (c) still ~+0.31 sharp -> clean one-shot acquaintance drives it "
      "-> 'close'. if it collapses -> professional-to-acquaintance contamination -> 'bound'.", flush=True)
pd.DataFrame(res).to_csv(f"{OUTD}/acq_contamination.csv", index=False)
print("[done] wrote output/acq_contamination.csv", flush=True)

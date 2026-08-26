# -*- coding: utf-8 -*-
"""6B step 38 — self-contained generator for the party-level exhibit: recompute the
rel_txn (acquaintance vs stranger) clean-window dose split and the contamination
robustness (strip professional signatures), then EMIT the paper macros
(numbers_partycut.tex) and a formatted table (tab_partycut.tex). Makes the §8.6
party-cut numbers reproducible from code rather than hand-typed.
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
DATA = str(_REP_PROJECT / "data"); TAB = str(_REP_PROJECT / "output" / "tables")
WIN = ("2017-01", "2019-03"); POST0 = "2018-09"

from _wild import wild_score_p, wild_p  # corrected shared WCR bootstrap


cc = pd.read_parquet(f"{DATA}/civil_case.parquet",
                     columns=["case_no","cause","prefecture_code","province","jmonth","rel_txn","monthly_rate_pct"])
ld = cc[cc["cause"]=="民间借贷纠纷"].copy(); ld["month"]=ld["jmonth"].astype(str).str[:7]
ld = ld[(ld["month"]>=WIN[0])&(ld["month"]<=WIN[1])]
org = pd.read_parquet(f"{DATA}/civil_party_orgflag.parquet").rename(columns={"案号":"case_no"})
ld = ld.merge(org[["case_no","first_org"]], on="case_no", how="left")
ld["acq"] = ld["rel_txn"].fillna(0).astype(int); ld["org"] = ld["first_org"].fillna(False).astype(bool)
sched = pd.read_parquet(f"{DATA}/panel_month.parquet")[["province","inspection_round"]].drop_duplicates()
ex = pd.read_parquet(f"{DATA}/exposure_v2.parquet")[["prefecture_code","exposure_v2_z"]]

def dose(df):
    g = (df.groupby(["prefecture_code","province","month"]).size().rename("n").reset_index()
         .merge(sched,on="province").merge(ex,on="prefecture_code")
         .dropna(subset=["exposure_v2_z","inspection_round"]))
    g["H"]=g["exposure_v2_z"]; g["pref"]=g["prefecture_code"]; g["treat"]=(g["inspection_round"]==1).astype(int)
    g["postc"]=(g["month"]>=POST0).astype(int); g["prov_id"]=pd.factorize(g["province"])[0]
    g["pt"]=g["postc"]*g["treat"]; g["pth"]=g["pt"]*g["H"]; g["ph"]=g["postc"]*g["H"]; g["y"]=np.arcsinh(g["n"])
    m=pf.feols("y ~ pth + ph + pt | pref + month",data=g,vcov={"CRV1":"prov_id"})
    return dict(b=m.coef()["pth"], se=m.se()["pth"], p=m.pvalue()["pth"],
                wp=wild_p("y ~ pth + ph + pt | pref + month",g,"pth"), N=len(df))

acq  = dose(ld[ld["acq"]==1]); strg = dose(ld[ld["acq"]==0])
lowint = (ld["monthly_rate_pct"].isna()) | (ld["monthly_rate_pct"]<=1.5)
bb = dose(ld[(ld["acq"]==1)&(~ld["org"])]); bc = dose(ld[(ld["acq"]==1)&(~ld["org"])&lowint])
org_share = ld.loc[ld["acq"]==1,"org"].mean()
print(f"acq {acq['b']:.3f}/{acq['wp']:.3f}  stranger {strg['b']:.3f}/{strg['wp']:.3f}  "
      f"clean {bc['b']:.3f}  org-share {org_share:.3f}", flush=True)

def st(x): s="***" if x<0.01 else "**" if x<0.05 else "*" if x<0.10 else ""; return s
def fmt(d): return f"{d['b']:.3f}{st(d['wp'])} & ({d['se']:.3f}) & {d['wp']:.3f} & {d['N']:,}"

with open(f"{TAB}/numbers_partycut.tex","w",encoding="utf-8") as f:
    m=lambda n,v:f.write(f"\\newcommand{{\\{n}}}{{{v}}}\n")
    f.write("% step 38 — party-level cut (auto-generated)\n")
    m("CutAcqFlow", f"{acq['b']:.2f}"); m("CutStrangerFlow", f"{strg['b']:.2f}")
    m("CutAcqCleanFlow", f"{bc['b']:.2f}"); m("CutAcqOrgShare", f"{org_share*100:.1f}")
    m("CutAcqFlowWildP", f"{acq['wp']:.3f}"); m("CutAcqN", f"{acq['N']:,}"); m("CutStrangerN", f"{strg['N']:,}")

with open(f"{TAB}/tab_partycut.tex","w",encoding="utf-8") as f:
    f.write("\\begin{tabular}{lcccc}\n\\toprule\n")
    f.write(" & Post$\\times$Treat$\\times H$ & (SE) & wild $p$ & Cases \\\\\n\\midrule\n")
    f.write("\\multicolumn{5}{l}{\\emph{Panel A. Acquaintance vs.\\ stranger lending flow}}\\\\\n")
    f.write(f"Acquaintance (rel.\\ txn) & {fmt(acq)} \\\\\n")
    f.write(f"Stranger & {fmt(strg)} \\\\\n")
    f.write("\\midrule\n\\multicolumn{5}{l}{\\emph{Panel B. Acquaintance rise, stripping professional signatures}}\\\\\n")
    f.write(f"All acquaintance & {fmt(acq)} \\\\\n")
    f.write(f"\\quad individual plaintiff only & {fmt(bb)} \\\\\n")
    f.write(f"\\quad {{}}+ low-interest only & {fmt(bc)} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
print("[done] wrote numbers_partycut.tex + tab_partycut.tex", flush=True)

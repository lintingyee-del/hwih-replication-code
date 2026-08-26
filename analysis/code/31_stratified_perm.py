# -*- coding: utf-8 -*-
"""6B step 31 — stratified randomization inference for the criminal de-militarization
joint index, answering the exchangeability critique. Free permutation across all 31
provinces is the wrong null if wave timing correlates with province crime. We stratify
provinces into pre-campaign crime terciles (province-mean exposure) and permute the
inspection clock ONLY WITHIN each tercile, preserving the crime-timing correlation.
Reports the wave x tercile crosstab (how confounded timing is with crime), the free
CrimJointP, and the stratified CrimJointP at the same REPS for apples-to-apples.
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
import numpy as np, pandas as pd, pyfixest as pf
DATA=str(_REP_PROJECT / "data")
REPS=2999; rng=np.random.default_rng(20260706)

pm=pd.read_parquet(f"{DATA}/panel_month.parquet")[
    ["province","insp_month","inspection_round"]].drop_duplicates()
pm["insp"]=pm["insp_month"].astype(str).str[:7]
# province pre-campaign crime intensity = province-mean exposure
ex=pd.read_parquet(f"{DATA}/exposure_v2.parquet").groupby("province")["exposure_v2_z"].mean()
pm=pm.merge(ex.rename("crime"),on="province",how="left")
pm["tercile"]=pd.qcut(pm["crime"],3,labels=[0,1,2]).astype(int)
PROVS=pm["province"].values; INSP_VALS=pm["insp"].values
TERC=pm["tercile"].values; ROUND=pm["inspection_round"].values
INSP_MAP0=dict(zip(PROVS,INSP_VALS))
print("wave x crime-tercile crosstab (rows=round, cols=tercile):",flush=True)
print(pd.crosstab(pm["inspection_round"],pm["tercile"]),flush=True)

def crim_frame(oc,fam,weight=True):
    k=pd.read_parquet(f"{DATA}/crim_panel_v2.parquet")
    k=k[(k["family"]==fam)&(k["n_cases"]>0)].dropna(subset=["exposure_v2_z"]).copy()
    k["month"]=k["jmonth"].astype(str).str[:7]; k["H"]=k["exposure_v2_z"]
    k["fe1"]=k["prefecture_code"]; k["fe2"]=k["province"]+"_"+k["month"]
    k["y"]=np.arcsinh(k["n_cases"]) if oc=="asinh_n" else k[oc]
    k["w"]=k["n_cases"].astype(float) if weight else 1.0
    return k,("w" if weight else None)
MARG=[crim_frame("y_backstop","market"),
      crim_frame("asinh_n","enforcementcrime",weight=False),
      crim_frame("y_detention_debt","enforcementcrime")]
def fit(d,w,im):
    x=d.copy(); ins=x["province"].map(im).values
    x["post"]=(x["month"].values>=ins).astype(int); x["px"]=x["post"].values*x["H"].values
    return float(pf.feols("y ~ px | fe1 + fe2",data=x,weights=w).coef()["px"])

groups=[np.where(TERC==t)[0] for t in (0,1,2)]
b_obs=np.array([fit(d,w,INSP_MAP0) for d,w in MARG])

def run(kind):
    B=np.full((REPS,3),np.nan)
    for r in range(REPS):
        v=INSP_VALS.copy()
        if kind=="free":
            v=rng.permutation(v)
        else:
            for g in groups: v[g]=rng.permutation(v[g])
        im=dict(zip(PROVS,v))
        for j,(d,w) in enumerate(MARG): B[r,j]=fit(d,w,im)
        if (r+1)%1000==0: print(f"  {kind} {r+1}/{REPS}",flush=True)
    mu,sd=B.mean(0),B.std(0,ddof=1)
    z=-(b_obs-mu)/sd; Z=-(B-mu)/sd
    return (1+np.sum(Z.sum(1)>=z.sum()))/(1+REPS)

p_free=run("free"); p_strat=run("stratified")
print(f"\n[obs] {np.round(b_obs,4)}",flush=True)
print(f"[result] CrimJointP free={p_free:.3f}  stratified(within crime-tercile)={p_strat:.3f}",flush=True)
pd.DataFrame([dict(scheme="free",p=p_free),dict(scheme="stratified_crimetercile",p=p_strat)]
             ).to_csv(str(_REP_PROJECT / 'output' / 'stratified_perm.csv').replace('\\', '/'),index=False)
print("[done] wrote stratified_perm.csv",flush=True)

# -*- coding: utf-8 -*-
"""6B step 30 — specification curve for the criminal de-militarization joint index,
answering the garden-of-forking-paths critique. Collect the wave-timing null
distribution ONCE for the 3 criminal margins (REPS draws), then compute the joint
randomization p for EVERY non-empty subset of margins under equal AND GLS weighting.
If \\CrimJointP is only significant for the full triple with detention-debt in it,
that is the vulnerability. Signs are theory-pinned (all criminal margins -).
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
from itertools import combinations
DATA=str(_REP_PROJECT / "data")
REPS=2999; rng=np.random.default_rng(20260706)

INSP=pd.read_parquet(f"{DATA}/panel_month.parquet")[["province","insp_month"]].drop_duplicates()
INSP["insp"]=INSP["insp_month"].astype(str).str[:7]
PROVS=INSP["province"].values; INSP_VALS=INSP.set_index("province").loc[PROVS,"insp"].values
INSP_MAP0=dict(zip(INSP["province"],INSP["insp"]))

def crim_frame(oc,fam,weight=True):
    k=pd.read_parquet(f"{DATA}/crim_panel_v2.parquet")
    k=k[(k["family"]==fam)&(k["n_cases"]>0)].dropna(subset=["exposure_v2_z"]).copy()
    k["month"]=k["jmonth"].astype(str).str[:7]; k["H"]=k["exposure_v2_z"]
    k["fe1"]=k["prefecture_code"]; k["fe2"]=k["province"]+"_"+k["month"]
    k["y"]=np.arcsinh(k["n_cases"]) if oc=="asinh_n" else k[oc]
    k["w"]=k["n_cases"].astype(float) if weight else 1.0
    return k, ("w" if weight else None)

NAMES=["backstop","enforceN","detdebt"]
MARG=[crim_frame("y_backstop","market"),
      crim_frame("asinh_n","enforcementcrime",weight=False),
      crim_frame("y_detention_debt","enforcementcrime")]

def fit(d,w,im):
    x=d.copy(); ins=x["province"].map(im).values
    x["post"]=(x["month"].values>=ins).astype(int); x["px"]=x["post"].values*x["H"].values
    return float(pf.feols("y ~ px | fe1 + fe2",data=x,weights=w).coef()["px"])

b_obs=np.array([fit(d,w,INSP_MAP0) for d,w in MARG])
B=np.full((REPS,3),np.nan)
for r in range(REPS):
    im=dict(zip(PROVS,rng.permutation(INSP_VALS)))
    for j,(d,w) in enumerate(MARG): B[r,j]=fit(d,w,im)
    if (r+1)%500==0: print(f"  {r+1}/{REPS}",flush=True)

mu,sd=B.mean(0),B.std(0,ddof=1)
z=-(b_obs-mu)/sd                      # sign -1 for all criminal margins
Z=-(B-mu)/sd
def pval(cols,gls=False):
    zo,ZZ=z[list(cols)],Z[:,list(cols)]
    if gls and len(cols)>1:
        w=np.linalg.solve(np.cov(ZZ,rowvar=False),np.ones(len(cols)))
    else: w=np.ones(len(cols))
    return (1+np.sum(ZZ@w>=zo@w))/(1+REPS)

print(f"\n[obs] {dict(zip(NAMES,np.round(b_obs,4)))}",flush=True)
print(f"\n{'subset':28s} {'k':>1s} {'p_eq':>6s} {'p_gls':>6s}",flush=True)
rows=[]
for k in (1,2,3):
    for cs in combinations(range(3),k):
        lab="+".join(NAMES[i] for i in cs)
        pe=pval(cs); pg=pval(cs,gls=True) if k>1 else pe
        rows.append((lab,k,pe,pg)); print(f"{lab:28s} {k:>1d} {pe:6.3f} {pg:6.3f}",flush=True)
full=[r for r in rows if r[1]==3][0]
dropone=[r for r in rows if r[1]==2]
print(f"\n[verdict] full triple p_eq={full[2]:.3f}; drop-one range "
      f"{min(r[2] for r in dropone):.3f}-{max(r[2] for r in dropone):.3f}",flush=True)
pd.DataFrame(rows,columns=["subset","k","p_eq","p_gls"]).to_csv(
    str(_REP_PROJECT / 'output' / 'index_speccurve.csv').replace('\\', '/'),index=False)
print("[done] wrote index_speccurve.csv",flush=True)

# -*- coding: utf-8 -*-
"""6B step 29 — pin the criminal de-militarization joint p at 9999 wave-timing
draws (criminal margins only, so each draw is 3 fast cell fits). Same construction
as step 27 Part A restricted to the 3 audited criminal margins. Patches
\\CrimJointP (and adds \\CrimJointDraws) in numbers_joint.tex.
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
import numpy as np, pandas as pd, pyfixest as pf, re
DATA=str(_REP_PROJECT / "data"); OUTD=str(_REP_PROJECT / "output")
REPS=9999; rng=np.random.default_rng(20260705)

INSP=pd.read_parquet(f"{DATA}/panel_month.parquet")[["province","insp_month"]].drop_duplicates()
INSP["insp"]=INSP["insp_month"].astype(str).str[:7]
PROVS=INSP["province"].values; INSP_VALS=INSP.set_index("province").loc[PROVS,"insp"].values
INSP_MAP0=dict(zip(INSP["province"],INSP["insp"]))

def crim_frame(outcome_col,family,weight=True):
    k=pd.read_parquet(f"{DATA}/crim_panel_v2.parquet")
    k=k[(k["family"]==family)&(k["n_cases"]>0)].dropna(subset=["exposure_v2_z"]).copy()
    k["month"]=k["jmonth"].astype(str).str[:7]; k["H"]=k["exposure_v2_z"]
    k["fe1"]=k["prefecture_code"]; k["fe2"]=k["province"]+"_"+k["month"]
    k["y"]=np.arcsinh(k["n_cases"]) if outcome_col=="asinh_n" else k[outcome_col]
    k["w"]=k["n_cases"].astype(float) if weight else 1.0
    return k, ("w" if weight else None)

MARG=[("crim_backstop",crim_frame("y_backstop","market")),
      ("crim_enforceN",crim_frame("asinh_n","enforcementcrime",weight=False)),
      ("crim_detdebt", crim_frame("y_detention_debt","enforcementcrime"))]

def fit(data,w,inspmap):
    d=data.copy(); ins=d["province"].map(inspmap).values
    d["post"]=(d["month"].values>=ins).astype(int); d["px"]=d["post"].values*d["H"].values
    return float(pf.feols("y ~ px | fe1 + fe2",data=d,weights=w).coef()["px"])

b_obs=np.array([fit(d,w,INSP_MAP0) for _,(d,w) in MARG])
print("[obs]",dict(zip([m for m,_ in MARG],np.round(b_obs,4))),flush=True)
K=3; B=np.full((REPS,K),np.nan)
for r in range(REPS):
    tm=dict(zip(PROVS,rng.permutation(INSP_VALS)))
    for j,(_,(d,w)) in enumerate(MARG): B[r,j]=fit(d,w,tm)
    if (r+1)%1000==0: print(f"   {r+1}/{REPS}",flush=True)
mu,sd=B.mean(0),B.std(0,ddof=1); z=-(b_obs-mu)/sd; Z=-(B-mu)/sd
p=(1+np.sum(Z.sum(1)>=z.sum()))/(1+REPS)
print(f"[done] CRIMINAL joint p ({REPS} draws) = {p:.4f}",flush=True)

nj=f"{OUTD}/tables/numbers_joint.tex"
with open(nj,encoding="utf-8") as fh: t=fh.read()
t=re.sub(r"\\newcommand\{\\CrimJointP\}\{[^}]*\}",f"\\\\newcommand{{\\\\CrimJointP}}{{{p:.3f}}}",t)
if "\\CrimJointDraws" not in t:
    t=t.replace("\\newcommand{\\CrimJointP}",f"\\newcommand{{\\CrimJointDraws}}{{{REPS:,}}}\n\\newcommand{{\\CrimJointP}}")
with open(nj,"w",encoding="utf-8") as fh: fh.write(t)
print(f"patched CrimJointP={p:.3f}",flush=True)

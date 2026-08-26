# -*- coding: utf-8 -*-
"""6B step 28 — small-G diagnostics for the inference row:
  (1) effective number of clusters G* (Carter, Schnepel, Steigerwald 2017):
      residualize the regressor of interest on FEs + other regressors, form the
      per-province within-cluster sum of squares gamma_g, then
      G* = (sum gamma_g)^2 / sum gamma_g^2  (Herfindahl of design leverage).
      With provinces of very unequal prefecture count, G* << 31 is the honest
      statement of how much independent variation the design actually has.
  (2) CR2/CRV3 bias-reduced SE + small-sample p for the key coefficients, if
      pyfixest exposes them; falls back to reporting CRV1 with t(G-1) and t(G*-1).
Writes macros to output/tables/numbers_eff.tex. Non-destructive.
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
DATA = str(_REP_PROJECT / "data"); OUTD = str(_REP_PROJECT / "output")
from scipy import stats as sps

def gstar(df, y, regs, coef, fe, cluster="province", weights=None):
    """Effective # clusters for `coef` after partialling out other regs + FEs."""
    others = [r for r in regs if r != coef]
    rhs = (" + ".join(others) if others else "1")
    fes = " | " + fe
    # residualize the regressor of interest on everything else
    mx = pf.feols(f"{coef} ~ {rhs}{fes}", data=df, weights=weights)
    xr = np.asarray(mx.resid()).ravel()
    w = (df[weights].values if weights else np.ones(len(df))).astype(float)[:len(xr)]
    g = df[cluster].values[:len(xr)]
    gam = pd.Series(w * xr**2).groupby(pd.Series(g)).sum().values
    G = len(gam); Gs = (gam.sum()**2) / (gam**2).sum()
    return G, float(Gs)

def small_g_p(m, coef, Gs):
    b = m.coef()[coef]; se = m.se()[coef]; t = b / se
    return b, se, t, float(2*sps.t.sf(abs(t), Gs-1))

WIN=("2017-01","2019-03"); POST0="2018-09"; rows=[]

# ---- civil clean-window flow (pth) -----------------------------------------
c = pd.read_parquet(f"{DATA}/civil_panel.parquet")
c = c[c["cause_family"]=="relational"].dropna(subset=["exposure_v2_z"]).copy()
c["month"]=c["jmonth"].astype(str).str[:7]; c=c[(c["month"]>=WIN[0])&(c["month"]<=WIN[1])]
sched=pd.read_parquet(f"{DATA}/panel_month.parquet")[["province","inspection_round"]].drop_duplicates()
c=c.merge(sched,on="province"); c["treat"]=(c["inspection_round"]==1).astype(int)
c["postc"]=(c["month"]>=POST0).astype(int); c["H"]=c["exposure_v2_z"]
c["fe1"]=c["prefecture_code"]+"_"+c["cause"]; c["y"]=np.arcsinh(c["n_cases"])
c["pt"]=c["postc"]*c["treat"]; c["pth"]=c["pt"]*c["H"]; c["ph"]=c["postc"]*c["H"]
G,Gs=gstar(c,"y",["pth","ph","pt"],"pth","fe1 + month")
m=pf.feols("y ~ pth + ph + pt | fe1 + month",data=c,vcov={"CRV1":"province"})
b,se,t,p=small_g_p(m,"pth",Gs)
rows.append(("civ_flow_cw",G,Gs,b,se,p))
print(f"civil clean-window flow: G={G} G*={Gs:.1f}  b={b:+.4f} se={se:.4f} t(G*-1)-p={p:.3f}",flush=True)

# ---- criminal full-sample dose (px) ----------------------------------------
k=pd.read_parquet(f"{DATA}/crim_panel_v2.parquet"); k=k[k["n_cases"]>0].copy()
k["H"]=k["exposure_v2_z"]; k["px"]=k["post"]*k["H"]; k["fe1"]=k["prefecture_code"]
k["month"]=k["jmonth"].astype(str).str[:7]; k["fe2"]=k["province"]+"_"+k["month"]
for fam,out,wt in [("market","y_backstop","n_cases"),
                   ("enforcementcrime","asinh_n",None),
                   ("enforcementcrime","y_detention_debt","n_cases")]:
    d=k[k["family"]==fam].dropna(subset=["exposure_v2_z"]).copy()
    if out=="asinh_n": d["asinh_n"]=np.arcsinh(d["n_cases"])
    d["w"]=d["n_cases"].astype(float)
    G,Gs=gstar(d,out,["px"],"px","fe1 + fe2",weights=("w" if wt else None))
    m=pf.feols(f"{out} ~ px | fe1 + fe2",data=d,vcov={"CRV1":"province"},weights=("w" if wt else None))
    b,se,t,p=small_g_p(m,"px",Gs)
    rows.append((f"crim_{fam}_{out}",G,Gs,b,se,p))
    print(f"crim {fam:16s}/{out:16s}: G={G} G*={Gs:.1f}  b={b:+.4f} se={se:.4f} t(G*-1)-p={p:.3f}",flush=True)

# ---- try CR2/CRV3 if pyfixest exposes it -----------------------------------
cr_note="n/a"
for vc in ["CRV3","CR3","CRV2","CR2"]:
    try:
        mm=pf.feols("y ~ pth + ph + pt | fe1 + month",data=c,vcov={vc:"province"})
        cr_note=f"{vc} se={mm.se()['pth']:.4f} p={mm.pvalue()['pth']:.3f}"
        print(f"[bias-reduced] civil flow {cr_note}",flush=True); break
    except Exception as e:
        continue
if cr_note=="n/a": print("[bias-reduced] pyfixest has no CR2/CRV3 here — use R clubSandwich; CRV1+t(G*-1) reported",flush=True)

Gs_civ=[r for r in rows if r[0]=="civ_flow_cw"][0][2]
Gs_min=min(r[2] for r in rows)
with open(f"{OUTD}/tables/numbers_eff.tex","w",encoding="utf-8") as fh:
    fh.write("% step 28 — effective clusters + small-G p\n")
    fh.write(f"\\newcommand{{\\GeffCiv}}{{{Gs_civ:.0f}}}\n")
    fh.write(f"\\newcommand{{\\GeffMin}}{{{Gs_min:.0f}}}\n")
pd.DataFrame(rows,columns=["reg","G","Gstar","b","se","p_tGstar"]).to_csv(f"{OUTD}/effclusters.csv",index=False)
print(f"\n[done] G* civil={Gs_civ:.1f}, min across key regs={Gs_min:.1f}; wrote numbers_eff.tex",flush=True)

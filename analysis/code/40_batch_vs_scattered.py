# -*- coding: utf-8 -*-
"""6B step 40 — operation-level confound test with the VALIDATED case-number signature
(step 39: run-length >=5 is a reliable batch marker, 3.8% false-cluster). Flag each
clean-window lending case as BATCH (its 民初 seq sits in a >=5 run of consecutive
民间借贷 case numbers within its court-year) vs SCATTERED, then split the clean-window
dose. Mechanical dumping of a professional book -> BATCH; backstop removal among
one-off borrowers -> SCATTERED. A rise on SCATTERED cannot be batch dumping.
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
import re, numpy as np, pandas as pd, pyfixest as pf
DATA=str(_REP_PROJECT / "data"); OUTD=str(_REP_PROJECT / "output")
WIN=("2017-01","2019-03"); POST0="2018-09"; RUNK=5
PAT=re.compile(r"）(.+?)民[一二三四]?(初|终|再)[^\d]*(\d+)号")
def parse(cn):
    if not isinstance(cn,str): return (None,None,None,None)
    y=re.search(r"(\d{4})",cn); m=PAT.search(cn)
    if not (y and m): return (None,None,None,None)
    return (m.group(1),int(y.group(1)),m.group(2),int(m.group(3)))

from _wild import wild_score_p, wild_p  # corrected shared WCR bootstrap


cc=pd.read_parquet(f"{DATA}/civil_case.parquet",columns=["case_no","cause","prefecture_code","province","jmonth"])
pr=cc["case_no"].map(parse)
cc[["court","year","div","seq"]]=pd.DataFrame(pr.tolist(),index=cc.index)
lend=cc[(cc["cause"]=="民间借贷纠纷")&(cc["div"]=="初")&cc["seq"].notna()].copy()
# flag batch: seq in a >=RUNK consecutive run within (court,year) among lending seqs
lend["batch"]=False
for (ct,yr),g in lend.groupby(["court","year"]):
    s=np.sort(g["seq"].unique())
    if len(s)<RUNK: continue
    runmark=set(); i=0
    while i<len(s):
        j=i
        while j+1<len(s) and s[j+1]==s[j]+1: j+=1
        if j-i+1>=RUNK: runmark.update(s[i:j+1])
        i=j+1
    if runmark:
        idx=g.index[g["seq"].isin(runmark)]; lend.loc[idx,"batch"]=True
lend["month"]=lend["jmonth"].astype(str).str[:7]
cw=lend[(lend["month"]>=WIN[0])&(lend["month"]<=WIN[1])].copy()
print(f"[batch flag] clean-window first-instance lending {len(cw):,}; batch (>= {RUNK}-run) share {cw['batch'].mean():.3f}",flush=True)

sched=pd.read_parquet(f"{DATA}/panel_month.parquet")[["province","inspection_round"]].drop_duplicates()
ex=pd.read_parquet(f"{DATA}/exposure_v2.parquet")[["prefecture_code","exposure_v2_z"]]
def dose(df,label):
    g=(df.groupby(["prefecture_code","province","month"]).size().rename("n").reset_index()
       .merge(sched,on="province").merge(ex,on="prefecture_code").dropna(subset=["exposure_v2_z","inspection_round"]))
    g["H"]=g["exposure_v2_z"];g["pref"]=g["prefecture_code"];g["treat"]=(g["inspection_round"]==1).astype(int)
    g["postc"]=(g["month"]>=POST0).astype(int);g["prov_id"]=pd.factorize(g["province"])[0]
    g["pt"]=g["postc"]*g["treat"];g["pth"]=g["pt"]*g["H"];g["ph"]=g["postc"]*g["H"];g["y"]=np.arcsinh(g["n"])
    m=pf.feols("y ~ pth + ph + pt | pref + month",data=g,vcov={"CRV1":"prov_id"})
    print(f"[{label:20s}] pth={m.coef()['pth']:+.4f} ({m.se()['pth']:.4f}) CRV1 p={m.pvalue()['pth']:.3f} "
          f"wild p={wild_p('y ~ pth + ph + pt | pref + month',g,'pth'):.3f} cases={len(df):,}",flush=True)
    return dict(group=label,b=m.coef()['pth'],se=m.se()['pth'],p=m.pvalue()['pth'],n=len(df))

print("\n===== batch (professional, >=5-run) vs scattered (one-off) lending flow =====",flush=True)
r=[dose(cw[~cw["batch"]],"scattered"),dose(cw[cw["batch"]],"batch"),dose(cw,"all first-instance")]
pd.DataFrame(r).to_csv(f"{OUTD}/batch_vs_scattered.csv",index=False)
print("\n[verdict] mechanism -> rise on SCATTERED; batch dumping of a professional book -> BATCH.",flush=True)

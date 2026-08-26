# -*- coding: utf-8 -*-
"""6B step 42 (a3) — CONGESTION SPILLOVER, truncation-robust. Does displaced lending load
slow OTHER civil causes' resolution in the same courts? Outcome = filing-to-judgment
duration of non-lending causes (cleanest: traffic-tort placebo). Clean-window triple-diff
(Post x Treat x H), same structure as the main civil design (step 36).

TRUNCATION FIX: data are judgments, so a filing cohort is only seen up to the frame end.
With the frame extended to judgment-month 2019-12 (step 41b), every filing in [2017-01,
2019-03] is fully observed up to a 270-day (9-month) horizon. Restricting dur<=270 makes
the observation window SYMMETRIC across pre/post cohorts, removing differential truncation.
Report: (i) capped log-duration (primary), (ii) 1[dur<=90] speed indicator, (iii) uncapped
log-duration (shows the truncation-driven negative for contrast).
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
WIN = ("2017-01", "2019-03"); POST0 = "2018-09"; H_DAYS = 270
PLACEBO = "机动车交通事故责任纠纷"; LEND = "民间借贷纠纷"

from _wild import wild_score_p, wild_p  # corrected shared WCR bootstrap


fil = pd.read_parquet(f"{DATA}/civil_filing.parquet").dropna(subset=["filing_ymd"]).rename(columns={"案号":"case_no"})[["case_no","filing_ymd"]]
cc = pd.read_parquet(f"{DATA}/civil_case.parquet", columns=["case_no","cause","prefecture_code","province","jmonth"])
d = cc.merge(fil, on="case_no", how="inner")
d["filing"] = pd.to_datetime(d["filing_ymd"], errors="coerce")
d["judg"] = pd.to_datetime(d["jmonth"].astype(str).str[:7], format="%Y-%m", errors="coerce") + pd.Timedelta(days=14)
d["dur"] = (d["judg"] - d["filing"]).dt.days
d["fmonth"] = d["filing"].dt.strftime("%Y-%m")
d = d[(d["dur"] >= 1) & (d["fmonth"].between(WIN[0], WIN[1]))]
print(f"[merged] {len(d):,} judged cases, filing in window; dur<=270 share {(d['dur']<=H_DAYS).mean():.3f}", flush=True)

sched = pd.read_parquet(f"{DATA}/panel_month.parquet")[["province","inspection_round"]].drop_duplicates()
ex = pd.read_parquet(f"{DATA}/exposure_v2.parquet")[["prefecture_code","exposure_v2_z"]]

def cells(df, ycol):
    c = (df.groupby(["prefecture_code","province","fmonth"]).agg(y=(ycol,"mean"), n=(ycol,"size")).reset_index()
           .rename(columns={"fmonth":"month"}).merge(sched, on="province").merge(ex, on="prefecture_code")
           .dropna(subset=["exposure_v2_z","inspection_round"]))
    c = c[c["n"] >= 5]
    c["H"]=c["exposure_v2_z"]; c["pref"]=c["prefecture_code"]
    c["treat"]=(c["inspection_round"]==1).astype(int); c["postc"]=(c["month"]>=POST0).astype(int)
    c["prov_id"]=pd.factorize(c["province"])[0]
    c["pt"]=c["postc"]*c["treat"]; c["pth"]=c["pt"]*c["H"]; c["ph"]=c["postc"]*c["H"]
    return c

def est(c, tag):
    m = pf.feols("y ~ pth + ph + pt | pref + month", data=c, weights="n", vcov={"CRV1":"prov_id"})
    wp = wild_score_p("y ~ pth + ph + pt | pref + month", c, "pth")
    b, se, p = m.coef()["pth"], m.se()["pth"], m.pvalue()["pth"]
    print(f"    [{tag:34s}] pth={b:+.4f} ({se:.4f}) CRV1 p={p:.3f} wild p={wp:.3f} cells={len(c)}", flush=True)
    return dict(spec=tag, b=b, se=se, p=p, wild_p=wp, cells=len(c))

res = []
for cname, sub in [("traffic-placebo", d[d.cause==PLACEBO]), ("all non-lending", d[d.cause!=LEND]),
                   ("lending own-cause", d[d.cause==LEND])]:
    print(f"[{cname}]  N={len(sub):,}", flush=True)
    cap = sub[sub["dur"] <= H_DAYS].copy(); cap["ld"] = np.log(cap["dur"]); cap["fast"] = (cap["dur"] <= 90).astype(float)
    res.append({**est(cells(cap, "ld"), f"{cname}: log-dur capped<=270 (PRIMARY)"), "grp": cname})
    res.append({**est(cells(cap, "fast"), f"{cname}: P(dur<=90 | dur<=270)"), "grp": cname})
    unc = sub.copy(); unc["ld"] = np.log(unc["dur"])
    res.append({**est(cells(unc, "ld"), f"{cname}: log-dur UNCAPPED (truncated)"), "grp": cname})

pd.DataFrame(res).to_csv(f"{OUTD}/e42_congestion_spillover.csv", index=False)
print(f"\n[done] -> {OUTD}/e42_congestion_spillover.csv", flush=True)

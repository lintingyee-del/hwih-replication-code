# -*- coding: utf-8 -*-
"""6B step 35 — item 5: sharpen the stake gradient (Proposition 3). Instead of many
thin, individually-insignificant claim-size bins, pre-define ONE mid-stake band
(20k-200k yuan, the range the model marks as the marginal formalization band) and test
whether the clean-window judicialization response CONCENTRATES there, with the tails
(<20k and >200k) as placebo. Concentrates the dispersed power; statistically correct.

Reports: (a) three separate band doses (low / mid / high); (b) the sharp mid-vs-tail
contrast (Post x Treat x H x Mid), tails pooled as within-prefecture-month placebo.
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
MID = (20000.0, 200000.0)

from _wild import wild_score_p, wild_p  # corrected shared WCR bootstrap


cc = pd.read_parquet(f"{DATA}/civil_case.parquet",
                     columns=["cause","prefecture_code","province","jmonth","amount_yuan"])
ld = cc[cc["cause"] == "民间借贷纠纷"].copy()
ld["month"] = ld["jmonth"].astype(str).str[:7]
ld = ld[(ld["month"] >= WIN[0]) & (ld["month"] <= WIN[1])]
print(f"[coverage] clean-window lending {len(ld)}; amount_yuan valid "
      f"{(ld['amount_yuan'] > 0).mean():.3f}", flush=True)
ld = ld[(ld["amount_yuan"] > 0) & (ld["amount_yuan"] < 1e8)].copy()
ld["band"] = np.where(ld["amount_yuan"] < MID[0], "low",
              np.where(ld["amount_yuan"] <= MID[1], "mid", "high"))
print("[band shares]", ld["band"].value_counts(normalize=True).round(3).to_dict(), flush=True)

sched = pd.read_parquet(f"{DATA}/panel_month.parquet")[["province","inspection_round"]].drop_duplicates()
ex = pd.read_parquet(f"{DATA}/exposure_v2.parquet")[["prefecture_code","exposure_v2_z"]]

def cells(df):
    g = (df.groupby(["prefecture_code","province","month"]).size().rename("n").reset_index()
         .merge(sched, on="province").merge(ex, on="prefecture_code")
         .dropna(subset=["exposure_v2_z","inspection_round"]))
    g["H"]=g["exposure_v2_z"]; g["pref"]=g["prefecture_code"]
    g["treat"]=(g["inspection_round"]==1).astype(int); g["postc"]=(g["month"]>=POST0).astype(int)
    g["prov_id"]=pd.factorize(g["province"])[0]
    g["pt"]=g["postc"]*g["treat"]; g["pth"]=g["pt"]*g["H"]; g["ph"]=g["postc"]*g["H"]
    g["y"]=np.arcsinh(g["n"])
    return g

print("\n===== (a) three separate band doses (clean window) =====", flush=True)
for b in ["low","mid","high"]:
    g = cells(ld[ld["band"]==b])
    m = pf.feols("y ~ pth + ph + pt | pref + month", data=g, vcov={"CRV1":"prov_id"})
    wp = wild_score_p("y ~ pth + ph + pt | pref + month", g, "pth")
    print(f"[{b:4s}] pth={m.coef()['pth']:+.4f} ({m.se()['pth']:.4f}) "
          f"CRV1 p={m.pvalue()['pth']:.3f} wild p={wp:.3f} N={m._N}", flush=True)

print("\n===== (b) sharp mid-vs-tail contrast (tails = placebo) =====", flush=True)
lo = cells(ld[ld["band"]!="mid"]).assign(is_mid=0)   # tails pooled
mi = cells(ld[ld["band"]=="mid"]).assign(is_mid=1)
st = pd.concat([lo, mi], ignore_index=True)
st["pref_mid"]=st["pref"]+"_"+st["is_mid"].astype(str)
st["prov_month"]=st["province"]+"_"+st["month"]
st["pthm"]=st["pth"]*st["is_mid"]; st["phm"]=st["ph"]*st["is_mid"]; st["ptm"]=st["pt"]*st["is_mid"]
m = pf.feols("y ~ pthm + phm + ptm + pth + ph + pt | pref_mid + prov_month",
             data=st, vcov={"CRV1":"prov_id"})
wp = wild_score_p("y ~ pthm + phm + ptm + pth + ph + pt | pref_mid + prov_month", st, "pthm")
print(f"[mid vs tail] Post x Treat x H x Mid = {m.coef()['pthm']:+.4f} "
      f"({m.se()['pthm']:.4f}) CRV1 p={m.pvalue()['pthm']:.3f} wild p={wp:.3f} N={m._N}", flush=True)
print("\n[verdict] Proposition 3 predicts the response concentrates in the mid band: "
      "positive mid-vs-tail contrast. If sharp, the inverted-U becomes a testable claim.", flush=True)
pd.DataFrame([dict(test="mid_vs_tail", coef=m.coef()['pthm'], se=m.se()['pthm'],
                   p=m.pvalue()['pthm'], wild_p=wp)]).to_csv(f"{OUTD}/stake_midband.csv", index=False)
print("[done] wrote output/stake_midband.csv", flush=True)

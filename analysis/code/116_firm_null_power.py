# -*- coding: utf-8 -*-
"""Power and support for the firm-quarter exit-hazard dose null reported in
Table tab:collspecs, Panel A (baseline b=-0.0048, CRV1 se=0.0059, wild p=0.347).

The paper reports this null without saying what it can and cannot detect. This
script produces the numbers for a short interpretive paragraph:
  - the minimum detectable effect at 80 percent power, 5 percent size
  - the confidence interval expressed as a share of the baseline quarterly
    hazard, which is the interpretable scale
  - the support: how many prefectures contribute at all, and how concentrated
    the surviving cross-sectional variation is

Rebuilds the hazard panel exactly as 92_collection_panel.py does.
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
import io
import os
import re
import sys

import numpy as np
import pandas as pd
import pyfixest as pf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from _wild import wild_score_p

BASE = str(_REP_PROJECT)
DATA = os.path.join(BASE, "data")
OUT = os.path.join(BASE, "output", "exposure_validity")
GS = str(_REP_REGISTRY).replace('\\', '/')
REGISTRY_HITS = (
    os.path.join(DATA, "derived", "registry_hits_deidentified.csv")
    if os.environ.get("HWIH_REPLICATION") == "1"
    else os.path.join(GS, "hits_clean.csv")
)
os.makedirs(OUT, exist_ok=True)
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


xw = pd.read_parquet(os.path.join(DATA, "court_xwalk.parquet"))
pat = re.compile(r"(?:.*?省|.*?自治区)?(.+?(?:市|州|盟|地区))中级人民法院$")
names = {}
for _, r in xw.iterrows():
    m = pat.match(str(r["court_name"]))
    if m:
        names.setdefault(m.group(1), []).append(r["prefecture_code"])
name2code = {nm: max(set(v), key=v.count) for nm, v in names.items()}
name2code.update({"北京市": "110100", "天津市": "120100",
                  "上海市": "310100", "重庆市": "500100"})


def city2code(c):
    c = str(c)
    if not c or c == "nan":
        return None
    cand = c if c.endswith(("市", "州", "盟", "地区")) else c + "市"
    if cand in name2code:
        return name2code[cand]
    for nm, cd in name2code.items():
        if nm[:-1] and nm[:-1] in c:
            return cd
    return None


exp = pd.read_parquet(os.path.join(DATA, "exposure_v2.parquet"))
codes = set(exp["prefecture_code"])
cp = pd.read_parquet(os.path.join(DATA, "civil_panel.parquet"),
                     columns=["province", "insp_month"]).drop_duplicates()
insp = cp.groupby("province")["insp_month"].first()

h = pd.read_csv(REGISTRY_HITS, dtype=str)
h = h.rename(columns={"所属城市": "district_raw"})
h["所属城市"] = h["所属省份"]
h["em"] = pd.to_datetime(h["成立日期"], errors="coerce")
h["exit"] = h["经营状态"].str.contains("注销|吊销", na=False)
h["xm"] = pd.to_datetime(h["核准日期"], errors="coerce").where(h["exit"])
h["prefecture_code"] = h["所属城市"].map(city2code)
h = h[h["prefecture_code"].isin(codes)].copy()
h = h.merge(exp[["prefecture_code", "exposure_v2_z", "province"]],
            on="prefecture_code")
h["insp_month"] = h["province"].map(insp)
h["fy"] = h["em"].dt.year

# ---- firm x quarter hazard panel, as in 92_collection_panel.py --------------
hz = h.dropna(subset=["em", "exposure_v2_z"]).copy()
hz["exq"] = hz["xm"].dt.to_period("Q")
rows = []
for q in pd.period_range("2016Q1", "2021Q4", freq="Q"):
    at_risk = hz[(hz["em"].dt.to_period("Q") < q)
                 & (hz["exq"].isna() | (hz["exq"] >= q))]
    rows.append(pd.DataFrame({
        "prefecture_code": at_risk["prefecture_code"].values,
        "province": at_risk["province"].values,
        "exposure_v2_z": at_risk["exposure_v2_z"].values,
        "fy": at_risk["fy"].fillna(0).astype(int).astype(str).values,
        "insp_month": at_risk["insp_month"].values,
        "q": str(q),
        "y": (at_risk["exq"] == q).astype(float).values}))
hp = pd.concat(rows, ignore_index=True)
hp["qend"] = pd.PeriodIndex(hp["q"], freq="Q").to_timestamp(how="end")
hp["post"] = (hp["qend"] >= pd.to_datetime(hp["insp_month"])).astype(int)
hp["postxH"] = hp["post"] * hp["exposure_v2_z"]
hp["provq"] = hp["province"] + "_" + hp["q"]
hp["prov_id"] = pd.factorize(hp["province"])[0]

m = pf.feols("y ~ postxH + C(fy) | prefecture_code + provq", data=hp,
             vcov={"CRV1": "prov_id"})
b = float(m.coef()["postxH"]); se = float(m.se()["postxH"])
pw = wild_score_p("y ~ postxH + C(fy) | prefecture_code + provq", hp, "postxH")
base = float(hp["y"].mean())
say(f"[replication] firm-quarter hazard DiD: b={b:.4f} se={se:.4f} "
    f"wild p={pw:.3f}   (paper table: -0.0048 / 0.0059 / 0.347)")
say(f"  firm-quarters {len(hp):,}; exit events {int(hp['y'].sum()):,}; "
    f"baseline quarterly hazard {base:.4f}")

mde = 2.802 * se
lo, hi = b - 1.96 * se, b + 1.96 * se
say(f"\n  MDE (80% power, 5% size) = {mde:.4f} per firm-quarter "
    f"= {mde / base:.0%} of the baseline hazard")
say(f"  95% CI = [{lo:.4f}, {hi:.4f}] "
    f"= [{lo / base:+.0%}, {hi / base:+.0%}] of the baseline hazard")
say(f"  the model predicts a POSITIVE coefficient (more exit where exposure is "
    f"higher); the interval's upper end is {hi / base:+.0%} of baseline")

# ---- support and concentration ---------------------------------------------
n_pref_panel = hp["prefecture_code"].nunique()
n_pref_exposure = exp["prefecture_code"].nunique()
say(f"\n  prefectures contributing any firm-quarter: {n_pref_panel} of "
    f"{n_pref_exposure} in the exposure sample "
    f"({1 - n_pref_panel / n_pref_exposure:.0%} contribute nothing)")
share = (hp.groupby("prefecture_code").size().sort_values(ascending=False)
         / len(hp))
say(f"  top 10 prefectures hold {share.head(10).sum():.0%} of firm-quarters; "
    f"top 25 hold {share.head(25).sum():.0%}")
ev = hp.groupby("prefecture_code")["y"].sum().sort_values(ascending=False)
say(f"  top 10 prefectures hold {ev.head(10).sum() / ev.sum():.0%} of exit "
    f"events")
# effective sample: Kish, on the exposure dose actually carried by the panel
w = hp.groupby("prefecture_code").size()
kish = (w.sum() ** 2) / (w ** 2).sum()
say(f"  Kish effective number of prefectures = {kish:.0f}")

pd.DataFrame([{
    "spec": "firm-quarter exit hazard DiD (paper Table tab:collspecs Panel A)",
    "beta": b, "se_crv1": se, "p_wild": pw,
    "firm_quarters": len(hp), "exit_events": int(hp["y"].sum()),
    "baseline_quarterly_hazard": base,
    "mde_80pct": mde, "mde_as_share_of_base": mde / base,
    "ci_low": lo, "ci_high": hi,
    "ci_low_share": lo / base, "ci_high_share": hi / base,
    "prefectures_in_panel": n_pref_panel,
    "prefectures_in_exposure": n_pref_exposure,
    "top10_share_firmquarters": float(share.head(10).sum()),
    "top10_share_exits": float(ev.head(10).sum() / ev.sum()),
    "kish_effective_prefectures": kish,
}]).to_csv(os.path.join(OUT, "firm_null_power.csv"), index=False,
           encoding="utf-8-sig")
with open(os.path.join(OUT, "firm_null_power_log.txt"), "w",
          encoding="utf-8") as fh:
    fh.write("\n".join(LOG))
say("\nDONE ->", OUT)

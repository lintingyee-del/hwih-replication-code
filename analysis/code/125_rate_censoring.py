# -*- coding: utf-8 -*-
"""6B step 125 - statutory-ceiling censoring in the recorded monthly rate.

The ex ante pricing null (Table tab:exante) is estimated on the monthly rate
recorded in the judgment. That rate is the rate the court recognises: the 2015
SPC provisions (Fa Shi [2015] No.18) protect annual interest through 24 percent
(2.00 percent a month) and void amounts above 36 percent (3.00 percent a month).
This script measures how much of the distribution sits on those two kinks, which
scopes what the null can bound.

Outputs: output/rate_censoring.csv and the two preamble macros.
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
import io, sys
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
DATA = str(_REP_PROJECT / "data")
OUT = str(_REP_PROJECT / "output")
CAP, VOID = 2.0, 3.0            # percent per month = 24 and 36 percent per year

df = pd.read_parquet(f"{DATA}/civil_case.parquet",
                     columns=["cause", "jmonth", "monthly_rate_pct"])
d = df[(df["cause"] == "民间借贷纠纷") & df["monthly_rate_pct"].notna()].copy()
# the raw extractor emits a small number of implausible values (max ~5e13)
d = d[(d["monthly_rate_pct"] > 0) & (d["monthly_rate_pct"] <= 10)]
d["yr"] = pd.to_datetime(d["jmonth"]).dt.year

rows = []
for name, lo, hi in [("pre 2016-17", 2016, 2017), ("pre 2014-17", 2014, 2017),
                     ("window 2014-20", 2014, 2020)]:
    r = d.loc[(d["yr"] >= lo) & (d["yr"] <= hi), "monthly_rate_pct"]
    rows.append(dict(sample=name, n=len(r), mean=r.mean(), median=r.median(),
                     at_cap=(r == CAP).mean(), at_or_above_cap=(r >= CAP).mean(),
                     above_cap=(r > CAP).mean(), at_void=(r == VOID).mean()))
for y in range(2014, 2021):
    r = d.loc[d["yr"] == y, "monthly_rate_pct"]
    rows.append(dict(sample=str(y), n=len(r), mean=r.mean(), median=r.median(),
                     at_cap=(r == CAP).mean(), at_or_above_cap=(r >= CAP).mean(),
                     above_cap=(r > CAP).mean(), at_void=(r == VOID).mean()))

res = pd.DataFrame(rows)
res.to_csv(f"{OUT}/rate_censoring.csv", index=False)
print(res.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

pre = res.loc[res["sample"] == "pre 2016-17"].iloc[0]
print(f"\n\newcommand{{\RateAtCapPct}}{{{round(100*pre.at_cap)}}}")
print(f"\newcommand{{\RateAtOrAboveCapPct}}{{{round(100*pre.at_or_above_cap)}}}")

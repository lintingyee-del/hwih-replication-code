# -*- coding: utf-8 -*-
"""Is the registered collection industry the paper's coercive backstop?

113 flagged 88.9% of the 6,738 registered "collection" firms as carrying a
licensed-credit context. If that classification is right, the national registry
exhibit in the paper is measuring the licensed bank-outsourced collection
industry, not the illegal coercive backstop, and the 2018 collapse it shows has
a competing explanation (the P2P crackdown and the 2018-19 regulatory push on
outsourced collection) that overlaps the confound the paper works to dismiss.

This script does two things:
  1. prints a random sample of snippets, flagged and unflagged, so the
     classifier can be checked by eye rather than trusted;
  2. re-runs the national entry/exit series separately for the licensed-credit
     subset and the vernacular subset (讨债 / 收数 / 商账 without a licensed
     context), which is the subset that plausibly proxies the backstop.
     If the collapse is present in the vernacular subset the exhibit survives
     with a relabel; if it is confined to the licensed subset it does not.
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
import statsmodels.api as sm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE = str(_REP_PROJECT)
OUT = os.path.join(BASE, "output", "exposure_validity")
GS = str(_REP_REGISTRY).replace('\\', '/')
REGISTRY_HITS = (
    os.path.join(BASE, "data", "derived", "registry_hits_deidentified.csv")
    if os.environ.get("HWIH_REPLICATION") == "1"
    else os.path.join(GS, "hits_clean.csv")
)
os.makedirs(OUT, exist_ok=True)
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


h = pd.read_csv(REGISTRY_HITS, dtype=str)
snip = h["snippet"].fillna("")
name = h["企业名称"].fillna("")
BANKISH = re.compile(r"银行|信用卡|信贷|金融机构|持牌|委托|资产管理|不良资产|"
                     r"保理|征信|应收账款|逾期户|贷后")
if "bank_context" in h.columns:
    h["bank_context"] = h["bank_context"].astype(str).str.lower().isin(
        {"true", "1", "yes"})
else:
    h["bank_context"] = snip.str.contains(BANKISH) | name.str.contains(BANKISH)
h["em"] = pd.to_datetime(h["成立日期"], errors="coerce")
h["exited"] = h["经营状态"].str.contains("注销|吊销", na=False)
h["xm"] = pd.to_datetime(h["核准日期"], errors="coerce").where(h["exited"])

# ---------------------------------------------------------------------------
# 1. eyeball sample
# ---------------------------------------------------------------------------
rng = np.random.default_rng(11)
say("=== SAMPLE: classified as licensed-credit context ===")
sub = h[h["bank_context"]]
for i in rng.choice(len(sub), 12, replace=False):
    r = sub.iloc[i]
    say(f"  [{r['matched_kw']}] {str(r['企业名称'])[:28]} :: "
        f"{str(r['snippet'])[:70]}")
say("\n=== SAMPLE: NOT classified as licensed-credit context ===")
sub0 = h[~h["bank_context"]]
for i in rng.choice(len(sub0), 12, replace=False):
    r = sub0.iloc[i]
    say(f"  [{r['matched_kw']}] {str(r['企业名称'])[:28]} :: "
        f"{str(r['snippet'])[:70]}")

# ---------------------------------------------------------------------------
# 2. national series by subset
# ---------------------------------------------------------------------------
say("\n=== national quarterly entries by subset ===")
h["vernacular"] = (~h["bank_context"]) & h["matched_kw"].isin(
    ["讨债", "收数", "商账"])
h["licensed"] = h["bank_context"]
say(f"  licensed-credit subset: {int(h['licensed'].sum())}")
say(f"  vernacular subset (讨债/收数/商账, no licensed context): "
    f"{int(h['vernacular'].sum())}")
say(f"  residual (催收 without licensed context): "
    f"{int((~h['licensed'] & ~h['vernacular']).sum())}")

q = pd.period_range("2013Q1", "2022Q4", freq="Q")
rows = []
for label, mask in [("licensed", h["licensed"]),
                    ("vernacular", h["vernacular"]),
                    ("residual_cuishou", ~h["licensed"] & ~h["vernacular"]),
                    ("all", pd.Series(True, index=h.index))]:
    g = h[mask]
    for qq in q:
        rows.append({"subset": label, "quarter": str(qq),
                     "entries": int((g["em"].dt.to_period("Q") == qq).sum()),
                     "exits": int((g["xm"].dt.to_period("Q") == qq).sum())})
nat = pd.DataFrame(rows)
nat.to_csv(os.path.join(OUT, "national_quarterly_by_subset.csv"),
           index=False, encoding="utf-8-sig")

# peak-to-2020 decline and a segmented trend break at 2018Q1 per subset
say("\n  entries: peak year -> 2020, and segmented break at 2018Q1")
res = []
for label in ["licensed", "vernacular", "residual_cuishou", "all"]:
    s = nat[nat["subset"].eq(label)].copy()
    s["qi"] = pd.PeriodIndex(s["quarter"], freq="Q")
    ann = s.groupby(s["qi"].dt.year)["entries"].sum()
    pk_year = int(ann.loc[2015:2019].idxmax())
    pk, v2020 = float(ann.loc[pk_year]), float(ann.get(2020, np.nan))
    dec = 1 - v2020 / pk if pk else np.nan
    # segmented trend on log(1+entries), break at 2018Q1, Newey-West
    w = s[(s["qi"] >= pd.Period("2014Q1")) & (s["qi"] <= pd.Period("2021Q4"))].copy()
    t = np.arange(len(w), dtype=float)
    brk = float((w["qi"] < pd.Period("2018Q1")).sum())
    Xd = np.column_stack([np.ones(len(w)), t, (t >= brk).astype(float),
                          np.clip(t - brk, 0, None)])
    y = np.log1p(w["entries"].values.astype(float))
    m = sm.OLS(y, Xd).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
    res.append({"subset": label, "n_firms": int(h[
        {"licensed": h["licensed"], "vernacular": h["vernacular"],
         "residual_cuishou": ~h["licensed"] & ~h["vernacular"],
         "all": pd.Series(True, index=h.index)}[label]].shape[0]),
        "peak_year": pk_year, "peak_entries": pk, "entries_2020": v2020,
        "decline_peak_to_2020": dec,
        "level_shift_2018Q1": m.params[2], "level_p": m.pvalues[2],
        "slope_break_2018Q1": m.params[3], "slope_p": m.pvalues[3]})
    say(f"    {label:18s} n={res[-1]['n_firms']:5d} peak {pk_year} "
        f"({pk:.0f}) -> 2020 ({v2020:.0f}) = {dec:+.1%}; "
        f"slope break {m.params[3]:+.4f} (p={m.pvalues[3]:.3g})")
rr = pd.DataFrame(res)
rr.to_csv(os.path.join(OUT, "registry_break_by_subset.csv"), index=False,
          encoding="utf-8-sig")

with open(os.path.join(OUT, "registry_construct_log.txt"), "w",
          encoding="utf-8") as fh:
    fh.write("\n".join(LOG))
say("\nDONE ->", OUT)

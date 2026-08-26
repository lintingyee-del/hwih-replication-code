# -*- coding: utf-8 -*-
"""A magnitude for the composition estimand, so it can lead the abstract.

The flow estimand currently leads the paper because it is the only one that
converts into plain language ("447 additional relational cases per year, 21
percent of pre-campaign workload"). The composition estimand is the robust one
(it survives province x month effects, CRV3, and the 51.5 percent deletion
ladder) but is reported only in asinh points, so it cannot carry a headline.

This script builds the same class of translation for composition, using the
identical conversion the flow number uses in 110_primary_civil_revised.py:

    increment_per_prefecture_year = baseline_cases_per_prefecture_year
                                    * (exp(beta) - 1)

Quantities produced
  1. acquaintance flow effect          exp(b_acq) - 1
  2. stranger flow effect              exp(b_str) - 1
  3. EXCESS acquaintance cases, i.e. the cases above what the stranger
     benchmark would have produced:  baseline_acq * (exp(b_acq) - exp(b_str))
     This is the differential's own magnitude and nets out anything that moves
     acquaintance and stranger lending together.
  4. the same at national scale
  5. the pre-campaign acquaintance share of classified lending, for context

Outputs (analysis/output/tables/)
  composition_magnitude.csv
  numbers_composition_magnitude.tex   drop-in macros
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
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import pyfixest as pf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from _wild import wild_score_p

DATA = str(_REP_PROJECT / "data")
TAB = str(_REP_PROJECT / "output" / "tables")
START, END, POST0 = "2017-01", "2019-03", "2018-09"
SUPPORT_START, SUPPORT_END = "2014-01", "2017-12"
BASE_YEARS = ["2016", "2017"]
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
VERSIONED_OUTPUTS = os.environ.get("HWIH_REPLICATION", "0") != "1"
os.makedirs(TAB, exist_ok=True)
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


schedule = (pd.read_parquet(f"{DATA}/panel_month.parquet")
            [["province", "inspection_round"]].drop_duplicates())
exposure = pd.read_parquet(f"{DATA}/exposure_v2.parquet")[
    ["prefecture_code", "province", "exposure_v2_z"]].drop_duplicates()
months = pd.DataFrame({"month": pd.period_range(START, END, freq="M").astype(str)})


def add_design(d):
    d = d.copy()
    d["treat"] = (d["inspection_round"] == 1).astype(int)
    d["postc"] = (d["month"] >= POST0).astype(int)
    d["pt"] = d["postc"] * d["treat"]
    d["ph"] = d["postc"] * d["exposure_v2_z"]
    d["pth"] = d["pt"] * d["exposure_v2_z"]
    d["prov_id"] = pd.factorize(d["province"])[0]
    return d


def fit(formula, data, coef):
    m = pf.feols(formula, data=data, vcov={"CRV1": "prov_id"})
    return (float(m.coef()[coef]), float(m.se()[coef]),
            wild_score_p(formula, data, coef))


# ---------------------------------------------------------------------------
# Rebuild the composition panel exactly as 110_primary_civil_revised.py does.
# ---------------------------------------------------------------------------
case = pd.read_parquet(f"{DATA}/civil_case.parquet",
                       columns=["cause", "prefecture_code", "province",
                                "jmonth", "rel_txn"])
case["month"] = case["jmonth"].astype(str).str[:7]
lending_all = case[case["cause"].eq("民间借贷纠纷")].copy()
lending = lending_all[lending_all["rel_txn"].notna()].copy()
lending["acq"] = lending["rel_txn"].astype(int)

comp_support = (lending[lending["month"].between(SUPPORT_START, SUPPORT_END)]
                [["prefecture_code", "province"]].drop_duplicates()
                .merge(exposure[["prefecture_code", "province"]],
                       on=["prefecture_code", "province"], how="inner"))
comp_counts = (lending[lending["month"].between(START, END)]
               .groupby(["prefecture_code", "province", "month", "acq"])
               .size().rename("n").reset_index())
comp = (comp_support.merge(months, how="cross")
        .merge(pd.DataFrame({"acq": [0, 1]}), how="cross")
        .merge(comp_counts, on=["prefecture_code", "province", "month", "acq"],
               how="left"))
comp["n"] = comp["n"].fillna(0).astype(float)
comp = comp.merge(schedule, on="province").merge(
    exposure[["prefecture_code", "exposure_v2_z"]], on="prefecture_code")
comp = add_design(comp)
comp["y"] = np.arcsinh(comp["n"])
comp["prefA"] = comp["prefecture_code"] + "_" + comp["acq"].astype(str)
comp["monthA"] = comp["month"] + "_" + comp["acq"].astype(str)
for t in ("pth", "ph", "pt"):
    comp[f"{t}A"] = comp[t] * comp["acq"]

b_diff, se_diff, p_diff = fit(
    "y ~ pthA + phA + ptA + pth + ph + pt | prefA + monthA", comp, "pthA")
b_acq, se_acq, p_acq = fit("y ~ pth + ph + pt | prefecture_code + month",
                           comp[comp["acq"].eq(1)].copy(), "pth")
b_str, se_str, p_str = fit("y ~ pth + ph + pt | prefecture_code + month",
                           comp[comp["acq"].eq(0)].copy(), "pth")
say(f"[replication check] acq-minus-stranger  b={b_diff:.4f} se={se_diff:.4f} "
    f"wild p={p_diff:.3f}   (paper: 0.182 / 0.055 / 0.012)")
say(f"[replication check] acquaintance flow   b={b_acq:.4f} wild p={p_acq:.3f}"
    f"   (paper: 0.282 / 0.007)")
say(f"[replication check] stranger flow       b={b_str:.4f} wild p={p_str:.3f}"
    f"   (paper: 0.100 / 0.325)")

# ---------------------------------------------------------------------------
# Baselines, on the same support the estimate uses.
# ---------------------------------------------------------------------------
support_codes = set(comp["prefecture_code"])
pre = lending[lending["month"].str[:4].isin(BASE_YEARS)
              & lending["prefecture_code"].isin(support_codes)]
n_pref = len(support_codes)
acq_pref_year = (pre[pre["acq"].eq(1)].groupby("prefecture_code").size()
                 .reindex(sorted(support_codes)).fillna(0).mean() / 2.0)
str_pref_year = (pre[pre["acq"].eq(0)].groupby("prefecture_code").size()
                 .reindex(sorted(support_codes)).fillna(0).mean() / 2.0)
cls_pref_year = acq_pref_year + str_pref_year
acq_share_pre = acq_pref_year / cls_pref_year

# classifiable share of the whole lending docket, for scaling to the full docket
pre_all = lending_all[lending_all["month"].str[:4].isin(BASE_YEARS)
                      & lending_all["prefecture_code"].isin(support_codes)]
classifiable = float(pre_all["rel_txn"].notna().mean())
all_pref_year = len(pre_all) / n_pref / 2.0

say(f"\nsupport: {n_pref} prefectures with pre-campaign classified lending")
say(f"pre-campaign (2016-17) per prefecture-year, classified lending:")
say(f"  acquaintance {acq_pref_year:,.0f} | stranger {str_pref_year:,.0f} "
    f"| classified total {cls_pref_year:,.0f}")
say(f"  acquaintance share of classified lending: {acq_share_pre:.1%}")
say(f"  classifiable share of the full lending docket: {classifiable:.1%} "
    f"(full docket {all_pref_year:,.0f} cases per prefecture-year)")

# ---------------------------------------------------------------------------
# Translations.
# ---------------------------------------------------------------------------
g_acq = np.exp(b_acq) - 1.0
g_str = np.exp(b_str) - 1.0
g_diff_ratio = np.exp(b_diff) - 1.0            # acquaintance grows this much more
excess_per_pref_year = acq_pref_year * (np.exp(b_acq) - np.exp(b_str))
acq_inc_per_pref_year = acq_pref_year * g_acq
excess_pct_of_acq = 100.0 * (np.exp(b_acq) - np.exp(b_str))
national_excess = excess_per_pref_year * n_pref

# Delta-method SE on the excess. The two separate flow coefficients are
# estimated on the same cells and are positively correlated, so treating them
# as independent overstates the variance badly. Write the excess through the
# jointly estimated differential instead,
#     excess = baseline_acq * exp(b_str) * (exp(b_diff) - 1),
# which is algebraically the same quantity but carries the differential's own
# standard error. The stranger benchmark is held fixed, which is the intended
# conditional statement.
excess_via_diff = acq_pref_year * np.exp(b_str) * (np.exp(b_diff) - 1.0)
excess_se = float(acq_pref_year * np.exp(b_str) * np.exp(b_diff) * se_diff)
# for reference only: the (wrong) independent-marginals version
cov_indep = np.array([[se_acq ** 2, 0.0], [0.0, se_str ** 2]])
grad = np.array([np.exp(b_acq), -np.exp(b_str)]) * acq_pref_year
excess_se_indep = float(np.sqrt(grad @ cov_indep @ grad))
assert abs(excess_via_diff - excess_per_pref_year) < 1e-6, (
    excess_via_diff, excess_per_pref_year)

say(f"\n=== translations, per standard deviation of exposure ===")
say(f"  acquaintance filings   +{g_acq:6.1%}")
say(f"  stranger filings       +{g_str:6.1%}")
say(f"  differential (ratio)   +{g_diff_ratio:6.1%} more for acquaintance")
say(f"  acquaintance increment {acq_inc_per_pref_year:,.0f} cases per "
    f"prefecture-year")
say(f"  EXCESS over the stranger benchmark: "
    f"{excess_per_pref_year:,.0f} cases per prefecture-year "
    f"(se {excess_se:,.0f} from the joint differential; {excess_se_indep:,.0f} "
    f"if the two flows are wrongly treated as independent), "
    f"= {excess_pct_of_acq:.0f} percentage points more growth than "
    f"stranger loans")
say(f"  national, one SD everywhere: {national_excess:,.0f} excess "
    f"acquaintance cases per year")

rows = [
    {"quantity": "acquaintance flow, asinh points per SD", "value": b_acq,
     "std_error": se_acq, "wild_p": p_acq},
    {"quantity": "stranger flow, asinh points per SD", "value": b_str,
     "std_error": se_str, "wild_p": p_str},
    {"quantity": "acquaintance minus stranger, asinh points per SD",
     "value": b_diff, "std_error": se_diff, "wild_p": p_diff},
    {"quantity": "acquaintance filings, percent per SD", "value": 100 * g_acq,
     "std_error": np.nan, "wild_p": p_acq},
    {"quantity": "stranger filings, percent per SD", "value": 100 * g_str,
     "std_error": np.nan, "wild_p": p_str},
    {"quantity": "differential, percent more for acquaintance per SD",
     "value": 100 * g_diff_ratio, "std_error": np.nan, "wild_p": p_diff},
    {"quantity": "pre-campaign acquaintance cases per prefecture-year",
     "value": acq_pref_year, "std_error": np.nan, "wild_p": np.nan},
    {"quantity": "pre-campaign stranger cases per prefecture-year",
     "value": str_pref_year, "std_error": np.nan, "wild_p": np.nan},
    {"quantity": "pre-campaign acquaintance share of classified lending",
     "value": acq_share_pre, "std_error": np.nan, "wild_p": np.nan},
    {"quantity": "classifiable share of the full lending docket",
     "value": classifiable, "std_error": np.nan, "wild_p": np.nan},
    {"quantity": "excess acquaintance cases per prefecture-year per SD",
     "value": excess_per_pref_year, "std_error": excess_se, "wild_p": p_diff},
    {"quantity": "excess acquaintance cases, SE if flows treated as independent",
     "value": excess_per_pref_year, "std_error": excess_se_indep,
     "wild_p": np.nan},
    {"quantity": "excess growth, percentage points above stranger loans",
     "value": excess_pct_of_acq, "std_error": np.nan, "wild_p": np.nan},
    {"quantity": "national excess acquaintance cases per year, one SD",
     "value": national_excess, "std_error": np.nan, "wild_p": np.nan},
    {"quantity": "prefectures on the composition support", "value": n_pref,
     "std_error": np.nan, "wild_p": np.nan},
]
out = pd.DataFrame(rows)
out.to_csv(os.path.join(TAB, "composition_magnitude.csv"), index=False,
           encoding="utf-8-sig")

macros = {
    "AcqPctPerSD": f"{100 * g_acq:.0f}",
    "StrPctPerSD": f"{100 * g_str:.0f}",
    "AcqDiffPctPerSD": f"{100 * g_diff_ratio:.0f}",
    "AcqCasesAbs": f"{excess_per_pref_year:,.0f}",
    "AcqCasesAbsSE": f"{excess_se:,.0f}",
    "AcqWorkloadPct": f"{excess_pct_of_acq:.0f}",
    "AcqBasePrefYear": f"{acq_pref_year:,.0f}",
    "StrBasePrefYear": f"{str_pref_year:,.0f}",
    "AcqSharePre": f"{100 * acq_share_pre:.0f}",
    "AcqClassifiable": f"{100 * classifiable:.0f}",
    "AcqNatOneSD": f"{national_excess / 1000:,.0f}",
    "AcqSupportPref": f"{n_pref}",
}
tex = "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items())
stems = ["numbers_composition_magnitude.tex"]
if VERSIONED_OUTPUTS:
    stems.insert(0, f"numbers_composition_magnitude_{STAMP}.tex")
for stem in stems:
    with open(os.path.join(TAB, stem), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(tex)
say("\n=== macros ===")
say(tex.strip())
say(f"\nwritten -> {TAB}")

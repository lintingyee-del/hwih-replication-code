# -*- coding: utf-8 -*-
"""Clean-window event study for acquaintance-minus-stranger lending flow.

The target coefficient in every non-reference calendar bin is
bin x wave-1 x exposure x acquaintance.  The regression includes all lower-order
bin interactions required by the pooled static composition equation and absorbs
prefecture-by-group and month-by-group fixed effects.  Two frozen samples are
reported: the paper's missing-as-stranger coding and the corrected classified-
text-only coding, both with explicit zero group cells.

Output: output/ext2124/composition_clean_calendar_eventstudy.csv
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import pyfixest as pf
from scipy import stats

from _wild import wild_score_p


DATA = str(_REP_PROJECT / "data")
OUT = str(_REP_PROJECT / "output" / "ext2124")
START, END, POST0 = "2017-01", "2019-03", "2018-09"
PRE_START, PRE_END = "2014-01", "2017-12"
BINS = [(-20, -13), (-12, -7), (-6, -1), (0, 6)]
REFERENCE = (-6, -1)
ROWS = []


schedule = (
    pd.read_parquet(f"{DATA}/panel_month.parquet")[["province", "inspection_round"]]
    .drop_duplicates()
)
exposure = pd.read_parquet(f"{DATA}/exposure_v2.parquet")[[
    "prefecture_code", "province", "exposure_v2_z"
]].drop_duplicates()
case = pd.read_parquet(
    f"{DATA}/civil_case.parquet",
    columns=["cause", "prefecture_code", "province", "jmonth", "rel_txn"],
)
case["month"] = case["jmonth"].astype(str).str[:7]
lending = case[case["cause"].eq("民间借贷纠纷")].copy()
months = pd.DataFrame({"month": pd.period_range(START, END, freq="M").astype(str)})
groups = pd.DataFrame({"acq": [0, 1]})


def make_panel(classified_only):
    d = lending.copy()
    if classified_only:
        d = d[d["rel_txn"].notna()].copy()
        d["acq"] = d["rel_txn"].astype(int)
    else:
        d["acq"] = d["rel_txn"].fillna(0).astype(int)
    support = (
        d[d["month"].between(PRE_START, PRE_END)][
            ["prefecture_code", "province"]
        ].drop_duplicates()
        .merge(exposure[["prefecture_code", "province"]],
               on=["prefecture_code", "province"], how="inner")
    )
    counts = (
        d[d["month"].between(START, END)]
        .groupby(["prefecture_code", "province", "month", "acq"])
        .size().rename("n").reset_index()
    )
    panel = support.merge(months, how="cross").merge(groups, how="cross").merge(
        counts, on=["prefecture_code", "province", "month", "acq"], how="left"
    )
    panel["n"] = panel["n"].fillna(0).astype(float)
    panel = (
        panel.merge(schedule, on="province")
        .merge(exposure[["prefecture_code", "exposure_v2_z"]], on="prefecture_code")
    )
    panel["y"] = np.arcsinh(panel["n"])
    panel["treat"] = (panel["inspection_round"] == 1).astype(int)
    panel["prov_id"] = pd.factorize(panel["province"])[0]
    panel["prefA"] = panel["prefecture_code"] + "_" + panel["acq"].astype(str)
    panel["monthA"] = panel["month"] + "_" + panel["acq"].astype(str)
    launch = pd.Period(POST0, freq="M").ordinal
    panel["cal_time"] = pd.PeriodIndex(panel["month"], freq="M").astype(int) - launch
    return panel


def run(panel, sample):
    targets = []
    controls = []
    lookup = {}
    for lo, hi in BINS:
        if (lo, hi) == REFERENCE:
            continue
        tag = f"m{abs(lo)}_m{abs(hi)}" if hi < 0 else f"p{lo}_p{hi}"
        indicator = panel["cal_time"].between(lo, hi).astype(float)
        ht = indicator * panel["treat"] * panel["exposure_v2_z"]
        target = f"HTA_{tag}"
        panel[target] = ht * panel["acq"]
        panel[f"HA_{tag}"] = indicator * panel["exposure_v2_z"] * panel["acq"]
        panel[f"TA_{tag}"] = indicator * panel["treat"] * panel["acq"]
        panel[f"HT_{tag}"] = ht
        panel[f"H_{tag}"] = indicator * panel["exposure_v2_z"]
        panel[f"T_{tag}"] = indicator * panel["treat"]
        targets.append(target)
        controls.extend([
            f"HA_{tag}", f"TA_{tag}", f"HT_{tag}", f"H_{tag}", f"T_{tag}"
        ])
        lookup[target] = (lo, hi)

    formula = f"y ~ {' + '.join(targets + controls)} | prefA + monthA"
    model = pf.feols(formula, data=panel, vcov={"CRV1": "prov_id"})
    names = list(model.coef().index)
    idx = [names.index(term) for term in targets]
    beta = model.coef().values[idx]
    vcov = model._vcov[np.ix_(idx, idx)]
    pre_idx = [i for i, term in enumerate(targets) if lookup[term][1] < 0]
    bpre = beta[pre_idx]
    vpre = vcov[np.ix_(pre_idx, pre_idx)]
    lead_wald = float(bpre.T @ np.linalg.pinv(vpre) @ bpre)
    lead_p = float(stats.chi2.sf(lead_wald, len(pre_idx)))

    for i, term in enumerate(targets):
        lo, hi = lookup[term]
        se = float(np.sqrt(vcov[i, i]))
        pw = wild_score_p(formula, panel, term)
        ROWS.append({
            "sample": sample,
            "bin_start": lo,
            "bin_end": hi,
            "reference_bin": "[-6,-1]",
            "coefficient": float(beta[i]),
            "std_error_crv1": se,
            "ci95_low": float(beta[i] - 1.96 * se),
            "ci95_high": float(beta[i] + 1.96 * se),
            "p_wild": float(pw),
            "joint_lead_p_chi2": lead_p,
            "n_obs": int(model._N),
            "formula": formula,
        })
        print(
            f"{sample:24s} [{lo:3d},{hi:2d}] b={float(beta[i]):+.6f} "
            f"se={se:.6f} wild={pw:.4f}",
            flush=True,
        )
    print(f"{sample:24s} joint-lead chi2 p={lead_p:.4f}", flush=True)


run(make_panel(False), "balanced_missing_as_stranger")
run(make_panel(True), "balanced_classified_only")

os.makedirs(OUT, exist_ok=True)
pd.DataFrame(ROWS).to_csv(f"{OUT}/composition_clean_calendar_eventstudy.csv", index=False)
print("written composition_clean_calendar_eventstudy.csv", flush=True)

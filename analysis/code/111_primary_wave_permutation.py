# -*- coding: utf-8 -*-
"""Wave-label permutation diagnostics for the two revised primary civil estimands.

Step 110 rebuilt both primary estimands on balanced panels (predetermined
2014--2017 support, zero-count cells restored, classified-flag cases only for
the composition contrast) but produced no permutation diagnostic.  The
permutation numbers still carried in the manuscript preamble predate that
rebuild:

  * \\RefPermCiv (0.116) belongs to the observed-case-support flow estimate
    (0.156), not to the balanced 0.190; and
  * \\CutAcqDiffPermP (0.070) belongs to step 81's pooled lending
    specification, which assigned unclassified cases to the stranger group
    (fillna(0)) on observed-case support and estimated 0.130, not to the
    balanced classified-only 0.182.

This script reassigns first-wave labels across the 31 provinces, holding the
treated count fixed at its observed value, and refits each estimand on the same
balanced panels step 110 uses.  Because inspection timing was not randomized,
the exercise is a diagnostic under conditional exchangeability, not exact
design-based randomization inference.
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
import shutil
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import pyfixest as pf


DATA = str(_REP_PROJECT / "data")
OUT = str(_REP_PROJECT / "output" / "ext2124")
START, END, POST0 = "2017-01", "2019-03", "2018-09"
SUPPORT_START, SUPPORT_END = "2014-01", "2017-12"
REPS = 999
SEED = 42
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
VERSIONED_OUTPUTS = os.environ.get("HWIH_REPLICATION", "0") != "1"

os.makedirs(OUT, exist_ok=True)

schedule = (
    pd.read_parquet(f"{DATA}/panel_month.parquet")[["province", "inspection_round"]]
    .drop_duplicates()
)
exposure = pd.read_parquet(f"{DATA}/exposure_v2.parquet")[[
    "prefecture_code", "province", "exposure_v2_z"
]].drop_duplicates()
months = pd.DataFrame({"month": pd.period_range(START, END, freq="M").astype(str)})


# ---------------------------------------------------------------------------
# Panel construction, mirroring step 110 exactly.
# ---------------------------------------------------------------------------
civil_panel = pd.read_parquet(f"{DATA}/civil_panel.parquet")
civil_panel["month"] = civil_panel["jmonth"].astype(str).str[:7]
rel = civil_panel[civil_panel["cause_family"].eq("relational")].copy()
flow_support = rel[rel["month"].between(SUPPORT_START, SUPPORT_END)][[
    "prefecture_code", "province", "cause"
]].drop_duplicates()
flow_counts = rel[rel["month"].between(START, END)][[
    "prefecture_code", "province", "cause", "month", "n_cases"
]]
flow = flow_support.merge(months, how="cross").merge(
    flow_counts, on=["prefecture_code", "province", "cause", "month"], how="left"
)
flow["n"] = flow["n_cases"].fillna(0).astype(float)
flow = (
    flow.merge(schedule, on="province")
    .merge(exposure[["prefecture_code", "exposure_v2_z"]], on="prefecture_code")
)
flow["y"] = np.arcsinh(flow["n"])
flow["pref_cause"] = flow["prefecture_code"] + "_" + flow["cause"]
flow["postc"] = (flow["month"] >= POST0).astype(int)
flow["prov_id"] = pd.factorize(flow["province"])[0]

case = pd.read_parquet(
    f"{DATA}/civil_case.parquet",
    columns=["cause", "prefecture_code", "province", "jmonth", "rel_txn"],
)
case["month"] = case["jmonth"].astype(str).str[:7]
lending_all = case[case["cause"].eq("民间借贷纠纷")].copy()
lending = lending_all[lending_all["rel_txn"].notna()].copy()
lending["acq"] = lending["rel_txn"].astype(int)
comp_support = (
    lending[lending["month"].between(SUPPORT_START, SUPPORT_END)][
        ["prefecture_code", "province"]
    ].drop_duplicates()
    .merge(exposure[["prefecture_code", "province"]],
           on=["prefecture_code", "province"], how="inner")
)
comp_counts = (
    lending[lending["month"].between(START, END)]
    .groupby(["prefecture_code", "province", "month", "acq"])
    .size().rename("n").reset_index()
)
composition = (
    comp_support.merge(months, how="cross")
    .merge(pd.DataFrame({"acq": [0, 1]}), how="cross")
    .merge(comp_counts, on=["prefecture_code", "province", "month", "acq"], how="left")
)
composition["n"] = composition["n"].fillna(0).astype(float)
composition = (
    composition.merge(schedule, on="province")
    .merge(exposure[["prefecture_code", "exposure_v2_z"]], on="prefecture_code")
)
composition["y"] = np.arcsinh(composition["n"])
composition["prefA"] = composition["prefecture_code"] + "_" + composition["acq"].astype(str)
composition["monthA"] = composition["month"] + "_" + composition["acq"].astype(str)
composition["postc"] = (composition["month"] >= POST0).astype(int)
composition["prov_id"] = pd.factorize(composition["province"])[0]


def apply_labels(data, treatment_map, interact_acq):
    d = data.copy()
    d["treat"] = d["province"].map(treatment_map).astype(int)
    d["pt"] = d["postc"] * d["treat"]
    d["ph"] = d["postc"] * d["exposure_v2_z"]
    d["pth"] = d["pt"] * d["exposure_v2_z"]
    if interact_acq:
        for term in ("pth", "ph", "pt"):
            d[f"{term}A"] = d[term] * d["acq"]
    return d


ESTIMANDS = {
    "balanced_relational_flow": {
        "data": flow,
        "formula": "y ~ pth + ph + pt | pref_cause + month",
        "coefficient": "pth",
        "interact_acq": False,
    },
    "classified_acquaintance_minus_stranger": {
        "data": composition,
        "formula": "y ~ pthA + phA + ptA + pth + ph + pt | prefA + monthA",
        "coefficient": "pthA",
        "interact_acq": True,
    },
}

rows = []
for name, spec in ESTIMANDS.items():
    data = spec["data"]
    province_schedule = data[["province", "inspection_round"]].drop_duplicates()
    observed_map = dict(zip(
        province_schedule["province"],
        (province_schedule["inspection_round"] == 1).astype(int),
    ))
    observed = apply_labels(data, observed_map, spec["interact_acq"])
    fit = pf.feols(spec["formula"], data=observed, vcov={"CRV1": "prov_id"})
    beta = float(fit.coef()[spec["coefficient"]])

    provinces = np.array(sorted(observed_map))
    n_treated = int(sum(observed_map.values()))
    rng = np.random.default_rng(SEED)
    permuted = np.empty(REPS)
    for draw in range(REPS):
        labels = np.zeros(len(provinces), dtype=int)
        labels[rng.choice(len(provinces), n_treated, replace=False)] = 1
        d = apply_labels(data, dict(zip(provinces, labels)), spec["interact_acq"])
        permuted[draw] = float(
            pf.feols(spec["formula"], data=d, vcov="iid").coef()[spec["coefficient"]]
        )
        if (draw + 1) % 200 == 0:
            print(f"{name}: {draw + 1}/{REPS}", flush=True)

    p_two = float((1 + np.sum(np.abs(permuted) >= abs(beta))) / (REPS + 1))
    p_one = float((1 + np.sum(np.sign(beta) * permuted >= abs(beta))) / (REPS + 1))
    rows.append({
        "estimand": name,
        "coefficient": spec["coefficient"],
        "estimate": beta,
        "se_crv1": float(fit.se()[spec["coefficient"]]),
        "p_crv1": float(fit.pvalue()[spec["coefficient"]]),
        "p_wave_permutation_two_sided": p_two,
        "p_wave_permutation_one_sided": p_one,
        "permutation_mean": float(permuted.mean()),
        "permutation_sd": float(permuted.std(ddof=1)),
        "permutation_q025": float(np.quantile(permuted, 0.025)),
        "permutation_q975": float(np.quantile(permuted, 0.975)),
        "permutation_reps": REPS,
        "seed": SEED,
        "province_clusters": len(provinces),
        "treated_provinces": n_treated,
        "regression_cells": int(fit._N),
        "formula": spec["formula"],
    })
    print(f"\n{name}: beta={beta:.4f}  two-sided perm p={p_two:.3f}  "
          f"one-sided={p_one:.3f}", flush=True)

result = pd.DataFrame(rows)
latest = f"{OUT}/primary_wave_permutation.csv"
if VERSIONED_OUTPUTS:
    timestamped = f"{OUT}/primary_wave_permutation_{STAMP}.csv"
    result.to_csv(timestamped, index=False)
    shutil.copyfile(timestamped, latest)
else:
    result.to_csv(latest, index=False)

print("\nRESULTS", flush=True)
print(result[[
    "estimand", "estimate", "se_crv1", "p_crv1",
    "p_wave_permutation_two_sided", "p_wave_permutation_one_sided",
    "permutation_sd", "regression_cells",
]].round(4).to_string(index=False), flush=True)
print(
    "\nInterpretation: inspection timing was not randomized, so these are "
    "diagnostics under conditional exchangeability, not exact randomization "
    "inference.",
    flush=True,
)

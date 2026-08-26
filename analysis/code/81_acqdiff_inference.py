# -*- coding: utf-8 -*-
"""Inference sensitivity for the clean-window acquaintance-minus-stranger result.

Rebuilds step 61's pooled lending specification, reports CRV1 and CRV3, and
reassigns first-wave labels across provinces while holding the treated count
fixed.  Because inspection timing was not randomized, the reassignment exercise
is a wave-permutation diagnostic under conditional exchangeability, not exact
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
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import pyfixest as pf

from _wild import wild_p


DATA = str(_REP_PROJECT / "data")
OUT = str(_REP_PROJECT / "output" / "ext2124")
WINDOW = ("2017-01", "2019-03")
POST0 = "2018-09"
REPS = 999
SEED = 42


cc = pd.read_parquet(
    f"{DATA}/civil_case.parquet",
    columns=["cause", "prefecture_code", "province", "jmonth", "rel_txn"],
)
ld = cc[cc["cause"] == "民间借贷纠纷"].copy()
ld["month"] = ld["jmonth"].astype(str).str[:7]
ld = ld[(ld["month"] >= WINDOW[0]) & (ld["month"] <= WINDOW[1])]
ld["acq"] = ld["rel_txn"].fillna(0).astype(int)
case_counts = ld.groupby("acq").size().to_dict()

sched = pd.read_parquet(f"{DATA}/panel_month.parquet")[
    ["province", "inspection_round"]
].drop_duplicates()
exposure = pd.read_parquet(f"{DATA}/exposure_v2.parquet")[
    ["prefecture_code", "exposure_v2_z"]
]

g = (
    ld.groupby(["prefecture_code", "province", "month", "acq"])
    .size()
    .rename("n")
    .reset_index()
    .merge(sched, on="province")
    .merge(exposure, on="prefecture_code")
    .dropna(subset=["exposure_v2_z", "inspection_round"])
)
g["H"] = g["exposure_v2_z"]
g["postc"] = (g["month"] >= POST0).astype(int)
g["y"] = np.arcsinh(g["n"])
g["prov_id"] = pd.factorize(g["province"])[0]
g["prefA"] = g["prefecture_code"] + "_" + g["acq"].astype(str)
g["monthA"] = g["month"] + "_" + g["acq"].astype(str)

FML = "y ~ pthA + phA + ptA + postcA + pth + ph + pt | prefA + monthA"


def add_treatment_terms(data, treatment_map):
    d = data.copy()
    d["treat"] = d["province"].map(treatment_map).astype(int)
    d["pt"] = d["postc"] * d["treat"]
    d["pth"] = d["pt"] * d["H"]
    d["ph"] = d["postc"] * d["H"]
    for name in ("pth", "ph", "pt", "postc"):
        d[f"{name}A"] = d[name] * d["acq"]
    return d


province_schedule = g[["province", "inspection_round"]].drop_duplicates()
observed_map = dict(
    zip(
        province_schedule["province"],
        (province_schedule["inspection_round"] == 1).astype(int),
    )
)
observed = add_treatment_terms(g, observed_map)

crv1 = pf.feols(FML, data=observed, vcov={"CRV1": "prov_id"})
crv3 = pf.feols(FML, data=observed, vcov={"CRV3": "prov_id"})
wild = wild_p(FML, observed, "pthA")
beta = float(crv1.coef()["pthA"])

provinces = np.array(sorted(observed_map))
n_treated = int(sum(observed_map.values()))
rng = np.random.default_rng(SEED)
permuted = np.empty(REPS)
for draw in range(REPS):
    labels = np.zeros(len(provinces), dtype=int)
    labels[rng.choice(len(provinces), n_treated, replace=False)] = 1
    d = add_treatment_terms(g, dict(zip(provinces, labels)))
    permuted[draw] = float(pf.feols(FML, data=d, vcov="iid").coef()["pthA"])
    if (draw + 1) % 100 == 0:
        print(f"wave permutations: {draw + 1}/{REPS}", flush=True)

permutation_p = float((1 + np.sum(np.abs(permuted) >= abs(beta))) / (REPS + 1))
result = pd.DataFrame(
    [
        {
            "estimand": "acquaintance_minus_stranger_pthA",
            "estimate": beta,
            "se_crv1": float(crv1.se()["pthA"]),
            "p_crv1": float(crv1.pvalue()["pthA"]),
            "se_crv3": float(crv3.se()["pthA"]),
            "p_crv3": float(crv3.pvalue()["pthA"]),
            "p_wild_score": float(wild),
            "p_wave_permutation": permutation_p,
            "permutation_reps": REPS,
            "province_clusters": len(provinces),
            "treated_provinces": n_treated,
            "regression_cells": int(crv1._N),
            "acquaintance_cases": int(case_counts.get(1, 0)),
            "stranger_cases": int(case_counts.get(0, 0)),
        }
    ]
)
os.makedirs(OUT, exist_ok=True)
result.to_csv(f"{OUT}/acqdiff_inference.csv", index=False)

print(result.to_string(index=False), flush=True)
print(
    "Interpretation: the wave-permutation p-value is a diagnostic under "
    "conditional exchangeability, not exact randomization inference.",
    flush=True,
)

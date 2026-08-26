# -*- coding: utf-8 -*-
"""Clean-window relational-minus-traffic gap with small-cluster inference.

Aggregates relational causes and the traffic-tort placebo to prefecture-month
totals, then estimates the first-wave dose design on their asinh difference.
This makes explicit that the relational-only flow coefficient and the placebo-
adjusted contrast are different estimands.
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


c = pd.read_parquet(f"{DATA}/civil_panel.parquet")
c["month"] = c["jmonth"].astype(str).str[:7]
c = c[(c["month"] >= WINDOW[0]) & (c["month"] <= WINDOW[1])].copy()
c["group"] = np.where(
    c["cause_family"].eq("relational"),
    "relational",
    np.where(c["cause_family"].eq("placebo"), "traffic", "other"),
)
c = c[c["group"].isin(["relational", "traffic"])]

g = (
    c.groupby(["prefecture_code", "province", "month", "group"], as_index=False)[
        "n_cases"
    ]
    .sum()
    .pivot_table(
        index=["prefecture_code", "province", "month"],
        columns="group",
        values="n_cases",
        fill_value=0,
    )
    .reset_index()
)
sched = pd.read_parquet(f"{DATA}/panel_month.parquet")[
    ["province", "inspection_round"]
].drop_duplicates()
exposure = pd.read_parquet(f"{DATA}/exposure_v2.parquet")[
    ["prefecture_code", "exposure_v2_z"]
]
g = (
    g.merge(sched, on="province")
    .merge(exposure, on="prefecture_code")
    .dropna(subset=["inspection_round", "exposure_v2_z"])
)
g["y"] = np.arcsinh(g["relational"]) - np.arcsinh(g["traffic"])
g["H"] = g["exposure_v2_z"]
g["treat"] = (g["inspection_round"] == 1).astype(int)
g["postc"] = (g["month"] >= POST0).astype(int)
g["pt"] = g["postc"] * g["treat"]
g["pth"] = g["pt"] * g["H"]
g["ph"] = g["postc"] * g["H"]
g["prov_id"] = pd.factorize(g["province"])[0]

fml = "y ~ pth + ph + pt | prefecture_code + month"
crv1 = pf.feols(fml, data=g, vcov={"CRV1": "prov_id"})
crv3 = pf.feols(fml, data=g, vcov={"CRV3": "prov_id"})
wild = wild_p(fml, g, "pth")

result = pd.DataFrame(
    [
        {
            "estimand": "clean_relational_minus_traffic_gap",
            "estimate": float(crv1.coef()["pth"]),
            "se_crv1": float(crv1.se()["pth"]),
            "p_crv1": float(crv1.pvalue()["pth"]),
            "se_crv3": float(crv3.se()["pth"]),
            "p_crv3": float(crv3.pvalue()["pth"]),
            "p_wild_score": float(wild),
            "province_clusters": int(g["prov_id"].nunique()),
            "regression_cells": int(crv1._N),
        }
    ]
)
os.makedirs(OUT, exist_ok=True)
result.to_csv(f"{OUT}/clean_traffic_gap.csv", index=False)
print(result.to_string(index=False))

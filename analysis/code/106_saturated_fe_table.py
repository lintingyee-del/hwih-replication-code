# -*- coding: utf-8 -*-
"""Reproduce the three saturated-FE rows reported in the paper appendix.

Output: output/ext2124/saturated_fe_table.csv
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

from _wild import wild_score_p


DATA = str(_REP_PROJECT / "data")
OUT = str(_REP_PROJECT / "output" / "ext2124")
WINDOW = ("2017-01", "2019-03")
SUPPORT = ("2014-01", "2017-12")
POST0 = "2018-09"
ROWS = []

schedule = (
    pd.read_parquet(f"{DATA}/panel_month.parquet")[["province", "inspection_round"]]
    .drop_duplicates()
)
exposure = pd.read_parquet(f"{DATA}/exposure_v2.parquet")[[
    "prefecture_code", "exposure_v2_z"
]]
months = pd.DataFrame({
    "month": pd.period_range(WINDOW[0], WINDOW[1], freq="M").astype(str)
})


def add_design(data):
    d = data.copy()
    d["treat"] = (d["inspection_round"] == 1).astype(int)
    d["postc"] = (d["month"] >= POST0).astype(int)
    d["prov_id"] = pd.factorize(d["province"])[0]
    d["pt"] = d["postc"] * d["treat"]
    d["pth"] = d["pt"] * d["exposure_v2_z"]
    d["ph"] = d["postc"] * d["exposure_v2_z"]
    d["prov_month"] = d["province"] + "_" + d["month"]
    return d


def run(label, formula, data, coefficient, fixed_effects):
    crv1 = pf.feols(formula, data=data, vcov={"CRV1": "prov_id"})
    crv3 = pf.feols(formula, data=data, vcov={"CRV3": "prov_id"})
    p_wild = wild_score_p(formula, data, coefficient)
    row = {
        "specification": label,
        "coefficient": float(crv1.coef()[coefficient]),
        "se_crv1": float(crv1.se()[coefficient]),
        "p_crv1": float(crv1.pvalue()[coefficient]),
        "p_wild_score": float(p_wild),
        "se_crv3": float(crv3.se()[coefficient]),
        "p_crv3": float(crv3.pvalue()[coefficient]),
        "bh_q_four_cause_asinh": np.nan,
        "n": int(crv1._N),
        "province_clusters": int(data["prov_id"].nunique()),
        "fixed_effects": fixed_effects,
        "formula": formula,
    }
    ROWS.append(row)
    print(
        f"{label:36s} b={row['coefficient']:+.6f} se1={row['se_crv1']:.6f} "
        f"p1={row['p_crv1']:.4f} wild={row['p_wild_score']:.4f} "
        f"p3={row['p_crv3']:.4f} N={row['n']:,}",
        flush=True,
    )


# 1. Judgment-dated total relational-cause flow.
civil = pd.read_parquet(f"{DATA}/civil_panel.parquet")
civil["month"] = civil["jmonth"].astype(str).str[:7]
relational = civil[civil["cause_family"].eq("relational")].copy()
total_support = relational[
    relational["month"].between(SUPPORT[0], SUPPORT[1])
][["prefecture_code", "province", "cause"]].drop_duplicates()
total_counts = relational[
    relational["month"].between(WINDOW[0], WINDOW[1])
][["prefecture_code", "province", "cause", "month", "n_cases"]]
total = total_support.merge(months, how="cross").merge(
    total_counts,
    on=["prefecture_code", "province", "cause", "month"],
    how="left",
)
total["n_cases"] = total["n_cases"].fillna(0).astype(float)
total = (
    total.merge(schedule, on="province")
    .merge(exposure, on="prefecture_code")
    .dropna(subset=["inspection_round", "exposure_v2_z"])
)
total = add_design(total)
total["y"] = np.arcsinh(total["n_cases"])
total["pref_cause"] = total["prefecture_code"] + "_" + total["cause"]
total["cause_month"] = total["cause"] + "_" + total["month"]
run(
    "Total relational-cause flow",
    "y ~ pth + ph | pref_cause + prov_month + cause_month",
    total,
    "pth",
    "prefecture_x_cause + province_x_month + cause_x_month",
)


# 2. Acquaintance-minus-stranger composition contrast.
case = pd.read_parquet(
    f"{DATA}/civil_case.parquet",
    columns=["cause", "prefecture_code", "province", "jmonth", "rel_txn"],
)
lending = case[
    case["cause"].eq("民间借贷纠纷") & case["rel_txn"].notna()
].copy()
lending["month"] = lending["jmonth"].astype(str).str[:7]
lending["acq"] = lending["rel_txn"].astype(int)
composition_support = (
    lending[lending["month"].between(SUPPORT[0], SUPPORT[1])][
        ["prefecture_code", "province"]
    ]
    .drop_duplicates()
    .merge(exposure[["prefecture_code"]], on="prefecture_code")
)
composition_counts = (
    lending[lending["month"].between(WINDOW[0], WINDOW[1])]
    .groupby(["prefecture_code", "province", "month", "acq"])
    .size()
    .rename("n")
    .reset_index()
)
composition = (
    composition_support.merge(months, how="cross")
    .merge(pd.DataFrame({"acq": [0, 1]}), how="cross")
    .merge(
        composition_counts,
        on=["prefecture_code", "province", "month", "acq"],
        how="left",
    )
    .merge(schedule, on="province")
    .merge(exposure, on="prefecture_code")
    .dropna(subset=["inspection_round", "exposure_v2_z"])
)
composition["n"] = composition["n"].fillna(0).astype(float)
composition = add_design(composition)
composition["y"] = np.arcsinh(composition["n"])
composition["prefA"] = composition["prefecture_code"] + "_" + composition["acq"].astype(str)
composition["monthA"] = composition["month"] + "_" + composition["acq"].astype(str)
for term in ("pth", "ph", "pt"):
    composition[f"{term}A"] = composition[term] * composition["acq"]
run(
    "Acquaintance minus stranger",
    "y ~ pthA + phA + ptA + pth + ph | prefA + prov_month + monthA",
    composition,
    "pthA",
    "prefecture_x_group + province_x_month + month_x_group",
)


# 3. Recourse-dispute flow on support fixed before the campaign.  Re-estimate
# the same balanced-asinh specification for all four relational causes so the
# reported recourse q-value is generated from its complete outcome family.
def balanced_cause(cause):
    cause_all = civil[civil["cause"] == cause].copy()
    support = cause_all[
        cause_all["month"].between("2014-01", "2017-12")
    ][["prefecture_code", "province"]].drop_duplicates()
    grid = support.merge(months, how="cross")
    counts = cause_all[
        cause_all["month"].between(WINDOW[0], WINDOW[1])
    ][["prefecture_code", "province", "month", "n_cases"]]
    d = grid.merge(
        counts, on=["prefecture_code", "province", "month"], how="left"
    )
    d["n"] = d["n_cases"].fillna(0)
    d = (
        d.merge(schedule, on="province")
        .merge(exposure, on="prefecture_code")
        .dropna(subset=["inspection_round", "exposure_v2_z"])
    )
    d = add_design(d)
    d["y"] = np.arcsinh(d["n"])
    return d


cause_formula = "y ~ pth + ph | prefecture_code + prov_month"
cause_p = {}
for cause in sorted(civil.loc[civil["cause_family"] == "relational", "cause"].unique()):
    d = balanced_cause(cause)
    cause_p[cause] = wild_score_p(cause_formula, d, "pth")

ordered = sorted(cause_p, key=cause_p.get)
raw_q = np.array([cause_p[cause] for cause in ordered]) * len(ordered) / np.arange(1, len(ordered) + 1)
adj_q = np.minimum.accumulate(raw_q[::-1])[::-1].clip(max=1.0)
cause_q = dict(zip(ordered, adj_q))

recourse = balanced_cause("追偿权纠纷")
run(
    "Recourse disputes, balanced panel",
    cause_formula,
    recourse,
    "pth",
    "prefecture + province_x_month",
)
ROWS[-1]["bh_q_four_cause_asinh"] = float(cause_q["追偿权纠纷"])
print(f"Recourse four-cause BH q={ROWS[-1]['bh_q_four_cause_asinh']:.4f}", flush=True)


os.makedirs(OUT, exist_ok=True)
result = pd.DataFrame(ROWS)
path = f"{OUT}/saturated_fe_table.csv"
result.to_csv(path, index=False)
print(f"written: {path}", flush=True)

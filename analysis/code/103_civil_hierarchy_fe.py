# -*- coding: utf-8 -*-
"""Fixed-effect audit for the pre-specified civil estimand hierarchy.

The script changes fixed effects only and reports every attempted variant for
the filing-clock, acquaintance-minus-stranger, and relational-minus-traffic
estimands.  It complements 102_civil_fe.py, which audits the primary
judgment-dated relational-flow outcome.

Output: output/ext2124/civil_hierarchy_fe.csv
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
POST0 = "2018-09"
ROWS = []

schedule = (
    pd.read_parquet(f"{DATA}/panel_month.parquet")[["province", "inspection_round"]]
    .drop_duplicates()
)
exposure = pd.read_parquet(f"{DATA}/exposure_v2.parquet")[[
    "prefecture_code", "exposure_v2_z"
]]


def fit(spec_id, estimand, formula, data, coefficient, fixed_effects, role):
    crv1 = pf.feols(formula, data=data, vcov={"CRV1": "prov_id"})
    crv3 = pf.feols(formula, data=data, vcov={"CRV3": "prov_id"})
    try:
        p_wild = wild_score_p(formula, data, coefficient)
    except Exception as exc:
        print(f"{spec_id}: wild-score failed: {exc}", flush=True)
        p_wild = np.nan
    beta = float(crv1.coef()[coefficient])
    row = {
        "spec_id": spec_id,
        "mode": "B_fixed_x_y_equation",
        "focus_side": "fixed_effects_only",
        "base_variable": estimand,
        "transformation": "asinh_count_or_predefined_asinh_difference",
        "model": "OLS_high_dimensional_FE",
        "sample_rule": "clean_window_2017-01_to_2019-03; wave1_vs_not_yet_treated",
        "controls": formula.split("|")[0].split("~", 1)[1].strip(),
        "fixed_effects": fixed_effects,
        "coefficient": beta,
        "std_error": float(crv1.se()[coefficient]),
        "p_value": float(crv1.pvalue()[coefficient]),
        "std_error_crv3": float(crv3.se()[coefficient]),
        "p_crv3": float(crv3.pvalue()[coefficient]),
        "p_wild": p_wild,
        "n_obs": int(crv1._N),
        "province_clusters": int(data["prov_id"].nunique()),
        "direction": "positive" if beta > 0 else "negative" if beta < 0 else "zero",
        "keep_or_drop": "retain_in_complete_audit_log",
        "reason": role,
    }
    ROWS.append(row)
    print(
        f"{spec_id:3s} {estimand:39s} b={beta:+.6f} "
        f"se1={row['std_error']:.6f} p1={row['p_value']:.4f} "
        f"se3={row['std_error_crv3']:.6f} p3={row['p_crv3']:.4f} "
        f"wild={p_wild:.4f} N={row['n_obs']:,}",
        flush=True,
    )


def treatment_terms(data):
    g = data.copy()
    g["treat"] = (g["inspection_round"] == 1).astype(int)
    g["postc"] = (g["month"] >= POST0).astype(int)
    g["prov_id"] = pd.factorize(g["province"])[0]
    g["pt"] = g["postc"] * g["treat"]
    g["pth"] = g["pt"] * g["exposure_v2_z"]
    g["ph"] = g["postc"] * g["exposure_v2_z"]
    g["prov_month"] = g["province"] + "_" + g["month"]
    return g


# Filing-clock timing validation: preserve the published extracted-date support.
case = pd.read_parquet(
    f"{DATA}/civil_case.parquet",
    columns=["case_no", "cause", "cause_family", "prefecture_code", "province", "jmonth"],
)
rel = case[case["cause_family"] == "relational"].copy()
filing = pd.read_parquet(f"{DATA}/civil_filing.parquet").rename(columns={"案号": "case_no"})
rel = rel.merge(filing[["case_no", "filing_ymd"]], on="case_no", how="left")
rel["fdate"] = pd.to_datetime(rel["filing_ymd"], errors="coerce")
rel["jdate"] = pd.to_datetime(rel["jmonth"], errors="coerce")
rel["duration"] = (rel["jdate"] - rel["fdate"]).dt.days
rel = rel[rel["fdate"].notna() & rel["duration"].between(0, 270)].copy()
rel["month"] = rel["fdate"].dt.strftime("%Y-%m")
filing_cells = (
    rel.groupby(["prefecture_code", "province", "cause", "month"])
    .size()
    .rename("n")
    .reset_index()
)
filing_cells = filing_cells[
    (filing_cells["month"] >= WINDOW[0]) & (filing_cells["month"] <= WINDOW[1])
]
filing_cells = (
    filing_cells.merge(schedule, on="province")
    .merge(exposure, on="prefecture_code")
    .dropna(subset=["inspection_round", "exposure_v2_z"])
)
filing_cells = treatment_terms(filing_cells)
filing_cells["y"] = np.arcsinh(filing_cells["n"])
filing_cells["pref_cause"] = filing_cells["prefecture_code"] + "_" + filing_cells["cause"]
filing_cells["cause_month"] = filing_cells["cause"] + "_" + filing_cells["month"]

fit(
    "F0", "filing_clock_relational_flow",
    "y ~ pth + ph + pt | pref_cause + month", filing_cells, "pth",
    "prefecture_x_cause + month", "published timing-validation anchor",
)
fit(
    "F1", "filing_clock_relational_flow",
    "y ~ pth + ph | pref_cause + prov_month", filing_cells, "pth",
    "prefecture_x_cause + province_x_month", "province-by-month saturation",
)
fit(
    "F2", "filing_clock_relational_flow",
    "y ~ pth + ph | pref_cause + prov_month + cause_month", filing_cells, "pth",
    "prefecture_x_cause + province_x_month + cause_x_month",
    "fully requested saturation",
)


# Primary composition contrast: preserve the published lending and rel_txn coding.
lending = pd.read_parquet(
    f"{DATA}/civil_case.parquet",
    columns=["cause", "prefecture_code", "province", "jmonth", "rel_txn"],
)
lending = lending[lending["cause"] == "民间借贷纠纷"].copy()
lending["month"] = lending["jmonth"].astype(str).str[:7]
lending = lending[(lending["month"] >= WINDOW[0]) & (lending["month"] <= WINDOW[1])]
lending["acq"] = lending["rel_txn"].fillna(0).astype(int)
composition = (
    lending.groupby(["prefecture_code", "province", "month", "acq"])
    .size()
    .rename("n")
    .reset_index()
    .merge(schedule, on="province")
    .merge(exposure, on="prefecture_code")
    .dropna(subset=["inspection_round", "exposure_v2_z"])
)
composition = treatment_terms(composition)
composition["y"] = np.arcsinh(composition["n"])
composition["prefA"] = composition["prefecture_code"] + "_" + composition["acq"].astype(str)
composition["monthA"] = composition["month"] + "_" + composition["acq"].astype(str)
for term in ("pth", "ph", "pt", "postc"):
    composition[f"{term}A"] = composition[term] * composition["acq"]

fit(
    "C0", "acquaintance_minus_stranger",
    "y ~ pthA + phA + ptA + postcA + pth + ph + pt | prefA + monthA",
    composition, "pthA", "prefecture_x_group + month_x_group",
    "published composition anchor",
)
fit(
    "C1", "acquaintance_minus_stranger",
    "y ~ pthA + phA + ptA + pth + ph + pt | prefA + prov_month",
    composition, "pthA", "prefecture_x_group + province_x_month",
    "existing province-by-month composition variant",
)
fit(
    "C2", "acquaintance_minus_stranger",
    "y ~ pthA + phA + ptA + pth + ph | prefA + prov_month + monthA",
    composition, "pthA",
    "prefecture_x_group + province_x_month + month_x_group",
    "fully saturated analogue of the reviewer-requested fixed effects",
)


# Placebo-docket validation: preserve the published prefecture-month gap.
civil = pd.read_parquet(f"{DATA}/civil_panel.parquet")
civil["month"] = civil["jmonth"].astype(str).str[:7]
civil = civil[(civil["month"] >= WINDOW[0]) & (civil["month"] <= WINDOW[1])].copy()
civil["group"] = np.where(
    civil["cause_family"].eq("relational"),
    "relational",
    np.where(civil["cause_family"].eq("placebo"), "traffic", "other"),
)
civil = civil[civil["group"].isin(["relational", "traffic"])]
gap = (
    civil.groupby(["prefecture_code", "province", "month", "group"], as_index=False)["n_cases"]
    .sum()
    .pivot_table(
        index=["prefecture_code", "province", "month"],
        columns="group", values="n_cases", fill_value=0,
    )
    .reset_index()
    .merge(schedule, on="province")
    .merge(exposure, on="prefecture_code")
    .dropna(subset=["inspection_round", "exposure_v2_z"])
)
gap = treatment_terms(gap)
gap["y"] = np.arcsinh(gap["relational"]) - np.arcsinh(gap["traffic"])

fit(
    "G0", "relational_minus_traffic_gap",
    "y ~ pth + ph + pt | prefecture_code + month", gap, "pth",
    "prefecture + month", "published placebo-docket validation anchor",
)
fit(
    "G1", "relational_minus_traffic_gap",
    "y ~ pth + ph | prefecture_code + prov_month", gap, "pth",
    "prefecture + province_x_month", "province-by-month placebo-docket validation",
)


os.makedirs(OUT, exist_ok=True)
result = pd.DataFrame(ROWS)
path = f"{OUT}/civil_hierarchy_fe.csv"
result.to_csv(path, index=False)
print(f"written: {path}", flush=True)

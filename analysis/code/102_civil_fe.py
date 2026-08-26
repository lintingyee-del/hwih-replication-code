# -*- coding: utf-8 -*-
"""Auditable fixed-effect sensitivity for the clean-window civil-flow estimand.

This is a bounded Mode-B specification family: the outcome concept, exposure,
clean window, treatment definition, and unit of observation are held fixed.
Only the fixed-effect saturation is changed in J0--J2.  J3--J4 keep the same
cells and fixed effects and change only the count-outcome functional form.

Output: output/ext2124/civil_fe_specs.csv
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


def direction(value):
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "zero"


def record_feols(spec_id, transformation, fixed_effects, formula, data, role):
    crv1 = pf.feols(formula, data=data, vcov={"CRV1": "prov_id"})
    crv3 = pf.feols(formula, data=data, vcov={"CRV3": "prov_id"})
    try:
        p_wild = wild_score_p(formula, data, "pth")
    except Exception as exc:
        print(f"{spec_id}: wild-score failed: {exc}", flush=True)
        p_wild = np.nan
    beta = float(crv1.coef()["pth"])
    row = {
        "spec_id": spec_id,
        "mode": "B_fixed_x_y_equation",
        "focus_side": "outcome_functional_form_only_for_J3_J4",
        "base_variable": "relational_cause_case_count",
        "transformation": transformation,
        "model": "OLS_high_dimensional_FE",
        "sample_rule": "judgment_month_cells; 2017-01_to_2019-03; wave1_vs_not_yet_treated",
        "controls": "post_x_exposure" + ("; post_x_treat" if " + pt " in formula else ""),
        "fixed_effects": fixed_effects,
        "coefficient": beta,
        "std_error": float(crv1.se()["pth"]),
        "p_value": float(crv1.pvalue()["pth"]),
        "std_error_crv3": float(crv3.se()["pth"]),
        "p_crv3": float(crv3.pvalue()["pth"]),
        "p_wild": p_wild,
        "n_obs": int(crv1._N),
        "province_clusters": int(data["prov_id"].nunique()),
        "direction": direction(beta),
        "keep_or_drop": "retain_in_complete_audit_log",
        "reason": role,
    }
    ROWS.append(row)
    print(
        f"{spec_id:3s} {transformation:7s} {fixed_effects:45s} "
        f"b={beta:+.6f} se1={row['std_error']:.6f} p1={row['p_value']:.4f} "
        f"se3={row['std_error_crv3']:.6f} p3={row['p_crv3']:.4f} "
        f"wild={p_wild:.4f} N={row['n_obs']:,}",
        flush=True,
    )


def record_ppml(spec_id, fixed_effects, formula, data, role):
    model = pf.fepois(formula, data=data, vcov={"CRV1": "prov_id"})
    beta = float(model.coef()["pth"])
    row = {
        "spec_id": spec_id,
        "mode": "B_fixed_x_y_equation",
        "focus_side": "outcome_functional_form_only_for_J3_J4",
        "base_variable": "relational_cause_case_count",
        "transformation": "level_count",
        "model": "Poisson_pseudo_maximum_likelihood_high_dimensional_FE",
        "sample_rule": "judgment_month_cells; 2017-01_to_2019-03; wave1_vs_not_yet_treated",
        "controls": "post_x_exposure",
        "fixed_effects": fixed_effects,
        "coefficient": beta,
        "std_error": float(model.se()["pth"]),
        "p_value": float(model.pvalue()["pth"]),
        "std_error_crv3": np.nan,
        "p_crv3": np.nan,
        "p_wild": np.nan,
        "n_obs": int(model._N),
        "province_clusters": int(data["prov_id"].nunique()),
        "direction": direction(beta),
        "keep_or_drop": "retain_in_complete_audit_log",
        "reason": role + "; nonlinear estimator does not use the OLS wild-score routine",
    }
    ROWS.append(row)
    print(
        f"{spec_id:3s} PPML   {fixed_effects:45s} "
        f"b={beta:+.6f} se1={row['std_error']:.6f} p1={row['p_value']:.4f} "
        f"N={row['n_obs']:,}",
        flush=True,
    )


# Freeze the published clean-window judgment-month support.
cells = pd.read_parquet(f"{DATA}/civil_panel.parquet")
cells = cells[cells["cause_family"] == "relational"].copy()
cells["month"] = cells["jmonth"].astype(str).str[:7]
cells = cells[(cells["month"] >= WINDOW[0]) & (cells["month"] <= WINDOW[1])]

schedule = (
    pd.read_parquet(f"{DATA}/panel_month.parquet")[["province", "inspection_round"]]
    .drop_duplicates()
)
cells = cells.merge(schedule, on="province", how="left")
cells = cells.dropna(subset=["exposure_v2_z", "inspection_round", "n_cases"])
cells["treat"] = (cells["inspection_round"] == 1).astype(int)
cells["postc"] = (cells["month"] >= POST0).astype(int)
cells["prov_id"] = pd.factorize(cells["province"])[0]
cells["pref_cause"] = cells["prefecture_code"].astype(str) + "_" + cells["cause"]
cells["month_fe"] = cells["month"]
cells["prov_month"] = cells["province"] + "_" + cells["month"]
cells["cause_month"] = cells["cause"] + "_" + cells["month"]
cells["asinh_n"] = np.arcsinh(cells["n_cases"])
cells["log1p_n"] = np.log1p(cells["n_cases"])
cells["pt"] = cells["postc"] * cells["treat"]
cells["pth"] = cells["pt"] * cells["exposure_v2_z"]
cells["ph"] = cells["postc"] * cells["exposure_v2_z"]

print(
    "PRECOMMITTED FAMILY: J0 published anchor; J1 province-by-month; "
    "J2 province-by-month plus cause-by-month; J3 log(1+y); J4 PPML. "
    "No window, sample, exposure, or treatment changes.",
    flush=True,
)

record_feols(
    "J0",
    "asinh",
    "prefecture_x_cause + month",
    "asinh_n ~ pth + ph + pt | pref_cause + month_fe",
    cells,
    "published anchor; must reproduce Table 3 before interpreting variants",
)
record_feols(
    "J1",
    "asinh",
    "prefecture_x_cause + province_x_month",
    "asinh_n ~ pth + ph | pref_cause + prov_month",
    cells,
    "reviewer-requested province-by-month saturation",
)
record_feols(
    "J2",
    "asinh",
    "prefecture_x_cause + province_x_month + cause_x_month",
    "asinh_n ~ pth + ph | pref_cause + prov_month + cause_month",
    cells,
    "fully requested saturation; candidate primary specification",
)
record_feols(
    "J3",
    "log1p",
    "prefecture_x_cause + province_x_month + cause_x_month",
    "log1p_n ~ pth + ph | pref_cause + prov_month + cause_month",
    cells,
    "admissible count transformation check on the identical cells",
)
record_ppml(
    "J4",
    "prefecture_x_cause + province_x_month + cause_x_month",
    "n_cases ~ pth + ph | pref_cause + prov_month + cause_month",
    cells,
    "admissible count-model check on the identical cells",
)

os.makedirs(OUT, exist_ok=True)
result = pd.DataFrame(ROWS)
path = f"{OUT}/civil_fe_specs.csv"
result.to_csv(path, index=False)
print(f"written: {path}", flush=True)

# -*- coding: utf-8 -*-
"""Reproduce the balanced criminal extensive margin used in the paper.

The outcome is an indicator that a prefecture records at least one enforcement
offense judgment in a month.  The panel restores months with zero cases.  The
script uses 9,999 null-imposed wild-score draws and joins the existing 9,999-draw
wave-permutation audit for the component and joint diagnostics.
"""

from __future__ import annotations

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

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf


CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
DATA = ROOT / "data"
OUT = ROOT / "output"
sys.path.insert(0, str(CODE))
from _wild import wild_score_p  # noqa: E402


WILD_REPS = 9_999
WILD_SEED = 42


def bh_adjust(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranked = values[order] * len(values) / np.arange(1, len(values) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    result = np.empty_like(ranked)
    result[order] = np.minimum(ranked, 1.0)
    return result


def balanced_panel(raw: pd.DataFrame) -> pd.DataFrame:
    meta = raw[
        ["prefecture_code", "province", "exposure_v2_z", "insp_month"]
    ].drop_duplicates("prefecture_code")
    months = pd.DataFrame(
        {
            "jmonth": pd.date_range(
                pd.to_datetime(raw["jmonth"]).min(),
                pd.to_datetime(raw["jmonth"]).max(),
                freq="MS",
            )
        }
    )
    panel = (
        meta.assign(_key=1)
        .merge(months.assign(_key=1), on="_key")
        .drop(columns="_key")
    )
    observed = raw.loc[
        raw["family"].eq("enforcementcrime"),
        ["prefecture_code", "jmonth", "n_cases"],
    ].copy()
    observed["jmonth"] = pd.to_datetime(observed["jmonth"])
    panel = panel.merge(observed, on=["prefecture_code", "jmonth"], how="left")
    panel["n_cases"] = panel["n_cases"].fillna(0.0)
    panel["any_case"] = panel["n_cases"].gt(0).astype(int)
    panel["month"] = panel["jmonth"].dt.to_period("M").astype(str)
    panel["inspection_month"] = (
        pd.to_datetime(panel["insp_month"]).dt.to_period("M").astype(str)
    )
    panel["post"] = panel["month"].ge(panel["inspection_month"]).astype(int)
    panel["exposure"] = panel["exposure_v2_z"]
    panel["post_exposure"] = panel["post"] * panel["exposure"]
    panel["prefecture"] = panel["prefecture_code"].astype(str)
    panel["province_month"] = panel["province"] + "_" + panel["month"]
    panel["province_id"] = pd.factorize(panel["province"])[0]
    return panel


def main() -> None:
    raw = pd.read_parquet(DATA / "crim_panel_v2.parquet")
    panel = balanced_panel(raw)
    formula = "any_case ~ post_exposure | prefecture + province_month"
    crv1 = pf.feols(formula, data=panel, vcov={"CRV1": "province_id"})
    crv3 = pf.feols(formula, data=panel, vcov={"CRV3": "province_id"})
    wild_p = wild_score_p(
        formula,
        panel,
        "post_exposure",
        cluster="province_id",
        reps=WILD_REPS,
        seed=WILD_SEED,
    )

    old = pd.read_csv(OUT / "results_v2.csv").set_index("tag")
    headline_tags = [
        "K2_market_backstop",
        "K2_market_relfail",
        "K2_market_formalization",
        "K2_enforcement_detentiondebt",
    ]
    family_p = np.r_[old.loc[headline_tags, "wild_p"].to_numpy(float), wild_p]
    family_q = bh_adjust(family_p)[-1]

    event = pd.read_csv(OUT / "criminal_eventstudy.csv")
    event = event[event["outcome"].eq("count_complete_any")].iloc[0]
    joint = pd.read_csv(OUT / "criminal_joint_ri.csv")
    joint = joint[joint["spec_id"].eq("J05_baseline_controls_complete_any")]
    free = joint[joint["scheme"].eq("free")].iloc[0]
    stratified = joint[joint["scheme"].eq("stratified_crime_tercile")].iloc[0]

    coefficient = float(crv1.coef()["post_exposure"])
    row = {
        "outcome": "indicator for any recorded enforcement case",
        "coefficient": coefficient,
        "std_error_crv1": float(crv1.se()["post_exposure"]),
        "p_crv1": float(crv1.pvalue()["post_exposure"]),
        "std_error_crv3": float(crv3.se()["post_exposure"]),
        "p_crv3": float(crv3.pvalue()["post_exposure"]),
        "p_wild": wild_p,
        "wild_reps": WILD_REPS,
        "wild_seed": WILD_SEED,
        "bh_q_five_headline_wild_p": float(family_q),
        "pretrend_joint_p_crv1": float(event["pretrend_joint_p"]),
        "pretrend_joint_p_wild": float(event["pretrend_wild_p"]),
        "component_perm_p_free": float(free["p_count"]),
        "component_perm_p_stratified": float(stratified["p_count"]),
        "joint_perm_p_free": float(free["joint_p_equal"]),
        "joint_perm_p_stratified": float(stratified["joint_p_equal"]),
        "n_fit": int(crv1._N),
        "preperiod_mean": float(panel.loc[panel["post"].eq(0), "any_case"].mean()),
    }
    destination = OUT / "criminal_balanced_headline.csv"
    pd.DataFrame([row]).to_csv(destination, index=False)
    print(pd.Series(row).to_string())
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Exploratory national-launch stake-gradient design.

This companion analysis is motivated by the timing diagnostics: the level-count
gradient begins near the January 2018 national campaign launch, before first-wave
central inspections.  It therefore estimates a separate and explicitly weaker
estimand, Post(January 2018) x pre-campaign exposure, in all eligible prefectures.

The output must not be described as the incremental causal effect of inspection.
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

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf
from scipy import stats as sps


CODE = Path(__file__).resolve().parent
ROOT = CODE.parent.parent
OUT = ROOT / "analysis/output"
SEED = 161803


def load_module(name: str, filename: str):
    path = CODE / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


xd = load_module("level_diagnostics", "99b_stake_level_diagnostics.py")
xg = xd.xg

FML = "y ~ ph_mid + ph | pref_group + prov_month + month_group"


def prepare(bin_panel: pd.DataFrame, transformation: str, cutoff: str = "2018-01") -> pd.DataFrame:
    d = bin_panel.copy()
    if transformation == "asinh":
        d["z"] = np.arcsinh(d["n"])
    elif transformation == "log1p":
        d["z"] = np.log1p(d["n"])
    elif transformation == "level":
        d["z"] = d["n"].astype(float)
    elif transformation == "any":
        d["z"] = (d["n"] > 0).astype(float)
    elif transformation == "level_cap99":
        caps = d.loc[d["month"] < "2018-01"].groupby("band", observed=True)["n"].quantile(0.99)
        d["z"] = np.minimum(d["n"], d["band"].map(caps).astype(float))
    else:
        raise ValueError(transformation)
    d["middle"] = d["band"].isin(xg.MIDDLE).astype(int)
    d["postn"] = (d["month"] >= cutoff).astype(int)
    d["ph"] = d["postn"] * d["H"]
    keys = ["prefecture_code", "province", "month", "H", "prov_id", "postn", "ph", "middle"]
    g = d.groupby(keys, as_index=False, observed=True)["z"].mean().rename(columns={"z": "y"})
    g["ph_mid"] = g["ph"] * g["middle"]
    g["pref_group"] = g["prefecture_code"] + "_" + g["middle"].astype(str)
    g["prov_month"] = g["province"] + "_" + g["month"]
    g["month_group"] = g["month"] + "_" + g["middle"].astype(str)
    return g


def fit(spec_id: str, panel: pd.DataFrame, transformation: str, clock: str) -> dict:
    m1 = pf.feols(FML, data=panel, vcov={"CRV1": "prov_id"})
    try:
        m3 = pf.feols(FML, data=panel, vcov={"CRV3": "prov_id"})
        se3 = float(m3.se()["ph_mid"])
        p3 = float(m3.pvalue()["ph_mid"])
    except Exception as exc:
        print(f"[CRV3 failed] {spec_id}: {exc}", flush=True)
        se3 = p3 = np.nan
    try:
        pwild = float(
            xg.wild_score_p(
                FML,
                panel,
                "ph_mid",
                cluster="prov_id",
                reps=xg.REPS_WILD,
                seed=SEED,
            )
        )
    except Exception as exc:
        print(f"[wild failed] {spec_id}: {exc}", flush=True)
        pwild = np.nan
    tail = float(m1.coef()["ph"])
    diff = float(m1.coef()["ph_mid"])
    row = {
        "spec_id": spec_id,
        "clock": clock,
        "transformation": transformation,
        "coefficient": diff,
        "std_error_crv1": float(m1.se()["ph_mid"]),
        "p_crv1": float(m1.pvalue()["ph_mid"]),
        "p_wild": pwild,
        "std_error_crv3": se3,
        "p_crv3": p3,
        "tail_response": tail,
        "middle_response": tail + diff,
        "n_obs": int(m1._N),
        "estimand": "Post(Jan 2018) x exposure x middle-minus-ends; not inspection effect",
    }
    print(
        f"{spec_id:27s} diff={diff:+.4f}; se={row['std_error_crv1']:.4f}, "
        f"p={row['p_crv1']:.3f}; wild={pwild:.3f}; CRV3 p={p3:.3f}",
        flush=True,
    )
    return row


def ppml(panel: pd.DataFrame) -> dict:
    try:
        m = pf.fepois(FML, data=panel, vcov={"CRV1": "prov_id"})
        return {
            "spec_id": "N6_judgment_ppml",
            "clock": "judgment",
            "transformation": "level count",
            "coefficient": float(m.coef()["ph_mid"]),
            "std_error_crv1": float(m.se()["ph_mid"]),
            "p_crv1": float(m.pvalue()["ph_mid"]),
            "p_wild": np.nan,
            "std_error_crv3": np.nan,
            "p_crv3": np.nan,
            "tail_response": float(m.coef()["ph"]),
            "middle_response": float(m.coef()["ph"] + m.coef()["ph_mid"]),
            "n_obs": int(m._N),
            "estimand": "Post(Jan 2018) x exposure x middle-minus-ends; not inspection effect",
        }
    except Exception as exc:
        return {
            "spec_id": "N6_judgment_ppml",
            "clock": "judgment",
            "transformation": "level count",
            "coefficient": np.nan,
            "std_error_crv1": np.nan,
            "p_crv1": np.nan,
            "p_wild": np.nan,
            "std_error_crv3": np.nan,
            "p_crv3": np.nan,
            "tail_response": np.nan,
            "middle_response": np.nan,
            "n_obs": 0,
            "estimand": f"PPML failed: {exc}",
        }


def prelaunch_placebos(bin_panel: pd.DataFrame) -> pd.DataFrame:
    pre = bin_panel[bin_panel["month"] <= "2017-12"].copy()
    rows = []
    for cutoff in ["2017-04", "2017-07", "2017-10"]:
        panel = prepare(pre, "level", cutoff)
        m = pf.feols(FML, data=panel, vcov={"CRV1": "prov_id"})
        rows.append(
            {
                "placebo_cutoff": cutoff,
                "coefficient": float(m.coef()["ph_mid"]),
                "std_error_crv1": float(m.se()["ph_mid"]),
                "p_crv1": float(m.pvalue()["ph_mid"]),
                "sample_end": "2017-12",
            }
        )
    return pd.DataFrame(rows)


def event_study(bin_panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = prepare(bin_panel, "level")
    periods = np.select(
        [
            d["month"].between("2017-01", "2017-06"),
            d["month"].between("2017-07", "2017-09"),
            d["month"].between("2017-10", "2017-12"),
            d["month"].between("2018-01", "2018-04"),
            d["month"].between("2018-05", "2018-08"),
            d["month"].between("2018-09", "2018-12"),
            d["month"].between("2019-01", "2019-03"),
        ],
        ["pre1", "pre2", "ref", "launch1", "launch2", "inspect1", "inspect2"],
        default="outside",
    )
    d["period"] = periods
    names = ["pre1", "pre2", "launch1", "launch2", "inspect1", "inspect2"]
    rhs = []
    for name in names:
        ind = (d["period"] == name).astype(int)
        d[f"h_{name}"] = ind * d["H"]
        d[f"hm_{name}"] = d[f"h_{name}"] * d["middle"]
        rhs.extend([f"h_{name}", f"hm_{name}"])
    formula = "y ~ " + " + ".join(rhs) + " | pref_group + prov_month + month_group"
    m = pf.feols(formula, data=d, vcov={"CRV1": "prov_id"})
    rows = []
    for name in names:
        coef = f"hm_{name}"
        rows.append(
            {
                "period": name,
                "coefficient": float(m.coef()[coef]),
                "std_error_crv1": float(m.se()[coef]),
                "p_crv1": float(m.pvalue()[coef]),
                "reference": "2017-10..2017-12",
            }
        )
    coef_names = list(m.coef().index)
    idx = [coef_names.index("hm_pre1"), coef_names.index("hm_pre2")]
    b = m.coef().values[idx]
    v = m._vcov[np.ix_(idx, idx)]
    stat = float(b @ np.linalg.pinv(v) @ b / len(idx))
    p = float(sps.f.sf(stat, len(idx), 30))
    joint = pd.DataFrame(
        [{"test": "2017 leads jointly zero", "f_stat": stat, "df_num": 2, "df_den": 30, "p_value": p}]
    )
    return pd.DataFrame(rows), joint


def main() -> None:
    print("== exploratory national-launch stake gradient ==", flush=True)
    pref = xg.load_prefecture_frame()
    cases = xg.read_lending_cases()
    judgment = xg.balanced_bin_panel(cases, pref, "judgment_month", "judgment")
    filing_cases = xg.filing_clock(cases)
    filing = xg.balanced_bin_panel(filing_cases, pref, "filing_month", "filing")

    rows = []
    for i, transformation in enumerate(["level", "level_cap99", "asinh", "log1p", "any"], start=1):
        rows.append(fit(f"N{i}_judgment_{transformation}", prepare(judgment, transformation), transformation, "judgment"))
    rows.append(ppml(prepare(judgment, "level")))
    rows.append(fit("N7_filing_level", prepare(filing, "level"), "level", "filing"))
    rows.append(fit("N8_filing_asinh", prepare(filing, "asinh"), "asinh", "filing"))

    placebo = prelaunch_placebos(judgment)
    event, joint = event_study(judgment)
    pd.DataFrame(rows).to_csv(OUT / "stake_national_launch_specs.csv", index=False)
    placebo.to_csv(OUT / "stake_national_launch_placebos.csv", index=False)
    event.to_csv(OUT / "stake_national_launch_event.csv", index=False)
    joint.to_csv(OUT / "stake_national_launch_event_joint.csv", index=False)
    print(placebo.to_string(index=False), flush=True)
    print(event.to_string(index=False), flush=True)
    print(joint.to_string(index=False), flush=True)
    print("Saved exploratory launch-date outputs; no paper file was modified.", flush=True)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Five-band decomposition of the defensible non-strict level specification."""

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


def load_module(name: str, filename: str):
    path = CODE / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


xg = load_module("gradient", "99_stake_gradient.py")


def fit(bin_panel: pd.DataFrame, clock: str, transformation: str) -> tuple[pd.DataFrame, dict]:
    d = bin_panel.copy()
    if transformation == "level":
        d["y"] = d["n"].astype(float)
    elif transformation == "asinh":
        d["y"] = np.arcsinh(d["n"])
    else:
        raise ValueError(transformation)
    d["pref_band"] = d["prefecture_code"] + "_" + d["band"].astype(str)
    d["prov_month"] = d["province"] + "_" + d["month"]
    d["month_band"] = d["month"] + "_" + d["band"].astype(str)
    for band in xg.BANDS:
        hit = (d["band"] == band).astype(int)
        d[f"pth_{band}"] = d["pth"] * hit
        d[f"ph_{band}"] = d["ph"] * hit
        if band != "q1":
            d[f"pt_{band}"] = d["pt"] * hit
    rhs = (
        [f"pth_{b}" for b in xg.BANDS]
        + [f"ph_{b}" for b in xg.BANDS]
        + [f"pt_{b}" for b in xg.BANDS if b != "q1"]
    )
    formula = "y ~ " + " + ".join(rhs) + " | pref_band + prov_month + month_band"
    m1 = pf.feols(formula, data=d, vcov={"CRV1": "prov_id"})
    m3 = pf.feols(formula, data=d, vcov={"CRV3": "prov_id"})
    names = list(m1.coef().index)
    w = np.zeros(len(names))
    weights = {"q1": -1 / 3, "q2": 1 / 2, "q3": 1 / 2, "q4": -1 / 3, "q5": -1 / 3}
    for band, weight in weights.items():
        w[names.index(f"pth_{band}")] = weight
    contrast = float(w @ m1.coef().values)
    se1 = float(np.sqrt(w @ m1._vcov @ w))
    se3 = float(np.sqrt(w @ m3._vcov @ w))
    p1 = float(2 * sps.t.sf(abs(contrast / se1), 30))
    p3 = float(2 * sps.t.sf(abs(contrast / se3), 30))
    bands = []
    for band in xg.BANDS:
        bands.append(
            {
                "clock": clock,
                "transformation": transformation,
                "band": band,
                "coefficient": float(m1.coef()[f"pth_{band}"]),
                "std_error_crv1": float(m1.se()[f"pth_{band}"]),
                "p_crv1": float(m1.pvalue()[f"pth_{band}"]),
                "n_obs": int(m1._N),
            }
        )
    summary = {
        "clock": clock,
        "transformation": transformation,
        "middle_minus_ends": contrast,
        "std_error_crv1": se1,
        "p_crv1_t30": p1,
        "std_error_crv3": se3,
        "p_crv3_t30": p3,
        "n_obs": int(m1._N),
        "fixed_effects": "prefecture x band; province x month; month x band",
    }
    print(pd.DataFrame(bands).to_string(index=False), flush=True)
    print(summary, flush=True)
    return pd.DataFrame(bands), summary


def main() -> None:
    pref = xg.load_prefecture_frame()
    cases = xg.read_lending_cases()
    judgment = xg.balanced_bin_panel(cases, pref, "judgment_month", "judgment")
    filing = xg.balanced_bin_panel(xg.filing_clock(cases), pref, "filing_month", "filing")
    band_frames = []
    summaries = []
    for clock, panel in [("judgment", judgment), ("filing", filing)]:
        for transformation in ["level", "asinh"]:
            bands, summary = fit(panel, clock, transformation)
            band_frames.append(bands)
            summaries.append(summary)
    pd.concat(band_frames, ignore_index=True).to_csv(OUT / "stake_five_band_coefficients.csv", index=False)
    pd.DataFrame(summaries).to_csv(OUT / "stake_five_band_contrasts.csv", index=False)
    print("Saved five-band outputs; no paper file was modified.", flush=True)


if __name__ == "__main__":
    main()

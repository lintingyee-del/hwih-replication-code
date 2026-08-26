# -*- coding: utf-8 -*-
"""Identification checks for the positive stake-gradient level contrast.

The treatment, window, sample, and original amount bins remain fixed.  This script
adds the province-by-month structure used by the earlier stake test, runs the same
checks under filing dates, and reports a complete set of pre-treatment timing tests.
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
SEED = 271828


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

FML_PM = (
    "y ~ pth_mid + ph_mid + pt_mid + pth + ph + pt "
    "| pref_group + prov_month"
)
FML_PM_MG = (
    "y ~ pth_mid + ph_mid + pt_mid + pth + ph "
    "| pref_group + prov_month + month_group"
)


def add_fe_ids(panel: pd.DataFrame) -> pd.DataFrame:
    d = panel.copy()
    d["prov_month"] = d["province"] + "_" + d["month"]
    return d


def fit(spec_id: str, panel: pd.DataFrame, formula: str, note: str) -> dict:
    m1 = pf.feols(formula, data=panel, vcov={"CRV1": "prov_id"})
    try:
        m3 = pf.feols(formula, data=panel, vcov={"CRV3": "prov_id"})
        se3 = float(m3.se()["pth_mid"])
        p3 = float(m3.pvalue()["pth_mid"])
    except Exception as exc:
        print(f"[CRV3 failed] {spec_id}: {exc}", flush=True)
        se3 = p3 = np.nan
    try:
        pwild = float(
            xg.wild_score_p(
                formula,
                panel,
                "pth_mid",
                cluster="prov_id",
                reps=xg.REPS_WILD,
                seed=SEED,
            )
        )
    except Exception as exc:
        print(f"[wild failed] {spec_id}: {exc}", flush=True)
        pwild = np.nan
    tail = float(m1.coef()["pth"])
    diff = float(m1.coef()["pth_mid"])
    row = {
        "spec_id": spec_id,
        "coefficient": diff,
        "std_error_crv1": float(m1.se()["pth_mid"]),
        "p_crv1": float(m1.pvalue()["pth_mid"]),
        "p_wild": pwild,
        "std_error_crv3": se3,
        "p_crv3": p3,
        "tail_response": tail,
        "middle_response": tail + diff,
        "n_obs": int(m1._N),
        "fixed_effects": formula.split("|")[-1].strip(),
        "note": note,
    }
    print(
        f"{spec_id:31s} diff={diff:+.4f}; CRV1 se={row['std_error_crv1']:.4f}, "
        f"p={row['p_crv1']:.3f}; wild={pwild:.3f}; CRV3 p={p3:.3f}",
        flush=True,
    )
    return row


def reset_post(bin_panel: pd.DataFrame, cutoff: str) -> pd.DataFrame:
    d = bin_panel.copy()
    d["postc"] = (d["month"] >= cutoff).astype(int)
    d["pt"] = d["postc"] * d["treat"]
    d["pth"] = d["pt"] * d["H"]
    d["ph"] = d["postc"] * d["H"]
    return d


def placebo_cutoffs(judgment: pd.DataFrame) -> pd.DataFrame:
    # The first actual inspection (Hebei pilot) arrives in July 2018.  June is
    # therefore the last uncontaminated month for a false-timing exercise.
    pre = judgment[judgment["month"] <= "2018-06"].copy()
    rows = []
    for cutoff in ["2017-07", "2017-10", "2018-01", "2018-04"]:
        d = reset_post(pre, cutoff)
        panel = add_fe_ids(xd.level_group_panel(d))
        m = pf.feols(FML_PM_MG, data=panel, vcov={"CRV1": "prov_id"})
        rows.append(
            {
                "placebo_cutoff": cutoff,
                "coefficient": float(m.coef()["pth_mid"]),
                "std_error_crv1": float(m.se()["pth_mid"]),
                "p_crv1": float(m.pvalue()["pth_mid"]),
                "n_obs": int(m._N),
                "sample_end": "2018-06",
            }
        )
        print(
            f"placebo {cutoff}: {rows[-1]['coefficient']:+.4f} "
            f"({rows[-1]['std_error_crv1']:.4f}), p={rows[-1]['p_crv1']:.3f}",
            flush=True,
        )
    return pd.DataFrame(rows)


def binned_event_study(level: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = add_fe_ids(level)
    periods = np.select(
        [
            d["month"].between("2017-01", "2017-06"),
            d["month"].between("2017-07", "2017-12"),
            d["month"].between("2018-01", "2018-03"),
            d["month"].between("2018-04", "2018-06"),
            d["month"].between("2018-07", "2018-09"),
            d["month"].between("2018-10", "2018-12"),
            d["month"].between("2019-01", "2019-03"),
        ],
        ["pre1", "pre2", "pre3", "ref", "arrival", "post1", "post2"],
        default="outside",
    )
    d["period"] = periods
    names = ["pre1", "pre2", "pre3", "arrival", "post1", "post2"]
    rhs = []
    for name in names:
        ind = (d["period"] == name).astype(int)
        d[f"h_{name}"] = ind * d["H"]
        d[f"hm_{name}"] = d[f"h_{name}"] * d["middle"]
        d[f"tm_{name}"] = ind * d["treat"] * d["middle"]
        d[f"th_{name}"] = ind * d["treat"] * d["H"]
        d[f"thm_{name}"] = d[f"th_{name}"] * d["middle"]
        rhs.extend([f"h_{name}", f"hm_{name}", f"tm_{name}", f"th_{name}", f"thm_{name}"])
    formula = "y ~ " + " + ".join(rhs) + " | pref_group + prov_month + month_group"
    m = pf.feols(formula, data=d, vcov={"CRV1": "prov_id"})
    out = []
    for name in names:
        coef = f"thm_{name}"
        out.append(
            {
                "period": name,
                "coefficient": float(m.coef()[coef]),
                "std_error_crv1": float(m.se()[coef]),
                "p_crv1": float(m.pvalue()[coef]),
                "reference_period": "2018-04..2018-06",
            }
        )
    out_df = pd.DataFrame(out)

    coef_names = list(m.coef().index)
    pre_names = [f"thm_pre{i}" for i in [1, 2, 3]]
    idx = [coef_names.index(name) for name in pre_names]
    b = m.coef().values[idx]
    v = m._vcov[np.ix_(idx, idx)]
    q = len(idx)
    stat = float(b @ np.linalg.pinv(v) @ b / q)
    p_joint = float(sps.f.sf(stat, q, 30))
    joint = pd.DataFrame(
        [{"test": "pre-period coefficients jointly zero", "f_stat": stat, "df_num": q, "df_den": 30, "p_value": p_joint}]
    )
    print(out_df.to_string(index=False), flush=True)
    print(f"binned-event pretrend F({q},30)={stat:.3f}, p={p_joint:.3f}", flush=True)
    return out_df, joint


def per_band(bin_panel: pd.DataFrame, clock: str, transform: str) -> pd.DataFrame:
    rows = []
    for band in xg.BANDS:
        d = bin_panel[bin_panel["band"] == band].copy()
        d["y"] = d["n"].astype(float) if transform == "level" else np.arcsinh(d["n"])
        d["pref"] = d["prefecture_code"]
        d["prov_month"] = d["province"] + "_" + d["month"]
        formula = "y ~ pth + ph | pref + prov_month"
        m = pf.feols(formula, data=d, vcov={"CRV1": "prov_id"})
        rows.append(
            {
                "clock": clock,
                "transform": transform,
                "band": band,
                "coefficient": float(m.coef()["pth"]),
                "std_error_crv1": float(m.se()["pth"]),
                "p_crv1": float(m.pvalue()["pth"]),
                "n_obs": int(m._N),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    print("== province-month and timing checks for the level result ==", flush=True)
    pref = xg.load_prefecture_frame()
    cases = xg.read_lending_cases()
    judgment = xg.balanced_bin_panel(cases, pref, "judgment_month", "judgment")
    jlevel = add_fe_ids(xd.level_group_panel(judgment))
    jlevel["clock"] = "judgment"

    filing_cases = xg.filing_clock(cases)
    filing = xg.balanced_bin_panel(filing_cases, pref, "filing_month", "filing")
    flevel = add_fe_ids(xd.level_group_panel(filing))
    flevel["clock"] = "filing"

    results = [
        fit(
            "I1_judgment_province_month",
            jlevel,
            FML_PM,
            "Fixed effects used by the earlier clean-window middle-versus-tail test",
        ),
        fit(
            "I2_judgment_pm_plus_month_group",
            jlevel,
            FML_PM_MG,
            "Adds nationwide amount-group-specific month shocks",
        ),
        fit(
            "I3_filing_province_month",
            flevel,
            FML_PM,
            "Filing clock with earlier clean-window fixed effects",
        ),
        fit(
            "I4_filing_pm_plus_month_group",
            flevel,
            FML_PM_MG,
            "Filing clock with province-month and amount-group month effects",
        ),
    ]

    for q, tag in [(0.99, "I5_judgment_cap_p99_pm"), (0.995, "I6_judgment_cap_p995_pm")]:
        capped, caps = xd.cap_from_preperiod(judgment, q)
        panel = add_fe_ids(xd.level_group_panel(capped, "n_capped"))
        results.append(fit(tag, panel, FML_PM_MG, f"Band-specific pre-period cap: {caps}"))

    placebo = placebo_cutoffs(judgment)
    event, joint = binned_event_study(jlevel)
    bands = pd.concat(
        [
            per_band(judgment, "judgment", "level"),
            per_band(judgment, "judgment", "asinh"),
            per_band(filing, "filing", "level"),
            per_band(filing, "filing", "asinh"),
        ],
        ignore_index=True,
    )

    pd.DataFrame(results).to_csv(OUT / "stake_identification_specs.csv", index=False)
    placebo.to_csv(OUT / "stake_level_placebos.csv", index=False)
    event.to_csv(OUT / "stake_level_event_study.csv", index=False)
    joint.to_csv(OUT / "stake_level_event_joint.csv", index=False)
    bands.to_csv(OUT / "stake_identification_per_band.csv", index=False)
    print("Saved identification outputs; no paper file was modified.", flush=True)


if __name__ == "__main__":
    main()

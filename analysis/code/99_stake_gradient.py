# -*- coding: utf-8 -*-
"""Auditable stake-gradient specification search for Proposition 3.

The research question, treatment, sample window, and original claim-size bins are fixed.
This script searches only admissible representations of the count outcome and compatible
estimators. It never patches the paper or existing result files.

Primary design
--------------
* Judgment clock, 2017-01 through 2019-03.
* First-wave provinces versus not-yet-treated provinces, post beginning 2018-09.
* Exposure is exposure_v2_z.
* Original five claim-size bins: <20k, 20--50k, 50--200k, 200k--1m, >1m yuan.
* A balanced prefecture x month x bin panel, including genuine zero-count cells.
* The shape estimand is the average response in the two middle bins minus the average
  response in the three end bins.

Outputs
-------
output/stake_spec_log.csv
output/stake_per_band.csv
output/stake_permutation.csv
output/stake_diagnostics.csv
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

import io
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf
from scipy import stats as sps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _wild import wild_score_p

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "analysis/data"
OUT = ROOT / "analysis/output"

WINDOW = ("2017-01", "2019-03")
POST0 = "2018-09"
AMOUNT_MIN = 100.0
AMOUNT_MAX = 5e8
REPS_WILD = 9_999
REPS_PERM = 999
SEED = 42

BANDS = ["q1", "q2", "q3", "q4", "q5"]
MIDDLE = {"q2", "q3"}
TAILS = {"q1", "q4", "q5"}

MODE = "direct_experiment"
FOCUS = "y"
BASE_VARIABLE = "private-lending case count by original claim-size bin"
TREATMENT = "Post x first-wave x exposure_v2_z"

rows: list[dict] = []
band_rows: list[dict] = []
diagnostics: list[dict] = []


def claim_bin(amount: pd.Series) -> pd.Series:
    return pd.cut(
        amount,
        bins=[AMOUNT_MIN, 20_000, 50_000, 200_000, 1_000_000, AMOUNT_MAX],
        labels=BANDS,
        right=False,
        include_lowest=True,
    ).astype("object")


def load_prefecture_frame() -> pd.DataFrame:
    exposure = pd.read_parquet(
        DATA / "exposure_v2.parquet",
        columns=["prefecture_code", "province", "exposure_v2_z"],
    )
    schedule = (
        pd.read_parquet(DATA / "panel_month.parquet", columns=["province", "inspection_round"])
        .drop_duplicates()
    )
    pref = (
        exposure.merge(schedule, on="province", how="inner")
        .dropna(subset=["exposure_v2_z", "inspection_round"])
        .drop_duplicates("prefecture_code")
    )
    pref["prefecture_code"] = pref["prefecture_code"].astype(str)
    return pref


def read_lending_cases() -> pd.DataFrame:
    cases = pd.read_parquet(
        DATA / "civil_case.parquet",
        columns=["case_no", "cause", "prefecture_code", "province", "jmonth", "amount_yuan"],
    )
    cases = cases[cases["cause"] == "民间借贷纠纷"].copy()
    cases["prefecture_code"] = cases["prefecture_code"].astype(str)
    cases["jmonth"] = pd.to_datetime(cases["jmonth"], errors="coerce")
    cases["judgment_month"] = cases["jmonth"].dt.strftime("%Y-%m")
    return cases


def filing_clock(cases: pd.DataFrame) -> pd.DataFrame:
    filing = pd.read_parquet(DATA / "civil_filing.parquet")
    case_col = filing.columns[0]
    filing = filing.rename(columns={case_col: "case_no"})[["case_no", "filing_ymd"]]
    filing["filing_date"] = pd.to_datetime(filing["filing_ymd"], errors="coerce")
    d = cases.merge(filing[["case_no", "filing_date"]], on="case_no", how="left")
    d["duration"] = (d["jmonth"] - d["filing_date"]).dt.days
    d = d[d["filing_date"].notna() & d["duration"].between(0, 270)].copy()
    d["filing_month"] = d["filing_date"].dt.strftime("%Y-%m")
    return d


def balanced_bin_panel(
    cases: pd.DataFrame,
    pref: pd.DataFrame,
    month_col: str,
    clock: str,
) -> pd.DataFrame:
    d = cases[cases[month_col].between(WINDOW[0], WINDOW[1])].copy()
    valid = d["amount_yuan"].between(AMOUNT_MIN, AMOUNT_MAX, inclusive="left")
    v = d[valid].copy()
    v["band"] = claim_bin(v["amount_yuan"])
    v = v[v["band"].notna()]
    counts = (
        v.groupby(["prefecture_code", month_col, "band"], observed=True)
        .size()
        .rename("n")
        .reset_index()
        .rename(columns={month_col: "month"})
    )

    months = pd.DataFrame({"month": pd.period_range(WINDOW[0], WINDOW[1], freq="M").astype(str)})
    bands = pd.DataFrame({"band": BANDS})
    grid = pref.assign(_k=1).merge(months.assign(_k=1), on="_k").merge(
        bands.assign(_k=1), on="_k"
    ).drop(columns="_k")
    grid = grid.merge(counts, on=["prefecture_code", "month", "band"], how="left")
    grid["n"] = grid["n"].fillna(0).astype(int)
    grid["clock"] = clock
    grid["H"] = grid["exposure_v2_z"]
    grid["treat"] = (grid["inspection_round"] == 1).astype(int)
    grid["postc"] = (grid["month"] >= POST0).astype(int)
    # Keep cluster codes stable when the parquet reader returns rows in a
    # different order (as can happen after the public-data rewrite).
    grid["prov_id"] = pd.factorize(grid["province"], sort=True)[0]
    grid["pt"] = grid["postc"] * grid["treat"]
    grid["pth"] = grid["pt"] * grid["H"]
    grid["ph"] = grid["postc"] * grid["H"]

    diagnostics.append(
        {
            "diagnostic": f"{clock}_balanced_panel",
            "value": len(grid),
            "detail": f"{grid.prefecture_code.nunique()} prefectures, {grid.month.nunique()} months, 5 bins",
        }
    )
    diagnostics.append(
        {
            "diagnostic": f"{clock}_zero_cell_share",
            "value": float((grid["n"] == 0).mean()),
            "detail": "share of prefecture-month-bin cells with zero valid-amount lending cases",
        }
    )
    diagnostics.append(
        {
            "diagnostic": f"{clock}_valid_amount_cases",
            "value": int(v.shape[0]),
            "detail": "cases in fixed amount support and clean window",
        }
    )
    return grid


def group_average_panel(bin_panel: pd.DataFrame, transformation: str) -> pd.DataFrame:
    d = bin_panel.copy()
    if transformation == "asinh":
        d["z"] = np.arcsinh(d["n"])
    elif transformation == "log1p":
        d["z"] = np.log1p(d["n"])
    elif transformation == "level":
        d["z"] = d["n"].astype(float)
    elif transformation == "any":
        d["z"] = (d["n"] > 0).astype(float)
    else:
        raise ValueError(transformation)

    d["middle"] = d["band"].isin(MIDDLE).astype(int)
    keys = [
        "prefecture_code", "province", "month", "inspection_round", "H", "treat",
        "postc", "prov_id", "pt", "pth", "ph", "middle",
    ]
    g = d.groupby(keys, as_index=False, observed=True)["z"].mean().rename(columns={"z": "y"})
    g["pref_group"] = g["prefecture_code"] + "_" + g["middle"].astype(str)
    g["month_group"] = g["month"] + "_" + g["middle"].astype(str)
    g["prov_month_group"] = (
        g["province"] + "_" + g["month"] + "_" + g["middle"].astype(str)
    )
    for name in ["pth", "ph", "pt"]:
        g[f"{name}_mid"] = g[name] * g["middle"]
    return g


FML_BASE = (
    "y ~ pth_mid + ph_mid + pt_mid + pth + ph + pt "
    "| pref_group + month_group"
)
FML_STRICT = (
    "y ~ pth_mid + ph_mid + pth + ph "
    "| pref_group + prov_month_group"
)


def add_row(
    *,
    spec_id: str,
    transformation: str,
    model: str,
    sample_rule: str,
    fixed_effects: str,
    estimate: float,
    se: float,
    p_crv1: float,
    n_obs: int,
    p_wild: float = np.nan,
    se_crv3: float = np.nan,
    p_crv3: float = np.nan,
    p_permutation: float = np.nan,
    tail_estimate: float = np.nan,
    mid_estimate: float = np.nan,
    keep_or_drop: str = "evaluate",
    reason: str = "",
    source: str = "99_stake_gradient.py",
) -> None:
    rows.append(
        {
            "spec_id": spec_id,
            "mode": MODE,
            "focus_side": FOCUS,
            "base_variable": BASE_VARIABLE,
            "treatment": TREATMENT,
            "transformation": transformation,
            "model": model,
            "sample_rule": sample_rule,
            "controls": "lower-order Post x Treat, Post x H, and their middle-bin interactions",
            "fixed_effects": fixed_effects,
            "coefficient": estimate,
            "std_error": se,
            "p_value": p_crv1,
            "p_wild": p_wild,
            "se_crv3": se_crv3,
            "p_crv3": p_crv3,
            "p_permutation": p_permutation,
            "n_obs": n_obs,
            "tail_estimate": tail_estimate,
            "mid_estimate": mid_estimate,
            "direction": "positive" if estimate > 0 else "negative" if estimate < 0 else "zero",
            "keep_or_drop": keep_or_drop,
            "reason": reason,
            "source": source,
        }
    )


def linear_shape(
    spec_id: str,
    panel: pd.DataFrame,
    transformation: str,
    fml: str = FML_BASE,
    strict: bool = False,
    weights: str | None = None,
) -> tuple[pf.estimation.feols_.Feols, pd.DataFrame]:
    m1 = pf.feols(fml, data=panel, vcov={"CRV1": "prov_id"}, weights=weights)
    try:
        m3 = pf.feols(fml, data=panel, vcov={"CRV3": "prov_id"}, weights=weights)
        se3 = float(m3.se()["pth_mid"])
        p3 = float(m3.pvalue()["pth_mid"])
    except Exception:
        se3 = p3 = np.nan
    try:
        wp = wild_score_p(
            fml, panel, "pth_mid", weights=weights, cluster="prov_id",
            reps=REPS_WILD, seed=SEED,
        )
    except Exception as exc:
        print(f"[wild failed] {spec_id}: {exc}", flush=True)
        wp = np.nan

    names = list(m1.coef().index)
    b = m1.coef()
    V = m1._vcov
    i_tail = names.index("pth")
    i_diff = names.index("pth_mid")
    tail = float(b["pth"])
    diff = float(b["pth_mid"])
    mid = tail + diff
    se_mid = float(np.sqrt(V[i_tail, i_tail] + V[i_diff, i_diff] + 2 * V[i_tail, i_diff]))
    p_mid = float(2 * sps.t.sf(abs(mid / se_mid), 30))

    add_row(
        spec_id=spec_id,
        transformation=transformation,
        model="FE-OLS" if not strict else "FE-OLS, province-month-group saturation",
        sample_rule=(
            f"{panel['clock'].iloc[0] if 'clock' in panel else 'judgment'} clock; "
            "2017-01..2019-03; first wave vs not-yet-treated; balanced zeros"
        ),
        fixed_effects=(
            "prefecture x middle-group; month x middle-group"
            if not strict else "prefecture x middle-group; province x month x middle-group"
        ),
        estimate=diff,
        se=float(m1.se()["pth_mid"]),
        p_crv1=float(m1.pvalue()["pth_mid"]),
        p_wild=wp,
        se_crv3=se3,
        p_crv3=p3,
        n_obs=int(m1._N),
        tail_estimate=tail,
        mid_estimate=mid,
        reason=(
            "Primary harmonized contrast" if spec_id == "J1_balanced_asinh" else
            "Admissible outcome/model variant around the fixed design"
        ),
    )
    print(
        f"{spec_id:28s} diff={diff:+.4f} (CRV1 {m1.se()['pth_mid']:.4f}, "
        f"p={m1.pvalue()['pth_mid']:.3f}; wild={wp:.3f}; CRV3 p={p3:.3f}) "
        f"tail={tail:+.4f} mid={mid:+.4f} (mid p={p_mid:.3f}) N={int(m1._N):,}",
        flush=True,
    )
    return m1, panel


def ppml_shape(spec_id: str, panel: pd.DataFrame) -> None:
    try:
        m = pf.fepois(FML_BASE, data=panel, vcov={"CRV1": "prov_id"})
        diff = float(m.coef()["pth_mid"])
        add_row(
            spec_id=spec_id,
            transformation="level count averaged within middle/end bins",
            model="Poisson pseudo-maximum-likelihood",
            sample_rule="judgment clock; fixed clean window; balanced zeros",
            fixed_effects="prefecture x middle-group; month x middle-group",
            estimate=diff,
            se=float(m.se()["pth_mid"]),
            p_crv1=float(m.pvalue()["pth_mid"]),
            n_obs=int(m._N),
            tail_estimate=float(m.coef()["pth"]),
            mid_estimate=float(m.coef()["pth"] + diff),
            reason="Count-model robustness; coefficient is a differential semi-elasticity",
        )
        print(
            f"{spec_id:28s} diff={diff:+.4f} (se {m.se()['pth_mid']:.4f}, "
            f"p={m.pvalue()['pth_mid']:.3f}) N={int(m._N):,}", flush=True,
        )
    except Exception as exc:
        print(f"[PPML failed] {spec_id}: {exc}", flush=True)
        add_row(
            spec_id=spec_id,
            transformation="level count averaged within middle/end bins",
            model="Poisson pseudo-maximum-likelihood",
            sample_rule="judgment clock; fixed clean window; balanced zeros",
            fixed_effects="prefecture x middle-group; month x middle-group",
            estimate=np.nan,
            se=np.nan,
            p_crv1=np.nan,
            n_obs=0,
            keep_or_drop="drop",
            reason=f"Estimator failed: {exc}",
        )


def per_band_estimates(bin_panel: pd.DataFrame, clock: str) -> None:
    for band in BANDS:
        d = bin_panel[bin_panel["band"] == band].copy()
        d["y"] = np.arcsinh(d["n"])
        d["pref"] = d["prefecture_code"]
        m = pf.feols(
            "y ~ pth + ph + pt | pref + month",
            data=d,
            vcov={"CRV1": "prov_id"},
        )
        wp = wild_score_p(
            "y ~ pth + ph + pt | pref + month", d, "pth",
            cluster="prov_id", reps=REPS_WILD, seed=SEED,
        )
        band_rows.append(
            {
                "clock": clock,
                "band": band,
                "estimate": float(m.coef()["pth"]),
                "se_crv1": float(m.se()["pth"]),
                "p_crv1": float(m.pvalue()["pth"]),
                "p_wild": wp,
                "n_obs": int(m._N),
            }
        )


def share_spec(bin_panel: pd.DataFrame, weighted: bool) -> None:
    d = bin_panel.copy()
    d["mid_n"] = d["n"] * d["band"].isin(MIDDLE).astype(int)
    keys = ["prefecture_code", "province", "month", "H", "treat", "postc", "prov_id", "pt", "pth", "ph"]
    g = d.groupby(keys, as_index=False)[["n", "mid_n"]].sum()
    g = g[g["n"] > 0].copy()
    g["y"] = g["mid_n"] / g["n"]
    g["pref"] = g["prefecture_code"]
    fml = "y ~ pth + ph + pt | pref + month"
    w = "n" if weighted else None
    m1 = pf.feols(fml, data=g, vcov={"CRV1": "prov_id"}, weights=w)
    try:
        m3 = pf.feols(fml, data=g, vcov={"CRV3": "prov_id"}, weights=w)
        se3 = float(m3.se()["pth"]); p3 = float(m3.pvalue()["pth"])
    except Exception:
        se3 = p3 = np.nan
    wp = wild_score_p(
        fml, g, "pth", weights=w, cluster="prov_id", reps=REPS_WILD, seed=SEED,
    )
    tag = "J7_mid_share_case_weighted" if weighted else "J6_mid_share_unweighted"
    add_row(
        spec_id=tag,
        transformation="middle-bin share among valid-amount lending cases",
        model="FE-OLS weighted by valid cases" if weighted else "FE-OLS",
        sample_rule="judgment clock; fixed clean window; prefecture-months with >=1 valid-amount case",
        fixed_effects="prefecture; month",
        estimate=float(m1.coef()["pth"]),
        se=float(m1.se()["pth"]),
        p_crv1=float(m1.pvalue()["pth"]),
        p_wild=wp,
        se_crv3=se3,
        p_crv3=p3,
        n_obs=int(m1._N),
        reason="Composition expression of the same original amount bins",
    )
    print(
        f"{tag:28s} b={m1.coef()['pth']:+.4f} (se {m1.se()['pth']:.4f}, "
        f"p={m1.pvalue()['pth']:.3f}; wild={wp:.3f}; CRV3 p={p3:.3f}) N={int(m1._N):,}",
        flush=True,
    )


def coverage_spec(cases: pd.DataFrame, pref: pd.DataFrame) -> None:
    d = cases[cases["judgment_month"].between(WINDOW[0], WINDOW[1])].copy()
    d["valid"] = d["amount_yuan"].between(AMOUNT_MIN, AMOUNT_MAX, inclusive="left").astype(int)
    g = (
        d.groupby(["prefecture_code", "judgment_month"])
        .agg(total=("case_no", "size"), valid=("valid", "sum"))
        .reset_index()
        .rename(columns={"judgment_month": "month"})
    )
    months = pd.DataFrame({"month": pd.period_range(WINDOW[0], WINDOW[1], freq="M").astype(str)})
    grid = pref.assign(_k=1).merge(months.assign(_k=1), on="_k").drop(columns="_k")
    g = grid.merge(g, on=["prefecture_code", "month"], how="left")
    g[["total", "valid"]] = g[["total", "valid"]].fillna(0)
    g = g[g["total"] > 0].copy()
    g["y"] = g["valid"] / g["total"]
    g["H"] = g["exposure_v2_z"]
    g["treat"] = (g["inspection_round"] == 1).astype(int)
    g["postc"] = (g["month"] >= POST0).astype(int)
    g["pt"] = g["postc"] * g["treat"]
    g["pth"] = g["pt"] * g["H"]
    g["ph"] = g["postc"] * g["H"]
    g["pref"] = g["prefecture_code"]
    g["prov_id"] = pd.factorize(g["province"], sort=True)[0]
    fml = "y ~ pth + ph + pt | pref + month"
    m = pf.feols(fml, data=g, vcov={"CRV1": "prov_id"}, weights="total")
    wp = wild_score_p(
        fml, g, "pth", weights="total", cluster="prov_id", reps=REPS_WILD, seed=SEED,
    )
    diagnostics.append(
        {
            "diagnostic": "amount_extraction_coverage_effect",
            "value": float(m.coef()["pth"]),
            "detail": f"se={m.se()['pth']:.4f}; CRV1 p={m.pvalue()['pth']:.3f}; wild p={wp:.3f}; N={int(m._N)}",
        }
    )
    diagnostics.append(
        {
            "diagnostic": "amount_extraction_rate",
            "value": float(d["valid"].mean()),
            "detail": f"{int(d.valid.sum()):,} valid of {len(d):,} lending cases in clean window",
        }
    )
    print(
        f"coverage effect: {m.coef()['pth']:+.4f} (se {m.se()['pth']:.4f}, "
        f"p={m.pvalue()['pth']:.3f}; wild={wp:.3f}); raw coverage={d.valid.mean():.3f}",
        flush=True,
    )


def wave_permutation(
    spec_id: str,
    panel: pd.DataFrame,
    fml: str,
    coef: str = "pth_mid",
) -> pd.DataFrame:
    observed_map = panel[["province", "treat"]].drop_duplicates().set_index("province")["treat"].to_dict()
    provinces = np.array(sorted(observed_map))
    treated_n = int(sum(observed_map.values()))

    def add_terms(data: pd.DataFrame, treatment_map: dict[str, int]) -> pd.DataFrame:
        d = data.copy()
        d["treat"] = d["province"].map(treatment_map).astype(int)
        d["pt"] = d["postc"] * d["treat"]
        d["pth"] = d["pt"] * d["H"]
        d["ph"] = d["postc"] * d["H"]
        for name in ["pth", "ph", "pt"]:
            d[f"{name}_mid"] = d[name] * d["middle"]
        return d

    observed = add_terms(panel, observed_map)
    beta = float(pf.feols(fml, data=observed, vcov="iid").coef()[coef])
    rng = np.random.default_rng(SEED)
    draws = np.empty(REPS_PERM)
    for r in range(REPS_PERM):
        labels = np.zeros(len(provinces), dtype=int)
        labels[rng.choice(len(provinces), treated_n, replace=False)] = 1
        d = add_terms(panel, dict(zip(provinces, labels)))
        draws[r] = float(pf.feols(fml, data=d, vcov="iid").coef()[coef])
        if (r + 1) % 100 == 0:
            print(f"{spec_id} wave permutations {r + 1}/{REPS_PERM}", flush=True)
    p = float((1 + np.sum(np.abs(draws) >= abs(beta))) / (REPS_PERM + 1))
    for row in rows:
        if row["spec_id"] == spec_id:
            row["p_permutation"] = p
    print(f"{spec_id} wave permutation: beta={beta:+.4f}, p={p:.3f}", flush=True)
    return pd.DataFrame({"spec_id": spec_id, "draw": np.arange(REPS_PERM), "estimate": draws})


def append_legacy_results() -> None:
    legacy_clean = pd.read_csv(OUT / "stake_midband.csv").iloc[0]
    add_row(
        spec_id="L1_legacy_clean_pooled_positive_cells",
        transformation="asinh of pooled middle versus pooled tails",
        model="FE-OLS",
        sample_rule="clean window; positive cells only; middle and tails pooled before transform",
        fixed_effects="prefecture x group; province x month",
        estimate=float(legacy_clean["coef"]),
        se=float(legacy_clean["se"]),
        p_crv1=float(legacy_clean["p"]),
        p_wild=float(legacy_clean["wild_p"]),
        n_obs=0,
        keep_or_drop="drop",
        reason="Provenance only: omits zero cells and pools unequal numbers of bins",
        source="35_stake_midband.py / output/stake_midband.csv",
    )
    hierarchy = pd.read_csv(OUT / "hierarchy_numbers.csv")
    h = hierarchy[hierarchy["tag"] == "stake_mid_vs_ends"].iloc[0]
    add_row(
        spec_id="L2_legacy_full_period_five_bin",
        transformation="asinh count, average middle minus average ends",
        model="FE-OLS",
        sample_rule="full 2014--2020 staggered sample; positive cells only",
        fixed_effects="prefecture x bin; province x month x bin",
        estimate=float(h["est"]),
        se=float(h["se"]),
        p_crv1=float(h["p"]),
        p_permutation=float(h["ri_p"]),
        n_obs=int(h["n"]),
        keep_or_drop="appendix_only",
        reason="Current paper contrast; different sample and zero-cell treatment from the headline clean window",
        source="78_hierarchy_numbers.py / output/hierarchy_numbers.csv",
    )


def main() -> None:
    print("== bounded stake-gradient search: fixed design, full failure log ==", flush=True)
    pref = load_prefecture_frame()
    cases = read_lending_cases()
    print(f"eligible prefectures={len(pref):,}; lending cases={len(cases):,}", flush=True)

    coverage_spec(cases, pref)

    judgment = balanced_bin_panel(cases, pref, "judgment_month", "judgment")
    per_band_estimates(judgment, "judgment")

    primary_panel = group_average_panel(judgment, "asinh")
    primary_panel["clock"] = "judgment"
    linear_shape("J1_balanced_asinh", primary_panel, "asinh bin counts")

    log_panel = group_average_panel(judgment, "log1p")
    log_panel["clock"] = "judgment"
    linear_shape("J2_balanced_log1p", log_panel, "log(1 + bin count)")

    level_panel = group_average_panel(judgment, "level")
    level_panel["clock"] = "judgment"
    linear_shape("J3_balanced_level", level_panel, "level bin count")

    any_panel = group_average_panel(judgment, "any")
    any_panel["clock"] = "judgment"
    linear_shape("J4_balanced_extensive", any_panel, "Pr(bin count > 0)")

    strict_panel = primary_panel.copy()
    linear_shape(
        "J5_balanced_asinh_strict_fe", strict_panel, "asinh bin counts",
        fml=FML_STRICT, strict=True,
    )

    share_spec(judgment, weighted=False)
    share_spec(judgment, weighted=True)
    ppml_shape("J8_balanced_ppml", level_panel)

    perm_frames = [wave_permutation("J1_balanced_asinh", primary_panel, FML_BASE)]

    try:
        filing_cases = filing_clock(cases)
        filing = balanced_bin_panel(filing_cases, pref, "filing_month", "filing")
        per_band_estimates(filing, "filing")
        filing_panel = group_average_panel(filing, "asinh")
        filing_panel["clock"] = "filing"
        linear_shape("F1_filing_balanced_asinh", filing_panel, "asinh bin counts")
        perm_frames.append(wave_permutation("F1_filing_balanced_asinh", filing_panel, FML_BASE))
    except Exception as exc:
        print(f"[filing clock failed] {exc}", flush=True)
        add_row(
            spec_id="F1_filing_balanced_asinh",
            transformation="asinh bin counts",
            model="FE-OLS",
            sample_rule="filing clock; fixed clean window; balanced zeros",
            fixed_effects="prefecture x middle-group; month x middle-group",
            estimate=np.nan, se=np.nan, p_crv1=np.nan, n_obs=0,
            keep_or_drop="drop", reason=f"Filing construction failed: {exc}",
        )

    append_legacy_results()

    spec_log = pd.DataFrame(rows)
    spec_log.to_csv(OUT / "stake_spec_log.csv", index=False)
    pd.DataFrame(band_rows).to_csv(OUT / "stake_per_band.csv", index=False)
    pd.concat(perm_frames, ignore_index=True).to_csv(
        OUT / "stake_permutation.csv", index=False
    )
    pd.DataFrame(diagnostics).to_csv(OUT / "stake_diagnostics.csv", index=False)

    print("\n== sorted admissible results (CRV1 p) ==", flush=True)
    show = spec_log[~spec_log["spec_id"].str.startswith("L")].sort_values("p_value")
    print(
        show[["spec_id", "coefficient", "std_error", "p_value", "p_wild", "p_crv3", "p_permutation"]]
        .to_string(index=False),
        flush=True,
    )
    print("\nSaved independent outputs; no paper or existing pipeline file was modified.", flush=True)


if __name__ == "__main__":
    main()

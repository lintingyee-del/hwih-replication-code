# -*- coding: utf-8 -*-
"""Cross-province neighbour-pair spillover audit.

This script is deliberately separate from the paper's production chain.  It tests
whether the neighbour-pair estimate could be amplified by displacement of private
enforcement into not-yet-inspected prefectures near first-wave provinces.

Pre-specified diagnostics
-------------------------
1. Control-side border versus interior: within later-wave provinces, compare
   prefectures within 200 km of a first-wave prefecture with prefectures in the
   same province farther than 200 km.  Province-by-month fixed effects absorb all
   later-wave province shocks.  The primary outcome is the paper's relational
   cause-cell asinh count; the relational-minus-traffic gap is corroborating.
2. Pair donut: re-estimate the paper's exact <=200 km neighbour-pair model after
   dropping pairs within 100 km.

The bounded search varies only defensible distance bands and the two outcomes
already used in the paper.  Every attempted specification is logged, including
small-sample and statistically weak results.  No manuscript file is modified.
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

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import pyfixest as pf

from _wild import wild_score_p


ROOT = Path(str(_REP_PROJECT))
DATA = ROOT / "data"
OUT = ROOT / "output" / "border_spillover"
OUT.mkdir(parents=True, exist_ok=True)

WINDOW = ("2017-01", "2019-03")
POST0 = "2018-09"
PRIMARY_DISTANCE = 200.0
PRIMARY_DONUT = (100.0, 200.0)
WILD_REPS = 9_999
WILD_SEED = 42


def haversine(la1, lo1, la2, lo2):
    r = np.pi / 180.0
    radius = 6371.0
    dla = (la2 - la1) * r
    dlo = (lo2 - lo1) * r
    a = (
        np.sin(dla / 2.0) ** 2
        + np.cos(la1 * r) * np.cos(la2 * r) * np.sin(dlo / 2.0) ** 2
    )
    return 2.0 * radius * np.arcsin(np.sqrt(a))


def bh_adjust(values):
    """Benjamini-Hochberg q-values, preserving missing entries."""
    x = np.asarray(values, float)
    out = np.full(len(x), np.nan)
    keep = np.isfinite(x)
    if not keep.any():
        return out
    p = x[keep]
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.minimum(q, 1.0)
    restored = np.empty(len(p))
    restored[order] = q
    out[np.flatnonzero(keep)] = restored
    return out


def load_panel():
    cp = pd.read_parquet(DATA / "civil_panel.parquet")
    cp["prefecture_code"] = cp["prefecture_code"].astype(str)
    cp["month"] = cp["jmonth"].astype(str).str[:7]
    cp = cp[
        (cp["month"] >= WINDOW[0])
        & (cp["month"] <= WINDOW[1])
    ].copy()

    sched = (
        pd.read_parquet(DATA / "panel_month.parquet")
        [["province", "inspection_round"]]
        .drop_duplicates()
    )
    cp = cp.merge(sched, on="province", how="left", validate="many_to_one")
    cp = cp.dropna(subset=["inspection_round", "exposure_v2_z"]).copy()
    cp["inspection_round"] = cp["inspection_round"].astype(int)
    cp["treat"] = (cp["inspection_round"] == 1).astype(int)
    cp["postc"] = (cp["month"] >= POST0).astype(int)
    cp["prov_id"] = pd.factorize(cp["province"])[0]
    cp["pref_cause"] = cp["prefecture_code"] + "_" + cp["cause"]
    cp["asinh_n"] = np.arcsinh(cp["n_cases"])
    cp["pt"] = cp["postc"] * cp["treat"]
    cp["ph"] = cp["postc"] * cp["exposure_v2_z"]
    cp["pth"] = cp["pt"] * cp["exposure_v2_z"]
    return cp


def build_geography(cp):
    cen = pd.read_csv(DATA / "pref_centroids.csv", dtype={"prefecture_code": str})
    pref = (
        cp[
            [
                "prefecture_code",
                "province",
                "inspection_round",
                "exposure_v2_z",
            ]
        ]
        .drop_duplicates("prefecture_code")
        .merge(cen, on="prefecture_code", how="left", validate="one_to_one")
        .dropna(subset=["lat", "lon"])
        .copy()
    )
    treated = pref[pref["inspection_round"] == 1].reset_index(drop=True)
    control = pref[pref["inspection_round"] != 1].reset_index(drop=True)
    dm = haversine(
        np.asarray(treated["lat"], float)[:, None],
        np.asarray(treated["lon"], float)[:, None],
        np.asarray(control["lat"], float)[None, :],
        np.asarray(control["lon"], float)[None, :],
    )
    ti = np.argmin(dm, axis=1)
    ci = np.argmin(dm, axis=0)
    td = dm.min(axis=1)
    cd = dm.min(axis=0)

    control = control.copy()
    control["distance_to_wave1_km"] = cd
    control["nearest_wave1_prefecture"] = np.asarray(
        treated.loc[ci, "prefecture_code"]
    )

    # Union of treated-to-control and control-to-treated nearest-neighbour links.
    pairs = {}
    for i in range(len(treated)):
        key = (
            treated.loc[i, "prefecture_code"],
            control.loc[ti[i], "prefecture_code"],
        )
        pairs[key] = float(td[i])
    for j in range(len(control)):
        key = (
            treated.loc[ci[j], "prefecture_code"],
            control.loc[j, "prefecture_code"],
        )
        pairs[key] = float(cd[j])
    pair_df = pd.DataFrame(
        [
            {
                "treated_prefecture": a,
                "control_prefecture": b,
                "distance_km": dist,
            }
            for (a, b), dist in sorted(pairs.items())
        ]
    )
    return pref, control, pair_df


def model_row(
    *,
    spec_id,
    family,
    outcome,
    model,
    target,
    transformation,
    sample_rule,
    controls,
    fixed_effects,
    expected_direction,
    status,
    reason,
    wild_p=np.nan,
    n_clusters=np.nan,
    n_near_pref=np.nan,
    n_far_pref=np.nan,
    n_pairs=np.nan,
):
    coef = float(model.coef()[target])
    se = float(model.se()[target])
    p = float(model.pvalue()[target])
    ci = model.confint().loc[target]
    direction = "negative" if coef < 0 else "positive" if coef > 0 else "zero"
    return {
        "spec_id": spec_id,
        "mode": "direct_experiment",
        "focus_side": "x" if family == "pair_donut" else "both",
        "family": family,
        "base_variable": outcome,
        "transformation": transformation,
        "model": "FE-OLS, CRV1 by province",
        "sample_rule": sample_rule,
        "controls": controls,
        "fixed_effects": fixed_effects,
        "target": target,
        "coefficient": coef,
        "std_error": se,
        "p_value": p,
        "wild_p": wild_p,
        "ci95_low": float(ci.iloc[0]),
        "ci95_high": float(ci.iloc[1]),
        "n_obs": int(model._N),
        "n_clusters": n_clusters,
        "n_near_pref": n_near_pref,
        "n_far_pref": n_far_pref,
        "n_pairs": n_pairs,
        "expected_direction": expected_direction,
        "direction": direction,
        "keep_or_drop": status,
        "reason": reason,
    }


def prepare_control_outcome(cp, control_geo, outcome):
    d = cp[cp["inspection_round"] != 1].copy()
    geo_cols = ["prefecture_code", "distance_to_wave1_km"]
    d = d.merge(
        control_geo[geo_cols],
        on="prefecture_code",
        how="inner",
        validate="many_to_one",
    )
    if outcome == "relational_cause_cells":
        d = d[d["cause_family"] == "relational"].copy()
        d["y"] = np.arcsinh(d["n_cases"])
        d["unit_fe"] = d["prefecture_code"] + "_" + d["cause"]
        d["cause_month"] = d["cause"] + "_" + d["month"]
        fe = "unit_fe + prov_month + cause_month"
    elif outcome in {
        "relational_aggregate",
        "traffic_aggregate",
        "relational_minus_traffic",
    }:
        g = (
            d[d["cause_family"].isin(["relational", "placebo"])]
            .groupby(
                [
                    "prefecture_code",
                    "province",
                    "month",
                    "inspection_round",
                    "exposure_v2_z",
                    "distance_to_wave1_km",
                    "cause_family",
                ],
                as_index=False,
            )["n_cases"]
            .sum()
        )
        wide = g.pivot_table(
            index=[
                "prefecture_code",
                "province",
                "month",
                "inspection_round",
                "exposure_v2_z",
                "distance_to_wave1_km",
            ],
            columns="cause_family",
            values="n_cases",
            fill_value=0,
        ).reset_index()
        wide.columns.name = None
        if outcome == "relational_aggregate":
            wide["y"] = np.arcsinh(wide["relational"])
        elif outcome == "traffic_aggregate":
            wide["y"] = np.arcsinh(wide["placebo"])
        else:
            wide["y"] = np.arcsinh(wide["relational"]) - np.arcsinh(
                wide["placebo"]
            )
        wide["unit_fe"] = wide["prefecture_code"]
        d = wide
        fe = "unit_fe + prov_month"
    else:
        raise ValueError(outcome)
    d["prov_month"] = d["province"] + "_" + d["month"]
    d["prov_id"] = pd.factorize(d["province"])[0]
    return d, fe


def fit_control_side(
    cp,
    control_geo,
    *,
    distance,
    outcome,
    interacted,
    placebo=False,
    compute_wild=False,
):
    d, fe = prepare_control_outcome(cp, control_geo, outcome)
    d["near"] = (d["distance_to_wave1_km"] <= distance).astype(int)

    # Match DLR's border-versus-state-interior logic: keep only later-wave
    # provinces containing both a near and an interior prefecture.
    overlap = (
        d[["province", "prefecture_code", "near"]]
        .drop_duplicates()
        .groupby("province")["near"]
        .nunique()
    )
    overlap = set(overlap[overlap == 2].index)
    d = d[d["province"].isin(overlap)].copy()

    pref_geo = d[
        ["prefecture_code", "near", "exposure_v2_z"]
    ].drop_duplicates("prefecture_code")
    h_near = float(pref_geo.loc[pref_geo["near"] == 1, "exposure_v2_z"].mean())
    d["Hc"] = d["exposure_v2_z"] - h_near

    if placebo:
        d = d[(d["month"] >= "2017-01") & (d["month"] <= "2017-12")].copy()
        d["post_test"] = (d["month"] >= "2017-07").astype(int)
        clock = "2017H2 versus 2017H1 placebo"
    else:
        d["post_test"] = (d["month"] >= POST0).astype(int)
        clock = f"post >= {POST0}"

    d["post_near"] = d["post_test"] * d["near"]
    d["post_H"] = d["post_test"] * d["Hc"]
    d["post_near_H"] = d["post_test"] * d["near"] * d["Hc"]

    rhs = "post_near + post_H"
    if interacted:
        rhs += " + post_near_H"
    fml = f"y ~ {rhs} | {fe}"
    model = pf.feols(fml, data=d, vcov={"CRV1": "prov_id"})

    unique_pref = d[["prefecture_code", "near"]].drop_duplicates()
    n_near = int(unique_pref["near"].sum())
    n_far = int((1 - unique_pref["near"]).sum())
    n_clusters = int(d["province"].nunique())
    tag = "full" if interacted else "additive"
    prefix = "P" if placebo else "C"
    spec_base = f"{prefix}_{outcome}_{int(distance)}km_{tag}"

    targets = ["post_near"] + (["post_near_H"] if interacted else [])
    rows = []
    for target in targets:
        wp = np.nan
        if compute_wild:
            wp = wild_score_p(
                fml,
                d,
                target,
                cluster="prov_id",
                reps=WILD_REPS,
                seed=WILD_SEED,
            )
        expected = "negative"
        target_desc = (
            "near-control post change at the near-group mean H"
            if target == "post_near"
            else "near-control differential post slope in own H"
        )
        primary = (
            (not placebo)
            and distance == PRIMARY_DISTANCE
            and outcome == "relational_cause_cells"
        )
        if primary:
            status = "primary"
        elif placebo:
            status = "diagnostic"
        elif distance == 100:
            status = "drop_small_sample"
        else:
            status = "exploratory"
        reason = (
            "Pre-specified direct control-side displacement test."
            if primary
            else "Pre-treatment placebo for the direct test."
            if placebo
            else "Only three overlap provinces; retained for disclosure, not interpretation."
            if distance == 100
            else "Admissible distance/outcome sensitivity retained in the full log."
        )
        rows.append(
            model_row(
                spec_id=f"{spec_base}_{target}",
                family="control_side",
                outcome=outcome,
                model=model,
                target=target,
                transformation=(
                    "asinh cause-cell count"
                    if outcome == "relational_cause_cells"
                    else "asinh aggregate relational count"
                    if outcome == "relational_aggregate"
                    else "asinh aggregate traffic count"
                    if outcome == "traffic_aggregate"
                    else "asinh relational minus asinh traffic"
                ),
                sample_rule=(
                    f"later-wave provinces with both <= {int(distance)} km and > "
                    f"{int(distance)} km prefectures; {clock}; H centered at near mean"
                ),
                controls="post x centered own H; fully interacted" if interacted else "post x centered own H",
                fixed_effects=fe.replace("unit_fe", "prefecture x cause" if outcome == "relational_cause_cells" else "prefecture"),
                expected_direction=expected,
                status=status,
                reason=reason + " Target: " + target_desc,
                wild_p=wp,
                n_clusters=n_clusters,
                n_near_pref=n_near,
                n_far_pref=n_far,
            )
        )

    joint = None
    if interacted:
        names = list(model.coef().index)
        R = np.zeros((2, len(names)))
        R[0, names.index("post_near")] = 1.0
        R[1, names.index("post_near_H")] = 1.0
        joint_test = model.wald_test(R=R, q=np.zeros(2), distribution="chi2")
        joint = {
            "spec_id": spec_base,
            "family": "control_side",
            "outcome": outcome,
            "distance_low_km": 0,
            "distance_high_km": distance,
            "placebo": placebo,
            "interacted": interacted,
            "joint_crv1_chi2": float(joint_test["statistic"]),
            "joint_crv1_p": float(joint_test["pvalue"]),
            "n_obs": int(model._N),
            "n_clusters": n_clusters,
            "n_near_pref": n_near,
            "n_far_pref": n_far,
            "h_center_near_mean": h_near,
        }
    return rows, joint


def control_side_lopo(cp, control_geo, *, distance, outcome):
    """Leave one overlap province out of the fully interacted control-side fit."""
    d, fe = prepare_control_outcome(cp, control_geo, outcome)
    d["near"] = (d["distance_to_wave1_km"] <= distance).astype(int)
    overlap = (
        d[["province", "prefecture_code", "near"]]
        .drop_duplicates()
        .groupby("province")["near"]
        .nunique()
    )
    overlap = sorted(overlap[overlap == 2].index)
    d = d[d["province"].isin(overlap)].copy()
    pref_geo = d[
        ["prefecture_code", "near", "exposure_v2_z"]
    ].drop_duplicates("prefecture_code")
    h_near = float(pref_geo.loc[pref_geo["near"] == 1, "exposure_v2_z"].mean())
    d["Hc"] = d["exposure_v2_z"] - h_near
    d["post_test"] = (d["month"] >= POST0).astype(int)
    d["post_near"] = d["post_test"] * d["near"]
    d["post_H"] = d["post_test"] * d["Hc"]
    d["post_near_H"] = d["post_test"] * d["near"] * d["Hc"]
    fml = f"y ~ post_near + post_H + post_near_H | {fe}"
    rows = []
    for omitted in overlap:
        sub = d[d["province"] != omitted].copy()
        sub["prov_id"] = pd.factorize(sub["province"])[0]
        model = pf.feols(fml, data=sub, vcov={"CRV1": "prov_id"})
        for target in ("post_near", "post_near_H"):
            rows.append(
                {
                    "distance_km": distance,
                    "outcome": outcome,
                    "omitted_province": omitted,
                    "target": target,
                    "coefficient": float(model.coef()[target]),
                    "std_error": float(model.se()[target]),
                    "p_value": float(model.pvalue()[target]),
                    "n_obs": int(model._N),
                    "n_clusters": int(sub["province"].nunique()),
                }
            )
    return rows


def control_side_eventstudy(cp, control_geo, *, distance, outcome, compute_wild=False):
    """Binned calendar dynamics relative to 2017H2.

    The bins separate the January 2018 national launch from the July-August
    first-wave rollout and the September 2018 clean post period.
    """
    d, fe = prepare_control_outcome(cp, control_geo, outcome)
    d["near"] = (d["distance_to_wave1_km"] <= distance).astype(int)
    overlap = (
        d[["province", "prefecture_code", "near"]]
        .drop_duplicates()
        .groupby("province")["near"]
        .nunique()
    )
    overlap = sorted(overlap[overlap == 2].index)
    d = d[d["province"].isin(overlap)].copy()
    pref_geo = d[
        ["prefecture_code", "near", "exposure_v2_z"]
    ].drop_duplicates("prefecture_code")
    h_near = float(pref_geo.loc[pref_geo["near"] == 1, "exposure_v2_z"].mean())
    d["Hc"] = d["exposure_v2_z"] - h_near

    def period(month):
        if month <= "2017-06":
            return "2017h1"
        if month <= "2017-12":
            return "ref_2017h2"
        if month <= "2018-06":
            return "2018h1_launch"
        if month <= "2018-08":
            return "2018julaug_rollout"
        return "post_sep2018"

    d["period"] = d["month"].map(period)
    bins = ["2017h1", "2018h1_launch", "2018julaug_rollout", "post_sep2018"]
    rhs = []
    for b in bins:
        flag = (d["period"] == b).astype(int)
        d[f"es_{b}_near"] = flag * d["near"]
        d[f"es_{b}_H"] = flag * d["Hc"]
        d[f"es_{b}_near_H"] = flag * d["near"] * d["Hc"]
        rhs.extend([f"es_{b}_near", f"es_{b}_H", f"es_{b}_near_H"])
    fml = f"y ~ {' + '.join(rhs)} | {fe}"
    model = pf.feols(fml, data=d, vcov={"CRV1": "prov_id"})
    ci = model.confint()
    rows = []
    for b in bins:
        for margin in ("near", "near_H"):
            target = f"es_{b}_{margin}"
            wp = np.nan
            if compute_wild and b == "post_sep2018":
                wp = wild_score_p(
                    fml,
                    d,
                    target,
                    cluster="prov_id",
                    reps=WILD_REPS,
                    seed=WILD_SEED,
                )
            rows.append(
                {
                    "distance_km": distance,
                    "outcome": outcome,
                    "reference_period": "2017h2",
                    "period": b,
                    "margin": margin,
                    "coefficient": float(model.coef()[target]),
                    "std_error": float(model.se()[target]),
                    "p_value": float(model.pvalue()[target]),
                    "wild_p": wp,
                    "ci95_low": float(ci.loc[target].iloc[0]),
                    "ci95_high": float(ci.loc[target].iloc[1]),
                    "n_obs": int(model._N),
                    "n_clusters": int(d["province"].nunique()),
                }
            )
    return rows


def fit_pair_donut(cp, pair_df, low, high, compute_wild=False):
    pairs = pair_df[
        (pair_df["distance_km"] > low) & (pair_df["distance_km"] <= high)
    ].copy()
    d0 = cp[cp["cause_family"] == "relational"].copy()
    stack = []
    for pair_id, row in pairs.reset_index(drop=True).iterrows():
        seg = d0[
            d0["prefecture_code"].isin(
                [row["treated_prefecture"], row["control_prefecture"]]
            )
        ].copy()
        seg["pair_id"] = pair_id
        seg["pair_month"] = str(pair_id) + "_" + seg["month"]
        seg["pair_pref_cause"] = str(pair_id) + "_" + seg["pref_cause"]
        stack.append(seg)
    if not stack:
        raise ValueError(f"No pairs in ({low}, {high}] km")
    d = pd.concat(stack, ignore_index=True)
    fml = "asinh_n ~ pth + ph + pt | pair_pref_cause + pair_month"
    model = pf.feols(fml, data=d, vcov={"CRV1": "prov_id"})
    wp = np.nan
    if compute_wild:
        wp = wild_score_p(
            fml,
            d,
            "pth",
            cluster="prov_id",
            reps=WILD_REPS,
            seed=WILD_SEED,
        )
    primary = (low, high) in [(0.0, 200.0), PRIMARY_DONUT]
    reason = (
        "Exact reproduction of the paper's <=200 km pair model."
        if (low, high) == (0.0, 200.0)
        else "Pre-specified donut removing all pairs within 100 km."
        if (low, high) == PRIMARY_DONUT
        else "Admissible mutually bounded distance sensitivity retained in the full log."
    )
    pair_status = (
        "primary"
        if primary
        else "drop_small_sample"
        if (low, high) == (0.0, 100.0)
        else "exploratory"
    )
    if (low, high) == (0.0, 100.0):
        reason = "Only eight pairs; retained for disclosure, not interpretation."
    row = model_row(
        spec_id=f"D_{int(low)}_{int(high)}km_pth",
        family="pair_donut",
        outcome="relational_cause_cells",
        model=model,
        target="pth",
        transformation="asinh count",
        sample_rule=f"nearest cross-province pairs with {int(low)} < distance <= {int(high)} km",
        controls="post x H and post x treated",
        fixed_effects="pair x prefecture x cause; pair x month",
        expected_direction="positive",
        status=pair_status,
        reason=reason,
        wild_p=wp,
        n_clusters=int(d["province"].nunique()),
        n_pairs=int(len(pairs)),
    )
    diag = {
        "spec_id": f"D_{int(low)}_{int(high)}km",
        "family": "pair_donut",
        "outcome": "relational_cause_cells",
        "distance_low_km": low,
        "distance_high_km": high,
        "placebo": False,
        "interacted": True,
        "joint_crv1_chi2": np.nan,
        "joint_crv1_p": np.nan,
        "n_obs": int(model._N),
        "n_clusters": int(d["province"].nunique()),
        "n_near_pref": np.nan,
        "n_far_pref": np.nan,
        "n_pairs": int(len(pairs)),
        "min_pair_distance": float(pairs["distance_km"].min()),
        "max_pair_distance": float(pairs["distance_km"].max()),
    }
    return row, diag


def main():
    warnings.filterwarnings("default")
    cp = load_panel()
    pref, control_geo, pair_df = build_geography(cp)
    pair_df.to_csv(OUT / "pair_definitions.csv", index=False)
    control_geo.to_csv(OUT / "control_distance_to_wave1.csv", index=False)

    rows = []
    diagnostics = []
    lopo_rows = []
    eventstudy_rows = []

    # Pre-specified direct test and its fully interacted version.
    for interacted in (False, True):
        rr, jj = fit_control_side(
            cp,
            control_geo,
            distance=PRIMARY_DISTANCE,
            outcome="relational_cause_cells",
            interacted=interacted,
            compute_wild=True,
        )
        rows.extend(rr)
        if jj is not None:
            diagnostics.append(jj)

    # Decompose the traffic-adjusted result into its two aggregate outcomes.
    for outcome in ("relational_aggregate", "traffic_aggregate"):
        rr, jj = fit_control_side(
            cp,
            control_geo,
            distance=PRIMARY_DISTANCE,
            outcome=outcome,
            interacted=True,
            compute_wild=True,
        )
        rows.extend(rr)
        if jj is not None:
            diagnostics.append(jj)

    # Corroborating traffic-adjusted outcome at the primary distance.
    for interacted in (False, True):
        rr, jj = fit_control_side(
            cp,
            control_geo,
            distance=PRIMARY_DISTANCE,
            outcome="relational_minus_traffic",
            interacted=interacted,
            compute_wild=True,
        )
        rows.extend(rr)
        if jj is not None:
            diagnostics.append(jj)

    # Pre-inspection placebo, same fully interacted specification.
    for outcome in ("relational_cause_cells", "relational_minus_traffic"):
        rr, jj = fit_control_side(
            cp,
            control_geo,
            distance=PRIMARY_DISTANCE,
            outcome=outcome,
            interacted=True,
            placebo=True,
            compute_wild=True,
        )
        rows.extend(rr)
        if jj is not None:
            diagnostics.append(jj)

    # bounded distance search.  Same model; CRV1 is logged for every attempt.
    for distance in (100.0, 150.0, 250.0):
        for outcome in ("relational_cause_cells", "relational_minus_traffic"):
            rr, jj = fit_control_side(
                cp,
                control_geo,
                distance=distance,
                outcome=outcome,
                interacted=True,
                compute_wild=(distance == 150.0),
            )
            rows.extend(rr)
            if jj is not None:
                diagnostics.append(jj)

    # At 150 km, decompose the exploratory traffic-adjusted signal too.
    for outcome in ("relational_aggregate", "traffic_aggregate"):
        rr, jj = fit_control_side(
            cp,
            control_geo,
            distance=150.0,
            outcome=outcome,
            interacted=True,
            compute_wild=True,
        )
        rows.extend(rr)
        if jj is not None:
            diagnostics.append(jj)

    # Influence diagnostics for the primary distance and the 150-km signal.
    for distance in (150.0, PRIMARY_DISTANCE):
        for outcome in (
            "relational_cause_cells",
            "relational_aggregate",
            "traffic_aggregate",
            "relational_minus_traffic",
        ):
            lopo_rows.extend(
                control_side_lopo(
                    cp,
                    control_geo,
                    distance=distance,
                    outcome=outcome,
                )
            )
            eventstudy_rows.extend(
                control_side_eventstudy(
                    cp,
                    control_geo,
                    distance=distance,
                    outcome=outcome,
                    compute_wild=outcome
                    in {"relational_aggregate", "relational_minus_traffic"},
                )
            )

    # Exact main model, primary donut, and a finite set of transparent bands.
    donut_bands = [
        (0.0, 100.0),
        (0.0, 150.0),
        (0.0, 200.0),
        (0.0, 250.0),
        (50.0, 200.0),
        (100.0, 200.0),
        (150.0, 200.0),
        (100.0, 250.0),
        (150.0, 250.0),
        (200.0, 250.0),
    ]
    for low, high in donut_bands:
        row, diag = fit_pair_donut(
            cp,
            pair_df,
            low,
            high,
            compute_wild=True,
        )
        rows.append(row)
        diagnostics.append(diag)

    log = pd.DataFrame(rows)
    log["bh_q_crv1"] = np.nan
    log["bh_q_wild"] = np.nan
    # Adjust within each result family.  Placebo rows are a separate diagnostic
    # family; the control-side post rows include every searched distance/outcome.
    family_keys = pd.Series(index=log.index, dtype="object")
    family_keys.loc[log["family"] == "pair_donut"] = "pair_donut"
    family_keys.loc[
        (log["family"] == "control_side") & log["spec_id"].str.startswith("C_")
    ] = "control_side_post"
    family_keys.loc[
        (log["family"] == "control_side") & log["spec_id"].str.startswith("P_")
    ] = "control_side_placebo"
    for key in family_keys.dropna().unique():
        ix = family_keys[family_keys == key].index
        log.loc[ix, "bh_q_crv1"] = bh_adjust(log.loc[ix, "p_value"])
        log.loc[ix, "bh_q_wild"] = bh_adjust(log.loc[ix, "wild_p"])
    log.to_csv(OUT / "spec_log.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(OUT / "model_diagnostics.csv", index=False)
    pd.DataFrame(lopo_rows).to_csv(OUT / "control_side_lopo.csv", index=False)
    pd.DataFrame(eventstudy_rows).to_csv(
        OUT / "control_side_eventstudy.csv", index=False
    )

    manifest = {
        "mode": "direct_experiment",
        "research_question": "Can cross-province displacement amplify the neighbour-pair judicialization estimate?",
        "primary_control_test": "Later-wave prefectures <=200 km from wave-1 versus same-province interior prefectures",
        "primary_donut": "100 < pair distance <= 200 km",
        "window": WINDOW,
        "post_start": POST0,
        "wild_reps": WILD_REPS,
        "wild_seed": WILD_SEED,
        "n_prefectures": int(pref["prefecture_code"].nunique()),
        "n_first_wave_prefectures": int((pref["inspection_round"] == 1).sum()),
        "n_later_wave_prefectures": int((pref["inspection_round"] != 1).sum()),
        "manuscript_modified": False,
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    show = log[
        log["spec_id"].isin(
            [
                "C_relational_cause_cells_200km_additive_post_near",
                "C_relational_cause_cells_200km_full_post_near",
                "C_relational_cause_cells_200km_full_post_near_H",
                "C_relational_minus_traffic_200km_full_post_near",
                "C_relational_minus_traffic_200km_full_post_near_H",
                "D_0_200km_pth",
                "D_100_200km_pth",
            ]
        )
    ][
        [
            "spec_id",
            "coefficient",
            "std_error",
            "p_value",
            "wild_p",
            "ci95_low",
            "ci95_high",
            "n_obs",
            "n_clusters",
            "n_pairs",
        ]
    ]
    print("\nPRIMARY RESULTS")
    print(show.to_string(index=False))
    print(f"\nFull specification log: {OUT / 'spec_log.csv'}")
    print(f"Diagnostics: {OUT / 'model_diagnostics.csv'}")


if __name__ == "__main__":
    main()

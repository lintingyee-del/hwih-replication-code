# -*- coding: utf-8 -*-
"""Validation gates for the two interpretable external-exposure candidates.

This script does not search new exposure definitions.  It takes the preferred
pre-campaign Baidu index and the economically closer but sparse vernacular firm
stock from code/116_external_exposure_xianzhu.py, then applies fixed gates:

  * CRV1, wild-score, and CRV3 inference under baseline and saturated FE;
  * relational and traffic aggregate decompositions;
  * calendar-bin event studies with joint lead tests;
  * leave-one-province-out stability;
  * 999 wave-label permutations holding the treated-province count fixed.

All outputs are diagnostics.  Inspection timing was not randomized, so the
wave-label exercise is not exact design-based randomization inference.
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

import numpy as np
import pandas as pd
import pyfixest as pf
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from _wild import wild_score_p


BASE = str(_REP_PROJECT)
DATA = os.path.join(BASE, "data")
SEARCH_OUT = os.path.join(BASE, "output", "external_exposure_xianzhu")
OUT = os.path.join(SEARCH_OUT, "validation")
os.makedirs(OUT, exist_ok=True)

WINDOW = ("2017-01", "2019-03")
SUPPORT = ("2014-01", "2017-12")
POST0 = "2018-09"
CAL_BINS = [(-20, -13), (-12, -7), (-6, -1), (0, 6)]
CAL_REF = (-6, -1)
REPS = 999
SEED = 42

FOCUS = {
    "baidu_combo_1417_asinh": "Baidu: 讨债 + 讨债公司, 2014-17 mean",
    "firm_vernacular_stock_asinh_density": (
        "Registry: vernacular collection-firm stock at end-2017"
    ),
}

LOG: list[str] = []


def say(*args) -> None:
    line = " ".join(str(x) for x in args)
    print(line, flush=True)
    LOG.append(line)


candidate_values = pd.read_csv(
    os.path.join(SEARCH_OUT, "candidate_values.csv"),
    dtype={"prefecture_code": str},
)
schedule = (
    pd.read_parquet(os.path.join(DATA, "panel_month.parquet"))[
        ["province", "inspection_round"]
    ]
    .drop_duplicates()
)
months = pd.DataFrame(
    {"month": pd.period_range(WINDOW[0], WINDOW[1], freq="M").astype(str)}
)


# ---------------------------------------------------------------------------
# Fixed outcome panels.
# ---------------------------------------------------------------------------
civil = pd.read_parquet(os.path.join(DATA, "civil_panel.parquet"))
civil["month"] = civil["jmonth"].astype(str).str[:7]
relational = civil[civil["cause_family"].eq("relational")].copy()
support = relational[relational["month"].between(SUPPORT[0], SUPPORT[1])][
    ["prefecture_code", "province", "cause"]
].drop_duplicates()
counts = relational[relational["month"].between(WINDOW[0], WINDOW[1])][
    ["prefecture_code", "province", "cause", "month", "n_cases"]
]
flow = (
    support.merge(months, how="cross")
    .merge(
        counts,
        on=["prefecture_code", "province", "cause", "month"],
        how="left",
    )
    .merge(schedule, on="province")
)
flow["n"] = flow["n_cases"].fillna(0.0)
flow["y"] = np.arcsinh(flow["n"])
flow["pref_cause"] = flow["prefecture_code"] + "_" + flow["cause"]
flow["cause_month"] = flow["cause"] + "_" + flow["month"]

rt = civil[civil["month"].between(WINDOW[0], WINDOW[1])].copy()
rt["group"] = np.where(
    rt["cause_family"].eq("relational"),
    "relational",
    np.where(rt["cause_family"].eq("placebo"), "traffic", "other"),
)
rt = rt[rt["group"].isin(["relational", "traffic"])]
aggregate = (
    rt.groupby(["prefecture_code", "province", "month", "group"], as_index=False)[
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
    .merge(schedule, on="province")
)
aggregate["y_relational"] = np.arcsinh(aggregate["relational"])
aggregate["y_traffic"] = np.arcsinh(aggregate["traffic"])
aggregate["y_gap"] = aggregate["y_relational"] - aggregate["y_traffic"]


def add_design(frame: pd.DataFrame, variant_id: str) -> pd.DataFrame:
    index = candidate_values[["prefecture_code", variant_id]].dropna()
    d = frame.merge(index, on="prefecture_code").copy()
    d["H"] = d[variant_id]
    d["treat"] = (d["inspection_round"] == 1).astype(int)
    d["postc"] = (d["month"] >= POST0).astype(int)
    d["pt"] = d["postc"] * d["treat"]
    d["ph"] = d["postc"] * d["H"]
    d["pth"] = d["pt"] * d["H"]
    d["prov_id"] = pd.factorize(d["province"])[0]
    d["prov_month"] = d["province"] + "_" + d["month"]
    return d


def fit_all(
    label: str,
    variant_id: str,
    outcome: str,
    fe_spec: str,
    formula: str,
    frame: pd.DataFrame,
) -> dict:
    m1 = pf.feols(formula, data=frame, vcov={"CRV1": "prov_id"})
    m3 = pf.feols(formula, data=frame, vcov={"CRV3": "prov_id"})
    return {
        "variant_id": variant_id,
        "label": label,
        "outcome": outcome,
        "fixed_effects": fe_spec,
        "coefficient": float(m1.coef()["pth"]),
        "se_crv1": float(m1.se()["pth"]),
        "p_crv1": float(m1.pvalue()["pth"]),
        "p_wild": float(wild_score_p(formula, frame, "pth")),
        "se_crv3": float(m3.se()["pth"]),
        "p_crv3": float(m3.pvalue()["pth"]),
        "n_obs": int(m1._N),
        "province_clusters": int(frame["prov_id"].nunique()),
        "formula": formula,
    }


say("=== Static inference and docket decomposition ===")
static_rows = []
for variant_id, label in FOCUS.items():
    dflow = add_design(flow, variant_id)
    dagg = add_design(aggregate, variant_id)
    flow_formulas = {
        "baseline": "y ~ pth + ph + pt | pref_cause + month",
        "province_month_saturated": (
            "y ~ pth + ph | pref_cause + prov_month + cause_month"
        ),
    }
    aggregate_formulas = {
        "baseline": "OUTCOME ~ pth + ph + pt | prefecture_code + month",
        "province_month_saturated": (
            "OUTCOME ~ pth + ph | prefecture_code + prov_month"
        ),
    }
    for fe_spec, formula in flow_formulas.items():
        row = fit_all(
            label,
            variant_id,
            "balanced_relational_cause_flow",
            fe_spec,
            formula,
            dflow,
        )
        static_rows.append(row)
        say(
            f"  {variant_id} flow {fe_spec}: b={row['coefficient']:+.4f} "
            f"wild={row['p_wild']:.4f} CRV3={row['p_crv3']:.4f}"
        )
    for outcome in ("y_relational", "y_traffic", "y_gap"):
        for fe_spec, template in aggregate_formulas.items():
            formula = template.replace("OUTCOME", outcome)
            row = fit_all(
                label, variant_id, outcome, fe_spec, formula, dagg
            )
            static_rows.append(row)
            say(
                f"  {variant_id} {outcome} {fe_spec}: "
                f"b={row['coefficient']:+.4f} wild={row['p_wild']:.4f}"
            )

static = pd.DataFrame(static_rows)
static.to_csv(
    os.path.join(OUT, "validation_static.csv"), index=False, encoding="utf-8-sig"
)


# ---------------------------------------------------------------------------
# Event-study gates for balanced relational-cause flow.
# ---------------------------------------------------------------------------
def event_study(variant_id: str, label: str, saturated: bool) -> pd.DataFrame:
    d = add_design(flow, variant_id)
    launch = pd.Period(POST0, freq="M").ordinal
    d["cal_time"] = pd.PeriodIndex(d["month"], freq="M").astype(int) - launch
    targets, controls, lookup = [], [], {}
    for lo, hi in CAL_BINS:
        if (lo, hi) == CAL_REF:
            continue
        tag = f"m{abs(lo)}_m{abs(hi)}" if hi < 0 else f"p{lo}_p{hi}"
        indicator = d["cal_time"].between(lo, hi).astype(float)
        target = f"HT_{tag}"
        d[target] = indicator * d["treat"] * d["H"]
        d[f"H_{tag}"] = indicator * d["H"]
        targets.append(target)
        controls.append(f"H_{tag}")
        if not saturated:
            d[f"T_{tag}"] = indicator * d["treat"]
            controls.append(f"T_{tag}")
        lookup[target] = (lo, hi)
    rhs = " + ".join(targets + controls)
    if saturated:
        formula = f"y ~ {rhs} | pref_cause + prov_month + cause_month"
        fe_label = "province_month_saturated"
    else:
        formula = f"y ~ {rhs} | pref_cause + month"
        fe_label = "baseline"
    model = pf.feols(formula, data=d, vcov={"CRV1": "prov_id"})
    names = list(model.coef().index)
    positions = [names.index(term) for term in targets]
    beta = model.coef().values[positions]
    vcov = model._vcov[np.ix_(positions, positions)]
    pre_positions = [
        i for i, term in enumerate(targets) if lookup[term][1] < 0
    ]
    bpre = beta[pre_positions]
    vpre = vcov[np.ix_(pre_positions, pre_positions)]
    joint = float(
        stats.chi2.sf(
            float(bpre.T @ np.linalg.pinv(vpre) @ bpre), len(pre_positions)
        )
    )
    rows = []
    for i, term in enumerate(targets):
        lo, hi = lookup[term]
        se = float(np.sqrt(vcov[i, i]))
        rows.append(
            {
                "variant_id": variant_id,
                "label": label,
                "fixed_effects": fe_label,
                "bin_start": lo,
                "bin_end": hi,
                "reference_bin": "[-6,-1]",
                "coefficient": float(beta[i]),
                "se_crv1": se,
                "p_wild": float(wild_score_p(formula, d, term)),
                "joint_lead_p_chi2": joint,
                "n_obs": int(model._N),
                "province_clusters": int(d["prov_id"].nunique()),
                "formula": formula,
            }
        )
    return pd.DataFrame(rows)


say("\n=== Calendar event-study gates ===")
event_frames = []
for variant_id, label in FOCUS.items():
    for saturated in (False, True):
        es = event_study(variant_id, label, saturated)
        event_frames.append(es)
        say(
            f"  {variant_id} {es['fixed_effects'].iloc[0]}: "
            f"leads=({es.iloc[0]['coefficient']:+.3f},"
            f"{es.iloc[1]['coefficient']:+.3f}) "
            f"joint={es['joint_lead_p_chi2'].iloc[0]:.3f}; "
            f"post={es.iloc[-1]['coefficient']:+.3f} "
            f"wild={es.iloc[-1]['p_wild']:.3f}"
        )
events = pd.concat(event_frames, ignore_index=True)
events.to_csv(
    os.path.join(OUT, "flow_eventstudy.csv"), index=False, encoding="utf-8-sig"
)


# ---------------------------------------------------------------------------
# Leave-one-province-out coefficient stability.
# ---------------------------------------------------------------------------
say("\n=== Leave-one-province-out flow stability ===")
lopo_rows = []
for variant_id, label in FOCUS.items():
    d = add_design(flow, variant_id)
    formulas = {
        "baseline": "y ~ pth + ph + pt | pref_cause + month",
        "province_month_saturated": (
            "y ~ pth + ph | pref_cause + prov_month + cause_month"
        ),
    }
    for fe_spec, formula in formulas.items():
        for province in sorted(d["province"].unique()):
            sample = d[d["province"].ne(province)]
            model = pf.feols(formula, data=sample, vcov="iid")
            lopo_rows.append(
                {
                    "variant_id": variant_id,
                    "label": label,
                    "fixed_effects": fe_spec,
                    "dropped_province": province,
                    "coefficient": float(model.coef()["pth"]),
                    "n_obs": int(model._N),
                }
            )
        block = [
            row
            for row in lopo_rows
            if row["variant_id"] == variant_id and row["fixed_effects"] == fe_spec
        ]
        values = np.array([row["coefficient"] for row in block])
        say(
            f"  {variant_id} {fe_spec}: range "
            f"[{values.min():+.4f},{values.max():+.4f}], "
            f"positive {int((values > 0).sum())}/{len(values)}"
        )
lopo = pd.DataFrame(lopo_rows)
lopo.to_csv(os.path.join(OUT, "flow_lopo.csv"), index=False, encoding="utf-8-sig")


# ---------------------------------------------------------------------------
# Wave-label permutation diagnostics.
# ---------------------------------------------------------------------------
def apply_labels(
    base: pd.DataFrame, treatment_map: dict[str, int]
) -> pd.DataFrame:
    d = base.copy()
    d["treat"] = d["province"].map(treatment_map).astype(int)
    d["pt"] = d["postc"] * d["treat"]
    d["pth"] = d["pt"] * d["H"]
    return d


say("\n=== Wave-label permutation diagnostics ===")
permutation_rows = []
for variant_id, label in FOCUS.items():
    d = add_design(flow, variant_id)
    formula = "y ~ pth + ph + pt | pref_cause + month"
    province_schedule = d[["province", "inspection_round"]].drop_duplicates()
    observed_map = dict(
        zip(
            province_schedule["province"],
            (province_schedule["inspection_round"] == 1).astype(int),
        )
    )
    observed = apply_labels(d, observed_map)
    fit = pf.feols(formula, data=observed, vcov={"CRV1": "prov_id"})
    beta = float(fit.coef()["pth"])
    provinces = np.array(sorted(observed_map))
    n_treated = int(sum(observed_map.values()))
    rng = np.random.default_rng(SEED)
    permuted = np.empty(REPS)
    for draw in range(REPS):
        labels = np.zeros(len(provinces), dtype=int)
        labels[rng.choice(len(provinces), n_treated, replace=False)] = 1
        sample = apply_labels(d, dict(zip(provinces, labels)))
        permuted[draw] = float(
            pf.feols(formula, data=sample, vcov="iid").coef()["pth"]
        )
        if (draw + 1) % 200 == 0:
            say(f"  {variant_id}: {draw + 1}/{REPS}")
    p_two = float((1 + np.sum(np.abs(permuted) >= abs(beta))) / (REPS + 1))
    p_one = float((1 + np.sum(permuted >= beta)) / (REPS + 1))
    permutation_rows.append(
        {
            "variant_id": variant_id,
            "label": label,
            "estimate": beta,
            "p_wave_permutation_two_sided": p_two,
            "p_wave_permutation_one_sided": p_one,
            "permutation_mean": float(permuted.mean()),
            "permutation_sd": float(permuted.std(ddof=1)),
            "permutation_q025": float(np.quantile(permuted, 0.025)),
            "permutation_q975": float(np.quantile(permuted, 0.975)),
            "permutation_reps": REPS,
            "seed": SEED,
            "province_clusters": len(provinces),
            "treated_provinces": n_treated,
            "n_obs": int(fit._N),
            "formula": formula,
        }
    )
    say(
        f"  {variant_id}: beta={beta:+.4f}, two-sided={p_two:.3f}, "
        f"one-sided={p_one:.3f}, permutation mean={permuted.mean():+.4f}"
    )
permutation = pd.DataFrame(permutation_rows)
permutation.to_csv(
    os.path.join(OUT, "flow_wave_permutation.csv"),
    index=False,
    encoding="utf-8-sig",
)

with open(os.path.join(OUT, "validation_log.txt"), "w", encoding="utf-8") as handle:
    handle.write("\n".join(LOG))

say("\nDONE ->", OUT)

# -*- coding: utf-8 -*-
"""Transparent exploratory grid under province-by-time fixed effects.

The grid is frozen before estimation and changes only pre-listed outcome
representations, aggregation, balanced-zero support, pre-period length, or
pre-existing cause disaggregation.  Treatment is always first-wave versus
not-yet-treated, the post date is always 2018-09 (2018Q4 for quarterly data),
the end date is always 2019-03, and exposure is always exposure_v2_z.

Outputs:
  output/ext2124/civil_exploratory_manifest.csv
  output/ext2124/civil_exploratory_all.csv
  output/ext2124/civil_exploratory_significant.csv
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
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import pyfixest as pf

from _wild import wild_score_p


DATA = str(_REP_PROJECT / "data")
OUT = str(_REP_PROJECT / "output" / "ext2124")
END = "2019-03"
POST0 = "2018-09"
ROWS = []


def month_sequence(start, end=END):
    return pd.period_range(start=start, end=end, freq="M").astype(str).tolist()


def add_design(data, time_col="month", post0=POST0):
    d = data.copy()
    d["treat"] = (d["inspection_round"] == 1).astype(int)
    d["postc"] = (d[time_col] >= post0).astype(int)
    d["prov_id"] = pd.factorize(d["province"])[0]
    d["pth"] = d["postc"] * d["treat"] * d["exposure_v2_z"]
    d["ph"] = d["postc"] * d["exposure_v2_z"]
    d["prov_time"] = d["province"] + "_" + d[time_col]
    return d


def bh_adjust(series):
    p = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    out = np.full(len(p), np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    if not len(valid):
        return out
    order = valid[np.argsort(p[valid])]
    ranked = p[order] * len(valid) / np.arange(1, len(valid) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out[order] = np.minimum(ranked, 1.0)
    return out


def fit(spec, data):
    formula = f"{spec['outcome_col']} ~ pth + ph | {spec['fixed_effects_formula']}"
    base = {
        **spec,
        "formula": formula,
        "coefficient": np.nan,
        "std_error_crv1": np.nan,
        "p_crv1": np.nan,
        "std_error_crv3": np.nan,
        "p_crv3": np.nan,
        "p_wild": np.nan,
        "n_obs": np.nan,
        "province_clusters": int(data["prov_id"].nunique()),
        "status": "failed",
        "error": "",
    }
    try:
        if spec["model"] == "PPML":
            model = pf.fepois(formula, data=data, vcov={"CRV1": "prov_id"})
            base.update(
                coefficient=float(model.coef()["pth"]),
                std_error_crv1=float(model.se()["pth"]),
                p_crv1=float(model.pvalue()["pth"]),
                n_obs=int(model._N),
                status="ok",
            )
        else:
            crv1 = pf.feols(formula, data=data, vcov={"CRV1": "prov_id"})
            crv3 = pf.feols(formula, data=data, vcov={"CRV3": "prov_id"})
            p_wild = wild_score_p(formula, data, "pth")
            base.update(
                coefficient=float(crv1.coef()["pth"]),
                std_error_crv1=float(crv1.se()["pth"]),
                p_crv1=float(crv1.pvalue()["pth"]),
                std_error_crv3=float(crv3.se()["pth"]),
                p_crv3=float(crv3.pvalue()["pth"]),
                p_wild=float(p_wild),
                n_obs=int(crv1._N),
                status="ok",
            )
    except Exception as exc:
        base["error"] = f"{type(exc).__name__}: {exc}"
    ROWS.append(base)
    if base["status"] == "ok":
        print(
            f"{spec['spec_id']:16s} {spec['family']:23s} "
            f"b={base['coefficient']:+.6f} p1={base['p_crv1']:.4f} "
            f"p3={base['p_crv3']:.4f} wild={base['p_wild']:.4f} "
            f"N={int(base['n_obs']):,}",
            flush=True,
        )
    else:
        print(f"{spec['spec_id']:16s} FAILED {base['error']}", flush=True)


# ---------------------------------------------------------------------------
# Build the fixed datasets before creating or estimating the manifest.
# ---------------------------------------------------------------------------
raw = pd.read_parquet(f"{DATA}/civil_panel.parquet")
raw["month"] = raw["jmonth"].astype(str).str[:7]
rel = raw[raw["cause_family"] == "relational"].copy()
schedule = (
    pd.read_parquet(f"{DATA}/panel_month.parquet")[["province", "inspection_round"]]
    .drop_duplicates()
)
exposure = pd.read_parquet(f"{DATA}/exposure_v2.parquet")[[
    "prefecture_code", "exposure_v2_z"
]]

pre = rel[(rel["month"] >= "2014-01") & (rel["month"] <= "2017-12")].copy()
support = pre[["prefecture_code", "province", "cause"]].drop_duplicates()
pref_support = support[["prefecture_code", "province"]].drop_duplicates()
cause_caps99 = pre.groupby("cause")["n_cases"].quantile(0.99)
cause_caps995 = pre.groupby("cause")["n_cases"].quantile(0.995)


def prepare_positive(start):
    d = rel[(rel["month"] >= start) & (rel["month"] <= END)].copy()
    d = d.merge(schedule, on="province", how="left")
    d = d.dropna(subset=["inspection_round", "exposure_v2_z", "n_cases"])
    d["n"] = d["n_cases"].astype(float)
    d["n_w99"] = np.minimum(d["n"], d["cause"].map(cause_caps99))
    d["n_w995"] = np.minimum(d["n"], d["cause"].map(cause_caps995))
    d["pref_cause"] = d["prefecture_code"] + "_" + d["cause"]
    d["cause_time"] = d["cause"] + "_" + d["month"]
    return add_design(d)


def prepare_balanced(start):
    months = pd.DataFrame({"month": month_sequence(start)})
    grid = support.merge(months, how="cross")
    counts = rel[(rel["month"] >= start) & (rel["month"] <= END)][
        ["prefecture_code", "province", "cause", "month", "n_cases"]
    ]
    d = grid.merge(
        counts, on=["prefecture_code", "province", "cause", "month"], how="left"
    )
    d["n"] = d["n_cases"].fillna(0).astype(float)
    d = d.merge(schedule, on="province").merge(exposure, on="prefecture_code")
    d = d.dropna(subset=["inspection_round", "exposure_v2_z"])
    d["n_w99"] = np.minimum(d["n"], d["cause"].map(cause_caps99))
    d["n_w995"] = np.minimum(d["n"], d["cause"].map(cause_caps995))
    d["pref_cause"] = d["prefecture_code"] + "_" + d["cause"]
    d["cause_time"] = d["cause"] + "_" + d["month"]
    return add_design(d)


def prepare_aggregate(start):
    months = pd.DataFrame({"month": month_sequence(start)})
    grid = pref_support.merge(months, how="cross")
    counts = (
        rel[(rel["month"] >= start) & (rel["month"] <= END)]
        .groupby(["prefecture_code", "province", "month"], as_index=False)["n_cases"]
        .sum()
    )
    d = grid.merge(counts, on=["prefecture_code", "province", "month"], how="left")
    d["n"] = d["n_cases"].fillna(0).astype(float)
    d = d.merge(schedule, on="province").merge(exposure, on="prefecture_code")
    d = d.dropna(subset=["inspection_round", "exposure_v2_z"])
    pre_total = (
        pre.groupby(["prefecture_code", "province", "month"], as_index=False)["n_cases"]
        .sum()["n_cases"]
    )
    d["n_w99"] = d["n"].clip(upper=float(pre_total.quantile(0.99)))
    d["n_w995"] = d["n"].clip(upper=float(pre_total.quantile(0.995)))
    return add_design(d)


datasets = {}
for start in ("2014-01", "2015-01", "2016-01", "2017-01"):
    datasets[f"positive_{start}"] = prepare_positive(start)
    datasets[f"balanced_{start}"] = prepare_balanced(start)
datasets["aggregate_2017-01"] = prepare_aggregate("2017-01")

# Quarterly aggregation is defined from the same balanced prefecture-month total.
quarter = datasets["aggregate_2017-01"].copy()
quarter["quarter"] = pd.PeriodIndex(quarter["month"], freq="M").asfreq("Q").astype(str)
quarter = (
    quarter.groupby(
        ["prefecture_code", "province", "quarter", "inspection_round", "exposure_v2_z"],
        as_index=False,
    )["n"]
    .sum()
)
quarter = add_design(quarter, time_col="quarter", post0="2018Q4")
datasets["aggregate_quarter"] = quarter

# Add deterministic transformations to every dataset.
for key, d in datasets.items():
    d["y_asinh"] = np.arcsinh(d["n"])
    d["y_log1p"] = np.log1p(d["n"])
    d["y_sqrt"] = np.sqrt(d["n"])
    d["y_level"] = d["n"]
    d["y_any"] = (d["n"] > 0).astype(float)
    if (d["n"] > 0).all():
        d["y_log"] = np.log(d["n"])
    if "n_w99" in d:
        d["y_win99"] = d["n_w99"]
        d["y_win995"] = d["n_w995"]


# ---------------------------------------------------------------------------
# Freeze the complete manifest before the first regression.
# ---------------------------------------------------------------------------
SPECS = []


def add_spec(spec_id, family, dataset, outcome, model, unit, interpretation,
             fixed_effects_formula, fixed_effects_label):
    SPECS.append({
        "spec_id": spec_id,
        "family": family,
        "dataset": dataset,
        "outcome_col": outcome,
        "model": model,
        "unit": unit,
        "interpretation": interpretation,
        "sample_rule": dataset,
        "fixed_effects_formula": fixed_effects_formula,
        "fixed_effects": fixed_effects_label,
        "treatment": "wave1_x_post2018-09_x_exposure_v2_z",
        "end_date": END,
    })


cell_fe = "pref_cause + prov_time + cause_time"
cell_fe_label = "prefecture_x_cause + province_x_month + cause_x_month"
for sid, outcome, model, label in (
    ("P_ASINH", "y_asinh", "OLS", "asinh positive-cell count"),
    ("P_LOG1P", "y_log1p", "OLS", "log(1+count) positive-cell"),
    ("P_LOG", "y_log", "OLS", "log(count) positive-cell"),
    ("P_SQRT", "y_sqrt", "OLS", "sqrt(count) positive-cell"),
    ("P_LEVEL", "y_level", "OLS", "raw positive-cell count"),
    ("P_WIN99", "y_win99", "OLS", "raw count capped at preperiod cause p99"),
    ("P_WIN995", "y_win995", "OLS", "raw count capped at preperiod cause p99.5"),
    ("P_PPML", "n", "PPML", "PPML positive-cell count"),
):
    add_spec(sid, "positive_transform", "positive_2017-01", outcome, model,
             "prefecture_cause_month", label, cell_fe, cell_fe_label)

for sid, outcome, model, label in (
    ("B_ASINH", "y_asinh", "OLS", "asinh balanced count"),
    ("B_LOG1P", "y_log1p", "OLS", "log(1+count) balanced count"),
    ("B_SQRT", "y_sqrt", "OLS", "sqrt balanced count"),
    ("B_LEVEL", "y_level", "OLS", "raw balanced count"),
    ("B_ANY", "y_any", "OLS", "positive-cell indicator"),
    ("B_WIN99", "y_win99", "OLS", "balanced count capped at preperiod cause p99"),
    ("B_WIN995", "y_win995", "OLS", "balanced count capped at preperiod cause p99.5"),
    ("B_PPML", "n", "PPML", "PPML balanced count"),
):
    add_spec(sid, "balanced_transform", "balanced_2017-01", outcome, model,
             "prefecture_cause_month", label, cell_fe, cell_fe_label)

agg_fe = "prefecture_code + prov_time"
agg_fe_label = "prefecture + province_x_month"
for sid, outcome, model, label in (
    ("A_ASINH", "y_asinh", "OLS", "asinh total relational count"),
    ("A_LOG1P", "y_log1p", "OLS", "log(1+total relational count)"),
    ("A_SQRT", "y_sqrt", "OLS", "sqrt total relational count"),
    ("A_LEVEL", "y_level", "OLS", "raw total relational count"),
    ("A_WIN99", "y_win99", "OLS", "total count capped at preperiod p99"),
    ("A_WIN995", "y_win995", "OLS", "total count capped at preperiod p99.5"),
    ("A_PPML", "n", "PPML", "PPML total relational count"),
):
    add_spec(sid, "aggregate_transform", "aggregate_2017-01", outcome, model,
             "prefecture_month", label, agg_fe, agg_fe_label)

# Disaggregate only along the four cause definitions already used by the paper.
balanced17 = datasets["balanced_2017-01"]
cause_order = (
    pre.groupby("cause")["n_cases"].sum().sort_values(ascending=False).index.tolist()
)
for idx, cause in enumerate(cause_order, start=1):
    key = f"cause_{idx}"
    datasets[key] = balanced17[balanced17["cause"] == cause].copy()
    add_spec(
        f"K{idx}_ASINH", "cause_disaggregation", key, "y_asinh", "OLS",
        "prefecture_month", f"asinh count; cause={cause}", agg_fe, agg_fe_label,
    )
    add_spec(
        f"K{idx}_PPML", "cause_disaggregation", key, "n", "PPML",
        "prefecture_month", f"PPML count; cause={cause}", agg_fe, agg_fe_label,
    )

# Longer pre-period starts; the post definition and clean-window end never move.
for start in ("2014-01", "2015-01", "2016-01"):
    suffix = start[:4]
    add_spec(
        f"WP_{suffix}", "positive_preperiod", f"positive_{start}", "y_asinh", "OLS",
        "prefecture_cause_month", f"positive-cell asinh; start={start}",
        cell_fe, cell_fe_label,
    )
    add_spec(
        f"WB_{suffix}", "balanced_preperiod", f"balanced_{start}", "y_asinh", "OLS",
        "prefecture_cause_month", f"balanced asinh; start={start}",
        cell_fe, cell_fe_label,
    )

quarter_fe = "prefecture_code + prov_time"
for sid, outcome, model, label in (
    ("Q_ASINH", "y_asinh", "OLS", "quarterly asinh total count"),
    ("Q_LOG1P", "y_log1p", "OLS", "quarterly log(1+total count)"),
    ("Q_LEVEL", "y_level", "OLS", "quarterly raw total count"),
    ("Q_PPML", "n", "PPML", "quarterly PPML total count"),
):
    add_spec(sid, "quarterly_aggregate", "aggregate_quarter", outcome, model,
             "prefecture_quarter", label, quarter_fe,
             "prefecture + province_x_quarter")

os.makedirs(OUT, exist_ok=True)
manifest_path = f"{OUT}/civil_exploratory_manifest.csv"
pd.DataFrame(SPECS).to_csv(manifest_path, index=False)
print(f"FROZEN MANIFEST: {len(SPECS)} specifications written to {manifest_path}", flush=True)


# ---------------------------------------------------------------------------
# Estimate every frozen specification; no result-dependent branching.
# ---------------------------------------------------------------------------
warnings.filterwarnings("ignore", message=".*singleton fixed effect.*")
for spec in SPECS:
    fit(spec, datasets[spec["dataset"]])

result = pd.DataFrame(ROWS)
result["primary_inference"] = np.where(result["model"].eq("OLS"), "wild_score", "CRV1")
result["primary_p"] = np.where(result["model"].eq("OLS"), result["p_wild"], result["p_crv1"])
for pcol in ("p_crv1", "p_crv3", "p_wild", "primary_p"):
    result[f"q_all_{pcol}"] = bh_adjust(result[pcol])
    result[f"q_family_{pcol}"] = np.nan
    for family, idx in result.groupby("family").groups.items():
        result.loc[idx, f"q_family_{pcol}"] = bh_adjust(result.loc[idx, pcol])

result["nominal_crv1_5"] = result["p_crv1"] < 0.05
result["nominal_crv3_5"] = result["p_crv3"] < 0.05
result["nominal_wild_5"] = result["p_wild"] < 0.05
result["primary_nominal_5"] = result["primary_p"] < 0.05
result["primary_family_bh_5"] = result["q_family_primary_p"] < 0.05
result["primary_all_bh_5"] = result["q_all_primary_p"] < 0.05

all_path = f"{OUT}/civil_exploratory_all.csv"
sig_path = f"{OUT}/civil_exploratory_significant.csv"
result.to_csv(all_path, index=False)
sig = result[
    result[["nominal_crv1_5", "nominal_crv3_5", "nominal_wild_5",
            "primary_family_bh_5", "primary_all_bh_5"]].any(axis=1)
].copy()
sig.to_csv(sig_path, index=False)

print(f"written all: {all_path}", flush=True)
print(f"written flagged: {sig_path}", flush=True)
print(
    f"SUMMARY total={len(result)} failed={(result.status != 'ok').sum()} "
    f"CRV1<.05={result.nominal_crv1_5.sum()} "
    f"CRV3<.05={result.nominal_crv3_5.sum()} "
    f"wild<.05={result.nominal_wild_5.sum()} "
    f"primary-family-BH<.05={result.primary_family_bh_5.sum()} "
    f"primary-all-BH<.05={result.primary_all_bh_5.sum()}",
    flush=True,
)

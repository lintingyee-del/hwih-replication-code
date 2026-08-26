# -*- coding: utf-8 -*-
"""Reconcile the civil estimands on common support.

This is a targeted adjudication exercise, not an open-ended robustness grid.  It
answers four questions raised by the paper's current presentation:

1. Does the clean-window cause-cell estimate survive explicit zero cells?
2. On one balanced prefecture-month panel, what are the separate relational and
   traffic responses, and therefore their exact difference?
3. Does the acquaintance-minus-stranger contrast survive explicit zero cells
   and exclusion of cases whose relationship flag is missing?
4. What do clean-window calendar-time coefficients look like for relational
   totals, traffic totals, and their exact difference?

Outputs:
  output/ext2124/civil_estimand_reconciliation.csv
  output/ext2124/civil_clean_calendar_eventstudy.csv
  output/ext2124/civil_estimand_reconciliation_support.csv
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
from scipy import stats

from _wild import wild_score_p


DATA = str(_REP_PROJECT / "data")
OUT = str(_REP_PROJECT / "output" / "ext2124")
START = "2017-01"
END = "2019-03"
POST0 = "2018-09"
PRE_SUPPORT_START = "2014-01"
PRE_SUPPORT_END = "2017-12"
ROWS = []
ES_ROWS = []
SUPPORT_ROWS = []


def month_sequence(start=START, end=END):
    return pd.period_range(start, end, freq="M").astype(str).tolist()


schedule = (
    pd.read_parquet(f"{DATA}/panel_month.parquet")[["province", "inspection_round"]]
    .drop_duplicates()
)
exposure = pd.read_parquet(f"{DATA}/exposure_v2.parquet")[[
    "prefecture_code", "province", "exposure_v2_z"
]].drop_duplicates()


def add_design(data):
    d = data.copy()
    d["treat"] = (d["inspection_round"] == 1).astype(int)
    d["postc"] = (d["month"] >= POST0).astype(int)
    d["pt"] = d["postc"] * d["treat"]
    d["pth"] = d["pt"] * d["exposure_v2_z"]
    d["ph"] = d["postc"] * d["exposure_v2_z"]
    d["prov_id"] = pd.factorize(d["province"])[0]
    d["prov_month"] = d["province"] + "_" + d["month"]
    return d


def fit(spec_id, family, outcome, formula, data, support, note):
    m1 = pf.feols(formula, data=data, vcov={"CRV1": "prov_id"})
    m3 = pf.feols(formula, data=data, vcov={"CRV3": "prov_id"})
    pw = wild_score_p(formula, data, "pth")
    beta = float(m1.coef()["pth"])
    row = {
        "spec_id": spec_id,
        "family": family,
        "outcome": outcome,
        "support": support,
        "formula": formula,
        "coefficient": beta,
        "std_error_crv1": float(m1.se()["pth"]),
        "p_crv1": float(m1.pvalue()["pth"]),
        "std_error_crv3": float(m3.se()["pth"]),
        "p_crv3": float(m3.pvalue()["pth"]),
        "p_wild": float(pw),
        "n_obs": int(m1._N),
        "province_clusters": int(data["prov_id"].nunique()),
        "exp_beta_minus_one": float(np.exp(beta) - 1),
        "note": note,
    }
    ROWS.append(row)
    print(
        f"{spec_id:18s} b={beta:+.6f} se={row['std_error_crv1']:.6f} "
        f"p1={row['p_crv1']:.4f} p3={row['p_crv3']:.4f} "
        f"wild={pw:.4f} N={row['n_obs']:,}",
        flush=True,
    )


def fit_weighted(spec_id, family, outcome, formula, data, weight, support, note):
    m1 = pf.feols(formula, data=data, weights=weight, vcov={"CRV1": "prov_id"})
    m3 = pf.feols(formula, data=data, weights=weight, vcov={"CRV3": "prov_id"})
    pw = wild_score_p(formula, data, "pth", weights=weight)
    beta = float(m1.coef()["pth"])
    row = {
        "spec_id": spec_id,
        "family": family,
        "outcome": outcome,
        "support": support,
        "formula": formula,
        "coefficient": beta,
        "std_error_crv1": float(m1.se()["pth"]),
        "p_crv1": float(m1.pvalue()["pth"]),
        "std_error_crv3": float(m3.se()["pth"]),
        "p_crv3": float(m3.pvalue()["pth"]),
        "p_wild": float(pw),
        "n_obs": int(m1._N),
        "province_clusters": int(data["prov_id"].nunique()),
        "exp_beta_minus_one": np.nan,
        "note": note,
    }
    ROWS.append(row)
    print(
        f"{spec_id:18s} b={beta:+.6f} se={row['std_error_crv1']:.6f} "
        f"p1={row['p_crv1']:.4f} p3={row['p_crv3']:.4f} "
        f"wild={pw:.4f} N={row['n_obs']:,}",
        flush=True,
    )


def record_support(panel, name, count_cols):
    row = {
        "panel": name,
        "rows": len(panel),
        "prefectures": panel["prefecture_code"].nunique(),
        "provinces": panel["province"].nunique(),
        "months": panel["month"].nunique(),
    }
    for col in count_cols:
        row[f"zero_share_{col}"] = float((panel[col] == 0).mean())
        row[f"sum_{col}"] = float(panel[col].sum())
    SUPPORT_ROWS.append(row)


# ---------------------------------------------------------------------------
# A. Four relational cause cells: published positive support versus a balanced
#    grid fixed using the entirely pre-campaign 2014-2017 support.
# ---------------------------------------------------------------------------
civil_panel = pd.read_parquet(f"{DATA}/civil_panel.parquet")
civil_panel["month"] = civil_panel["jmonth"].astype(str).str[:7]
rel = civil_panel[civil_panel["cause_family"].eq("relational")].copy()

pre_rel = rel[
    rel["month"].between(PRE_SUPPORT_START, PRE_SUPPORT_END)
][["prefecture_code", "province", "cause"]].drop_duplicates()
months = pd.DataFrame({"month": month_sequence()})
rel_counts = rel[rel["month"].between(START, END)][[
    "prefecture_code", "province", "cause", "month", "n_cases"
]]

positive_cells = (
    rel_counts.merge(schedule, on="province")
    .merge(exposure[["prefecture_code", "exposure_v2_z"]], on="prefecture_code")
)
positive_cells["n"] = positive_cells["n_cases"].astype(float)
positive_cells["y"] = np.arcsinh(positive_cells["n"])
positive_cells["pref_cause"] = (
    positive_cells["prefecture_code"] + "_" + positive_cells["cause"]
)
positive_cells["cause_month"] = positive_cells["cause"] + "_" + positive_cells["month"]
positive_cells = add_design(positive_cells)

balanced_cells = pre_rel.merge(months, how="cross").merge(
    rel_counts,
    on=["prefecture_code", "province", "cause", "month"],
    how="left",
)
balanced_cells["n"] = balanced_cells["n_cases"].fillna(0).astype(float)
balanced_cells = (
    balanced_cells.merge(schedule, on="province")
    .merge(exposure[["prefecture_code", "exposure_v2_z"]], on="prefecture_code")
)
balanced_cells["y"] = np.arcsinh(balanced_cells["n"])
balanced_cells["pref_cause"] = (
    balanced_cells["prefecture_code"] + "_" + balanced_cells["cause"]
)
balanced_cells["cause_month"] = balanced_cells["cause"] + "_" + balanced_cells["month"]
balanced_cells = add_design(balanced_cells)
record_support(balanced_cells, "relational_cause_cells", ["n"])

fit(
    "CC_POS_LOW", "cause_cell", "asinh relational cause count",
    "y ~ pth + ph + pt | pref_cause + month", positive_cells,
    "observed positive cells only",
    "Published clean-window equation reproduced on its original support.",
)
fit(
    "CC_BAL_LOW", "cause_cell", "asinh relational cause count",
    "y ~ pth + ph + pt | pref_cause + month", balanced_cells,
    "2014-2017 prefecture-cause support crossed with every clean-window month",
    "Same equation after restoring zero-count cells.",
)
fit(
    "CC_BAL_SAT", "cause_cell", "asinh relational cause count",
    "y ~ pth + ph | pref_cause + prov_month + cause_month", balanced_cells,
    "2014-2017 prefecture-cause support crossed with every clean-window month",
    "Balanced cells with province-by-month and cause-by-month fixed effects.",
)


# ---------------------------------------------------------------------------
# B. Relational and traffic totals on exactly the same balanced pref-month grid.
#    OLS linearity makes the gap coefficient exactly relational minus traffic.
# ---------------------------------------------------------------------------
rt = civil_panel[civil_panel["cause_family"].isin(["relational", "placebo"])].copy()
rt["group"] = np.where(rt["cause_family"].eq("relational"), "relational", "traffic")
pre_rt_pref = (
    rt[rt["month"].between(PRE_SUPPORT_START, PRE_SUPPORT_END)][
        ["prefecture_code", "province"]
    ].drop_duplicates()
    .merge(exposure[["prefecture_code", "province"]],
           on=["prefecture_code", "province"], how="inner")
)
rt_counts = (
    rt[rt["month"].between(START, END)]
    .groupby(["prefecture_code", "province", "month", "group"], as_index=False)["n_cases"]
    .sum()
    .pivot(index=["prefecture_code", "province", "month"], columns="group", values="n_cases")
    .reset_index()
)
aggregate = pre_rt_pref.merge(months, how="cross").merge(
    rt_counts, on=["prefecture_code", "province", "month"], how="left"
)
for col in ("relational", "traffic"):
    aggregate[col] = aggregate[col].fillna(0).astype(float)
aggregate = (
    aggregate.merge(schedule, on="province")
    .merge(exposure[["prefecture_code", "exposure_v2_z"]], on="prefecture_code")
)
aggregate["y_relational"] = np.arcsinh(aggregate["relational"])
aggregate["y_traffic"] = np.arcsinh(aggregate["traffic"])
aggregate["y_gap"] = aggregate["y_relational"] - aggregate["y_traffic"]
aggregate = add_design(aggregate)
record_support(aggregate, "relational_traffic_prefecture_month", ["relational", "traffic"])

for suffix, formula in (
    ("LOW", "{y} ~ pth + ph + pt | prefecture_code + month"),
    ("SAT", "{y} ~ pth + ph | prefecture_code + prov_month"),
):
    for short, ycol in (
        ("REL", "y_relational"),
        ("TRF", "y_traffic"),
        ("GAP", "y_gap"),
    ):
        fit(
            f"AGG_{short}_{suffix}", "prefecture_month_common_support", ycol,
            formula.format(y=ycol), aggregate,
            "same balanced 2014-2017 relational-or-traffic prefecture support",
            "Separate responses and their exact within-panel difference.",
        )


# ---------------------------------------------------------------------------
# C. Acquaintance versus stranger lending.  Compare the published observed-cell
#    coding with balanced group cells, then exclude missing relationship flags.
# ---------------------------------------------------------------------------
case = pd.read_parquet(
    f"{DATA}/civil_case.parquet",
    columns=["cause", "prefecture_code", "province", "jmonth", "rel_txn"],
)
case["month"] = case["jmonth"].astype(str).str[:7]
lending = case[case["cause"].eq("民间借贷纠纷")].copy()


def make_composition(flag_rule, balance):
    if flag_rule == "missing_as_stranger":
        d = lending.copy()
        d["acq"] = d["rel_txn"].fillna(0).astype(int)
    elif flag_rule == "classified_only":
        d = lending[lending["rel_txn"].notna()].copy()
        d["acq"] = d["rel_txn"].astype(int)
    else:
        raise ValueError(flag_rule)

    support = (
        d[d["month"].between(PRE_SUPPORT_START, PRE_SUPPORT_END)][
            ["prefecture_code", "province"]
        ].drop_duplicates()
        .merge(exposure[["prefecture_code", "province"]],
               on=["prefecture_code", "province"], how="inner")
    )
    counts = (
        d[d["month"].between(START, END)]
        .groupby(["prefecture_code", "province", "month", "acq"])
        .size().rename("n").reset_index()
    )
    if balance:
        groups = pd.DataFrame({"acq": [0, 1]})
        out = support.merge(months, how="cross").merge(groups, how="cross").merge(
            counts, on=["prefecture_code", "province", "month", "acq"], how="left"
        )
        out["n"] = out["n"].fillna(0).astype(float)
    else:
        out = counts
    out = (
        out.merge(schedule, on="province")
        .merge(exposure[["prefecture_code", "exposure_v2_z"]], on="prefecture_code")
    )
    out = add_design(out)
    out["y"] = np.arcsinh(out["n"])
    out["prefA"] = out["prefecture_code"] + "_" + out["acq"].astype(str)
    out["monthA"] = out["month"] + "_" + out["acq"].astype(str)
    for term in ("pth", "ph", "pt"):
        out[f"{term}A"] = out[term] * out["acq"]
    return out


composition_specs = {
    "CMP_POS_PUB": make_composition("missing_as_stranger", False),
    "CMP_BAL_PUB": make_composition("missing_as_stranger", True),
    "CMP_BAL_CC": make_composition("classified_only", True),
}
for name, d in composition_specs.items():
    record_support(d, name, ["n"])

def fit_composition(name, d, formula, support_note, note):
    m1 = pf.feols(formula, data=d, vcov={"CRV1": "prov_id"})
    m3 = pf.feols(formula, data=d, vcov={"CRV3": "prov_id"})
    pw = wild_score_p(formula, d, "pthA")
    beta = float(m1.coef()["pthA"])
    ROWS.append({
        "spec_id": name,
        "family": "acquaintance_minus_stranger",
        "outcome": "asinh lending count; coefficient pthA",
        "support": support_note,
        "formula": formula,
        "coefficient": beta,
        "std_error_crv1": float(m1.se()["pthA"]),
        "p_crv1": float(m1.pvalue()["pthA"]),
        "std_error_crv3": float(m3.se()["pthA"]),
        "p_crv3": float(m3.pvalue()["pthA"]),
        "p_wild": float(pw),
        "n_obs": int(m1._N),
        "province_clusters": int(d["prov_id"].nunique()),
        "exp_beta_minus_one": float(np.exp(beta) - 1),
        "note": note,
    })
    print(
        f"{name:18s} b={beta:+.6f} se={float(m1.se()['pthA']):.6f} "
        f"p1={float(m1.pvalue()['pthA']):.4f} p3={float(m3.pvalue()['pthA']):.4f} "
        f"wild={pw:.4f} N={int(m1._N):,}",
        flush=True,
    )


# The generic fit function targets pth, so estimate the pooled composition
# equations explicitly for their pthA coefficient.  month-by-group fixed effects
# absorb the group-specific post indicator, which is therefore not written as a
# redundant regressor.
for name, d in composition_specs.items():
    flag = "classified relationship flags only" if name.endswith("CC") else "missing flag coded as stranger"
    balanced = "balanced prefecture-month-group cells" if "BAL" in name else "observed positive group cells"
    fit_composition(
        name,
        d,
        "y ~ pthA + phA + ptA + pth + ph + pt | prefA + monthA",
        f"{balanced}; {flag}",
        "Tests zero-cell selection and treatment of unclassified relationship flags.",
    )

for name in ("CMP_BAL_PUB", "CMP_BAL_CC"):
    d = composition_specs[name]
    flag = "classified relationship flags only" if name.endswith("CC") else "missing flag coded as stranger"
    fit_composition(
        f"{name}_SAT",
        d,
        "y ~ pthA + phA + ptA + pth + ph | prefA + prov_month + monthA",
        f"balanced prefecture-month-group cells; {flag}",
        "Province-by-month saturated composition analogue on balanced cells.",
    )


# Availability of the relationship flag is text availability, not a negative
# relationship classification: regexp_matches(NULL, ...) returns NULL.  Test
# whether that availability itself follows the clean-window treatment gradient.
availability = (
    lending[lending["month"].between(START, END)]
    .assign(flag_available=lambda x: x["rel_txn"].notna().astype(int))
    .groupby(["prefecture_code", "province", "month"], as_index=False)
    .agg(n_total=("flag_available", "size"), n_available=("flag_available", "sum"))
)
availability["share_available"] = availability["n_available"] / availability["n_total"]
availability = (
    availability.merge(schedule, on="province")
    .merge(exposure[["prefecture_code", "exposure_v2_z"]], on="prefecture_code")
)
availability = add_design(availability)
fit_weighted(
    "CMP_FLAG_AVAIL", "relationship_flag_availability", "share with nonmissing relationship flag",
    "share_available ~ pth + ph + pt | prefecture_code + month",
    availability, "n_total", "prefecture-months with at least one lending judgment",
    "Case-weighted test for differential availability of the text-based relationship flag.",
)
fit_weighted(
    "CMP_FLAG_AVAIL_SAT", "relationship_flag_availability", "share with nonmissing relationship flag",
    "share_available ~ pth + ph | prefecture_code + prov_month",
    availability, "n_total", "prefecture-months with at least one lending judgment",
    "Province-by-month saturated test for differential flag availability.",
)


# ---------------------------------------------------------------------------
# D. Filing-clock reconciliation.  The published construction subtracts a full
# filing date from a judgment date already truncated to the first of its month;
# this mechanically drops almost every same-month filing.  Report that support
# and a corrected 0-to-9 calendar-month lag support, then place relational and
# traffic filings on the same balanced prefecture-month grid.
# ---------------------------------------------------------------------------
case_clock = pd.read_parquet(
    f"{DATA}/civil_case.parquet",
    columns=["case_no", "cause", "cause_family", "prefecture_code", "province", "jmonth"],
)
case_clock = case_clock[case_clock["cause_family"].isin(["relational", "placebo"])].copy()
filing_dates = pd.read_parquet(f"{DATA}/civil_filing.parquet").rename(columns={"案号": "case_no"})
case_clock = case_clock.merge(filing_dates[["case_no", "filing_ymd"]], on="case_no", how="left")
case_clock["fdate"] = pd.to_datetime(case_clock["filing_ymd"], errors="coerce")
case_clock["jdate_truncated"] = pd.to_datetime(case_clock["jmonth"], errors="coerce")
case_clock["day_lag_from_truncated_jdate"] = (
    case_clock["jdate_truncated"] - case_clock["fdate"]
).dt.days
valid_dates = case_clock["fdate"].notna() & case_clock["jdate_truncated"].notna()
case_clock.loc[valid_dates, "month_lag"] = (
    pd.PeriodIndex(case_clock.loc[valid_dates, "jdate_truncated"], freq="M").astype(int)
    - pd.PeriodIndex(case_clock.loc[valid_dates, "fdate"], freq="M").astype(int)
)
case_clock["filing_month"] = case_clock["fdate"].dt.strftime("%Y-%m")
case_clock["group"] = np.where(
    case_clock["cause_family"].eq("relational"), "relational", "traffic"
)


def make_filing_aggregate(rule):
    if rule == "published_day_lag":
        keep = valid_dates & case_clock["day_lag_from_truncated_jdate"].between(0, 270)
    elif rule == "calendar_month_lag":
        keep = valid_dates & case_clock["month_lag"].between(0, 9)
    else:
        raise ValueError(rule)
    d = case_clock[keep & case_clock["filing_month"].between(START, END)].copy()
    counts = (
        d.groupby(["prefecture_code", "province", "filing_month", "group"])
        .size().rename("n").reset_index()
        .pivot(index=["prefecture_code", "province", "filing_month"], columns="group", values="n")
        .reset_index().rename(columns={"filing_month": "month"})
    )
    out = pre_rt_pref.merge(months, how="cross").merge(
        counts, on=["prefecture_code", "province", "month"], how="left"
    )
    for col in ("relational", "traffic"):
        out[col] = out[col].fillna(0).astype(float)
        out[f"y_{col}"] = np.arcsinh(out[col])
    out["y_gap"] = out["y_relational"] - out["y_traffic"]
    out = (
        out.merge(schedule, on="province")
        .merge(exposure[["prefecture_code", "exposure_v2_z"]], on="prefecture_code")
    )
    return add_design(out)


filing_panels = {
    "FDAY": make_filing_aggregate("published_day_lag"),
    "FMON": make_filing_aggregate("calendar_month_lag"),
}
for label, d in filing_panels.items():
    record_support(d, label, ["relational", "traffic"])
    support_note = (
        "published 0-270 day rule using first-of-judgment-month"
        if label == "FDAY" else "corrected 0-9 calendar-month filing-to-judgment lag"
    )
    for short, ycol in (("REL", "y_relational"), ("TRF", "y_traffic"), ("GAP", "y_gap")):
        fit(
            f"{label}_{short}_LOW", "filing_clock_common_support", ycol,
            f"{ycol} ~ pth + ph + pt | prefecture_code + month", d,
            support_note,
            "Balanced relational and traffic filing counts on a common support.",
        )


# ---------------------------------------------------------------------------
# E. Clean-window calendar event study on the common aggregate panel.
#    Reference period is the six months before September 2018.
# ---------------------------------------------------------------------------
CAL_BINS = [(-20, -13), (-12, -7), (-6, -1), (0, 6)]
CAL_REF = (-6, -1)
calendar0 = pd.Period(POST0, freq="M")
aggregate["cal_time"] = (
    pd.PeriodIndex(aggregate["month"], freq="M").astype(int) - calendar0.ordinal
)

target_terms = []
controls = []
bin_lookup = {}
for lo, hi in CAL_BINS:
    if (lo, hi) == CAL_REF:
        continue
    tag = f"m{abs(lo)}_m{abs(hi)}" if hi < 0 else f"p{lo}_p{hi}"
    indicator = aggregate["cal_time"].between(lo, hi).astype(float)
    target = f"H_T_{tag}"
    treat_term = f"T_{tag}"
    dose_term = f"H_{tag}"
    aggregate[target] = indicator * aggregate["treat"] * aggregate["exposure_v2_z"]
    aggregate[treat_term] = indicator * aggregate["treat"]
    aggregate[dose_term] = indicator * aggregate["exposure_v2_z"]
    target_terms.append(target)
    controls.extend([treat_term, dose_term])
    bin_lookup[target] = (lo, hi)

es_rhs = " + ".join(target_terms + controls)
for outcome in ("y_relational", "y_traffic", "y_gap"):
    formula = f"{outcome} ~ {es_rhs} | prefecture_code + month"
    model = pf.feols(formula, data=aggregate, vcov={"CRV1": "prov_id"})
    names = list(model.coef().index)
    idx = [names.index(term) for term in target_terms]
    beta = model.coef().values[idx]
    vcov = model._vcov[np.ix_(idx, idx)]
    # Joint CRV1 Wald test for the two pre coefficients.
    pre_idx = [i for i, term in enumerate(target_terms) if bin_lookup[term][1] < 0]
    bp = beta[pre_idx]
    vp = vcov[np.ix_(pre_idx, pre_idx)]
    wald = float(bp.T @ np.linalg.pinv(vp) @ bp)
    lead_p_chi2 = float(stats.chi2.sf(wald, len(pre_idx)))
    for i, term in enumerate(target_terms):
        lo, hi = bin_lookup[term]
        se = float(np.sqrt(vcov[i, i]))
        pw = wild_score_p(formula, aggregate, term)
        ES_ROWS.append({
            "design": "aggregate_common_support",
            "outcome": outcome,
            "bin_start": lo,
            "bin_end": hi,
            "reference_bin": "[-6,-1]",
            "coefficient": float(beta[i]),
            "std_error_crv1": se,
            "ci95_low": float(beta[i] - 1.96 * se),
            "ci95_high": float(beta[i] + 1.96 * se),
            "p_wild": float(pw),
            "joint_lead_p_chi2": lead_p_chi2,
            "n_obs": int(model._N),
            "formula": formula,
        })
        print(
            f"ES {outcome:14s} [{lo:3d},{hi:2d}] "
            f"b={float(beta[i]):+.6f} se={se:.6f} wild={pw:.4f}",
            flush=True,
        )
    print(f"ES {outcome:14s} joint-lead chi2 p={lead_p_chi2:.4f}", flush=True)


def run_cause_cell_calendar_es(data, design_label):
    d = data.copy()
    d["cal_time"] = (
        pd.PeriodIndex(d["month"], freq="M").astype(int) - calendar0.ordinal
    )
    terms = []
    other = []
    lookup = {}
    for lo, hi in CAL_BINS:
        if (lo, hi) == CAL_REF:
            continue
        tag = f"m{abs(lo)}_m{abs(hi)}" if hi < 0 else f"p{lo}_p{hi}"
        indicator = d["cal_time"].between(lo, hi).astype(float)
        target = f"H_T_{tag}"
        tterm = f"T_{tag}"
        hterm = f"H_{tag}"
        d[target] = indicator * d["treat"] * d["exposure_v2_z"]
        d[tterm] = indicator * d["treat"]
        d[hterm] = indicator * d["exposure_v2_z"]
        terms.append(target)
        other.extend([tterm, hterm])
        lookup[target] = (lo, hi)
    formula = f"y ~ {' + '.join(terms + other)} | pref_cause + month"
    model = pf.feols(formula, data=d, vcov={"CRV1": "prov_id"})
    names = list(model.coef().index)
    idx = [names.index(term) for term in terms]
    beta = model.coef().values[idx]
    vcov = model._vcov[np.ix_(idx, idx)]
    pre_idx = [i for i, term in enumerate(terms) if lookup[term][1] < 0]
    bp = beta[pre_idx]
    vp = vcov[np.ix_(pre_idx, pre_idx)]
    wald = float(bp.T @ np.linalg.pinv(vp) @ bp)
    lead_p_chi2 = float(stats.chi2.sf(wald, len(pre_idx)))
    for i, term in enumerate(terms):
        lo, hi = lookup[term]
        se = float(np.sqrt(vcov[i, i]))
        pw = wild_score_p(formula, d, term)
        ES_ROWS.append({
            "design": design_label,
            "outcome": "y_relational_cause_cell",
            "bin_start": lo,
            "bin_end": hi,
            "reference_bin": "[-6,-1]",
            "coefficient": float(beta[i]),
            "std_error_crv1": se,
            "ci95_low": float(beta[i] - 1.96 * se),
            "ci95_high": float(beta[i] + 1.96 * se),
            "p_wild": float(pw),
            "joint_lead_p_chi2": lead_p_chi2,
            "n_obs": int(model._N),
            "formula": formula,
        })
        print(
            f"ES {design_label:18s} [{lo:3d},{hi:2d}] "
            f"b={float(beta[i]):+.6f} se={se:.6f} wild={pw:.4f}",
            flush=True,
        )
    print(f"ES {design_label:18s} joint-lead chi2 p={lead_p_chi2:.4f}", flush=True)


run_cause_cell_calendar_es(positive_cells, "cause_cell_positive")
run_cause_cell_calendar_es(balanced_cells, "cause_cell_balanced")


os.makedirs(OUT, exist_ok=True)
pd.DataFrame(ROWS).to_csv(f"{OUT}/civil_estimand_reconciliation.csv", index=False)
pd.DataFrame(ES_ROWS).to_csv(f"{OUT}/civil_clean_calendar_eventstudy.csv", index=False)
pd.DataFrame(SUPPORT_ROWS).to_csv(
    f"{OUT}/civil_estimand_reconciliation_support.csv", index=False
)
print("written reconciliation outputs", flush=True)

# -*- coding: utf-8 -*-
"""Revised primary civil results used by the submission manuscript.

The script implements only the four approved revisions:

1. clean-window relational flow on a balanced prefecture-cause-month panel,
   including its same-support saturated fixed-effect analogue;
2. acquaintance-minus-stranger flow on balanced cells among cases with a
   classified relationship flag;
3. clean-window calendar dynamics for the composition contrast; and
4. clean-window calendar dynamics for balanced relational flow.

For the composition profile, the script also reports the paper's transparent
relative-magnitudes calculation: the largest absolute step across the two leads
and the omitted reference period, and the value of Mbar at which the 95 percent
robust interval for the post coefficient first includes zero.

Every generated text/CSV output is written first with a timestamp and then to a
fixed latest filename.  Figure rendering is handled by
manuscript/figures/gen_fig_civil_revised.py.
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
import shutil
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import pyfixest as pf
from scipy import stats

from _wild import wild_score_p


DATA = str(_REP_PROJECT / "data")
OUT = str(_REP_PROJECT / "output" / "ext2124")
TAB = str(_REP_PROJECT / "output" / "tables")
START, END, POST0 = "2017-01", "2019-03", "2018-09"
SUPPORT_START, SUPPORT_END = "2014-01", "2017-12"
CAL_BINS = [(-20, -13), (-12, -7), (-6, -1), (0, 6)]
CAL_REF = (-6, -1)
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
VERSIONED_OUTPUTS = os.environ.get("HWIH_REPLICATION", "0") != "1"

os.makedirs(OUT, exist_ok=True)
os.makedirs(TAB, exist_ok=True)


def write_versioned_csv(frame, directory, stem):
    latest = os.path.join(directory, f"{stem}.csv")
    if VERSIONED_OUTPUTS:
        timestamped = os.path.join(directory, f"{stem}_{STAMP}.csv")
        frame.to_csv(timestamped, index=False)
        shutil.copyfile(timestamped, latest)
        return timestamped, latest
    frame.to_csv(latest, index=False)
    return latest,


def write_versioned_text(text, directory, stem, suffix="tex"):
    latest = os.path.join(directory, f"{stem}.{suffix}")
    if VERSIONED_OUTPUTS:
        timestamped = os.path.join(directory, f"{stem}_{STAMP}.{suffix}")
        with open(timestamped, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        shutil.copyfile(timestamped, latest)
        return timestamped, latest
    with open(latest, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    return latest,


schedule = (
    pd.read_parquet(f"{DATA}/panel_month.parquet")[["province", "inspection_round"]]
    .drop_duplicates()
)
exposure = pd.read_parquet(f"{DATA}/exposure_v2.parquet")[[
    "prefecture_code", "province", "exposure_v2_z"
]].drop_duplicates()
months = pd.DataFrame({"month": pd.period_range(START, END, freq="M").astype(str)})


def add_design(data):
    d = data.copy()
    d["treat"] = (d["inspection_round"] == 1).astype(int)
    d["postc"] = (d["month"] >= POST0).astype(int)
    d["pt"] = d["postc"] * d["treat"]
    d["ph"] = d["postc"] * d["exposure_v2_z"]
    d["pth"] = d["pt"] * d["exposure_v2_z"]
    d["prov_id"] = pd.factorize(d["province"])[0]
    d["prov_month"] = d["province"] + "_" + d["month"]
    return d


def fit_target(formula, data, coefficient, weights=None):
    m1 = pf.feols(formula, data=data, weights=weights, vcov={"CRV1": "prov_id"})
    m3 = pf.feols(formula, data=data, weights=weights, vcov={"CRV3": "prov_id"})
    pw = wild_score_p(formula, data, coefficient, weights=weights)
    return {
        "coefficient": float(m1.coef()[coefficient]),
        "std_error_crv1": float(m1.se()[coefficient]),
        "p_crv1": float(m1.pvalue()[coefficient]),
        "std_error_crv3": float(m3.se()[coefficient]),
        "p_crv3": float(m3.pvalue()[coefficient]),
        "p_wild": float(pw),
        "n_obs": int(m1._N),
        "province_clusters": int(data["prov_id"].nunique()),
        "formula": formula,
    }


def simple_calendar_eventstudy(data, outcome, unit_fe, design):
    d = data.copy()
    launch = pd.Period(POST0, freq="M").ordinal
    d["cal_time"] = pd.PeriodIndex(d["month"], freq="M").astype(int) - launch
    targets, controls, lookup = [], [], {}
    for lo, hi in CAL_BINS:
        if (lo, hi) == CAL_REF:
            continue
        tag = f"m{abs(lo)}_m{abs(hi)}" if hi < 0 else f"p{lo}_p{hi}"
        indicator = d["cal_time"].between(lo, hi).astype(float)
        target = f"HT_{tag}"
        tterm = f"T_{tag}"
        hterm = f"H_{tag}"
        d[target] = indicator * d["treat"] * d["exposure_v2_z"]
        d[tterm] = indicator * d["treat"]
        d[hterm] = indicator * d["exposure_v2_z"]
        targets.append(target)
        controls.extend([tterm, hterm])
        lookup[target] = (lo, hi)
    formula = f"{outcome} ~ {' + '.join(targets + controls)} | {unit_fe} + month"
    model = pf.feols(formula, data=d, vcov={"CRV1": "prov_id"})
    names = list(model.coef().index)
    index = [names.index(term) for term in targets]
    beta = model.coef().values[index]
    vcov = model._vcov[np.ix_(index, index)]
    pre_index = [i for i, term in enumerate(targets) if lookup[term][1] < 0]
    bpre = beta[pre_index]
    vpre = vcov[np.ix_(pre_index, pre_index)]
    joint_p = float(stats.chi2.sf(float(bpre.T @ np.linalg.pinv(vpre) @ bpre), len(pre_index)))
    rows = []
    for i, term in enumerate(targets):
        lo, hi = lookup[term]
        se = float(np.sqrt(vcov[i, i]))
        rows.append({
            "design": design,
            "bin_start": lo,
            "bin_end": hi,
            "reference_bin": "[-6,-1]",
            "coefficient": float(beta[i]),
            "std_error_crv1": se,
            "ci95_low": float(beta[i] - 1.96 * se),
            "ci95_high": float(beta[i] + 1.96 * se),
            "p_wild": float(wild_score_p(formula, d, term)),
            "joint_lead_p_chi2": joint_p,
            "n_obs": int(model._N),
            "formula": formula,
        })
    return pd.DataFrame(rows)


def composition_calendar_eventstudy(data):
    d = data.copy()
    launch = pd.Period(POST0, freq="M").ordinal
    d["cal_time"] = pd.PeriodIndex(d["month"], freq="M").astype(int) - launch
    targets, controls, lookup = [], [], {}
    for lo, hi in CAL_BINS:
        if (lo, hi) == CAL_REF:
            continue
        tag = f"m{abs(lo)}_m{abs(hi)}" if hi < 0 else f"p{lo}_p{hi}"
        indicator = d["cal_time"].between(lo, hi).astype(float)
        ht = indicator * d["treat"] * d["exposure_v2_z"]
        target = f"HTA_{tag}"
        d[target] = ht * d["acq"]
        d[f"HA_{tag}"] = indicator * d["exposure_v2_z"] * d["acq"]
        d[f"TA_{tag}"] = indicator * d["treat"] * d["acq"]
        d[f"HT_{tag}"] = ht
        d[f"H_{tag}"] = indicator * d["exposure_v2_z"]
        d[f"T_{tag}"] = indicator * d["treat"]
        targets.append(target)
        controls.extend([
            f"HA_{tag}", f"TA_{tag}", f"HT_{tag}", f"H_{tag}", f"T_{tag}"
        ])
        lookup[target] = (lo, hi)
    formula = f"y ~ {' + '.join(targets + controls)} | prefA + monthA"
    model = pf.feols(formula, data=d, vcov={"CRV1": "prov_id"})
    names = list(model.coef().index)
    index = [names.index(term) for term in targets]
    beta = model.coef().values[index]
    vcov = model._vcov[np.ix_(index, index)]
    pre_index = [i for i, term in enumerate(targets) if lookup[term][1] < 0]
    bpre = beta[pre_index]
    vpre = vcov[np.ix_(pre_index, pre_index)]
    joint_p = float(stats.chi2.sf(float(bpre.T @ np.linalg.pinv(vpre) @ bpre), len(pre_index)))
    rows = []
    for i, term in enumerate(targets):
        lo, hi = lookup[term]
        se = float(np.sqrt(vcov[i, i]))
        rows.append({
            "design": "balanced_classified_acquaintance_minus_stranger",
            "bin_start": lo,
            "bin_end": hi,
            "reference_bin": "[-6,-1]",
            "coefficient": float(beta[i]),
            "std_error_crv1": se,
            "ci95_low": float(beta[i] - 1.96 * se),
            "ci95_high": float(beta[i] + 1.96 * se),
            "p_wild": float(wild_score_p(formula, d, term)),
            "joint_lead_p_chi2": joint_p,
            "n_obs": int(model._N),
            "formula": formula,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Balanced clean-window relational flow.
# ---------------------------------------------------------------------------
civil_panel = pd.read_parquet(f"{DATA}/civil_panel.parquet")
civil_panel["month"] = civil_panel["jmonth"].astype(str).str[:7]
rel = civil_panel[civil_panel["cause_family"].eq("relational")].copy()
flow_support = rel[rel["month"].between(SUPPORT_START, SUPPORT_END)][[
    "prefecture_code", "province", "cause"
]].drop_duplicates()
flow_counts = rel[rel["month"].between(START, END)][[
    "prefecture_code", "province", "cause", "month", "n_cases"
]]
flow = flow_support.merge(months, how="cross").merge(
    flow_counts,
    on=["prefecture_code", "province", "cause", "month"],
    how="left",
)
flow["n"] = flow["n_cases"].fillna(0).astype(float)
flow = (
    flow.merge(schedule, on="province")
    .merge(exposure[["prefecture_code", "exposure_v2_z"]], on="prefecture_code")
)
flow = add_design(flow)
flow["y"] = np.arcsinh(flow["n"])
flow["pref_cause"] = flow["prefecture_code"] + "_" + flow["cause"]
flow["cause_month"] = flow["cause"] + "_" + flow["month"]
flow_static = fit_target(
    "y ~ pth + ph + pt | pref_cause + month", flow, "pth"
)
flow_saturated = fit_target(
    "y ~ pth + ph | pref_cause + prov_month + cause_month", flow, "pth"
)
flow_es = simple_calendar_eventstudy(
    flow, "y", "pref_cause", "balanced_relational_cause_flow"
)


# ---------------------------------------------------------------------------
# Balanced acquaintance-versus-stranger flow among classified-text cases.
# ---------------------------------------------------------------------------
case = pd.read_parquet(
    f"{DATA}/civil_case.parquet",
    columns=["cause", "prefecture_code", "province", "jmonth", "rel_txn"],
)
case["month"] = case["jmonth"].astype(str).str[:7]
lending_all = case[case["cause"].eq("民间借贷纠纷")].copy()
lending = lending_all[lending_all["rel_txn"].notna()].copy()
lending["acq"] = lending["rel_txn"].astype(int)
comp_support = (
    lending[lending["month"].between(SUPPORT_START, SUPPORT_END)][
        ["prefecture_code", "province"]
    ].drop_duplicates()
    .merge(exposure[["prefecture_code", "province"]],
           on=["prefecture_code", "province"], how="inner")
)
comp_counts = (
    lending[lending["month"].between(START, END)]
    .groupby(["prefecture_code", "province", "month", "acq"])
    .size().rename("n").reset_index()
)
composition = (
    comp_support.merge(months, how="cross")
    .merge(pd.DataFrame({"acq": [0, 1]}), how="cross")
    .merge(comp_counts, on=["prefecture_code", "province", "month", "acq"], how="left")
)
composition["n"] = composition["n"].fillna(0).astype(float)
composition = (
    composition.merge(schedule, on="province")
    .merge(exposure[["prefecture_code", "exposure_v2_z"]], on="prefecture_code")
)
composition = add_design(composition)
composition["y"] = np.arcsinh(composition["n"])
composition["prefA"] = composition["prefecture_code"] + "_" + composition["acq"].astype(str)
composition["monthA"] = composition["month"] + "_" + composition["acq"].astype(str)
for term in ("pth", "ph", "pt"):
    composition[f"{term}A"] = composition[term] * composition["acq"]

composition_formula = "y ~ pthA + phA + ptA + pth + ph + pt | prefA + monthA"
composition_static = fit_target(composition_formula, composition, "pthA")
composition_saturated = fit_target(
    "y ~ pthA + phA + ptA + pth + ph | prefA + prov_month + monthA",
    composition,
    "pthA",
)
stranger_static = fit_target(
    "y ~ pth + ph + pt | prefecture_code + month",
    composition[composition["acq"].eq(0)].copy(),
    "pth",
)
acquaintance_static = fit_target(
    "y ~ pth + ph + pt | prefecture_code + month",
    composition[composition["acq"].eq(1)].copy(),
    "pth",
)
composition_es = composition_calendar_eventstudy(composition)


# Text-availability neutrality check on the full lending docket.
availability = (
    lending_all[lending_all["month"].between(START, END)]
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
availability_static = fit_target(
    "share_available ~ pth + ph + pt | prefecture_code + month",
    availability,
    "pth",
    weights="n_total",
)


case_counts = {
    "acquaintance": int(
        lending[lending["month"].between(START, END) & lending["acq"].eq(1)].shape[0]
    ),
    "stranger": int(
        lending[lending["month"].between(START, END) & lending["acq"].eq(0)].shape[0]
    ),
}
case_counts["classified_total"] = case_counts["acquaintance"] + case_counts["stranger"]

static_rows = []
for spec_id, result in (
    ("balanced_relational_flow", flow_static),
    ("balanced_relational_flow_saturated", flow_saturated),
    ("classified_acquaintance_minus_stranger", composition_static),
    ("classified_acquaintance_minus_stranger_saturated", composition_saturated),
    ("classified_acquaintance_flow", acquaintance_static),
    ("classified_stranger_flow", stranger_static),
    ("relationship_flag_availability", availability_static),
):
    static_rows.append({"spec_id": spec_id, **result})
static = pd.DataFrame(static_rows)

flow_csv = write_versioned_csv(flow_es, OUT, "clean_flow_eventstudy_revised")
comp_csv = write_versioned_csv(composition_es, OUT, "composition_eventstudy_revised")
static_csv = write_versioned_csv(static, OUT, "primary_civil_revised")


def event_value(frame, lo, field):
    return float(frame.loc[frame["bin_start"].eq(lo), field].iloc[0])


comp_lead_early = event_value(composition_es, -20, "coefficient")
comp_lead_late = event_value(composition_es, -12, "coefficient")
comp_post = event_value(composition_es, 0, "coefficient")
comp_post_se = event_value(composition_es, 0, "std_error_crv1")
comp_pre_step_bound = max(
    abs(comp_lead_late - comp_lead_early),
    abs(comp_lead_late),
)
comp_breakdown_mbar = max(
    (abs(comp_post) - 1.96 * comp_post_se) / comp_pre_step_bound,
    0.0,
)
composition_rr = pd.DataFrame([{
    "design": "balanced_classified_acquaintance_minus_stranger",
    "target": "post bin [0,6]",
    "theta": comp_post,
    "std_error_crv1": comp_post_se,
    "pre_step_bound_B": comp_pre_step_bound,
    "breakdown_Mbar_95pct": comp_breakdown_mbar,
}])
rr_csv = write_versioned_csv(composition_rr, OUT, "composition_rr_revised")


beta = flow_static["coefficient"]
pre1617 = rel[rel["month"].str[:4].isin(["2016", "2017"])]
pre_per_pref_year = pre1617.groupby("prefecture_code")["n_cases"].sum().mean() / 2.0
increment_per_year = pre_per_pref_year * (np.exp(beta) - 1)
workload_pct = 100 * (np.exp(beta) - 1)

macros = {
    "StackedCivFlow": f"{flow_static['coefficient']:.3f}",
    "StackedCivFlowSE": f"{flow_static['std_error_crv1']:.3f}",
    "StackedCivFlowP": f"{flow_static['p_crv1']:.3f}",
    "StackedCivFlowWildP": f"{flow_static['p_wild']:.3f}",
    "StackedCivFlowCRVThreeP": f"{flow_static['p_crv3']:.3f}",
    "StackedCivFlowN": f"{flow_static['n_obs']:,}",
    "SatCivFlow": f"{flow_saturated['coefficient']:.4f}",
    "SatCivFlowSE": f"{flow_saturated['std_error_crv1']:.4f}",
    "SatCivFlowWildP": f"{flow_saturated['p_wild']:.3f}",
    "SatCivFlowN": f"{flow_saturated['n_obs']:,}",
    "RelCasesAbs": f"{increment_per_year:,.0f}",
    "RelWorkloadPct": f"{workload_pct:.0f}",
    "CutAcqDiff": f"{composition_static['coefficient']:.3f}",
    "CutAcqDiffSE": f"{composition_static['std_error_crv1']:.3f}",
    "CutAcqDiffP": f"{composition_static['p_crv1']:.3f}",
    "CutAcqDiffWildP": f"{composition_static['p_wild']:.3f}",
    "CutAcqDiffCRVThreeP": f"{composition_static['p_crv3']:.3f}",
    "CutAcqDiffN": f"{composition_static['n_obs']:,}",
    "CutAcqFlow": f"{acquaintance_static['coefficient']:.3f}",
    "CutAcqFlowSE": f"{acquaintance_static['std_error_crv1']:.3f}",
    "CutAcqFlowWildP": f"{acquaintance_static['p_wild']:.3f}",
    "CutStrangerFlow": f"{stranger_static['coefficient']:.3f}",
    "CutStrangerFlowSE": f"{stranger_static['std_error_crv1']:.3f}",
    "CutStrangerFlowWildP": f"{stranger_static['p_wild']:.3f}",
    "CutAcqN": f"{case_counts['acquaintance']:,}",
    "CutStrangerN": f"{case_counts['stranger']:,}",
    "CutClassifiedN": f"{case_counts['classified_total']:,}",
    "SatAcqDiff": f"{composition_saturated['coefficient']:.4f}",
    "SatAcqDiffSE": f"{composition_saturated['std_error_crv1']:.4f}",
    "SatAcqDiffWildP": f"{composition_saturated['p_wild']:.3f}",
    "SatAcqDiffN": f"{composition_saturated['n_obs']:,}",
    "RelFlagAvailability": f"{availability_static['coefficient']:.3f}",
    "RelFlagAvailabilityWildP": f"{availability_static['p_wild']:.3f}",
    "CompESLeadEarly": f"{comp_lead_early:.3f}",
    "CompESLeadLate": f"{comp_lead_late:.3f}",
    "CompESPost": f"{comp_post:.3f}",
    "CompESPostSE": f"{comp_post_se:.3f}",
    "CompESPostWildP": f"{event_value(composition_es, 0, 'p_wild'):.3f}",
    "CompESLeadJointP": f"{composition_es['joint_lead_p_chi2'].iloc[0]:.3f}",
    "CompESPreStepBound": f"{comp_pre_step_bound:.3f}",
    "CompESBreakdownMbar": f"{comp_breakdown_mbar:.2f}",
    "FlowESLeadEarly": f"{event_value(flow_es, -20, 'coefficient'):.3f}",
    "FlowESLeadLate": f"{event_value(flow_es, -12, 'coefficient'):.3f}",
    "FlowESPost": f"{event_value(flow_es, 0, 'coefficient'):.3f}",
    "FlowESPostSE": f"{event_value(flow_es, 0, 'std_error_crv1'):.3f}",
    "FlowESPostWildP": f"{event_value(flow_es, 0, 'p_wild'):.3f}",
    "FlowESLeadJointP": f"{flow_es['joint_lead_p_chi2'].iloc[0]:.3f}",
}
macro_text = "% Revised primary civil results; generated by code/110_primary_civil_revised.py.\n"
macro_text += "\n".join(
    f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in macros.items()
) + "\n"
macro_paths = write_versioned_text(macro_text, TAB, "numbers_civil_revised")

print("STATIC RESULTS", flush=True)
print(static[[
    "spec_id", "coefficient", "std_error_crv1", "p_crv1", "p_wild", "p_crv3", "n_obs"
]].round(6).to_string(index=False), flush=True)
print("\nFLOW DYNAMICS", flush=True)
print(flow_es[[
    "bin_start", "bin_end", "coefficient", "std_error_crv1", "p_wild", "joint_lead_p_chi2"
]].round(6).to_string(index=False), flush=True)
print("\nCOMPOSITION DYNAMICS", flush=True)
print(composition_es[[
    "bin_start", "bin_end", "coefficient", "std_error_crv1", "p_wild", "joint_lead_p_chi2"
]].round(6).to_string(index=False), flush=True)
print("\nCOMPOSITION RELATIVE-MAGNITUDES SENSITIVITY", flush=True)
print(composition_rr.round(6).to_string(index=False), flush=True)
print(f"\nCASE COUNTS {case_counts}", flush=True)
print(f"FLOW TRANSLATION +{increment_per_year:.0f} cases/year, {workload_pct:.1f}%", flush=True)
print("OUTPUTS", static_csv, flow_csv, comp_csv, rr_csv, macro_paths, sep="\n", flush=True)

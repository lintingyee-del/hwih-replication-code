# -*- coding: utf-8 -*-
"""Auditable bounded-skill search for the three criminal-side outcomes.

The research question, treatment, and identifying fixed effects are held fixed:
    Post-inspection x pre-campaign coercive-capacity exposure,
    prefecture FE, province-by-time FE, province-clustered inference.

The search changes one feature at a time. It covers transformations of the current
outcomes, aggregation frequencies, symmetric event windows, and count models that
restore zero-count prefecture-months. It does not overwrite the paper or any baseline
result. Every attempted specification, including failures, is written to one CSV.

Usage (project venv):
  AutoFigure-main/.venv/Scripts/python.exe analysis/code/81_criminal_specs.py
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

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"
OUT = ROOT / "output"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE))
from _wild import wild_score_p  # noqa: E402

REPS = int(os.environ.get("HWIH_WILD_REPS", "9999"))
SEED = int(os.environ.get("HWIH_SEED", "20260715"))
rows: list[dict] = []


def bh_adjust(values: pd.Series) -> pd.Series:
    """Benjamini-Hochberg adjustment, preserving missing values."""
    out = pd.Series(np.nan, index=values.index, dtype=float)
    good = values.dropna().astype(float)
    if good.empty:
        return out
    order = good.sort_values().index
    p = good.loc[order].to_numpy()
    m = len(p)
    q = p * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out.loc[order] = np.minimum(q, 1.0)
    return out


def record_failure(spec_id: str, outcome_family: str, base_variable: str,
                   transformation: str, model: str, sample_rule: str,
                   controls: str, fixed_effects: str, reason: str,
                   source: str = "new_search") -> None:
    rows.append({
        "spec_id": spec_id,
        "mode": "direct_experiment",
        "focus_side": "y",
        "outcome_family": outcome_family,
        "base_variable": base_variable,
        "transformation": transformation,
        "model": model,
        "sample_rule": sample_rule,
        "controls": controls,
        "fixed_effects": fixed_effects,
        "coefficient": np.nan,
        "std_error": np.nan,
        "p_value": np.nan,
        "wild_p": np.nan,
        "n_obs": np.nan,
        "direction": "failed",
        "keep_or_drop": "drop",
        "reason": reason,
        "source": source,
    })


def run_feols(spec_id: str, outcome_family: str, base_variable: str,
              transformation: str, data: pd.DataFrame, fml: str, coef: str,
              sample_rule: str, controls: str, fixed_effects: str,
              weights: str | None = None, reason: str = "admissible search",
              focus_side: str = "y") -> None:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = pf.feols(fml, data=data, vcov={"CRV1": "prov_id"}, weights=weights)
            wp = wild_score_p(
                fml, data, coef, weights=weights, cluster="prov_id",
                reps=REPS, seed=SEED,
            )
        b = float(m.coef()[coef])
        se = float(m.se()[coef])
        p = float(m.pvalue()[coef])
        n = int(m._N)
        rows.append({
            "spec_id": spec_id,
            "mode": "direct_experiment",
            "focus_side": focus_side,
            "outcome_family": outcome_family,
            "base_variable": base_variable,
            "transformation": transformation,
            "model": "FE-OLS" + (" WLS" if weights else ""),
            "sample_rule": sample_rule,
            "controls": controls,
            "fixed_effects": fixed_effects,
            "coefficient": b,
            "std_error": se,
            "p_value": p,
            "wild_p": wp,
            "n_obs": n,
            "direction": "negative" if b < 0 else "positive",
            "keep_or_drop": "evaluate",
            "reason": reason,
            "source": "new_search",
        })
        print(f"{spec_id:34s} b={b:+.5f} se={se:.5f} p={p:.3f} "
              f"wild={wp:.3f} N={n:,}", flush=True)
    except Exception as exc:  # retain every failed attempt in the ledger
        print(f"{spec_id:34s} FAILED: {exc}", flush=True)
        record_failure(spec_id, outcome_family, base_variable, transformation,
                       "FE-OLS", sample_rule, controls, fixed_effects,
                       f"estimation failed: {type(exc).__name__}: {exc}")


def run_fepois(spec_id: str, outcome_family: str, base_variable: str,
               data: pd.DataFrame, fml: str, coef: str, sample_rule: str,
               fixed_effects: str, reason: str) -> None:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = pf.fepois(fml, data=data, vcov={"CRV1": "prov_id"})
        b = float(m.coef()[coef])
        se = float(m.se()[coef])
        p = float(m.pvalue()[coef])
        n = int(m._N)
        rows.append({
            "spec_id": spec_id,
            "mode": "direct_experiment",
            "focus_side": "y",
            "outcome_family": outcome_family,
            "base_variable": base_variable,
            "transformation": "level count",
            "model": "FE-PPML",
            "sample_rule": sample_rule,
            "controls": "none",
            "fixed_effects": fixed_effects,
            "coefficient": b,
            "std_error": se,
            "p_value": p,
            "wild_p": np.nan,
            "n_obs": n,
            "direction": "negative" if b < 0 else "positive",
            "keep_or_drop": "evaluate",
            "reason": reason + "; wild-score routine is linear-model only",
            "source": "new_search",
        })
        print(f"{spec_id:34s} b={b:+.5f} se={se:.5f} p={p:.3f} "
              f"wild=NA N={n:,}", flush=True)
    except Exception as exc:
        print(f"{spec_id:34s} FAILED: {exc}", flush=True)
        record_failure(spec_id, outcome_family, base_variable, "level count",
                       "FE-PPML", sample_rule, "none", fixed_effects,
                       f"estimation failed: {type(exc).__name__}: {exc}")


def add_common(d: pd.DataFrame, time_col: str = "month") -> pd.DataFrame:
    x = d.copy()
    x["pref"] = x["prefecture_code"].astype(str)
    x["prov_id"] = pd.factorize(x["province"])[0]
    x["prov_time"] = x["province"] + "_" + x[time_col].astype(str)
    x["H"] = x["exposure_v2_z"]
    x["px"] = x["post"] * x["H"]
    return x


def aggregate_share(d: pd.DataFrame, y: str, freq: str) -> pd.DataFrame:
    """Case-weighted share in quarter/half-year; drop inspection period."""
    x = d.dropna(subset=[y, "x_doclen", "n_cases"]).copy()
    dt = pd.to_datetime(x["jmonth"])
    ins = pd.to_datetime(x["insp_month"])
    if freq == "quarter":
        x["period"] = dt.dt.to_period("Q").astype(str)
        x["insp_period"] = ins.dt.to_period("Q").astype(str)
    elif freq == "halfyear":
        x["period"] = dt.dt.year.astype(str) + "H" + np.where(dt.dt.month <= 6, "1", "2")
        x["insp_period"] = ins.dt.year.astype(str) + "H" + np.where(ins.dt.month <= 6, "1", "2")
    else:
        raise ValueError(freq)
    x = x[x["period"] != x["insp_period"]].copy()
    x["yn"] = x[y] * x["n_cases"]
    x["docn"] = x["x_doclen"] * x["n_cases"]
    g = (x.groupby([
        "prefecture_code", "province", "period", "insp_period", "exposure_v2_z"
    ], as_index=False)
         .agg(yn=("yn", "sum"), n_cases=("n_cases", "sum"), docn=("docn", "sum")))
    g[y] = g["yn"] / g["n_cases"]
    g["x_doclen"] = g["docn"] / g["n_cases"]
    g["log_doclen"] = np.log1p(g["x_doclen"])
    g["post"] = (g["period"] > g["insp_period"]).astype(int)
    return add_common(g, "period")


print(f"bounded criminal search: wild reps={REPS}, seed={SEED}", flush=True)
raw = pd.read_parquet(DATA / "crim_panel_v2.parquet")
raw["month"] = pd.to_datetime(raw["jmonth"]).dt.to_period("M").astype(str)
raw["log_doclen"] = np.log1p(raw["x_doclen"])
raw = add_common(raw, "month")


# ---------------------------------------------------------------------------
# 1. Current content-share outcomes: change one feature at a time.
# ---------------------------------------------------------------------------
share_defs = [
    ("backstop", "market", "y_backstop"),
    ("detention", "enforcementcrime", "y_detention_debt"),
]

for family_name, docket, y in share_defs:
    d = raw[(raw["family"] == docket) & (raw["n_cases"] > 0)].copy()
    d["y_asinh"] = np.arcsinh(d[y])
    d["y_log1p"] = np.log1p(d[y])
    shrunk = (d[y] * d["n_cases"] + 0.5) / (d["n_cases"] + 1.0)
    d["y_logit_shrunk"] = np.log(shrunk / (1.0 - shrunk))

    prefix = "B" if family_name == "backstop" else "D"
    base_var = y
    fixed = "prefecture + province-month"
    run_feols(
        f"{prefix}00_baseline", family_name, base_var, "level share", d,
        f"{y} ~ px + x_doclen | pref + prov_time", "px",
        "all positive-docket prefecture-months", "mean document length", fixed,
        weights="n_cases", reason="published baseline; preserve exact estimand",
    )
    run_feols(
        f"{prefix}01_no_doc_control", family_name, base_var, "level share", d,
        f"{y} ~ px | pref + prov_time", "px",
        "same as baseline", "none", fixed, weights="n_cases",
        reason="tests dependence on a potentially post-treatment narration control",
    )
    run_feols(
        f"{prefix}02_log_doc_control", family_name, base_var, "level share", d,
        f"{y} ~ px + log_doclen | pref + prov_time", "px",
        "same as baseline", "log(1 + mean document length)", fixed,
        weights="n_cases", reason="reduces leverage from extreme document lengths",
    )
    run_feols(
        f"{prefix}03_unweighted", family_name, base_var, "level share", d,
        f"{y} ~ px + log_doclen | pref + prov_time", "px",
        "same cells; each prefecture-month receives equal weight",
        "log(1 + mean document length)", fixed, weights=None,
        reason="cell-average estimand; less efficient when small cells are noisy",
    )
    run_feols(
        f"{prefix}04_asinh_share", family_name, base_var, "asinh(share)", d,
        "y_asinh ~ px + log_doclen | pref + prov_time", "px",
        "same as baseline", "log(1 + mean document length)", fixed,
        weights="n_cases", reason="monotone transformation retaining zeros",
    )
    run_feols(
        f"{prefix}05_log1p_share", family_name, base_var, "log(1 + share)", d,
        "y_log1p ~ px + log_doclen | pref + prov_time", "px",
        "same as baseline", "log(1 + mean document length)", fixed,
        weights="n_cases", reason="monotone transformation retaining zeros",
    )
    run_feols(
        f"{prefix}06_logit_shrunk", family_name, base_var,
        "logit((flag_count + 0.5)/(n + 1))", d,
        "y_logit_shrunk ~ px + log_doclen | pref + prov_time", "px",
        "same as baseline", "log(1 + mean document length)", fixed,
        weights="n_cases", reason="bounded-share model with continuity correction",
    )

    for cutoff in (24, 18, 12):
        w = d[d["event_time"].between(-cutoff, cutoff)].copy()
        run_feols(
            f"{prefix}{7 + (24-cutoff)//6:02d}_event_pm{cutoff}",
            family_name, base_var, "level share", w,
            f"{y} ~ px + x_doclen | pref + prov_time", "px",
            f"symmetric event window [-{cutoff}, +{cutoff}] months",
            "mean document length", fixed, weights="n_cases",
            reason="localizes the inspection contrast; exploratory window sensitivity",
        )

    for freq, suffix in (("quarter", "qtr"), ("halfyear", "half")):
        a = aggregate_share(d, y, freq)
        run_feols(
            f"{prefix}{10 if freq == 'quarter' else 11:02d}_{suffix}",
            family_name, base_var, f"{freq} case-weighted level share", a,
            f"{y} ~ px + log_doclen | pref + prov_time", "px",
            f"{freq} aggregation; inspection {freq} dropped",
            "log(1 + mean document length)",
            f"prefecture + province-{freq}", weights="n_cases",
            reason="reduces noise from thin monthly cells without changing the indicator",
        )


# ---------------------------------------------------------------------------
# 2. Enforcement count: retain baseline, then restore zero-count cells.
# ---------------------------------------------------------------------------
en_pos = raw[(raw["family"] == "enforcementcrime") & (raw["n_cases"] > 0)].copy()
en_pos["asinh_n"] = np.arcsinh(en_pos["n_cases"])
en_pos["log1p_n"] = np.log1p(en_pos["n_cases"])
run_feols(
    "C00_baseline_positive_asinh", "enforcement_count", "n_cases", "asinh(count)",
    en_pos, "asinh_n ~ px | pref + prov_time", "px",
    "positive enforcement-docket cells only", "none",
    "prefecture + province-month", reason="published baseline",
)
run_feols(
    "C01_positive_log1p", "enforcement_count", "n_cases", "log(1 + count)",
    en_pos, "log1p_n ~ px | pref + prov_time", "px",
    "positive enforcement-docket cells only", "none",
    "prefecture + province-month", reason="same selected cells; alternate zero-safe transform",
)

meta = (raw[["prefecture_code", "province", "exposure_v2_z", "insp_month"]]
        .drop_duplicates("prefecture_code"))
month_frame = pd.DataFrame({
    "jmonth": pd.date_range(pd.to_datetime(raw["jmonth"]).min(),
                             pd.to_datetime(raw["jmonth"]).max(), freq="MS")
})
full = meta.assign(_k=1).merge(month_frame.assign(_k=1), on="_k").drop(columns="_k")
obs = (raw[raw["family"] == "enforcementcrime"]
       [["prefecture_code", "jmonth", "n_cases"]].copy())
obs["jmonth"] = pd.to_datetime(obs["jmonth"])
full = full.merge(obs, on=["prefecture_code", "jmonth"], how="left")
full["n_cases"] = full["n_cases"].fillna(0.0)
full["month"] = full["jmonth"].dt.to_period("M").astype(str)
full["insp"] = pd.to_datetime(full["insp_month"]).dt.to_period("M").astype(str)
full["post"] = (full["month"] >= full["insp"]).astype(int)
full["event_time"] = ((full["jmonth"].dt.year - pd.to_datetime(full["insp_month"]).dt.year) * 12
                      + full["jmonth"].dt.month - pd.to_datetime(full["insp_month"]).dt.month)
full["asinh_n"] = np.arcsinh(full["n_cases"])
full["log1p_n"] = np.log1p(full["n_cases"])
full["any_n"] = (full["n_cases"] > 0).astype(int)
full = add_common(full, "month")

run_feols(
    "C02_complete_asinh", "enforcement_count", "n_cases", "asinh(count)",
    full, "asinh_n ~ px | pref + prov_time", "px",
    "balanced prefecture-month panel; zero-count cells restored", "none",
    "prefecture + province-month",
    reason="preferred support correction for a count outcome",
)
run_feols(
    "C03_complete_log1p", "enforcement_count", "n_cases", "log(1 + count)",
    full, "log1p_n ~ px | pref + prov_time", "px",
    "balanced prefecture-month panel; zero-count cells restored", "none",
    "prefecture + province-month",
    reason="zero-safe transformation on the complete count support",
)
run_feols(
    "C04_complete_any", "enforcement_count", "n_cases", "1(count > 0)",
    full, "any_n ~ px | pref + prov_time", "px",
    "balanced prefecture-month panel; zero-count cells restored", "none",
    "prefecture + province-month",
    reason="extensive-margin supplement; changes the estimand",
)
run_fepois(
    "C05_complete_ppml", "enforcement_count", "n_cases", full,
    "n_cases ~ px | pref + prov_time", "px",
    "balanced prefecture-month panel; zero-count cells restored",
    "prefecture + province-month",
    reason="count-support-matched model",
)

for cutoff, sid in ((24, 6), (18, 7), (12, 8)):
    w = full[full["event_time"].between(-cutoff, cutoff)].copy()
    run_feols(
        f"C{sid:02d}_complete_event_pm{cutoff}", "enforcement_count", "n_cases",
        "asinh(count)", w, "asinh_n ~ px | pref + prov_time", "px",
        f"balanced panel; symmetric event window [-{cutoff}, +{cutoff}] months",
        "none", "prefecture + province-month",
        reason="local count response; exploratory window sensitivity",
    )

full["quarter"] = full["jmonth"].dt.to_period("Q").astype(str)
full["insp_quarter"] = pd.to_datetime(full["insp_month"]).dt.to_period("Q").astype(str)
q = (full.groupby([
    "prefecture_code", "province", "quarter", "insp_quarter", "exposure_v2_z"
], as_index=False).agg(n_cases=("n_cases", "sum")))
q = q[q["quarter"] != q["insp_quarter"]].copy()
q["post"] = (q["quarter"] > q["insp_quarter"]).astype(int)
q["asinh_n"] = np.arcsinh(q["n_cases"])
q["log1p_n"] = np.log1p(q["n_cases"])
q = add_common(q, "quarter")
run_feols(
    "C09_complete_qtr_asinh", "enforcement_count", "n_cases",
    "asinh(quarterly count)", q, "asinh_n ~ px | pref + prov_time", "px",
    "balanced quarterly panel; inspection quarter dropped", "none",
    "prefecture + province-quarter",
    reason="reduces monthly count noise; preserves complete support",
)
run_feols(
    "C10_complete_qtr_log1p", "enforcement_count", "n_cases",
    "log(1 + quarterly count)", q, "log1p_n ~ px | pref + prov_time", "px",
    "balanced quarterly panel; inspection quarter dropped", "none",
    "prefecture + province-quarter",
    reason="alternate zero-safe quarterly transformation",
)
run_fepois(
    "C11_complete_qtr_ppml", "enforcement_count", "n_cases", q,
    "n_cases ~ px | pref + prov_time", "px",
    "balanced quarterly panel; inspection quarter dropped",
    "prefecture + province-quarter",
    reason="quarterly count-support-matched model",
)


# ---------------------------------------------------------------------------
# 3. Last-stage transformations of the existing exposure index. These do not
#    invent a new proxy: they only reduce tail leverage or use ordinal exposure.
# ---------------------------------------------------------------------------
ex = (raw[["prefecture_code", "exposure_v2_z"]]
      .drop_duplicates("prefecture_code").set_index("prefecture_code"))
for label, qlo, qhi, rank in [
    ("w1", 0.01, 0.99, False),
    ("w5", 0.05, 0.95, False),
    ("rank", None, None, True),
]:
    h = ex["exposure_v2_z"].copy()
    if rank:
        h = h.rank(method="average", pct=True)
    else:
        h = h.clip(h.quantile(qlo), h.quantile(qhi))
    h = (h - h.mean()) / h.std(ddof=1)
    hname = f"H_{label}"
    pxname = f"px_{label}"
    for frame in (raw, en_pos, full):
        frame[hname] = frame["prefecture_code"].map(h)
        frame[pxname] = frame["post"] * frame[hname]

    mkx = raw[(raw["family"] == "market") & (raw["n_cases"] > 0)].copy()
    dex = raw[(raw["family"] == "enforcementcrime") & (raw["n_cases"] > 0)].copy()
    run_feols(
        f"X_{label}_backstop", "backstop", "y_backstop",
        f"exposure {label} transformation; level share", mkx,
        f"y_backstop ~ {pxname} + x_doclen | pref + prov_time", pxname,
        "baseline positive-docket cells", "mean document length",
        "prefecture + province-month", weights="n_cases",
        reason="last-stage exposure-tail sensitivity; not selected by fit",
        focus_side="x",
    )
    run_feols(
        f"X_{label}_detention", "detention", "y_detention_debt",
        f"exposure {label} transformation; level share", dex,
        f"y_detention_debt ~ {pxname} + x_doclen | pref + prov_time", pxname,
        "baseline positive-docket cells", "mean document length",
        "prefecture + province-month", weights="n_cases",
        reason="last-stage exposure-tail sensitivity; not selected by fit",
        focus_side="x",
    )
    run_feols(
        f"X_{label}_count_positive", "enforcement_count", "n_cases",
        f"exposure {label} transformation; asinh(count)", en_pos,
        f"asinh_n ~ {pxname} | pref + prov_time", pxname,
        "positive enforcement-docket cells", "none",
        "prefecture + province-month",
        reason="last-stage exposure-tail sensitivity; retains published count support",
        focus_side="x",
    )
    run_feols(
        f"X_{label}_count_any", "enforcement_count", "n_cases",
        f"exposure {label} transformation; 1(count > 0)", full,
        f"any_n ~ {pxname} | pref + prov_time", pxname,
        "balanced prefecture-month panel; zero-count cells restored", "none",
        "prefecture + province-month",
        reason="last-stage exposure-tail sensitivity on extensive margin",
        focus_side="x",
    )


# ---------------------------------------------------------------------------
# 4. Historical admissible attempts: retained for the audit trail, not folded
#    into the current-search BH correction because they were not predeclared here.
# ---------------------------------------------------------------------------
hist_path = OUT / "referee_robustness.csv"
if hist_path.exists():
    hist = pd.read_csv(hist_path).set_index("tag")
    for tag, label, reason in [
        ("E1_enf_H1415", "2014-15 split-half exposure",
         "more significant but unstable across split halves; not a main specification"),
        ("E1_enf_H1617", "2016-17 split-half exposure",
         "same concept and model; later half is uninformative"),
        ("E2_v1_backstop_baseline", "legacy v1 backstop dictionary",
         "stronger but superseded by the audited v2 dictionary; not adopted"),
        ("E2_v2_backstop_dropmafia", "v2 backstop, mafia cells dropped",
         "robustness cut; not a replacement for the audited baseline"),
    ]:
        if tag not in hist.index:
            continue
        z = hist.loc[tag]
        b = float(z["est"])
        rows.append({
            "spec_id": f"HIST_{tag}",
            "mode": "direct_experiment",
            "focus_side": "x" if "H14" in tag or "H16" in tag else "y",
            "outcome_family": "enforcement_count" if "enf_H" in tag else "backstop",
            "base_variable": "historical",
            "transformation": label,
            "model": "historical FE-OLS",
            "sample_rule": "see referee_robustness.csv",
            "controls": "see source script",
            "fixed_effects": "prefecture + province-month",
            "coefficient": b,
            "std_error": float(z["se"]),
            "p_value": float(z["p"]),
            "wild_p": float(z["wild_p"]),
            "n_obs": int(z["n"]),
            "direction": "negative" if b < 0 else "positive",
            "keep_or_drop": "drop" if "H14" in tag or "v1" in tag else "appendix",
            "reason": reason,
            "source": "historical",
        })

hierarchy_path = OUT / "hierarchy_numbers.csv"
if hierarchy_path.exists():
    h = pd.read_csv(hierarchy_path)
    z = h[h["tag"] == "demil_index"].iloc[0]
    rows.append({
        "spec_id": "HIST_outcome_level_equal_weight_index",
        "mode": "direct_experiment", "focus_side": "y",
        "outcome_family": "joint_index", "base_variable": "three criminal outcomes",
        "transformation": "pre-period standardized equal-weight outcome index",
        "model": "historical FE-OLS", "sample_rule": "common merged panel",
        "controls": "none", "fixed_effects": "prefecture + province-month",
        "coefficient": float(z["est"]), "std_error": float(z["se"]),
        "p_value": float(z["p"]), "wild_p": float(z["wild_p"]),
        "n_obs": int(z["n"]), "direction": "negative",
        "keep_or_drop": "drop",
        "reason": f"CRV1/wild look strong but wave-timing RI p={float(z['ri_p']):.3f}; not robust",
        "source": "historical",
    })


result = pd.DataFrame(rows)
new = result["source"].eq("new_search")
result.loc[new, "bh_q_crv1_within_outcome"] = (
    result.loc[new].groupby("outcome_family", group_keys=False)["p_value"].apply(bh_adjust)
)
result.loc[new, "bh_q_wild_within_outcome"] = (
    result.loc[new].groupby("outcome_family", group_keys=False)["wild_p"].apply(bh_adjust)
)
result["search_reps"] = np.where(new, REPS, np.nan)

csv_path = OUT / "criminal_specs.csv"
result.to_csv(csv_path, index=False, encoding="utf-8-sig")

summary = []
summary.append("# bounded criminal-side specification search")
summary.append("")
summary.append(f"- Wild-score draws per new linear specification: {REPS}")
summary.append("- Treatment and fixed effects held fixed unless the row explicitly says otherwise.")
summary.append("- BH q-values apply only within each outcome family in this newly declared search.")
summary.append("- Historical attempts are retained but excluded from the new-search q-value denominator.")
summary.append("")
for fam in ["backstop", "detention", "enforcement_count", "joint_index"]:
    z = result[result["outcome_family"] == fam].copy()
    if z.empty:
        continue
    summary.append(f"## {fam}")
    summary.append("")
    summary.append("| spec | b | se | CRV1 p | wild p | wild BH q | N | source |")
    summary.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for _, r in z.iterrows():
        def fmt(v, k=3):
            return "" if pd.isna(v) else f"{float(v):.{k}f}"
        summary.append(
            f"| {r['spec_id']} | {fmt(r['coefficient'],4)} | {fmt(r['std_error'],4)} | "
            f"{fmt(r['p_value'])} | {fmt(r['wild_p'])} | "
            f"{fmt(r.get('bh_q_wild_within_outcome', np.nan))} | "
            f"{'' if pd.isna(r['n_obs']) else int(r['n_obs']):} | {r['source']} |"
        )
    summary.append("")

md_path = OUT / "criminal_specs.md"
md_path.write_text("\n".join(summary).rstrip() + "\n", encoding="utf-8")
print(f"[done] wrote {csv_path}", flush=True)
print(f"[done] wrote {md_path}", flush=True)

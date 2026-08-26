# -*- coding: utf-8 -*-
"""Admissible specification search for the civil level estimand under saturated FE.

Target: the balanced clean-window relational-cause flow, which is 0.1896
(wild p=0.013) with prefecture-by-cause and month FE but attenuates to 0.0962
(wild p=0.162) once province-by-month and cause-by-month FE are added
(Appendix saturated-FE table). This script holds the research question, window,
support, treatment, and clustering fixed and searches only:

  (i)  FE decomposition: which saturation layer absorbs the coefficient;
  (ii) outcome transform: asinh (baseline), log1p, extensive margin;
  (iii) estimation: PPML on counts (Chen & Roth 2023 scale-dependence of asinh
        motivates the count model as the principled alternative), and WLS with
        predetermined pre-period caseload weights (the level estimand's stated
        role is magnitude, which the unweighted cell regression does not target:
        equal weights let near-empty prefecture-cause cells dominate).

Inference: CRV1 + null-imposed wild-score bootstrap (9,999 draws) for linear
rows; PPML rows report CRV1 only (paper precedent for nonlinear rows). BH
adjustment across this round's linear wild p-values is reported in the log.
No paper output is patched.
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

from _wild import wild_score_p

DATA = str(_REP_PROJECT / "data")
OUT = str(_REP_PROJECT / "output")
WINDOW = ("2017-01", "2019-03")
SUPPORT = ("2014-01", "2017-12")
POST0 = "2018-09"
ROWS = []

schedule = (
    pd.read_parquet(f"{DATA}/panel_month.parquet")[["province", "inspection_round"]]
    .drop_duplicates()
)
exposure = pd.read_parquet(f"{DATA}/exposure_v2.parquet")[[
    "prefecture_code", "exposure_v2_z"
]]
months = pd.DataFrame({
    "month": pd.period_range(WINDOW[0], WINDOW[1], freq="M").astype(str)
})

civil = pd.read_parquet(f"{DATA}/civil_panel.parquet")
civil["month"] = civil["jmonth"].astype(str).str[:7]
relational = civil[civil["cause_family"].eq("relational")].copy()

support_pairs = relational[
    relational["month"].between(SUPPORT[0], SUPPORT[1])
][["prefecture_code", "province", "cause"]].drop_duplicates()

pre_totals = (
    relational[relational["month"].between(SUPPORT[0], SUPPORT[1])]
    .groupby(["prefecture_code", "cause"])["n_cases"].sum()
    .rename("pre_total")
    .reset_index()
)
n_support_months = len(pd.period_range(SUPPORT[0], SUPPORT[1], freq="M"))

counts = relational[
    relational["month"].between(WINDOW[0], WINDOW[1])
][["prefecture_code", "province", "cause", "month", "n_cases"]]

total = support_pairs.merge(months, how="cross").merge(
    counts, on=["prefecture_code", "province", "cause", "month"], how="left"
)
total["n_cases"] = total["n_cases"].fillna(0).astype(float)
total = (
    total.merge(schedule, on="province")
    .merge(exposure, on="prefecture_code")
    .merge(pre_totals, on=["prefecture_code", "cause"], how="left")
    .dropna(subset=["inspection_round", "exposure_v2_z"])
)
total["pre_mean"] = total["pre_total"].fillna(0) / n_support_months
total["treat"] = (total["inspection_round"] == 1).astype(int)
total["postc"] = (total["month"] >= POST0).astype(int)
total["prov_id"] = pd.factorize(total["province"])[0]
total["pt"] = total["postc"] * total["treat"]
total["pth"] = total["pt"] * total["exposure_v2_z"]
total["ph"] = total["postc"] * total["exposure_v2_z"]
total["prov_month"] = total["province"] + "_" + total["month"]
total["pref_cause"] = total["prefecture_code"] + "_" + total["cause"]
total["cause_month"] = total["cause"] + "_" + total["month"]
total["y_asinh"] = np.arcsinh(total["n_cases"])
total["y_log1p"] = np.log1p(total["n_cases"])
total["y_any"] = total["n_cases"].gt(0).astype(float)

print(f"panel rows={len(total):,}  pref-cause pairs={total['pref_cause'].nunique():,}",
      flush=True)


def run_linear(label, formula, data, weights=None):
    fit = pf.feols(formula, data=data, weights=weights, vcov={"CRV1": "prov_id"})
    crv3 = pf.feols(formula, data=data, weights=weights, vcov={"CRV3": "prov_id"})
    pw = wild_score_p(formula, data, "pth", weights=weights)
    row = {
        "spec": label,
        "estimator": "OLS" if weights is None else f"WLS[{weights}]",
        "coef_pth": float(fit.coef()["pth"]),
        "se_crv1": float(fit.se()["pth"]),
        "p_crv1": float(fit.pvalue()["pth"]),
        "p_crv3": float(crv3.pvalue()["pth"]),
        "p_wild": float(pw),
        "n": int(fit._N),
        "formula": formula,
    }
    ROWS.append(row)
    print(f"{label:52s} b={row['coef_pth']:+.4f} se={row['se_crv1']:.4f} "
          f"p1={row['p_crv1']:.4f} wild={row['p_wild']:.4f} p3={row['p_crv3']:.4f}",
          flush=True)
    return row


def run_ppml(label, formula, data):
    fit = pf.fepois(formula, data=data, vcov={"CRV1": "prov_id"})
    row = {
        "spec": label,
        "estimator": "PPML",
        "coef_pth": float(fit.coef()["pth"]),
        "se_crv1": float(fit.se()["pth"]),
        "p_crv1": float(fit.pvalue()["pth"]),
        "p_crv3": np.nan,
        "p_wild": np.nan,
        "n": int(fit._N),
        "formula": formula,
    }
    ROWS.append(row)
    print(f"{label:52s} b={row['coef_pth']:+.4f} se={row['se_crv1']:.4f} "
          f"p1={row['p_crv1']:.4f} (PPML, CRV1 only)", flush=True)
    return row


BASE_FE = "pref_cause + month"
SAT_FE = "pref_cause + prov_month + cause_month"
MID_FE_PM = "pref_cause + prov_month"
MID_FE_CM = "pref_cause + month + cause_month"

# --- Panel 1: reproduction and FE decomposition (asinh) ---
run_linear("S00 asinh, baseline FE (repro 0.1896)",
           f"y_asinh ~ pth + ph + pt | {BASE_FE}", total)
run_linear("S01 asinh, + cause-month FE only",
           f"y_asinh ~ pth + ph + pt | {MID_FE_CM}", total)
run_linear("S02 asinh, + province-month FE only",
           f"y_asinh ~ pth + ph | {MID_FE_PM}", total)
run_linear("S03 asinh, saturated (repro 0.0962)",
           f"y_asinh ~ pth + ph | {SAT_FE}", total)

# --- Panel 2: transforms under saturated FE ---
run_linear("S04 log1p, saturated",
           f"y_log1p ~ pth + ph | {SAT_FE}", total)
run_linear("S05 any-case, saturated",
           f"y_any ~ pth + ph | {SAT_FE}", total)

# --- Panel 3: estimators targeting magnitude ---
run_linear("S06 asinh, baseline FE, pre-period WLS",
           f"y_asinh ~ pth + ph + pt | {BASE_FE}", total, weights="pre_mean")
run_linear("S07 asinh, saturated, pre-period WLS",
           f"y_asinh ~ pth + ph | {SAT_FE}", total, weights="pre_mean")
run_ppml("S08 PPML counts, baseline FE",
         f"n_cases ~ pth + ph + pt | {BASE_FE}", total)
run_ppml("S09 PPML counts, saturated",
         f"n_cases ~ pth + ph | {SAT_FE}", total)

# --- BH across this round's NEW linear wild p-values (S04-S07; S00-S03 are
#     reproductions/decompositions, not candidate replacements) ---
new = [r for r in ROWS if r["spec"].split()[0] in {"S04", "S05", "S06", "S07"}]
pvals = np.array([r["p_wild"] for r in new])
order = np.argsort(pvals)
ranked = pvals[order] * len(pvals) / np.arange(1, len(pvals) + 1)
ranked = np.minimum.accumulate(ranked[::-1])[::-1]
for i, idx in enumerate(order):
    new[idx]["bh_q_round"] = float(min(ranked[i], 1.0))
for r in ROWS:
    r.setdefault("bh_q_round", np.nan)

out = pd.DataFrame(ROWS)
path = f"{OUT}/saturated_repair.csv"
out.to_csv(path, index=False, encoding="utf-8-sig")
print(f"[done] wrote {path}", flush=True)

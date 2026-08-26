# -*- coding: utf-8 -*-
"""Studentized wave-label permutation for the two primary civil estimands.

111_primary_wave_permutation.py ranks raw coefficients across permuted
first-wave labels. Per Young (2019) and MacKinnon & Webb (2020), the
studentized statistic is preferred when precision varies across assignments.
This script reruns the same two estimands and the same label scheme (10 of 31
provinces first-wave) reporting raw and CRV1-studentized permutation p-values
side by side from the same draws. REPS=999 matches the paper's diagnostics.
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

DATA = str(_REP_PROJECT / "data")
OUT = str(_REP_PROJECT / "output")
WINDOW = ("2017-01", "2019-03")
SUPPORT = ("2014-01", "2017-12")
POST0 = "2018-09"
REPS = int(os.environ.get("PERM_REPS", "999"))
SEED = int(os.environ.get("PERM_SEED", "20260706"))

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

# --- balanced relational flow panel (as 106/121) ---
civil = pd.read_parquet(f"{DATA}/civil_panel.parquet")
civil["month"] = civil["jmonth"].astype(str).str[:7]
relational = civil[civil["cause_family"].eq("relational")].copy()
support_pairs = relational[
    relational["month"].between(SUPPORT[0], SUPPORT[1])
][["prefecture_code", "province", "cause"]].drop_duplicates()
counts = relational[
    relational["month"].between(WINDOW[0], WINDOW[1])
][["prefecture_code", "province", "cause", "month", "n_cases"]]
flow = support_pairs.merge(months, how="cross").merge(
    counts, on=["prefecture_code", "province", "cause", "month"], how="left"
)
flow["n_cases"] = flow["n_cases"].fillna(0).astype(float)
flow = flow.merge(exposure, on="prefecture_code").dropna(subset=["exposure_v2_z"])
flow["postc"] = (flow["month"] >= POST0).astype(int)
flow["y"] = np.arcsinh(flow["n_cases"])
flow["pref_cause"] = flow["prefecture_code"] + "_" + flow["cause"]
flow["prov_id"] = pd.factorize(flow["province"])[0]

# --- classified acquaintance/stranger composition panel (as 106) ---
case = pd.read_parquet(
    f"{DATA}/civil_case.parquet",
    columns=["cause", "prefecture_code", "province", "jmonth", "rel_txn"],
)
lending = case[case["cause"].eq("民间借贷纠纷") & case["rel_txn"].notna()].copy()
lending["month"] = lending["jmonth"].astype(str).str[:7]
lending["acq"] = lending["rel_txn"].astype(int)
comp_support = (
    lending[lending["month"].between(SUPPORT[0], SUPPORT[1])][
        ["prefecture_code", "province"]
    ].drop_duplicates()
    .merge(exposure[["prefecture_code"]], on="prefecture_code")
)
comp_counts = (
    lending[lending["month"].between(WINDOW[0], WINDOW[1])]
    .groupby(["prefecture_code", "province", "month", "acq"])
    .size().rename("n").reset_index()
)
comp = (
    comp_support.merge(months, how="cross")
    .merge(pd.DataFrame({"acq": [0, 1]}), how="cross")
    .merge(comp_counts, on=["prefecture_code", "province", "month", "acq"], how="left")
    .merge(exposure, on="prefecture_code")
    .dropna(subset=["exposure_v2_z"])
)
comp["n"] = comp["n"].fillna(0).astype(float)
comp["postc"] = (comp["month"] >= POST0).astype(int)
comp["y"] = np.arcsinh(comp["n"])
comp["prefA"] = comp["prefecture_code"] + "_" + comp["acq"].astype(str)
comp["monthA"] = comp["month"] + "_" + comp["acq"].astype(str)
comp["prov_id"] = pd.factorize(comp["province"])[0]


def apply_labels(d, tmap, interact_acq):
    d = d.copy()
    d["treat"] = d["province"].map(tmap).astype(int)
    d["pt"] = d["postc"] * d["treat"]
    d["ph"] = d["postc"] * d["exposure_v2_z"]
    d["pth"] = d["pt"] * d["exposure_v2_z"]
    if interact_acq:
        for term in ("pth", "ph", "pt"):
            d[f"{term}A"] = d[term] * d["acq"]
    return d


ESTIMANDS = {
    "balanced_relational_flow": {
        "data": flow,
        "formula": "y ~ pth + ph + pt | pref_cause + month",
        "coefficient": "pth",
        "interact_acq": False,
    },
    "classified_acquaintance_minus_stranger": {
        "data": comp,
        "formula": "y ~ pthA + phA + ptA + pth + ph + pt | prefA + monthA",
        "coefficient": "pthA",
        "interact_acq": True,
    },
}

rows = []
for name, spec in ESTIMANDS.items():
    data = spec["data"]
    provinces = np.array(sorted(data["province"].unique()))
    sched = schedule[schedule["province"].isin(provinces)]
    observed_map = dict(zip(sched["province"],
                            (sched["inspection_round"] == 1).astype(int)))
    n_treated = int(sum(observed_map.values()))
    d0 = apply_labels(data, observed_map, spec["interact_acq"])
    fit0 = pf.feols(spec["formula"], data=d0, vcov={"CRV1": "prov_id"})
    b0 = float(fit0.coef()[spec["coefficient"]])
    t0 = float(fit0.tstat()[spec["coefficient"]])
    print(f"{name}: observed b={b0:+.4f} t={t0:+.3f} "
          f"({n_treated}/{len(provinces)} treated)", flush=True)

    rng = np.random.default_rng(SEED)
    perm_b = np.empty(REPS)
    perm_t = np.empty(REPS)
    for r in range(REPS):
        labels = np.zeros(len(provinces), dtype=int)
        labels[rng.choice(len(provinces), n_treated, replace=False)] = 1
        d = apply_labels(data, dict(zip(provinces, labels)), spec["interact_acq"])
        f = pf.feols(spec["formula"], data=d, vcov={"CRV1": "prov_id"})
        perm_b[r] = float(f.coef()[spec["coefficient"]])
        perm_t[r] = float(f.tstat()[spec["coefficient"]])
        if (r + 1) % 100 == 0:
            print(f"{name}: {r + 1}/{REPS}", flush=True)

    rows.append({
        "estimand": name,
        "b_obs": b0,
        "t_obs": t0,
        "p_raw_two_sided": float((1 + np.sum(np.abs(perm_b) >= abs(b0))) / (REPS + 1)),
        "p_raw_one_sided": float((1 + np.sum(perm_b >= b0)) / (REPS + 1)),
        "p_stud_two_sided": float((1 + np.sum(np.abs(perm_t) >= abs(t0))) / (REPS + 1)),
        "p_stud_one_sided": float((1 + np.sum(perm_t >= t0)) / (REPS + 1)),
        "reps": REPS,
        "seed": SEED,
    })
    print(pd.Series(rows[-1]).to_string(), flush=True)

out = pd.DataFrame(rows)
path = f"{OUT}/civil_perm_studentized.csv"
out.to_csv(path, index=False, encoding="utf-8-sig")
print(f"[done] wrote {path}", flush=True)

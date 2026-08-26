# -*- coding: utf-8 -*-
"""Relational-versus-traffic reconciliation for the 200 km border designs.

The published geographic checks use positive relational cause cells only.  This
targeted script holds the prefecture sample, matching rule, time window, treatment
terms, and fixed effects fixed while putting relational totals, traffic totals,
and their exact difference on one balanced prefecture-month support.

Output: output/ext2124/border_placebo_reconciliation.csv
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
OUT = str(_REP_PROJECT / "output" / "ext2124")
START, END, POST0 = "2017-01", "2019-03", "2018-09"
D_KM = 200.0
ROWS = []


def haversine(la1, lo1, la2, lo2):
    radius = 6371.0
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    a = (
        np.sin((la2 - la1) / 2) ** 2
        + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    )
    return 2 * radius * np.arcsin(np.sqrt(a))


def fit(spec, sample, outcome, formula, data, sample_note):
    m1 = pf.feols(formula, data=data, vcov={"CRV1": "prov_id"})
    m3 = pf.feols(formula, data=data, vcov={"CRV3": "prov_id"})
    pw = wild_score_p(formula, data, "pth")
    row = {
        "spec": spec,
        "sample": sample,
        "outcome": outcome,
        "coefficient": float(m1.coef()["pth"]),
        "std_error_crv1": float(m1.se()["pth"]),
        "p_crv1": float(m1.pvalue()["pth"]),
        "std_error_crv3": float(m3.se()["pth"]),
        "p_crv3": float(m3.pvalue()["pth"]),
        "p_wild": float(pw),
        "n_obs": int(m1._N),
        "province_clusters": int(data["prov_id"].nunique()),
        "prefectures": int(data["prefecture_code"].nunique()),
        "sample_note": sample_note,
        "formula": formula,
    }
    ROWS.append(row)
    print(
        f"{spec:20s} {outcome:12s} b={row['coefficient']:+.6f} "
        f"se={row['std_error_crv1']:.6f} wild={pw:.4f} N={row['n_obs']:,}",
        flush=True,
    )


raw = pd.read_parquet(f"{DATA}/civil_panel.parquet")
raw["month"] = raw["jmonth"].astype(str).str[:7]
raw = raw[raw["cause_family"].isin(["relational", "placebo"])].copy()
raw["group"] = np.where(raw["cause_family"].eq("relational"), "relational", "traffic")

exposure = pd.read_parquet(f"{DATA}/exposure_v2.parquet")[[
    "prefecture_code", "province", "exposure_v2_z"
]].drop_duplicates()
schedule = (
    pd.read_parquet(f"{DATA}/panel_month.parquet")[["province", "inspection_round"]]
    .drop_duplicates()
)
support = (
    raw[raw["month"].between("2014-01", "2017-12")][
        ["prefecture_code", "province"]
    ].drop_duplicates()
    .merge(exposure[["prefecture_code", "province"]],
           on=["prefecture_code", "province"], how="inner")
)
months = pd.DataFrame({"month": pd.period_range(START, END, freq="M").astype(str)})
counts = (
    raw[raw["month"].between(START, END)]
    .groupby(["prefecture_code", "province", "month", "group"], as_index=False)["n_cases"]
    .sum()
    .pivot(index=["prefecture_code", "province", "month"], columns="group", values="n_cases")
    .reset_index()
)
panel = support.merge(months, how="cross").merge(
    counts, on=["prefecture_code", "province", "month"], how="left"
)
for col in ("relational", "traffic"):
    panel[col] = panel[col].fillna(0).astype(float)
    panel[f"y_{col}"] = np.arcsinh(panel[col])
panel["y_gap"] = panel["y_relational"] - panel["y_traffic"]
panel = (
    panel.merge(schedule, on="province")
    .merge(exposure[["prefecture_code", "exposure_v2_z"]], on="prefecture_code")
)
panel["treat"] = (panel["inspection_round"] == 1).astype(int)
panel["postc"] = (panel["month"] >= POST0).astype(int)
panel["pt"] = panel["postc"] * panel["treat"]
panel["ph"] = panel["postc"] * panel["exposure_v2_z"]
panel["pth"] = panel["pt"] * panel["exposure_v2_z"]
panel["prov_id"] = pd.factorize(panel["province"])[0]
panel["pcode2"] = panel["prefecture_code"].str[:2]


# Reproduce the paper's centroid-based nearest-across-province matching rule.
centroids = pd.read_csv(f"{DATA}/pref_centroids.csv", dtype={"prefecture_code": str})
pref = panel[["prefecture_code", "pcode2", "treat"]].drop_duplicates("prefecture_code")
pref = pref.merge(centroids, on="prefecture_code", how="left").dropna(subset=["lat", "lon"])
treated = pref[pref["treat"].eq(1)].reset_index(drop=True)
control = pref[pref["treat"].eq(0)].reset_index(drop=True)
distance = haversine(
    np.asarray(treated["lat"], float)[:, None],
    np.asarray(treated["lon"], float)[:, None],
    np.asarray(control["lat"], float)[None, :],
    np.asarray(control["lon"], float)[None, :],
)
distance = np.where(
    np.asarray(treated["pcode2"], object)[:, None]
    != np.asarray(control["pcode2"], object)[None, :],
    distance,
    np.inf,
)
t_near = distance.min(axis=1)
c_near = distance.min(axis=0)
treated_border = set(treated.loc[t_near <= D_KM, "prefecture_code"])
control_border = set(control.loc[c_near <= D_KM, "prefecture_code"])
border_panel = panel[panel["prefecture_code"].isin(treated_border | control_border)].copy()

pairs = set()
t_index = np.argmin(distance, axis=1)
for i in range(len(treated)):
    if t_near[i] <= D_KM:
        pairs.add((treated.loc[i, "prefecture_code"], control.loc[t_index[i], "prefecture_code"]))
c_index = np.argmin(distance, axis=0)
for j in range(len(control)):
    if c_near[j] <= D_KM:
        pairs.add((treated.loc[c_index[j], "prefecture_code"], control.loc[j, "prefecture_code"]))

stack = []
for pair_id, (a, b) in enumerate(sorted(pairs)):
    segment = panel[panel["prefecture_code"].isin([a, b])].copy()
    segment["pair_pref"] = str(pair_id) + "_" + segment["prefecture_code"]
    segment["pair_month"] = str(pair_id) + "_" + segment["month"]
    stack.append(segment)
pair_panel = pd.concat(stack, ignore_index=True)

samples = (
    ("full", panel, "y_{out} ~ pth + ph + pt | prefecture_code + month",
     "balanced full clean-window support"),
    ("pborder_200km", border_panel,
     "y_{out} ~ pth + ph + pt | prefecture_code + month",
     f"centroid distance <= {D_KM:.0f} km from nearest differently timed cross-province prefecture"),
    ("dlr_pairs_200km", pair_panel,
     "y_{out} ~ pth + ph + pt | pair_pref + pair_month",
     f"{len(pairs)} nearest differently timed cross-province pairs within {D_KM:.0f} km"),
)
for sample_name, data, template, note in samples:
    for short, yname in (("relational", "relational"), ("traffic", "traffic"), ("gap", "gap")):
        fit(
            f"{sample_name}_{short}", sample_name, short,
            template.format(out=yname), data, note,
        )

os.makedirs(OUT, exist_ok=True)
pd.DataFrame(ROWS).to_csv(f"{OUT}/border_placebo_reconciliation.csv", index=False)
print("written border_placebo_reconciliation.csv", flush=True)

# -*- coding: utf-8 -*-
"""Joint randomization-inference audit after restoring zero count cells.

This script is a follow-up to 81_criminal_specs.py. It keeps the two
content-share measures fixed and varies only the enforcement-count support or
transformation. It reports free wave-timing permutation and the more conservative
within-pre-campaign-crime-tercile permutation. No paper or baseline output is patched.
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
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf
from pyfixest.estimation.internals.demean_ import demean_within

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "output"
REPS = int(os.environ.get("HWIH_RI_REPS", "9999"))
SEED = int(os.environ.get("HWIH_RI_SEED", "20260715"))
BATCH = int(os.environ.get("HWIH_RI_BATCH", "200"))


def month_number(s: pd.Series) -> np.ndarray:
    d = pd.to_datetime(s)
    return (d.dt.year * 12 + d.dt.month).to_numpy(np.int32)


def prepare(d: pd.DataFrame, ycols: list[str], weight: str | None,
            controls: list[str] | None = None):
    controls = controls or []
    need = ycols + ["prefecture_code", "province", "month", "H"]
    need += controls
    if weight:
        need.append(weight)
    x = d.dropna(subset=need).reset_index(drop=True).copy()
    fe = np.column_stack([
        pd.factorize(x["prefecture_code"])[0],
        pd.factorize(x["province"] + "_" + x["month"])[0],
    ]).astype(np.uint32)
    w = x[weight].to_numpy(float) if weight else None
    wr = np.ones(len(x), dtype=float) if w is None else w
    C = None
    Cbread = None
    if controls:
        c0 = x[controls].to_numpy(float)
        C, ok, _ = demean_within(c0, fe, w)
        if not ok:
            raise RuntimeError("demeaning failed for controls")
        Cbread = np.linalg.pinv(C.T @ (wr[:, None] * C))
    yr = {}
    for y in ycols:
        z, ok, _ = demean_within(x[y].to_numpy(float)[:, None], fe, w)
        if not ok:
            raise RuntimeError(f"demeaning failed for {y}")
        if C is not None:
            z = z - C @ (Cbread @ (C.T @ (wr[:, None] * z)))
        yr[y] = z[:, 0]
    return {
        "data": x,
        "fe": fe,
        "weights": w,
        "wr": wr,
        "yr": yr,
        "C": C,
        "Cbread": Cbread,
        "province": x["province"].to_numpy(),
        "month_num": x["month_num"].to_numpy(np.int32),
        "H": x["H"].to_numpy(float),
    }


def coef_for_x(obj, xmat: np.ndarray, y: str) -> np.ndarray:
    xr, ok, _ = demean_within(xmat, obj["fe"], obj["weights"])
    if not ok:
        raise RuntimeError("demeaning failed for treatment matrix")
    wr = obj["wr"][:, None]
    if obj["C"] is not None:
        C = obj["C"]
        xr = xr - C @ (obj["Cbread"] @ (C.T @ (wr * xr)))
    numerator = (wr * obj["yr"][y][:, None] * xr).sum(axis=0)
    denominator = (wr * xr * xr).sum(axis=0)
    return numerator / denominator


def generate_assignments(rng: np.random.Generator, scheme: str, n: int,
                         insp_vals: np.ndarray, terc: np.ndarray) -> np.ndarray:
    ans = np.empty((n, len(insp_vals)), dtype=np.int32)
    groups = [np.where(terc == t)[0] for t in np.unique(terc)]
    for r in range(n):
        if scheme == "free":
            ans[r] = rng.permutation(insp_vals)
        else:
            ans[r] = insp_vals.copy()
            for g in groups:
                ans[r, g] = rng.permutation(insp_vals[g])
    return ans


def x_from_assignments(obj, assignments: np.ndarray,
                       prov_lookup: dict[str, int]) -> np.ndarray:
    pidx = np.array([prov_lookup[p] for p in obj["province"]], dtype=int)
    assigned_by_row = assignments[:, pidx].T
    return ((obj["month_num"][:, None] >= assigned_by_row).astype(float)
            * obj["H"][:, None])


raw = pd.read_parquet(DATA / "crim_panel_v2.parquet")
raw["month"] = pd.to_datetime(raw["jmonth"]).dt.to_period("M").astype(str)
raw["month_num"] = month_number(raw["jmonth"])
raw["H"] = raw["exposure_v2_z"]

back = raw[(raw["family"] == "market") & (raw["n_cases"] > 0)].copy()
det = raw[(raw["family"] == "enforcementcrime") & (raw["n_cases"] > 0)].copy()
pos = det.copy()
pos["asinh_pos"] = np.arcsinh(pos["n_cases"])

meta = (raw[["prefecture_code", "province", "exposure_v2_z", "insp_month"]]
        .drop_duplicates("prefecture_code"))
months = pd.DataFrame({
    "jmonth": pd.date_range(pd.to_datetime(raw["jmonth"]).min(),
                             pd.to_datetime(raw["jmonth"]).max(), freq="MS")
})
full = meta.assign(_k=1).merge(months.assign(_k=1), on="_k").drop(columns="_k")
obs = (raw[raw["family"] == "enforcementcrime"]
       [["prefecture_code", "jmonth", "n_cases"]].copy())
obs["jmonth"] = pd.to_datetime(obs["jmonth"])
full = full.merge(obs, on=["prefecture_code", "jmonth"], how="left")
full["n_cases"] = full["n_cases"].fillna(0.0)
full["month"] = full["jmonth"].dt.to_period("M").astype(str)
full["month_num"] = month_number(full["jmonth"])
full["H"] = full["exposure_v2_z"]
full["asinh_full"] = np.arcsinh(full["n_cases"])
full["log1p_full"] = np.log1p(full["n_cases"])
full["any_full"] = (full["n_cases"] > 0).astype(float)

objects = {
    "back": prepare(back, ["y_backstop"], "n_cases"),
    "det": prepare(det, ["y_detention_debt"], "n_cases"),
    "back_doc": prepare(back, ["y_backstop"], "n_cases", ["x_doclen"]),
    "det_doc": prepare(det, ["y_detention_debt"], "n_cases", ["x_doclen"]),
    "pos": prepare(pos, ["asinh_pos"], None),
    "full": prepare(full, ["asinh_full", "log1p_full", "any_full"], None),
}

sched = (raw[["province", "insp_month"]].drop_duplicates()
         .sort_values("province").reset_index(drop=True))
provs = sched["province"].to_numpy()
prov_lookup = {p: i for i, p in enumerate(provs)}
insp_vals = month_number(sched["insp_month"])
crime = (pd.read_parquet(DATA / "exposure_v2.parquet")
         .groupby("province")["exposure_v2_z"].mean().reindex(provs))
terc = pd.qcut(crime, 3, labels=[0, 1, 2]).astype(int).to_numpy()

observed_assignment = insp_vals[None, :]
observed = {
    "back": coef_for_x(objects["back"],
                       x_from_assignments(objects["back"], observed_assignment, prov_lookup),
                       "y_backstop")[0],
    "det": coef_for_x(objects["det"],
                      x_from_assignments(objects["det"], observed_assignment, prov_lookup),
                      "y_detention_debt")[0],
    "back_doc": coef_for_x(objects["back_doc"],
                           x_from_assignments(objects["back_doc"], observed_assignment, prov_lookup),
                           "y_backstop")[0],
    "det_doc": coef_for_x(objects["det_doc"],
                          x_from_assignments(objects["det_doc"], observed_assignment, prov_lookup),
                          "y_detention_debt")[0],
    "positive_asinh": coef_for_x(objects["pos"],
                                 x_from_assignments(objects["pos"], observed_assignment, prov_lookup),
                                 "asinh_pos")[0],
    "complete_asinh": coef_for_x(objects["full"],
                                 x_from_assignments(objects["full"], observed_assignment, prov_lookup),
                                 "asinh_full")[0],
    "complete_log1p": coef_for_x(objects["full"],
                                 x_from_assignments(objects["full"], observed_assignment, prov_lookup),
                                 "log1p_full")[0],
    "complete_any": coef_for_x(objects["full"],
                               x_from_assignments(objects["full"], observed_assignment, prov_lookup),
                               "any_full")[0],
}
print("observed coefficients", {k: round(v, 5) for k, v in observed.items()}, flush=True)

variants = {
    "J00_positive_asinh": ("back", "det", "positive_asinh"),
    "J01_complete_asinh": ("back", "det", "complete_asinh"),
    "J02_complete_log1p": ("back", "det", "complete_log1p"),
    "J03_complete_any": ("back", "det", "complete_any"),
    "J04_baseline_controls_positive_asinh": ("back_doc", "det_doc", "positive_asinh"),
    "J05_baseline_controls_complete_any": ("back_doc", "det_doc", "complete_any"),
}
rows = []

for scheme_i, scheme in enumerate(["free", "stratified_crime_tercile"]):
    rng = np.random.default_rng(SEED + scheme_i)
    draws = {k: np.empty(REPS, dtype=float) for k in observed}
    done = 0
    while done < REPS:
        n = min(BATCH, REPS - done)
        a = generate_assignments(rng, scheme, n, insp_vals, terc)
        xb = x_from_assignments(objects["back"], a, prov_lookup)
        xd = x_from_assignments(objects["det"], a, prov_lookup)
        xbd = x_from_assignments(objects["back_doc"], a, prov_lookup)
        xdd = x_from_assignments(objects["det_doc"], a, prov_lookup)
        xp = x_from_assignments(objects["pos"], a, prov_lookup)
        xf = x_from_assignments(objects["full"], a, prov_lookup)
        sl = slice(done, done + n)
        draws["back"][sl] = coef_for_x(objects["back"], xb, "y_backstop")
        draws["det"][sl] = coef_for_x(objects["det"], xd, "y_detention_debt")
        draws["back_doc"][sl] = coef_for_x(objects["back_doc"], xbd, "y_backstop")
        draws["det_doc"][sl] = coef_for_x(objects["det_doc"], xdd, "y_detention_debt")
        draws["positive_asinh"][sl] = coef_for_x(objects["pos"], xp, "asinh_pos")
        draws["complete_asinh"][sl] = coef_for_x(objects["full"], xf, "asinh_full")
        draws["complete_log1p"][sl] = coef_for_x(objects["full"], xf, "log1p_full")
        draws["complete_any"][sl] = coef_for_x(objects["full"], xf, "any_full")
        done += n
        if done % 1000 < BATCH or done == REPS:
            print(f"{scheme}: {done}/{REPS}", flush=True)

    for spec_id, (back_key, det_key, count_key) in variants.items():
        keys = [back_key, det_key, count_key]
        b = np.array([observed[k] for k in keys])
        B = np.column_stack([draws[k] for k in keys])
        mu = B.mean(axis=0)
        sd = B.std(axis=0, ddof=1)
        z_obs = -(b - mu) / sd
        Z = -(B - mu) / sd
        p_ind = [(1 + np.sum(Z[:, j] >= z_obs[j])) / (1 + REPS) for j in range(3)]
        t_obs = float(z_obs.sum())
        t_draw = Z.sum(axis=1)
        p_eq = float((1 + np.sum(t_draw >= t_obs)) / (1 + REPS))
        cov = np.cov(Z, rowvar=False)
        w = np.linalg.solve(cov, np.ones(3))
        p_gls = float((1 + np.sum(Z @ w >= z_obs @ w)) / (1 + REPS))
        def subset_p(cols):
            cols = list(cols)
            return float((1 + np.sum(Z[:, cols].sum(axis=1) >= z_obs[cols].sum()))
                         / (1 + REPS))
        rows.append({
            "spec_id": spec_id,
            "scheme": scheme,
            "count_variant": count_key,
            "b_backstop": b[0],
            "b_detention": b[1],
            "b_count": b[2],
            "p_backstop": p_ind[0],
            "p_detention": p_ind[1],
            "p_count": p_ind[2],
            "joint_p_equal": p_eq,
            "joint_p_gls": p_gls,
            "p_backstop_plus_detention": subset_p([0, 1]),
            "p_backstop_plus_count": subset_p([0, 2]),
            "p_detention_plus_count": subset_p([1, 2]),
            "reps": REPS,
        })
        print(f"{scheme:27s} {spec_id:22s} joint eq={p_eq:.4f} gls={p_gls:.4f} "
              f"individual={np.round(p_ind,4)} pairs="
              f"{np.round([subset_p([0,1]), subset_p([0,2]), subset_p([1,2])],4)}",
              flush=True)

out = pd.DataFrame(rows)
path = OUT / "criminal_joint_ri.csv"
out.to_csv(path, index=False, encoding="utf-8-sig")
print(f"[done] wrote {path}", flush=True)

# -*- coding: utf-8 -*-
"""Studentized wave-timing randomization inference for the three criminal margins.

Motivation. 82_joint_ri.py ranks raw coefficients across permuted
inspection schedules (its z-transform uses one global mean/sd, a monotone map,
so per-outcome p-values equal raw-coefficient RI). With staggered timing the
precision of the estimated dose coefficient varies sharply across hypothetical
schedules, and the standard recommendation (Young 2019; MacKinnon & Webb 2020;
Imbens & Rosenbaum 2005) is to permute a studentized statistic. This script
recomputes the same draws (same seed, batching, and assignment generation as
82) and reports BOTH statistics side by side:

  raw:         b_r ranked against b_obs                (reproduces 82)
  studentized: t_r = b_r / se_CRV1(b_r) ranked vs t_obs (correction candidate)

Specs are the paper's Table 3 family (share outcomes weighted by n_cases with
the document-length control; balanced any-case panel) plus the no-control
variants. Nothing in the paper is patched by this script.
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
    gprov = pd.factorize(x["province"])[0]
    return {
        "data": x,
        "fe": fe,
        "weights": w,
        "wr": wr,
        "yr": yr,
        "C": C,
        "Cbread": Cbread,
        "province": x["province"].to_numpy(),
        "gprov": gprov,
        "n_prov": int(gprov.max() + 1),
        "month_num": x["month_num"].to_numpy(np.int32),
        "H": x["H"].to_numpy(float),
    }


def resid_x(obj, xmat: np.ndarray) -> np.ndarray:
    xr, ok, _ = demean_within(xmat, obj["fe"], obj["weights"])
    if not ok:
        raise RuntimeError("demeaning failed for treatment matrix")
    wr = obj["wr"][:, None]
    if obj["C"] is not None:
        C = obj["C"]
        xr = xr - C @ (obj["Cbread"] @ (C.T @ (wr * xr)))
    return xr


def coef_and_t(obj, xmat: np.ndarray, y: str):
    """Per-column FWL coefficient and CRV1-studentized t (dof factors constant
    across draws, so they cancel in permutation ranking)."""
    xr = resid_x(obj, xmat)
    wr = obj["wr"][:, None]
    yv = obj["yr"][y][:, None]
    sxx = (wr * xr * xr).sum(axis=0)
    b = (wr * yv * xr).sum(axis=0) / sxx
    e = yv - xr * b[None, :]
    m = wr * xr * e
    G = obj["n_prov"]
    S = np.zeros((G, m.shape[1]))
    np.add.at(S, obj["gprov"], m)
    V = (S * S).sum(axis=0) / (sxx * sxx)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = b / np.sqrt(V)
    return b, t


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
full["any_full"] = (full["n_cases"] > 0).astype(float)

objects = {
    "back": prepare(back, ["y_backstop"], "n_cases"),
    "det": prepare(det, ["y_detention_debt"], "n_cases"),
    "back_doc": prepare(back, ["y_backstop"], "n_cases", ["x_doclen"]),
    "det_doc": prepare(det, ["y_detention_debt"], "n_cases", ["x_doclen"]),
    "full": prepare(full, ["any_full"], None),
}
ykey = {"back": "y_backstop", "det": "y_detention_debt",
        "back_doc": "y_backstop", "det_doc": "y_detention_debt",
        "full": "any_full"}

sched = (raw[["province", "insp_month"]].drop_duplicates()
         .sort_values("province").reset_index(drop=True))
provs = sched["province"].to_numpy()
prov_lookup = {p: i for i, p in enumerate(provs)}
insp_vals = month_number(sched["insp_month"])
crime = (pd.read_parquet(DATA / "exposure_v2.parquet")
         .groupby("province")["exposure_v2_z"].mean().reindex(provs))
terc = pd.qcut(crime, 3, labels=[0, 1, 2]).astype(int).to_numpy()

observed_assignment = insp_vals[None, :]
obs_b = {}
obs_t = {}
for k, o in objects.items():
    b, t = coef_and_t(o, x_from_assignments(o, observed_assignment, prov_lookup),
                      ykey[k])
    obs_b[k], obs_t[k] = float(b[0]), float(t[0])
    print(f"observed {k:10s} b={obs_b[k]:+.6f} t={obs_t[k]:+.3f}", flush=True)

variants = {
    "J03_complete_any": ("back", "det", "full"),
    "J05_baseline_controls_complete_any": ("back_doc", "det_doc", "full"),
}
rows = []

for scheme_i, scheme in enumerate(["free", "stratified_crime_tercile"]):
    rng = np.random.default_rng(SEED + scheme_i)
    draws_b = {k: np.empty(REPS, dtype=float) for k in objects}
    draws_t = {k: np.empty(REPS, dtype=float) for k in objects}
    done = 0
    while done < REPS:
        n = min(BATCH, REPS - done)
        a = generate_assignments(rng, scheme, n, insp_vals, terc)
        sl = slice(done, done + n)
        for k, o in objects.items():
            xm = x_from_assignments(o, a, prov_lookup)
            b, t = coef_and_t(o, xm, ykey[k])
            draws_b[k][sl] = b
            draws_t[k][sl] = t
        done += n
        if done % 1000 < BATCH or done == REPS:
            print(f"{scheme}: {done}/{REPS}", flush=True)

    for spec_id, keys in variants.items():
        rec = {"spec_id": spec_id, "scheme": scheme, "reps": REPS}
        # one-sided in the model direction (decline) for each statistic
        zs_obs, zs_draw = [], []
        for label, k in zip(["backstop", "detention", "count"], keys):
            B, T = draws_b[k], draws_t[k]
            rec[f"b_{label}"] = obs_b[k]
            rec[f"t_{label}"] = obs_t[k]
            rec[f"p_raw_{label}"] = float(
                (1 + np.sum(B <= obs_b[k])) / (1 + REPS))
            rec[f"p_stud_{label}"] = float(
                (1 + np.sum(T <= obs_t[k])) / (1 + REPS))
            mu, sd = T.mean(), T.std(ddof=1)
            zs_obs.append(-(obs_t[k] - mu) / sd)
            zs_draw.append(-(T - mu) / sd)
        Z = np.column_stack(zs_draw)
        z0 = np.array(zs_obs)
        rec["joint_p_stud_equal"] = float(
            (1 + np.sum(Z.sum(axis=1) >= z0.sum())) / (1 + REPS))
        cov = np.cov(Z, rowvar=False)
        w = np.linalg.solve(cov, np.ones(3))
        rec["joint_p_stud_gls"] = float(
            (1 + np.sum(Z @ w >= z0 @ w)) / (1 + REPS))
        Bz = np.column_stack([-(draws_b[k] - draws_b[k].mean()) / draws_b[k].std(ddof=1)
                              for k in keys])
        b0 = np.array([-(obs_b[k] - draws_b[k].mean()) / draws_b[k].std(ddof=1)
                       for k in keys])
        rec["joint_p_raw_equal"] = float(
            (1 + np.sum(Bz.sum(axis=1) >= b0.sum())) / (1 + REPS))
        rows.append(rec)
        print(f"{scheme:26s} {spec_id:36s} "
              f"raw=({rec['p_raw_backstop']:.4f},{rec['p_raw_detention']:.4f},"
              f"{rec['p_raw_count']:.4f}) "
              f"stud=({rec['p_stud_backstop']:.4f},{rec['p_stud_detention']:.4f},"
              f"{rec['p_stud_count']:.4f}) "
              f"joint raw={rec['joint_p_raw_equal']:.4f} "
              f"stud={rec['joint_p_stud_equal']:.4f} "
              f"gls={rec['joint_p_stud_gls']:.4f}", flush=True)

out = pd.DataFrame(rows)
path = OUT / "ri_studentized_criminal.csv"
out.to_csv(path, index=False, encoding="utf-8-sig")
print(f"[done] wrote {path}", flush=True)

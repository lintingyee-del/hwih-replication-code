# -*- coding: utf-8 -*-
"""Event-study diagnostics for bounded criminal specifications.

Checks whether the promising extensive-margin count and short-window content-share
results have credible pre-period paths. This is diagnostic only and does not patch TeX.
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

from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "output"
BINS = [(-24, -19), (-18, -13), (-12, -7), (-6, -1),
        (0, 5), (6, 11), (12, 17), (18, 28)]
REF = (-6, -1)


def wild_joint_lead_p(x, y, leads, post_terms, weights=None, controls="",
                      reps=9999, seed=20260715):
    """Null-imposed wild cluster score test for all lead coefficients."""
    aux_terms = list(post_terms) + ([controls] if controls else [])
    aux = " + ".join(aux_terms) if aux_terms else "1"
    need = [y, "pref", "prov_month", "prov_id"] + leads + aux_terms
    if weights:
        need.append(weights)
    z = x.dropna(subset=list(dict.fromkeys(need))).reset_index(drop=True)
    my = pf.feols(f"{y} ~ {aux} | pref + prov_month", data=z, weights=weights,
                  fixef_rm="none")
    u = np.asarray(my.resid()).ravel()
    xt = []
    for lead in leads:
        mx = pf.feols(f"{lead} ~ {aux} | pref + prov_month", data=z,
                      weights=weights, fixef_rm="none")
        xt.append(np.asarray(mx.resid()).ravel())
    X = np.column_stack(xt)
    w = z[weights].to_numpy(float) if weights else np.ones(len(z))
    g = pd.factorize(z["prov_id"])[0]
    scores = np.zeros((g.max() + 1, len(leads)))
    for j in range(len(leads)):
        np.add.at(scores[:, j], g, w * X[:, j] * u)
    S = scores.sum(axis=0)
    V = scores.T @ scores
    Vi = np.linalg.pinv(V)
    T = float(S @ Vi @ S)
    W = np.random.default_rng(seed).choice([-1.0, 1.0], size=(reps, len(scores)))
    Sb = W @ scores
    Tb = np.einsum("bi,ij,bj->b", Sb, Vi, Sb)
    return float((1 + np.sum(Tb >= T)) / (1 + reps))


def fit_es(tag, d, y, weights=None, controls=""):
    x = d[d["event_time"].between(-24, 28)].copy()
    terms = []
    for lo, hi in BINS:
        if (lo, hi) == REF:
            continue
        nm = f"b_{lo}_{hi}".replace("-", "m")
        x[nm] = x["event_time"].between(lo, hi).astype(float) * x["H"]
        terms.append(nm)
    rhs = " + ".join(terms) + (f" + {controls}" if controls else "")
    fml = f"{y} ~ {rhs} | pref + prov_month"
    m = pf.feols(fml, data=x, vcov={"CRV1": "prov_id"}, weights=weights)
    names = list(m.coef().index)
    leads = terms[:3]
    idx = [names.index(t) for t in leads]
    b = m.coef()[leads].to_numpy()
    V = m._vcov[np.ix_(idx, idx)]
    stat = float(b @ np.linalg.solve(V, b))
    pre_p = float(1 - stats.chi2.cdf(stat, len(leads)))
    post_terms = terms[3:]
    pre_wild_p = wild_joint_lead_p(
        x, y, leads, post_terms, weights=weights, controls=controls
    )
    out = []
    for term in terms:
        lo, hi = term[2:].replace("m", "-").rsplit("_", 1)
        out.append({
            "outcome": tag,
            "bin_lo": int(lo),
            "bin_hi": int(hi),
            "estimate": float(m.coef()[term]),
            "std_error": float(m.se()[term]),
            "p_value": float(m.pvalue()[term]),
            "pretrend_joint_p": pre_p,
            "pretrend_wild_p": pre_wild_p,
            "n_obs": int(m._N),
        })
    print(f"{tag}: pretrend joint p={pre_p:.3f}; wild-score joint p={pre_wild_p:.3f}; "
          f"N={int(m._N):,}")
    print(pd.DataFrame(out)[["bin_lo", "bin_hi", "estimate", "std_error"]]
          .round(4).to_string(index=False))
    return out


raw = pd.read_parquet(DATA / "crim_panel_v2.parquet")
raw["month"] = pd.to_datetime(raw["jmonth"]).dt.to_period("M").astype(str)
raw["H"] = raw["exposure_v2_z"]
raw["log_doclen"] = np.log1p(raw["x_doclen"])
raw["pref"] = raw["prefecture_code"].astype(str)
raw["prov_month"] = raw["province"] + "_" + raw["month"]
raw["prov_id"] = pd.factorize(raw["province"])[0]

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
full["insp"] = pd.to_datetime(full["insp_month"])
full["event_time"] = ((full["jmonth"].dt.year - full["insp"].dt.year) * 12
                      + full["jmonth"].dt.month - full["insp"].dt.month)
full["H"] = full["exposure_v2_z"]
full["pref"] = full["prefecture_code"].astype(str)
full["prov_month"] = full["province"] + "_" + full["month"]
full["prov_id"] = pd.factorize(full["province"])[0]
full["asinh_n"] = np.arcsinh(full["n_cases"])
full["any_n"] = (full["n_cases"] > 0).astype(float)

rows = []
rows += fit_es("backstop_raw_doc", back, "y_backstop", "n_cases", "x_doclen")
rows += fit_es("backstop_no_doc", back, "y_backstop", "n_cases")
rows += fit_es("detention_raw_doc", det, "y_detention_debt", "n_cases", "x_doclen")
rows += fit_es("detention_log_doc", det, "y_detention_debt", "n_cases", "log_doclen")
rows += fit_es("detention_no_doc", det, "y_detention_debt", "n_cases")
rows += fit_es("count_complete_asinh", full, "asinh_n")
rows += fit_es("count_complete_any", full, "any_n")

path = OUT / "criminal_eventstudy.csv"
pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
print(f"[done] wrote {path}")

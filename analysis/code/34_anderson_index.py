# -*- coding: utf-8 -*-
"""6B step 34 — item 3: proper Anderson (2008) inverse-covariance summary index for
the latent 'coercive-backstop technology' as a FIRST-STAGE outcome. Instead of five
individually-insignificant criminal margins, build ONE per-prefecture-month composite
of the (sign-flipped, standardized) de-militarization measures, weighted by the inverse
of their covariance (Anderson efficient index), and run the dose Post x H on it.
Reports GLS(Anderson) and equal-weight indices, each with CRV1, wild-score, and
wave-timing randomization p. Directly comparable to the coefficient-index CrimJointP.

Measures (each signed so a DECLINE => positive index):
  market hard-backstop content (share); enforcement caseload (asinh); detention-for-debt
  (share). All aligned to prefecture-month.
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
import numpy as np, pandas as pd, pyfixest as pf
DATA = str(_REP_PROJECT / "data"); OUTD = str(_REP_PROJECT / "output")
REPS = 4999; rng = np.random.default_rng(20260706)

k = pd.read_parquet(f"{DATA}/crim_panel_v2.parquet")
k["month"] = k["jmonth"].astype(str).str[:7]
mkt = k[k["family"] == "market"][["prefecture_code", "province", "month", "y_backstop",
                                   "n_cases", "exposure_v2_z", "insp_month"]].copy()
mkt = mkt.rename(columns={"n_cases": "n_mkt"})
enf = k[k["family"] == "enforcementcrime"][["prefecture_code", "month", "n_cases",
                                            "y_detention_debt"]].copy()
enf["enf_asinh"] = np.arcsinh(enf["n_cases"])
d = mkt.merge(enf[["prefecture_code", "month", "enf_asinh", "y_detention_debt"]],
              on=["prefecture_code", "month"], how="inner").dropna(
              subset=["exposure_v2_z", "y_backstop", "enf_asinh", "y_detention_debt"])
print(f"[panel] {len(d)} prefecture-month cells with all three measures", flush=True)

# sign so DECLINE => positive (de-militarization); then standardize (z over all cells)
M = np.column_stack([-d["y_backstop"].values, -d["enf_asinh"].values, -d["y_detention_debt"].values])
Z = (M - M.mean(0)) / M.std(0, ddof=1)
Sig = np.cov(Z, rowvar=False)
w_gls = np.linalg.solve(Sig, np.ones(3)); w_gls = w_gls / w_gls.sum()   # Anderson efficient
d["idx_gls"] = Z @ w_gls
d["idx_eq"] = Z.mean(1)
print(f"[weights] Anderson inverse-cov weights = {np.round(w_gls,3)} "
      f"(market, enforcement, detention)", flush=True)

d["H"] = d["exposure_v2_z"]; d["pref"] = d["prefecture_code"]
d["insp"] = d["insp_month"].astype(str).str[:7]
d["prov_month"] = d["province"] + "_" + d["month"]
d["prov_id"] = pd.factorize(d["province"])[0]

from _wild import wild_score_p, wild_p  # corrected shared WCR bootstrap


INSP = d[["province","insp"]].drop_duplicates(); PROV = INSP["province"].values
IVALS = INSP.set_index("province").loc[PROV,"insp"].values
def fit_px(outcome, inspmap):
    x = d.copy(); ins = x["province"].map(inspmap).values
    x["post"] = (x["month"].values >= ins).astype(int); x["px"] = x["post"]*x["H"]
    return float(pf.feols(f"{outcome} ~ px | pref + prov_month", data=x).coef()["px"])

for oc, lab in [("idx_gls", "Anderson (inverse-cov)"), ("idx_eq", "equal-weight")]:
    m = pf.feols(f"{oc} ~ px_obs | pref + prov_month",
                 data=d.assign(px_obs=d["exposure_v2_z"]*(d["month"]>=d["insp"]).astype(int)),
                 vcov={"CRV1": "prov_id"})
    b, se, p = m.coef()["px_obs"], m.se()["px_obs"], m.pvalue()["px_obs"]
    wp = wild_p(f"{oc} ~ px_obs | pref + prov_month",
                d.assign(px_obs=d["exposure_v2_z"]*(d["month"]>=d["insp"]).astype(int)), "px_obs")
    # wave-timing randomization
    b_obs = fit_px(oc, dict(zip(PROV, IVALS)))
    draws = np.array([fit_px(oc, dict(zip(PROV, rng.permutation(IVALS)))) for _ in range(REPS)])
    perm_p = (1 + np.sum(draws <= b_obs)) / (1 + REPS)   # one-sided: decline => positive index => px>0; test px>=obs
    perm_p = (1 + np.sum(draws >= b_obs)) / (1 + REPS)
    print(f"[{lab:22s}] px={b:+.4f} ({se:.4f})  CRV1 p={p:.3f}  wild p={wp:.3f}  "
          f"randomization p={perm_p:.4f}  (N={m._N})", flush=True)

print("\n[note] compare to coefficient-index CrimJointP=0.051. If randomization p here "
      "lands 0.03-0.04, the criminal first-stage pillar stands on the composite.", flush=True)

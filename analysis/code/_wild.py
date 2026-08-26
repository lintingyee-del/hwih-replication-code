# -*- coding: utf-8 -*-
"""Corrected null-imposed wild-score cluster bootstrap (WCR), shared across 6B scripts.

Fixes a row-misalignment bug in the previous inline copies. The old version
residualized y and x in two separate feols calls, truncated both to the shorter
length with [:n], and then pulled cluster ids (and weights) from the *original*
df's first n rows. Because feols drops singleton fixed-effect groups by default
(and micro-level LHS variables can be NaN), the residual vectors, weights, and
cluster ids were silently misaligned. That scrambled the per-cluster scores and
produced spurious wild p-values in BOTH directions (fake significance where scores
happened to align in sign, fake insignificance elsewhere).

Fix: assemble ONE common estimation sample (drop rows missing any variable the fit
uses), residualize on it with singleton removal disabled (fixef_rm="none") so the
residuals align 1:1 with that sample, and take weights and cluster ids from the
same frame. Singletons kept this way receive residual 0 and contribute nothing to
any cluster score, so the point estimate is identical to the singleton-dropped fit
(verified: the FWL coefficient recovered from these residuals reproduces the feols
coefficient to machine precision, weighted and unweighted).

Method is unchanged: null-imposed (restricted) wild score bootstrap with Rademacher
weights on per-cluster scores; the studentizing denominator is invariant to the
sign flips, so comparing |sum s_g| to |sum W_g s_g| is the studentized WCR test.
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
import numpy as np
import pandas as pd
import pyfixest as pf


def wild_score_p(fml, df, coef, weights=None, cluster="prov_id", reps=9_999, seed=42):
    lhs, rhs = fml.split("~", 1)
    parts = rhs.split("|")
    others = [t.strip() for t in parts[0].split("+") if t.strip() != coef]
    fe_cols = [c.strip() for c in parts[1].split("+")] if len(parts) > 1 else []
    fes = "|" + parts[1] if len(parts) > 1 else ""
    aux = " + ".join(others) if others else "1"
    lhs = lhs.strip()
    # one common estimation sample: drop rows missing any variable the fit will use
    need = [lhs, coef, cluster] + others + fe_cols + ([weights] if weights else [])
    need = [c for c in dict.fromkeys(need) if c in df.columns]
    d = df.dropna(subset=need).reset_index(drop=True)
    # FWL residualization under H0 (coef excluded from the auxiliary regression);
    # keep singletons (fixef_rm="none") so residuals align 1:1 with d
    my = pf.feols(f"{lhs} ~ {aux} {fes}", data=d, weights=weights, fixef_rm="none")
    mx = pf.feols(f"{coef} ~ {aux} {fes}", data=d, weights=weights, fixef_rm="none")
    yt = np.asarray(my.resid()).ravel()
    xt = np.asarray(mx.resid()).ravel()
    assert len(yt) == len(d) == len(xt), (len(yt), len(xt), len(d))
    w = (d[weights].values if weights else np.ones(len(d))).astype(float)
    # Assign bootstrap columns by the sorted cluster label, not by the order in
    # which clusters happen to appear in the input.  Parquet/DuckDB scans do not
    # promise a stable row order; first-occurrence factorization therefore made
    # the finite-replication p-value change after an otherwise lossless public-
    # data rewrite.  The score statistic is order invariant, and the bootstrap
    # draw should be as well.
    g = pd.factorize(d[cluster].values, sort=True)[0]
    s0 = np.zeros(g.max() + 1)
    np.add.at(s0, g, w * xt * yt)
    T = abs(s0.sum())
    W = np.random.default_rng(seed).choice([-1.0, 1.0], size=(reps, len(s0)))
    return float((1 + np.sum(np.abs(W @ s0) >= T)) / (1 + reps))


# some scripts import the estimator under this name
wild_p = wild_score_p

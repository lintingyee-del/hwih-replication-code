# -*- coding: utf-8 -*-
"""6B step 47 — M1 event study through event-time +66 on the 6A-coded panel.

Estimand (per the reframing): no reversion of the de-militarization margin in
the released record under the post-2020 permanent-enforcement regime. Not
"persistence of the campaign": post-2020 mixes the original waves with the
normalized successor regime.

Spec: share outcomes on target-docket fact-section cases, event-time bins x H
(exposure_v2_z), prefecture + province x month FE, province CRV1, weights =
cell denominator, min-cell >= 20, donut Sep-Dec 2021 (source transition).
Placebo-differenced variant subtracts the placebo-docket coercive share within
prefecture-month. Plateau test: joint equality of post-2020 bins; drift test:
linear trend across post-2020 bins.
Output: output/ext2124/persist_es.csv, persist_tests.csv
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
import pandas as pd, numpy as np, pyfixest as pf, duckdb, os, sys, io
from scipy import stats as sps

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
DATA = str(_REP_PROJECT / "data")
OUT = str(_REP_PROJECT / "output" / "ext2124")
SRC = str(_REP_CASE_ARCHIVE)

con = duckdb.connect()
pan = con.sql(f"SELECT * FROM '{OUT}/persist_panel.parquet'").df()
exp = con.sql(f"SELECT prefecture_code, province, exposure_v2_z AS H FROM '{DATA}/exposure_v2.parquet'").df()
insp = con.sql(f"""
  SELECT province, min(strftime(inspection_start_date, '%Y-%m')) AS insp_ym
  FROM '{SRC}' WHERE inspection_start_date IS NOT NULL GROUP BY 1""").df()

df = pan.merge(exp, on="prefecture_code", how="inner").merge(insp, on="province", how="inner")
ym_i = pd.PeriodIndex(df["ym"], freq="M")
im_i = pd.PeriodIndex(df["insp_ym"], freq="M")
df["event_time"] = (ym_i - im_i).map(lambda x: x.n)
df["prov_id"] = pd.factorize(df["province"])[0]
df["prov_month"] = df["province"] + "_" + df["ym"]
df["pref"] = df["prefecture_code"]
df["d_share"] = df["sh_coercive"] - df["sh_coercive_placebo"]

DONUT = {"2021-09", "2021-10", "2021-11", "2021-12"}
BINS = [(-24,-19),(-18,-13),(-12,-7),(-6,-1),(0,5),(6,11),(12,17),(18,23),
        (24,29),(30,35),(36,41),(42,47),(48,53),(54,66)]
POST20 = [(24,29),(30,35),(36,41),(42,47),(48,53),(54,66)]

def run_es(dd, ycol, wcol, label, mincell=20, donut=True):
    d = dd[dd[wcol] >= mincell].copy()
    if donut:
        d = d[~d["ym"].isin(DONUT)]
    d = d.dropna(subset=[ycol, "H"])
    terms = []
    for lo, hi in BINS:
        if (lo, hi) == (-6, -1):
            continue
        nm = f"b_{lo}_{hi}".replace("-", "m")
        d[nm] = ((d["event_time"] >= lo) & (d["event_time"] <= hi)).astype(float) * d["H"]
        terms.append(nm)
    m = pf.feols(f"{ycol} ~ {' + '.join(terms)} | pref + prov_month",
                 data=d, vcov={"CRV1": "prov_id"}, weights=wcol)
    names = list(m.coef().index)
    rows = []
    for t in terms:
        lo, hi = t[2:].replace("m", "-").rsplit("_", 1)
        rows.append(dict(spec=label, bin_lo=int(lo), bin_hi=int(hi),
                         est=m.coef()[t], se=m.se()[t], p=m.pvalue()[t], n=int(m._N)))
    # plateau: joint equality of post-2020 bins; drift: linear trend across them
    pterms = [f"b_{lo}_{hi}".replace("-", "m") for lo, hi in POST20]
    idx = [names.index(t) for t in pterms]
    b = m.coef()[pterms].values
    V = m._vcov[np.ix_(idx, idx)]
    R = np.zeros((len(pterms) - 1, len(pterms)))
    for i in range(len(pterms) - 1):
        R[i, i], R[i, i + 1] = 1.0, -1.0
    W = float((R @ b) @ np.linalg.solve(R @ V @ R.T, R @ b))
    p_eq = float(1 - sps.chi2.cdf(W, R.shape[0]))
    mid = np.array([(lo + hi) / 2 for lo, hi in POST20])
    c = (mid - mid.mean()); c = c / (c @ c)
    drift = float(c @ b); drift_se = float(np.sqrt(c @ V @ c))
    # pre-trend joint test (leads earlier than -6)
    lterms = [t for t in terms if t.startswith("b_m") and abs(int(t[2:].replace("m","-").rsplit("_",1)[0])) >= 7]
    li = [names.index(t) for t in lterms]
    lb = m.coef()[lterms].values
    lV = m._vcov[np.ix_(li, li)]
    p_pre = float(1 - sps.chi2.cdf(float(lb @ np.linalg.solve(lV, lb)), len(lterms)))
    tests = dict(spec=label, p_plateau_eq=p_eq, drift_per_month=drift,
                 drift_se=drift_se, p_pretrend=p_pre, n=int(m._N))
    print(f"[{label}] N={m._N}  pre-trend p={p_pre:.3f}  post-bin equality p={p_eq:.3f}  "
          f"drift/mo={drift:.5f} (se {drift_se:.5f})")
    for r in rows:
        print(f"   [{r['bin_lo']:+d},{r['bin_hi']:+d}]  {r['est']: .5f} ({r['se']:.5f})")
    return rows, tests

all_rows, all_tests = [], []
for ycol, wcol, label in [
    ("sh_coercive", "n_target_fact", "coercive_share"),
    ("d_share", "n_target_fact", "coercive_share_placebo_diff"),
    ("sh_coercive_debt", "n_target_fact", "coercive_debt_share"),
    ("sh_relational", "n_target_fact", "relational_share"),
]:
    r, t = run_es(df, ycol, wcol, label)
    all_rows += r; all_tests.append(t)

# robustness: no donut, min-cell 10
r, t = run_es(df, "sh_coercive", "n_target_fact", "coercive_share_nodonut", donut=False)
all_rows += r; all_tests.append(t)
r, t = run_es(df, "sh_coercive", "n_target_fact", "coercive_share_min10", mincell=10)
all_rows += r; all_tests.append(t)

pd.DataFrame(all_rows).to_csv(f"{OUT}/persist_es.csv", index=False)
pd.DataFrame(all_tests).to_csv(f"{OUT}/persist_tests.csv", index=False)
print("written:", f"{OUT}/persist_es.csv", f"{OUT}/persist_tests.csv")

# -*- coding: utf-8 -*-
"""6B step 46 — M1 groundwork: prefecture-month criminal panel 2014-2024 from the
6A case-level parquet (which carries mechanism flags through 2024), plus the
composition gates the persistence extension must pass before any estimation.

Outputs to output/ext2124/:
  persist_panel.parquet   prefecture x month cells: counts, fact-section rate,
                          coercive/relational/ex-ante shares (fact-section cases,
                          target dockets), offense-family mix, mean fact length
  gates_composition.csv   prefecture-level 2019->2022/23 changes in predetermined
                          docket attributes regressed on exposure H_c (31-province
                          CRV1), one row per attribute x horizon
Notes: outcomes here use the 6A coding consistently across the 2020 boundary, so
the extension series is internally comparable; in-window replication against the
paper's v2-dictionary results is a separate check.
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
import duckdb, os, sys, io
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
SRC = str(_REP_CASE_ARCHIVE)
EXP = str(_REP_PROJECT / 'data' / 'exposure_v2.parquet').replace('\\', '/')
OUT = str(_REP_PROJECT / "output" / "ext2124")
os.makedirs(OUT, exist_ok=True)

con = duckdb.connect()
con.sql("SET threads TO 8; SET memory_limit='16GB'")

con.sql(f"""
CREATE OR REPLACE TEMP VIEW cases AS
SELECT prefecture_code, province,
  strftime(judgment_date, '%Y-%m') AS ym,
  year(judgment_date) AS yr,
  placebo_crime, illegal_market_crime,
  has_fact_section::INT AS has_fact,
  fact_text_length,
  GREATEST(COALESCE(coercive_physical_violence,0), COALESCE(coercive_threat,0),
           COALESCE(coercive_illegal_detention,0), COALESCE(coercive_soft_violence,0)) AS coercive_any,
  COALESCE(coercive_debt_collection,0) AS coercive_debt,
  GREATEST(COALESCE(relational_acquaintance,0), COALESCE(relational_kinship_hometown,0),
           COALESCE(relational_introducer,0)) AS relational_any,
  GREATEST(COALESCE(ex_ante_deposit_prepayment,0), COALESCE(ex_ante_guarantee_collateral,0),
           COALESCE(ex_ante_identity_contact_verification,0)) AS exante_any,
  (crime_name SIMILAR TO '.*(非法拘禁|寻衅滋事|聚众斗殴|敲诈勒索|强迫交易|黑社会性质组织).*')::INT AS violenf
FROM '{SRC}'
WHERE prefecture_code IS NOT NULL AND judgment_date >= DATE '2014-01-01'
  AND judgment_date <= DATE '2024-10-31'
""")

con.sql(f"""
COPY (
  SELECT prefecture_code, ym, yr,
    COUNT(*) AS n_all,
    SUM((placebo_crime=0)::INT) AS n_target,
    SUM((placebo_crime=0 AND has_fact=1)::INT) AS n_target_fact,
    SUM((placebo_crime=1 AND has_fact=1)::INT) AS n_placebo_fact,
    SUM((illegal_market_crime=1)::INT) AS n_market,
    AVG(has_fact) AS sh_fact,
    median(fact_text_length) AS med_factlen,
    -- shares within target-docket, fact-section cases (the M1 objects)
    AVG(CASE WHEN placebo_crime=0 AND has_fact=1 THEN coercive_any END) AS sh_coercive,
    AVG(CASE WHEN placebo_crime=0 AND has_fact=1 THEN coercive_debt END) AS sh_coercive_debt,
    AVG(CASE WHEN placebo_crime=0 AND has_fact=1 THEN relational_any END) AS sh_relational,
    AVG(CASE WHEN placebo_crime=0 AND has_fact=1 THEN exante_any END) AS sh_exante,
    -- placebo-docket counterpart (traffic/dangerous-driving etc.)
    AVG(CASE WHEN placebo_crime=1 AND has_fact=1 THEN coercive_any END) AS sh_coercive_placebo,
    -- offense-family mix (predetermined at filing)
    SUM(violenf) AS n_violenf,
    SUM((placebo_crime=1)::INT) AS n_placebo
  FROM cases GROUP BY 1,2,3
) TO '{OUT}/persist_panel.parquet' (FORMAT PARQUET)
""")
pan = con.sql(f"SELECT * FROM '{OUT}/persist_panel.parquet'").df()
print(f"panel: {len(pan):,} prefecture-month cells, "
      f"{pan.prefecture_code.nunique()} prefectures, {pan.ym.min()}..{pan.ym.max()}")

# ---------------------------------------------------------------- gates
exp = con.sql(f"SELECT prefecture_code, province, exposure_v2_z AS H FROM '{EXP}'").df()
grp_cols = {
    "log_release": lambda d: np.log(d.n_all.sum()),
    "sh_fact": lambda d: (d.sh_fact * d.n_all).sum() / d.n_all.sum(),
    "violent_offense_mix": lambda d: d.n_violenf.sum() / max(d.n_target.sum(), 1),
    "placebo_mix": lambda d: d.n_placebo.sum() / d.n_all.sum(),
    "med_factlen": lambda d: np.average(d.med_factlen.dropna()) if d.med_factlen.notna().any() else np.nan,
    "target_share": lambda d: d.n_target.sum() / d.n_all.sum(),
}

def yearly(pan, years):
    sub = pan[pan.yr.isin(years)]
    out = []
    for pc, d in sub.groupby("prefecture_code"):
        row = {"prefecture_code": pc}
        for k, f in grp_cols.items():
            try:
                row[k] = f(d)
            except Exception:
                row[k] = np.nan
        out.append(row)
    return pd.DataFrame(out)


def crv1(y, x, cluster):
    m = ~(np.isnan(y) | np.isnan(x))
    y, x, cl = y[m], x[m], cluster[m]
    X = np.column_stack([np.ones(len(x)), x])
    XtXi = np.linalg.inv(X.T @ X)
    b = XtXi @ X.T @ y
    u = y - X @ b
    G = len(np.unique(cl))
    meat = np.zeros((2, 2))
    for g in np.unique(cl):
        s = X[cl == g].T @ u[cl == g]
        meat += np.outer(s, s)
    V = (G/(G-1)) * ((len(y)-1)/(len(y)-2)) * XtXi @ meat @ XtXi
    return b[1], np.sqrt(V[1, 1]), len(y), G


base = yearly(pan, [2019]).set_index("prefecture_code")
rows = []
for label, years in [("2021", [2021]), ("2022", [2022]), ("2023", [2023]),
                     ("2022-23", [2022, 2023]), ("2021-24", [2021, 2022, 2023, 2024])]:
    post = yearly(pan, years).set_index("prefecture_code")
    for k in grp_cols:
        d = (post[k] - base[k]).rename("dy").to_frame().join(
            exp.set_index("prefecture_code")[["H", "province"]], how="inner").dropna()
        if len(d) < 50:
            continue
        beta, se, n, G = crv1(d.dy.to_numpy(float), d.H.to_numpy(float),
                              d.province.to_numpy())
        rows.append({"horizon": label, "attribute": k, "beta_H": beta, "se": se,
                     "t": beta/se, "n_pref": n, "n_clusters": G})
res = pd.DataFrame(rows)
res.to_csv(f"{OUT}/gates_composition.csv", index=False)
print(res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

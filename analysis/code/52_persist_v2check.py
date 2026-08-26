# -*- coding: utf-8 -*-
"""6B step 52 — pre-trend cross-check: does the 6A-coded coercive share behave
like the paper's own v2-dictionary backstop share (d_backstop from the 06_sweep
extracts) over the common 2014-2020 window? Same spec both sides: bins x H,
prefecture + province-month FE, weights = cell n, min cell 20, CRV1 province.
Also re-runs both with an alternative omitted period (-12..-1) to diagnose the
role of the -6..-1 anchor (possible pre-inspection enforcement run-up).
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
import duckdb, glob, os, sys, io
import numpy as np
import pandas as pd
import pyfixest as pf
from scipy import stats as sps

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
DATA = str(_REP_PROJECT / "data")
OUT = str(_REP_PROJECT / "output" / "ext2124")
SRC = str(_REP_CASE_ARCHIVE)
PLACEBO = "('危险驾驶','交通肇事','过失致人死亡','盗窃')"

con = duckdb.connect()
con.sql("SET threads TO 6; SET memory_limit='12GB'")

files = sorted(glob.glob(str(_REP_PROJECT / "data" / "derived" / "extract_2014_2020" / 'crim_20*.parquet').replace('\\', '/')))
print(f"v2 extract files: {len(files)}")
con.sql(f"""
CREATE OR REPLACE TEMP VIEW v2p AS
SELECT x.court, strftime(TRY_CAST(x.judgment_date AS DATE), '%Y-%m') AS ym,
       w.prefecture_code, w.province,
       (x.crime NOT IN {PLACEBO})::INT AS target, x.d_backstop
FROM read_parquet({files}) x
JOIN '{DATA}/court_xwalk.parquet' w ON x.court = w.court_name
WHERE x.judgment_date IS NOT NULL
""")
v2 = con.sql("""
SELECT prefecture_code, province, ym,
       COUNT(*) FILTER (target=1) AS n_t,
       AVG(d_backstop) FILTER (target=1) AS sh_backstop
FROM v2p GROUP BY 1,2,3
""").df()
print(f"v2 panel: {len(v2):,} cells, {v2.prefecture_code.nunique()} prefectures, "
      f"{v2.ym.min()}..{v2.ym.max()}")

exp = con.sql(f"SELECT prefecture_code, province, exposure_v2_z AS H FROM '{DATA}/exposure_v2.parquet'").df()
insp = con.sql(f"""
  SELECT province, min(strftime(inspection_start_date, '%Y-%m')) AS insp_ym
  FROM '{SRC}' WHERE inspection_start_date IS NOT NULL GROUP BY 1""").df()

pan6a = con.sql(f"SELECT * FROM '{OUT}/persist_panel.parquet'").df()
pan6a = pan6a[pan6a.ym <= "2020-12"]


def prep(df, ycol, ncol):
    d = df.drop(columns=["province"], errors="ignore")
    d = d.merge(exp, on="prefecture_code").merge(insp, on="province")
    d["event_time"] = (pd.PeriodIndex(d["ym"], freq="M")
                       - pd.PeriodIndex(d["insp_ym"], freq="M")).map(lambda x: x.n)
    d["prov_id"] = pd.factorize(d["province"])[0]
    d["prov_month"] = d["province"] + "_" + d["ym"]
    d["pref"] = d["prefecture_code"]
    d = d[d[ncol] >= 20].dropna(subset=[ycol, "H"])
    return d


BINS = [(-24,-19),(-18,-13),(-12,-7),(-6,-1),(0,5),(6,11),(12,17),(18,27)]

def es(d, ycol, wcol, label, omit=(-6,-1)):
    terms = []
    for lo, hi in BINS:
        if (lo, hi) == omit or (omit == (-12,-1) and lo in (-12,-6)):
            continue
        nm = f"b_{lo}_{hi}".replace("-", "m")
        d = d.copy()
        d[nm] = ((d["event_time"] >= lo) & (d["event_time"] <= hi)).astype(float) * d["H"]
        terms.append(nm)
    m = pf.feols(f"{ycol} ~ {' + '.join(terms)} | pref + prov_month",
                 data=d, vcov={"CRV1": "prov_id"}, weights=wcol)
    names = list(m.coef().index)
    leads = [t for t in terms if t.startswith("b_m")]
    li = [names.index(t) for t in leads]
    lb = m.coef()[leads].values
    lV = m._vcov[np.ix_(li, li)]
    p_pre = float(1 - sps.chi2.cdf(float(lb @ np.linalg.solve(lV, lb)), len(leads)))
    print(f"[{label}] N={m._N} pre-trend p={p_pre:.3f}")
    for t in terms:
        lo, hi = t[2:].replace("m", "-").rsplit("_", 1)
        print(f"   [{int(lo):+d},{int(hi):+d}]  {m.coef()[t]: .5f} ({m.se()[t]:.5f})")
    return p_pre


d_v2 = prep(v2.rename(columns={"n_t": "n"}), "sh_backstop", "n")
d_6a = prep(pan6a, "sh_coercive", "n_target_fact")

es(d_v2, "sh_backstop", "n", "v2_dictionary_2014-20, omit -6..-1")
es(d_6a, "sh_coercive", "n_target_fact", "6A_coding_2014-20, omit -6..-1")
es(d_v2, "sh_backstop", "n", "v2_dictionary, omit -12..-1", omit=(-12,-1))
es(d_6a, "sh_coercive", "n_target_fact", "6A_coding, omit -12..-1", omit=(-12,-1))

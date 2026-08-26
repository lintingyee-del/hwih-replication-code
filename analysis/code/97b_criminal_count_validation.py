# -*- coding: utf-8 -*-
"""Pre-declared validation of the only candidate from step 97.

The candidate is the enforcement-crime count with an exposure-specific linear
trend.  To break the mechanical link between the 2014--2017 count component of
H and the outcome, this script rebuilds H from 2014--2015 only and estimates on
2016--2020 (or 2016--2019).  It reports staggered and national clocks under
asinh OLS and PPML.  The symmetric 2016--2017 exposure is retained as a
diagnostic, not as an independent holdout.
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
import os
import sys

import duckdb
import numpy as np
import pandas as pd
import pyfixest as pf
from statsmodels.stats.multitest import multipletests

from _wild import wild_score_p

sys.stdout.reconfigure(encoding="utf-8")
BASE = str(_REP_PROJECT)
DATA = os.path.join(BASE, "data")
OUT = os.path.join(BASE, "output", "spec_universe")
EXT = str(_REP_PROJECT / "data" / "derived" / "extract_2014_2020")
os.makedirs(OUT, exist_ok=True)

con = duckdb.connect()
con.sql("SET threads TO 10; SET memory_limit='20GB'")
con.sql(f"CREATE TABLE xwalk AS SELECT * FROM '{DATA}/court_xwalk.parquet'")
con.sql(f"""
CREATE TABLE crim0 AS
SELECT c.crime, c.d_backstop_collection, x.prefecture_code,
       TRY_CAST(c.judgment_date AS DATE) AS jdate
FROM read_parquet('{EXT}/crim_*.parquet') c
LEFT JOIN xwalk x ON c.court = x.court_name
WHERE x.prefecture_code IS NOT NULL
""")
viol = ("'非法拘禁','寻衅滋事','聚众斗殴','敲诈勒索','强迫交易',"
        "'组织、领导、参加黑社会性质组织'")
halves = {}
for tag, lo, hi in [("early", "2014-01-01", "2015-12-31"),
                    ("late", "2016-01-01", "2017-12-31")]:
    halves[tag] = con.sql(f"""
      SELECT prefecture_code, COUNT(*) AS n_pre,
        AVG((crime IN ({viol}))::INT) AS violent_share,
        AVG(d_backstop_collection) AS backstop_collect_rate
      FROM crim0
      WHERE jdate BETWEEN DATE '{lo}' AND DATE '{hi}'
      GROUP BY 1 HAVING COUNT(*) >= 150
    """).df()
con.close()

common = set(halves["early"]["prefecture_code"]) & set(halves["late"]["prefecture_code"])
for tag in halves:
    h = halves[tag][halves[tag]["prefecture_code"].isin(common)].copy()
    z1 = (h["violent_share"] - h["violent_share"].mean()) / h["violent_share"].std(ddof=1)
    z2 = ((h["backstop_collect_rate"] - h["backstop_collect_rate"].mean())
          / h["backstop_collect_rate"].std(ddof=1))
    h[f"H_{tag}"] = (z1 + z2) / 2
    halves[tag] = h[["prefecture_code", f"H_{tag}"]]
h = (halves["early"].merge(halves["late"], on="prefecture_code")
     .sort_values("prefecture_code").reset_index(drop=True))
h.to_csv(os.path.join(OUT, "criminal_split_half_exposure.csv"), index=False,
         encoding="utf-8-sig")

k = pd.read_parquet(os.path.join(DATA, "crim_panel_v2.parquet"))
k = k[(k["family"] == "enforcementcrime") & (k["n_cases"] > 0)].copy()
k = k.merge(h, on="prefecture_code", how="inner")
k["jmonth"] = pd.to_datetime(k["jmonth"])
k["insp_month"] = pd.to_datetime(k["insp_month"])
k["month"] = k["jmonth"].dt.strftime("%Y-%m")
k["year"] = k["jmonth"].dt.year
k["t"] = ((k["jmonth"].dt.year - 2016) * 12 + k["jmonth"].dt.month - 1).astype(float)
k["post_staggered"] = (k["jmonth"] >= k["insp_month"]).astype(int)
k["post_national"] = (k["jmonth"] >= pd.Timestamp("2018-01-01")).astype(int)
k["pref"] = k["prefecture_code"].astype(str)
k["prov_month"] = k["province"] + "_" + k["month"]
k["prov_id"] = pd.factorize(k["province"])[0]

rows = []
for htag in ("early", "late"):
    for timing in ("staggered", "national"):
        for end in (2019, 2020):
            for method in ("asinh", "ppml"):
                d = k[(k["year"] >= 2016) & (k["year"] <= end)].copy()
                d["H"] = d[f"H_{htag}"]
                d["Ht"] = d["H"] * (d["t"] - d["t"].mean())
                d["X"] = d[f"post_{timing}"] * d["H"]
                fml = ("y ~ X + Ht | pref + prov_month" if method == "asinh"
                       else "n_cases ~ X + Ht | pref + prov_month")
                if method == "asinh":
                    d["y"] = np.arcsinh(d["n_cases"])
                    m = pf.feols(fml, data=d, vcov={"CRV1": "prov_id"})
                    wp = wild_score_p(fml, d, "X", cluster="prov_id",
                                      reps=9_999, seed=42)
                else:
                    m = pf.fepois(fml, data=d, vcov={"CRV1": "prov_id"})
                    wp = np.nan
                rec = {
                    "exposure_half": htag,
                    "timing": timing,
                    "start": 2016,
                    "end": end,
                    "method": method,
                    "beta": float(m.coef()["X"]),
                    "se": float(m.se()["X"]),
                    "p_crv1": float(m.pvalue()["X"]),
                    "p_wild": wp,
                    "n_input": len(d),
                    "n_fit": int(m._N),
                }
                rows.append(rec)
                ptxt = f"{wp:.4f}" if np.isfinite(wp) else "--"
                print(f"H={htag:5s} {timing:9s} end={end} {method:6s} "
                      f"b={rec['beta']:+.5f} se={rec['se']:.5f} "
                      f"p={rec['p_crv1']:.4f} wild={ptxt} "
                      f"N={rec['n_fit']}/{rec['n_input']}", flush=True)

r = pd.DataFrame(rows)
r["p_for_bh"] = np.where(r["p_wild"].notna(), r["p_wild"], r["p_crv1"])
r["q_all"] = multipletests(r["p_for_bh"], method="fdr_bh")[1]
r["q_early"] = np.nan
idx = r.index[r["exposure_half"] == "early"]
r.loc[idx, "q_early"] = multipletests(r.loc[idx, "p_for_bh"], method="fdr_bh")[1]
r.to_csv(os.path.join(OUT, "criminal_count_split_half_validation.csv"),
         index=False, encoding="utf-8-sig")
print(f"split-half correlation = {h['H_early'].corr(h['H_late']):.3f}; N={len(h)}",
      flush=True)
print("early-half rows:", flush=True)
print(r[r["exposure_half"] == "early"].to_string(index=False), flush=True)

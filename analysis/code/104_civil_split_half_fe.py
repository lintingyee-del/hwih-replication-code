# -*- coding: utf-8 -*-
"""Requested-FE audit using the paper's pre-existing split-half exposures.

The three regressions use the identical common sample and specification.  They
change only H: the full 2014--17 index, the 2014--15 half, or the 2016--17 half.

Output: output/ext2124/civil_split_half_fe.csv
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

import duckdb
import numpy as np
import pandas as pd
import pyfixest as pf

from _wild import wild_score_p


DATA = str(_REP_PROJECT / "data")
RAW = str(_REP_PROJECT / "data" / "derived" / "extract_2014_2020")
OUT = str(_REP_PROJECT / "output" / "ext2124")
WINDOW = ("2017-01", "2019-03")
POST0 = "2018-09"
ROWS = []


# Rebuild the two exposure halves exactly as in 22_referee_robustness.py.
con = duckdb.connect()
con.sql("SET threads TO 10; SET memory_limit='20GB'")
con.sql(f"CREATE TABLE xwalk AS SELECT * FROM '{DATA}/court_xwalk.parquet'")
con.sql(f"""
CREATE TABLE crim0 AS
SELECT c.crime, c.d_backstop_collection, x.prefecture_code,
       TRY_CAST(c.judgment_date AS DATE) AS jdate
FROM read_parquet('{RAW}/crim_*.parquet') c
LEFT JOIN xwalk x ON c.court = x.court_name
WHERE x.prefecture_code IS NOT NULL
  AND TRY_CAST(c.judgment_date AS DATE) BETWEEN DATE '2014-01-01' AND DATE '2017-12-31'
""")
violent = (
    "'非法拘禁','寻衅滋事','聚众斗殴','敲诈勒索','强迫交易',"
    "'组织、领导、参加黑社会性质组织'"
)
halves = {}
for tag, lo, hi in (("h1", "2014-01-01", "2015-12-31"),
                    ("h2", "2016-01-01", "2017-12-31")):
    halves[tag] = con.sql(f"""
        SELECT prefecture_code, COUNT(*) AS n_pre,
          AVG((crime IN ({violent}))::INT) AS violent_share,
          AVG(d_backstop_collection) AS backstop_collect_rate
        FROM crim0
        WHERE jdate BETWEEN DATE '{lo}' AND DATE '{hi}'
        GROUP BY 1 HAVING COUNT(*) >= 150
    """).df()
con.close()

common_prefectures = halves["h1"][["prefecture_code"]].merge(
    halves["h2"][["prefecture_code"]], on="prefecture_code"
)
for tag in ("h1", "h2"):
    d = halves[tag][halves[tag]["prefecture_code"].isin(common_prefectures["prefecture_code"])].copy()
    z_violent = (d["violent_share"] - d["violent_share"].mean()) / d["violent_share"].std()
    z_backstop = (
        (d["backstop_collect_rate"] - d["backstop_collect_rate"].mean())
        / d["backstop_collect_rate"].std()
    )
    d[f"H_{tag}"] = (z_violent + z_backstop) / 2
    halves[tag] = d[["prefecture_code", f"H_{tag}"]]
split = halves["h1"].merge(halves["h2"], on="prefecture_code")


cells = pd.read_parquet(f"{DATA}/civil_panel.parquet")
cells = cells[cells["cause_family"] == "relational"].copy()
cells["month"] = cells["jmonth"].astype(str).str[:7]
cells = cells[(cells["month"] >= WINDOW[0]) & (cells["month"] <= WINDOW[1])]
schedule = (
    pd.read_parquet(f"{DATA}/panel_month.parquet")[["province", "inspection_round"]]
    .drop_duplicates()
)
cells = (
    cells.merge(schedule, on="province")
    .merge(split, on="prefecture_code")
    .dropna(subset=["inspection_round", "exposure_v2_z", "H_h1", "H_h2", "n_cases"])
)
cells["treat"] = (cells["inspection_round"] == 1).astype(int)
cells["postc"] = (cells["month"] >= POST0).astype(int)
cells["prov_id"] = pd.factorize(cells["province"])[0]
cells["pref_cause"] = cells["prefecture_code"] + "_" + cells["cause"]
cells["prov_month"] = cells["province"] + "_" + cells["month"]
cells["cause_month"] = cells["cause"] + "_" + cells["month"]
cells["y"] = np.arcsinh(cells["n_cases"])

formula = "y ~ pth + ph | pref_cause + prov_month + cause_month"
for spec_id, h_col, label in (
    ("X0", "exposure_v2_z", "full_2014_17_index_on_common_sample"),
    ("X1", "H_h1", "split_half_2014_15"),
    ("X2", "H_h2", "split_half_2016_17"),
):
    d = cells.copy()
    d["pth"] = d["postc"] * d["treat"] * d[h_col]
    d["ph"] = d["postc"] * d[h_col]
    crv1 = pf.feols(formula, data=d, vcov={"CRV1": "prov_id"})
    crv3 = pf.feols(formula, data=d, vcov={"CRV3": "prov_id"})
    p_wild = wild_score_p(formula, d, "pth")
    beta = float(crv1.coef()["pth"])
    row = {
        "spec_id": spec_id,
        "mode": "B_fixed_x_y_equation",
        "focus_side": "x_preexisting_measurement_variant",
        "base_variable": "pre_campaign_coercive_capacity",
        "transformation": label,
        "model": "OLS_high_dimensional_FE",
        "sample_rule": "identical_split_half_common_sample; judgment_month; 2017-01_to_2019-03",
        "controls": "post_x_exposure",
        "fixed_effects": "prefecture_x_cause + province_x_month + cause_x_month",
        "coefficient": beta,
        "std_error": float(crv1.se()["pth"]),
        "p_value": float(crv1.pvalue()["pth"]),
        "std_error_crv3": float(crv3.se()["pth"]),
        "p_crv3": float(crv3.pvalue()["pth"]),
        "p_wild": float(p_wild),
        "n_obs": int(crv1._N),
        "province_clusters": int(d["prov_id"].nunique()),
        "direction": "positive" if beta > 0 else "negative" if beta < 0 else "zero",
        "keep_or_drop": "retain_in_complete_audit_log",
        "reason": "pre-existing split-half exposure check; never a replacement selected by p-value",
    }
    ROWS.append(row)
    print(
        f"{spec_id} {label:38s} b={beta:+.6f} se1={row['std_error']:.6f} "
        f"p1={row['p_value']:.4f} se3={row['std_error_crv3']:.6f} "
        f"p3={row['p_crv3']:.4f} wild={p_wild:.4f} N={row['n_obs']:,}",
        flush=True,
    )

os.makedirs(OUT, exist_ok=True)
result = pd.DataFrame(ROWS)
path = f"{OUT}/civil_split_half_fe.csv"
result.to_csv(path, index=False)
print(f"written: {path}", flush=True)

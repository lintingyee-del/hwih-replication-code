# -*- coding: utf-8 -*-
"""6B step 08 — build v2 layers from <restricted-source-path> sweep outputs.

Outputs (data/):
  court_xwalk.parquet     court name -> prefecture_code/province (from 6A archive)
  civil_case.parquet      cleaned civil cases with parsed rate/amount/orig_year
  civil_panel.parquet     prefecture x month x cause cells
  crim_panel_v2.parquet   prefecture x month x family cells, new dictionaries
  exposure_v2.parquet     2014-17 violent-enforcement exposure (non-same-source)
  civil_flow.csv          civil sample flow
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
import duckdb, os, csv

SIX_A = str(_REP_CASE_ARCHIVE)
EXT = str(_REP_PROJECT / "data" / "derived" / "extract_2014_2020")
OUT = str(_REP_PROJECT / "data")
con = duckdb.connect()
con.sql("SET threads TO 10; SET memory_limit='20GB'")

# ---------------- court -> prefecture crosswalk from the 6A archive ----------
con.sql(f"""
CREATE OR REPLACE TABLE xwalk AS
SELECT court_name, arg_max(prefecture_code, cnt) AS prefecture_code,
       arg_max(province, cnt) AS province
FROM (
  SELECT court_name, prefecture_code, province, COUNT(*) AS cnt
  FROM '{SIX_A}'
  WHERE court_name IS NOT NULL AND prefecture_code<>'' AND province<>''
  GROUP BY 1,2,3
) GROUP BY 1
""")
con.sql(f"COPY xwalk TO '{OUT}/court_xwalk.parquet' (FORMAT PARQUET)")

# ---------------- inspection schedule (province -> round, start month) -------
con.sql(f"""
CREATE OR REPLACE TABLE sched AS
SELECT province, ANY_VALUE(inspection_round) AS round,
       ANY_VALUE(date_trunc('month', inspection_start_date)) AS insp_month
FROM '{SIX_A}' WHERE province<>'' AND inspection_start_date IS NOT NULL
GROUP BY 1
""")

# ---------------- civil case layer -------------------------------------------
flow = []
con.sql(f"""
CREATE OR REPLACE TABLE civ0 AS
SELECT * FROM read_parquet('{EXT}/civil_*.parquet', filename=true)
""")
flow.append(("V0 raw civil extract rows", con.sql("SELECT COUNT(*) FROM civ0").fetchone()[0]))
con.sql("""
CREATE OR REPLACE TABLE civ1 AS
SELECT c.*, x.prefecture_code, x.province, s.round, s.insp_month,
  TRY_CAST(judgment_date AS DATE) AS jdate
FROM civ0 c
LEFT JOIN xwalk x ON c.court = x.court_name
LEFT JOIN sched s USING (province)
""")
flow.append(("V1 court matched to prefecture",
             con.sql("SELECT COUNT(*) FROM civ1 WHERE prefecture_code IS NOT NULL").fetchone()[0]))
con.sql("""
CREATE OR REPLACE TABLE civil_case AS
SELECT case_no, court, cause, doc_type, doc_len, proceeding,
  prefecture_code, province, round, insp_month,
  date_trunc('month', jdate) AS jmonth,
  (date_trunc('month', jdate) >= insp_month)::INT AS post,
  datediff('month', insp_month, date_trunc('month', jdate)) AS event_time,
  rel_txn, credit_fail, rel_fail, evid_any, evid_iou, evid_guarantee,
  evid_transfer, evid_chat, backstop_collection, backstop_any, judic_any,
  (doc_type='mediation')::INT AS mediated,
  CASE WHEN cause IN ('民间借贷纠纷','保证合同纠纷','追偿权纠纷','合伙协议纠纷')
       THEN 'relational'
       WHEN cause IN ('买卖合同纠纷','租赁合同纠纷') THEN 'commercial'
       ELSE 'placebo' END AS cause_family,
  COALESCE(TRY_CAST(rate_月pct AS DOUBLE),
           TRY_CAST(rate_月分 AS DOUBLE),
           TRY_CAST(rate_年pct AS DOUBLE)/12.0) AS monthly_rate_pct,
  TRY_CAST(replace(replace(amt_str, ',', ''), '，','') AS DOUBLE)
    * CASE WHEN amt_is_wan=1 THEN 10000 ELSE 1 END AS amount_yuan,
  TRY_CAST(orig_year AS INT) AS orig_year
FROM civ1
WHERE prefecture_code IS NOT NULL AND jdate IS NOT NULL
  AND jdate BETWEEN DATE '2014-01-01' AND DATE '2020-12-31'
""")
flow.append(("V2 dated, geocoded, 2014-2020", con.sql("SELECT COUNT(*) FROM civil_case").fetchone()[0]))
flow.append(("V2a lending cases", con.sql("SELECT COUNT(*) FROM civil_case WHERE cause='民间借贷纠纷'").fetchone()[0]))
flow.append(("V2b placebo (traffic) cases", con.sql("SELECT COUNT(*) FROM civil_case WHERE cause_family='placebo'").fetchone()[0]))
con.sql(f"COPY civil_case TO '{OUT}/civil_case.parquet' (FORMAT PARQUET)")

# ---------------- exposure v2: violent private enforcement, 2014-17 ----------
# built ONLY from violent-enforcement crimes and detention x debt co-occurrence;
# disjoint from relational/evidence outcome dictionaries
con.sql(f"""
CREATE OR REPLACE TABLE crim0 AS
SELECT c.*, x.prefecture_code, x.province, TRY_CAST(judgment_date AS DATE) AS jdate
FROM read_parquet('{EXT}/crim_*.parquet') c
LEFT JOIN xwalk x ON c.court = x.court_name
""")
con.sql("""
CREATE OR REPLACE TABLE exposure_v2 AS
WITH pre AS (
  SELECT prefecture_code, province,
    COUNT(*) AS n_pre,
    AVG((crime IN ('非法拘禁','寻衅滋事','聚众斗殴','敲诈勒索','强迫交易',
                   '组织、领导、参加黑社会性质组织'))::INT) AS violent_share,
    AVG(detention_debt) AS detention_debt_rate,
    AVG(d_backstop_collection) AS backstop_collect_rate
  FROM crim0
  WHERE prefecture_code IS NOT NULL
    AND jdate BETWEEN DATE '2014-01-01' AND DATE '2017-12-31'
  GROUP BY 1,2
)
SELECT *,
  ((violent_share - AVG(violent_share) OVER())/STDDEV(violent_share) OVER()
   + (backstop_collect_rate - AVG(backstop_collect_rate) OVER())
     /STDDEV(backstop_collect_rate) OVER())/2 AS exposure_v2_z
FROM pre WHERE n_pre >= 300
""")
con.sql(f"COPY exposure_v2 TO '{OUT}/exposure_v2.parquet' (FORMAT PARQUET)")

# ---------------- civil panel -------------------------------------------------
con.sql(f"""
CREATE OR REPLACE TABLE civil_panel AS
SELECT c.prefecture_code, c.province, c.cause, c.cause_family, c.jmonth,
  ANY_VALUE(c.post) AS post, ANY_VALUE(c.event_time) AS event_time,
  ANY_VALUE(c.insp_month) AS insp_month,
  COUNT(*) AS n_cases,
  AVG(rel_txn::DOUBLE) AS y_rel_txn,
  AVG(rel_fail::DOUBLE) AS y_rel_fail,
  AVG(evid_iou::DOUBLE) AS y_evid_iou,
  AVG(evid_transfer::DOUBLE) AS y_evid_transfer,
  AVG(evid_any::DOUBLE) AS y_evid_any,
  AVG(backstop_collection::DOUBLE) AS y_backstop_collect,
  AVG(mediated::DOUBLE) AS y_mediated,
  AVG(doc_len) AS x_doclen,
  MEDIAN(monthly_rate_pct) AS y_rate_med,
  e.exposure_v2_z
FROM civil_case c JOIN exposure_v2 e USING (prefecture_code)
GROUP BY c.prefecture_code, c.province, c.cause, c.cause_family, c.jmonth, e.exposure_v2_z
""")
flow.append(("V3 civil panel cells", con.sql("SELECT COUNT(*) FROM civil_panel").fetchone()[0]))
con.sql(f"COPY civil_panel TO '{OUT}/civil_panel.parquet' (FORMAT PARQUET)")

# ---------------- criminal panel v2 (new dictionaries) ------------------------
con.sql("""
CREATE OR REPLACE TABLE crim_panel AS
SELECT prefecture_code, province,
  CASE WHEN crime IN ('赌博','开设赌场','组织卖淫','非法经营','走私普通货物、物品') THEN 'market'
       WHEN crime IN ('危险驾驶','交通肇事','过失致人死亡') THEN 'placebo'
       WHEN crime='盗窃' THEN 'theft'
       WHEN crime='故意伤害' THEN 'violence'
       WHEN crime IN ('非法拘禁','寻衅滋事','聚众斗殴','强迫交易','敲诈勒索',
                      '组织、领导、参加黑社会性质组织') THEN 'enforcementcrime'
       WHEN crime='诈骗' THEN 'fraud' ELSE 'other' END AS family,
  date_trunc('month', jdate) AS jmonth,
  COUNT(*) AS n_cases,
  AVG(d_backstop) AS y_backstop,
  AVG(d_backstop_collection) AS y_backstop_collect,
  AVG(detention_debt) AS y_detention_debt,
  AVG(d_rel_txn) AS y_rel_txn,
  AVG(d_rel_fail) AS y_rel_fail,
  AVG(d_formalization) AS y_formalization,
  AVG(fraud_telecom) AS y_fraud_telecom,
  AVG(fraud_acquaintance) AS y_fraud_acq,
  AVG(doc_len) AS x_doclen
FROM crim0
WHERE prefecture_code IS NOT NULL
  AND jdate BETWEEN DATE '2014-01-01' AND DATE '2020-12-31'
GROUP BY 1,2,3,4
""")
con.sql(f"""
CREATE OR REPLACE TABLE crim_panel_v2 AS
SELECT p.*, e.exposure_v2_z, s.insp_month,
  (p.jmonth >= s.insp_month)::INT AS post,
  datediff('month', s.insp_month, p.jmonth) AS event_time
FROM crim_panel p
JOIN exposure_v2 e USING (prefecture_code)
JOIN sched s ON p.province = s.province
""")
flow.append(("V4 criminal v2 panel cells", con.sql("SELECT COUNT(*) FROM crim_panel_v2").fetchone()[0]))
con.sql(f"COPY crim_panel_v2 TO '{OUT}/crim_panel_v2.parquet' (FORMAT PARQUET)")

with open(f"{OUT}/civil_flow.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh); w.writerow(["step", "n"]); w.writerows(flow)
for s, n in flow: print(f"  {s}: {n:,}")
print("build v2 done")

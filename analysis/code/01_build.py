# -*- coding: utf-8 -*-
"""HWIH replication pipeline: step 01 — case-level cleaning, exposure, panels, audits.

Input : 6a_case_level_analysis_text_v02.parquet (4,877,456 criminal judgments)
Output: data/case_clean.parquet, data/panel_month.parquet, data/exposure.parquet,
        data/audit_publication.csv, data/sample_flow.csv, data/offense_date_qc.csv
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
import duckdb, os

SRC = str(_REP_CASE_ARCHIVE)
OUT = str(_REP_PROJECT / "data")
os.makedirs(OUT, exist_ok=True)
con = duckdb.connect()

# ---------------------------------------------------------------- crime groups
# analysis_group:
#   market   — relational-intensive illegal markets (outcomes measured here)
#   placebo  — low relational dependence (traffic family)
#   theft    — secondary placebo
#   violence — 故意伤害: violence-margin outcome, not a placebo
#   direct   — campaign-direct targets (涉黑, 敲诈勒索): exposure only
#   fraud    — 诈骗: excluded (telecom vs ordinary unsplittable without text)
GROUP_SQL = """
CASE
  WHEN crime_name IN ('赌博罪','开设赌场罪','组织卖淫罪','非法经营罪','走私普通货物、物品罪')
       THEN 'market'
  WHEN crime_name IN ('危险驾驶罪','交通肇事罪','过失致人死亡罪') THEN 'placebo'
  WHEN crime_name = '盗窃罪' THEN 'theft'
  WHEN crime_name = '故意伤害罪' THEN 'violence'
  WHEN crime_name IN ('组织、领导、参加黑社会性质组织罪','敲诈勒索罪') THEN 'direct'
  WHEN crime_name = '诈骗罪' THEN 'fraud'
  ELSE 'other' END
"""

# ------------------------------------------------------- sample flow accounting
steps = []
n0 = con.sql(f"SELECT COUNT(*) FROM '{SRC}'").fetchone()[0]
steps.append(("S0 raw archive", n0))
n1 = con.sql(f"SELECT COUNT(*) FROM '{SRC}' WHERE province<>'' AND prefecture_code<>''").fetchone()[0]
steps.append(("S1 valid province+prefecture geocode", n1))
n2 = con.sql(f"""SELECT COUNT(*) FROM '{SRC}' WHERE province<>'' AND prefecture_code<>''
    AND judgment_month IS NOT NULL""").fetchone()[0]
steps.append(("S2 + valid judgment month", n2))
n3 = con.sql(f"""SELECT COUNT(*) FROM '{SRC}' WHERE province<>'' AND prefecture_code<>''
    AND judgment_month BETWEEN DATE '2014-01-01' AND DATE '2020-12-31'""").fetchone()[0]
steps.append(("S3 + main window 2014-2020 (publication-collapse guard)", n3))

# ------------------------------------------------------------------ case_clean
con.sql(f"""
CREATE OR REPLACE TABLE case_clean AS
SELECT *,
  {GROUP_SQL} AS analysis_group,
  -- mechanism composites (denominator: has_fact_section=1)
  GREATEST(coercive_physical_violence, coercive_threat, coercive_illegal_detention,
           coercive_soft_violence, coercive_debt_collection, coercive_territorial_control,
           coercive_protection_fee, coercive_official_protection) AS hard_backstop_any,
  GREATEST(relational_acquaintance, relational_kinship_hometown, relational_introducer,
           relational_long_term_or_repeated, relational_reputation, relational_closed_group)
           AS relational_any,
  GREATEST(ex_ante_deposit_prepayment, ex_ante_guarantee_collateral,
           ex_ante_escrow_or_platform) AS formalization_any,
  CASE WHEN GREATEST(relational_acquaintance, relational_kinship_hometown,
       relational_introducer, relational_long_term_or_repeated, relational_reputation,
       relational_closed_group)=1 AND dispute_resolution=1 THEN 1 ELSE 0 END
       AS relational_failure_visible,
  date_trunc('month', inspection_start_date) AS insp_month,
  (judgment_month >= date_trunc('month', inspection_start_date))::INT AS post_judgment,
  datediff('month', date_trunc('month', inspection_start_date), judgment_month)
    AS event_time
FROM '{SRC}'
WHERE province<>'' AND prefecture_code<>'' AND judgment_month IS NOT NULL
  AND judgment_month BETWEEN DATE '2014-01-01' AND DATE '2020-12-31'
""")
n4 = con.sql("SELECT COUNT(*) FROM case_clean WHERE analysis_group IN ('market','placebo','theft','violence')").fetchone()[0]
steps.append(("S4 analysis crimes (market+placebo+theft+violence; direct->exposure, fraud excluded)", n4))
n5 = con.sql("""SELECT COUNT(*) FROM case_clean WHERE analysis_group IN ('market','placebo','theft','violence')
    AND has_fact_section=1""").fetchone()[0]
steps.append(("S5 mechanism-outcome subsample (has_fact_section=1)", n5))

# ---------------------------------------------------------------- exposure (H)
# Pre-period 2014-2017, prefecture level; NOT built from outcome families:
#   (a) direct-target crime share: (涉黑+敲诈勒索) / all judgments
#   (b) coercive text-flag rate among fact-section cases (coercive dictionary is
#       disjoint from relational / ex-ante outcome dictionaries)
con.sql(f"""
CREATE OR REPLACE TABLE exposure AS
WITH pre AS (
  SELECT prefecture_code, province,
    COUNT(*) AS n_pre,
    AVG(({GROUP_SQL} = 'direct')::INT) AS direct_share,
    AVG(CASE WHEN has_fact_section=1 THEN
      GREATEST(coercive_physical_violence, coercive_threat, coercive_illegal_detention,
               coercive_soft_violence, coercive_debt_collection, coercive_territorial_control,
               coercive_protection_fee, coercive_official_protection) END) AS coercive_rate,
    COUNT(*) FILTER (WHERE has_fact_section=1) AS n_fact_pre
  FROM '{SRC}'
  WHERE province<>'' AND prefecture_code<>''
    AND judgment_month BETWEEN DATE '2014-01-01' AND DATE '2017-12-31'
  GROUP BY 1,2
)
SELECT *,
  (direct_share  - AVG(direct_share)  OVER()) / STDDEV(direct_share)  OVER() AS direct_share_z,
  (coercive_rate - AVG(coercive_rate) OVER()) / STDDEV(coercive_rate) OVER() AS coercive_rate_z,
  ((direct_share - AVG(direct_share) OVER()) / STDDEV(direct_share) OVER()
   + (coercive_rate - AVG(coercive_rate) OVER()) / STDDEV(coercive_rate) OVER())/2
   AS exposure_z
FROM pre WHERE n_pre >= 200 AND n_fact_pre >= 50
""")

# ---------------------------------------------------------------------- panel
con.sql("""
CREATE OR REPLACE TABLE panel AS
SELECT c.prefecture_code, c.province, c.analysis_group, c.judgment_month,
  c.insp_month, c.post_judgment, c.event_time,
  ANY_VALUE(c.inspection_round) AS inspection_round,
  COUNT(*) AS n_cases,
  COUNT(*) FILTER (WHERE has_fact_section=1) AS n_fact,
  AVG(CASE WHEN has_fact_section=1 THEN hard_backstop_any END)         AS y_backstop,
  AVG(CASE WHEN has_fact_section=1 THEN relational_any END)            AS y_relational,
  AVG(CASE WHEN has_fact_section=1 THEN relational_failure_visible END) AS y_rel_failure,
  AVG(CASE WHEN has_fact_section=1 THEN formalization_any END)         AS y_formalization,
  AVG(CASE WHEN has_fact_section=1 THEN fact_text_length END)          AS x_factlen,
  AVG(has_fact_section::DOUBLE)                                        AS x_factshare,
  AVG(spans_inspection::DOUBLE)                                        AS x_spanshare
FROM case_clean c
WHERE analysis_group IN ('market','placebo','theft','violence')
GROUP BY 1,2,3,4,5,6,7
""")
con.sql(f"""
CREATE OR REPLACE TABLE panel_x AS
SELECT p.*, e.exposure_z, e.direct_share_z, e.coercive_rate_z
FROM panel p JOIN exposure e USING (prefecture_code)
""")
n6 = con.sql("SELECT SUM(n_cases) FROM panel_x").fetchone()[0]
steps.append(("S6 cases entering estimation panel (exposure-matched prefectures)", int(n6)))
ncell = con.sql("SELECT COUNT(*) FROM panel_x").fetchone()[0]
steps.append(("S6b panel cells (prefecture x month x group)", ncell))

# --------------------------------------------------------------------- audits
con.sql(f"""COPY (
  SELECT province, year(judgment_date) AS yr, {GROUP_SQL} AS grp, COUNT(*) AS n
  FROM '{SRC}' WHERE province<>'' GROUP BY 1,2,3 ORDER BY 1,2,3
) TO '{OUT}/audit_publication.csv' (HEADER)""")
con.sql(f"""COPY (
  SELECT offense_date_quality, spans_inspection, COUNT(*) AS n,
         AVG(post_by_offense_end) AS post_share
  FROM case_clean GROUP BY 1,2 ORDER BY 3 DESC
) TO '{OUT}/offense_date_qc.csv' (HEADER)""")

import csv
with open(f"{OUT}/sample_flow.csv","w",newline="",encoding="utf-8") as fh:
    w = csv.writer(fh); w.writerow(["step","n"]); w.writerows(steps)

con.sql(f"COPY case_clean TO '{OUT}/case_clean.parquet' (FORMAT PARQUET)")
con.sql(f"COPY panel_x   TO '{OUT}/panel_month.parquet' (FORMAT PARQUET)")
con.sql(f"COPY exposure  TO '{OUT}/exposure.parquet' (FORMAT PARQUET)")
print("SAMPLE FLOW:")
for s,n in steps: print(f"  {s}: {n:,}")
print("prefectures with exposure:", con.sql("SELECT COUNT(*) FROM exposure").fetchone()[0])

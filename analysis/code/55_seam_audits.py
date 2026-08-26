# -*- coding: utf-8 -*-
"""6B step 55 — two seam audits for the release-selection threat.

A. SURVIVAL AUDIT via the 2024-10 backfill snapshot: the s41 October-2024 file
   is 98% re-released judgments dated 2014-2024. Match its civil lending
   documents by 案号 against the 2014-2020 archive (civil_case.parquet) and
   regress reappearance on the case's own content flags (relational, coercive
   collection, IOU) interacted with H. Content-selective suppression predicts
   negative backstop x H; content-neutral mirroring predicts ~0.
B. COURT-LEVEL RELEASE-SURVIVAL GATE: court (案号 code) level, 2019 baseline
   volume and composition vs 2022 volume; regress log retention and survival
   on H and on the court's own 2019 docket composition.
Output: output/ext2124/seam_audits.csv
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
import duckdb, glob, sys, io
import numpy as np
import pandas as pd
import pyfixest as pf

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
DATA = str(_REP_PROJECT / "data")
OUT = str(_REP_PROJECT / "output" / "ext2124")
con = duckdb.connect()
con.sql("SET threads TO 8; SET memory_limit='16GB'")
rows = []

# ---------------- A: backfill survival audit (civil lending) ----------------
bf = str(_REP_PROJECT / "data" / "derived" / "extract_2021_2024" / 'civ_2024_10.parquet').replace('\\', '/')
n_by_year = con.sql(f"""
  SELECT year(TRY_CAST(judgment_date AS DATE)) y, COUNT(*) n FROM '{bf}'
  WHERE cause='民间借贷纠纷' GROUP BY 1 ORDER BY 1""").df()
print("backfill lending docs by judgment year:")
print(n_by_year.to_string(index=False))

con.sql(f"""
CREATE OR REPLACE TEMP VIEW arch AS
SELECT case_no, prefecture_code, province,
  rel_txn::INT AS rel_txn, backstop_collection::INT AS backstop_collection,
  evid_iou::INT AS evid_iou, doc_len,
  strftime(jmonth, '%Y') AS jyr
FROM '{DATA}/civil_case.parquet'
WHERE cause='民间借贷纠纷' AND doc_type='judgment'
  AND jmonth BETWEEN TIMESTAMP '2017-01-01' AND TIMESTAMP '2020-12-31'
""")
con.sql(f"""
CREATE OR REPLACE TEMP VIEW bfk AS
SELECT DISTINCT case_no FROM '{bf}'
WHERE cause='民间借贷纠纷' AND doc_type='judgment'
  AND TRY_CAST(judgment_date AS DATE) < DATE '2021-01-01'
""")
surv = con.sql("""
SELECT a.*, (b.case_no IS NOT NULL)::INT AS reappear
FROM arch a LEFT JOIN bfk b USING (case_no)
""").df()
exp = con.sql(f"SELECT prefecture_code, exposure_v2_z AS H FROM '{DATA}/exposure_v2.parquet'").df()
surv = surv.merge(exp, on="prefecture_code", how="inner")
surv["prov_id"] = pd.factorize(surv["province"])[0]
surv["pref_yr"] = surv["prefecture_code"] + "_" + surv["jyr"]
print(f"survival sample: {len(surv):,} archive lending judgments 2017-2020; "
      f"reappearance rate = {surv.reappear.mean():.4f}")

for flagcol in ("backstop_collection", "rel_txn", "evid_iou"):
    surv["fx"] = surv[flagcol] * surv["H"]
    m = pf.feols(f"reappear ~ fx + {flagcol} | pref_yr",
                 data=surv, vcov={"CRV1": "prov_id"})
    est, se, p = m.coef()["fx"], m.se()["fx"], m.pvalue()["fx"]
    lvl = m.coef()[flagcol]
    print(f"A. reappear ~ {flagcol} x H: {est:.5f} (se {se:.5f}, p {p:.3f}); "
          f"level {lvl:.5f}")
    rows.append(dict(audit="backfill_survival", object=flagcol,
                     est=est, se=se, p=p, level_est=lvl, n=int(m._N)))

# ---------------- B: court-level release survival ----------------
ANO = "[（(]\\s*20[0-9]{2}\\s*[）)]\\s*([\\p{Han}]{1,3}[0-9]{0,4})"
con.sql(f"""
CREATE OR REPLACE TEMP VIEW pre AS
SELECT regexp_extract(case_no, '{ANO}', 1) AS ano, prefecture_code, province,
  COUNT(*) AS n_pre, AVG(rel_txn::INT) AS rel_pre, AVG(evid_iou::INT) AS iou_pre
FROM '{DATA}/civil_case.parquet'
WHERE cause='民间借贷纠纷' AND doc_type='judgment'
  AND jmonth BETWEEN TIMESTAMP '2019-01-01' AND TIMESTAMP '2019-12-31'
GROUP BY 1,2,3 HAVING COUNT(*) >= 10 AND ano <> ''
""")
files22 = glob.glob(str(_REP_PROJECT / "data" / "derived" / "extract_2021_2024" / 'civ_2022_*.parquet').replace('\\', '/'))
con.sql(f"""
CREATE OR REPLACE TEMP VIEW post AS
SELECT ano_code AS ano, COUNT(*) AS n_post
FROM read_parquet({files22})
WHERE cause='民间借贷纠纷' AND doc_type='judgment'
GROUP BY 1
""")
courts = con.sql("""
SELECT p.*, COALESCE(q.n_post, 0) AS n_post,
  (COALESCE(q.n_post,0) > 0)::INT AS survives,
  ln(COALESCE(q.n_post,0) + 1) - ln(n_pre) AS log_retention
FROM pre p LEFT JOIN post q USING (ano)
""").df()
courts = courts.merge(exp, on="prefecture_code", how="inner")
courts["prov_id"] = pd.factorize(courts["province"])[0]
print(f"\nB. courts with >=10 lending judgments in 2019: {len(courts):,}; "
      f"share surviving in 2022 release: {courts.survives.mean():.3f}")
for y in ("survives", "log_retention"):
    m = pf.feols(f"{y} ~ H + rel_pre + iou_pre", data=courts,
                 vcov={"CRV1": "prov_id"}, weights="n_pre")
    for c in ("H", "rel_pre", "iou_pre"):
        print(f"   {y} ~ {c}: {m.coef()[c]:.5f} (se {m.se()[c]:.5f}, p {m.pvalue()[c]:.3f})")
        rows.append(dict(audit="court_survival", object=f"{y}~{c}",
                         est=m.coef()[c], se=m.se()[c], p=m.pvalue()[c],
                         level_est=np.nan, n=int(m._N)))

pd.DataFrame(rows).to_csv(f"{OUT}/seam_audits.csv", index=False)
print("written:", f"{OUT}/seam_audits.csv")

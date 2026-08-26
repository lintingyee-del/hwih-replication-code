# -*- coding: utf-8 -*-
"""6B step 57 — publication-delay content selection test in the successor
regime. Lending judgments dated 2022-01..2023-06 appear either in their
contemporaneous monthly file or in a later batch (notably the 2024 files).
If publication screens on content, late-published documents differ in content,
and differentially with exposure H. Spec: flag ~ late x H + late, prefecture x
judgment-month FE, province CRV1, case-level collapsed to cells.
Output: output/ext2124/pubdelay.csv
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
files = glob.glob(str(_REP_PROJECT / "data" / "derived" / "extract_2021_2024" / 'civ_2*.parquet').replace('\\', '/'))

ABBR = {"京": "11", "津": "12", "冀": "13", "晋": "14", "蒙": "15", "内": "15",
        "辽": "21", "吉": "22", "黑": "23", "沪": "31", "苏": "32", "浙": "33",
        "皖": "34", "闽": "35", "赣": "36", "鲁": "37", "豫": "41", "鄂": "42",
        "湘": "43", "粤": "44", "桂": "45", "琼": "46", "渝": "50", "川": "51",
        "黔": "52", "滇": "53", "云": "53", "藏": "54", "陕": "61", "甘": "62",
        "青": "63", "宁": "64", "新": "65"}
abbr_sql = "CASE " + " ".join(f"WHEN ab='{k}' THEN '{v}'" for k, v in ABBR.items()) + " ELSE NULL END"

cells = con.sql(f"""
WITH base AS (
  SELECT rel_txn, backstop_collection, evid_iou,
    TRY_CAST(judgment_date AS DATE) AS jdate,
    CAST(regexp_extract(filename, 'civ_(20[0-9]{{2}})_', 1) AS INT) AS fyear,
    regexp_extract(ano_code, '^([\\p{{Han}}])', 1) AS ab,
    regexp_extract(ano_code, '([0-9]+)$', 1) AS code
  FROM read_parquet({files}, filename=true)
  WHERE cause='民间借贷纠纷' AND doc_type='judgment'
), tagged AS (
  SELECT *, year(jdate) AS jyear, strftime(jdate, '%Y-%m') AS jmonth,
    {abbr_sql} AS provcode,
    (fyear > year(jdate))::INT AS late
  FROM base
  WHERE jdate BETWEEN DATE '2022-01-01' AND DATE '2023-06-30'
)
SELECT
  CASE WHEN provcode IN ('11','12','31','50') THEN provcode || '0000'
       WHEN length(code) = 4 THEN provcode || substr(code,1,2) || '00'
       WHEN length(code) = 2 THEN provcode || code || '00'
       ELSE NULL END AS prefecture_code,
  jmonth, late, COUNT(*) AS n,
  AVG(rel_txn) AS rel_txn, AVG(backstop_collection) AS backstop_collection,
  AVG(evid_iou) AS evid_iou
FROM tagged WHERE provcode IS NOT NULL
GROUP BY 1,2,3
""").df()
exp = con.sql(f"SELECT prefecture_code, province, exposure_v2_z AS H FROM '{DATA}/exposure_v2.parquet'").df()
d = cells.dropna(subset=["prefecture_code"]).merge(exp, on="prefecture_code")
d["prov_id"] = pd.factorize(d["province"])[0]
d["pref_jm"] = d["prefecture_code"] + "_" + d["jmonth"]
d["lateH"] = d["late"] * d["H"]
print(f"cells: {len(d):,}; docs: {d.n.sum():,.0f}; late share: "
      f"{(d.late * d.n).sum() / d.n.sum():.3f}")
rows = []
for y in ("rel_txn", "backstop_collection", "evid_iou"):
    m = pf.feols(f"{y} ~ lateH + late | pref_jm", data=d,
                 vcov={"CRV1": "prov_id"}, weights="n")
    print(f"{y}: late x H = {m.coef()['lateH']:.5f} (se {m.se()['lateH']:.5f}, "
          f"p {m.pvalue()['lateH']:.3f}); late level = {m.coef()['late']:.5f} "
          f"(se {m.se()['late']:.5f})")
    rows.append(dict(outcome=y, lateH=m.coef()["lateH"], se_lateH=m.se()["lateH"],
                     p_lateH=m.pvalue()["lateH"], late=m.coef()["late"],
                     se_late=m.se()["late"], n=int(m._N)))
pd.DataFrame(rows).to_csv(f"{OUT}/pubdelay.csv", index=False)
print("written:", f"{OUT}/pubdelay.csv")

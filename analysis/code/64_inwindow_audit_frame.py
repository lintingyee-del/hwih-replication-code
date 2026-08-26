# -*- coding: utf-8 -*-
"""6B step 64 — in-window dual-coding frame: upgrade the 2014-2020 validation
from machine-gold + single-human-check to genuine dual-coder blind adjudication.
Anchor months 2017-06, 2018-06, 2019-06, 2020-06 (macro schema, direct paths);
120 lending judgments + 60 target-crime docs per month, deterministic hash
sampling, same rubric as the post-2020 pool (audit2124/RUBRIC.md).
Output: data/audit2124/coding_sheet_{civ,crim}_inwindow.csv
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
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
BASE = str(_REP_JUDGMENTS)
OUTD = str(_REP_PROJECT / "data" / "audit2124")
CRIME_RX = ("赌博|开设赌场|组织卖淫|非法经营|走私普通货物|危险驾驶|交通肇事|过失致人死亡"
            "|盗窃|故意伤害|黑社会性质组织|敲诈勒索|诈骗|非法拘禁|寻衅滋事|聚众斗殴|强迫交易")

def month_path(y, m):
    if y == 2020:
        cands = glob.glob(f"{BASE}/2020_Court_Judgments_CSV_Extracted_Partial/**/"
                          f"2020年{m:02d}月*.csv", recursive=True)
        return cands[0] if cands else None
    return (f"{BASE}/{y}_Court_Judgments_CSV/{y}_MacroData_Court_Judgments_CSV/"
            f"{y}年{m:02d}月裁判文书数据.csv")

con = duckdb.connect()
con.sql("SET threads TO 10; SET memory_limit='20GB'")
civ_pool, crim_pool = [], []
for y in (2017, 2018, 2019, 2020):
    p = month_path(y, 6)
    if p is None or not os.path.exists(p):
        alts = glob.glob(f"{BASE}/2020_Court_Judgments_CSV_Extracted_Partial/**/*.csv",
                         recursive=True) if y == 2020 else []
        p = sorted(alts)[0] if alts else None
        if p is None:
            print(f"{y}: no file found, skipped")
            continue
    fu = p.replace("\\", "/").replace("'", "''")
    ym = f"{y}-06"
    civ = con.sql(f"""
      SELECT '{ym}' AS ym, 'macro-inwindow' AS source, 案号 AS case_no, 案件名称, 案由,
             裁判日期 AS judgment_date, 全文
      FROM (SELECT * FROM read_csv('{fu}', auto_detect=true, sample_size=2000,
                                   ignore_errors=true, all_varchar=true)
            WHERE 案件类型='民事案件' AND 案由='民间借贷纠纷' AND 案件名称 LIKE '%判决%')
      ORDER BY hash(案号) LIMIT 120""").df()
    crim = con.sql(f"""
      SELECT '{ym}' AS ym, 'macro-inwindow' AS source, 案号 AS case_no, 案件名称, 案由,
             裁判日期 AS judgment_date, 全文
      FROM (SELECT * FROM read_csv('{fu}', auto_detect=true, sample_size=2000,
                                   ignore_errors=true, all_varchar=true)
            WHERE 案件类型='刑事案件'
              AND regexp_matches(COALESCE(案由, 案件名称, ''), '{CRIME_RX}'))
      ORDER BY hash(案号) LIMIT 60""").df()
    civ_pool.append(civ); crim_pool.append(crim)
    print(f"{ym}: civ {len(civ)}, crim {len(crim)} from {os.path.basename(p)}")

civ = pd.concat(civ_pool, ignore_index=True)
crim = pd.concat(crim_pool, ignore_index=True)
civ.insert(0, "doc_id", ["WC%04d" % i for i in range(1, len(civ) + 1)])
crim.insert(0, "doc_id", ["WK%04d" % i for i in range(1, len(crim) + 1)])
for g in ["gold_is_judgment", "gold_rel_txn", "gold_evid_iou", "gold_orig_year", "coder_notes"]:
    civ[g] = ""
for g in ["gold_is_target_crime", "gold_backstop", "gold_detention_debt", "coder_notes"]:
    crim[g] = ""
civ.to_csv(f"{OUTD}/coding_sheet_civ_inwindow.csv", index=False, encoding="utf-8-sig")
crim.to_csv(f"{OUTD}/coding_sheet_crim_inwindow.csv", index=False, encoding="utf-8-sig")
print(f"in-window pools: civ {len(civ)}, crim {len(crim)} -> {OUTD}")

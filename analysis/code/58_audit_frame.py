# -*- coding: utf-8 -*-
"""6B step 58 — stratified gold-standard re-audit pool for the post-2020
extension (referee-spec: by source regime x anchor month; H-tercile
stratification applied at sheet-assembly). Anchor months: 2021-03, 2021-08
(macro), 2022-03, 2022-09, 2023-03 (ws), 2024-03, 2024-08 (s41).
Per month: 120 civil lending judgments + 60 criminal target-offense docs,
full text, deterministic hash sampling.
Output: data/audit2124/{civ,crim}_pool.parquet + coding-sheet CSVs + rubric.
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
import duckdb, os, sys, io, zipfile, shutil

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
BASE = str(_REP_JUDGMENTS)
MZ = BASE + "/2021_2024_Court_Judgments_Monthly_Zips"
TMP = str(_REP_RESTRICTED / "work" / "scan_tmp").replace('\\', '/')
OUTD = str(_REP_PROJECT / "data" / "audit2124")
os.makedirs(OUTD, exist_ok=True)
CRIME_RX = ("赌博|开设赌场|组织卖淫|非法经营|走私普通货物|危险驾驶|交通肇事|过失致人死亡"
            "|盗窃|故意伤害|黑社会性质组织|敲诈勒索|诈骗|非法拘禁|寻衅滋事|聚众斗殴|强迫交易")

ANCHORS = [
    ("2021-03", "macro", ("path", f"{BASE}/2021_Court_Judgments_CSV_Extracted_Jan_Oct/2021_MacroData_Court_Judgments_CSV/2021年03月裁判文书数据.csv")),
    ("2021-08", "macro", ("path", f"{BASE}/2021_Court_Judgments_CSV_Extracted_Jan_Oct/2021_MacroData_Court_Judgments_CSV/2021年08月裁判文书数据.csv")),
    ("2022-03", "ws", ("zip", f"{MZ}/2022_Court_Judgments_Monthly_Zips/ws_2022_03.zip")),
    ("2022-09", "ws", ("zip", f"{MZ}/2022_Court_Judgments_Monthly_Zips/ws_2022_09.zip")),
    ("2023-03", "ws", ("zip", f"{MZ}/2023_Court_Judgments_Monthly_Zips/ws_2023_03.zip")),
    ("2024-03", "s41", ("nested", f"{MZ}/2024-1.zip", "s41_202403.zip")),
    ("2024-08", "s41", ("nested", f"{MZ}/2024-2.zip", "s41_202408.zip")),
]


def fixname(n):
    try:
        return n.encode("cp437").decode("gbk")
    except Exception:
        return n


def biggest_csv(zf):
    return max((i for i in zf.infolist() if not i.is_dir()
                and i.filename.lower().endswith(".csv") and "__MACOSX" not in i.filename),
               key=lambda i: i.file_size)


def materialize(g):
    if g[0] == "path":
        return g[1], []
    if g[0] == "zip":
        with zipfile.ZipFile(g[1]) as zf, zf.open(biggest_csv(zf)) as src, \
             open(os.path.join(TMP, "aud.csv"), "wb") as out:
            shutil.copyfileobj(src, out, 64 * 1024 * 1024)
        return os.path.join(TMP, "aud.csv"), [os.path.join(TMP, "aud.csv")]
    with zipfile.ZipFile(g[1]) as zo:
        m = [i for i in zo.infolist() if fixname(i.filename).endswith(g[2])
             and "__MACOSX" not in i.filename][0]
        with zo.open(m) as src, open(os.path.join(TMP, "aud_inner.zip"), "wb") as out:
            shutil.copyfileobj(src, out, 64 * 1024 * 1024)
    with zipfile.ZipFile(os.path.join(TMP, "aud_inner.zip")) as zi, \
         zi.open(biggest_csv(zi)) as src, open(os.path.join(TMP, "aud.csv"), "wb") as out:
        shutil.copyfileobj(src, out, 64 * 1024 * 1024)
    return os.path.join(TMP, "aud.csv"), [os.path.join(TMP, "aud.csv"),
                                          os.path.join(TMP, "aud_inner.zip")]


con = duckdb.connect()
con.sql("SET threads TO 10; SET preserve_insertion_order=false; SET memory_limit='20GB'")
for ym, source, getter in ANCHORS:
    civ_dest = f"{OUTD}/pool_civ_{ym}.parquet".replace("\\", "/")
    if os.path.exists(civ_dest):
        continue
    csv_path, cleanup = materialize(getter)
    fu = csv_path.replace("\\", "/").replace("'", "''")
    cm = ("案由 = '民间借贷纠纷'" if source == "macro" else "案由 LIKE '%民间借贷纠纷%'")
    con.sql(f"""COPY (
      SELECT '{ym}' AS ym, '{source}' AS source, 案号 AS case_no, 案件名称, 案由,
             裁判日期 AS judgment_date, 全文
      FROM (SELECT * FROM read_csv('{fu}', auto_detect=true, sample_size=2000,
                                   ignore_errors=true, all_varchar=true)
            WHERE 案件类型='民事案件' AND {cm} AND 案件名称 LIKE '%判决%')
      ORDER BY hash(案号) LIMIT 120
    ) TO '{civ_dest}' (FORMAT PARQUET)""")
    con.sql(f"""COPY (
      SELECT '{ym}' AS ym, '{source}' AS source, 案号 AS case_no, 案件名称, 案由,
             裁判日期 AS judgment_date, 全文
      FROM (SELECT * FROM read_csv('{fu}', auto_detect=true, sample_size=2000,
                                   ignore_errors=true, all_varchar=true)
            WHERE 案件类型='刑事案件'
              AND regexp_matches(COALESCE(案由, 案件名称, ''), '{CRIME_RX}'))
      ORDER BY hash(案号) LIMIT 60
    ) TO '{OUTD}/pool_crim_{ym}.parquet' (FORMAT PARQUET)""")
    print(f"{ym} [{source}] sampled", flush=True)
    for p in cleanup:
        try:
            os.remove(p)
        except OSError:
            pass

import glob
import pandas as pd
civ = pd.concat([pd.read_parquet(f) for f in glob.glob(f"{OUTD}/pool_civ_*.parquet")])
crim = pd.concat([pd.read_parquet(f) for f in glob.glob(f"{OUTD}/pool_crim_*.parquet")])
civ.insert(0, "doc_id", ["C%04d" % i for i in range(1, len(civ) + 1)])
crim.insert(0, "doc_id", ["K%04d" % i for i in range(1, len(crim) + 1)])
for g in ["gold_is_judgment", "gold_rel_txn", "gold_evid_iou", "gold_orig_year", "coder_notes"]:
    civ[g] = ""
for g in ["gold_is_target_crime", "gold_backstop", "gold_detention_debt", "coder_notes"]:
    crim[g] = ""
civ.to_csv(f"{OUTD}/coding_sheet_civ.csv", index=False, encoding="utf-8-sig")
crim.to_csv(f"{OUTD}/coding_sheet_crim.csv", index=False, encoding="utf-8-sig")
civ.drop(columns=["全文"]).to_csv(f"{OUTD}/frame_index_civ.csv", index=False, encoding="utf-8-sig")
crim.drop(columns=["全文"]).to_csv(f"{OUTD}/frame_index_crim.csv", index=False, encoding="utf-8-sig")
print(f"civil pool {len(civ)} docs, criminal pool {len(crim)} docs -> {OUTD}")

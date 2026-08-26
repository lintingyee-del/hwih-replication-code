# -*- coding: utf-8 -*-
"""6B step 69 — total archive document counts by year x case type, 2014-2020,
from the raw monthly CSVs (all causes, not just the 17-offense/7-cause analysis
sets). Denominator side for the official-statistics coverage benchmark.
Restartable via output presence. Output: output/ext2124/archive_totals.csv"""

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
import duckdb, glob, os, re, sys, time, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
BASE = str(_REP_JUDGMENTS)
OUT = str(_REP_PROJECT / 'output' / 'ext2124' / 'archive_totals.csv').replace('\\', '/')
YEAR_DIRS = {
    2014: "2014_Court_Judgments_CSV", 2015: "2015_Court_Judgments_CSV",
    2016: "2016_Court_Judgments_CSV", 2017: "2017_Court_Judgments_CSV",
    2018: "2018_Court_Judgments_CSV", 2019: "2019_Court_Judgments_CSV",
    2020: "2020_Court_Judgments_CSV_Extracted_Partial",
}
done = set()
if os.path.exists(OUT):
    with open(OUT, encoding="utf-8") as fh:
        done = {ln.split(",")[0] for ln in fh if "," in ln}
else:
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("ym,case_type,doc_class,n\n")

con = duckdb.connect()
con.sql("SET threads TO 10; SET preserve_insertion_order=false; SET memory_limit='20GB'")
files = []
for yr, yd in sorted(YEAR_DIRS.items()):
    for f in sorted(glob.glob(os.path.join(BASE, yd, "**", "*.csv"), recursive=True)):
        m = re.search(r"(20[12][0-9])年([01][0-9])月", os.path.basename(f))
        if m:
            files.append((f"{m.group(1)}-{m.group(2)}", f))
print(f"{len(files)} monthly files, {len(done)} done", flush=True)
for ym, f in files:
    if ym in done:
        continue
    t0 = time.time()
    fu = f.replace("\\", "/").replace("'", "''")
    try:
        rows = con.sql(f"""
          SELECT COALESCE(案件类型,'NA'),
            CASE WHEN 案件名称 LIKE '%判决%' THEN 'judgment'
                 WHEN 案件名称 LIKE '%裁定%' THEN 'ruling'
                 WHEN 案件名称 LIKE '%调解%' THEN 'mediation' ELSE 'other' END,
            COUNT(*)
          FROM read_csv('{fu}', auto_detect=true, sample_size=2000,
                        ignore_errors=true, all_varchar=true)
          GROUP BY 1,2""").fetchall()
        with open(OUT, "a", encoding="utf-8") as fh:
            for ct, dc, n in rows:
                fh.write(f"{ym},{ct},{dc},{n}\n")
        print(f"{ym} done in {time.time()-t0:.0f}s", flush=True)
    except Exception as e:
        print(f"{ym} FAILED {type(e).__name__}: {str(e)[:150]}", flush=True)
print("DONE", flush=True)

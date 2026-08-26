# -*- coding: utf-8 -*-
"""Map E: archive layout: files per year, case-type and target-cause counts on samples."""

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
import glob, os, sys, io
import duckdb

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
BASE = str(_REP_JUDGMENTS)
con = duckdb.connect()

# enumerate monthly csvs per year folder (2014-2020)
layout = {}
for ydir in sorted(os.listdir(BASE)):
    full = os.path.join(BASE, ydir)
    if not os.path.isdir(full): continue
    csvs = glob.glob(os.path.join(full, "**", "*.csv"), recursive=True)
    if csvs:
        layout[ydir] = (len(csvs), sum(os.path.getsize(c) for c in csvs)/1e9)
        print(f"{ydir}: {len(csvs)} csv, {layout[ydir][1]:.1f} GB")

# sample one file: case-type mix and target causes
f = glob.glob(os.path.join(BASE, "2017_Court_Judgments_CSV", "*", "*06*.csv"))[0].replace("\\", "/")
q = f"""
SELECT 案件类型, COUNT(*) n FROM read_csv('{f}', auto_detect=true, sample_size=1000,
  ignore_errors=true) GROUP BY 1 ORDER BY 2 DESC
"""
print(con.sql(q).df().to_string(index=False))
q2 = f"""
SELECT 案由, COUNT(*) n FROM read_csv('{f}', auto_detect=true, sample_size=1000, ignore_errors=true)
WHERE 案由 IN ('民间借贷纠纷','买卖合同纠纷','保证合同纠纷','追偿权纠纷','合伙协议纠纷',
               '机动车交通事故责任纠纷','租赁合同纠纷')
GROUP BY 1 ORDER BY 2 DESC
"""
print(con.sql(q2).df().to_string(index=False))

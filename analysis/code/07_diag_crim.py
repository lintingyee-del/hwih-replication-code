# -*- coding: utf-8 -*-
"""Diagnose criminal-row 案由 values in a raw monthly CSV."""

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
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
BASE = str(_REP_JUDGMENTS)
f = glob.glob(os.path.join(BASE, "2014_Court_Judgments_CSV", "**", "*01*.csv"),
              recursive=True)[0].replace("\\", "/")
con = duckdb.connect(); con.sql("SET threads TO 4")
df = con.sql(f"""
  SELECT 案由, COUNT(*) n FROM read_csv('{f}', auto_detect=true, sample_size=2000,
    ignore_errors=true)
  WHERE 案件类型 = '刑事案件' GROUP BY 1 ORDER BY 2 DESC LIMIT 25
""").df()
print(df.to_string(index=False))
print("----- null share -----")
print(con.sql(f"""
  SELECT COUNT(*) total, COUNT(案由) nonnull FROM read_csv('{f}', auto_detect=true,
    sample_size=2000, ignore_errors=true) WHERE 案件类型='刑事案件'
""").df().to_string(index=False))

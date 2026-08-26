# -*- coding: utf-8 -*-
"""Inspect raw monthly judgment CSVs on E: — schema, case-type mix, row anchors."""

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
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
BASE = str(_REP_JUDGMENTS)
files = sorted(glob.glob(os.path.join(BASE, "2017_Court_Judgments_CSV", "*", "*.csv")))
print("2017 files:", len(files))
f = files[5]
print("sample file:", os.path.basename(f), f"{os.path.getsize(f)/1e9:.2f} GB")

head = pd.read_csv(f, nrows=200, encoding="utf-8", encoding_errors="replace")
with open(str(_REP_PROJECT / 'data' / 'raw_csv_schema.txt').replace('\\', '/'), "w", encoding="utf-8") as out:
    out.write(f"file: {f}\ncolumns ({len(head.columns)}):\n")
    for c in head.columns:
        out.write(f"  {c} | example: {str(head[c].dropna().iloc[0])[:80] if head[c].notna().any() else 'NA'}\n")
print("columns:", list(head.columns))

# case-type composition: look for 案件类型/案由 column
tc = [c for c in head.columns if "类型" in c or "案由" in c or "案件" in c]
print("type-ish columns:", tc)
if tc:
    chunks = pd.read_csv(f, usecols=tc[:2], chunksize=200_000, encoding="utf-8",
                         encoding_errors="replace")
    agg = {}
    for ch in chunks:
        for v, n in ch[tc[0]].value_counts().items():
            agg[v] = agg.get(v, 0) + int(n)
    print("case-type counts:", dict(sorted(agg.items(), key=lambda x: -x[1])[:10]))

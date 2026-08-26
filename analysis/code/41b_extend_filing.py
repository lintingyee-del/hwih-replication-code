# -*- coding: utf-8 -*-
"""6B step 41b — extend the filing-date extraction to judgment months 2019-04..2019-12
(pre-COVID) so that post-cohort filings (2018-09..2019-03) are observed to completion, not
right-truncated at the 2019-03 frame edge. Append to civil_filing.parquet (dedup on 案号).
This lets the congestion spillover use a truncation-robust outcome (resolved-within-X-days)."""

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
import re, glob, time, pandas as pd
BASE = str(_REP_JUDGMENTS)
OUT = str(_REP_PROJECT / 'data' / 'civil_filing.parquet').replace('\\', '/')
CAUSES = set(pd.read_parquet(str(_REP_PROJECT / 'data' / 'civil_case.parquet').replace('\\', '/'), columns=["cause"])["cause"].unique())
D = r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日"
FIL = re.compile(rf"(?:立案受理|立案|受理)[^。；;]{{0,18}}?{D}|{D}[^。；;]{{0,6}}?(?:立案|受理)")
def extract(t):
    if not isinstance(t, str): return None
    m = FIL.search(t[:6000])
    if not m: return None
    g = m.groups(); y, mo, dy = (g[0:3] if g[0] else g[3:6])
    try: return f"{int(y):04d}-{int(mo):02d}-{int(dy):02d}"
    except: return None

months = [(2019, m) for m in range(4, 13)]
paths = [f"{BASE}/{y}_Court_Judgments_CSV/{y}_MacroData_Court_Judgments_CSV/{y}年{m:02d}月裁判文书数据.csv" for y, m in months]
out = []
for p in paths:
    if not glob.glob(p): print(f"[skip] {p}", flush=True); continue
    t0 = time.time(); got = 0
    for ch in pd.read_csv(p, chunksize=60000, encoding="utf-8", usecols=["案号","案由","全文"], dtype=str, on_bad_lines="skip"):
        m = ch[ch["案由"].isin(CAUSES)].copy()
        if len(m) == 0: continue
        m["filing_ymd"] = m["全文"].map(extract)
        out.append(m[["案号","案由","filing_ymd"]]); got += len(m)
    print(f"[ok] {p.split('/')[-1]} civil={got} {time.time()-t0:.0f}s", flush=True)

new = pd.concat(out, ignore_index=True)
old = pd.read_parquet(OUT)
comb = pd.concat([old, new], ignore_index=True).drop_duplicates("案号")
comb.to_parquet(OUT, index=False)
print(f"\n[done] old {len(old):,} + new {len(new):,} -> {len(comb):,} unique; "
      f"filing extractable {comb['filing_ymd'].notna().mean():.3f}", flush=True)

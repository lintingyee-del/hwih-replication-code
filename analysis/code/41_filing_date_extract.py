# -*- coding: utf-8 -*-
"""6B step 41 (a1) — extract the civil FILING / 立案受理 date from judgment text for the
clean window, keyed by case_no, for the collected civil causes. Powers the congestion-
spillover design: duration = 裁判日期 - filing. Autopsy (validated) put extractability at
~70% and durations clean (median 69d). Output: data/civil_filing.parquet (案号, 案由,
filing_ymd). Usage: python 41_filing_date_extract.py [test]  (test = one month, timed)."""

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
import sys, re, glob, time, pandas as pd
BASE = str(_REP_JUDGMENTS)
OUT = str(_REP_PROJECT / 'data' / 'civil_filing.parquet').replace('\\', '/')
TEST = len(sys.argv) > 1 and sys.argv[1] == "test"
CAUSES = set(pd.read_parquet(str(_REP_PROJECT / 'data' / 'civil_case.parquet').replace('\\', '/'),
                             columns=["cause"])["cause"].unique())
print(f"[causes] {len(CAUSES)} collected civil causes", flush=True)
D = r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日"
FIL = re.compile(rf"(?:立案受理|立案|受理)[^。；;]{{0,18}}?{D}|{D}[^。；;]{{0,6}}?(?:立案|受理)")
def extract(t):
    if not isinstance(t, str): return None
    m = FIL.search(t[:6000])  # regex is proximity-constrained to 立案/受理, so wider window stays precise
    if not m: return None
    g = m.groups(); y, mo, dy = (g[0:3] if g[0] else g[3:6])
    try: return f"{int(y):04d}-{int(mo):02d}-{int(dy):02d}"
    except: return None

def month_paths():
    ym = [(2017, m) for m in range(1, 13)] + [(2018, m) for m in range(1, 13)] + [(2019, m) for m in range(1, 4)]
    if TEST: ym = [(2018, 6)]
    return [f"{BASE}/{y}_Court_Judgments_CSV/{y}_MacroData_Court_Judgments_CSV/{y}年{m:02d}月裁判文书数据.csv" for y, m in ym]

out = []
for p in month_paths():
    if not glob.glob(p): print(f"[skip] {p}", flush=True); continue
    t0 = time.time(); got = 0
    for ch in pd.read_csv(p, chunksize=60000, encoding="utf-8", usecols=["案号","案由","全文"], dtype=str, on_bad_lines="skip"):
        m = ch[ch["案由"].isin(CAUSES)].copy()
        if len(m) == 0: continue
        m["filing_ymd"] = m["全文"].map(extract)
        out.append(m[["案号","案由","filing_ymd"]]); got += len(m)
    print(f"[ok] {p.split('/')[-1]} civil={got} rate={pd.concat(out[-1:])['filing_ymd'].notna().mean():.3f} {time.time()-t0:.0f}s", flush=True)

df = pd.concat(out, ignore_index=True).drop_duplicates("案号")
df.to_parquet(OUT, index=False)
print(f"\n[done] {len(df):,} civil cases; filing extractable {df['filing_ymd'].notna().mean():.3f} -> {OUT}", flush=True)

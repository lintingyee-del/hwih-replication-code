# -*- coding: utf-8 -*-
"""6B step 43 (a2) — build the GOLD-STANDARD audit sample for the filing-date extractor.
The extractor is a regex; its accuracy is month-invariant, so we audit on one raw month.
Sample first-instance (民初) 7-cause cases: cases where the regex EXTRACTED a 立案/受理 date
(audit PRECISION: is it the true filing date?) and cases where it did NOT (audit RECALL:
did we miss a stated filing date?). For each, save an 800-char text window + the machine
date. Output JSON is handed to verification agents; no date is revealed as 'correct'.
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
import re, json, pandas as pd
F = str(_REP_JUDGMENTS / '2018_Court_Judgments_CSV' / '2018_MacroData_Court_Judgments_CSV' / '2018年06月裁判文书数据.csv').replace('\\', '/')
OUT = str(_REP_PACKAGE / "restricted_data" / "source_data").replace('\\', '/')
CAUSES = set(pd.read_parquet(str(_REP_PROJECT / 'data' / 'civil_case.parquet').replace('\\', '/'), columns=["cause"])["cause"].unique())
D = r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日"
FIL = re.compile(rf"(?:立案受理|立案|受理)[^。；;]{{0,18}}?{D}|{D}[^。；;]{{0,6}}?(?:立案|受理)")
FIRST = re.compile(r"民初|民商初|民一初|民再")  # first-instance only (match civil_case universe)
def ext(t):
    if not isinstance(t, str): return None, None
    m = FIL.search(t[:6000])
    if not m: return None, None
    g = m.groups(); y, mo, dy = (g[0:3] if g[0] else g[3:6])
    try: return f"{int(y):04d}-{int(mo):02d}-{int(dy):02d}", m.span()
    except: return None, None

hit, miss = [], []
for ch in pd.read_csv(F, chunksize=60000, encoding="utf-8", usecols=["案号","案由","裁判日期","全文"], dtype=str, on_bad_lines="skip"):
    m = ch[(ch["案由"].isin(CAUSES)) & (ch["案号"].fillna("").str.contains(FIRST))]
    for _, r in m.iterrows():
        t = r["全文"];  date, span = ext(t)
        if date and span:
            s = max(0, span[0]-120); win = t[s:span[1]+80]
            if len(hit) < 160: hit.append(dict(case=r["案号"], cause=r["案由"], judg=r["裁判日期"], machine_date=date, window=win))
        elif isinstance(t, str) and len(t) > 200:
            if len(miss) < 60: miss.append(dict(case=r["案号"], cause=r["案由"], judg=r["裁判日期"], window=t[:900]))
    if len(hit) >= 160 and len(miss) >= 60: break

json.dump({"hit": hit, "miss": miss}, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"[audit sample] precision cases (extracted) {len(hit)}; recall cases (not extracted) {len(miss)} -> {OUT}", flush=True)

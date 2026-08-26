# -*- coding: utf-8 -*-
"""6B step 32 — extract organizational-vs-individual PLAINTIFF flag for lending
judgments in the clean window (2017-01..2019-03), from the raw 当事人 field on E:.
This powers Cut 1: does the judicialization rise load on individual/acquaintance
lending or on commercial/professional (org) lending? Confound (busted professional
book) -> org side; backstop-removal mechanism -> individual side. Opposite signs.

Names are anonymized per-document (张某1), so cross-case repeat-counting of
individuals is unreliable; organization names survive enough to detect. First-listed
party approximates the plaintiff (原告) in first-instance cases (~91% of the docket).
Output: data/civil_party_orgflag.parquet  (案号, first_org, any_org, proc).
Usage: python 32_party_org_extract.py [test]
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
import sys, re, glob, pandas as pd
BASE = str(_REP_JUDGMENTS)
OUT = str(_REP_PROJECT / 'data' / 'civil_party_orgflag.parquet').replace('\\', '/')
TEST = len(sys.argv) > 1 and sys.argv[1] == "test"

ORG = re.compile(r"公司|小额贷款|小贷|投资|担保|典当|资产管理|信用社|合作社|财务|融资")
def month_paths():
    ym = [(2017, m) for m in range(1, 13)] + [(2018, m) for m in range(1, 13)] \
         + [(2019, m) for m in range(1, 4)]
    if TEST: ym = [(2018, 6)]
    ps = []
    for y, m in ym:
        p = f"{BASE}/{y}_Court_Judgments_CSV/{y}_MacroData_Court_Judgments_CSV/{y}年{m:02d}月裁判文书数据.csv"
        ps.append(p)
    return ps

def first_party(s):
    if not isinstance(s, str) or not s: return ""
    return re.split(r"[,，、;；]", s)[0]

rows = []
for p in month_paths():
    if not glob.glob(p):
        print(f"[skip missing] {p}", flush=True); continue
    got = 0
    try:
        it = pd.read_csv(p, chunksize=100000, encoding="utf-8",
                         usecols=["案号", "当事人", "案由", "审理程序"], dtype=str,
                         on_bad_lines="skip")
        for ch in it:
            m = ch[ch["案由"] == "民间借贷纠纷"].copy()
            if len(m) == 0: continue
            fp = m["当事人"].map(first_party)
            m["first_org"] = fp.str.contains(ORG, na=False)
            m["any_org"] = m["当事人"].fillna("").str.contains(ORG)
            m["proc"] = m["审理程序"]
            rows.append(m[["案号", "first_org", "any_org", "proc"]])
            got += len(m)
    except Exception as e:
        print(f"[error] {p}: {str(e)[:100]}", flush=True); continue
    print(f"[ok] {p.split('/')[-1]}  lending={got}", flush=True)

out = pd.concat(rows, ignore_index=True).drop_duplicates("案号")
out.to_parquet(OUT, index=False)
print(f"\n[done] {len(out)} lending case flags -> {OUT}", flush=True)
print("first_org share:", round(out['first_org'].mean(), 3),
      "| any_org share:", round(out['any_org'].mean(), 3), flush=True)

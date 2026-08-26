# -*- coding: utf-8 -*-
"""6B step 77 — prosecution-date extractor audit sample, adjudication-ready.

Re-exports the 2018-09 audit sample with snippets adequate for adjudication:
hit rows carry the +/-60/+40 window around the match; no-hit rows carry the
full head the extractor searched (3,500 chars), so false negatives are
assessable. Splits into three slices for parallel adjudication.

Output: output/audit_prosdate/slice_{1,2,3}.csv (+ full.csv)
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
import os, re, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd

BASE = str(_REP_JUDGMENTS)
OUTD = str(_REP_PROJECT / "output" / "audit_prosdate")
os.makedirs(OUTD, exist_ok=True)

FAM = ("赌博", "开设赌场", "组织卖淫", "非法经营", "走私普通货物、物品", "危险驾驶",
       "交通肇事", "过失致人死亡", "盗窃", "故意伤害", "非法拘禁", "寻衅滋事",
       "聚众斗殴", "强迫交易", "敲诈勒索", "组织、领导、参加黑社会性质组织", "诈骗")
D = r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日"
PROS = re.compile(rf"{D}[^。]{{0,40}}?向本院提起公诉")

p = (f"{BASE}/2018_Court_Judgments_CSV/2018_MacroData_Court_Judgments_CSV/"
     f"2018年09月裁判文书数据.csv")
snips = []
for ch in pd.read_csv(p, chunksize=60000, encoding="utf-8",
                      usecols=["案号", "案件类型", "案由", "全文"], dtype=str,
                      on_bad_lines="skip"):
    a = ch[(ch["案件类型"] == "刑事案件") & ch["案由"].isin(FAM)
           & ch["案号"].str.contains("刑初", na=False)]
    for _, r in a.iterrows():
        t = r["全文"] if isinstance(r["全文"], str) else ""
        mm = PROS.search(t[:3500])
        if mm:
            lo = max(0, mm.start() - 80)
            snips.append(dict(case_no=r["案号"], anyou=r["案由"], hit=1,
                              date=f"{int(mm.group(1)):04d}-{int(mm.group(2)):02d}-{int(mm.group(3)):02d}",
                              snippet=t[lo:mm.end() + 60]))
        else:
            snips.append(dict(case_no=r["案号"], anyou=r["案由"], hit=0, date="",
                              snippet=t[:3500]))
sn = pd.DataFrame(snips)
aud = pd.concat([sn[sn["hit"] == 1].sample(min(200, int((sn["hit"] == 1).sum())),
                                           random_state=42),
                 sn[sn["hit"] == 0].sample(min(100, int((sn["hit"] == 0).sum())),
                                           random_state=42)]).reset_index(drop=True)
aud["row_id"] = aud.index
aud.to_csv(f"{OUTD}/full.csv", index=False, encoding="utf-8-sig")
for k in range(3):
    aud.iloc[k::3].to_csv(f"{OUTD}/slice_{k+1}.csv", index=False,
                          encoding="utf-8-sig")
print(f"exported {len(aud)} rows ({int((aud['hit']==1).sum())} hits, "
      f"{int((aud['hit']==0).sum())} misses) -> 3 slices", flush=True)

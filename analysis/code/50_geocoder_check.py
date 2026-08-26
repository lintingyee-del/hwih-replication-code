# -*- coding: utf-8 -*-
"""6B step 50 — validate the 案号 court-code geocoder on 2021 Jan-Sep macro months,
where court names (-> court_xwalk prefecture) and case numbers coexist.
Parsed prefecture = province-abbrev code + first two digits of the court code
(4-digit = basic court, first2 = prefecture; 2-digit = intermediate court).
Reports coverage and agreement vs the court-name crosswalk."""

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
import duckdb, sys, io
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
BASE = str(_REP_JUDGMENTS)
XW = str(_REP_PROJECT / 'data' / 'court_xwalk.parquet').replace('\\', '/')

ABBR = {"京": "11", "津": "12", "冀": "13", "晋": "14", "蒙": "15", "内": "15",
        "辽": "21", "吉": "22", "黑": "23", "沪": "31", "苏": "32", "浙": "33",
        "皖": "34", "闽": "35", "赣": "36", "鲁": "37", "豫": "41", "鄂": "42",
        "湘": "43", "粤": "44", "桂": "45", "琼": "46", "渝": "50", "川": "51",
        "黔": "52", "滇": "53", "云": "53", "藏": "54", "陕": "61", "甘": "62",
        "青": "63", "宁": "64", "新": "65"}

con = duckdb.connect()
con.sql("SET threads TO 8; SET memory_limit='16GB'")
rows = []
for m in ("03", "08"):
    f = (f"{BASE}/2021_Court_Judgments_CSV_Extracted_Jan_Oct/"
         f"2021_MacroData_Court_Judgments_CSV/2021年{m}月裁判文书数据.csv")
    d = con.sql(f"""
      SELECT 法院 AS court,
        regexp_extract(案号, '[（(]\\s*20[0-9]{{2}}\\s*[）)]\\s*([\\p{{Han}}]{{1,3}})([0-9]{{0,4}})', 1) AS ab,
        regexp_extract(案号, '[（(]\\s*20[0-9]{{2}}\\s*[）)]\\s*([\\p{{Han}}]{{1,3}})([0-9]{{0,4}})', 2) AS code
      FROM read_csv('{f}', auto_detect=true, sample_size=2000, ignore_errors=true,
                    all_varchar=true)
      WHERE 案件类型='民事案件' AND 案号 IS NOT NULL
    """).df()
    xw = pd.read_parquet(XW)
    d = d.merge(xw, left_on="court", right_on="court_name", how="left")
    d["provcode"] = d["ab"].str[:1].map(ABBR)
    # multi-char abbrevs (兵团 etc.) stay unmapped -> excluded
    d.loc[d["ab"].str.len() > 1, "provcode"] = None
    MUNI = {"11", "12", "31", "50"}  # direct-administered municipalities: prefecture = province

    def parse_pref(r):
        if pd.isna(r["provcode"]) or not isinstance(r["code"], str):
            return None
        if r["provcode"] in MUNI:
            return r["provcode"] + "0000"
        c = r["code"]
        if len(c) == 4:
            return r["provcode"] + c[:2] + "00"
        if len(c) == 2:
            return r["provcode"] + c + "00"
        if len(c) == 0:
            return "HIGHCOURT"
        return None
    d["pref_parsed"] = d.apply(parse_pref, axis=1)
    both = d.dropna(subset=["pref_parsed", "prefecture_code"])
    both = both[both["pref_parsed"] != "HIGHCOURT"]
    agree = (both["pref_parsed"] == both["prefecture_code"]).mean()
    rows.append(dict(month=f"2021-{m}", n=len(d),
                     sh_parseable=d["pref_parsed"].notna().mean(),
                     sh_highcourt=(d["pref_parsed"] == "HIGHCOURT").mean(),
                     sh_xwalk=d["prefecture_code"].notna().mean(),
                     n_both=len(both), agreement=agree))
    print(rows[-1])
    dis = both[both["pref_parsed"] != both["prefecture_code"]]
    print("  top disagreements (parsed vs xwalk):")
    print(dis.groupby(["pref_parsed", "prefecture_code"]).size()
          .sort_values(ascending=False).head(8).to_string())
pd.DataFrame(rows).to_csv(
    str(_REP_PROJECT / 'output' / 'ext2124' / 'geocoder_check.csv').replace('\\', '/'), index=False)

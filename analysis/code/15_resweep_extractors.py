# -*- coding: utf-8 -*-
"""6B step 15 — v2 extractors re-sweep, motivated by gold-standard audit:
  telecom-fraud P=0.35 -> modus-based patterns
  offense-year agree=0.69 -> earliest plausible month-anchored year
  origination-year agree=0.75 -> multi-pattern, loan-context anchored, min
  monthly-rate agree=0.83 -> added patterns (月利息/年息/按月利率/分利)
Outputs <restricted-source-path>, x2_civ_YYYY_MM.parquet (case_no keyed).
Restartable; logs to resweep_log.txt.
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
import duckdb, glob, os, re, time

BASE = str(_REP_JUDGMENTS)
OUT = str(_REP_PROJECT / "data" / "derived" / "extract_2014_2020")
LOG = os.path.join(OUT, "resweep_log.txt")
YEAR_DIRS = {2014: "2014_Court_Judgments_CSV", 2015: "2015_Court_Judgments_CSV",
             2016: "2016_Court_Judgments_CSV", 2017: "2017_Court_Judgments_CSV",
             2018: "2018_Court_Judgments_CSV", 2019: "2019_Court_Judgments_CSV",
             2020: "2020_Court_Judgments_CSV_Extracted_Partial"}
CRIME_RX = ("赌博|开设赌场|组织卖淫|非法经营|走私普通货物|危险驾驶|交通肇事|过失致人死亡"
            "|盗窃|故意伤害|黑社会性质组织|敲诈勒索|诈骗|非法拘禁|寻衅滋事|聚众斗殴|强迫交易")
CRIMES = ("('赌博','开设赌场','组织卖淫','非法经营','走私普通货物、物品','危险驾驶','交通肇事',"
          "'过失致人死亡','盗窃','故意伤害','组织、领导、参加黑社会性质组织','敲诈勒索','诈骗',"
          "'非法拘禁','寻衅滋事','聚众斗殴','强迫交易')")

# earliest plausible month-anchored year in fact text
YEARLIST = "list_transform(regexp_extract_all(全文, '(20[01][0-9])年[0-9]{1,2}月', 1), x -> CAST(x AS INT))"
CRIM_SQL = f"""
SELECT 案号 AS case_no,
  list_min(list_filter({YEARLIST}, y -> y >= 2005)) AS offense_year_v2,
  len({YEARLIST}) AS n_month_years,
  CASE WHEN 案由='诈骗' THEN (
    regexp_matches(全文, '电信(?:网络)?诈骗|网络诈骗|冒充(?:客服|公检法|警察|领导|老板)'
      || '|刷单|裸聊|杀猪盘|网络(?:投资|贷款|交友|赌博)平台|虚假(?:网站|链接|App|购物网站)'
      || '|通过(?:网络|微信|QQ|陌陌|探探)(?:结识|认识)[^。]{{0,30}}(?:被害人|骗)'
      || '|群发(?:短信|信息)|改号软件|木马(?:链接|程序)')
  )::INT ELSE NULL END AS fraud_telecom_v2
FROM read_csv('__FILE__', auto_detect=true, sample_size=2000, ignore_errors=true)
WHERE 案件类型 = '刑事案件'
  AND (案由 IN {CRIMES} OR (案由 IS NULL AND regexp_matches(案件名称, '{CRIME_RX}')))
"""

ORIGLIST = ("list_transform(regexp_extract_all(全文, "
            "'(20[01][0-9])年[0-9]{1,2}月[^。]{0,40}?(?:借款|出借|借给|贷款|签订|出具)', 1), "
            "x -> CAST(x AS INT))")
ORIGLIST2 = ("list_transform(regexp_extract_all(全文, "
             "'(?:借款|出借|借给)[^。]{0,25}?(20[01][0-9])年', 1), x -> CAST(x AS INT))")
CIV_SQL = f"""
SELECT 案号 AS case_no,
  list_min(list_filter(list_concat({ORIGLIST}, {ORIGLIST2}), y -> y >= 2000)) AS orig_year_v2,
  COALESCE(
    TRY_CAST(regexp_extract(全文, '月利[率息][为按约]?([0-9]+(?:[.．][0-9]+)?)[%％]', 1) AS DOUBLE),
    TRY_CAST(regexp_extract(全文, '按?月利率([0-9]+(?:[.．][0-9]+)?)[%％]?', 1) AS DOUBLE),
    TRY_CAST(regexp_extract(全文, '月[息利][为按约]?([0-9]+(?:[.．][0-9]+)?)分', 1) AS DOUBLE),
    TRY_CAST(regexp_extract(全文, '([0-9]+(?:[.．][0-9]+)?)分利', 1) AS DOUBLE),
    TRY_CAST(regexp_extract(全文, '年利[率息][为按约]?([0-9]+(?:[.．][0-9]+)?)[%％]', 1) AS DOUBLE)/12.0,
    TRY_CAST(regexp_extract(全文, '年息([0-9]+(?:[.．][0-9]+)?)分', 1) AS DOUBLE)*10.0/12.0
  ) AS monthly_rate_v2
FROM read_csv('__FILE__', auto_detect=true, sample_size=2000, ignore_errors=true)
WHERE 案件类型 = '民事案件' AND 案由 = '民间借贷纠纷'
  AND (案件名称 LIKE '%判决%' OR 案件名称 LIKE '%调解%')
"""

def log(msg):
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(f"{time.strftime('%H:%M:%S')} {msg}\n")

con = duckdb.connect()
con.sql("SET threads TO 10; SET preserve_insertion_order=false; SET memory_limit='24GB'")
files = []
for yr, yd in sorted(YEAR_DIRS.items()):
    for f in sorted(glob.glob(os.path.join(BASE, yd, "**", "*.csv"), recursive=True)):
        m = re.search(r"(20[12][0-9])年([01][0-9])月", os.path.basename(f))
        if m: files.append((m.group(1), m.group(2), f))
log(f"START resweep over {len(files)} files")
for yr, mo, f in files:
    fu = f.replace("\\", "/").replace("'", "''")
    for kind, sql in [("x2_crim", CRIM_SQL), ("x2_civ", CIV_SQL)]:
        dest = os.path.join(OUT, f"{kind}_{yr}_{mo}.parquet").replace("\\", "/")
        if os.path.exists(dest): continue
        t0 = time.time()
        try:
            con.sql(f"COPY ({sql.replace('__FILE__', fu)}) TO '{dest}' (FORMAT PARQUET)")
            n = con.sql(f"SELECT COUNT(*) FROM '{dest}'").fetchone()[0]
            log(f"{kind}_{yr}_{mo}: {n:,} rows in {time.time()-t0:.0f}s")
        except Exception as e:
            log(f"{kind}_{yr}_{mo}: FAILED {type(e).__name__}: {str(e)[:150]}")
log("DONE")
print("resweep complete")

# -*- coding: utf-8 -*-
"""6B: one streaming pass over the raw monthly judgment CSVs (2014-2020).

Per month, two extracts written to <restricted-source-path>
  civil_YYYY_MM.parquet — target civil causes + traffic placebo, with regex-coded
    evidence/relational/failure/collection flags, interest & amount & origination
    strings (parsed downstream)
  crim_YYYY_MM.parquet  — 17 offense names, five co-occurrence dictionaries
    (proximity windows), telecom-fraud split, offense-year candidates
Restartable: skips months whose outputs already exist. Logs to sweep_log.txt.
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
import duckdb, glob, os, re, sys, time, traceback

BASE = str(_REP_JUDGMENTS)
OUT = str(_REP_PROJECT / "data" / "derived" / "extract_2014_2020")
LOG = os.path.join(OUT, "sweep_log.txt")
os.makedirs(OUT, exist_ok=True)

YEAR_DIRS = {
    2014: "2014_Court_Judgments_CSV", 2015: "2015_Court_Judgments_CSV",
    2016: "2016_Court_Judgments_CSV", 2017: "2017_Court_Judgments_CSV",
    2018: "2018_Court_Judgments_CSV", 2019: "2019_Court_Judgments_CSV",
    2020: "2020_Court_Judgments_CSV_Extracted_Partial",
}
CIVIL_CAUSES = "('民间借贷纠纷','买卖合同纠纷','保证合同纠纷','追偿权纠纷','合伙协议纠纷','租赁合同纠纷','机动车交通事故责任纠纷')"
# raw-dump criminal 案由 has no 罪 suffix; ~20% null 案由 backfilled from 案件名称
CRIMES = ("('赌博','开设赌场','组织卖淫','非法经营','走私普通货物、物品',"
          "'危险驾驶','交通肇事','过失致人死亡','盗窃','故意伤害',"
          "'组织、领导、参加黑社会性质组织','敲诈勒索','诈骗',"
          "'非法拘禁','寻衅滋事','聚众斗殴','强迫交易')")
CRIME_RX = ("赌博|开设赌场|组织卖淫|非法经营|走私普通货物|危险驾驶|交通肇事|过失致人死亡"
            "|盗窃|故意伤害|黑社会性质组织|敲诈勒索|诈骗|非法拘禁|寻衅滋事|聚众斗殴|强迫交易")

# ---- regex fragments (RE2). proximity co-occurrence via bounded gaps ----
REL = "(?:朋友|亲戚|亲友|同乡|老乡|同事|同学|熟人|介绍人?|经人介绍|合作多年|长期合作)"
TXN = "(?:借款|出借|欠款|货款|合伙|担保|保证|赊|贷)"
FAIL = "(?:拖欠|欠款未还|未偿还|拒不(?:归还|支付|偿还)|失信|违约|无力偿还|赖账)"
EVID = "(?:借条|欠条|借据|合同|协议书?|担保书|保证人|转账|汇款|银行流水|微信(?:聊天)?记录|支付宝)"
BACKSTOP = "(?:威胁|恐吓|殴打|拘禁|滋扰|骚扰|软暴力|喷漆|堵门|跟踪|保护伞|打招呼|通风报信|徇私)"
COLLECT = "(?:催收|讨债|索要|讨要|追讨|上门)"
JUDIC = "(?:诉至|起诉|报警|报案|申请(?:强制)?执行|财产保全|判令)"
TELECOM = "(?:电信|网络|电话|短信|微信|QQ|刷单|冒充(?:客服|公检法|领导)|信息网络|网上|平台)"

def prox(a, b, gap=60):
    return f"(?:{a}.{{0,{gap}}}{b}|{b}.{{0,{gap}}}{a})"

def flag(pat):  # SQL boolean int from regex on 全文
    p = pat.replace("'", "''")
    return f"regexp_matches(全文, '{p}')::INT"

CIVIL_SQL = f"""
SELECT 案号 AS case_no, 法院 AS court, 所属地区 AS region, 案由 AS cause,
  审理程序 AS proceeding, 裁判日期 AS judgment_date, 公开日期 AS publish_date,
  CASE WHEN 案件名称 LIKE '%调解%' THEN 'mediation'
       WHEN 案件名称 LIKE '%判决%' THEN 'judgment' ELSE 'other' END AS doc_type,
  length(全文) AS doc_len,
  {flag(prox(REL, TXN))}   AS rel_txn,
  {flag(FAIL)}             AS credit_fail,
  {flag(prox(REL, FAIL))}  AS rel_fail,
  {flag(EVID)}             AS evid_any,
  {flag('借条|欠条|借据')}   AS evid_iou,
  {flag('担保|保证人')}      AS evid_guarantee,
  {flag('转账|汇款|银行流水')} AS evid_transfer,
  {flag('微信|聊天记录|支付宝')} AS evid_chat,
  {flag(prox(COLLECT, BACKSTOP, 80))} AS backstop_collection,
  {flag(BACKSTOP)}         AS backstop_any,
  {flag(JUDIC)}            AS judic_any,
  regexp_extract(全文, '月利率[为按约]?([0-9]+(?:[.．][0-9]+)?)[%％]', 1) AS rate_月pct,
  regexp_extract(全文, '月[息利][为按约]?([0-9]+(?:[.．][0-9]+)?)分', 1) AS rate_月分,
  regexp_extract(全文, '年利率[为按约]?([0-9]+(?:[.．][0-9]+)?)[%％]', 1) AS rate_年pct,
  regexp_extract(全文, '(?:借款|出借|贷款)[^。]{{0,30}}?([0-9][0-9,，]*(?:[.．][0-9]+)?)万?元', 1) AS amt_str,
  regexp_matches(全文, '(?:借款|出借|贷款)[^。]{{0,30}}?[0-9][0-9,，]*(?:[.．][0-9]+)?万元')::INT AS amt_is_wan,
  regexp_extract(全文, '(20[01][0-9])年[0-9]{{1,2}}月[0-9]{{1,2}}日[^。]{{0,25}}?(?:向|借款|出借|签订)', 1) AS orig_year,
  当事人 AS parties
FROM read_csv('__FILE__', auto_detect=true, sample_size=2000, ignore_errors=true)
WHERE 案件类型 = '民事案件' AND 案由 IN {CIVIL_CAUSES}
  AND (案件名称 LIKE '%判决%' OR 案件名称 LIKE '%调解%')
"""

CRIM_SQL = f"""
SELECT 案号 AS case_no, 法院 AS court, 所属地区 AS region,
  COALESCE(案由, regexp_extract(案件名称, '{CRIME_RX}', 0)) AS crime,
  审理程序 AS proceeding, 裁判日期 AS judgment_date, length(全文) AS doc_len,
  {flag(prox(REL, TXN))}    AS d_rel_txn,
  {flag(FAIL)}              AS d_credit_fail,
  {flag(prox(REL, FAIL))}   AS d_rel_fail,
  {flag(EVID)}              AS d_formalization,
  {flag(BACKSTOP)}          AS d_backstop,
  {flag(prox(COLLECT, BACKSTOP, 80))} AS d_backstop_collection,
  {flag(JUDIC)}             AS d_judic,
  CASE WHEN 案由='诈骗' THEN {flag(TELECOM)} ELSE NULL END AS fraud_telecom,
  CASE WHEN 案由='诈骗' THEN {flag(prox(REL, '(?:骗|诈骗)', 40))} ELSE NULL END AS fraud_acquaintance,
  {flag(prox('(?:非法拘禁|拘禁|扣押)', '(?:债|欠款|借款|讨要|索要|催收)', 60))} AS detention_debt,
  regexp_extract(全文, '(20[01][0-9])年[0-9]{{1,2}}月[^。]{{0,12}}?(?:期?间|作案|案发|开始|至)', 1) AS offense_year_1,
  regexp_extract(全文, '自(20[01][0-9])年', 1) AS offense_year_2,
  len(regexp_extract_all(全文, '20[01][0-9]年')) AS n_year_mentions
FROM read_csv('__FILE__', auto_detect=true, sample_size=2000, ignore_errors=true)
WHERE 案件类型 = '刑事案件'
  AND (案由 IN {CRIMES} OR (案由 IS NULL AND regexp_matches(案件名称, '{CRIME_RX}')))
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
log(f"START sweep over {len(files)} monthly files")
for yr, mo, f in files:
    tag = f"{yr}_{mo}"
    fu = f.replace("\\", "/").replace("'", "''")
    for kind, sql in [("civil", CIVIL_SQL), ("crim", CRIM_SQL)]:
        dest = os.path.join(OUT, f"{kind}_{tag}.parquet").replace("\\", "/")
        if os.path.exists(dest):
            continue
        t0 = time.time()
        try:
            con.sql(f"COPY ({sql.replace('__FILE__', fu)}) TO '{dest}' (FORMAT PARQUET)")
            n = con.sql(f"SELECT COUNT(*) FROM '{dest}'").fetchone()[0]
            log(f"{kind}_{tag}: {n:,} rows in {time.time()-t0:.0f}s")
        except Exception as e:
            log(f"{kind}_{tag}: FAILED {type(e).__name__}: {str(e)[:200]}")
log("DONE")
print("sweep complete")

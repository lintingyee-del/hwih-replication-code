# -*- coding: utf-8 -*-
"""6B step 48 — extraction sweep over the 2021-2024 raw files, both layers,
using the SAME dictionary fragments as 06_sweep so the measure series is
internally consistent from 2014 through 2024.

Sources/schemas as in step 45: macro 15-col (2021-01..10), ws 8-col
(2021-11..2023-12), s41 (2024-01..10, ws-like schema).
Per month, writes to <restricted-source-path>
  civ_YYYY_MM.parquet    target civil causes + traffic placebo, dictionary flags,
                         rate/amount/orig-year/filing strings, 案号 court code
  crim_YYYY_MM.parquet   17 offenses, five dictionaries, 案号 court code
  audit_civ_YYYY_MM.parquet / audit_crim_YYYY_MM.parquet
                         random full-text samples (25/15 rows) for the
                         post-2020 gold-standard re-audit pool
Restartable: months with existing civ_ output are skipped. Log: sweep_log.txt.
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
import duckdb, os, sys, time, zipfile, shutil, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
BASE = str(_REP_JUDGMENTS)
MZ = BASE + "/2021_2024_Court_Judgments_Monthly_Zips"
TMP = str(_REP_RESTRICTED / "work" / "scan_tmp").replace('\\', '/')
OUT = str(_REP_PROJECT / "data" / "derived" / "extract_2021_2024")
LOG = os.path.join(OUT, "sweep_log.txt")
os.makedirs(TMP, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

# ---- dictionary fragments: identical to 06_sweep ----
REL = "(?:朋友|亲戚|亲友|同乡|老乡|同事|同学|熟人|介绍人?|经人介绍|合作多年|长期合作)"
TXN = "(?:借款|出借|欠款|货款|合伙|担保|保证|赊|贷)"
FAIL = "(?:拖欠|欠款未还|未偿还|拒不(?:归还|支付|偿还)|失信|违约|无力偿还|赖账)"
EVID = "(?:借条|欠条|借据|合同|协议书?|担保书|保证人|转账|汇款|银行流水|微信(?:聊天)?记录|支付宝)"
BACKSTOP = "(?:威胁|恐吓|殴打|拘禁|滋扰|骚扰|软暴力|喷漆|堵门|跟踪|保护伞|打招呼|通风报信|徇私)"
COLLECT = "(?:催收|讨债|索要|讨要|追讨|上门)"
JUDIC = "(?:诉至|起诉|报警|报案|申请(?:强制)?执行|财产保全|判令)"
TELECOM = "(?:电信|网络|电话|短信|微信|QQ|刷单|冒充(?:客服|公检法|领导)|信息网络|网上|平台)"
CAUSES = ["民间借贷纠纷", "买卖合同纠纷", "保证合同纠纷", "追偿权纠纷",
          "合伙协议纠纷", "租赁合同纠纷", "机动车交通事故责任纠纷"]
CRIME_RX = ("赌博|开设赌场|组织卖淫|非法经营|走私普通货物|危险驾驶|交通肇事|过失致人死亡"
            "|盗窃|故意伤害|黑社会性质组织|敲诈勒索|诈骗|非法拘禁|寻衅滋事|聚众斗殴|强迫交易")
ANO = "[（(]\\s*20[0-9]{2}\\s*[）)]\\s*([\\p{Han}]{1,3}[0-9]{0,4})"
FILING = "(?:立案|受理)[^。；;]{0,18}?20[0-2][0-9]\\s*年\\s*[0-9]{1,2}\\s*月\\s*[0-9]{1,2}\\s*日"
ORIG = "(20[0-2][0-9])年[0-9]{1,2}月[0-9]{1,2}日[^。]{0,25}?(?:向|借款|出借|签订)"


def prox(a, b, gap=60):
    return f"(?:{a}.{{0,{gap}}}{b}|{b}.{{0,{gap}}}{a})"


def flag(pat):
    p = pat.replace("'", "''")
    return f"regexp_matches(全文, '{p}')::INT"


def rex(pat, grp=1):
    p = pat.replace("'", "''")
    return f"regexp_extract(全文, '{p}', {grp})"


CIV_COMMON = f"""
  CASE WHEN 案件名称 LIKE '%调解%' THEN 'mediation'
       WHEN 案件名称 LIKE '%判决%' THEN 'judgment' ELSE 'other' END AS doc_type,
  length(全文) AS doc_len,
  regexp_extract(案号, '{ANO}', 1) AS ano_code,
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
  {rex("月利率[为按约]?([0-9]+(?:[.．][0-9]+)?)[%％]")} AS rate_月pct,
  {rex("月[息利][为按约]?([0-9]+(?:[.．][0-9]+)?)分")} AS rate_月分,
  {rex("年利率[为按约]?([0-9]+(?:[.．][0-9]+)?)[%％]")} AS rate_年pct,
  {rex("(?:借款|出借|贷款)[^。]{0,30}?([0-9][0-9,，]*(?:[.．][0-9]+)?)万?元")} AS amt_str,
  regexp_matches(全文, '(?:借款|出借|贷款)[^。]{{0,30}}?[0-9][0-9,，]*(?:[.．][0-9]+)?万元')::INT AS amt_is_wan,
  {rex(ORIG)} AS orig_year,
  {rex(FILING, 0)} AS filing_raw
"""

CRIM_COMMON = f"""
  length(全文) AS doc_len,
  regexp_extract(案号, '{ANO}', 1) AS ano_code,
  {flag(prox(REL, TXN))}    AS d_rel_txn,
  {flag(FAIL)}              AS d_credit_fail,
  {flag(prox(REL, FAIL))}   AS d_rel_fail,
  {flag(EVID)}              AS d_formalization,
  {flag(BACKSTOP)}          AS d_backstop,
  {flag(prox(COLLECT, BACKSTOP, 80))} AS d_backstop_collection,
  {flag(JUDIC)}             AS d_judic,
  {flag(prox('(?:非法拘禁|拘禁|扣押)', '(?:债|欠款|借款|讨要|索要|催收)', 60))} AS detention_debt
"""

MACRO_CIV = f"""
SELECT 案号 AS case_no, 法院 AS court, 所属地区 AS region, 案由 AS cause,
  审理程序 AS proceeding, 裁判日期 AS judgment_date, 公开日期 AS publish_date,
  'macro' AS source, {CIV_COMMON}
FROM read_csv('__FILE__', auto_detect=true, sample_size=2000, ignore_errors=true, all_varchar=true)
WHERE 案件类型 = '民事案件' AND 案由 IN ({','.join("'" + c + "'" for c in CAUSES)})
  AND (案件名称 LIKE '%判决%' OR 案件名称 LIKE '%调解%')
"""

WS_CAUSE = " OR ".join(f"案由 LIKE '%{c}%'" for c in CAUSES)
WS_CAUSE_EXTRACT = "regexp_extract(案由, '(" + "|".join(CAUSES) + ")', 1)"
WS_CIV = f"""
SELECT 案号 AS case_no, NULL AS court, 所属地区 AS region,
  {WS_CAUSE_EXTRACT} AS cause,
  审理程序 AS proceeding, 裁判日期 AS judgment_date, NULL AS publish_date,
  'ws' AS source, {CIV_COMMON}
FROM read_csv('__FILE__', auto_detect=true, sample_size=2000, ignore_errors=true, all_varchar=true)
WHERE 案件类型 = '民事案件' AND ({WS_CAUSE})
  AND (案件名称 LIKE '%判决%' OR 案件名称 LIKE '%调解%')
"""

MACRO_CRIM = f"""
SELECT 案号 AS case_no, 法院 AS court, 所属地区 AS region,
  COALESCE(案由, regexp_extract(案件名称, '{CRIME_RX}', 0)) AS crime,
  审理程序 AS proceeding, 裁判日期 AS judgment_date, 'macro' AS source, {CRIM_COMMON},
  CASE WHEN 案由='诈骗' THEN {flag(TELECOM)} ELSE NULL END AS fraud_telecom
FROM read_csv('__FILE__', auto_detect=true, sample_size=2000, ignore_errors=true, all_varchar=true)
WHERE 案件类型 = '刑事案件'
  AND (regexp_matches(COALESCE(案由,''), '{CRIME_RX}') OR (案由 IS NULL AND regexp_matches(案件名称, '{CRIME_RX}')))
"""

WS_CRIM = f"""
SELECT 案号 AS case_no, NULL AS court, 所属地区 AS region,
  COALESCE(regexp_extract(案由, '({CRIME_RX})', 1),
           regexp_extract(案件名称, '{CRIME_RX}', 0)) AS crime,
  审理程序 AS proceeding, 裁判日期 AS judgment_date, 'ws' AS source, {CRIM_COMMON},
  CASE WHEN 案由 LIKE '%诈骗%' THEN {flag(TELECOM)} ELSE NULL END AS fraud_telecom
FROM read_csv('__FILE__', auto_detect=true, sample_size=2000, ignore_errors=true, all_varchar=true)
WHERE 案件类型 = '刑事案件'
  AND (regexp_matches(COALESCE(案由,''), '{CRIME_RX}') OR regexp_matches(案件名称, '{CRIME_RX}'))
"""

AUDIT_CIV = """
SELECT 案号 AS case_no, 案件名称, 案由, 裁判日期, 全文
FROM read_csv('__FILE__', auto_detect=true, sample_size=2000, ignore_errors=true, all_varchar=true)
WHERE 案件类型='民事案件' AND 案由 __CAUSEMATCH__ AND 案件名称 LIKE '%判决%'
USING SAMPLE 25 ROWS
"""
AUDIT_CRIM = f"""
SELECT 案号 AS case_no, 案件名称, 案由, 裁判日期, 全文
FROM read_csv('__FILE__', auto_detect=true, sample_size=2000, ignore_errors=true, all_varchar=true)
WHERE 案件类型='刑事案件' AND regexp_matches(COALESCE(案由,案件名称,''), '{CRIME_RX}')
USING SAMPLE 15 ROWS
"""


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def fixname(n):
    try:
        return n.encode("cp437").decode("gbk")
    except Exception:
        return n


def month_plan():
    plan = []
    for m in range(1, 11):
        p = (f"{BASE}/2021_Court_Judgments_CSV_Extracted_Jan_Oct/"
             f"2021_MacroData_Court_Judgments_CSV/2021年{m:02d}月裁判文书数据.csv")
        plan.append((f"2021_{m:02d}", "macro", ("path", p)))
    for m in (11, 12):
        plan.append((f"2021_{m}", "ws", ("nested", f"{BASE}/2021.zip", f"ws_2021_{m}.zip")))
    for y in (2022, 2023):
        for m in range(1, 13):
            plan.append((f"{y}_{m:02d}", "ws",
                         ("zip", f"{MZ}/{y}_Court_Judgments_Monthly_Zips/ws_{y}_{m:02d}.zip")))
    for m in range(1, 11):
        outer = f"{MZ}/2024-1.zip" if m <= 4 else f"{MZ}/2024-2.zip"
        plan.append((f"2024_{m:02d}", "ws", ("nested", outer, f"s41_2024{m:02d}.zip")))
    return plan


def biggest_csv(zf):
    cands = [i for i in zf.infolist()
             if not i.is_dir() and i.filename.lower().endswith(".csv")
             and "__MACOSX" not in i.filename]
    return max(cands, key=lambda i: i.file_size)


def materialize(getter):
    kind = getter[0]
    if kind == "path":
        return getter[1], []
    if kind == "zip":
        with zipfile.ZipFile(getter[1]) as zf:
            info = biggest_csv(zf)
            dest = os.path.join(TMP, "sweep_cur.csv")
            with zf.open(info) as src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out, 64 * 1024 * 1024)
        return dest, [dest]
    outer_path, inner_name = getter[1], getter[2]
    inner_tmp = os.path.join(TMP, "sweep_inner.zip")
    with zipfile.ZipFile(outer_path) as zo:
        match = [i for i in zo.infolist()
                 if fixname(i.filename).endswith(inner_name) and "__MACOSX" not in i.filename]
        if not match:
            raise FileNotFoundError(f"{inner_name} not in {outer_path}")
        with zo.open(match[0]) as src, open(inner_tmp, "wb") as out:
            shutil.copyfileobj(src, out, 64 * 1024 * 1024)
    with zipfile.ZipFile(inner_tmp) as zi:
        info = biggest_csv(zi)
        dest = os.path.join(TMP, "sweep_cur.csv")
        with zi.open(info) as src, open(dest, "wb") as out:
            shutil.copyfileobj(src, out, 64 * 1024 * 1024)
    return dest, [dest, inner_tmp]


def main():
    con = duckdb.connect()
    con.sql("SET threads TO 10; SET preserve_insertion_order=false; SET memory_limit='20GB'")
    plan = month_plan()
    log(f"START sweep 2124: {len(plan)} months")
    for tag, schema, getter in plan:
        if os.path.exists(os.path.join(OUT, f"civ_{tag}.parquet")):
            continue
        t0 = time.time()
        cleanup = []
        try:
            csv_path, cleanup = materialize(getter)
            fu = csv_path.replace("\\", "/").replace("'", "''")
            civ = (MACRO_CIV if schema == "macro" else WS_CIV).replace("__FILE__", fu)
            crim = (MACRO_CRIM if schema == "macro" else WS_CRIM).replace("__FILE__", fu)
            cm = ("IN (" + ",".join("'" + c + "'" for c in CAUSES) + ")"
                  if schema == "macro" else "LIKE '%民间借贷纠纷%'")
            aud_c = AUDIT_CIV.replace("__FILE__", fu).replace("__CAUSEMATCH__", cm)
            aud_k = AUDIT_CRIM.replace("__FILE__", fu)
            for name, sql in [("crim", crim), ("audit_civ", aud_c),
                              ("audit_crim", aud_k), ("civ", civ)]:
                dest = os.path.join(OUT, f"{name}_{tag}.parquet").replace("\\", "/")
                con.sql(f"COPY ({sql}) TO '{dest}' (FORMAT PARQUET)")
            nc = con.sql(f"SELECT COUNT(*) FROM '{OUT}/civ_{tag}.parquet'").fetchone()[0]
            nk = con.sql(f"SELECT COUNT(*) FROM '{OUT}/crim_{tag}.parquet'").fetchone()[0]
            log(f"{tag} [{schema}] civ={nc:,} crim={nk:,} in {time.time()-t0:.0f}s")
        except Exception as e:
            log(f"{tag} FAILED {type(e).__name__}: {str(e)[:250]}")
        finally:
            for p in cleanup:
                try:
                    os.remove(p)
                except OSError:
                    pass
    log("DONE")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""6B step 45 — availability scan of the 2021-2024 raw judgment files (civil focus).

Sources (three provider batches, two schemas):
  2021-01..10  macrodatas 15-col CSVs, already extracted
  2021-11..12  ws_ 8-col CSVs, nested zips inside 2021.zip
  2022-01..2023-12  ws_ monthly zips
  2024-01..10  s41_ monthly zips nested inside 2024-1.zip / 2024-2.zip

Per month, writes tidy rows to output/ext2124/:
  casetype.csv   ym, case_type, n
  causes.csv     ym, cause, n_all, n_judg, n_med
  mjd.csv        ym + quality stats on 民间借贷判决 (doclen, orig-year hit,
                 rel_txn rate, prefecture-code and province coverage, date sanity)
  origyr.csv     ym, contract origination year, n  (民间借贷判决 only)
Restartable: months already present in casetype.csv are skipped.
Usage: python 45_scan_2021_2024.py [test]   (test = 2022-01 only)
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
import duckdb, os, re, sys, time, zipfile, shutil

BASE = str(_REP_JUDGMENTS)
MZ = BASE + "/2021_2024_Court_Judgments_Monthly_Zips"
TMP = str(_REP_RESTRICTED / "work" / "scan_tmp").replace('\\', '/')
OUT = str(_REP_PROJECT / "output" / "ext2124")
LOG = os.path.join(OUT, "scan_log.txt")
os.makedirs(TMP, exist_ok=True)
os.makedirs(OUT, exist_ok=True)
TEST = len(sys.argv) > 1 and sys.argv[1] == "test"

CAUSES = ["民间借贷纠纷", "买卖合同纠纷", "保证合同纠纷", "追偿权纠纷",
          "合伙协议纠纷", "租赁合同纠纷", "机动车交通事故责任纠纷"]
REL = "(?:朋友|亲戚|亲友|同乡|老乡|同事|同学|熟人|介绍人?|经人介绍|合作多年|长期合作)"
TXN = "(?:借款|出借|欠款|货款|合伙|担保|保证|赊|贷)"
RELTXN = f"(?:{REL}.{{0,60}}{TXN}|{TXN}.{{0,60}}{REL})"
ORIG = "(20[0-2][0-9])年[0-9]{1,2}月[0-9]{1,2}日[^。]{0,25}?(?:向|借款|出借|签订)"
PREF = "[（(]\\s*20[0-9]{2}\\s*[）)]\\s*([^0-9]{1,4}[0-9]{2})"


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
    plan = []  # (ym, schema, getter) — getter returns a CSV path on disk
    for m in range(1, 11):
        p = (f"{BASE}/2021_Court_Judgments_CSV_Extracted_Jan_Oct/"
             f"2021_MacroData_Court_Judgments_CSV/2021年{m:02d}月裁判文书数据.csv")
        plan.append((f"2021-{m:02d}", "macro", ("path", p)))
    for m in (11, 12):
        plan.append((f"2021-{m:02d}", "ws",
                     ("nested", f"{BASE}/2021.zip", f"ws_2021_{m}.zip")))
    for y in (2022, 2023):
        for m in range(1, 13):
            plan.append((f"{y}-{m:02d}", "ws",
                         ("zip", f"{MZ}/{y}_Court_Judgments_Monthly_Zips/ws_{y}_{m:02d}.zip")))
    for m in range(1, 11):
        outer = f"{MZ}/2024-1.zip" if m <= 4 else f"{MZ}/2024-2.zip"
        plan.append((f"2024-{m:02d}", "ws",
                     ("nested", outer, f"s41_2024{m:02d}.zip")))
    if TEST:
        plan = [p for p in plan if p[0] == "2022-01"]
    return plan


def biggest_csv(zf):
    cands = [i for i in zf.infolist()
             if not i.is_dir() and i.filename.lower().endswith(".csv")
             and "__MACOSX" not in i.filename]
    return max(cands, key=lambda i: i.file_size)


def materialize(getter):
    """Return (csv_path, cleanup_list)."""
    kind = getter[0]
    if kind == "path":
        return getter[1], []
    if kind == "zip":
        with zipfile.ZipFile(getter[1]) as zf:
            info = biggest_csv(zf)
            dest = os.path.join(TMP, "cur.csv")
            with zf.open(info) as src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out, 64 * 1024 * 1024)
        return dest, [dest]
    if kind == "nested":  # outer zip -> inner zip (match by suffix) -> csv
        outer_path, inner_name = getter[1], getter[2]
        inner_tmp = os.path.join(TMP, "inner.zip")
        with zipfile.ZipFile(outer_path) as zo:
            match = [i for i in zo.infolist()
                     if fixname(i.filename).endswith(inner_name)
                     and "__MACOSX" not in i.filename]
            if not match:
                raise FileNotFoundError(f"{inner_name} not in {outer_path}")
            with zo.open(match[0]) as src, open(inner_tmp, "wb") as out:
                shutil.copyfileobj(src, out, 64 * 1024 * 1024)
        with zipfile.ZipFile(inner_tmp) as zi:
            info = biggest_csv(zi)
            dest = os.path.join(TMP, "cur.csv")
            with zi.open(info) as src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out, 64 * 1024 * 1024)
        return dest, [dest, inner_tmp]


def q(pat):
    return pat.replace("'", "''")


def scan_month(con, ym, schema, csv_path):
    fu = csv_path.replace("\\", "/")
    rd = (f"read_csv('{fu}', auto_detect=true, sample_size=2000, "
          f"ignore_errors=true, all_varchar=true)")
    cause_match = ((lambda c: f"案由 = '{c}'") if schema == "macro"
                   else (lambda c: f"案由 LIKE '%{c}%'"))
    con.sql(f"CREATE OR REPLACE TEMP VIEW raw AS SELECT * FROM {rd}")

    ct = con.sql("SELECT COALESCE(案件类型,'NA') t, COUNT(*) n FROM raw GROUP BY 1").fetchall()

    aggs = []
    for c in CAUSES:
        cm = cause_match(c)
        aggs += [
            f"SUM((案件类型='民事案件' AND {cm})::INT) AS \"{c}_all\"",
            f"SUM((案件类型='民事案件' AND {cm} AND 案件名称 LIKE '%判决%')::INT) AS \"{c}_judg\"",
            f"SUM((案件类型='民事案件' AND {cm} AND 案件名称 LIKE '%调解%')::INT) AS \"{c}_med\"",
        ]
    cause_row = con.sql("SELECT " + ", ".join(aggs) + " FROM raw").fetchone()

    mjd_where = (f"案件类型='民事案件' AND {cause_match('民间借贷纠纷')} "
                 f"AND 案件名称 LIKE '%判决%'")
    mjd = con.sql(f"""
      SELECT COUNT(*) n,
        median(length(全文)) med_len,
        AVG((length(全文) >= 400)::INT) sh_len400,
        AVG(regexp_matches(全文, '{q(ORIG)}')::INT) orig_hit,
        AVG(regexp_matches(全文, '{q(RELTXN)}')::INT) rel_txn,
        COUNT(DISTINCT regexp_extract(案号, '{q(PREF)}', 1)) n_prefcode,
        COUNT(DISTINCT 所属地区) n_region,
        AVG((substr(裁判日期,1,7) = '{ym}')::INT) sh_date_in_month,
        min(裁判日期) dmin, max(裁判日期) dmax
      FROM raw WHERE {mjd_where}""").fetchone()

    oy = con.sql(f"""
      SELECT regexp_extract(全文, '{q(ORIG)}', 1) y, COUNT(*) n
      FROM raw WHERE {mjd_where} GROUP BY 1 HAVING y <> '' ORDER BY 1""").fetchall()

    with open(os.path.join(OUT, "casetype.csv"), "a", encoding="utf-8") as fh:
        for t, n in ct:
            fh.write(f"{ym},{t},{n}\n")
    with open(os.path.join(OUT, "causes.csv"), "a", encoding="utf-8") as fh:
        for i, c in enumerate(CAUSES):
            a, j, m = cause_row[3 * i], cause_row[3 * i + 1], cause_row[3 * i + 2]
            fh.write(f"{ym},{c},{a or 0},{j or 0},{m or 0}\n")
    with open(os.path.join(OUT, "mjd.csv"), "a", encoding="utf-8") as fh:
        n, ml, s4, oh, rt, np_, nr, sdm, dmin, dmax = mjd
        fh.write(f"{ym},{n},{ml},{s4},{oh},{rt},{np_},{nr},{sdm},{dmin},{dmax}\n")
    with open(os.path.join(OUT, "origyr.csv"), "a", encoding="utf-8") as fh:
        for y, n in oy:
            fh.write(f"{ym},{y},{n}\n")


def main():
    done = set()
    ctp = os.path.join(OUT, "casetype.csv")
    if os.path.exists(ctp):
        with open(ctp, encoding="utf-8") as fh:
            done = {ln.split(",")[0] for ln in fh if "," in ln}
    con = duckdb.connect()
    con.sql("SET threads TO 10; SET preserve_insertion_order=false; "
            "SET memory_limit='20GB'")
    plan = month_plan()
    log(f"START scan: {len(plan)} months, {len(done)} already done")
    for ym, schema, getter in plan:
        if ym in done:
            continue
        t0 = time.time()
        cleanup = []
        try:
            csv_path, cleanup = materialize(getter)
            sz = os.path.getsize(csv_path) / 1e9
            scan_month(con, ym, schema, csv_path)
            log(f"{ym} [{schema}] {sz:.1f}GB done in {time.time()-t0:.0f}s")
        except Exception as e:
            log(f"{ym} FAILED {type(e).__name__}: {str(e)[:300]}")
        finally:
            for p in cleanup:
                try:
                    os.remove(p)
                except OSError:
                    pass
    log("DONE")


if __name__ == "__main__":
    main()

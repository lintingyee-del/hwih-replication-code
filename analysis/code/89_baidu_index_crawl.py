# -*- coding: utf-8 -*-
"""Baidu Index city-month panel for private debt-collection demand keywords.

Modes
  --test-cookie          validate the cookie file
  --preflight            check which candidate words are indexed and have
                         2014 history at the national level (the make-or-break
                         gate: unindexed words have NO retrievable history)
  --crawl                word-group x city sweep, daily data 2014-2021,
                         checkpointed per city; safe to interrupt and rerun
  --aggregate            daily shards -> city x month panel csv

Cookie: log into https://index.baidu.com in a browser, copy the full Cookie
header of any index.baidu.com request (must contain BDUSS=...), save it as a
single line in analysis/data/baidu_cookie.txt.
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
import argparse, csv, json, os, random, sys, time

sys.stdout.reconfigure(encoding="utf-8")
from qdata.baidu_index import CITY_CODE, get_search_index, test_cookies
from qdata.errors import QdataError

DATA = str(_REP_BAIDU)
RAW = os.path.join(DATA, "raw")
COOKIE_FILE = os.path.join(DATA, "baidu_cookie.txt")

CANDIDATES = ["讨债公司", "讨债", "要账", "要债", "催收公司", "催收",
              "债务催收", "专业讨债", "请人讨债", "收数公司"]
START, END = "2014-01-01", "2021-12-31"

# qdata keeps the four direct-administered municipalities in PROVINCE_CODE
# rather than CITY_CODE.  For Baidu Index geography they are the city-level
# units needed by the prefecture panel, so include their documented area codes
# explicitly in the city sweep.
DIRECT_MUNICIPALITY_CODES = {
    "北京": "911",
    "上海": "910",
    "天津": "923",
    "重庆": "904",
}


def load_cookie():
    if not os.path.exists(COOKIE_FILE):
        sys.exit(f"缺 cookie 文件: {COOKIE_FILE}\n"
                 "浏览器登录 index.baidu.com -> F12 Network -> 任一 index.baidu.com 请求"
                 " -> 复制完整 Cookie 头(含 BDUSS=) -> 存成单行文本。")
    ck = open(COOKIE_FILE, encoding="utf-8").read().strip()
    if "BDUSS" not in ck:
        sys.exit("cookie 里没有 BDUSS 字段,不是登录态 cookie。")
    return ck


def probe_word(word, cookie, start, end):
    """Return (ok, n_days_nonzero, total) for one word at national level."""
    try:
        rows = [r for r in get_search_index(
            keywords_list=[[word]], start_date=start, end_date=end,
            cookies=cookie, area=0) if r["type"] == "all"]
        nz = sum(1 for r in rows if int(r["index"]) > 0)
        tot = sum(int(r["index"]) for r in rows)
        return True, nz, tot
    except QdataError as e:
        return False, str(e), None
    except Exception as e:
        return False, f"{type(e).__name__}: {e}", None


def preflight(cookie):
    print("=== 收录自查(全国级) ===")
    keep = []
    for w in CANDIDATES:
        ok14, nz14, tot14 = probe_word(w, cookie, "2014-01-01", "2014-12-31")
        time.sleep(random.uniform(3, 5))
        ok20, nz20, tot20 = probe_word(w, cookie, "2020-01-01", "2020-12-31")
        time.sleep(random.uniform(3, 5))
        if ok14 and nz14 > 0:
            verdict = "可用(有2014历史)"
            keep.append(w)
        elif ok20 and (nz20 or 0) > 0:
            verdict = "仅近期有数据,无2014历史 -- 放弃"
        else:
            verdict = f"未收录/失败: {nz14 if not ok14 else nz20}"
        print(f"  {w}: 2014非零天数={nz14 if ok14 else '-'} 年总量={tot14 if ok14 else '-'} | "
              f"2020非零天数={nz20 if ok20 else '-'} => {verdict}")
    print("\n通过预检的词:", keep)
    with open(os.path.join(DATA, "preflight_keep.json"), "w", encoding="utf-8") as f:
        json.dump(keep, f, ensure_ascii=False)
    return keep


def fetch_city(cookie, words, code):
    """Fetch one city with chunk-level retry (10 date chunks x [data+key] calls;
    retrying a failed 300-day chunk costs 2 requests, not the whole city)."""
    from qdata.baidu_index import common as bic
    from qdata.baidu_index.baidu_index import format_data
    rows = []
    for sd, ed in bic.get_time_range_list(START, END):
        last = None
        for attempt in range(5):
            try:
                ej = bic.get_encrypt_json(start_date=sd, end_date=ed,
                                          keywords=[[w] for w in words],
                                          type="search", area=int(code), cookies=cookie)
                key = bic.get_key(ej["data"]["uniqid"], cookie)
                for ed_ in ej["data"]["userIndexes"]:
                    for kind in ["all", "pc", "wise"]:
                        ed_[kind]["data"] = bic.decrypt_func(key, ed_[kind]["data"])
                    rows.extend(format_data(ed_))
                last = None
                break
            except Exception as e:
                last = e
                time.sleep(8 + attempt * 7)
        if last is not None:
            raise last
    return rows


def crawl(cookie, words):
    os.makedirs(RAW, exist_ok=True)
    groups = [words[i:i + 5] for i in range(0, len(words), 5)]
    city_codes = {**CITY_CODE, **DIRECT_MUNICIPALITY_CODES}
    cities = sorted(city_codes.items(), key=lambda kv: int(kv[1]))
    done_log = os.path.join(DATA, "crawl_done.txt")

    def load_done():
        return (set(open(done_log, encoding="utf-8").read().split())
                if os.path.exists(done_log) else set())

    for sweep in range(1, 31):
        done = load_done()
        todo = [(gi, grp, city, code) for gi, grp in enumerate(groups)
                for city, code in cities if f"g{gi}_{code}" not in done]
        if not todo:
            print("全部完成"); return
        print(f"== 第{sweep}轮清扫: 剩 {len(todo)} 个市 ==")
        fails = 0
        for gi, grp, city, code in todo:
            try:
                rows = fetch_city(cookie, grp, code)
                fp = os.path.join(RAW, f"city_{code}_g{gi}.csv")
                with open(fp, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["keyword", "type", "date", "index", "city", "citycode"])
                    for r in rows:
                        w.writerow(["|".join(r["keyword"]), r["type"], r["date"],
                                    r["index"], city, code])
                with open(done_log, "a", encoding="utf-8") as f:
                    f.write(f"g{gi}_{code}\n")
                fails = 0
                print(f"ok {city}({code}) g{gi}: {len(rows)} rows")
            except QdataError as e:
                msg = str(e)
                if "频繁" in msg or "REQUEST_LIMITED" in msg:
                    print(f"RATE-LIMIT {city}({code}),暂停30分钟")
                    time.sleep(1800)
                    fails = 0
                    continue
                if "cookie" in msg.lower() or "登录" in msg:
                    sys.exit(f"cookie 失效,退出;重新扫码后重跑续传。({msg})")
                fails += 1
                print(f"FAIL {city}({code}) g{gi}: {e}")
                if fails >= 8:
                    print("连败8次,本轮中止,休息后进下一轮")
                    break
                time.sleep(120)
            except Exception as e:
                fails += 1
                print(f"ERR {city}({code}) g{gi}: {type(e).__name__}: {e}")
                if fails >= 8:
                    print("连败8次,本轮中止,休息后进下一轮")
                    break
                time.sleep(60)
            time.sleep(random.uniform(8, 14))
        done = load_done()
        remaining = [(gi, code) for gi, grp in enumerate(groups)
                     for city, code in cities if f"g{gi}_{code}" not in done]
        if not remaining:
            print("全部完成")
            return
        time.sleep(900)
    print("30轮后仍未全齐,退出;重跑续传")


def aggregate():
    import pandas as pd, glob
    parts = []
    for fp in glob.glob(os.path.join(RAW, "city_*.csv")):
        df = pd.read_csv(fp)
        parts.append(df[df["type"] == "all"])
    d = pd.concat(parts, ignore_index=True)
    d["ym"] = d["date"].str[:7]
    m = (d.groupby(["keyword", "city", "citycode", "ym"])["index"]
           .agg(["mean", "sum", "max"]).reset_index())
    out = os.path.join(DATA, "baidu_index_city_month.csv")
    m.to_csv(out, index=False, encoding="utf-8-sig")
    print("wrote", out, len(m), "rows,", d["city"].nunique(), "cities")


if __name__ == "__main__":
    os.makedirs(DATA, exist_ok=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-cookie", action="store_true")
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--crawl", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--words", nargs="*", default=None)
    a = ap.parse_args()
    if a.aggregate:
        aggregate(); sys.exit()
    ck = load_cookie()
    if a.test_cookie:
        print("cookie 有效" if test_cookies(ck) else "cookie 无效"); sys.exit()
    if a.preflight:
        preflight(ck); sys.exit()
    if a.crawl:
        words = a.words
        if not words:
            kfp = os.path.join(DATA, "preflight_keep.json")
            words = json.load(open(kfp, encoding="utf-8")) if os.path.exists(kfp) else None
        if not words:
            sys.exit("先跑 --preflight 或用 --words 指定词表")
        crawl(ck, words)

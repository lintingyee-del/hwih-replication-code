# -*- coding: utf-8 -*-
"""6B step 71 — mechanism scan over the raw archive (two passes, one run).

Pass 1 (civil, clean window 2017-01..2019-03, relational causes):
  - lawyer/firm pairs from the party header (委托诉讼代理人 ... 律师事务所/法律服务所),
    the identity-bearing field behind the book-dump worst-case bound;
  - citations of CRIMINAL case numbers anywhere in the text (刑初/刑终/刑字第...号),
    the civil side of the criminal-civil cross-citation linkage.
  Output: data/mech_civil_scan.parquet (案号, 案由, jmonth, lawyer_keys, n_lawyer,
          crim_cites, n_crimcite)

Pass 2 (criminal, months mirroring <restricted-source-path> coverage):
  - prosecution date (向本院提起公诉) from the head, fallback court-acceptance date,
    for the 17 analysis offenses -> the earlier clock for the de-militarization
    margins;
  - citations of CIVIL case numbers (民初/民终...号): for analysis-offense cases all
    years, and for ANY criminal case 2018+ whose text mentions 民初 (催收/套路贷/
    虚假诉讼 networks name the suits they ran).
  Outputs: data/mech_crim_prosdate.parquet (案号, 案由, pros_ymd, src)
           data/mech_crim_civcite.parquet (案号, 案由, ym, in_offenses, civ_cites,
           n_civcite)

Usage: python 71_mech_scan.py [test]   (test = one month per pass)
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
import sys, re, glob, time, pandas as pd

BASE = str(_REP_JUDGMENTS)
DATA = str(_REP_PROJECT / "data")
TEST = len(sys.argv) > 1 and sys.argv[1] == "test"

# ---------------- shared regexes ---------------------------------------------
D = r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日"
# lawyer name + firm (律师事务所 or 基层法律服务所), party header only
LAW = re.compile(r"委托(?:诉讼)?代理人[：:，,]?\s*([^，。；、：:\s]{2,12})[，、,]"
                 r"[^。；]{0,40}?([^，。；、\s]{2,30}?(?:律师事务所|法律服务所))")
# criminal case numbers cited inside civil text
CRIMNO = re.compile(r"[（(]\s*(?:19|20)\d{2}\s*[）)]\s*[^（）()。；，、\s]{1,14}刑"
                    r"[^（）()。；，、\s]{0,8}\d{1,6}\s*号")
# civil case numbers cited inside criminal text
CIVNO = re.compile(r"[（(]\s*(?:19|20)\d{2}\s*[）)]\s*[^（）()。；，、\s]{1,14}民"
                   r"[^（）()。；，、\s]{0,8}\d{1,6}\s*号")
# prosecution date: "...于2018年5月10日向本院提起公诉"
PROS = re.compile(rf"{D}[^。]{{0,40}}?向本院提起公诉")
# fallback: acceptance date after the prosecution sentence, or generic 本院立案
ACC = re.compile(rf"(?:提起公诉|移送起诉)[^。]{{0,40}}?本院[^。]{{0,12}}?于?{D}"
                 rf"[^。]{{0,8}}?(?:立案|受理)|本院于{D}立案")


def norm_no(s):
    return s.replace(" ", "").replace("(", "（").replace(")", "）")


def fmt_d(g3):
    try:
        return f"{int(g3[0]):04d}-{int(g3[1]):02d}-{int(g3[2]):02d}"
    except Exception:
        return None


def month_path(y, m):
    return (f"{BASE}/{y}_Court_Judgments_CSV/{y}_MacroData_Court_Judgments_CSV/"
            f"{y}年{m:02d}月裁判文书数据.csv")


# ============================================================================
# Pass 1 — civil relational causes, clean window
# ============================================================================
cc_meta = pd.read_parquet(f"{DATA}/civil_case.parquet",
                          columns=["cause", "cause_family"]).drop_duplicates()
REL_CAUSES = set(cc_meta.loc[cc_meta["cause_family"] == "relational", "cause"])
print(f"[p1] relational causes: {sorted(REL_CAUSES)}", flush=True)

ym1 = ([(2017, m) for m in range(1, 13)] + [(2018, m) for m in range(1, 13)]
       + [(2019, m) for m in range(1, 4)])
if TEST: ym1 = [(2018, 6)]

out1 = []
for y, m in ym1:
    p = month_path(y, m)
    if not glob.glob(p):
        print(f"[p1 skip] {p}", flush=True); continue
    t0 = time.time(); got = 0
    for ch in pd.read_csv(p, chunksize=60000, encoding="utf-8",
                          usecols=["案号", "案由", "全文"], dtype=str,
                          on_bad_lines="skip"):
        d = ch[ch["案由"].isin(REL_CAUSES)].copy()
        if len(d) == 0: continue
        txt = d["全文"].fillna("")
        d["lawyer_keys"] = txt.map(
            lambda t: ";".join(sorted({f"{a}|{b}" for a, b in
                                       LAW.findall(t[:4500])})))
        d["crim_cites"] = txt.map(
            lambda t: ";".join(sorted({norm_no(x) for x in
                                       CRIMNO.findall(t[:80000])})))
        d["n_lawyer"] = d["lawyer_keys"].str.count(r"\|")
        d["n_crimcite"] = d["crim_cites"].map(lambda s: 0 if not s else s.count(";") + 1)
        d["jmonth"] = f"{y:04d}-{m:02d}"
        out1.append(d[["案号", "案由", "jmonth", "lawyer_keys", "n_lawyer",
                       "crim_cites", "n_crimcite"]])
        got += len(d)
    r = out1[-1] if out1 else None
    lw = (r["n_lawyer"] > 0).mean() if r is not None else float("nan")
    print(f"[p1 ok] {y}-{m:02d} rel={got} lawyer_rate~{lw:.3f} "
          f"{time.time()-t0:.0f}s", flush=True)

d1 = pd.concat(out1, ignore_index=True).drop_duplicates("案号")
d1.to_parquet(f"{DATA}/mech_civil_scan.parquet", index=False)
print(f"[p1 done] {len(d1):,} relational cases; lawyer coverage "
      f"{(d1['n_lawyer']>0).mean():.3f}; crim-citation share "
      f"{(d1['n_crimcite']>0).mean():.4f}", flush=True)

# ============================================================================
# Pass 2 — criminal: prosecution dates (analysis offenses) + civil citations
# ============================================================================
offc = pd.concat([pd.read_parquet(f, columns=["crime"])
                  for f in sorted(glob.glob(str(_REP_PROJECT / "data" / "derived" / "extract_2014_2020" / 'crim_2018_*.parquet').replace('\\', '/')))])
OFFENSES = set(offc["crime"].unique())
print(f"[p2] {len(OFFENSES)} analysis offenses", flush=True)

months2 = sorted({f.split("crim_")[1][:7].replace("_", "-")
                  for f in glob.glob(str(_REP_PROJECT / "data" / "derived" / "extract_2014_2020" / 'crim_*.parquet').replace('\\', '/'))})
if TEST: months2 = ["2019-06"]

pros_out, cite_out = [], []
for ym in months2:
    y, m = int(ym[:4]), int(ym[5:7])
    p = month_path(y, m)
    if not glob.glob(p):
        print(f"[p2 skip] {p}", flush=True); continue
    t0 = time.time(); n_off = n_cite = 0
    for ch in pd.read_csv(p, chunksize=60000, encoding="utf-8",
                          usecols=["案号", "案件类型", "案由", "全文"], dtype=str,
                          on_bad_lines="skip"):
        cr = ch[ch["案件类型"] == "刑事案件"]
        if len(cr) == 0: continue
        inoff = cr["案由"].isin(OFFENSES)
        # (a) prosecution dates for analysis offenses
        a = cr[inoff].copy()
        if len(a):
            def pros_date(t):
                if not isinstance(t, str): return None, None
                mm = PROS.search(t[:3500])
                if mm: return fmt_d(mm.groups()[0:3]), "pros"
                mm = ACC.search(t[:3500])
                if mm:
                    g = mm.groups()
                    g3 = g[0:3] if g[0] else g[3:6]
                    return fmt_d(g3), "acc"
                return None, None
            res = a["全文"].map(pros_date)
            a["pros_ymd"] = res.map(lambda x: x[0])
            a["src"] = res.map(lambda x: x[1])
            pros_out.append(a[["案号", "案由", "pros_ymd", "src"]])
            n_off += len(a)
        # (b) civil-case-number citations: offense cases always; other criminal
        #     cases from 2018 on if the text mentions 民初
        b = cr[inoff | ((ym >= "2018-01")
                        & cr["全文"].str.contains("民初", na=False))].copy()
        if len(b):
            b["civ_cites"] = b["全文"].fillna("").map(
                lambda t: ";".join(sorted({norm_no(x) for x in
                                           CIVNO.findall(t[:80000])})))
            b = b[b["civ_cites"] != ""]
            if len(b):
                b["n_civcite"] = b["civ_cites"].map(lambda s: s.count(";") + 1)
                b["ym"] = ym
                b["in_offenses"] = b["案由"].isin(OFFENSES)
                cite_out.append(b[["案号", "案由", "ym", "in_offenses",
                                   "civ_cites", "n_civcite"]])
                n_cite += len(b)
    print(f"[p2 ok] {ym} offenses={n_off} citing={n_cite} "
          f"{time.time()-t0:.0f}s", flush=True)

d2 = pd.concat(pros_out, ignore_index=True).drop_duplicates("案号")
d2.to_parquet(f"{DATA}/mech_crim_prosdate.parquet", index=False)
d3 = (pd.concat(cite_out, ignore_index=True).drop_duplicates("案号")
      if cite_out else pd.DataFrame(columns=["案号", "案由", "ym", "in_offenses",
                                             "civ_cites", "n_civcite"]))
d3.to_parquet(f"{DATA}/mech_crim_civcite.parquet", index=False)
print(f"[p2 done] prosecution dates: {len(d2):,} offense cases, coverage "
      f"{d2['pros_ymd'].notna().mean():.3f} "
      f"(pros {(d2['src']=='pros').mean():.3f}); "
      f"civil-citing criminal cases: {len(d3):,}", flush=True)
print("step 71 complete", flush=True)

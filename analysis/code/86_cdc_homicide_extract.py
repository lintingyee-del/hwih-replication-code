# -*- coding: utf-8 -*-
"""Extract homicide (他杀及后遗症, X85-Y09) death counts/rates from
中国死因监测数据集 volumes (chapter 7: region x sex x age x cause).

Chapter 7 text layer carries no region/sex captions. Blocks are delimited by
the U000 (全死因) row and follow the fixed volume order
  {全国, 东部, 中部, 西部} x {城乡合计, 城市, 农村} x {合计, 男性, 女性}
(region-major, then urban/rural, then sex). Labels are assigned by position
and verified with adding-up identities (male+female=both, urban+rural=all,
E+C+W=national), which hold exactly for integer death counts.
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
import fitz, os, re, sys, json

BASE = str(_REP_MORTALITY)

TARGETS = {"全死因": "all_cause", "伤害": "injury_all", "自杀及后遗症": "suicide",
           "他杀及后遗症": "homicide", "道路交通事故": "traffic_acc"}

# age columns in ch.7 tables: 合计, 0,1,5,10,15,20,...,85 (20 columns)
AGE_COLS = ["total", "a0", "a1", "a5", "a10", "a15", "a20", "a25", "a30", "a35",
            "a40", "a45", "a50", "a55", "a60", "a65", "a70", "a75", "a80", "a85"]

REGIONS = ["全国", "东部", "中部", "西部"]
URBRUR = ["城乡合计", "城市", "农村"]
SEXES = ["合计", "男性", "女性"]

# Block order verified on the 2017 volume via adding-up identities:
# national (all/urban/rural), then E/C/W all, E/C/W urban, E/C/W rural.
BLOCK_ORDER = [("全国", "城乡合计"), ("全国", "城市"), ("全国", "农村"),
               ("东部", "城乡合计"), ("中部", "城乡合计"), ("西部", "城乡合计"),
               ("东部", "城市"), ("中部", "城市"), ("西部", "城市"),
               ("东部", "农村"), ("中部", "农村"), ("西部", "农村")]


def parse_chapter7(doc, p_from, p_to, rates=False):
    """Tokenize ch.7 pages, split into U000-anchored blocks, pull target rows."""
    numpat = re.compile(r"^-?[\d,]+(\.\d+)?$")
    ucode = re.compile(r"^U\d{3}$")
    blocks, cur = [], None
    for i in range(p_from, p_to):
        text = doc[i].get_text()
        # drop page furniture lines
        lines = [l.strip() for l in text.split("\n")]
        toks = []
        for l in lines:
            if re.match(r"^(中国死因监测数据集|续表|第七章|7\.\d|疾病$|编码$|疾病名称$|合计.*岁|.*岁～$|\d+ 岁～)", l):
                continue
            toks.extend(l.split())
        j = 0
        while j < len(toks):
            t = toks[j]
            if ucode.match(t):
                # start of a row: collect name tokens then numbers
                j += 1
                name_parts, nums = [], []
                while j < len(toks) and not numpat.match(toks[j]):
                    if ucode.match(toks[j]):
                        break
                    name_parts.append(toks[j]); j += 1
                while j < len(toks) and numpat.match(toks[j]):
                    nums.append(toks[j]); j += 1
                name = re.sub(r"^[0-9a-zA-Z\.\s]*", "", "".join(name_parts))
                name = re.sub(r"^(Ⅰ|Ⅱ|Ⅲ|Ⅳ)\.?", "", name)
                if name == "全死因":
                    cur = {}
                    blocks.append(cur)
                if cur is not None and name in TARGETS and TARGETS[name] not in cur:
                    if nums:
                        v = nums[0].replace(",", "")
                        cur[TARGETS[name]] = float(v) if rates else int(float(v))
                        if name == "他杀及后遗症" and not rates and len(nums) >= 14:
                            # ages 15-59 = columns 15~,20~,...,55~ (indices 5..13)
                            cur["homicide_15_59"] = sum(
                                int(float(x.replace(",", ""))) for x in nums[5:14])
            else:
                j += 1
    return blocks


def label_blocks(n):
    labs = []
    for reg, ur in BLOCK_ORDER:
        for sx in SEXES:
            labs.append((reg, ur, sx))
    return labs[:n]


def validate(rows):
    """Adding-up identities on homicide counts; returns list of check strings."""
    d = {(r["region"], r["urbrur"], r["sex"]): r.get("homicide") for r in rows}
    checks = []
    def ok(a, b, tag):
        if None in (a, b):
            checks.append(f"SKIP {tag}")
        else:
            checks.append(("PASS" if a == b else f"FAIL({a} vs {b})") + " " + tag)
    for reg in REGIONS:
        for ur in URBRUR:
            ok(d.get((reg, ur, "合计")),
               (d.get((reg, ur, "男性")) or 0) + (d.get((reg, ur, "女性")) or 0),
               f"{reg}{ur} 男+女")
        ok(d.get((reg, "城乡合计", "合计")),
           (d.get((reg, "城市", "合计")) or 0) + (d.get((reg, "农村", "合计")) or 0),
           f"{reg} 城+乡")
    for ur in URBRUR:
        ok(d.get(("全国", ur, "合计")),
           sum((d.get((r, ur, "合计")) or 0) for r in ["东部", "中部", "西部"]),
           f"东+中+西=全国({ur})")
    return checks


def open_combined(paths):
    """Open one or more PDF parts as a single document, dropping duplicated
    book pages across part boundaries (detected via the leading page-number
    token on each page)."""
    out = fitz.open()
    seen = set()
    for p in paths:
        src = fitz.open(p)
        for i in range(len(src)):
            first = src[i].get_text().strip().split("\n", 1)[0].strip()
            key = first if re.match(r"^\d{1,4}$", first) else None
            if key is not None and key in seen:
                continue
            if key is not None:
                seen.add(key)
            out.insert_pdf(src, from_page=i, to_page=i)
        src.close()
    return out


def extract_volume(doc, year):
    # locate 7.1 (counts) and 7.2 (rates) section starts
    p71 = p72 = pend = None
    for i in range(20, len(doc)):
        t = doc[i].get_text()[:400]
        if p71 is None and re.search(r"7\.1\s*地区别", t):
            p71 = i
        elif p71 is not None and p72 is None and i > p71 and re.search(r"7\.2\s*地区别", t):
            p72 = i
        elif p72 is not None and pend is None and i > p72 and ("附录" in t and "监测点" in t):
            pend = i
            break
    if pend is None:
        pend = len(doc)
    out = []
    for sec, (a, b), rates in [("counts", (p71, p72), False), ("rates", (p72, pend), True)]:
        if a is None or b is None:
            print(f"  [{year}] section {sec}: NOT FOUND (p71={p71}, p72={p72})"); continue
        blocks = parse_chapter7(doc, a, b, rates=rates)
        labs = label_blocks(len(blocks))
        print(f"  [{year}] {sec}: pages {a+1}-{b}, blocks={len(blocks)} (expect 36)")
        for (reg, ur, sx), blk in zip(labs, blocks):
            row = {"year": year, "measure": sec, "region": reg, "urbrur": ur, "sex": sx}
            row.update(blk)
            out.append(row)
    return out


VOLUMES = {
    2014: [os.path.join(BASE, "中国死因监测数据集2014", "2014年死因数据集", f)
           for f in ["2014死因监测数据集（上）.pdf", "2014死因监测数据集（中）.pdf", "2014死因监测数据集（下）.pdf"]],
    2015: [os.path.join(BASE, "中国死因监测数据集2015", f)
           for f in ["1数据集2016.27.pdf", "2数据集2016.27.pdf", "3数据集2016.27.pdf"]],
    # 2016: scanned images, no text layer -- needs OCR, handled separately
    2017: [os.path.join(BASE, "中国死因监测数据集2017.pdf")],
    2018: [os.path.join(BASE, "中国死因监测数据集2018.pdf")],
    2019: [os.path.join(BASE, "中国死因监测数据集2019.pdf")],
    2020: [os.path.join(BASE, "中国死因监测数据集2020.pdf")],
    2021: [os.path.join(BASE, "2021死因监测数据集.pdf")],
}

OUTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "cdc_homicide")


if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)
    all_rows, log = [], []
    for year, paths in sorted(VOLUMES.items()):
        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            log.append(f"[{year}] MISSING FILES: {missing}"); print(log[-1]); continue
        doc = open_combined(paths)
        rows = extract_volume(doc, year)
        doc.close()
        cnt = [r for r in rows if r["measure"] == "counts"]
        checks = validate(cnt)
        nfail = sum(1 for c in checks if c.startswith("FAIL"))
        nskip = sum(1 for c in checks if c.startswith("SKIP"))
        log.append(f"[{year}] blocks={len(rows)} identity checks: "
                   f"{len(checks)-nfail-nskip} PASS / {nfail} FAIL / {nskip} SKIP")
        print(log[-1])
        for c in checks:
            if not c.startswith("PASS"):
                log.append(f"   {year}: {c}"); print(log[-1])
        nat = next((r for r in cnt if (r["region"], r["urbrur"], r["sex"]) == ("全国", "城乡合计", "合计")), {})
        print(f"   全国他杀死亡数 N={nat.get('homicide')}  全死因={nat.get('all_cause')}")
        all_rows.extend(rows)
    # derive rates from counts + implied population (pop = all-cause count/rate);
    # raw extracted rates kept as cross-check columns
    key = lambda r: (r["year"], r["region"], r["urbrur"], r["sex"])
    counts = {key(r): r for r in all_rows if r["measure"] == "counts"}
    rates = {key(r): r for r in all_rows if r["measure"] == "rates"}
    final = []
    for k, c in counts.items():
        rr = rates.get(k, {})
        pop = None
        if c.get("all_cause") and rr.get("all_cause"):
            pop = c["all_cause"] / rr["all_cause"] * 1e5
        row = {"year": k[0], "region": k[1], "urbrur": k[2], "sex": k[3],
               "pop_implied": round(pop) if pop else None}
        for v in ["all_cause", "injury_all", "suicide", "homicide", "traffic_acc",
                  "homicide_15_59"]:
            row[v + "_n"] = c.get(v)
            row[v + "_rate"] = round(c[v] / pop * 1e5, 4) if (pop and c.get(v) is not None) else None
        row["homicide_rate_raw"] = rr.get("homicide")
        row["suicide_rate_raw"] = rr.get("suicide")
        final.append(row)
    # cross-check derived vs raw homicide rates
    bad = [(r["year"], r["region"], r["urbrur"], r["sex"], r["homicide_rate"], r["homicide_rate_raw"])
           for r in final if r["homicide_rate"] is not None and r["homicide_rate_raw"] is not None
           and abs(r["homicide_rate"] - r["homicide_rate_raw"]) > 0.011]
    print(f"\n他杀率 推导vs原始 偏差>0.01 的格子: {len(bad)}")
    for b in bad[:10]:
        print("  ", b)
    import csv
    cols = list(final[0].keys())
    fp = os.path.join(OUTDIR, "cdc_homicide_panel.csv")
    with open(fp, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(final, key=lambda x: (x["year"], x["region"], x["urbrur"], x["sex"])):
            w.writerow(r)
    with open(os.path.join(OUTDIR, "extract_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log))
    print("\nwrote", fp, f"({len(final)} rows)")

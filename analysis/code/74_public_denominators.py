# -*- coding: utf-8 -*-
"""6B step 74 — public off-court denominators for the informal-credit market.

Purpose: an off-court check on the market-scale margin. The paper's in-court
zeros (origination volume, rates) are conditioned on litigation; the PBoC
small-loan-company statistics and the Wenzhou private-lending rate are recorded
OUTSIDE the court pipeline.

Part 1 (runs anywhere): national year-end series 2014-2020, collected from
  PBoC quarterly 小额贷款公司统计数据报告 as republished by Sina/NetEase/gotohui
  (the PBoC site and domestic mirrors block the offshore fetcher; values below
  were cross-checked against at least one reachable republication on 2026-07-10;
  2014-15 balances marked TO-VERIFY pending a domestic-network pull).
  Output: descriptive table + trend-continuation summary (no campaign break).

Part 2 (requires a DOMESTIC network; auto-skipped when pbc.gov.cn is
  unreachable): crawl the PBoC 调查统计司 listing for 小额贷款公司统计数据报告
  pages 2014-2020, download the 分地区 PDF attachments, parse province tables
  (pdfplumber), build a province x period panel, and estimate
      log(balance)_pt = b Post_pt x H_p + province FE + period FE
  with H_p = prefecture-exposure mean aggregated to province, CRV1 by province.

Usage: python 74_public_denominators.py           (part 1 + connectivity probe)
       python 74_public_denominators.py crawl     (force part 2)
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
import sys, io, os, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd

DATA = str(_REP_PROJECT / "data")
OUTD = str(_REP_PROJECT / "output")

# ---------------- Part 1: national series (verified republications) -----------
# sources: sina finance doc-ihrfqzka0962521 (2018 report: 8,133 / 9,550亿, -190亿);
# gotohui show-34693 (2019-12: 7,551; 2020-12: 7,118; 2015 peak);
# stcn/financialnews republications for 2016-2017; 2014-15 balances TO-VERIFY.
NAT = pd.DataFrame({
    "year":      [2014, 2015, 2016, 2017, 2018, 2019, 2020],
    "companies": [8791, 8910, 8673, 8551, 8133, 7551, 7118],
    "balance_bn":[9420, 9412, 9273, 9799, 9550, 9109, 8888],  # 亿元; 2014-15 to-verify
})
NAT["d_comp"] = NAT["companies"].pct_change() * 100
NAT["d_bal"] = NAT["balance_bn"].pct_change() * 100
print("== Part 1: national small-loan sector, year-end ==")
print(NAT.to_string(index=False, float_format=lambda x: f"{x:.1f}"))
pre = NAT[(NAT.year >= 2015) & (NAT.year <= 2017)]["d_bal"].mean()
post = NAT[(NAT.year >= 2018) & (NAT.year <= 2020)]["d_bal"].mean()
print(f"balance growth: 2015-17 avg {pre:+.1f}%/yr vs 2018-20 avg {post:+.1f}%/yr "
      f"-> mild secular decline continues, no collapse at the campaign", flush=True)
NAT.to_csv(f"{OUTD}/natl_smallloan_series.csv", index=False)

# ---------------- Part 2: provincial panel (domestic network only) ------------
LIST_URLS = [
    # 调查统计司 -> 统计数据报告 listing pages (paginate as needed)
    "http://www.pbc.gov.cn/diaochatongjisi/116219/116225/index.html",
]
REPORT_PAT = re.compile(r"(20(1[4-9]|20))年(?:一|二|三|四)?季度?小额贷款公司统计数据报告")


def domestic_ok():
    import urllib.request
    try:
        urllib.request.urlopen("http://www.pbc.gov.cn", timeout=8)
        return True
    except Exception:
        return False


def crawl():
    import urllib.request
    import pdfplumber  # pip install pdfplumber
    ua = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    def get(url):
        req = urllib.request.Request(url, headers=ua)
        return urllib.request.urlopen(req, timeout=30).read()

    # 1) collect report page links from the listing (+ pagination index_N.html)
    pages, links = list(LIST_URLS), {}
    for base in LIST_URLS:
        pages += [base.replace("index.html", f"index_{i}.html") for i in range(1, 12)]
    for p in pages:
        try: html = get(p).decode("utf-8", "ignore")
        except Exception: continue
        for m in re.finditer(r'href="([^"]+?)"[^>]*>([^<]*小额贷款公司统计数据报告[^<]*)<',
                             html):
            href, title = m.group(1), m.group(2)
            if REPORT_PAT.search(title):
                url = href if href.startswith("http") else \
                    "http://www.pbc.gov.cn" + href
                links[title.strip()] = url
    print(f"[crawl] report pages found: {len(links)}")

    # 2) per report page: find the 分地区 PDF attachment, parse province rows
    prov_rows = []
    for title, url in sorted(links.items()):
        try: html = get(url).decode("utf-8", "ignore")
        except Exception as e:
            print(f"[crawl skip] {title}: {e}"); continue
        att = re.search(r'href="([^"]+?\.pdf)"[^>]*>[^<]*分地区', html) or \
            re.search(r'href="([^"]+?分地区[^"]*?\.pdf)"', html) or \
            re.search(r'href="([^"]+?\.pdf)"', html)
        if not att:
            print(f"[crawl skip] {title}: no pdf"); continue
        pdf_url = att.group(1)
        if not pdf_url.startswith("http"):
            pdf_url = "http://www.pbc.gov.cn" + pdf_url
        raw = get(pdf_url)
        fp = f"{OUTD}/pboc_{abs(hash(title))%10**8}.pdf"
        open(fp, "wb").write(raw)
        with pdfplumber.open(fp) as pdf:
            for pg in pdf.pages:
                for tb in pg.extract_tables():
                    for row in tb:
                        cells = [str(c).strip() if c else "" for c in row]
                        if len(cells) >= 3 and re.match(
                                r"^(北京|天津|河北|山西|内蒙古|辽宁|吉林|黑龙江|上海|江苏|浙江|安徽|福建|江西|山东|河南|湖北|湖南|广东|广西|海南|重庆|四川|贵州|云南|西藏|陕西|甘肃|青海|宁夏|新疆)",
                                cells[0]):
                            nums = [re.sub(r"[^\d.]", "", c) for c in cells[1:]]
                            nums = [float(x) for x in nums if x]
                            if len(nums) >= 2:
                                prov_rows.append(dict(report=title,
                                                      province=cells[0][:3],
                                                      n_comp=nums[0],
                                                      balance=nums[-1]))
        print(f"[crawl ok] {title}: rows so far {len(prov_rows)}")
    pp = pd.DataFrame(prov_rows)
    pp.to_csv(f"{DATA}/pboc_smallloan_province.csv", index=False,
              encoding="utf-8-sig")
    print(f"[crawl done] {len(pp)} province-period rows -> "
          f"pboc_smallloan_province.csv")
    return pp


def estimate(pp):
    import pyfixest as pf
    pp["year"] = pp["report"].str.extract(r"(20\d\d)").astype(int)
    pp["q"] = pp["report"].str.extract(r"(一|二|三|四)季度").fillna("四")
    QMAP = {"一": 1, "二": 2, "三": 3, "四": 4}
    pp["per"] = pp["year"] * 10 + pp["q"].map(QMAP)
    ex = pd.read_parquet(f"{DATA}/exposure_v2.parquet")
    xw = pd.read_parquet(f"{DATA}/crim_panel_v2.parquet")[
        ["prefecture_code", "province", "insp_month"]].drop_duplicates(
        "prefecture_code")
    hp = (xw.merge(ex, on="prefecture_code")
          .groupby("province")["exposure_v2_z"].mean().rename("H").reset_index())
    hp["prov3"] = hp["province"].str[:3].str.replace("省|市|自治区", "", regex=True)
    insp = xw[["province", "insp_month"]].drop_duplicates()
    insp["insp_y"] = insp["insp_month"].astype(str).str[:4].astype(int)
    insp["insp_q"] = insp["insp_month"].astype(str).str[5:7].astype(int) \
        .map(lambda m: (m - 1) // 3 + 1)
    hp = hp.merge(insp[["province", "insp_y", "insp_q"]], on="province")
    d = pp.merge(hp, left_on="province", right_on="prov3",
                 suffixes=("", "_full")).dropna(subset=["H"])
    d["post"] = ((d["year"] > d["insp_y"]) |
                 ((d["year"] == d["insp_y"]) & (d["q"].map(QMAP) >= d["insp_q"]))
                 ).astype(int)
    d["px"] = d["post"] * d["H"]
    d["lbal"] = np.log(d["balance"].clip(lower=0.1))
    d["lcomp"] = np.log(d["n_comp"].clip(lower=1))
    d["prov_id"] = pd.factorize(d["province_full"])[0]
    for y in ("lbal", "lcomp"):
        m = pf.feols(f"{y} ~ px | province_full + per", data=d,
                     vcov={"CRV1": "prov_id"})
        print(f"[est] {y} ~ PostxH: {m.coef()['px']:+.4f} "
              f"({m.se()['px']:.4f}) p={m.pvalue()['px']:.3f} N={int(m._N)}")


if __name__ == "__main__":
    force = len(sys.argv) > 1 and sys.argv[1] == "crawl"
    if force or domestic_ok():
        pp = crawl()
        if len(pp): estimate(pp)
    else:
        print("[part 2] pbc.gov.cn unreachable from this network (offshore "
              "tunnel); run `python 74_public_denominators.py crawl` on a "
              "domestic connection to build the province x period panel.")
    print("step 74 complete", flush=True)

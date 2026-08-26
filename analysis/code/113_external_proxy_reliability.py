# -*- coding: utf-8 -*-
"""Do the two external proxies carry any signal at all?

112 found near-zero correlation between H_c and both external indices, and a
failed substitution. That result is only informative about H_c if the proxies
are themselves reliable measures of *something*. This script asks:

  1. split-half reliability of each proxy (odd vs even founding years for the
     firm index; keyword 讨债 vs 讨债公司 for the search index). A proxy that
     does not correlate with itself cannot be used to judge H_c.
  2. construct validity of the firm index: what industry are the 6,738
     registered "collection" firms actually in? If they are bank-commissioned
     credit-card collectors, they measure the licensed formal-credit margin,
     not the illegal coercive backstop the paper studies.
  3. how much within-province variance each proxy has, since that is the
     variation the dose design uses.
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
import io
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE = str(_REP_PROJECT)
DATA = os.path.join(BASE, "data")
OUT = os.path.join(BASE, "output", "exposure_validity")
GS = str(_REP_REGISTRY).replace('\\', '/')
BD = str(_REP_BAIDU / 'baidu_index_city_month.csv').replace('\\', '/')
if os.environ.get("HWIH_REPLICATION") == "1":
    REGISTRY_HITS = os.path.join(DATA, "derived", "registry_hits_deidentified.csv")
    REGISTRY_AGG = os.path.join(DATA, "derived", "registry_aggregate")
    BD = os.path.join(DATA, "derived", "baidu_index_city_month.csv")
else:
    REGISTRY_HITS = os.path.join(GS, "hits_clean.csv")
    REGISTRY_AGG = os.path.join(GS, "agg")
PRE_START, PRE_END = "2014-01", "2017-12"
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def z(s):
    s = pd.to_numeric(s, errors="coerce")
    return (s - s.mean()) / s.std(ddof=1)


def within(frame, col, group="province"):
    return frame[col] - frame.groupby(group)[col].transform("mean")


xw = pd.read_parquet(os.path.join(DATA, "court_xwalk.parquet"))
_pat = re.compile(r"(?:.*?省|.*?自治区)?(.+?(?:市|州|盟|地区))中级人民法院$")
_names = {}
for _, r in xw.iterrows():
    m = _pat.match(str(r["court_name"]))
    if m:
        _names.setdefault(m.group(1), []).append(r["prefecture_code"])
NAME2CODE = {nm: max(set(v), key=v.count) for nm, v in _names.items()}
NAME2CODE.update({"北京市": "110100", "天津市": "120100",
                  "上海市": "310100", "重庆市": "500100"})


def city2code(c):
    c = str(c)
    if not c or c == "nan":
        return None
    cand = c if c.endswith(("市", "州", "盟", "地区")) else c + "市"
    if cand in NAME2CODE:
        return NAME2CODE[cand]
    for nm, cd in NAME2CODE.items():
        if nm[:-1] and nm[:-1] in c:
            return cd
    return None


exposure = pd.read_parquet(os.path.join(DATA, "exposure_v2.parquet"))

# ---------------------------------------------------------------------------
# 2. What industry are the registered "collection" firms actually in?
# ---------------------------------------------------------------------------
say("=== construct validity of the firm index ===")
h = pd.read_csv(REGISTRY_HITS, dtype=str)
h = h.rename(columns={"所属城市": "district_raw"})
h["所属城市"] = h["所属省份"]
h["prefecture_code"] = h["所属城市"].map(city2code)
h["em"] = pd.to_datetime(h["成立日期"], errors="coerce")
h["exited"] = h["经营状态"].str.contains("注销|吊销", na=False)
h["xm"] = pd.to_datetime(h["核准日期"], errors="coerce").where(h["exited"])
snip = h["snippet"].fillna("")
name = h["企业名称"].fillna("")

BANKISH = re.compile(r"银行|信用卡|信贷|金融机构|持牌|委托|资产管理|不良资产|"
                     r"保理|征信|应收账款|逾期户|贷后")
NAMEHIT = (
    h["name_hit"].astype(str).str.lower().isin({"true", "1", "yes"})
    if "name_hit" in h.columns else h["matched_field"].eq("名称")
)
if "bank_context" in h.columns:
    h["bank_context"] = h["bank_context"].astype(str).str.lower().isin(
        {"true", "1", "yes"})
else:
    h["bank_context"] = snip.str.contains(BANKISH) | name.str.contains(BANKISH)
say(f"  firms: {len(h)}")
say(f"  matched on company NAME (not scope): {int(NAMEHIT.sum())} "
    f"({NAMEHIT.mean():.1%})")
say(f"  snippet or name carries a licensed-credit context "
    f"(bank / credit card / NPL / factoring / receivables): "
    f"{int(h['bank_context'].sum())} ({h['bank_context'].mean():.1%})")
for kw, sub in h.groupby("matched_kw"):
    say(f"    keyword {kw}: n={len(sub)}, licensed-credit context "
        f"{sub['bank_context'].mean():.1%}")

# ---------------------------------------------------------------------------
# 1a. Split-half reliability of the firm index (odd vs even founding year)
# ---------------------------------------------------------------------------
say("\n=== split-half reliability: firm index ===")
agg = []
import glob
for fp in sorted(glob.glob(os.path.join(REGISTRY_AGG, "*.csv"))):
    a = pd.read_csv(fp, dtype={"city": str, "month": str})
    a = a[a["month"].between(PRE_START, PRE_END)]
    if len(a):
        agg.append(a)
agg = pd.concat(agg, ignore_index=True)
agg["prefecture_code"] = agg["city"].map(city2code)
den = (agg.dropna(subset=["prefecture_code"])
       .groupby("prefecture_code", as_index=False)
       .agg(all_entries=("entries_all", "sum")))

pre = h[(h["em"] >= "2014-01-01") & (h["em"] <= "2017-12-31")].copy()
pre["yr"] = pre["em"].dt.year
halves = {}
for lab, yrs in [("A_2014_2015", [2014, 2015]), ("B_2016_2017", [2016, 2017])]:
    halves[lab] = (pre[pre["yr"].isin(yrs)].groupby("prefecture_code").size()
                   .rename(lab))
fh = den.merge(pd.concat(halves.values(), axis=1).reset_index()
               .rename(columns={"index": "prefecture_code"}),
               on="prefecture_code", how="left")
for lab in halves:
    fh[lab] = fh[lab].fillna(0.0)
    fh[lab + "_d"] = np.arcsinh(fh[lab] / fh["all_entries"] * 1e6)
fh = fh.merge(exposure[["prefecture_code", "province"]], on="prefecture_code")
r, p = stats.pearsonr(fh["A_2014_2015_d"], fh["B_2016_2017_d"])
sr, sp = stats.spearmanr(fh["A_2014_2015_d"], fh["B_2016_2017_d"])
fh["Aw"] = within(fh, "A_2014_2015_d")
fh["Bw"] = within(fh, "B_2016_2017_d")
rw, pw_ = stats.pearsonr(fh["Aw"], fh["Bw"])
say(f"  n={len(fh)}  split-half r={r:.3f} (p={p:.2g}) spearman={sr:.3f}")
say(f"  within-province split-half r={rw:.3f} (p={pw_:.2g})")
sb_full = 2 * r / (1 + r) if r > -1 else np.nan
sb_within = 2 * rw / (1 + rw) if rw > -1 else np.nan
say(f"  Spearman-Brown reliability of the full index: {sb_full:.3f} "
    f"(within-province {sb_within:.3f})")

# ---------------------------------------------------------------------------
# 1b. Split-half reliability of the search index (two keywords)
# ---------------------------------------------------------------------------
say("\n=== split-half reliability: Baidu search index ===")
bd = pd.read_csv(BD, dtype={"ym": str, "city": str})
bd["prefecture_code"] = bd["city"].map(city2code)
bd = bd[bd["ym"].between(PRE_START, PRE_END)].dropna(subset=["prefecture_code"])
kw = {}
for k in ["讨债", "讨债公司", "收数公司"]:
    kw[k] = (bd[bd["keyword"].eq(k)].groupby("prefecture_code")["mean"].mean()
             .rename(k))
bh = pd.concat(kw.values(), axis=1).reset_index()
bh = bh.merge(exposure[["prefecture_code", "province"]], on="prefecture_code")
for k in kw:
    bh[k + "_a"] = np.arcsinh(bh[k])
r2, p2 = stats.pearsonr(bh["讨债_a"], bh["讨债公司_a"])
bh["Aw"] = within(bh, "讨债_a")
bh["Bw"] = within(bh, "讨债公司_a")
rw2, pw2 = stats.pearsonr(bh["Aw"], bh["Bw"])
say(f"  n={len(bh)}  讨债 vs 讨债公司 r={r2:.3f} (p={p2:.2g})")
say(f"  within-province r={rw2:.3f} (p={pw2:.2g})")
say(f"  Spearman-Brown reliability: {2 * r2 / (1 + r2):.3f} "
    f"(within-province {2 * rw2 / (1 + rw2):.3f})")

# ---------------------------------------------------------------------------
# 3. Variance decomposition of every index
# ---------------------------------------------------------------------------
say("\n=== within-province share of variance (the dose margin) ===")
X = pd.read_csv(os.path.join(OUT, "external_indices.csv"),
                dtype={"prefecture_code": str})
rows = []
for c, lab in [("exposure_v2_z", "H_c composite"),
               ("violent_share_z", "H_c: violent share"),
               ("backstop_rate_z", "H_c: collection narrative"),
               ("firm_stock_z", "firm density (external)"),
               ("baidu_z", "Baidu intensity (external)")]:
    d = X.dropna(subset=[c, "province"])
    tot = d[c].var(ddof=1)
    w = (d[c] - d.groupby("province")[c].transform("mean")).var(ddof=1)
    rows.append({"index": lab, "n": len(d), "var_total": tot,
                 "var_within_province": w, "within_share": w / tot})
    say(f"  {lab}: n={len(d)} within-province share = {w / tot:.2f}")
vd = pd.DataFrame(rows)

pd.DataFrame([
    {"proxy": "collection-firm density", "n": len(fh),
     "split_half_r": r, "split_half_within_r": rw,
     "spearman_brown": sb_full, "spearman_brown_within": sb_within,
     "licensed_credit_context_share": float(h["bank_context"].mean())},
    {"proxy": "Baidu search intensity", "n": len(bh),
     "split_half_r": r2, "split_half_within_r": rw2,
     "spearman_brown": 2 * r2 / (1 + r2),
     "spearman_brown_within": 2 * rw2 / (1 + rw2),
     "licensed_credit_context_share": np.nan},
]).to_csv(os.path.join(OUT, "proxy_reliability.csv"), index=False,
          encoding="utf-8-sig")
vd.to_csv(os.path.join(OUT, "variance_decomposition.csv"), index=False,
          encoding="utf-8-sig")

# attenuation ceiling: with reliability rho_xx, an observed correlation cannot
# exceed sqrt(rho_xx * rho_yy). Report what the firm/baidu proxies could ever show.
say("\n=== attenuation ceiling on the validation test ===")
for lab, rel in [("firm density", sb_within), ("Baidu intensity",
                                               2 * rw2 / (1 + rw2))]:
    ceil = np.sqrt(max(rel, 0) * 1.0)
    say(f"  {lab}: within-province reliability {rel:.3f} -> even a perfectly "
        f"valid H_c could correlate at most {ceil:.3f} with it")

with open(os.path.join(OUT, "proxy_reliability_log.txt"), "w",
          encoding="utf-8") as fh_:
    fh_.write("\n".join(LOG))
say("\nDONE ->", OUT)

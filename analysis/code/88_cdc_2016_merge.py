# -*- coding: utf-8 -*-
"""Merge the 2016 volume (scanned; visually read, identity-validated) into
cdc_homicide_panel.csv.

The 2016 volume of 中国死因监测数据集 has no text layer. Its 72 chapter-7
injury pages were read visually (pages_2016_raw.json) and validated before
this merge: 95/95 adding-up identities on counts across five causes and
108/108 per-age-column identities on the homicide 15-55 vector. Population
is implied from the injury-all count/rate pair (rate rounding => ~0.04%).
all_cause is not read for 2016 (U000 pages not needed for the series).
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
import json, os
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "output", "cdc_homicide")

BLOCK_ORDER = [("全国", "城乡合计"), ("全国", "城市"), ("全国", "农村"),
               ("东部", "城乡合计"), ("中部", "城乡合计"), ("西部", "城乡合计"),
               ("东部", "城市"), ("中部", "城市"), ("西部", "城市"),
               ("东部", "农村"), ("中部", "农村"), ("西部", "农村")]
SEXES = ["合计", "男性", "女性"]
LABS = [(r, u, s) for r, u in BLOCK_ORDER for s in SEXES]
VARS = ["homicide", "suicide", "traffic", "intentional", "injury_all"]

pages = json.load(open(os.path.join(OUT, "pages_2016_raw.json"), encoding="utf-8"))
cnt = [p for p in pages if p["kind"] == "counts"]
rts = [p for p in pages if p["kind"] == "rates"]
assert len(cnt) == 36 and len(rts) == 36

# re-run the identity gate before touching the panel
fails = 0
d = dict(zip(LABS, cnt))
for v in VARS:
    g = lambda r, u, s: d[(r, u, s)][v]
    for r, u in BLOCK_ORDER:
        fails += g(r, u, "合计") != g(r, u, "男性") + g(r, u, "女性")
    for r in ["全国", "东部", "中部", "西部"]:
        fails += g(r, "城乡合计", "合计") != g(r, "城市", "合计") + g(r, "农村", "合计")
    for u in ["城乡合计", "城市", "农村"]:
        fails += g("全国", u, "合计") != sum(g(rr, u, "合计") for rr in ["东部", "中部", "西部"])
assert fails == 0, f"{fails} identity failures -- do not merge"

rd = dict(zip(LABS, rts))
rows = []
for lab in LABS:
    c, r = d[lab], rd[lab]
    pop = c["injury_all"] / r["injury_all"] * 1e5
    ages = c.get("homicide_ages_15_55") or []
    h1559 = sum(ages) if len(ages) == 9 else None
    row = {"year": 2016, "region": lab[0], "urbrur": lab[1], "sex": lab[2],
           "pop_implied": round(pop),
           "all_cause_n": None, "all_cause_rate": None,
           "injury_all_n": c["injury_all"], "injury_all_rate": round(c["injury_all"] / pop * 1e5, 4),
           "suicide_n": c["suicide"], "suicide_rate": round(c["suicide"] / pop * 1e5, 4),
           "homicide_n": c["homicide"], "homicide_rate": round(c["homicide"] / pop * 1e5, 4),
           "traffic_acc_n": c["traffic"], "traffic_acc_rate": round(c["traffic"] / pop * 1e5, 4),
           "homicide_15_59_n": h1559,
           "homicide_15_59_rate": round(h1559 / pop * 1e5, 4) if h1559 is not None else None,
           "homicide_rate_raw": r["homicide"], "suicide_rate_raw": r["suicide"]}
    rows.append(row)

fp = os.path.join(OUT, "cdc_homicide_panel.csv")
panel = pd.read_csv(fp)
panel = panel[panel.year != 2016]
panel = pd.concat([panel, pd.DataFrame(rows)], ignore_index=True)
panel = panel.sort_values(["year", "region", "urbrur", "sex"])
panel.to_csv(fp, index=False, encoding="utf-8-sig")
nat = [r for r in rows if (r["region"], r["urbrur"], r["sex"]) == ("全国", "城乡合计", "合计")][0]
print(f"merged 2016: 全国他杀 N={nat['homicide_n']} rate={nat['homicide_rate']:.2f} "
      f"(书中率 {nat['homicide_rate_raw']}); panel rows={len(panel)}")

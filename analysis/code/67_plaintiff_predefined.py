# -*- coding: utf-8 -*-
"""6B step 67 — plaintiff composition with PRE-DETERMINED creditor types.
Types fixed on the 2014-2016 base period, strictly before the clean window:
incumbent = plaintiff appeared in the prefecture's lending docket in 2014-16;
professional = >= 5 lending cases in 2014-16. Outcomes: share of the cell's
window filings brought by each pre-defined type. Immune to the mechanical
volume effect (type does not depend on window behavior). Stock rerouting
(Proposition p:stock) predicts positive Post x Treat x H on both shares.
Output: output/ext2124/plaintiff_predefined.csv
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
import duckdb, glob, sys, io
import numpy as np
import pandas as pd
import pyfixest as pf
from _wild import wild_p

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
DATA = str(_REP_PROJECT / "data")
OUT = str(_REP_PROJECT / "output" / "ext2124")
WINDOW = ("2017-01", "2019-03"); POST0 = "2018-09"

con = duckdb.connect()
con.sql("SET threads TO 8; SET memory_limit='16GB'")
files = sorted(glob.glob(str(_REP_PROJECT / "data" / "derived" / "extract_2014_2020" / 'civil_20*.parquet').replace('\\', '/')))
df = con.sql(f"""
SELECT w.prefecture_code, w.province,
  strftime(TRY_CAST(x.judgment_date AS DATE), '%Y-%m') AS month,
  trim(split_part(x.parties, ',', 1)) AS plaintiff
FROM read_parquet({files}) x
JOIN '{DATA}/court_xwalk.parquet' w ON x.court = w.court_name
WHERE x.cause = '民间借贷纠纷' AND x.doc_type = 'judgment'
  AND x.parties IS NOT NULL AND length(trim(split_part(x.parties, ',', 1))) >= 2
""").df()
df = df[df["month"].notna()]
df = df[~df["plaintiff"].str.contains("某", na=False)]
ORG_RX = "公司|银行|信用社|合作社|小额贷款|担保|投资|典当|基金|事务所|中心|厂"
df = df[~df["plaintiff"].str.contains(ORG_RX, na=False, regex=True)]

base = df[(df["month"] >= "2014-01") & (df["month"] <= "2016-12")]
bcount = (base.groupby(["prefecture_code", "plaintiff"]).size().rename("k").reset_index())
bcount["incumbent"] = True
bcount["pro_pre"] = bcount["k"] >= 5
win = df[(df["month"] >= WINDOW[0]) & (df["month"] <= WINDOW[1])].merge(
    bcount[["prefecture_code", "plaintiff", "incumbent", "pro_pre"]],
    on=["prefecture_code", "plaintiff"], how="left")
win["incumbent"] = win["incumbent"].fillna(False).astype(float)
win["pro_pre"] = win["pro_pre"].fillna(False).astype(float)
cells = (win.groupby(["prefecture_code", "province", "month"])
         .agg(n=("plaintiff", "size"), sh_incumbent=("incumbent", "mean"),
              sh_pro_pre=("pro_pre", "mean")).reset_index())

sched = pd.read_parquet(f"{DATA}/panel_month.parquet")[["province", "inspection_round"]].drop_duplicates()
ex = pd.read_parquet(f"{DATA}/exposure_v2.parquet")[["prefecture_code", "exposure_v2_z"]]
g = cells.merge(sched, on="province").merge(ex, on="prefecture_code")
g = g.dropna(subset=["exposure_v2_z", "inspection_round"])
g = g[g["n"] >= 10]
g["treat"] = (g["inspection_round"] == 1).astype(int)
g["postc"] = (g["month"] >= POST0).astype(int)
g["prov_id"] = pd.factorize(g["province"])[0]
g["pref"] = g["prefecture_code"]
g["pt"] = g["postc"] * g["treat"]
g["pth"] = g["pt"] * g["exposure_v2_z"]
g["ph"] = g["postc"] * g["exposure_v2_z"]
rows = []
for y in ("sh_incumbent", "sh_pro_pre"):
    fml = f"{y} ~ pth + ph + pt | pref + month"
    m = pf.feols(fml, data=g, vcov={"CRV1": "prov_id"}, weights="n")
    wp = wild_p(fml, g, "pth")
    b0 = np.average(g.loc[g.postc == 0, y], weights=g.loc[g.postc == 0, "n"])
    print(f"{y:14s} pth={m.coef()['pth']: .4f} (se {m.se()['pth']:.4f}) "
          f"p={m.pvalue()['pth']:.4f} wild={wp:.3f} base={b0:.3f} N={int(m._N):,}")
    rows.append(dict(outcome=y, est=m.coef()["pth"], se=m.se()["pth"],
                     p_crv1=m.pvalue()["pth"], p_wild=wp, base=b0, n=int(m._N)))
pd.DataFrame(rows).to_csv(f"{OUT}/plaintiff_predefined.csv", index=False)
print("written")

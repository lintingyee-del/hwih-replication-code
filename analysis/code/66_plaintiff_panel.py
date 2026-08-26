# -*- coding: utf-8 -*-
"""6B step 66 — plaintiff-level linkage (the feasible actor-level test).
From the 2014-2020 lending-judgment extracts (parties field), build per
prefecture-month: (i) first-time plaintiff share (individual, first appearance
since 2014-01 in the prefecture), (ii) professional share (plaintiff with >= 5
lending cases in the trailing 12 months), (iii) organizational-plaintiff share.
Clean-window DiD (Post x Treat x H) on each. Descriptive composition margins;
the design-grade test with PRE-DETERMINED creditor types (immune to the
mechanical volume effect) is step 67. Output: output/ext2124/plaintiff_panel.csv
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
print(f"lending judgments with plaintiff: {len(df):,}")
df = df[df["month"].notna()]
df["red"] = df["plaintiff"].str.contains("某", na=False)
ORG_RX = "公司|银行|信用社|合作社|小额贷款|担保|投资|典当|基金|事务所|中心|厂"
df["org"] = df["plaintiff"].str.contains(ORG_RX, na=False, regex=True)
ind = df[~df["red"] & ~df["org"]].copy()
print(f"individual non-redacted: {len(ind):,} ({len(ind)/len(df):.2%}); "
      f"org {df['org'].mean():.2%}, redacted {df['red'].mean():.2%}")

ind = ind.sort_values(["prefecture_code", "plaintiff", "month"])
ind["first_ever"] = ~ind.duplicated(["prefecture_code", "plaintiff"])
pm = (ind.groupby(["prefecture_code", "plaintiff", "month"]).size()
      .rename("k").reset_index())
pm["mi"] = pd.PeriodIndex(pm["month"], freq="M").astype("int64")
pm = pm.sort_values(["prefecture_code", "plaintiff", "mi"])
out = []
for (pc, pl), g in pm.groupby(["prefecture_code", "plaintiff"], sort=False):
    if len(g) == 1:
        out.append((g["k"].iloc[0],))
        continue
    mi = g["mi"].to_numpy(); k = g["k"].to_numpy()
    roll = [k[(mi > mi[i] - 12) & (mi <= mi[i])].sum() for i in range(len(g))]
    out.append(tuple(roll))
pm["roll12"] = [v for tup in out for v in tup]
ind = ind.merge(pm[["prefecture_code", "plaintiff", "month", "roll12"]],
                on=["prefecture_code", "plaintiff", "month"], how="left")
ind["pro"] = ind["roll12"] >= 5

cell_i = (ind.groupby(["prefecture_code", "province", "month"])
          .agg(n=("plaintiff", "size"), first_share=("first_ever", "mean"),
               pro_share=("pro", "mean")).reset_index())
cell_o = (df.groupby(["prefecture_code", "province", "month"])
          .agg(n_all=("org", "size"), org_share=("org", "mean")).reset_index())
cells = cell_i.merge(cell_o, on=["prefecture_code", "province", "month"])

sched = pd.read_parquet(f"{DATA}/panel_month.parquet")[["province", "inspection_round"]].drop_duplicates()
ex = pd.read_parquet(f"{DATA}/exposure_v2.parquet")[["prefecture_code", "exposure_v2_z"]]
g = cells.merge(sched, on="province").merge(ex, on="prefecture_code")
g = g.dropna(subset=["exposure_v2_z", "inspection_round"])
g = g[(g["month"] >= WINDOW[0]) & (g["month"] <= WINDOW[1]) & (g["n"] >= 10)]
g["treat"] = (g["inspection_round"] == 1).astype(int)
g["postc"] = (g["month"] >= POST0).astype(int)
g["prov_id"] = pd.factorize(g["province"])[0]
g["pref"] = g["prefecture_code"]
g["pt"] = g["postc"] * g["treat"]
g["pth"] = g["pt"] * g["exposure_v2_z"]
g["ph"] = g["postc"] * g["exposure_v2_z"]

rows = []
for y, w in [("first_share", "n"), ("pro_share", "n"), ("org_share", "n_all")]:
    fml = f"{y} ~ pth + ph + pt | pref + month"
    m = pf.feols(fml, data=g, vcov={"CRV1": "prov_id"}, weights=w)
    wp = wild_p(fml, g, "pth")
    base = np.average(g.loc[g.postc == 0, y], weights=g.loc[g.postc == 0, w])
    print(f"{y:12s} pth={m.coef()['pth']: .4f} (se {m.se()['pth']:.4f}) "
          f"p={m.pvalue()['pth']:.4f} wild={wp:.3f} base={base:.3f} N={int(m._N):,}")
    rows.append(dict(outcome=y, est=m.coef()["pth"], se=m.se()["pth"],
                     p_crv1=m.pvalue()["pth"], p_wild=wp, base=base, n=int(m._N)))
pd.DataFrame(rows).to_csv(f"{OUT}/plaintiff_panel.csv", index=False)
print("written:", f"{OUT}/plaintiff_panel.csv")

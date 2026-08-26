# -*- coding: utf-8 -*-
"""6B step 61 — two referee fixes.
A. Acquaintance-vs-stranger DIFFERENCE test: pool the two clean-window lending
   segments and estimate Post x Treat x H x Acq (segment-specific prefecture
   and month FE), province CRV1 + wild-score p. Decides the abstract's
   "concentrated in acquaintance loans" wording.
B. 2018 release-dip neutrality gate: 2017-to-2018 change in log releases on
   exposure H_c, for criminal (17 offenses), civil lending judgments, and the
   traffic placebo. Output: output/ext2124/acqdiff_2018gate.csv
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
import duckdb, sys, io
import numpy as np
import pandas as pd
import pyfixest as pf
from _wild import wild_p

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
DATA = str(_REP_PROJECT / "data")
OUT = str(_REP_PROJECT / "output" / "ext2124")
WIN = ("2017-01", "2019-03")
POST0 = "2018-09"
rows = []

# ---------------- A: difference test ----------------
cc = pd.read_parquet(f"{DATA}/civil_case.parquet",
                     columns=["cause", "prefecture_code", "province", "jmonth", "rel_txn"])
ld = cc[cc["cause"] == "民间借贷纠纷"].copy()
ld["month"] = ld["jmonth"].astype(str).str[:7]
ld = ld[(ld["month"] >= WIN[0]) & (ld["month"] <= WIN[1])]
ld["acq"] = ld["rel_txn"].fillna(0).astype(int)
sched = pd.read_parquet(f"{DATA}/panel_month.parquet")[["province", "inspection_round"]].drop_duplicates()
ex = pd.read_parquet(f"{DATA}/exposure_v2.parquet")[["prefecture_code", "exposure_v2_z"]]

g = (ld.groupby(["prefecture_code", "province", "month", "acq"]).size().rename("n")
     .reset_index().merge(sched, on="province").merge(ex, on="prefecture_code")
     .dropna(subset=["exposure_v2_z", "inspection_round"]))
g["H"] = g["exposure_v2_z"]
g["treat"] = (g["inspection_round"] == 1).astype(int)
g["postc"] = (g["month"] >= POST0).astype(int)
g["prov_id"] = pd.factorize(g["province"])[0]
g["pt"] = g["postc"] * g["treat"]
g["pth"] = g["pt"] * g["H"]
g["ph"] = g["postc"] * g["H"]
g["y"] = np.arcsinh(g["n"])
for c in ("pth", "ph", "pt", "postc"):
    g[f"{c}A"] = g[c] * g["acq"]
g["prefA"] = g["prefecture_code"] + "_" + g["acq"].astype(str)
g["monthA"] = g["month"] + "_" + g["acq"].astype(str)

fml = "y ~ pthA + phA + ptA + postcA + pth + ph + pt | prefA + monthA"
m = pf.feols(fml, data=g, vcov={"CRV1": "prov_id"})
wp = wild_p(fml, g, "pthA")
print(f"A. acq-minus-stranger difference (pthA): {m.coef()['pthA']:.4f} "
      f"(se {m.se()['pthA']:.4f}), CRV1 p={m.pvalue()['pthA']:.4f}, wild p={wp:.3f}, "
      f"N={int(m._N):,}")
rows.append(dict(test="acq_minus_stranger_pthA", est=m.coef()["pthA"],
                 se=m.se()["pthA"], p_crv1=m.pvalue()["pthA"], p_wild=wp, n=int(m._N)))

# ---------------- B: 2018 dip gate ----------------
con = duckdb.connect()
crim = con.sql(f"""
  SELECT prefecture_code, yr, SUM(n_all) n FROM
  (SELECT prefecture_code, yr, n_all FROM '{OUT}/persist_panel.parquet')
  WHERE yr IN (2017, 2018) GROUP BY 1,2""").df()
civ = con.sql(f"""
  SELECT prefecture_code, year(jmonth) yr,
    SUM((cause='民间借贷纠纷' AND doc_type='judgment')::INT) n_lend,
    SUM((cause='机动车交通事故责任纠纷' AND doc_type='judgment')::INT) n_traffic
  FROM '{DATA}/civil_case.parquet' WHERE year(jmonth) IN (2017, 2018)
  GROUP BY 1,2""").df()
exp = pd.read_parquet(f"{DATA}/exposure_v2.parquet")[["prefecture_code", "province", "exposure_v2_z"]]

def gate(df, col, label):
    w = df.pivot_table(index="prefecture_code", columns="yr", values=col, aggfunc="sum")
    w = w[(w[2017] >= 10) & (w[2018] >= 1)].copy()
    w["dy"] = np.log(w[2018]) - np.log(w[2017])
    d = w.reset_index().merge(exp, on="prefecture_code").dropna(subset=["exposure_v2_z"])
    d["prov_id"] = pd.factorize(d["province"])[0]
    d["H"] = d["exposure_v2_z"]
    m = pf.feols("dy ~ H", data=d, vcov={"CRV1": "prov_id"}, weights=None)
    print(f"B. 2018-dip gate [{label}]: beta_H = {m.coef()['H']:.4f} "
          f"(se {m.se()['H']:.4f}, p {m.pvalue()['H']:.3f}), n={len(d)}, "
          f"mean dy = {d.dy.mean():.3f}")
    rows.append(dict(test=f"dip2018_{label}", est=m.coef()["H"], se=m.se()["H"],
                     p_crv1=m.pvalue()["H"], p_wild=np.nan, n=len(d)))

gate(crim, "n", "criminal")
gate(civ, "n_lend", "lending")
gate(civ, "n_traffic", "traffic")

pd.DataFrame(rows).to_csv(f"{OUT}/acqdiff_2018gate.csv", index=False)
print("written:", f"{OUT}/acqdiff_2018gate.csv")

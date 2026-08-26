# -*- coding: utf-8 -*-
"""6B step 13 — C4 stake gradient: litigation response by claim-size bin.
Model predicts an inverted U: response concentrated in mid-sized stakes
(Proposition 3). Cells: prefecture x month x amount-bin among lending cases."""

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
import pandas as pd, numpy as np, pyfixest as pf, duckdb, os

DATA = str(_REP_PROJECT / "data")
OUTD = str(_REP_PROJECT / "output")
con = duckdb.connect()

cov = con.sql(f"""SELECT COUNT(*) n, COUNT(amount_yuan) has_amt,
  AVG((amount_yuan BETWEEN 100 AND 5e8)::INT) plausible
  FROM '{DATA}/civil_case.parquet' WHERE cause='民间借贷纠纷'""").df()
print("amount coverage:", cov.to_string(index=False))

cells = con.sql(f"""
WITH ld AS (
  SELECT prefecture_code, province, jmonth, post, insp_month, amount_yuan
  FROM '{DATA}/civil_case.parquet'
  WHERE cause='民间借贷纠纷' AND amount_yuan BETWEEN 100 AND 5e8
), b AS (
  SELECT *, CASE WHEN amount_yuan < 20000 THEN 'q1_lt2w'
                 WHEN amount_yuan < 50000 THEN 'q2_2_5w'
                 WHEN amount_yuan < 200000 THEN 'q3_5_20w'
                 WHEN amount_yuan < 1000000 THEN 'q4_20_100w'
                 ELSE 'q5_gt100w' END AS bin
  FROM ld
)
SELECT b.prefecture_code, b.province, b.jmonth, b.bin,
  ANY_VALUE(post) AS post, COUNT(*) AS n_cases, e.exposure_v2_z
FROM b JOIN '{DATA}/exposure_v2.parquet' e USING (prefecture_code)
GROUP BY 1,2,3,4,7
""").df()
cells["asinh_n"] = np.arcsinh(cells["n_cases"])
cells["month"] = cells["jmonth"].astype(str)
cells["prov_id"] = pd.factorize(cells["province"])[0]
cells["prov_month"] = cells["province"] + "_" + cells["month"]
cells["pref_bin"] = cells["prefecture_code"] + "_" + cells["bin"]
cells["px"] = cells["post"] * cells["exposure_v2_z"]

rows = []
for q in sorted(cells["bin"].unique()):
    d = cells[cells["bin"] == q].copy()
    d["pref"] = d["prefecture_code"]
    m = pf.feols("asinh_n ~ px | pref + prov_month", data=d, vcov={"CRV1": "prov_id"})
    rows.append(dict(tag=f"C4_{q}", coef="px", est=m.coef()["px"], se=m.se()["px"],
                     p=m.pvalue()["px"], wild_p=np.nan, n=int(m._N)))
    print(f"C4 {q}: {m.coef()['px']: .5f} ({m.se()['px']:.5f}) p={m.pvalue()['px']:.4f}")

old = pd.read_csv(f"{OUTD}/results_v2.csv")
new = pd.concat([old[~old["tag"].str.startswith("C4_")], pd.DataFrame(rows)])
new.to_csv(f"{OUTD}/results_v2.csv", index=False)

L = [r"\begin{tabular}{lccc}", r"\toprule",
     r"Claim-size bin & Post$\times$Exposure & (SE) & $N$ \\ \midrule"]
LAB = {"q1_lt2w": "$<$20k yuan", "q2_2_5w": "20--50k", "q3_5_20w": "50--200k",
       "q4_20_100w": "200k--1m", "q5_gt100w": "$>$1m"}
for r0 in rows:
    q = r0["tag"][3:]
    st = "***" if r0["p"] < .01 else "**" if r0["p"] < .05 else "*" if r0["p"] < .1 else ""
    L.append(f"{LAB[q]} & {r0['est']:.4f}{st} & ({r0['se']:.4f}) & {int(r0['n']):,} \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
open(f"{OUTD}/tables/tab_c4_stakes.tex", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("C4 done")

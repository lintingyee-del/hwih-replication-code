# -*- coding: utf-8 -*-
"""6B step 53 — M2: ex-ante documentation across contract-origination cohorts,
judged within fixed post-2020 observation windows.

Design (per the stress-tested spec): observation windows = judgment years
2021, 2022, 2024 (2023H2 decays; 2023 kept as robustness). Cohort = extracted
origination year (annual). Estimand: cohort x H interactions identified WITHIN
cohort-x-loan-age cells (age = judgment year - origination year), prefecture
and province x judgment-month FE, so common duration-documentation gradients
and within-window release selection difference out. Reference cohort: 2017
(last fully pre-campaign year). Outcomes: IOU, transfer, guarantee
documentation. Banned outcomes (per referee triage): interest rates (4xLPR
censoring), acquaintance share (paper's own ex-post margin).
Placebo read: cohorts 2014-2016 vs the 2017 reference must show ~0.
Source split: macro (2021-01..09) vs ws (2021-11..2022-12, 2024) sign check.
Output: output/ext2124/m2_cohort.csv
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

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
DATA = str(_REP_PROJECT / "data")
OUT = str(_REP_PROJECT / "output" / "ext2124")

ABBR = {"京": "11", "津": "12", "冀": "13", "晋": "14", "蒙": "15", "内": "15",
        "辽": "21", "吉": "22", "黑": "23", "沪": "31", "苏": "32", "浙": "33",
        "皖": "34", "闽": "35", "赣": "36", "鲁": "37", "豫": "41", "鄂": "42",
        "湘": "43", "粤": "44", "桂": "45", "琼": "46", "渝": "50", "川": "51",
        "黔": "52", "滇": "53", "云": "53", "藏": "54", "陕": "61", "甘": "62",
        "青": "63", "宁": "64", "新": "65"}
MUNI = {"11", "12", "31", "50"}

con = duckdb.connect()
con.sql("SET threads TO 8; SET memory_limit='16GB'")
files = glob.glob(str(_REP_PROJECT / "data" / "derived" / "extract_2021_2024" / 'civ_*.parquet').replace('\\', '/'))

abbr_sql = "CASE " + " ".join(
    f"WHEN ab='{k}' THEN '{v}'" for k, v in ABBR.items()) + " ELSE NULL END"

con.sql(f"""
CREATE OR REPLACE TEMP VIEW loans AS
WITH base AS (
  SELECT case_no, source, doc_type, doc_len,
    TRY_CAST(judgment_date AS DATE) AS jdate,
    TRY_CAST(orig_year AS INT) AS cohort,
    evid_iou, evid_transfer, evid_guarantee,
    regexp_extract(ano_code, '^([\\p{{Han}}])', 1) AS ab,
    regexp_extract(ano_code, '([0-9]+)$', 1) AS code
  FROM read_parquet({files})
  WHERE cause = '民间借贷纠纷' AND doc_type = 'judgment'
)
SELECT *,
  year(jdate) AS jyear, strftime(jdate, '%Y-%m') AS jmonth,
  {abbr_sql} AS provcode,
  year(jdate) - cohort AS age
FROM base
WHERE cohort BETWEEN 2014 AND 2022 AND jdate IS NOT NULL
""")

con.sql(f"""
CREATE OR REPLACE TEMP VIEW cells AS
SELECT
  CASE WHEN provcode IN ('11','12','31','50') THEN provcode || '0000'
       WHEN length(code) = 4 THEN provcode || substr(code,1,2) || '00'
       WHEN length(code) = 2 THEN provcode || code || '00'
       ELSE NULL END AS prefecture_code,
  source, cohort, jyear, jmonth,
  LEAST(GREATEST(age, 0), 7) AS ageb,
  COUNT(*) AS n,
  AVG(evid_iou) AS iou, AVG(evid_transfer) AS transfer,
  AVG(evid_guarantee) AS guarantee
FROM loans
WHERE provcode IS NOT NULL AND age BETWEEN 0 AND 10
GROUP BY 1,2,3,4,5,6
""")
cells = con.sql("SELECT * FROM cells WHERE prefecture_code IS NOT NULL").df()
exp = con.sql(f"SELECT prefecture_code, province, exposure_v2_z AS H FROM '{DATA}/exposure_v2.parquet'").df()
df = cells.merge(exp, on="prefecture_code", how="inner")
print(f"cells: {len(df):,}; loans covered: {df.n.sum():,.0f}; "
      f"prefectures: {df.prefecture_code.nunique()}")
print(df.groupby("cohort").n.sum().to_string())

df["prov_id"] = pd.factorize(df["province"])[0]
df["prov_jm"] = df["province"] + "_" + df["jmonth"]
df["coh_age"] = df["cohort"].astype(str) + "_" + df["ageb"].astype(str)
df["pref"] = df["prefecture_code"]
REF = 2017
COHORTS = [c for c in range(2014, 2023) if c != REF]
for c in COHORTS:
    df[f"cx{c}"] = (df["cohort"] == c).astype(float) * df["H"]

def run(d, label, outcomes=("iou", "transfer", "guarantee")):
    import scipy.stats as sps
    res = []
    have = [c for c in COHORTS if (d["cohort"] == c).any()]
    terms = " + ".join(f"cx{c}" for c in have)
    for y in outcomes:
        m = pf.feols(f"{y} ~ {terms} | coh_age + pref + prov_jm",
                     data=d, vcov={"CRV1": "prov_id"}, weights="n")
        names = list(m.coef().index)
        got = [c for c in have if f"cx{c}" in names]
        for c in got:
            res.append(dict(sample=label, outcome=y, cohort=c,
                            est=m.coef()[f"cx{c}"], se=m.se()[f"cx{c}"],
                            p=m.pvalue()[f"cx{c}"], n=int(m._N)))
        pre = [f"cx{c}" for c in (2014, 2015, 2016) if f"cx{c}" in names]
        p_pl = np.nan
        if len(pre) >= 2:
            b = m.coef()[pre].values
            V = m._vcov[np.ix_([names.index(t) for t in pre],
                               [names.index(t) for t in pre])]
            p_pl = float(1 - sps.chi2.cdf(float(b @ np.linalg.solve(V, b)), len(pre)))
        camp = [f"cx{c}" for c in (2019, 2020) if f"cx{c}" in names]
        est_c = se_c = np.nan
        if camp:
            bc = m.coef()[camp].values
            Vc = m._vcov[np.ix_([names.index(t) for t in camp],
                                [names.index(t) for t in camp])]
            w = np.full(len(camp), 1.0 / len(camp))
            est_c, se_c = float(w @ bc), float(np.sqrt(w @ Vc @ w))
        print(f"[{label}] {y}: campaign(19-20)xH = {est_c:.5f} (se {se_c:.5f}); "
              f"placebo-cohort joint p = {p_pl:.3f}; N={m._N}")
        res.append(dict(sample=label, outcome=y, cohort=9999, est=est_c, se=se_c,
                        p=np.nan, n=int(m._N)))
        res.append(dict(sample=label, outcome=y, cohort=-1, est=p_pl, se=np.nan,
                        p=np.nan, n=int(m._N)))
    return res

all_res = []
main = df[df.jyear.isin([2021, 2022, 2024])]
all_res += run(main, "windows_21_22_24")
all_res += run(df[df.jyear == 2021], "window_2021")
all_res += run(df[df.jyear == 2022], "window_2022")
all_res += run(df[df.jyear == 2024], "window_2024")
all_res += run(main[main.source == "macro"], "macro_only")
all_res += run(main[main.source == "ws"], "ws_only")

pd.DataFrame(all_res).to_csv(f"{OUT}/m2_cohort.csv", index=False)
print("written:", f"{OUT}/m2_cohort.csv")

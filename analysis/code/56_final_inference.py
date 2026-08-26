# -*- coding: utf-8 -*-
"""6B step 56 — submission-grade inference for the extension results.

Part 1 (M1 final spec): coercive-share event study to +66 with cell-level
fact-length control, donut, min-cell 20; pooled post-2020 coefficient with
CRV1 + wild-score bootstrap p; Rambachan-Roth relative-magnitude readout
(B = max |lead|, breakdown Mbar for the pooled coefficient).
Part 2 (M2 headline): campaign-cohort pooled coefficient (2019-20 x H) with
CRV1 + wild-score bootstrap p, per outcome, on the 3-window sample.
Output: output/ext2124/final_inference.csv
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
from scipy import stats as sps
from _wild import wild_score_p

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
DATA = str(_REP_PROJECT / "data")
OUT = str(_REP_PROJECT / "output" / "ext2124")
SRC = str(_REP_CASE_ARCHIVE)
con = duckdb.connect()
con.sql("SET threads TO 8; SET memory_limit='16GB'")
res = []

# ---------------- Part 1: M1 final ----------------
pan = con.sql(f"SELECT * FROM '{OUT}/persist_panel.parquet'").df()
exp = con.sql(f"SELECT prefecture_code, province, exposure_v2_z AS H FROM '{DATA}/exposure_v2.parquet'").df()
insp = con.sql(f"""SELECT province, min(strftime(inspection_start_date, '%Y-%m')) AS insp_ym
  FROM '{SRC}' WHERE inspection_start_date IS NOT NULL GROUP BY 1""").df()
d = pan.merge(exp, on="prefecture_code").merge(insp, on="province")
d["event_time"] = (pd.PeriodIndex(d["ym"], freq="M")
                   - pd.PeriodIndex(d["insp_ym"], freq="M")).map(lambda x: x.n)
d["prov_id"] = pd.factorize(d["province"])[0]
d["prov_month"] = d["province"] + "_" + d["ym"]
d["pref"] = d["prefecture_code"]
d = d[~d["ym"].isin({"2021-09", "2021-10", "2021-11", "2021-12"})]
d = d[d["n_target_fact"] >= 20].dropna(subset=["sh_coercive", "H", "med_factlen"])
d["factlen_k"] = d["med_factlen"] / 1000.0

BINS = [(-24,-19),(-18,-13),(-12,-7),(0,5),(6,11),(12,17),(18,23)]
terms = []
for lo, hi in BINS:
    nm = f"b_{lo}_{hi}".replace("-", "m")
    d[nm] = ((d["event_time"] >= lo) & (d["event_time"] <= hi)).astype(float) * d["H"]
    terms.append(nm)
d["post2124"] = (d["event_time"] >= 24).astype(float) * d["H"]
fml = f"sh_coercive ~ post2124 + {' + '.join(terms)} + factlen_k | pref + prov_month"
m = pf.feols(fml, data=d, vcov={"CRV1": "prov_id"}, weights="n_target_fact")
names = list(m.coef().index)
leads = [t for t in terms if t.startswith("b_m")]
B = float(np.max(np.abs(m.coef()[leads].values)))
theta, se = float(m.coef()["post2124"]), float(m.se()["post2124"])
breakdown = max(0.0, (abs(theta) - 1.645 * se) / B) if B > 0 else np.inf
li = [names.index(t) for t in leads]
lb = m.coef()[leads].values
lV = m._vcov[np.ix_(li, li)]
p_pre = float(1 - sps.chi2.cdf(float(lb @ np.linalg.solve(lV, lb)), len(leads)))
try:
    wp = wild_score_p(fml, d, "post2124", "n_target_fact", "prov_id")
except Exception as e:
    print("wild failed:", e)
    wp = np.nan
print(f"[M1 final] post-2020 pooled x H = {theta:.5f} (se {se:.5f}) "
      f"CRV1 p={m.pvalue()['post2124']:.4f} wild p={wp:.3f}")
print(f"           pre-trend joint p={p_pre:.3f}; RR: B={B:.4f}, breakdown Mbar={breakdown:.2f}")
for t in terms:
    print(f"   {t}: {m.coef()[t]: .5f} ({m.se()[t]:.5f})")
res.append(dict(part="M1", object="post2124_pooled", est=theta, se=se,
                p_crv1=float(m.pvalue()["post2124"]), p_wild=wp,
                B=B, breakdown_Mbar=breakdown, pretrend_p=p_pre, n=int(m._N)))

# ---------------- Part 2: M2 bootstrap ----------------
ABBR = {"京": "11", "津": "12", "冀": "13", "晋": "14", "蒙": "15", "内": "15",
        "辽": "21", "吉": "22", "黑": "23", "沪": "31", "苏": "32", "浙": "33",
        "皖": "34", "闽": "35", "赣": "36", "鲁": "37", "豫": "41", "鄂": "42",
        "湘": "43", "粤": "44", "桂": "45", "琼": "46", "渝": "50", "川": "51",
        "黔": "52", "滇": "53", "云": "53", "藏": "54", "陕": "61", "甘": "62",
        "青": "63", "宁": "64", "新": "65"}
abbr_sql = "CASE " + " ".join(f"WHEN ab='{k}' THEN '{v}'" for k, v in ABBR.items()) + " ELSE NULL END"
files = glob.glob(str(_REP_PROJECT / "data" / "derived" / "extract_2021_2024" / 'civ_*.parquet').replace('\\', '/'))
cells = con.sql(f"""
WITH base AS (
  SELECT source, TRY_CAST(judgment_date AS DATE) AS jdate,
    TRY_CAST(orig_year AS INT) AS cohort,
    evid_iou, evid_transfer, evid_guarantee,
    regexp_extract(ano_code, '^([\\p{{Han}}])', 1) AS ab,
    regexp_extract(ano_code, '([0-9]+)$', 1) AS code
  FROM read_parquet({files})
  WHERE cause = '民间借贷纠纷' AND doc_type = 'judgment'
), loans AS (
  SELECT *, year(jdate) AS jyear, strftime(jdate, '%Y-%m') AS jmonth,
    {abbr_sql} AS provcode, year(jdate) - cohort AS age
  FROM base WHERE cohort BETWEEN 2014 AND 2022 AND jdate IS NOT NULL
)
SELECT
  CASE WHEN provcode IN ('11','12','31','50') THEN provcode || '0000'
       WHEN length(code) = 4 THEN provcode || substr(code,1,2) || '00'
       WHEN length(code) = 2 THEN provcode || code || '00'
       ELSE NULL END AS prefecture_code,
  cohort, jyear, jmonth, LEAST(GREATEST(age, 0), 7) AS ageb,
  COUNT(*) AS n, AVG(evid_iou) AS iou, AVG(evid_transfer) AS transfer,
  AVG(evid_guarantee) AS guarantee
FROM loans WHERE provcode IS NOT NULL AND age BETWEEN 0 AND 10
GROUP BY 1,2,3,4,5
""").df()
m2 = cells.dropna(subset=["prefecture_code"]).merge(exp, on="prefecture_code")
m2 = m2[m2.jyear.isin([2021, 2022, 2024])].copy()
m2["prov_id"] = pd.factorize(m2["province"])[0]
m2["prov_jm"] = m2["province"] + "_" + m2["jmonth"]
m2["coh_age"] = m2["cohort"].astype(str) + "_" + m2["ageb"].astype(str)
m2["pref"] = m2["prefecture_code"]
m2["cxCAMP"] = m2["cohort"].isin([2019, 2020]).astype(float) * m2["H"]
for c in (2014, 2015, 2016, 2018, 2021, 2022):
    m2[f"cx{c}"] = (m2["cohort"] == c).astype(float) * m2["H"]
oth = " + ".join(f"cx{c}" for c in (2014, 2015, 2016, 2018, 2021, 2022))
for y in ("iou", "transfer", "guarantee"):
    fml2 = f"{y} ~ cxCAMP + {oth} | coh_age + pref + prov_jm"
    mm = pf.feols(fml2, data=m2, vcov={"CRV1": "prov_id"}, weights="n")
    try:
        wp2 = wild_score_p(fml2, m2, "cxCAMP", "n", "prov_id")
    except Exception as e:
        print("wild failed:", e)
        wp2 = np.nan
    print(f"[M2 final] {y}: campaign x H = {mm.coef()['cxCAMP']:.5f} "
          f"(se {mm.se()['cxCAMP']:.5f}) CRV1 p={mm.pvalue()['cxCAMP']:.4f} wild p={wp2:.3f}")
    res.append(dict(part="M2", object=y, est=float(mm.coef()["cxCAMP"]),
                    se=float(mm.se()["cxCAMP"]), p_crv1=float(mm.pvalue()["cxCAMP"]),
                    p_wild=wp2, B=np.nan, breakdown_Mbar=np.nan,
                    pretrend_p=np.nan, n=int(mm._N)))

pd.DataFrame(res).to_csv(f"{OUT}/final_inference.csv", index=False)
print("written:", f"{OUT}/final_inference.csv")

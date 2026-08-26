# -*- coding: utf-8 -*-
"""6B step 17 — re-run C2 (origination cohorts) and C3 (rates) with validated
v2 extractors: orig_year_v2 (gold agreement .91) and monthly_rate_v2 (recall .90)."""

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
import pandas as pd, numpy as np, pyfixest as pf, duckdb

DATA = str(_REP_PROJECT / "data")
OUTD = str(_REP_PROJECT / "output")
EXT = str(_REP_PROJECT / "data" / "derived" / "extract_2014_2020")
con = duckdb.connect(); con.sql("SET threads TO 10; SET memory_limit='20GB'")
rows = []

from _wild import wild_score_p, wild_p  # corrected shared WCR bootstrap


def run(tag, fml, df, coef, weights=None):
    m = pf.feols(fml, data=df, vcov={"CRV1": "prov_id"}, weights=weights)
    try: wp = wild_score_p(fml, df, coef, weights)
    except Exception: wp = np.nan
    rows.append(dict(tag=tag, coef=coef, est=m.coef()[coef], se=m.se()[coef],
                     p=m.pvalue()[coef], wild_p=wp, n=int(m._N)))
    print(f"{tag:34s} {m.coef()[coef]: .5f} ({m.se()[coef]:.5f}) p={m.pvalue()[coef]:.4f} "
          f"wild={wp:.3f} N={m._N}")

ld = con.sql(f"""
SELECT c.case_no, c.prefecture_code, c.province, c.jmonth, c.post, c.insp_month,
  c.evid_iou, c.evid_transfer, c.rel_txn, c.doc_len,
  e.exposure_v2_z AS H, x.orig_year_v2, x.monthly_rate_v2
FROM '{DATA}/civil_case.parquet' c
JOIN '{DATA}/exposure_v2.parquet' e USING (prefecture_code)
LEFT JOIN read_parquet('{EXT}/x2_civ_*.parquet') x USING (case_no)
WHERE c.cause = '民间借贷纠纷'
""").df()
ld["prov_id"] = pd.factorize(ld["province"])[0]
ld["month"] = ld["jmonth"].astype(str)
ld["prov_month"] = ld["province"] + "_" + ld["month"]
ld["pref"] = ld["prefecture_code"]
ld["logdoclen"] = np.log(ld["doc_len"].clip(lower=1))
ld["px"] = ld["post"] * ld["H"]

# ---- C3 v2: rates (multi-pattern, recall .90), window <= 2020-07 -------------
lr = ld[(ld["monthly_rate_v2"] > 0) & (ld["monthly_rate_v2"] <= 10)
        & (ld["month"] <= "2020-07")].copy()
run("C3v2_rate", "monthly_rate_v2 ~ px + logdoclen | pref + prov_month", lr, "px")
# relational subsample: the model's premium prediction is about relational credit
lrr = lr[lr["rel_txn"] == 1].copy()
run("C3v2_rate_relational", "monthly_rate_v2 ~ px + logdoclen | pref + prov_month",
    lrr, "px")

# ---- C2 v2: origination cohorts (agreement .91) ------------------------------
oc = ld[(ld["orig_year_v2"] >= 2012) & (ld["orig_year_v2"] <= 2020)].copy()
oc["insp_year"] = pd.to_datetime(oc["insp_month"]).dt.year
oc["post_cohort"] = (oc["orig_year_v2"] >= oc["insp_year"]).astype(int)
oc["pcx"] = oc["post_cohort"] * oc["H"]
oc["oy"] = oc["orig_year_v2"].astype(int).astype(str)
for y, tag in [("evid_iou", "C2v2_orig_iou"), ("evid_transfer", "C2v2_orig_transfer")]:
    run(tag, f"{y} ~ pcx + logdoclen | pref + oy + month", oc, "pcx")
# C6 exit margin: origination volume by cohort-year x prefecture
ov = oc.groupby(["prefecture_code", "province", "prov_id", "oy"], as_index=False) \
    .agg(n=("case_no", "count"), H=("H", "first"), insp_year=("insp_year", "first"))
ov["asinh_n"] = np.arcsinh(ov["n"])
ov["post_cohort"] = (ov["oy"].astype(int) >= ov["insp_year"]).astype(int)
ov["pcx"] = ov["post_cohort"] * ov["H"]
ov["pref"] = ov["prefecture_code"]
run("C6v2_orig_volume", "asinh_n ~ pcx | pref + oy", ov, "pcx")

old = pd.read_csv(f"{OUTD}/results_v2.csv")
new = pd.concat([old[~old["tag"].str.contains("v2_", na=False) |
                     ~old["tag"].str.startswith(("C2v2","C3v2","C6v2"))],
                 pd.DataFrame(rows)])
new = new.drop_duplicates(subset=["tag","coef"], keep="last")
new.to_csv(f"{OUTD}/results_v2.csv", index=False)
print("C2/C3 v2 re-runs saved")
